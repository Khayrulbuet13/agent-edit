"""The harness server: owns the kernel and the backend, answers requests from RUN/queue.

The server runs on a compute node and the agent's shell on the login node; they share a
filesystem, so all traffic is files (request written to a temp name, then renamed; the server
answers the same way). The agent only ever sees which='train'.
"""

from __future__ import annotations

import ast
import importlib
import json
import os
import re
import shutil
import signal
import socket
import threading
import time
import traceback
from pathlib import Path

from ..config import has_module
from .files import HB_S, POLL_S, STALE_S, ensure_dirs, now, pid_alive, read_json, write_atomic
from .guards import check_keys, precheck
from .kernel import Kernel
from .names import Names
from .overlays import SHEET_K
from .prompt import build_prompt, build_reading_prompt
from .recipe import CachedTrain, digest, spec_key
from .recipe import check as check_recipe


def _reading_module():
    if not has_module("harness.reading"):
        raise SystemExit("--reading needs harness/reading.py, which this tree does not have")
    return importlib.import_module(f"{__package__}.reading")


def materialize(obj, outdir: Path, prefix="r"):
    """JSON-able copy; arrays with more than 16 elements go to .npy files."""
    import numpy as np
    import torch

    if isinstance(obj, torch.Tensor):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        if obj.size <= 16:
            return obj.tolist()
        outdir.mkdir(parents=True, exist_ok=True)
        p = outdir / (re.sub(r"[^A-Za-z0-9_.-]", "_", prefix) + ".npy")
        np.save(p, obj)
        return {"npy": str(p), "shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {str(k): materialize(v, outdir, f"{prefix}.{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [materialize(v, outdir, f"{prefix}.{i}") for i, v in enumerate(obj)]
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    return repr(obj)


def strip_private(res):
    """A backend result without its private '_' keys (see Backend)."""
    if isinstance(res, dict):
        return {k: v for k, v in res.items() if not (isinstance(k, str) and k.startswith("_"))}
    return res


def short(obj, max_list=30):
    """Display-only copy: long scalar lists are cut."""
    if isinstance(obj, dict):
        return {k: short(v, max_list) for k, v in obj.items()}
    if isinstance(obj, list):
        if all(isinstance(x, (int, float, str, bool)) or x is None for x in obj):
            if len(obj) > max_list:
                return f"{obj[:max_list]} ... ({len(obj)} items; all in result.json)"
            return str(obj)
        return [short(v, max_list) for v in obj[:max_list]]
    return obj


def picture_note(res, sheet=False) -> str:
    """Overlay folders, put before the JSON: a long answer is cut at max_chars (and agents pipe it
    through head), which hid the paths at the end, most often the edited head's in "both"."""
    parts = [res, res.get("original"), res.get("edited")] if isinstance(res, dict) else []
    out = []
    for r in parts:
        if not (isinstance(r, dict) and isinstance(r.get("overlays"), dict)):
            continue
        d = r["overlays"]["dir"]
        if sheet:
            out.append(f"Open {d}/sheet.png first: the {SHEET_K} worst images of this pass\n")
        out.append(f"Error pictures ({len(r['overlays']['files'])} PNG, open them): {d}\n")
    return "".join(out)


def _per_image(res):
    v = res.get("per_image_f1") if isinstance(res, dict) else None
    if isinstance(v, dict):
        return v
    if isinstance(v, list):
        return {str(i): x for i, x in enumerate(v)}
    return None


def paired(ro, re_):
    a, b = _per_image(ro), _per_image(re_)
    out = {}
    if "macro_f1" in ro and "macro_f1" in re_:
        out["macro_f1_delta"] = re_["macro_f1"] - ro["macro_f1"]
    if a and b:
        keys = [k for k in a if k in b]
        if len(keys) > 1:
            d = [float(b[k]) - float(a[k]) for k in keys]
            m = sum(d) / len(d)
            sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
            out.update(per_image_f1_delta_mean=m, per_image_f1_delta_se=sd / len(d) ** 0.5,
                       n_images=len(d), better=sum(x > 0 for x in d), worse=sum(x < 0 for x in d))
    return out


RECIPE_TIMEOUT = 300.0
RETAIN_SLACK = 1.0  # points; the prompt's rule


class Server:
    """Flags (all off by default: the prompt and harness are then the plain ones): overlays = pictures of the
    errors with every generate; overlay_sheet = also tile the SHEET_K worst of those into one
    sheet.png (needs overlays); retention = the fwd {"which": "retain"} check and its prompt rule;
    recipe = the kernel's `train` object and the recipe.py check at final; hide_names = class_<k>
    instead of class names in everything the agent sees; permute_names = every class name becomes
    ANOTHER class's real name (a seeded derangement, `permute_seed`) instead -- unlike hide_names,
    nothing in the prompt says so, since the point is to see whether the agent leans on a name it
    still believes; reading = no head to edit, the agent names the backend's shuffled class rows
    instead (cannot combine with the other flags). hide_names and permute_names are mutually
    exclusive.

    seed_recipe (not a serve flag, so not pinned the same way) copies a recipe.py from another
    project into work/seed_recipe.py as a warm start; seed_recipe_source overrides the project
    label shown to the agent (default: the seed file's grandparent folder name)."""

    def __init__(self, run, backend, budget, max_exec=300, exec_timeout=900.0, max_chars=10_000,
                 max_images=256, frozen=(), deadline_min=None, project=None, overlays=False, retention=False,
                 recipe=False, hide_names=False, permute_names=False, permute_seed=None, reading=False,
                 overlay_sheet=False, seed_recipe=None, seed_recipe_source=None, meta=None):
        self.run = Path(run).resolve()
        self.backend, self.budget, self.max_exec = backend, budget, max_exec
        self.exec_timeout, self.max_chars, self.max_images = exec_timeout, max_chars, max_images
        self.deadline = time.time() + 60 * deadline_min if deadline_min else None
        self.project = project
        self.stop_reason = None
        self._closing = False
        self.finished, self.finish_reason = False, None
        self.seq = 0
        self.t_start = time.time()
        ensure_dirs(self.run)
        self._acquire_lock()
        self._hb = threading.Thread(target=self._heartbeat, daemon=True)
        self._hb.start()

        st = read_json(self.run / "state.json", {})
        if st.get("finished"):
            self._release_lock()
            raise SystemExit(f"run {self.run} is already finished")
        self.passes = int(st.get("passes_used", 0))
        self.execs = int(st.get("exec_used", 0))
        self.restarts = int(st.get("restarts", 0)) + (1 if st else 0)
        self.restart_note = bool(st)
        self.seq = int(st.get("seq", 0))
        self.final_note = ""

        self.flags = {"overlays": bool(overlays), "retention": bool(retention), "recipe": bool(recipe),
                      "hide_names": bool(hide_names), "permute_names": bool(permute_names),
                      "reading": bool(reading), "overlay_sheet": bool(overlay_sheet)}
        self.overlays, self.retention, self.recipe = self.flags["overlays"], self.flags["retention"], self.flags["recipe"]
        self.overlay_sheet = self.flags["overlay_sheet"]
        self.names = None
        self.permute_seed = permute_seed
        self._reading = None
        self.seed_recipe_source = None
        self._seed_recipe_label = None
        try:
            self._setup_flags(meta, budget, max_exec, max_images, exec_timeout, project, seed_recipe, seed_recipe_source)
        except BaseException:
            self._release_lock()
            raise

        orig = {k: v.detach().cpu().float().contiguous().clone() for k, v in backend.initial_weights().items()}
        frozen = list(frozen) + list(backend.frozen_patterns())
        self.orig_digest = digest(orig) if self.recipe else None
        train = CachedTrain(self.run / "log" / "reads.json", orig) if self.recipe else None
        self.kernel = Kernel(self.run, orig, frozen, self._log, train=train)
        if self._reading:
            self.kernel.ns.update(ROWS=self._reading.rows, POOL_FEATURES=self._reading.pool_features,
                                  POOL_NAMES=list(self._reading.pool_names), CLASS_NAMES=list(self._reading.class_names),
                                  submit_names=lambda ans: self._reading.submit(self.kernel.work, ans))
        # the prompt before state.json: a client that sees a live server must find its prompt
        if project:
            pp = self.run / "AGENT_PROMPT.md"
            if not pp.exists():
                if self._reading:
                    pp.write_text(build_reading_prompt(project, self.run, len(self._reading.key),
                                                        self._reading.class_names, max_exec, exec_timeout, max_chars,
                                                        task=backend.project_info().get("task", "")))
                else:
                    info = backend.project_info()
                    tr = self.names.text if self.names else (lambda x: x)
                    classes = [tr(c) for c in info.get("classes", [])]
                    pp.write_text(build_prompt(project, budget, max_exec, self.run, tr(backend.describe()),
                                               classes, tr(info.get("task", "")),
                                               exec_timeout, max_chars, max_images,
                                               n_shot=info.get("n_shot"), gate_open=info.get("gate_open", False),
                                               model_intro=info.get("model_intro"), overlays=self.overlays,
                                               overlay_legend=backend.overlay_legend, overlay_sheet=self.overlay_sheet,
                                               retention=self.retain_orig["about"] if self.retention else None,
                                               recipe=self.recipe, hide_names=self.flags["hide_names"],
                                               seed_recipe_source=self._seed_recipe_label,
                                               metric=info.get("metric"), metric_how=info.get("metric_how")))
        self._write_state()
        self._fail_orphans()
        self._log({"kind": "serve_start", "budget": budget, "max_exec": max_exec, "flags": self.flags,
                   "resumed": bool(st), "passes_used": self.passes, "exec_used": self.execs,
                   "restored_edits": sorted(self.kernel.edits)})
        print(f"[harness] serving {self.run} (passes {self.passes}/{budget}, runs {self.execs}/{max_exec})", flush=True)

    def _setup_flags(self, meta, budget, max_exec, max_images, exec_timeout, project, seed_recipe=None,
                     seed_recipe_source=None):
        """Check the backend supports the flags, and pin the flags in RUN/harness.json: a resumed run
        must keep the flags it started with."""
        from ..backends.base import Backend

        be = self.backend
        if self.overlays and type(be).render_errors is Backend.render_errors:
            raise SystemExit(f"--overlays: {type(be).__name__} has no render_errors()")
        if self.overlay_sheet and not self.overlays:
            raise SystemExit("--overlay-sheet needs --overlays")
        if self.overlay_sheet and not getattr(be, "supports_sheet", False):
            raise SystemExit(f"--overlay-sheet: {type(be).__name__} has no sheet support")
        if seed_recipe:
            src = Path(seed_recipe).resolve()
            if not src.is_file():
                raise SystemExit(f"--seed-recipe: {src} is not a file")
            self.seed_recipe_source = str(src)
            dst = self.run / "work" / "seed_recipe.py"
            if not dst.exists():
                shutil.copyfile(src, dst)
            self._seed_recipe_label = seed_recipe_source or src.parent.parent.name
        if self.flags["hide_names"] and self.flags["permute_names"]:
            raise SystemExit("--hide-names and --permute-names are mutually exclusive")
        if self.flags["hide_names"]:
            self.names = Names(be.project_info().get("classes", []), secret=be.secret_words())
            be.class_labels = dict(self.names.shown)
        elif self.flags["permute_names"]:
            if self.permute_seed is None:
                raise SystemExit("--permute-names needs the run's seed (--backend-arg seed=S)")
            self.names = Names(be.project_info().get("classes", []), secret=be.secret_words(),
                               permute=True, seed=self.permute_seed)
            be.class_labels = dict(self.names.shown)
        if self.retention:
            self.retain_orig = be.retention(None)
            if self.retain_orig is None:
                raise SystemExit(f"--retention: {type(be).__name__}.retention() gives None, so this backend has no "
                                 "held-out set of its original task for this project. Serve it without --retention.")
        if self.flags["reading"]:
            reading = _reading_module()
            if type(be).reading_task is Backend.reading_task:
                raise SystemExit(f"--reading: {type(be).__name__} has no reading_task()")
            if self.overlays or self.retention or self.recipe or self.names:
                raise SystemExit("--reading cannot be combined with --overlays/--retention/--recipe/"
                                 "--hide-names/--permute-names")
            task = be.reading_task()
            if task is None:
                raise SystemExit(f"--reading: {type(be).__name__}.reading_task() gave None for this project")
            self._reading = reading.ReadingTask(task)
            if not (self.run / "work" / "rows.npy").exists():
                self._reading.write_task(self.run / "work")
            self._reading.write_key(reading.key_path(self.run))
        hp = self.run / "harness.json"
        old = read_json(hp, None)
        if old is not None:
            if old.get("flags", {}) != self.flags:
                raise SystemExit(f"{hp} says this run started with flags {old.get('flags')}; got {self.flags}")
            if (old.get("seed_recipe_source") or None) != (self.seed_recipe_source or None):
                raise SystemExit(f"{hp} says this run started with seed_recipe_source "
                                 f"{old.get('seed_recipe_source')!r}; got {self.seed_recipe_source!r}")
            return
        pool = be.n_images() if callable(getattr(be, "n_images", None)) else None
        rec = {"flags": self.flags, "project": project, "budget": budget, "max_exec": max_exec,
               "max_images": max_images, "pool_images": min(pool, max_images) if pool else None,
               "exec_timeout": exec_timeout, "started": now(), **(meta or {})}
        if self.seed_recipe_source:
            rec["seed_recipe_source"] = self.seed_recipe_source
        if self.names:
            rec["names"] = self.names.shown
        if self.retention:
            rec["retention_original"] = self.retain_orig
        write_atomic(hp, json.dumps(rec, indent=1))

    def _lock_path(self):
        return self.run / "serve.lock"

    def _acquire_lock(self):
        p = self._lock_path()
        info = read_json(p, None) if p.exists() else None
        me = socket.gethostname()
        if info:
            age = time.time() - float(info.get("heartbeat", 0))
            same = info.get("host") == me
            alive = pid_alive(int(info.get("pid", -1))) if same else age < STALE_S
            if alive and age < STALE_S:
                raise SystemExit(f"another server holds {p}: host={info.get('host')} "
                                 f"pid={info.get('pid')} heartbeat {age:.0f}s ago")
            print(f"[harness] taking over stale lock (host={info.get('host')} pid={info.get('pid')} "
                  f"heartbeat {age:.0f}s ago)", flush=True)
        self._write_lock()

    def _write_lock(self):
        write_atomic(self._lock_path(), json.dumps(
            {"host": socket.gethostname(), "pid": os.getpid(), "start": self.t_start, "heartbeat": time.time()}))

    def _heartbeat(self):
        while not self._closing:
            time.sleep(HB_S)
            if self._closing:
                return
            try:
                self._write_lock()
            except OSError:
                pass

    def _release_lock(self):
        self._closing = True
        info = read_json(self._lock_path(), {})
        if info.get("pid") == os.getpid():
            self._lock_path().unlink(missing_ok=True)

    def _write_state(self, finished=None, reason=None):
        if finished:
            self.finished, self.finish_reason = True, reason
        finished, reason = self.finished, self.finish_reason
        write_atomic(self.run / "state.json", json.dumps({
            "budget": self.budget, "max_exec": self.max_exec, "passes_used": self.passes,
            "exec_used": self.execs, "finished": finished, "finish_reason": reason,
            "restarts": self.restarts, "seq": self.seq, "updated": now(),
            "host": socket.gethostname(), "pid": os.getpid()}, indent=1))

    def _log(self, rec):
        rec = {"ts": now(), **rec}
        with open(self.run / "log" / "turns.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")

    def _fail_orphans(self):
        """Requests claimed by a dead server have no answer: answer them with an error."""
        q = self.run / "queue"
        for c in (q / "claimed").glob("req_*.json"):
            req = read_json(c, {})
            rid = req.get("id")
            if rid and not (q / f"resp_{rid}.json").exists():
                write_atomic(q / f"resp_{rid}.json", json.dumps(
                    {"ok": False, "text": "ERROR: the server restarted while this request was "
                     "running. It was not counted; send it again."}))

    def serve(self):
        for s in (signal.SIGTERM, signal.SIGINT):
            signal.signal(s, lambda *_: setattr(self, "stop_reason", "signal"))
        q = self.run / "queue"
        try:
            while True:
                if self.stop_reason:
                    self._finalize(None, f"auto: {self.stop_reason}")
                    return
                if self.deadline and time.time() > self.deadline:
                    self._finalize(None, "auto: deadline")
                    return
                reqs = sorted(q.glob("req_*.json"))
                if not reqs:
                    time.sleep(POLL_S)
                    continue
                path = reqs[0]
                claimed = q / "claimed" / path.name
                try:
                    os.replace(path, claimed)
                except OSError:
                    continue  # the client withdrew it
                req = read_json(claimed, None)
                if not isinstance(req, dict) or "id" not in req:
                    continue
                try:
                    resp, done = self._handle(req)
                except BaseException as e:  # noqa: BLE001
                    if isinstance(e, KeyboardInterrupt):
                        raise
                    resp, done = {"ok": False, "text": f"ERROR (server): {e!r}"}, False
                    self._log({"kind": "server_error", "error": traceback.format_exc()[-3000:]})
                write_atomic(q / f"resp_{req['id']}.json", json.dumps(resp))
                self._write_state()
                if done:
                    return
                if self.passes >= self.budget and self.execs >= self.max_exec:
                    self._finalize(None, "auto: budget and exec limit used")
                    return
        finally:
            self._release_lock()

    def _footer(self):
        return f"[runs {self.execs}/{self.max_exec} | passes {self.passes}/{self.budget}]"

    def _handle(self, req):
        kind = req.get("kind")
        if kind == "ping":
            return {"ok": True, "text": "pong"}, False
        if kind == "final":
            self._finalize(req.get("summary"), "agent")
            return {"ok": True, "text": f"final submitted. Run finished. {self._footer()}\n"
                    f"edited tensors: {len(self.kernel.edits)}{self.final_note}"}, True
        if kind == "exec":
            return self._do_exec(req), False
        if kind == "fwd":
            return self._do_fwd(req), False
        return {"ok": False, "text": f"ERROR: unknown request kind {kind!r}"}, False

    def _do_exec(self, req):
        code = req.get("code", "")
        if not isinstance(code, str) or not code.strip():
            return {"ok": False, "text": "ERROR: empty code"}
        if len(code) > 200_000:
            return {"ok": False, "text": "ERROR: code too long (200,000 chars max)"}
        self.seq += 1
        why = precheck(code, self.kernel.work)
        if why:
            self._log({"kind": "exec_rejected", "seq": self.seq, "code": code, "reason": why})
            return {"ok": False, "text": f"REJECTED (not run, not counted): {why}\n{self._footer()}"}
        try:
            ast.parse(code)
        except SyntaxError as e:
            self._log({"kind": "exec_syntax_error", "seq": self.seq, "code": code})
            return {"ok": False, "text": f"SyntaxError (not counted): {e}\n{self._footer()}"}
        if self.execs >= self.max_exec:
            return {"ok": False, "text": f"ERROR: all {self.max_exec} code runs are used. "
                    "Use the fwd tool if passes remain, then call final."}
        self.execs += 1
        self._log({"kind": "exec", "seq": self.seq, "run": self.execs, "code": code})
        t0 = time.time()
        text, ok = self.kernel.run_code(code, self.exec_timeout, self.max_chars)
        dt = time.time() - t0
        self._log({"kind": "exec_result", "seq": self.seq, "run": self.execs, "ok": ok,
                   "seconds": round(dt, 2), "output": text[:30_000]})
        head = ""
        if self.restart_note:
            self.restart_note = False
            head = (f"[the server was restarted: kernel variables were lost. Saved edits were "
                    f"restored: {sorted(self.kernel.edits)}]\n")
        done_note = "\n[all code runs are used]" if self.execs >= self.max_exec else ""
        return {"ok": ok, "text": f"{head}{text}{done_note}\n{self._footer()} {dt:.1f}s"}

    def _do_fwd(self, req):
        if self.flags["reading"]:
            return {"ok": False, "text": "ERROR: fwd is not available in reading mode (no head to run). "
                    "Use exec and submit_names({...}), then final."}
        spec = req.get("request")
        if not isinstance(spec, dict):
            return {"ok": False, "text": "ERROR: request must be a JSON object"}
        agent_spec = spec
        if self.names:
            spec = self.names.inp(spec)
        err = None
        typ, model = spec.get("type"), spec.get("model", "edited")
        which = spec.get("which", "train")
        if self.passes >= self.budget:
            return {"ok": False, "text": f"ERROR: forward-pass budget used up ({self.passes}/"
                    f"{self.budget}). Call final."}
        if which == "retain":
            return self._do_retain()
        if typ not in ("generate", "read"):
            err = "type must be 'generate' or 'read'"
        elif model not in (("original", "edited", "both") if typ == "generate" else ("original", "edited")):
            err = "model must be 'original' or 'edited' (generate also accepts 'both')"
        elif which != "train":
            err = "only which='train' is available" + (" (or 'retain')" if self.retention else "")
        n = spec.get("n_images")
        if not err and typ == "generate":
            if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= self.max_images:
                err = f"n_images must be an integer in 1..{self.max_images}"
        if not err and typ == "read" and n is not None:
            if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= self.max_images:
                err = f"n_images must be an integer in 1..{self.max_images}"
        if err:
            return {"ok": False, "text": f"ERROR (not counted): {err}\n{self._footer()}"}

        self.seq += 1
        k = self.passes + 1
        outdir = self.kernel.work / f"fwd_{k}"
        shutil.rmtree(outdir, ignore_errors=True)
        self._log({"kind": "fwd", "seq": self.seq, "pass": k, "request": agent_spec})
        t0 = time.time()
        try:
            w = None if model == "original" else self.kernel.current_weights()
            note = "" if (model == "original" or self.kernel.edits) else \
                "[note: no edits saved yet, so 'edited' equals the original]\n"
            if typ == "generate":
                if model == "both":
                    ro = self._generate(None, n, outdir, "original")
                    re_ = self._generate(self.kernel.current_weights(), n, outdir, "edited")
                    res = {"original": ro, "edited": re_, "paired": paired(ro, re_)}
                else:
                    res = self._generate(w, n, outdir, model)
            else:
                s = {k_: v for k_, v in spec.items() if k_ not in ("type", "model")}
                s["which"] = "train"
                res = strip_private(self.backend.read(w, s))
            check_keys(res)
            if self.names:
                res = self.names.out(res)
            res = materialize(res, outdir)
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "result.json").write_text(json.dumps(res, indent=1))
            if typ == "read" and self.recipe:
                self._index_read(agent_spec, w, outdir, k)
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            shutil.rmtree(outdir, ignore_errors=True)
            tb = traceback.format_exc()[-1200:]
            self._log({"kind": "fwd_failed", "seq": self.seq, "error": tb})
            msg = f"ERROR (pass not counted): {e!r}\n{tb}\n{self._footer()}"
            return {"ok": False, "text": self.names.text(msg) if self.names else msg}
        self.passes += 1
        dt = time.time() - t0
        self._log({"kind": "fwd_result", "seq": self.seq, "pass": self.passes,
                   "seconds": round(dt, 1), "result": res if len(json.dumps(res)) < 20_000
                   else f"see work/fwd_{k}/result.json"})
        text = json.dumps(short(res), indent=1)
        if len(text) > self.max_chars:
            text = text[: self.max_chars] + f"\n[cut; full result in {outdir}/result.json]"
        left = self.budget - self.passes
        tail = "\n[budget used up: call final]" if left == 0 else ""
        return {"ok": True, "text": f"{note}pass {self.passes}/{self.budget} done ({dt:.1f}s). "
                f"Files in {outdir}\n{picture_note(res, self.overlay_sheet)}{text}{tail}\n{self._footer()}"}

    def _generate(self, weights, n, outdir, label):
        res = self.backend.generate(weights, n, "train")
        if self.overlays:
            d = outdir / f"overlays_{label}"
            kw = {"sheet_k": SHEET_K} if self.overlay_sheet else {}
            files = self.backend.render_errors(res, d, **kw) or []
            res = {**strip_private(res), "overlays": {"dir": str(d), "files": [str(f) for f in files],
                                                      "legend": self.backend.overlay_legend}}
        return strip_private(res)

    def _index_read(self, agent_spec, w, outdir, k):
        """Remember which read (spec, weights) produced fwd_<k>, so the kernel's train.read can serve it."""
        key_w = self.orig_digest if w is None else digest(w)
        p = self.run / "log" / "reads.json"
        idx = read_json(p, {})
        idx[spec_key(agent_spec) + "@" + key_w] = {"spec": spec_key(agent_spec), "original": key_w == self.orig_digest,
                                                  "dir": str(outdir), "pass": k}
        write_atomic(p, json.dumps(idx, indent=1))

    def _do_retain(self):
        if not self.retention:
            return {"ok": False, "text": f"ERROR (not counted): the retain check is off in this run\n{self._footer()}"}
        self.seq += 1
        k = self.passes + 1
        outdir = self.kernel.work / f"fwd_{k}"
        shutil.rmtree(outdir, ignore_errors=True)
        self._log({"kind": "fwd", "seq": self.seq, "pass": k, "request": {"which": "retain"}})
        t0 = time.time()
        try:
            res = self._retain_result(self.kernel.current_weights())
            outdir.mkdir(parents=True, exist_ok=True)
            (outdir / "result.json").write_text(json.dumps(res, indent=1))
        except BaseException as e:  # noqa: BLE001
            if isinstance(e, KeyboardInterrupt):
                raise
            shutil.rmtree(outdir, ignore_errors=True)
            tb = traceback.format_exc()[-1200:]
            self._log({"kind": "fwd_failed", "seq": self.seq, "error": tb})
            return {"ok": False, "text": f"ERROR (pass not counted): {e!r}\n{tb}\n{self._footer()}"}
        self.passes += 1
        dt = time.time() - t0
        self._log({"kind": "fwd_result", "seq": self.seq, "pass": self.passes, "seconds": round(dt, 1), "result": res})
        tail = "\n[budget used up: call final]" if self.passes >= self.budget else ""
        return {"ok": True, "text": f"pass {self.passes}/{self.budget} done ({dt:.1f}s). Files in {outdir}\n"
                f"{json.dumps(res, indent=1)}{tail}\n{self._footer()}"}

    def _retain_result(self, weights) -> dict:
        o = self.retain_orig
        e = self.backend.retention(weights)
        return {"metric": o["metric"], "n_images": o["n_images"], "original": round(o["score"], 4),
                "edited": round(e["score"], 4), "difference": round(e["score"] - o["score"], 4),
                "within_1_point": bool(e["score"] >= o["score"] - RETAIN_SLACK)}

    def _finalize(self, summary, reason):
        extra = {"passes_used": self.passes, "budget": self.budget, "exec_used": self.execs,
                 "max_exec": self.max_exec, "finish_reason": reason, "finished_at": now()}
        write_atomic(self.run / "edit" / "manifest.json",
                     json.dumps(self.kernel.manifest(final=True, extra=extra), indent=1))
        self._final_checks(extra)
        m = self.kernel.manifest()
        stats = (f"\n\n---\nedited tensors: {m['n_edited_tensors']}, changed weights: "
                 f"{m['changed_weights']} of {m['total_weights']} ({100 * m['fraction']:.4f}%), "
                 f"passes {self.passes}/{self.budget}, code runs {self.execs}/{self.max_exec}, "
                 f"finish: {reason}\n")
        body = summary.strip() if summary and summary.strip() else \
            "(placeholder: no summary was submitted. Use `final --summary-file F` next time.)"
        write_atomic(self.run / "final_answer.md", "# Final answer\n\n" + body + stats)
        self._write_state(finished=True, reason=reason)
        self._log({"kind": "final", "reason": reason, "summary": summary})
        print(f"[harness] finished ({reason})", flush=True)

    def _final_checks(self, extra):
        """RUN/final.json: how the run ended, plus the retention of the submitted head and the recipe
        check. Neither check can block final: a failure is recorded, not raised."""
        fin = {**extra, "flags": self.flags}
        if self.flags["reading"]:
            answered = (self.kernel.work / "answer.json").exists()
            fin["reading"] = {"answered": answered}
            if not answered:
                self.final_note += "\n[no answer.json submitted: call submit_names({...}) before final next time.]"
        if self.retention:
            try:
                fin["retention"] = self._retain_result(self.kernel.current_weights())
            except Exception as e:  # noqa: BLE001
                fin["retention"] = {"error": repr(e)}
        if self.recipe:
            src = self.kernel.work / "recipe.py"
            try:
                res = check_recipe(src, self.kernel.train, self.kernel.orig, self.kernel.edits,
                                   self.run / "log" / "recipe_sandbox", timeout=RECIPE_TIMEOUT)
                if src.exists():
                    shutil.copyfile(src, self.run / "recipe.py")
            except Exception as e:  # noqa: BLE001
                res = {"present": src.exists(), "reproduces": False, "error": repr(e), "message": f"recipe check failed: {e!r}"}
            fin["recipe"] = {k: v for k, v in res.items() if k != "message"}
            self.final_note = f"\n{res['message']}"
            self._log({"kind": "recipe_check", "reproduces": res.get("reproduces"), "max_abs_diff": res.get("max_abs_diff"),
                       "error": res.get("error")})
        write_atomic(self.run / "final.json", json.dumps(fin, indent=1))


if has_module("harness.server_lab"):  # names of an optional sibling module, when the tree has it
    from .server_lab import *  # noqa: F403
