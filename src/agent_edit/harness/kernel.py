"""The agent's persistent Python kernel over the editable weights (CPU float32 tensors)."""

from __future__ import annotations

import ast
import builtins
import contextlib
import difflib
import fnmatch
import io
import json
import os
import signal
import threading
import traceback
from pathlib import Path

from .files import read_json, safe_name, write_atomic
from .guards import DENY_NAMES, bad_module, confine, inside, trusted


class EditRejected(ValueError):
    pass


def sandbox_builtins(work: Path) -> dict:
    """Builtins for agent code: no eval/exec/compile, guarded import, open() only inside `work`."""
    work = Path(work)

    def safe_open(file, *a, **k):
        if isinstance(file, int):
            raise PermissionError("open() of file descriptors is not allowed")
        p = Path(os.fspath(file))
        if not p.is_absolute():
            p = work / p
        if not inside(str(p), work, relative_ok=False):
            raise PermissionError(f"open() only works inside {work}")
        return open(p, *a, **k)

    b = dict(builtins.__dict__)
    for n in DENY_NAMES | {"exit", "quit"}:
        b.pop(n, None)
    b["open"] = safe_open
    b["__import__"] = guarded_import
    return b


def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if level == 0:
        m = bad_module(name) or next(
            (x for x in (bad_module(f"{name}.{f}") for f in (fromlist or ())) if x), None)
        if m:
            raise ImportError(m)
    return __import__(name, globals, locals, fromlist, level)


class _Capture(io.TextIOBase):
    KEEP = 200_000

    def __init__(self):
        self.parts, self.kept, self.total = [], 0, 0

    def write(self, s):
        self.total += len(s)
        if self.kept < self.KEEP:
            s2 = s[: self.KEEP - self.kept]
            self.parts.append(s2)
            self.kept += len(s2)
        return len(s)

    def flush(self):
        pass

    def getvalue(self):
        return "".join(self.parts)


class Kernel:
    """Persistent namespace. W holds the current weights (edits applied);
    the originals live in self.orig and are never exposed by reference."""

    def __init__(self, run: Path, orig: dict, frozen: list, log, train=None):
        import numpy as np
        import torch

        self.torch, self.run, self._log_fn = torch, run, log
        self.work = (run / "work").resolve()
        self.orig = orig
        self.frozen = list(frozen)
        self.edits: dict = {}
        self.total = sum(t.numel() for t in orig.values())
        self.W = {k: v.clone() for k, v in orig.items()}
        self.train = train

        self.ns = {
            "__builtins__": sandbox_builtins(self.work), "__name__": "__kernel__",
            "torch": torch, "np": np, "numpy": np, "W": self.W, "NAMES": sorted(orig),
            "WORK": str(self.work), "save_edit": self.save_edit, "drop_edit": self.drop_edit,
            "edited": self.edited, "load": lambda n: self.W[n].clone(),
            "load_original": lambda n: self.orig[n].clone(),
        }
        if train is not None:
            self.ns.update(train=train, check_recipe=self.check_recipe)
        self._load_saved_edits()

    def log(self, rec):
        with trusted():  # the run log lives outside WORK, and save_edit logs while agent code runs
            self._log_fn(rec)

    def _edit_path(self, name):
        return self.run / "edit" / (safe_name(name) + ".pt")

    def _load_saved_edits(self):
        man = read_json(self.run / "edit" / "manifest.json", {})
        for name in man.get("edited", {}):
            p = self._edit_path(name)
            if name in self.orig and p.exists():
                t = self.torch.load(p, weights_only=True)
                if t.shape == self.orig[name].shape:
                    self.edits[name] = t
                    self.W[name] = t.clone()

    def current_weights(self) -> dict:
        return {n: self.edits.get(n, self.orig[n]) for n in self.orig}

    def edited(self):
        return {n: int((t != self.orig[n]).sum()) for n, t in self.edits.items()}

    def manifest(self, final=False, extra=None) -> dict:
        info = {n: {"shape": list(t.shape), "changed": int((t != self.orig[n]).sum()),
                    "file": self._edit_path(n).name} for n, t in self.edits.items()}
        changed = sum(v["changed"] for v in info.values())
        m = {"edited": info, "n_edited_tensors": len(info), "changed_weights": changed,
             "total_weights": self.total, "fraction": changed / max(self.total, 1), "final": final}
        m.update(extra or {})
        return m

    def _write_manifest(self):
        write_atomic(self.run / "edit" / "manifest.json", json.dumps(self.manifest(), indent=1))

    def save_edit(self, name, tensor):
        torch = self.torch
        if not isinstance(name, str) or name not in self.orig:
            close = difflib.get_close_matches(str(name), list(self.orig), n=3)
            raise EditRejected(f"unknown tensor name {name!r}. Closest: {close}. See NAMES.")
        for pat in self.frozen:
            if fnmatch.fnmatch(name, pat):
                raise EditRejected(f"{name!r} is frozen (matches {pat!r}); edits to it are refused.")
        if isinstance(tensor, self.ns["np"].ndarray):
            tensor = torch.from_numpy(tensor)
        if not isinstance(tensor, torch.Tensor):
            raise EditRejected(f"need a torch.Tensor, got {type(tensor).__name__}")
        t = tensor.detach().cpu()
        o = self.orig[name]
        if t.dtype != o.dtype:
            raise EditRejected(f"dtype mismatch: got {t.dtype}, need {o.dtype} (use .float())")
        if t.shape != o.shape:
            raise EditRejected(f"shape mismatch: got {tuple(t.shape)}, need {tuple(o.shape)}")
        if not bool(torch.isfinite(t).all()):
            raise EditRejected("tensor has NaN or inf")
        od, nd = o.double().flatten(), t.double().flatten()
        denom = float((od * od).sum())
        c = float((nd * od).sum()) / denom if denom > 0 else 0.0
        scale = float(od.abs().max())
        if torch.allclose(nd, c * od, rtol=1e-5, atol=1e-6 * max(scale, 1e-12)):
            if abs(c - 1.0) < 1e-6 or denom == 0:
                raise EditRejected("tensor is identical to the original (no change).")
            # a 1-element tensor (CLIP's logit_scale) can only change by a rescale
            if o.numel() > 1:
                raise EditRejected(f"tensor equals the original times {c:.6g}: whole-tensor "
                                   "rescales are refused. Change specific parts instead.")
        self.edits[name] = t.clone().contiguous()
        self.W[name] = t.clone()
        p = self._edit_path(name)
        tmp = p.with_name("." + p.name + ".tmp")
        with trusted():
            torch.save(self.edits[name], tmp)
            os.replace(tmp, p)
            self._write_manifest()
        n = int((t != o).sum())
        self.log({"kind": "save_edit", "name": name, "changed": n, "numel": t.numel()})
        return f"saved {name}: {n} of {t.numel()} elements changed ({100 * n / t.numel():.2f}%)"

    def drop_edit(self, name):
        if name not in self.edits:
            return f"no edit saved for {name!r}"
        del self.edits[name]
        self.W[name] = self.orig[name].clone()
        with trusted():
            self._edit_path(name).unlink(missing_ok=True)
            self._write_manifest()
        self.log({"kind": "drop_edit", "name": name})
        return f"dropped edit of {name}"

    def run_code(self, code: str, timeout: float, max_chars: int):
        """-> (text, ok). A trailing expression is printed, like a notebook cell."""
        tree = ast.parse(code)
        last = None
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            last = ast.Expression(tree.body.pop().value)
            ast.fix_missing_locations(last)
        cap = _Capture()
        err = ""
        use_alarm = threading.current_thread() is threading.main_thread() and timeout > 0

        def on_alarm(*_):
            raise TimeoutError(f"code run exceeded {timeout:.0f}s")

        if use_alarm:
            old = signal.signal(signal.SIGALRM, on_alarm)
            signal.setitimer(signal.ITIMER_REAL, timeout)
        try:
            with contextlib.redirect_stdout(cap), contextlib.redirect_stderr(cap), self.torch.no_grad(), \
                    confine(self.work):
                try:
                    exec(compile(tree, "<cell>", "exec"), self.ns)
                    if last is not None:
                        v = eval(compile(last, "<cell>", "eval"), self.ns)
                        if v is not None:
                            print(repr(v))
                except BaseException as e:  # noqa: BLE001 - the kernel must survive anything
                    if isinstance(e, (KeyboardInterrupt, SystemExit)) and not use_alarm:
                        raise
                    frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename != __file__]
                    err = ("Traceback (most recent call last):\n"
                           + "".join(traceback.format_list(frames))
                           + "".join(traceback.format_exception_only(type(e), e)))
        finally:
            if use_alarm:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old)
        out = cap.getvalue()
        if len(out) > max_chars:
            out = out[:max_chars] + f"\n[output cut: showing first {max_chars} of {cap.total} chars]"
        if err:
            out += ("\n" if out and not out.endswith("\n") else "") + err[-2500:]
        return out, not err

    def check_recipe(self):
        """Run WORK/recipe.py the way the final check does and say whether it reproduces the saved edits."""
        from .recipe import check

        with trusted():
            res = check(self.work / "recipe.py", self.train, self.orig, self.edits, self.work / ".recipe_sandbox",
                        timeout=None)
        self.log({"kind": "check_recipe", **{k: v for k, v in res.items() if k != "per_tensor"}})
        return res["message"]
