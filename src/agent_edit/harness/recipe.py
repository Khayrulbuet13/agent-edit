"""The recipe deliverable (--recipe): WORK/recipe.py defines `edit(train, weights) -> {name: tensor}`.

`weights` holds copies of the original tensors. `train.read(spec, weights=None)` returns what a fwd
read with `spec` returned on the original head (or on `weights`), big arrays as numpy. Inside a run,
CachedTrain serves only reads the agent already paid a pass for, so the recipe costs no budget and
can be checked against the saved edits. In a replay, LiveTrain runs the same reads on another train
pool with no agent.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shutil
import signal
import threading
import time
import traceback
from pathlib import Path

from .files import read_json

TOL = 1e-5
_NOT_SPEC = ("type", "model", "which")


def digest(weights: dict) -> str:
    h = hashlib.sha1()
    for n in sorted(weights):
        t = weights[n].detach().cpu().float().contiguous()
        h.update(f"{n}{tuple(t.shape)}".encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()[:16]


def spec_key(spec: dict) -> str:
    return json.dumps({k: v for k, v in spec.items() if k not in _NOT_SPEC}, sort_keys=True)


def load_arrays(obj):
    """A materialized result with its {"npy": path} entries loaded back as arrays."""
    import numpy as np

    if isinstance(obj, dict):
        if set(obj) == {"npy", "shape", "dtype"}:
            return np.load(obj["npy"])
        return {k: load_arrays(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [load_arrays(v) for v in obj]
    return obj


class _Train:
    def __init__(self, orig: dict):
        self._orig = orig

    def _spec(self, spec, weights):
        if not isinstance(spec, dict):
            raise TypeError("spec must be a dict, like the JSON of a fwd read")
        if spec.get("type", "read") != "read":
            raise ValueError("train only serves reads (type 'read')")
        if spec.get("model", "original") != "original":
            raise ValueError("train.read reads the original head; for other weights pass weights=<dict of tensors>")
        if weights is None:
            return spec, None
        bad = sorted(set(weights) - set(self._orig))
        if bad:
            raise KeyError(f"unknown tensor names in weights: {bad}")
        return spec, {**self._orig, **weights}


class CachedTrain(_Train):
    """train.read() answered from this run's earlier fwd reads (index in RUN/log/reads.json)."""

    def __init__(self, index: Path, orig: dict):
        super().__init__(orig)
        self._index = Path(index)
        self._orig_digest = digest(orig)

    def read(self, spec: dict, weights: dict | None = None) -> dict:
        from .guards import trusted

        spec, full = self._spec(spec, weights)
        key = spec_key(spec) + "@" + (self._orig_digest if full is None else digest(full))
        with trusted():
            idx = read_json(self._index, {})
            if key not in idx:
                done = [f"{v['spec']} on {'the original head' if v['original'] else 'other weights'}" for v in idx.values()]
                raise LookupError(f"no fwd read with spec {spec_key(spec)} on these weights in this run. Run it with the "
                                  f"fwd tool first (one pass); train.read can then serve it for free. Reads so far: {done}")
            return load_arrays(json.loads((Path(idx[key]["dir"]) / "result.json").read_text()))

    def specs(self) -> list:
        return [v["spec"] for v in read_json(self._index, {}).values()]


class LiveTrain(_Train):
    """train.read() run on a backend's train pool, the way the harness answers a fwd read.
    `all_images` = (source pool size, target pool size): a read of the whole source pool reads
    the whole target pool, so a recipe that used every image keeps doing so at another N."""

    def __init__(self, backend, orig: dict, tmp: Path, names=None, all_images=None):
        super().__init__(orig)
        self._backend, self._tmp, self._names, self._n = backend, Path(tmp), names, 0
        self._all = all_images

    def read(self, spec: dict, weights: dict | None = None) -> dict:
        from .guards import check_keys, trusted
        from .server import materialize, strip_private

        spec, full = self._spec(spec, weights)
        s = {k: v for k, v in spec.items() if k not in _NOT_SPEC}
        if self._all and s.get("n_images") == self._all[0]:
            s["n_images"] = self._all[1]
        with trusted():
            if self._names:
                s = self._names.inp(s)
            res = strip_private(self._backend.read(full, {**s, "which": "train"}))
            check_keys(res)
            if self._names:
                res = self._names.out(res)
            self._n += 1
            return load_arrays(materialize(res, self._tmp / f"read_{self._n}"))


@contextlib.contextmanager
def _alarm(timeout):
    if not timeout or threading.current_thread() is not threading.main_thread():
        yield
        return

    def on_alarm(*_):
        raise TimeoutError(f"recipe ran longer than {timeout:.0f}s")

    old = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def _validate(out, orig: dict) -> dict:
    import numpy as np
    import torch

    if not isinstance(out, dict):
        raise TypeError(f"edit() must return a dict name -> tensor, got {type(out).__name__}")
    edits = {}
    for n, t in out.items():
        if n not in orig:
            raise KeyError(f"edit() returned an unknown tensor name {n!r}")
        if isinstance(t, np.ndarray):
            t = torch.from_numpy(t)
        if not isinstance(t, torch.Tensor) or not t.is_floating_point():
            raise TypeError(f"{n}: need a float tensor, got {type(t).__name__}")
        t = t.detach().cpu().float()
        if t.shape != orig[n].shape:
            raise ValueError(f"{n}: shape {tuple(t.shape)}, need {tuple(orig[n].shape)}")
        if not bool(torch.isfinite(t).all()):
            raise ValueError(f"{n}: NaN or inf")
        edits[n] = t
    return edits


def run_recipe(source: str, train, orig: dict, sandbox: Path, timeout=None) -> dict:
    """Run recipe source in the agent sandbox: guarded builtins, no file access outside the
    (empty) `sandbox` folder. -> {'ok', 'edits' or 'error', 'seconds'}."""
    import numpy as np
    import torch

    from .guards import confine, precheck
    from .kernel import sandbox_builtins

    sandbox = Path(sandbox)
    shutil.rmtree(sandbox, ignore_errors=True)
    sandbox.mkdir(parents=True)
    why = precheck(source, sandbox)
    if why:
        return {"ok": False, "error": f"refused by the pre-check: {why}", "seconds": 0.0}
    ns = {"__builtins__": sandbox_builtins(sandbox), "__name__": "__recipe__", "torch": torch, "np": np, "numpy": np}
    t0 = time.time()
    try:
        with _alarm(timeout), torch.no_grad(), confine(sandbox, reads=True):
            exec(compile(source, "recipe.py", "exec"), ns)
            fn = ns.get("edit")
            if not callable(fn):
                raise TypeError("recipe.py must define edit(train, weights)")
            out = fn(train, {k: v.clone() for k, v in orig.items()})
        edits = _validate(out, orig)
    except BaseException as e:  # noqa: BLE001 - agent code may raise anything
        if isinstance(e, KeyboardInterrupt) and not timeout:
            raise
        frames = [f for f in traceback.extract_tb(e.__traceback__) if f.filename == "recipe.py"]
        err = "".join(traceback.format_list(frames)) + "".join(traceback.format_exception_only(type(e), e))
        return {"ok": False, "error": err[-2500:], "seconds": round(time.time() - t0, 2)}
    return {"ok": True, "edits": edits, "seconds": round(time.time() - t0, 2)}


def compare(edits: dict, saved: dict, orig: dict) -> dict:
    """Max abs difference between the recipe's head and the saved head, per tensor either touches."""
    per = {}
    for n in sorted(set(edits) | set(saved)):
        a, b = edits.get(n, orig[n]).float(), saved.get(n, orig[n]).float()
        per[n] = float((a - b).abs().max()) if a.numel() else 0.0
    mx = max(per.values(), default=0.0)
    return {"reproduces": bool(mx <= TOL), "max_abs_diff": mx, "tolerance": TOL, "per_tensor": per,
            "recipe_tensors": sorted(edits), "saved_tensors": sorted(saved)}


def check(path: Path, train, orig: dict, saved: dict, sandbox: Path, timeout=None) -> dict:
    """Does recipe.py, run on this run's train data, give back the saved edits?"""
    path = Path(path)
    if not path.exists():
        return {"present": False, "reproduces": False, "message": f"no recipe: {path} does not exist"}
    source = path.read_text()
    out = {"present": True, "sha256": hashlib.sha256(source.encode()).hexdigest()}
    r = run_recipe(source, train, orig, sandbox, timeout)
    out["seconds"] = r["seconds"]
    if not r["ok"]:
        out.update(ok=False, reproduces=False, error=r["error"],
                   message=f"recipe.py failed, so it does not reproduce the saved edits:\n{r['error']}")
        return out
    c = compare(r["edits"], saved, orig)
    out.update(ok=True, **c)
    worst = sorted(c["per_tensor"].items(), key=lambda kv: -kv[1])[:5]
    out["message"] = (f"recipe.py reproduces the saved edits (max abs diff {c['max_abs_diff']:.3g}, "
                      f"{len(c['per_tensor'])} tensors)" if c["reproduces"] else
                      f"recipe.py does NOT reproduce the saved edits: max abs diff {c['max_abs_diff']:.3g} "
                      f"(limit {TOL:g}); largest: {worst}; recipe returns {c['recipe_tensors']}, "
                      f"saved edits are {c['saved_tensors']}")
    return out
