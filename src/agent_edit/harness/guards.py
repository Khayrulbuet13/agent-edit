"""Best-effort guards on the agent's code and on backend results.

The static pre-check, the guarded import/open in the kernel, and the save_edit checks stop
honest mistakes and obvious shortcuts. A determined agent can get around them (the kernel lives
in the server process). The rule "no fitting of weights to labels" cannot be checked by code:
it rests on the prompt plus the audit log (every exec block is logged verbatim).

Paths are checked twice. The pre-check reads string literals in the path argument of known file
calls. At run time, `confine` (an audit hook plus torch.save/torch.load wrappers, since torch
writes through C++ that raises no audit event) refuses writes outside WORK while agent code runs,
and every file outside WORK and the Python install while a recipe runs.
"""

from __future__ import annotations

import ast
import contextlib
import os
import re
import sys
import threading
from pathlib import Path

from ..config import PKG

DENY_MODULES = {
    "os", "sys", "subprocess", "socket", "urllib", "urllib3", "requests", "http", "httpx",
    "aiohttp", "ftplib", "smtplib", "shutil", "ctypes", "multiprocessing", "importlib",
    "pathlib", "builtins", "pickle", "io", "glob", "tempfile", "signal", "threading",
    "inspect", "gc", "runpy", "code", "posix", "nt", "pty", "imp",
    PKG, "sam3", "transformers", "huggingface_hub", "timm",
}
DENY_ATTRS = {
    "backward": "backward passes are not allowed",
    "requires_grad_": "gradients are not allowed",
    "autograd": "torch.autograd is not allowed",
    "optim": "optimizers are not allowed",
    "set_grad_enabled": "gradients are not allowed",
    "enable_grad": "gradients are not allowed",
    "vjp": "gradients are not allowed",
    "jvp": "gradients are not allowed",
    "jacrev": "gradients are not allowed",
    "jacfwd": "gradients are not allowed",
    "hessian": "gradients are not allowed",
    "grad_and_value": "gradients are not allowed",
}
DENY_DUNDER = {"__import__", "__builtins__", "__globals__", "__subclasses__", "__code__",
               "__closure__", "__loader__", "__spec__", "__getattribute__"}
DENY_NAMES = {"eval", "exec", "compile", "breakpoint", "input"}
DENY_STRINGS = {"autograd", "optim", "backward", "requires_grad_"}
MODEL_WORDS = re.compile(r"head_forward|load_head_weights|from_pretrained|load_model|build_sam|sam3", re.I)
# file calls -> (position, keyword) of their path argument
_NP_PATH = {"save": (0, "file"), "savez": (0, "file"), "savez_compressed": (0, "file"), "savetxt": (0, "fname"),
            "load": (0, "file"), "fromfile": (0, "file"), "loadtxt": (0, "fname"), "genfromtxt": (0, "fname"),
            "memmap": (0, "filename")}
PATH_ARG = {"torch.save": (1, "f"), "torch.load": (0, "f"),
            **{f"{m}.{fn}": v for m in ("np", "numpy") for fn, v in _NP_PATH.items()}}
# methods that write their first argument as a file (array.tofile, plt.savefig, df.to_csv, ...)
PATH_METHODS = {"tofile": "fid", "savefig": "fname", "imsave": "fname", "imwrite": "filename", "to_csv": "path_or_buf",
                "to_json": "path_or_buf", "to_parquet": "path", "to_pickle": "path", "write_text": None, "write_bytes": None}
NO_MODEL = "loading or running the model is not allowed in the kernel (use the fwd tool)"


def _dotted(node) -> str:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def bad_module(full: str):
    top = full.split(".")[0]
    if top in DENY_MODULES:
        return f"importing '{top}' is not allowed"
    for p in ("torch.autograd", "torch.optim", "torch.func"):
        if full == p or full.startswith(p + "."):
            return f"'{p}' is not allowed"
    if MODEL_WORDS.search(full):
        return NO_MODEL
    return None


def inside(path: str, work: Path, relative_ok: bool) -> bool:
    p = Path(path)
    if not p.is_absolute():
        if not relative_ok:
            return False
        p = work / p
    try:
        p.resolve().relative_to(work.resolve())
        return True
    except ValueError:
        return False


def _check_node(node, work: Path):
    if isinstance(node, ast.Import):
        for a in node.names:
            m = bad_module(a.name)
            if m:
                return m
    elif isinstance(node, ast.ImportFrom):
        mod = node.module or ""
        m = bad_module(mod)
        if m:
            return m
        for a in node.names:
            m = bad_module(f"{mod}.{a.name}") if mod else bad_module(a.name)
            if m:
                return m
            if mod == "torch" and a.name in ("func", "autograd", "optim"):
                return f"'torch.{a.name}' is not allowed"
    elif isinstance(node, ast.Attribute):
        if node.attr in DENY_ATTRS:
            return f"'.{node.attr}': {DENY_ATTRS[node.attr]}"
        if node.attr in DENY_DUNDER:
            return f"'{node.attr}' is not allowed"
        if MODEL_WORDS.search(node.attr):
            return NO_MODEL
        d = _dotted(node)
        if d == "torch.func" or d.startswith("torch.func."):
            return "'torch.func' is not allowed"
    elif isinstance(node, ast.Name):
        if node.id in DENY_NAMES or node.id in DENY_DUNDER:
            return f"'{node.id}' is not allowed"
        if MODEL_WORDS.search(node.id):
            return NO_MODEL
    elif isinstance(node, ast.Constant) and isinstance(node.value, str):
        if node.value in DENY_STRINGS:
            return f"string '{node.value}' is not allowed (looks like a getattr bypass)"
    elif isinstance(node, ast.keyword):
        if node.arg == "requires_grad" and not (isinstance(node.value, ast.Constant) and node.value.value is False):
            return "requires_grad must be False"
    elif isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        val = getattr(node, "value", None)
        for t in targets:
            if isinstance(t, ast.Attribute) and t.attr == "requires_grad" and not (
                    isinstance(val, ast.Constant) and val.value is False):
                return "requires_grad must be False"
    elif isinstance(node, ast.Call):
        return _check_call(node, work)
    return None


def _literal(node):
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _path_literals(node: ast.Call, pos: int, kw) -> list[str]:
    """String literals a file call may use as a path: the path argument itself, plus any other
    literal argument that looks like a path (absolute, home, or with '..')."""
    out = []
    arg = node.args[pos] if len(node.args) > pos else next((k.value for k in node.keywords if k.arg == kw), None)
    if _literal(arg) is not None:
        out.append(_literal(arg))
    for a in list(node.args) + [k.value for k in node.keywords]:
        lit = _literal(a)
        if lit is not None and a is not arg and (lit.startswith(("/", "~")) or ".." in lit):
            out.append(lit)
    return out


def _check_call(node: ast.Call, work: Path):
    fn = node.func
    name = _dotted(fn)
    if isinstance(fn, ast.Name) and fn.id == "open":
        for lit in _path_literals(node, 0, "file"):
            if not inside(lit, work, relative_ok=True):
                return f"open() only works inside {work}"
        return None
    if name in PATH_ARG:
        pos, kw = PATH_ARG[name]
    elif isinstance(fn, ast.Attribute) and fn.attr in PATH_METHODS:
        pos, kw = 0, PATH_METHODS[fn.attr]
        name = f".{fn.attr}()"
    else:
        return None
    for lit in _path_literals(node, pos, kw):
        if not inside(lit, work, relative_ok=False):
            return f"{name} paths must be absolute and inside {work} (use WORK + '/name')"
    return None


def precheck(code: str, work: Path):
    """A reason string if the code is refused, else None. Syntax errors pass (the server reports them)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        msg = _check_node(node, work)
        if msg:
            return f"line {getattr(node, 'lineno', '?')}: {msg}"
    return None


# Backends must never hand GT masks or eval data to the agent.
FORBIDDEN_KEY = re.compile(r"gt_mask|raw_gt|eval_|eval$", re.I)


def check_keys(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if FORBIDDEN_KEY.search(str(k)):
                raise RuntimeError(f"backend returned forbidden key {path}{k!r} (GT masks / eval data)")
            check_keys(v, f"{path}{k}.")
    elif isinstance(obj, (list, tuple)):
        for v in obj[:50]:
            check_keys(v, path)


# Runtime confinement. One audit hook per process (they cannot be removed); it acts only on
# the thread that runs agent code, so the server's heartbeat thread keeps writing its lock file.
_STATE: dict | None = None
_HOOKED = False
_DEPTH = 0
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_WRITE_EVENTS = {"os.remove", "os.rename", "os.mkdir", "os.rmdir", "os.truncate", "os.symlink", "os.link",
                 "os.chmod", "os.chown", "shutil.rmtree", "shutil.copyfile", "shutil.move"}


def _lib_roots() -> list[Path]:
    # not all of sys.path: with `python -m` its first entry is the cwd, which may hold data
    roots = {sys.prefix, sys.base_prefix, sys.exec_prefix, str(Path(__file__).resolve().parents[1]),
             *[p for p in sys.path if "site-packages" in p or "dist-packages" in p]}
    return [Path(r).resolve() for r in roots if os.path.isdir(r)]


def _under(path, roots) -> bool:
    p = Path(os.fsdecode(path) if isinstance(path, bytes) else os.fspath(path))
    if not p.is_absolute():
        p = Path(os.getcwd()) / p
    p = p.resolve()
    return any(p == r or r in p.parents for r in roots)


def _audit(event, args):
    st = _STATE
    if st is None or threading.get_ident() != st["tid"]:
        return
    if event == "open":
        path, mode, flags = args
        if path is None or isinstance(path, int):
            return
        writing = (any(c in mode for c in "wax+") if isinstance(mode, str)
                   else bool((flags or 0) & _WRITE_FLAGS))
        if writing or st["reads"]:
            roots = st["roots"] if writing else st["roots"] + st["lib"]
            if not _under(path, roots):
                raise PermissionError(f"{'writing' if writing else 'reading'} {os.fsdecode(path) if isinstance(path, bytes) else path} "
                                      f"is not allowed: files must stay inside {st['roots'][0]}")
    elif event in _WRITE_EVENTS:
        for a in args[:2]:
            if isinstance(a, (str, bytes, os.PathLike)) and not _under(a, st["roots"]):
                raise PermissionError(f"{event} outside {st['roots'][0]} is not allowed")


def _torch_io(fn, pos):
    def guarded(*a, **k):
        st = _STATE
        f = a[pos] if len(a) > pos else k.get("f")
        if st is not None and threading.get_ident() == st["tid"] and isinstance(f, (str, bytes, os.PathLike)):
            if not _under(f, st["roots"]):
                raise PermissionError(f"torch file paths must be inside {st['roots'][0]} (use WORK + '/name')")
        return fn(*a, **k)
    guarded.__wrapped__ = fn
    return guarded


@contextlib.contextmanager
def confine(root: Path, reads: bool = False):
    """While the block runs on this thread: no writes outside `root`; with reads=True, no reads
    outside `root` either (the Python install stays readable, for imports). Nests."""
    global _STATE, _HOOKED, _DEPTH
    import torch
    import torch.serialization as ts

    if not _HOOKED:
        sys.addaudithook(_audit)
        _HOOKED = True
    prev = _STATE
    _STATE = {"tid": threading.get_ident(), "roots": [Path(root).resolve()], "reads": reads, "lib": _lib_roots()}
    if _DEPTH == 0:
        for m in (torch, ts):
            m.save, m.load = _torch_io(ts.save, 1), _torch_io(ts.load, 0)
    _DEPTH += 1
    try:
        yield
    finally:
        _DEPTH -= 1
        if _DEPTH == 0:
            save, load = ts.save.__wrapped__, ts.load.__wrapped__
            for m in (torch, ts):
                m.save, m.load = save, load
        _STATE = prev


@contextlib.contextmanager
def trusted():
    """Harness code that runs inside a confined block (save_edit, train.read) is not confined."""
    global _STATE
    prev = _STATE
    _STATE = None
    try:
        yield
    finally:
        _STATE = prev
