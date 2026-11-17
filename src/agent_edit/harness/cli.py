"""Harness commands. Both `<pkg> <cmd>` and `python -m <pkg>.harness <cmd>` land here.

    serve  --run-dir RUN --budget B --max-exec 300 --backend module:factory [--backend-arg k=v ...]
           [--overlays] [--overlay-sheet] [--retention] [--recipe] [--hide-names] [--permute-names]
           [--reading] [--seed-recipe PATH] [--seed-recipe-source NAME]
    exec   RUN -c 'code' | --file F | -
    fwd    RUN --json '{...}'
    status RUN
    final  RUN [--summary-file F]
    wait   RUN ID
    prompt --project P --budget B --run-dir RUN --backend module:factory

Only `serve` and `prompt --backend` import torch; the client commands use the standard library.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

from ..config import PKG, has_module
from . import client
from .client import ClientError

COMMANDS = ("serve", "exec", "fwd", "status", "final", "wait", "prompt")


def load_backend(spec: str, args_kv):
    mod, _, fac = spec.partition(":")
    if not fac:
        raise SystemExit("--backend must look like module:factory")
    kw = dict(a.split("=", 1) for a in (args_kv or []))
    return getattr(importlib.import_module(mod), fac)(**kw)


def add_commands(sub) -> None:
    s = sub.add_parser("serve", help="run the harness server for one run folder")
    s.add_argument("--run-dir", required=True)
    s.add_argument("--budget", type=int, required=True)
    s.add_argument("--max-exec", type=int, default=300)
    s.add_argument("--backend", required=True, help="module:factory")
    s.add_argument("--backend-arg", action="append", help="key=value passed to the factory")
    s.add_argument("--project")
    s.add_argument("--frozen", action="append", default=[], help="fnmatch pattern of frozen tensors")
    s.add_argument("--exec-timeout", type=float, default=900)
    s.add_argument("--max-images", type=int, default=256)
    s.add_argument("--deadline-min", type=float, help="auto-finalize after this many minutes")
    s.add_argument("--seed-recipe", help="path to a recipe.py from another project/run; copied into "
                   "work/seed_recipe.py as a warm start for the agent")
    s.add_argument("--seed-recipe-source", help="label of --seed-recipe's source project shown to the agent "
                   "(default: inferred from its path, runs/<project>/<run>/recipe.py)")
    _add_flags(s)

    e = sub.add_parser("exec", help="run code in the run's kernel")
    e.add_argument("run")
    e.add_argument("code_or_dash", nargs="?", help="'-' reads code from stdin")
    e.add_argument("-c", dest="code")
    e.add_argument("--file")
    e.add_argument("--timeout", type=float, default=1000)

    f = sub.add_parser("fwd", help="spend one forward pass (generate or read)")
    f.add_argument("run")
    f.add_argument("--json", dest="js")
    f.add_argument("--json-file")
    f.add_argument("--timeout", type=float, default=7200)

    st = sub.add_parser("status", help="passes and code runs used")
    st.add_argument("run")

    fi = sub.add_parser("final", help="submit the saved edits and end the run")
    fi.add_argument("run")
    fi.add_argument("--summary-file")
    fi.add_argument("--summary")

    w = sub.add_parser("wait", help="wait for the answer to a timed-out request")
    w.add_argument("run")
    w.add_argument("id")
    w.add_argument("--timeout", type=float, default=7200)

    p = sub.add_parser("prompt", help="write AGENT_PROMPT.md without starting a server")
    p.add_argument("--project", required=True)
    p.add_argument("--budget", type=int, required=True)
    p.add_argument("--max-exec", type=int, default=300)
    p.add_argument("--run-dir")
    p.add_argument("--backend")
    p.add_argument("--backend-arg", action="append")
    p.add_argument("--describe-file")
    p.add_argument("--classes", help="comma separated; overrides backend.project_info()")
    p.add_argument("--max-images", type=int, default=256)
    p.add_argument("--exec-timeout", type=float, default=900)
    p.add_argument("--wrapped", nargs="?", const="-", metavar="FILE",
                   help="needs --run-dir: print the full text an agent gets (run wrapper + task) to stdout, or to FILE")
    _add_flags(p)


def _add_flags(p) -> None:
    p.add_argument("--overlays", action="store_true", help="draw the train errors of every generate (backend.render_errors)")
    p.add_argument("--overlay-sheet", action="store_true", help="needs --overlays: also tile the worst-scored "
                   "images of each generate into one labelled sheet.png")
    p.add_argument("--retention", action="store_true",
                   help="fwd {\"which\": \"retain\"} scores the original task; the prompt asks to keep it within 1 point")
    p.add_argument("--recipe", action="store_true", help="the agent also submits WORK/recipe.py; checked at final")
    p.add_argument("--hide-names", action="store_true", help="class names become class_<k> in all the agent sees")
    p.add_argument("--permute-names", action="store_true",
                   help="class names become a seeded derangement of the real names (seed: --backend-arg seed=S); "
                        "mutually exclusive with --hide-names")
    p.add_argument("--reading", action="store_true",
                   help="reading mode: no head to edit, the agent names the backend's shuffled class rows "
                        "(backend.reading_task()); cannot combine with the other flags"
                   if has_module("harness.reading") else argparse.SUPPRESS)


def _run_file(run, path) -> Path:
    """A file the client sends must be inside the run folder, or the client would read any file
    for an agent whose shell may only call these commands (and echo it back in an error)."""
    p, r = Path(path).resolve(), Path(run).resolve()
    if r not in p.parents:
        raise ClientError(f"{path} is not inside the run folder {r}; put the file in {r / 'work'}")
    return p


def run(a) -> int:
    try:
        if a.cmd == "serve":
            from .server import Server

            backend = load_backend(a.backend, a.backend_arg)
            backend_args = dict(x.split("=", 1) for x in (a.backend_arg or []))
            meta = {"backend": a.backend, "backend_args": backend_args}
            srv = Server(a.run_dir, backend, a.budget, a.max_exec, a.exec_timeout, max_images=a.max_images,
                         frozen=a.frozen, deadline_min=a.deadline_min, project=a.project, overlays=a.overlays,
                         retention=a.retention, recipe=a.recipe, hide_names=a.hide_names,
                         permute_names=a.permute_names, permute_seed=backend_args.get("seed"),
                         reading=a.reading, overlay_sheet=a.overlay_sheet, seed_recipe=a.seed_recipe,
                         seed_recipe_source=a.seed_recipe_source, meta=meta)
            srv.serve()
            return 0
        if a.cmd == "exec":
            if a.code is not None:
                code = a.code
            elif a.file:
                code = _run_file(a.run, a.file).read_text()
            elif a.code_or_dash == "-":
                code = sys.stdin.read()
            else:
                raise ClientError("give -c CODE, --file F, or '-' for stdin")
            return client.print_resp(client.send(a.run, {"kind": "exec", "code": code}, a.timeout))
        if a.cmd == "fwd":
            raw = a.js if a.js is not None else (_run_file(a.run, a.json_file).read_text() if a.json_file else None)
            if raw is None:
                raise ClientError("give --json '{...}' or --json-file F")
            try:
                spec = json.loads(raw)
            except ValueError as ex:
                raise ClientError(f"request is not valid JSON: {ex}") from None
            return client.print_resp(client.send(a.run, {"kind": "fwd", "request": spec}, a.timeout))
        if a.cmd == "status":
            return client.status(a.run)
        if a.cmd == "wait":
            return client.print_resp(client.wait(a.run, a.id, a.timeout))
        if a.cmd == "final":
            summary = _run_file(a.run, a.summary_file).read_text() if a.summary_file else a.summary
            return client.final(a.run, summary)
        if a.cmd == "prompt":
            return _prompt(a)
    except ClientError as ex:
        print(f"ERROR: {ex}")
        return 2
    return 0


def _prompt(a) -> int:
    from .prompt import build_prompt, render_prompt

    if a.reading:
        raise ClientError("--reading is only for serve, not prompt (it needs a live run folder)")
    describe, classes, task, shot_kw, be = "", [], "", {}, None
    if a.describe_file:
        describe = Path(a.describe_file).read_text()
    if a.backend:
        be = load_backend(a.backend, a.backend_arg)
        describe = describe or be.describe()
        info = be.project_info()
        classes, task = info.get("classes", []), info.get("task", "")
        shot_kw = {"n_shot": info.get("n_shot"), "gate_open": info.get("gate_open", False),
                   "model_intro": info.get("model_intro"), "metric": info.get("metric"),
                   "metric_how": info.get("metric_how")}
    if not describe:
        raise ClientError("need --backend or --describe-file")
    if a.classes:
        classes = [c.strip() for c in a.classes.split(",") if c.strip()]
    if a.hide_names and a.permute_names:
        raise ClientError("--hide-names and --permute-names are mutually exclusive")
    if a.hide_names or a.permute_names:
        from .names import Names

        secret = be.secret_words() if be else []
        if a.permute_names:
            seed = dict(x.split("=", 1) for x in (a.backend_arg or [])).get("seed")
            if seed is None:
                raise ClientError("--permute-names needs the run's seed (--backend-arg seed=S)")
            nm = Names(classes, secret=secret, permute=True, seed=seed)
        else:
            nm = Names(classes, secret=secret)
        describe, task, classes = nm.text(describe), nm.text(task), [nm.text(c) for c in classes]
    if a.retention and be is None:
        raise ClientError("--retention needs --backend")
    retain = be.retention(None) if a.retention else None
    if a.retention and retain is None:
        raise ClientError("--retention: this backend has no held-out set of its original task (retention() gives None)")
    shot_kw.update(overlays=a.overlays, overlay_legend=be.overlay_legend if be else "", overlay_sheet=a.overlay_sheet,
                   retention=retain["about"] if retain else None, recipe=a.recipe, hide_names=a.hide_names)
    if a.wrapped and not a.run_dir:
        raise ClientError("--wrapped needs --run-dir")
    run_dir = Path(a.run_dir).resolve() if a.run_dir else Path("<RUN>")
    text = build_prompt(a.project, a.budget, a.max_exec, run_dir, describe, classes, task,
                        a.exec_timeout, 10_000, a.max_images, **shot_kw)
    if a.run_dir:
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "AGENT_PROMPT.md").write_text(text)
        if a.wrapped:
            full = render_prompt(run_dir, a.project, a.budget, a.max_exec)
            if a.wrapped == "-":
                print(full)
            else:
                Path(a.wrapped).write_text(full)
            return 0
        print(run_dir / "AGENT_PROMPT.md")
    else:
        print(text)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog=f"python -m {PKG}.harness")
    add_commands(ap.add_subparsers(dest="cmd", required=True))
    return run(ap.parse_args(argv))
