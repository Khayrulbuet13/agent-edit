"""The command line.

A command whose modules are missing from the tree is not registered: it does not show in --help
and its arguments are not accepted. COMMANDS is the registry; `config.has_module` decides what is
present.

Client commands (exec, fwd, status, final, wait) import no torch.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Callable
from dataclasses import dataclass

from . import config
from .config import CLI, PKG, has_module
from .harness import cli as harness_cli

# baseline kind -> the module that holds it
BASELINE_MODULES = {"c2": "baselines.probe", "c5r": "baselines.adam", "c5f": "baselines.decoder_ft",
                    "lwf": "baselines.decoder_ft"}
BASELINE_HELP = {"c2": "linear probe", "c5r": "cosine head (rows trained with Adam)",
                 "c5f": "full decoder fine-tuning", "lwf": "fine-tuning with a learning-without-forgetting term"}
SCORE_MODULES = {"pmt": "backends.pmt.evaluate"}
CACHE_MODULES = {"pmt": "backends.pmt.cache"}


def _present(table: dict) -> list[str]:
    return [k for k, mod in table.items() if has_module(mod)]


def _import(rel: str, what: str):
    if not has_module(rel):
        raise SystemExit(f"{what} is not available: this tree has no {rel}")
    return importlib.import_module(f"{PKG}.{rel}")


@dataclass(frozen=True)
class Command:
    """`needs`: module paths relative to the package. The command exists when any one of them is
    present, or always when `needs` is empty. `group`: the line of --help that lists it."""

    name: str
    add: Callable
    run: Callable
    needs: tuple[str, ...] = ()
    group: str = ""

    def present(self) -> bool:
        return not self.needs or any(has_module(m) for m in self.needs)


def _add_score(sub):
    s = sub.add_parser("score", help="score a finished run on the hidden eval set (eval.json)")
    s.add_argument("run", help="run folder, or <project>/<run> under $AGENT_EDIT_RUNS")
    s.add_argument("--out", help="where to write eval.json (default: inside the run folder)")
    s.add_argument("--model", default="auto", choices=["auto", *_present(SCORE_MODULES)],
                   help="auto: the model comes from the run's project and name")


def _run_score(a) -> int:
    _import(SCORE_MODULES["pmt"], "the PMT scorer").score_run(config.resolve_run(a.run), a.out)
    return 0


def _add_baselines(sub):
    kinds = _present(BASELINE_MODULES)
    b = sub.add_parser("baselines", help="no-agent baselines")
    b.add_argument("method", choices=kinds, help=", ".join(f"{k} {BASELINE_HELP[k]}" for k in kinds))
    b.add_argument("project")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--n", type=int, default=20)
    b.add_argument("--protocol", default="cv", choices=["cv"],
                   help="cv: 5-fold CV pick on the train images (one object per image sets)")
    b.add_argument("--fold", type=int, help="c5f: run only this CV fold (0..4), e.g. one GPU job per fold")
    b.add_argument("--order", type=int, default=0, help="c5f/lwf: batch-shuffle seed, independent of --seed (the data draw)")
    b.add_argument("--kd-weight", type=float, help="lwf: distillation weight against the stock decoder (its own grid knob)")
    b.add_argument("--out-dir", help="default $AGENT_EDIT_RESULTS/pmt_fg/baselines")


def _run_baselines(a) -> int:
    from .baselines import run as baselines

    baselines.main(a.method, a.project, a.seed, a.n, a.out_dir, a.fold, a.order, a.kd_weight)
    return 0


def _add_formula(sub):
    f = sub.add_parser("formula", help="run the formula baseline on PROJECT")
    f.add_argument("project")
    f.add_argument("--n", type=int, help="train shots N")
    f.add_argument("--seed", type=int)
    f.add_argument("--out", required=True, help="output folder")


def _run_formula(a) -> int:
    argv = [a.project]
    for flag, val in (("--n", a.n), ("--seed", a.seed), ("--out", a.out)):
        if val is not None:
            argv += [flag, str(val)]
    return int(_import("baselines.formula", "the formula baseline").main(argv) or 0)


def _add_retention(sub):
    r = sub.add_parser("retention", help="COCO mask mAP with a PMT run's rows appended (writes e2.json)")
    r.add_argument("runs", nargs="+")
    r.add_argument("--limit", type=int)


def _run_retention(a) -> int:
    from .analysis import retention

    retention.run([config.resolve_run(r) for r in a.runs], a.limit)
    return 0


def _add_cache(sub):
    ca = sub.add_parser("cache", help="build frozen-feature caches (GPU)")
    ca.add_argument("model", choices=_present(CACHE_MODULES))
    ca.add_argument("projects", nargs="*")
    ca.add_argument("--tile", type=int, default=0)
    ca.add_argument("--stride", type=int)
    ca.add_argument("--batch", type=int, default=0)
    ca.add_argument("--shard", default="0/1", help="i/n: this job takes every n-th image from i")
    ca.add_argument("--splits", default="train_pool,eval")


def _run_cache(a) -> int:
    from .backends.pmt import cache

    if not a.projects:
        raise SystemExit(f"usage: {CLI} cache pmt PROJECT [PROJECT ...] [--tile T --stride S]")
    stride = a.stride if a.stride is not None else a.tile // 2
    shard, nshards = (int(v) for v in a.shard.split("/"))
    for p in a.projects:
        cache.build(p, splits=tuple(a.splits.split(",")), tile=a.tile, stride=stride, batch=a.batch,
                    shard=shard, nshards=nshards)
    return 0


def _add_replay(sub):
    rp = sub.add_parser("replay", help="run a run's recipe.py on another train pool and score it (no agent)")
    rp.add_argument("run")
    rp.add_argument("--project", help="default: the run's project")
    rp.add_argument("--seed", type=int, help="default: the run's seed")
    rp.add_argument("--n", type=int, help="train shots N; default: the run's")
    rp.add_argument("--out", help="default RUN/replay/<project>_s<S>_N<N>.json")
    rp.add_argument("--no-eval", action="store_true", help="train score only")


def _run_replay(a) -> int:
    from .analysis import replay

    rec = replay.replay(config.resolve_run(a.run), a.project, a.seed, a.n, a.out, with_eval=not a.no_eval)
    return 0 if rec["ok"] else 1


def _add_env(sub):
    sub.add_parser("env", help="show the AGENT_EDIT_* paths")


def _run_env(a) -> int:
    for k, v in config.check().items():
        print(f"{k}={v or '(not set)'}")
    return 0


# `group` of a command -> its line in --help
GROUP_LABELS = {"scoring": "Scoring", "setup": "Setup"}

# The order is the order in --help.
COMMANDS = [
    Command("score", _add_score, _run_score, group="scoring"),
    Command("baselines", _add_baselines, _run_baselines, tuple(BASELINE_MODULES.values()), "scoring"),
    Command("formula", _add_formula, _run_formula, ("baselines.formula",), "scoring"),
    Command("retention", _add_retention, _run_retention, ("analysis.retention",), "scoring"),
    Command("cache", _add_cache, _run_cache, tuple(CACHE_MODULES.values()), "setup"),
    Command("replay", _add_replay, _run_replay, ("analysis.replay",), "scoring"),
    Command("env", _add_env, _run_env, group="setup"),
]


def _lab():
    """The module of lab-only commands, when this tree has one. Its COMMANDS is the whole registry,
    in the order of --help, and stands in for the one above."""
    if has_module("cli_lab"):
        from . import cli_lab

        return cli_lab


def active() -> list[Command]:
    lab = _lab()
    return [c for c in (lab.COMMANDS if lab else COMMANDS) if c.present()]


def description(cmds: list[Command], labels: dict[str, str]) -> str:
    lines = ["Harness (the agent's loop):  " + ", ".join(harness_cli.COMMANDS)]
    for group, label in labels.items():
        lines.append(f"{label + ':':<29}{', '.join(c.name for c in cmds if c.group == group)}")
    return (f"The `{CLI}` command.\n\n" + "\n".join(lines) + "\n\n"
            f"Run `{CLI} <command> -h` for options. Client commands (exec, fwd, status, final, wait) import no torch.\n")


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    lab = _lab()
    if lab and lab.passthrough(argv):
        return 0
    cmds = active()
    ap = argparse.ArgumentParser(prog=CLI, description=description(cmds, lab.GROUP_LABELS if lab else GROUP_LABELS),
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    harness_cli.add_commands(sub)
    for c in cmds:
        c.add(sub)
    a = ap.parse_args(argv)
    for c in cmds:
        if c.name == a.cmd:
            return c.run(a)
    return harness_cli.run(a)


if __name__ == "__main__":
    sys.exit(main())
