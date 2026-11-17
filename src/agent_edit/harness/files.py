"""Run-folder file helpers shared by server and client. Standard library only."""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path

HB_S = float(os.environ.get("AGENT_EDIT_HARNESS_HB_S", "5"))  # heartbeat period
STALE_S = float(os.environ.get("AGENT_EDIT_HARNESS_STALE_S", "30"))  # a lock older than this is dead
POLL_S = 0.25


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name("." + path.name + f".{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def safe_name(name: str) -> str:
    return name.replace("/", "%2F")


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def ensure_dirs(run: Path) -> None:
    for d in ("queue", "queue/claimed", "work", "edit", "log"):
        (run / d).mkdir(parents=True, exist_ok=True)
