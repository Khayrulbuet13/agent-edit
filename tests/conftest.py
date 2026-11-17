import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PACK_MODE = False  # True when the data root was taken from this tree, not from the environment


def pytest_configure(config):
    """A tree that carries data/manifests needs no setup: point the data root at it."""
    global PACK_MODE
    pack = ROOT / "data"
    if not os.environ.get("AGENT_EDIT_DATA") and (pack / "manifests").is_dir():
        os.environ["AGENT_EDIT_DATA"] = str(pack)
        PACK_MODE = True


@pytest.fixture(autouse=True)
def _empty_roots(monkeypatch, tmp_path_factory):
    """With only the manifests at hand, point the image and cache roots at empty folders.

    Manifests and backends build paths from these roots; a test that needs a file under them finds
    nothing and skips."""
    if PACK_MODE:
        for var, name in (("AGENT_EDIT_DATA_RAW", "raw"), ("AGENT_EDIT_CACHE", "cache")):
            if not os.environ.get(var):
                monkeypatch.setenv(var, str(tmp_path_factory.getbasetemp() / f"empty_{name}"))


def need_manifest(*projects):
    """Skip the calling test unless every named project has its manifest on disk."""
    from agent_edit import config

    try:
        root = config.data_dir() / "manifests"
    except config.ConfigError:
        pytest.skip("no data root: set AGENT_EDIT_DATA to a folder with manifests/")
    for project in projects:
        if not (root / f"{project}.json").exists():
            pytest.skip(f"no manifest for {project} under {root}")


@pytest.fixture
def make_server(tmp_path):
    """An in-process harness Server on the toy backend; requests go straight to its handler."""
    from agent_edit.backends.fake import FakeBackend
    from agent_edit.harness.server import Server

    made = []

    def make(backend=None, name="run", budget=6, max_exec=20, **flags):
        srv = Server(tmp_path / name, backend or FakeBackend(), budget, max_exec, exec_timeout=60, max_images=64,
                     project="toy", meta={"backend": "agent_edit.backends.fake:FakeBackend", "backend_args": {}}, **flags)
        made.append(srv)
        return srv

    yield make
    for srv in made:
        srv._release_lock()


def ask(srv, kind, **kw):
    """-> (ok, text, done)"""
    resp, done = srv._handle({"kind": kind, **kw})
    return resp["ok"], resp["text"], done


def fwd(srv, **spec):
    return ask(srv, "fwd", request=spec)


def result(srv, k):
    return json.loads((srv.run / "work" / f"fwd_{k}" / "result.json").read_text())
