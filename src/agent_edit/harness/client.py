"""Client side of the file queue: exec, fwd, status, final, wait. Standard library only,
so the agent's login-node shell needs no torch."""

from __future__ import annotations

import contextlib
import json
import time
import uuid
from pathlib import Path

from ..config import PKG
from .files import STALE_S, read_json, write_atomic


class ClientError(Exception):
    pass


def server_state(run: Path):
    st = read_json(run / "state.json", None)
    lock = read_json(run / "serve.lock", None)
    age = (time.time() - float(lock["heartbeat"])) if lock else None
    return st, lock, age


def send(run: Path, req: dict, timeout: float):
    run = Path(run).resolve()
    q = run / "queue"
    if not q.is_dir():
        raise ClientError(f"{run} is not a run folder (no queue/)")
    st, lock, age = server_state(run)
    if st and st.get("finished"):
        raise ClientError("the run is finished; nothing more can be run.")
    if age is None or age > STALE_S:
        raise ClientError("the server is not running (no fresh heartbeat). Not sent.")
    rid = uuid.uuid4().hex[:12]
    req = {**req, "id": rid}
    name = f"req_{time.time_ns()}_{rid}.json"
    write_atomic(q / name, json.dumps(req))
    resp_path = q / f"resp_{rid}.json"
    t0, delay, last_check = time.time(), 0.15, time.time()
    while True:
        if resp_path.exists():
            resp = read_json(resp_path, None)
            if resp is not None:
                resp_path.unlink(missing_ok=True)
                return resp
        now = time.time()
        if now - t0 > timeout:
            raise ClientError(f"no answer after {timeout:.0f}s. The request (id {rid}) may still be "
                              f"running. Wait for it with: python -m {PKG}.harness wait {run} {rid}")
        if now - last_check > 5:
            last_check = now
            _, _, age = server_state(run)
            if age is None or age > 2 * STALE_S:
                with contextlib.suppress(OSError):
                    (q / name).unlink()
                raise ClientError("the server died while waiting (no heartbeat). Request withdrawn.")
        time.sleep(delay)
        delay = min(delay * 1.3, 1.5)


def wait(run: Path, rid: str, timeout: float):
    p = Path(run) / "queue" / f"resp_{rid}.json"
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = read_json(p, None)
        if r is not None:
            p.unlink(missing_ok=True)
            return r
        time.sleep(1)
    raise ClientError(f"still no answer after {timeout:.0f}s")


def print_resp(resp) -> int:
    print(resp.get("text", ""))
    return 0 if resp.get("ok") else 1


def status(run) -> int:
    run = Path(run).resolve()
    st, lock, age = server_state(run)
    if not st:
        print("no state.json: server never started here")
        return 1
    if st.get("finished"):
        srv = f"finished ({st.get('finish_reason')})"
    elif age is not None and age < STALE_S:
        srv = f"alive (host {lock['host']}, heartbeat {age:.0f}s ago)"
    else:
        srv = "NOT RUNNING (stale or no lock)"
    man = read_json(run / "edit" / "manifest.json", {})
    print(f"server: {srv}")
    print(f"passes: {st['passes_used']}/{st['budget']} used ({st['budget'] - st['passes_used']} left)")
    print(f"code runs: {st['exec_used']}/{st['max_exec']} used ({st['max_exec'] - st['exec_used']} left)")
    print(f"edited tensors: {man.get('n_edited_tensors', 0)}; changed weights: "
          f"{man.get('changed_weights', 0)} of {man.get('total_weights', '?')}")
    return 0


def final(run, summary) -> int:
    run = Path(run).resolve()
    st, _, _ = server_state(run)
    if st and st.get("finished"):
        fa = run / "final_answer.md"
        if summary and fa.exists() and "(placeholder" in fa.read_text():
            txt = fa.read_text()
            tail = txt[txt.rfind("\n---\n"):]
            fa.write_text("# Final answer\n\n" + summary.strip() + tail)
            print("run already finished; summary written to final_answer.md")
            return 0
        print("run already finished")
        return 1
    return print_resp(send(run, {"kind": "final", "summary": summary}, 600))
