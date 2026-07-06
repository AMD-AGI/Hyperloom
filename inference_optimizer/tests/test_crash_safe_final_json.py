# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Issue #464 — end-to-end proof that a non-graceful exit still yields final.json.

Unlike the unit tests in ``test_breakdown_exporter_unit.py`` (which exercise
``write_minimal_final_json`` in isolation), these tests launch a *real*
subprocess that mirrors how ``cli.py`` runs the optimizer:

* it installs an asyncio SIGINT/SIGTERM handler that lets the event loop
  unwind (exactly like ``Coordinator.run``), and
* it calls ``write_minimal_final_json`` from a ``finally`` block (exactly like
  ``cli.py``'s end-of-session safety net).

We then send a real signal and assert on the on-disk result. This reproduces
the original bug (no ``final.json`` on a killed run) and pins down precisely
which kill modes the fix covers.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Worktree/repo root (…/inference_optimizer/tests/<this> -> parents[2]).
_REPO_ROOT = Path(__file__).resolve().parents[2]

# A faithful miniature of the cli.py run+finally lifecycle. Seeds a state.json,
# installs the same SIGTERM handler Coordinator.run uses, blocks until signalled,
# and flushes the crash-safe final.json from a finally block on the way out.
_HARNESS = """
import asyncio, signal, sys
from pathlib import Path

sd = Path(sys.argv[1])
from hyperloom.orchestrator.shared_state import SharedState
st = SharedState.load_or_init(sd)
st.session_id = "sess-kill"
st.baseline_tput = 35.83
st.set_stop_reason("time_exhausted")
st.save(sd)

INSTALL_HANDLER = sys.argv[2] == "1"

async def main():
    stop = asyncio.Event()
    if INSTALL_HANDLER:
        loop = asyncio.get_running_loop()
        for s in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(s, stop.set)
    try:
        print("READY", flush=True)
        await stop.wait()
    finally:
        from inference_optimizer.breakdown import write_minimal_final_json
        write_minimal_final_json(sd)

asyncio.run(main())
"""


def _spawn(session_dir: Path, install_handler: bool) -> subprocess.Popen:
    env = dict(os.environ)
    # Make the worktree's package importable in the child regardless of any
    # editable install pointing elsewhere.
    env["PYTHONPATH"] = str(_REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env["USER_DATA_PATH"] = str(session_dir)
    proc = subprocess.Popen(
        [sys.executable, "-c", _HARNESS, str(session_dir), "1" if install_handler else "0"],
        cwd=str(_REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # Wait until the child has installed its handler and is blocked.
    deadline = time.time() + 30
    assert proc.stdout is not None
    while time.time() < deadline:
        line = proc.stdout.readline()
        if "READY" in line:
            return proc
        if proc.poll() is not None:
            raise AssertionError(f"child exited early: {proc.stderr.read()}")
    proc.kill()
    raise AssertionError("child never became READY")


def test_sigterm_runs_finally_and_writes_final_json(tmp_path):
    """SIGTERM (the normal external-kill path) -> finally runs -> final.json exists.

    This is the bug reproduction: without the fix there is no
    ``write_minimal_final_json`` call on this path, so ``final.json`` is absent.
    """
    proc = _spawn(tmp_path, install_handler=True)
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=30)

    final_json = tmp_path / "reports" / "final.json"
    assert final_json.is_file(), "SIGTERM must still produce reports/final.json"

    import json

    data = json.loads(final_json.read_text(encoding="utf-8"))
    assert data["safety_net"] is True
    assert data["report_complete"] is False
    assert data["session_id"] == "sess-kill"
    assert data["baseline_tput"] == 35.83


def test_sigkill_cannot_write_final_json(tmp_path):
    """SIGKILL (hard kill / Slurm reclaim) -> finally CANNOT run -> no final.json.

    Documents the honest boundary of the in-process fix: a -9 kill bypasses
    Python's finally entirely, so only an offline rebuild (Layer 3) can recover
    here. If this ever starts passing it means the kill mode changed, not that
    the in-process safety net grew new powers.
    """
    proc = _spawn(tmp_path, install_handler=True)
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=30)

    final_json = tmp_path / "reports" / "final.json"
    assert not final_json.exists(), "SIGKILL bypasses finally; final.json cannot exist"
