# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end proof that a non-graceful exit still yields final.json.

Launches a real subprocess that mirrors how ``cli.py`` runs the optimizer: it
installs an asyncio SIGINT/SIGTERM handler that lets the event loop unwind, and
calls ``write_minimal_final_json`` from a ``finally`` block. A real signal is
sent and the on-disk result asserted, pinning down which kill modes the fix
covers.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Worktree/repo root.
_REPO_ROOT = Path(__file__).resolve().parents[4]

# Miniature of the cli.py run+finally lifecycle: seed state.json, install the
# SIGTERM handler, block until signalled, flush the crash-safe final.json.
_HARNESS = """
import asyncio, signal, sys
from pathlib import Path

sd = Path(sys.argv[1])
from hyperloom.orchestrator.state.shared_state import SharedState
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
        from hyperloom.inference_optimizer.breakdown import write_minimal_final_json
        write_minimal_final_json(sd)

asyncio.run(main())
"""


def _child_env() -> dict[str, str]:
    """Environment whose ``hyperloom`` is this worktree's, not the installed one.

    Both children below assert something about *these* sources, so inheriting a
    bare environment would quietly test whichever copy is installed -- passing or
    failing for reasons that have nothing to do with the worktree. ``src`` is what
    makes the package importable under the src layout; the repo root stays for the
    top-level helpers the harness reaches for.
    """
    env = dict(os.environ)
    roots = (str(_REPO_ROOT / "src"), str(_REPO_ROOT))
    env["PYTHONPATH"] = os.pathsep.join((*roots, env.get("PYTHONPATH", "")))
    return env


def _spawn(session_dir: Path, install_handler: bool) -> subprocess.Popen:
    env = _child_env()
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
    """SIGTERM (the normal external-kill path) -> finally runs -> final.json exists."""
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

    Documents the boundary of the in-process fix: a -9 kill bypasses Python's
    finally entirely, so only an offline rebuild can recover here.
    """
    proc = _spawn(tmp_path, install_handler=True)
    proc.send_signal(signal.SIGKILL)
    proc.wait(timeout=30)

    final_json = tmp_path / "reports" / "final.json"
    assert not final_json.exists(), "SIGKILL bypasses finally; final.json cannot exist"


def test_crash_safe_platform_does_not_drag_in_the_orchestrator():
    """The safety net must not import the subsystems that may have just failed.

    The orchestrator's report renderer imports the message bus and a SQLite
    connection layer at module level, so reaching it for this record would
    acquire that dependency after a run has already died -- and if the crash
    came from that subsystem, the record explaining it is what breaks.
    Run in a subprocess because the import is global and one test cannot unsee it.
    """
    probe = (
        "import sys;"
        "from hyperloom.inference_optimizer.breakdown.exporter import _crash_safe_platform;"
        "import hyperloom;"
        "rec = _crash_safe_platform('mi355x');"
        "assert rec.get('status'), rec;"
        "heavy = [m for m in ("
        "  'hyperloom.orchestrator.actions.executors.report',"
        "  'hyperloom.orchestrator.bus.message_bus',"
        ") if m in sys.modules];"
        "print('FROM:' + (hyperloom.__file__ or ''));"
        "print('LEAKED:' + ','.join(heavy) if heavy else 'CLEAN')"
    )
    out = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(_REPO_ROOT),
        env=_child_env(),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert out.returncode == 0, out.stderr
    # Without this the child imports whichever ``hyperloom`` is installed, and a
    # green result would say nothing about the sources under test.
    origin = next(ln[len("FROM:") :] for ln in out.stdout.splitlines() if ln.startswith("FROM:"))
    assert origin.startswith(str(_REPO_ROOT / "src")), f"child imported {origin}"
    assert "CLEAN" in out.stdout, out.stdout
