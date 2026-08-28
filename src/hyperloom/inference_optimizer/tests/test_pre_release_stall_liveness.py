# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression guard for the pre-release E2E leg liveness (stall) check.

``bootstrap-pre-release.sh`` blocks after the demo turn until ``optimize`` writes
``reports/final.json``, and declares a leg dead when nothing has been written for
``LEG_STALL_GRACE_S``. Two properties of that check killed legs that were provably
still alive (run 1.0.1a0.dev202608280354+ci, both baremetal-sglang legs):

* the idle window was measured against absolute file mtimes, so the minutes spent
  inside the two non-streaming ``claude --print`` turns were charged to the leg and
  the very first loop iteration condemned it;
* only ``$session`` was watched, while the agent's launcher, its setup/install logs
  and ``install.sh``'s caches land elsewhere under the leg root -- one leg was reaped
  26s after it last wrote a file.

These tests exercise the real ``leg_idle_s`` helper out of the script.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

_STALL_GRACE_S = 600


def _find_bootstrap() -> Path | None:
    """Locate the in-pod bootstrap; None when running from an installed wheel."""
    for parent in Path(__file__).resolve().parents:
        candidate = parent / ".github" / "pre-release" / "bootstrap-pre-release.sh"
        if candidate.is_file():
            return candidate
    return None


_BOOTSTRAP = _find_bootstrap()

pytestmark = pytest.mark.skipif(
    _BOOTSTRAP is None,
    reason="pre-release liveness guard needs the source checkout (.github/pre-release/)",
)


@pytest.fixture(scope="module")
def script() -> str:
    assert _BOOTSTRAP is not None
    return _BOOTSTRAP.read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def leg_idle_fn(script: str) -> str:
    """Slice the ``leg_idle_s`` function out of the script so it can be run alone."""
    lines = script.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("leg_idle_s() {"))
    end = next(i for i, line in enumerate(lines[start:], start) if line == "}")
    return "\n".join(lines[start : end + 1])


def _idle(leg_idle_fn: str, root: Path, loop_start: int, now: int) -> int:
    """Run leg_idle_s(root, loop_start, now) in bash and return its answer."""
    proc = subprocess.run(
        ["bash", "-c", f'{leg_idle_fn}\nleg_idle_s "$1" "$2" "$3"', "_", str(root), str(loop_start), str(now)],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(proc.stdout.strip())


def _write(path: Path, age_s: int, now: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    stamp = now - age_s
    os.utime(path, (stamp, stamp))


def test_pre_loop_gap_is_not_charged_to_the_leg(leg_idle_fn: str, tmp_path: Path) -> None:
    """The observed kill: nothing written for 629s, but the loop just started."""
    now = int(time.time())
    _write(tmp_path / "session" / ".session_dir", age_s=629, now=now)
    idle = _idle(leg_idle_fn, tmp_path, loop_start=now - 5, now=now)
    assert idle == 5
    assert idle < _STALL_GRACE_S


def test_writes_outside_the_session_dir_count_as_progress(leg_idle_fn: str, tmp_path: Path) -> None:
    """The agent's own logs live next to the workspace, not under session/."""
    now = int(time.time())
    loop_start = now - 700
    _write(tmp_path / "session" / ".session_dir", age_s=700, now=now)
    _write(tmp_path / "setup_sglang_retry.log", age_s=26, now=now)
    idle = _idle(leg_idle_fn, tmp_path, loop_start=loop_start, now=now)
    assert idle == 26
    assert idle < _STALL_GRACE_S


def test_a_genuinely_idle_tree_is_still_reaped(leg_idle_fn: str, tmp_path: Path) -> None:
    """A hung launch must still be caught once the loop itself has waited it out."""
    now = int(time.time())
    _write(tmp_path / "session" / ".session_dir", age_s=900, now=now)
    idle = _idle(leg_idle_fn, tmp_path, loop_start=now - 900, now=now)
    assert idle == 900
    assert idle >= _STALL_GRACE_S


def test_empty_tree_falls_back_to_the_loop_start(leg_idle_fn: str, tmp_path: Path) -> None:
    now = int(time.time())
    assert _idle(leg_idle_fn, tmp_path, loop_start=now - 42, now=now) == 42


def test_stall_check_watches_the_leg_root(script: str) -> None:
    """Pin the call site: the check must be scoped to $root, not $session."""
    assert 'idle="$(leg_idle_s "$root" "$start_ts" "$now")"' in script
    assert 'no file written under $root' in script


def test_agent_turns_are_mirrored_to_nfs(script: str) -> None:
    """SaFE deletes a failed leg's pod, so the agent transcript must reach NFS."""
    assert 'agent_log="${session}/agent-${leg}.log"' in script
    assert script.count('tee -a "$agent_log"') == 2


def test_dockerd_never_runs_on_vfs(script: str) -> None:
    """vfs copies every layer in full and evicted the host pod twice (200Gi, 1792Gi).

    Only deduplicating drivers may be attempted, and a leg with none available has to
    fail rather than quietly reproduce the eviction.
    """
    assert "--storage-driver=vfs" not in script
    drivers = next(
        line for line in script.splitlines() if line.startswith("DOCKER_DRIVERS=")
    )
    assert "overlay2" in drivers
    assert "vfs" not in drivers
