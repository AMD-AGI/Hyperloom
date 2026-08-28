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
import re
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
    assert 'idle="$(leg_idle_s "$root" "$grace_ts" "$now")"' in script
    assert 'no file written under $root' in script


def test_agent_turns_are_mirrored_to_nfs(script: str) -> None:
    """SaFE deletes a failed leg's pod, so every agent turn must reach NFS."""
    assert 'agent_log="${session}/agent-${leg}.log"' in script
    # Every turn goes through the one helper, which is the only place that tees, so no
    # invocation can bypass the transcript.
    assert script.count('tee -a "$alog"') == 1
    invocations = [
        line
        for line in script.splitlines()
        if "claude --print --dangerously-skip-permissions" in line
        and not line.lstrip().startswith("#")
    ]
    assert invocations == ['  claude --print --dangerously-skip-permissions "$@" 2>&1 | tee -a "$alog"']


def _leg_run_started(script: str, session: Path) -> bool:
    """Run the real leg_run_started() against a session tree."""
    lines = script.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("leg_run_started() {"))
    end = next(i for i, line in enumerate(lines[start:], start) if line == "}")
    fn = "\n".join(lines[start : end + 1])
    proc = subprocess.run(
        ["bash", "-c", f'{fn}\nleg_run_started "$1"', "_", str(session)],
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0


def test_launch_detection_needs_the_nested_run_dir(script: str, tmp_path: Path) -> None:
    """optimize writes state.json into $session/<model>/<ts>-<rand>/, never at the top."""
    session = tmp_path / "session"
    session.mkdir()
    assert not _leg_run_started(script, session)

    # A stray state.json directly under the session must not count as a launch.
    (session / "state.json").write_text("{}", encoding="utf-8")
    assert not _leg_run_started(script, session)

    nested = session / "Qwen3-8B" / "20260828T053223Z-4351891f"
    nested.mkdir(parents=True)
    (nested / "state.json").write_text("{}", encoding="utf-8")
    assert _leg_run_started(script, session)


def test_setup_marker_matches_every_setup_prompt(script: str) -> None:
    """The retry gate greps a literal the prompts must actually ask the agent to print."""
    assert 'grep -qiE "setup complete: ${run_mode}/${backend}" "$agent_log"' in script
    prompts = _BOOTSTRAP.parent / "prompts" / "pre-release"  # type: ignore[union-attr]
    for run_mode in ("baremetal", "docker"):
        for backend in ("vllm", "sglang"):
            prompt = prompts / f"setup-{run_mode}-{backend}.md"
            assert f"setup complete: {run_mode}/{backend}" in prompt.read_text(encoding="utf-8")


def test_an_early_turn_is_re_driven_not_fatal(script: str) -> None:
    """A turn that ends without finishing leaves nothing running; ask again, bounded."""
    assert 'max_demo_redrives="${LEG_DEMO_REDRIVES:-2}"' in script
    assert 'demo_redrives=$(( demo_redrives + 1 ))' in script


def test_setup_is_budgeted_in_time_not_in_turns(script: str) -> None:
    """A count-based cap of 3 turns was ~4min of wall clock and killed live installs.

    A framework install runs 10-30min, and each turn ends after ~60-90s, so the budget
    has to be a deadline plus a liveness check -- never a small turn count.
    """
    assert "LEG_TURN_ATTEMPTS" not in script
    assert 'setup_deadline_s="${LEG_SETUP_DEADLINE_S:-2700}"' in script
    assert 'setup_stall_s="${LEG_SETUP_STALL_S:-600}"' in script
    assert 'sidle="$(leg_idle_s "$root" "$setup_t0" "$snow")"' in script
    # The stall check must gate the failure, i.e. a progressing install is never reaped.
    assert 'if [ "$sidle" -ge "$setup_stall_s" ]; then' in script


def test_follow_up_turns_resume_the_same_conversation(script: str) -> None:
    """Re-feeding the prompt as a fresh turn throws away what the agent already knows.

    `--session-id` opens the leg's conversation and `--resume` continues it, so a
    follow-up turn still knows what it launched and which log it was watching.
    """
    assert 'agent_turn "$agent_log" --session-id "$uuid" < "$setup_prompt"' in script
    assert script.count('--resume "$uuid"') == 3  # setup nudge, demo turn, demo re-drive
    for nudge in ("SETUP_RESUME_NUDGE", "DEMO_RESUME_NUDGE"):
        assert f"{nudge}='" in script
    # Resuming can fail (no such session); that must degrade, not kill the leg.
    assert script.count("could not resume session") == 3


def test_leg_session_uuid_is_stable_and_well_formed(script: str, tmp_path: Path) -> None:
    """--session-id rejects anything that is not a UUID."""
    lines = script.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("leg_session_uuid() {"))
    end = next(i for i, line in enumerate(lines[start:], start) if line == "}")
    fn = "\n".join(lines[start : end + 1])

    def uuid_for(leg: str, version: str) -> str:
        proc = subprocess.run(
            ["bash", "-c", f'{fn}\nleg_session_uuid "$1" "$2"', "_", leg, version],
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()

    a = uuid_for("baremetal-vllm-3h", "1.0.0.dev1+ci")
    assert re.fullmatch(r"[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-8[0-9a-f]{3}-[0-9a-f]{12}", a)
    assert a == uuid_for("baremetal-vllm-3h", "1.0.0.dev1+ci")  # stable across turns
    # Distinct per leg (four legs share the docker host) and per run.
    assert a != uuid_for("baremetal-vllm-12h", "1.0.0.dev1+ci")
    assert a != uuid_for("baremetal-vllm-3h", "1.0.0.dev2+ci")


def test_re_drive_does_not_extend_the_hard_deadline(script: str) -> None:
    """The stall grace restarts per turn, but the pod-deadline clock must not.

    bootstrap's own deadline has to stay below the SaFE pod timeout, or SaFE pre-empts
    the pod mid-wait and the clean failure path is lost.
    """
    body = script.split("run_leg() {", 1)[1]
    assert body.count('start_ts="$(date +%s)"') == 1
    assert 'grace_ts="$(date +%s)"' in body
    assert 'elapsed=$(( now - start_ts ))' in body
    assert 'idle="$(leg_idle_s "$root" "$grace_ts" "$now")"' in body


def test_demo_prompts_forbid_ending_the_turn_early(script: str) -> None:
    prompts = _BOOTSTRAP.parent / "prompts" / "pre-release"  # type: ignore[union-attr]
    for hours in (3, 12):
        text = (prompts / f"demo-{hours}h.md").read_text(encoding="utf-8")
        assert "single non-interactive turn" in text
        assert "setsid nohup" in text
        assert "state.json" in text


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
