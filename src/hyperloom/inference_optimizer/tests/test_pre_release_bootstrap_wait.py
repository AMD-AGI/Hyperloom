# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Drives bootstrap's leg wait loop against real state.json files.

The loop decides when a leg is finished, and getting that wrong is expensive:
`run_leg` returning ends the pod. stop_reason is stamped on the transition INTO
CLOSE and reports/final.json is written by its first step, so a loop that exits
on either one tears the pod down while the closeout is still running. That is
not hypothetical -- two 12h baremetal legs died 9s and 20s after their
stop_reason appeared, both carrying the highest validated gains of that run and
neither leaving a finished report.

Only the "baremetal + kernel gain + 12h" combination reproduces it live: the
docker legs run their optimizer in a nested container that outlives run_leg, and
the 3h legs are --no-kernel so their closeout finishes inside one poll interval.
Rather than wait twelve hours for that combination, this extracts the loop and
feeds it the state.json sequence the doomed legs actually wrote.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

import pytest

_HERE = Path(__file__).resolve()


def _github_root() -> Path | None:
    for parent in _HERE.parents:
        candidate = parent / ".github"
        if candidate.is_dir():
            return candidate
    return None


_GITHUB = _github_root()

pytestmark = [
    pytest.mark.skipif(_GITHUB is None, reason="needs the source checkout (.github/)"),
    pytest.mark.skipif(shutil.which("jq") is None, reason="the wait loop reads state.json with jq"),
]


def _wait_loop() -> str:
    """Return the real wait loop, from its `while` down to the matching `done`."""
    assert _GITHUB is not None
    text = (_GITHUB / "pre-release" / "bootstrap-pre-release.sh").read_text(encoding="utf-8")
    body = text.split("run_leg() {", 1)[1]
    # Anchored on the newline: the same condition appears one level deeper inside
    # the session-discovery branch, and an unanchored match lands there instead.
    start = body.index('\n    if [ -n "$real_sdir" ]; then') + 1
    end = body.index("\n  done\n", start)
    return "  while :; do\n" + body[start:end] + "\n  done\n"


def _clean_stop_reason_fn() -> str:
    """Return the script's own clean-terminal predicate.

    Restating the vocabulary in the harness would let it drift from the script
    and assert a classification the script does not actually make.
    """
    assert _GITHUB is not None
    text = (_GITHUB / "pre-release" / "bootstrap-pre-release.sh").read_text(encoding="utf-8")
    m = re.search(r"^is_clean_stop_reason\(\) \{\n.*?^\}", text, re.S | re.M)
    assert m, "could not locate is_clean_stop_reason() in bootstrap"
    return m.group(0)


_HARNESS = """\
set -euo pipefail
log() {{ echo "[log] $*"; }}
publish_state_for_poll() {{ :; }}
{clean_fn}
# Each pass advances the clock and swaps in the next state.json, so the loop sees
# the same sequence a real closeout writes instead of one frozen snapshot.
passes=0
sleep() {{
  passes=$(( passes + 1 ))
  elapsed=$(( elapsed + 1 ))
  if [ -f "{states}/$passes.json" ]; then cp "{states}/$passes.json" "{state_json}"; fi
  if [ "$passes" -ge {max_passes} ]; then echo "[log] STILL_WAITING"; exit 42; fi
}}
leg=testleg
session={session}
real_sdir={real_sdir}
state_json={state_json}
final_json={final_json}
elapsed=0
deadline_s=100000
wait_interval=0

wait_for_leg() {{
{loop}
}}
wait_for_leg
rc=$?
echo "[log] RETURNED $rc"
exit $rc
"""


def _run(tmp_path: Path, *, snapshots: list[dict], final_json: bool, loop: str) -> tuple[int, str]:
    """Feed `snapshots` to the loop one pass at a time; returns (exit code, output)."""
    real_sdir = tmp_path / "sdir"
    (real_sdir / "reports").mkdir(parents=True)
    state_json = real_sdir / "state.json"
    final = real_sdir / "reports" / "final.json"
    if final_json:
        final.write_text("{}", encoding="utf-8")

    states = tmp_path / "states"
    states.mkdir()
    state_json.write_text(json.dumps(snapshots[0]), encoding="utf-8")
    for i, snap in enumerate(snapshots[1:], start=1):
        (states / f"{i}.json").write_text(json.dumps(snap), encoding="utf-8")

    script = tmp_path / "harness.sh"
    script.write_text(
        _HARNESS.format(
            states=states,
            state_json=state_json,
            session=tmp_path / "session",
            real_sdir=real_sdir,
            final_json=final,
            max_passes=len(snapshots) + 2,
            clean_fn=_clean_stop_reason_fn(),
            loop=loop,
        ),
        encoding="utf-8",
    )
    proc = subprocess.run(["bash", str(script)], text=True, capture_output=True, check=False)
    return proc.returncode, proc.stdout + proc.stderr


# The sequence baremetal-sglang-12h wrote: sweep_done lands on the transition into
# CLOSE, and close_sequence_done only flips once the seven-step sequence ends.
_MID_CLOSE = {"stop_reason": "sweep_done", "close_sequence_done": False}
_CLOSED = {"stop_reason": "sweep_done", "close_sequence_done": True}


def test_the_loop_waits_out_the_closeout_before_ending_the_leg(tmp_path: Path) -> None:
    """Pin the pass the loop returns on, not merely that it did not return early.

    The loop logs the pass it finished on, and the harness advances that counter
    once per pass, so the number says how long it waited. Asserting it exactly
    proves the loop read all three snapshots and returned on the one that flipped
    close_sequence_done -- an assertion that it "did not return" would also hold
    if the loop never managed to read state at all.
    """
    code, out = _run(tmp_path, snapshots=[_MID_CLOSE, _MID_CLOSE, _CLOSED], final_json=False, loop=_wait_loop())
    assert code == 0, f"the loop never finished:\n{out}"
    assert "[log] leg testleg state.json stop_reason='sweep_done' after 2s; demo complete" in out


def test_final_json_mid_closeout_does_not_end_the_leg(tmp_path: Path) -> None:
    """final.json exists from CLOSE step 1 onward, long before the sequence ends."""
    code, out = _run(tmp_path, snapshots=[_MID_CLOSE, _MID_CLOSE, _CLOSED], final_json=True, loop=_wait_loop())
    assert code == 0, f"the loop never finished:\n{out}"
    assert "[log] leg testleg final.json present and close sequence done after 2s; demo complete" in out


def test_the_leg_ends_once_the_close_sequence_is_done(tmp_path: Path) -> None:
    code, out = _run(tmp_path, snapshots=[_MID_CLOSE, _CLOSED], final_json=True, loop=_wait_loop())
    assert code == 0, f"the loop did not finish after close_sequence_done:\n{out}"
    assert "RETURNED 0" in out


def test_a_dirty_stop_reason_still_fails_the_leg(tmp_path: Path) -> None:
    code, out = _run(
        tmp_path,
        snapshots=[{"stop_reason": "baseline_failed", "close_sequence_done": True}],
        final_json=False,
        loop=_wait_loop(),
    )
    assert code == 1, f"a non-clean terminal must fail the leg:\n{out}"


def test_the_previous_loop_ended_the_leg_mid_closeout(tmp_path: Path) -> None:
    """The regression this guards: the pre-fix loop exits while CLOSE is running."""
    previous = textwrap.dedent("""\
      while :; do
        if [ -n "$real_sdir" ]; then
          if [ -f "$final_json" ]; then
            log "leg $leg final.json present after ${elapsed}s; demo complete"
            return 0
          fi
          stop=""
          [ -f "$state_json" ] && stop="$(jq -r '.stop_reason // ""' "$state_json" 2>/dev/null || echo "")"
          if [ -n "$stop" ]; then
            if is_clean_stop_reason "$stop"; then
              log "leg $leg state.json stop_reason='$stop' after ${elapsed}s; demo complete"
              return 0
            fi
            return 1
          fi
        fi
        sleep "$wait_interval"
      done
    """)
    code, out = _run(tmp_path, snapshots=[_MID_CLOSE, _MID_CLOSE], final_json=False, loop=previous)
    assert code == 0, f"expected the pre-fix loop to end the leg mid-closeout, got:\n{out}"
    assert "RETURNED 0" in out
