# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""GEAK's kernel wins must reach the stream the dashboard reads.

The TOP Model dashboard's GEAK column is
``summary_by_source["kernel_agent"]["by_backend"]["geak"]``, which
``collect_recorded_optimizations`` builds from the recorder's ``operations`` and
``adoptions`` streams. ``_record_geak_kernel_journey`` used to replay every GEAK
kernel with ``route_strategy="legacy_only"``, and all three recorder entry
points return *before* writing either record on that route. A session therefore
finished with a kept, validated, positive-gain kernel and no operation naming
it, which is the same shape a session with no kernel agent at all leaves.

These tests pin the path end to end so the column cannot silently go quiet
again.
"""

from __future__ import annotations

import json
from pathlib import Path

from hyperloom.inference_optimizer.breakdown import collectors
from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts, instrument
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.state.shared_state import SharedState

BASELINE_TPUT = 1000.0


def _coord(tmp_path: Path) -> Coordinator:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        baseline_tput=BASELINE_TPUT,
        current_best={"action": "explore", "tput": BASELINE_TPUT},
        model_path="/models/gemma",
        gpu_type="mi300x",
        isl=1024,
        osl=1024,
        conc=64,
    )
    return coord


def _journey(tmp_path: Path, kernels: list[dict]) -> str:
    path = tmp_path / "kernel_journey.json"
    path.write_text(
        json.dumps(
            {
                "discovery_runs": [
                    {"source": "profile", "status": "success", "hot_kernels": []}
                ],
                "kernels": kernels,
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _kernel(kid: str, *, gain: float, before: float, after: float) -> dict:
    return {
        "kernel_id": kid,
        "dispatch": {"dispatched": True, "backends": ["geak"], "skip_reason": ""},
        "backend_result": {
            "kernel_id": kid,
            "attempts": [
                {
                    "backend": "geak",
                    "attempt_id": f"{kid}-geak-0",
                    "status": "succeeded",
                    "decision": "KEEP",
                    "compile_passed": True,
                    "correctness_passed": True,
                }
            ],
            "verification": {"best_attempt_id": f"{kid}-geak-0", "best_backend": "geak"},
        },
        "e2e": {
            "kernel_id": kid,
            "integrated": True,
            "e2e_gain_pct": gain,
            "validated": True,
            "decision": "KEEP",
            # The pair the delta was judged against. GEAK publishes these so
            # the collector can state each win in percentage points of the one
            # session baseline instead of summing percentages of moving ones.
            "base_tput": before,
            "new_tput": after,
        },
    }


def _column(session_dir: Path) -> dict:
    """Run the real assembler and the real collector, and return the row the
    dashboard renders for the kernel agent."""
    warnings: list[str] = []
    parts = assemble_parts(session_dir, warnings=warnings)
    out = collectors.collect_recorded_optimizations(
        "session",
        [r for r in parts.get("operations") or [] if isinstance(r, dict)],
        [r for r in parts.get("measurements") or [] if isinstance(r, dict)],
        [r for r in parts.get("adoptions") or [] if isinstance(r, dict)],
        [r for r in parts.get("artifacts") or [] if isinstance(r, dict)],
        [],
        [],
        warnings,
    )
    return (out.get("summary_by_source") or {}).get("kernel_agent") or {}


def _record_baseline(session_dir: Path) -> None:
    """Every real session measures its baseline before the kernel agent runs;
    it is the denominator every gain below is stated against."""
    instrument.record_action_operation(
        session_dir,
        action="baseline",
        task_id="baseline-0",
        status="succeeded",
        decision="KEEP",
        result={"baseline_tput": BASELINE_TPUT, "ts": "2026-01-01T00:00:00Z"},
    )


def test_kept_geak_kernel_reaches_the_kernel_agent_column(tmp_path: Path) -> None:
    """A kept GEAK kernel is credited to the ``geak`` backend, not dropped."""
    coord = _coord(tmp_path)
    _record_baseline(tmp_path)
    result = {
        "status": "ok",
        "kernel_journey_path": _journey(
            tmp_path,
            [_kernel("fast_attn", gain=12.0, before=1000.0, after=1120.0)],
        ),
    }
    coord._record_geak_kernel_journey(result)

    parts = assemble_parts(tmp_path)
    operations = [r for r in parts.get("operations") or [] if isinstance(r, dict)]
    # The regression itself: on the legacy route this list was empty, which is
    # indistinguishable from a session whose records were lost.
    assert operations, "GEAK wrote no operation; the dashboard cannot see it"

    column = _column(tmp_path)
    assert column.get("keeps") == 1
    geak = (column.get("by_backend") or {}).get("geak") or {}
    assert geak.get("keeps") == 1
    assert geak.get("total_gain_pct") > 0.0
    # Credited to GEAK by name, never parked in the catch-all bucket.
    assert not (column.get("by_backend") or {}).get("unattributed", {}).get("keeps")


def test_gain_is_stated_in_points_of_the_session_baseline(tmp_path: Path) -> None:
    """Two stacked wins sum to the total the workload actually moved.

    Each kernel's own percentage is measured against wherever the previous one
    left off, so the percentages do not compose. Published with their
    throughput pair they are converted to points of the one session baseline,
    and those do.
    """
    coord = _coord(tmp_path)
    _record_baseline(tmp_path)
    coord._record_geak_kernel_journey(
        {
            "status": "ok",
            "kernel_journey_path": _journey(
                tmp_path,
                [
                    # +10% of 1000, then +10% of 1100. Naively summed that reads
                    # as 20%; the workload actually moved 21 points.
                    _kernel("k1", gain=10.0, before=1000.0, after=1100.0),
                    _kernel("k2", gain=10.0, before=1100.0, after=1210.0),
                ],
            ),
        }
    )
    geak = (_column(tmp_path).get("by_backend") or {}).get("geak") or {}
    assert geak.get("keeps") == 2
    assert geak.get("total_gain_pct") == 21.0


def test_reverted_geak_kernel_is_not_credited(tmp_path: Path) -> None:
    """The revert path is on the canonical stream too, so a kernel that was
    taken back out does not keep the credit it was given."""
    coord = _coord(tmp_path)
    _record_baseline(tmp_path)
    kernel = _kernel("regressed", gain=-3.0, before=1000.0, after=970.0)
    kernel["e2e"].update(integrated=False, validated=False, decision="REVERT")
    coord._record_geak_kernel_journey(
        {"status": "ok", "kernel_journey_path": _journey(tmp_path, [kernel])}
    )
    geak = (_column(tmp_path).get("by_backend") or {}).get("geak") or {}
    assert not geak.get("keeps")
