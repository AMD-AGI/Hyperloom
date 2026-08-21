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


def _kernel_without_throughput_pair(kid: str, *, gain: float) -> dict:
    """The shape every real campaign artifact has today.

    All 36 KEEP blocks under ``/shared_nfs/hyperloom-claw`` publish
    ``e2e_gain_pct`` and no ``base_tput``/``new_tput``. The fixture above adds the
    pair, so it exercises the contract GEAK is moving to rather than the files on
    disk — and the difference decides whether a number may be summed.
    """
    kernel = _kernel(kid, gain=gain, before=0.0, after=0.0)
    kernel["e2e"].pop("base_tput", None)
    kernel["e2e"].pop("new_tput", None)
    return kernel


def test_keep_without_a_throughput_pair_is_visible_but_not_summed(tmp_path: Path) -> None:
    """Visible as a keep, absent from the total.

    A local ``e2e_gain_pct`` is measured against whatever baseline the executor
    held at the time. Projecting it as points of the session baseline is a unit
    error: replaying the real journeys that way summed 36 local deltas into
    +348.6 pp of a session that never moved that far. Withholding the gain must
    not also withhold the keep, or the column reads zero again — which is the
    bug the canonical-stream change was written to fix.
    """
    coord = _coord(tmp_path)
    _record_baseline(tmp_path)
    result = {
        "kernel_journey_path": _journey(
            tmp_path, [_kernel_without_throughput_pair("k_nopair", gain=29.994)]
        )
    }
    coord._record_geak_kernel_journey(result)

    geak = (_column(tmp_path).get("by_backend") or {}).get("geak") or {}
    assert geak.get("keeps") == 1, geak
    assert geak.get("non_attributable_keeps") == 1, geak
    assert geak.get("total_gain_pct") == 0.0, geak


def test_replayed_geak_kernel_is_not_parented_under_the_forge_route(tmp_path: Path) -> None:
    """The tree must not assert a GEAK kernel ran beneath Forge.

    Dropping ``legacy_only`` alone leaves the default route, which names the
    parent operation ``kernel_agent_forge``. The per-kernel strategy is corrected
    to ``geak`` afterwards, so the dashboard column fills either way — but a
    reader walking parents to answer "which optimizer produced this kernel?"
    gets Forge, and a replay after a process restart can mint further Forge route
    operations for kernels Forge never dispatched.
    """
    coord = _coord(tmp_path)
    _record_baseline(tmp_path)
    result = {
        "kernel_journey_path": _journey(
            tmp_path, [_kernel("k_route", gain=5.0, before=1000.0, after=1050.0)]
        )
    }
    coord._record_geak_kernel_journey(result)

    warnings: list[str] = []
    parts = assemble_parts(tmp_path, warnings=warnings)
    operations = [r for r in parts.get("operations") or [] if isinstance(r, dict)]
    names = {str(op.get("name") or "") for op in operations}
    assert "k_route" in names, sorted(names)
    assert "kernel_agent_forge" not in names, sorted(names)

    kernel_ops = [op for op in operations if str(op.get("name") or "") == "k_route"]
    assert kernel_ops, operations
    parents = {str(op.get("parent_operation_id") or "") for op in kernel_ops}
    geak_route_ids = {
        str(op.get("operation_id") or "")
        for op in operations
        if str(op.get("name") or "") == "geak"
    }
    assert geak_route_ids, sorted(names)
    assert parents <= geak_route_ids, (parents, geak_route_ids)


def _route_ops(session_dir: Path) -> tuple[set[str], set[str]]:
    """(operation names, route subject names) written under ``session_dir``."""
    warnings: list[str] = []
    parts = assemble_parts(session_dir, warnings=warnings)
    operations = [r for r in parts.get("operations") or [] if isinstance(r, dict)]
    names = {str(op.get("name") or "") for op in operations}
    subjects = {
        str((op.get("subject") or {}).get("name") or "")
        for op in operations
        if str((op.get("subject") or {}).get("subject_type") or "") == "kernel_optimizer_route"
    }
    return names, subjects


def test_reverting_a_geak_kernel_stays_on_the_geak_route(tmp_path: Path) -> None:
    """Withdrawing a kernel must not re-file it under another optimizer.

    HONEST SCOPE: this passes with and without the route argument on the revert
    call, because `record_kernel_e2e` mints no route operation on that path —
    checked both after a KEEP replay and standalone, as a restart would do. The
    review reported the revert falling back to Forge; the call did, but the
    fallback is inert today. The argument is still passed, because a call that
    does not name its route is one change inside the recorder away from being
    wrong, and this pins the outcome so that change cannot land quietly.

    The load-bearing check for the route argument itself is
    ``test_every_journey_replay_call_names_its_route``, which reads the source.
    """
    coord = _coord(tmp_path)
    _record_baseline(tmp_path)
    kernel = _kernel("k_revert_route", gain=5.0, before=1000.0, after=1050.0)
    coord._record_geak_kernel_journey(
        {"status": "ok", "kernel_journey_path": _journey(tmp_path, [kernel])}
    )
    # `_reject_*` lives on KernelPhase and is not among the methods Coordinator
    # delegates, so bind it directly rather than reaching through the facade.
    from hyperloom.orchestrator.phases.kernel import KernelPhase

    KernelPhase._reject_geak_kernel_journey(
        coord,
        {"status": "ok", "kernel_journey_path": _journey(tmp_path, [kernel])},
        measured_tput=BASELINE_TPUT,
        current_best_tput=BASELINE_TPUT * 1.2,
        provenance="revalidation",
    )

    names, _ = _route_ops(tmp_path)
    assert "kernel_agent_forge" not in names, sorted(names)
    geak = (_column(tmp_path).get("by_backend") or {}).get("geak") or {}
    assert not geak.get("keeps"), geak


def test_the_geak_route_subject_names_geak(tmp_path: Path) -> None:
    """Operation and subject must name the same optimizer.

    ``record_native_kernel_run_start`` derives the operation's name and strategy
    from the route, but the subject payload kept a literal ``kernel_agent_forge``.
    A GEAK replay then wrote ``operation.name=geak`` beside
    ``subject.name=kernel_agent_forge`` — one record naming two optimizers, and
    the subject is what identity lookups resolve against.
    """
    coord = _coord(tmp_path)
    _record_baseline(tmp_path)
    coord._record_geak_kernel_journey(
        {
            "kernel_journey_path": _journey(
                tmp_path, [_kernel("k_subject", gain=5.0, before=1000.0, after=1050.0)]
            )
        }
    )

    names, subjects = _route_ops(tmp_path)
    assert "geak" in names, sorted(names)
    assert subjects == {"geak"}, subjects


def test_every_journey_replay_call_names_its_route() -> None:
    """Derived from the source, not from a list a seventh call site can miss.

    Both route defects were the same shape: a recorder call that did not name its
    route and silently took Forge. Six call sites carry ``route_strategy="geak"``
    today; this fails if one is added without it, rather than waiting for a
    reviewer to notice the provenance is wrong.
    """
    import re

    source = Path(
        __file__
    ).resolve().parents[3] / "hyperloom" / "orchestrator" / "phases" / "kernel.py"
    text = source.read_text(encoding="utf-8")
    replay = text[text.index("def _record_geak_kernel_journey") :]
    unrouted = []
    for match in re.finditer(r"instrument\.record_kernel_(\w+)\(", replay):
        depth, i = 0, match.end() - 1
        while i < len(replay):
            if replay[i] == "(":
                depth += 1
            elif replay[i] == ")":
                depth -= 1
                if depth == 0:
                    break
            i += 1
        call = replay[match.start() : i + 1]
        if 'route_strategy="geak"' not in call:
            unrouted.append(
                (match.group(1), replay[: match.start()].count("\n"))
            )
    assert not unrouted, f"recorder calls in the GEAK replay with no GEAK route: {unrouted}"
