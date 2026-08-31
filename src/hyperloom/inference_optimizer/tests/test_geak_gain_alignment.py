# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Alignment / credibility unit tests for the GEAK e2e gain path.

Covers the three coupling points that keep Hyperloom's reported gain honest and
consistent with GEAK's own e2e speedup:

  * #3 — the same-harness (2b) revalidation DECISION only stamps ``validated``
    when the ran config's identity matches AND the win actually engaged, else it
    hands off to the GEAK-harness (2a) fallback.
  * #5 — promote records a PROVISIONAL cross-harness gain that is internally
    consistent with ``current_best.tput`` / ``baseline_tput`` (cold-to-cold),
    never a hot-final-over-cold-baseline ratio, and never stamps ``validated``.
  * 2a — the GEAK-harness fallback validates using GEAK's OWN reported speedup
    (``throughput_speedup`` on the promoted basis), so Hyperloom's validated
    number equals GEAK's headline instead of an inflated hot A/B.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown.collectors.attribution import _geak_contribution
from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts
from hyperloom.inference_optimizer.breakdown.reporters._renderers.final import render as render_final
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.coordinator_helpers import (
    _geak_result_has_material,
    _geak_revalidation_decision,
    _normalize_geak_overlay_dir,
)
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task


# ── #3: same-harness (2b) revalidation decision ──────────────────────────────


def test_revalidation_validated_when_identity_and_engagement_hold() -> None:
    assert (
        _geak_revalidation_decision(
            measured=115.0,
            baseline=100.0,
            got_hash="abc",
            expected_hash="abc",
            min_engaged_gain_pct=2.0,
        )
        == "validated"
    )


def test_revalidation_fallback_on_config_identity_mismatch() -> None:
    # Engaged (15% > 2%) but the ran config's fingerprint drifted → fall back.
    assert (
        _geak_revalidation_decision(
            measured=115.0,
            baseline=100.0,
            got_hash="WRONG",
            expected_hash="abc",
            min_engaged_gain_pct=2.0,
        )
        == "fallback"
    )


def test_revalidation_fallback_when_not_engaged() -> None:
    # Identity matches but the win collapsed back to (near-)baseline → fall back.
    assert (
        _geak_revalidation_decision(
            measured=101.0,
            baseline=100.0,
            got_hash="abc",
            expected_hash="abc",
            min_engaged_gain_pct=2.0,
        )
        == "fallback"
    )


def test_revalidation_identity_skipped_when_no_expected_hash() -> None:
    # No pinned expected hash → identity check is skipped; engagement decides.
    assert (
        _geak_revalidation_decision(
            measured=115.0,
            baseline=100.0,
            got_hash="",
            expected_hash="",
            min_engaged_gain_pct=2.0,
        )
        == "validated"
    )


@pytest.mark.parametrize("measured,baseline", [(0.0, 100.0), (115.0, 0.0), (None, 100.0)])
def test_revalidation_fallback_on_bad_measurement(measured, baseline) -> None:
    assert (
        _geak_revalidation_decision(
            measured=measured,
            baseline=baseline,
            got_hash="abc",
            expected_hash="abc",
            min_engaged_gain_pct=2.0,
        )
        == "fallback"
    )


def test_normalize_geak_overlay_dir_picks_overlay_subdir(tmp_path: Path) -> None:
    final = tmp_path / "final"
    (final / "overlay").mkdir(parents=True)
    assert _normalize_geak_overlay_dir(str(final)) == str(final / "overlay")


def test_normalize_geak_overlay_dir_keeps_real_overlay(tmp_path: Path) -> None:
    overlay = tmp_path / "final" / "overlay"
    overlay.mkdir(parents=True)
    assert _normalize_geak_overlay_dir(str(overlay)) == str(overlay)


def test_normalize_geak_overlay_dir_empty_passthrough() -> None:
    assert _normalize_geak_overlay_dir("") == ""


def test_revalidation_no_promote_when_not_beating_current_best() -> None:
    # Engaged over baseline + identity matches, but does not beat current_best.
    assert (
        _geak_revalidation_decision(
            measured=9623.0,
            baseline=7380.7,
            got_hash="abc",
            expected_hash="abc",
            min_engaged_gain_pct=2.0,
            current_best=10067.9,
        )
        == "no_promote"
    )


def test_revalidation_validated_when_beating_current_best() -> None:
    # Beats current_best (and baseline + identity) -> a real KEEP.
    assert (
        _geak_revalidation_decision(
            measured=10500.0,
            baseline=7380.7,
            got_hash="abc",
            expected_hash="abc",
            min_engaged_gain_pct=2.0,
            current_best=10067.9,
        )
        == "validated"
    )


# ── Shared Coordinator fixture ───────────────────────────────────────────────


def _coord(tmp_path: Path, *, baseline: float, best_tput: float) -> Coordinator:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        baseline_tput=baseline,
        current_best={"action": "explore", "tput": best_tput},
        model_path="/models/gemma",
        gpu_type="mi300x",
        isl=1024,
        osl=1024,
        conc=64,
    )
    return coord


# ── 2a: GEAK-harness fallback validates on GEAK's OWN promoted-basis speedup ──


@pytest.mark.asyncio
async def test_geak_harness_fallback_writes_measured_headline(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Rebench-first 2a: the headline is written from the GEAK-harness MEASURED
    throughput (not a self-reported speedup), and it lifts current_best +
    optimization_stack the same way the orchestrator (2b) path does."""
    base = 2844.209
    measured = base * 1.088  # what the GEAK-harness replay actually measured
    coord = _coord(tmp_path, baseline=base, best_tput=3042.941)
    coord.shared_state.geak_result = {
        "status": "ok",
        "throughput_speedup": 1.088,
        "final_throughput_basis": "cold",
        "accepted_config": {"flags": "--max-num-batched-tokens 24576", "env": "VLLM_ROCM_USE_AITER=0"},
        "final_overlay": "",
        "validated_regimes": [{"isl": 1024, "osl": 1024, "conc": 64}],
        "alignment_metrics": {
            "hot_geak_speedup": 1.1329,  # the INFLATED number we must NOT use
            "cold_geak_speedup": 1.088,
            "final_basis": "cold",
        },
    }

    async def _fake_sweep(**_kwargs):
        # bench-e2e replay measured this throughput at the validated regime.
        return {"status": "succeeded", "best_for_each_conc": {"64": {"output_throughput": measured}}}

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._geak_sweep.sweep_via_geak",
        _fake_sweep,
    )

    out = await coord._validate_geak_via_geak_harness(reason="unit")

    assert out["validated"] is True
    ss = coord.shared_state
    expected_pct = (measured - base) / base * 100.0
    # Validated == the MEASURED same-harness total (≈+8.8%), NOT the hot A/B (+13.29%).
    assert ss.cumulative_gain_validated == pytest.approx(expected_pct, abs=1e-6)
    assert ss.cumulative_gain_validated != pytest.approx(13.29, abs=0.05)
    assert ss.resume_pending_revalidation is False
    # Rebench-first writes the headline HERE: current_best.tput == measured, and
    # the geak_e2e stack entry now exists.
    assert ss.current_best["tput"] == pytest.approx(measured)
    assert any(e.get("action") == "geak_e2e" for e in ss.optimization_stack)
    assert not ss.geak_pending  # candidate cleared on promote


@pytest.mark.asyncio
async def test_geak_harness_fallback_no_promote_below_current_best(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2a: a GEAK-harness replay measured below current_best must NOT overwrite it."""
    base = 7380.7
    current_best = 10067.9
    measured = 9623.0  # beats baseline but loses to current_best
    coord = _coord(tmp_path, baseline=base, best_tput=current_best)
    coord.shared_state.optimization_stack = [
        {"action": "explore", "variant_name": "kv-cache-fp8", "tput": current_best}
    ]
    coord.shared_state.geak_result = {
        "status": "ok",
        "throughput_speedup": 1.088,
        "final_throughput_basis": "cold",
        "accepted_config": {"flags": "--kv-cache-dtype fp8", "env": "TP=1"},
        "final_overlay": "",
        "validated_regimes": [{"isl": 1024, "osl": 1024, "conc": 64}],
        "alignment_metrics": {"cold_geak_speedup": 1.088, "final_basis": "cold"},
    }
    journey_path = tmp_path / "kernel_journey.json"
    journey_path.write_text(
        json.dumps(
            {
                "kernels": [
                    {
                        "kernel_id": "candidate-kernel",
                        "e2e": {
                            "integrated": True,
                            "e2e_gain_pct": 1.686,
                            "validated": True,
                            "decision": "KEEP",
                        },
                    },
                    {
                        "kernel_id": "already-reverted",
                        "e2e": {
                            "integrated": False,
                            "e2e_gain_pct": -1.0,
                            "validated": False,
                            "decision": "REVERT",
                        },
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    coord.shared_state.geak_result["kernel_journey_path"] = str(journey_path)
    coord._record_geak_kernel_journey(coord.shared_state.geak_result)
    provisional_rows = {row["kernel_id"]: row for row in assemble_parts(tmp_path)["kernel_journey"]["kernels"]}
    provisional = provisional_rows["candidate-kernel"]["e2e"]
    assert provisional["decision"] == "KEEP"
    assert provisional["validated"] is True

    async def _fake_sweep(**_kwargs):
        return {"status": "succeeded", "best_for_each_conc": {"64": {"output_throughput": measured}}}

    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._geak_sweep.sweep_via_geak",
        _fake_sweep,
    )

    await coord._validate_geak_via_geak_harness(reason="unit")

    ss = coord.shared_state
    assert ss.current_best["tput"] == pytest.approx(current_best)
    assert not any(e.get("action") == "geak_e2e" for e in ss.optimization_stack)
    assert ss.cumulative_gain_validated == pytest.approx(0.0)
    assert not ss.geak_pending
    rejected = assemble_parts(tmp_path)
    rejected_rows = {row["kernel_id"]: row for row in rejected["kernel_journey"]["kernels"]}
    e2e = rejected_rows["candidate-kernel"]["e2e"]
    assert e2e["decision"] == "REVERT"
    assert e2e["validated"] is False
    assert e2e["integrated"] is False
    assert e2e["e2e_gain_pct"] is None
    assert e2e["self_reported_e2e_gain_pct"] == pytest.approx(1.686)
    assert e2e["revalidation_measured_tput"] == pytest.approx(measured)
    assert e2e["revalidation_current_best_tput"] == pytest.approx(current_best)
    assert e2e["rejection_reason"] == "rebench_did_not_beat_current_best"
    untouched = rejected_rows["already-reverted"]["e2e"]
    assert untouched["decision"] == "REVERT"
    assert untouched["e2e_gain_pct"] == pytest.approx(-1.0)
    assert "rejection_reason" not in untouched

    coord.phase_kernel._reject_geak_kernel_journey(
        coord.shared_state.geak_result,
        measured_tput=measured,
        current_best_tput=current_best,
        provenance="geak_same_harness_geak",
    )
    repeated = assemble_parts(tmp_path)["kernel_journey"]["kernels"]
    assert len(repeated) == 2


# ── Fix B: report renders a PROVISIONAL gain honestly (not "+0.00% validated") ─


def _final_breakdown(*, pending: bool, gain_v: float) -> dict:
    return {
        "session": {"image": ""},
        "baseline": {"throughput_tok_s_per_gpu": 2844.2},
        "final": {
            "throughput_tok_s_per_gpu": 3236.5,
            "cumulative_gain_pct_validated": gain_v,
            "revalidation_pending": pending,
            "validated_at_stack_len": 2,
            "validated_ts": "",
            "action_path": ["geak_e2e"],
        },
    }


def test_report_shows_provisional_not_zero_validated() -> None:
    """A cross-harness provisional (validated pending) must not read as +0.00%."""
    sec = render_final(
        _final_breakdown(
            pending=True,
            gain_v=0.0,  # collectors coerces a pending/unstamped validated to 0.0
        )
    )
    facts = " ".join(sec.key_facts)
    warns = " ".join(sec.warnings)
    assert "PENDING same-harness revalidation" in facts
    assert "Validated cumulative gain" not in facts  # must NOT claim validation
    assert "+0.00%" not in facts  # must NOT read as no-op
    assert "PROVISIONAL" in warns and "cross-harness" in warns


def test_report_shows_validated_when_same_harness_confirmed() -> None:
    """A same-harness validated gain renders as authoritative, no provisional tag."""
    sec = render_final(
        _final_breakdown(
            pending=False,
            gain_v=13.5,
        )
    )
    facts = " ".join(sec.key_facts)
    assert "Validated cumulative gain" in facts
    assert "Provisional" not in facts


# ── 2b: validated is stamped ONLY from the same-harness (orchestrator) rebench ─


def _revalidate_task(*, expected_hash: str) -> Task:
    return Task(
        task_id="reval-1",
        kind="explore",
        state="succeeded",
        params={
            "source": "resume_stack_revalidate",
            "geak_fallback": True,
            "expected_cfg_hash": expected_hash,
        },
        idempotency_key="reval-1",
    )


@pytest.mark.asyncio
async def test_2b_stamps_validated_from_orchestrator_rebench(tmp_path: Path) -> None:
    """decision==validated → validated == (measured − baseline)/baseline, same harness."""
    base, measured = 2844.209, 3270.0  # ~+14.97%, engaged + identity matches
    coord = _coord(tmp_path, baseline=base, best_tput=3236.489)
    coord.shared_state.optimization_stack = [{"action": "geak_e2e", "variant_name": "geak_e2e", "tput": 3236.489}]
    coord.shared_state.resume_pending_revalidation = True

    # Guard: the GEAK-harness fallback must NOT be taken on the validated path.
    async def _must_not_fallback(**_kwargs):
        raise AssertionError("2a fallback must not run when 2b validates")

    coord._validate_geak_via_geak_harness = _must_not_fallback  # type: ignore[assignment]

    result = {
        "output_throughput": measured,
        "best_variant": {"fingerprint": "abc"},
        "winners": [],
    }
    await coord._promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="abc"))

    ss = coord.shared_state
    expected_pct = (measured - base) / base * 100.0
    assert ss.cumulative_gain_validated == pytest.approx(expected_pct, abs=1e-6)
    assert ss.resume_pending_revalidation is False
    assert ss.cumulative_gain_validated_stack_len == 1


@pytest.mark.asyncio
async def test_2b_identity_mismatch_defers_to_geak_harness(tmp_path: Path) -> None:
    """decision==fallback (config drift) → NO validated stamp; 2a is invoked."""
    base, measured = 2844.209, 3270.0  # engaged, but fingerprint won't match
    coord = _coord(tmp_path, baseline=base, best_tput=3236.489)
    coord.shared_state.optimization_stack = [{"action": "geak_e2e", "variant_name": "geak_e2e", "tput": 3236.489}]
    coord.shared_state.resume_pending_revalidation = True
    coord.shared_state.geak_pending = {
        "status": "awaiting_rebench",
        "revalidation_task_id": "reval-1",
    }

    called = {"n": 0}

    async def _fallback(**_kwargs):
        called["n"] += 1
        return {"validated": False}

    coord._validate_geak_via_geak_harness = _fallback  # type: ignore[assignment]

    result = {
        "output_throughput": measured,
        "best_variant": {"fingerprint": "DRIFTED"},  # != expected "abc"
        "winners": [],
    }
    await coord._promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="abc"))

    ss = coord.shared_state
    # 2b did NOT stamp validated (still 0); it deferred to the GEAK harness (2a).
    assert called["n"] == 1
    assert ss.cumulative_gain_validated == pytest.approx(0.0)
    assert not ss.geak_pending
    assert ss.resume_pending_revalidation is False
    assert ss.geak_result["revalidation_status"] == "fallback_failed"


@pytest.mark.asyncio
async def test_2b_no_promote_when_rebench_loses_to_current_best(tmp_path: Path) -> None:
    """A GEAK rebench that beats baseline but loses to current_best is measured, not a KEEP."""
    base, current_best, measured = 7380.7, 10067.9, 9623.0
    coord = _coord(tmp_path, baseline=base, best_tput=current_best)
    coord.shared_state.optimization_stack = [
        {"action": "explore", "variant_name": "kv-cache-fp8", "tput": current_best}
    ]
    coord.shared_state.resume_pending_revalidation = True
    coord.shared_state.geak_pending = {"status": "awaiting_rebench"}

    async def _must_not_fallback(**_kwargs):
        raise AssertionError("2a fallback must not run for a measured no-promote")

    coord._validate_geak_via_geak_harness = _must_not_fallback  # type: ignore[assignment]

    result = {
        "output_throughput": measured,
        "best_variant": {"fingerprint": "abc"},
        "winners": [],
    }
    await coord._promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="abc"))

    ss = coord.shared_state
    assert ss.current_best["tput"] == pytest.approx(current_best)
    assert ss.cumulative_gain_validated == pytest.approx(0.0)
    assert not any(e.get("action") == "geak_e2e" for e in ss.optimization_stack)
    assert ss.resume_pending_revalidation is False
    assert not ss.geak_pending
    assert ss.geak_result["revalidation_status"] == "no_promote"


# ── Rebench-first: candidate recorded, headline deferred to measured rebench ──


def _ok_result(*, final: float, base_for_gain: float | None = None) -> dict:
    return {
        "status": "ok",
        "final_throughput_tok_s": final,
        "final_throughput_basis": "cold",
        "throughput_speedup": 1.088,
        "accepted_config": {"flags": "--max-num-batched-tokens 24576", "env": "VLLM_ROCM_USE_AITER=0"},
        "final_overlay": "",
        "final_launch_script": "/x/launch.sh",
        "bench_script": "/x/bench.sh",
        "eval_dir": "/x/eval",
        "alignment_metrics": {"cold_geak_speedup": 1.088, "final_basis": "cold"},
    }


def test_record_candidate_writes_pending_not_headline(tmp_path: Path) -> None:
    """`_record_geak_candidate` stores an audit-only pending candidate and
    leaves current_best / optimization_stack / the gain ledger untouched."""
    base = 2844.209
    coord = _coord(tmp_path, baseline=base, best_tput=3042.941)
    before_best = dict(coord.shared_state.current_best)
    coord._record_geak_candidate(_ok_result(final=3236.489))

    ss = coord.shared_state
    # Headline is UNCHANGED — no premature promote.
    assert ss.current_best == before_best
    assert ss.cumulative_gain_validated == pytest.approx(0.0)
    assert not any(e.get("action") == "geak_e2e" for e in ss.optimization_stack)
    # The candidate is recorded as pending with audit-only self-reported numbers.
    pend = ss.geak_pending
    assert pend.get("status") == "awaiting_rebench"
    assert pend.get("self_reported_tput") == pytest.approx(3236.489)
    assert pend.get("self_reported_gain_pct") == pytest.approx((3236.489 - base) / base * 100.0)
    assert pend.get("accepted_flags") == "--max-num-batched-tokens 24576"
    assert pend.get("accepted_envs") == {"VLLM_ROCM_USE_AITER": "0"}


def test_promote_from_candidate_writes_measured_headline(tmp_path: Path) -> None:
    """`_promote_geak_from_candidate` lifts the headline from a MEASURED
    tput (never the self-reported number) and clears the pending candidate."""
    base = 2844.209
    measured = 3270.0
    coord = _coord(tmp_path, baseline=base, best_tput=3042.941)
    result = _ok_result(final=3236.489)  # self-reported win
    coord.shared_state.geak_result = result
    coord._record_geak_candidate(result)
    assert coord.shared_state.geak_pending.get("status") == "awaiting_rebench"
    coord._promote_geak_from_candidate(
        result,
        measured_tput=measured,
    )
    ss = coord.shared_state
    expected_pct = (measured - base) / base * 100.0
    # Headline uses the MEASURED tput, not the self-reported 3236.489.
    assert ss.current_best["tput"] == pytest.approx(measured)
    assert ss.current_best["extra_server_args"] == "--max-num-batched-tokens 24576"
    assert ss.current_best["extra_envs"].get("VLLM_ROCM_USE_AITER") == "0"
    assert ss.cumulative_gain_validated == pytest.approx(expected_pct)
    assert ss.resume_pending_revalidation is False
    geak_entry = next(e for e in ss.optimization_stack if e.get("action") == "geak_e2e")
    # A flags/env win with no proven overlay moved the CONFIG lever. Stamping
    # ``kernel`` from the task kind alone would put ``lever_buckets`` in direct
    # conflict with ``_geak_contribution``, which reads the same entry.
    assert geak_entry["lever_kind"] == "config"
    assert not ss.geak_pending


def test_promote_with_a_proven_overlay_stamps_the_kernel_lever(tmp_path: Path) -> None:
    """The lever follows the overlay proof, not the task kind."""
    base = 2844.209
    coord = _coord(tmp_path, baseline=base, best_tput=3042.941)
    result = _ok_result(final=3236.489)
    result["accepted_kernels"] = ["fused_moe"]
    coord.shared_state.geak_result = result
    coord._promote_geak_from_candidate(result, measured_tput=3270.0, overlay_loaded=True)

    entry = next(e for e in coord.shared_state.optimization_stack if e.get("action") == "geak_e2e")
    assert entry["lever_kind"] == "kernel"


def _journey_with_validated_keeps(tmp_path: Path, ratios: list[float]) -> str:
    """Write a journey whose KEEPs each carry a validated ``(base,new)`` pair.

    Those are exactly the rows ``record_kernel_e2e`` sums into the GEAK column,
    so they are the share the route-level attempt must NOT claim again.
    """
    path = tmp_path / "kernel_journey.json"
    path.write_text(
        json.dumps(
            {
                "kernels": [
                    {
                        "kernel_id": f"k{i}",
                        "e2e": {
                            "kernel_id": f"k{i}",
                            "integrated": True,
                            "validated": True,
                            "decision": "KEEP",
                            "base_tput": 1000.0,
                            "new_tput": 1000.0 * r,
                            "e2e_gain_pct": (r - 1.0) * 100.0,
                        },
                    }
                    for i, r in enumerate(ratios)
                ]
            }
        ),
        encoding="utf-8",
    )
    return str(path)


def _e2e_attempt_pair(session_dir: Path) -> tuple[float, float] | None:
    """Return the ``(before, after)`` the route-level attempt recorded."""
    parts = assemble_parts(session_dir)
    ops = {
        r.get("operation_id"): r
        for r in parts.get("operations") or []
        if isinstance(r, dict)
        and r.get("name") == "geak_e2e"
        and r.get("kind") in {"kernel_optimization", "gemm_tuning"}
    }
    if not ops:
        return None
    by_name = {
        (m.get("name"), m.get("operation_id")): m.get("value")
        for m in parts.get("measurements") or []
        if isinstance(m, dict)
    }
    op_id = next(iter(ops))
    return float(by_name[("baseline_throughput", op_id)]), float(by_name[("final_throughput", op_id)])


def test_route_attempt_starts_where_the_per_kernel_ledger_stops(tmp_path: Path) -> None:
    """The residual, not the whole route delta, is what the attempt records.

    Two KEEPs with validated pairs (+50 and +20 tok/s) are already summed into
    the GEAK column by ``record_kernel_e2e``; recording the attempt from the
    pre-GEAK tput would credit those 70 tok/s a second time.
    """
    base = 2844.209
    pre_geak = 3042.941
    measured = 3400.0
    coord = _coord(tmp_path, baseline=base, best_tput=pre_geak)
    result = _ok_result(final=3236.489)
    result["accepted_kernels"] = ["k0", "k1"]
    result["kernel_journey_path"] = _journey_with_validated_keeps(tmp_path, [1.05, 1.02])

    coord._promote_geak_from_candidate(result, measured_tput=measured, overlay_loaded=True)

    from hyperloom.orchestrator.phases.kernel import KernelPhase

    claimed = KernelPhase._geak_journey_attributed_delta(result)
    assert claimed == pytest.approx(50.0 + 20.0)
    before, after = _e2e_attempt_pair(tmp_path)
    assert before == pytest.approx(pre_geak + claimed)
    assert after == pytest.approx(measured)

    # The point of holding back an ABSOLUTE tok/s rather than a speedup ratio:
    # both records divide by the same session baseline, so the per-kernel
    # credits and the route credit telescope to exactly the measured route
    # lift, leaving nothing for ``unattributed_gain_pct`` to absorb.
    ledger_pct = claimed / base * 100.0
    route_pct = (after - before) / base * 100.0
    assert ledger_pct + route_pct == pytest.approx((measured - pre_geak) / base * 100.0)


def test_route_attempt_survives_when_only_the_config_remainder_is_left(tmp_path: Path) -> None:
    """A journey KEEP must not suppress the whole attempt.

    The predicate this replaced was boolean: any attributable KEEP dropped the
    route row entirely, so an env/flag win measured in the same promotion never
    reached the ledger at all.
    """
    coord = _coord(tmp_path, baseline=2844.209, best_tput=3000.0)
    result = _ok_result(final=3236.489)
    result["kernel_journey_path"] = _journey_with_validated_keeps(tmp_path, [1.05])

    coord._promote_geak_from_candidate(result, measured_tput=3400.0, overlay_loaded=True)

    before, after = _e2e_attempt_pair(tmp_path)
    assert before == pytest.approx(3000.0 + 50.0)
    assert after == pytest.approx(3400.0)


def test_noise_sized_residual_is_not_recorded_as_an_attempt(tmp_path: Path) -> None:
    """A keep worth 0.05% is measurement noise, not an optimization."""
    coord = _coord(tmp_path, baseline=2844.209, best_tput=3000.0)
    result = _ok_result(final=3236.489)
    result["kernel_journey_path"] = _journey_with_validated_keeps(tmp_path, [1.05])

    # 3050.0 is the claimed share; the promotion measured barely above it.
    coord._promote_geak_from_candidate(result, measured_tput=3050.0 * 1.0005, overlay_loaded=True)

    assert _e2e_attempt_pair(tmp_path) is None


def test_unproven_overlay_leaves_the_full_delta_to_the_route(tmp_path: Path) -> None:
    """Without overlay proof the per-kernel ledger claims nothing.

    ``_reject_geak_kernel_journey`` refuses to credit KEEPs whose overlay was
    not proven loaded, so holding back their delta here would erase gain no
    other record holds.
    """
    coord = _coord(tmp_path, baseline=2844.209, best_tput=3000.0)
    result = _ok_result(final=3236.489)
    result["kernel_journey_path"] = _journey_with_validated_keeps(tmp_path, [1.05])

    coord._promote_geak_from_candidate(result, measured_tput=3400.0, overlay_loaded=False)

    before, _ = _e2e_attempt_pair(tmp_path)
    assert before == pytest.approx(3000.0)


def test_report_shows_pending_candidate_excluded_from_headline() -> None:
    """A pending GEAK candidate renders as an audit note + warning and is
    NOT presented as a validated headline gain."""
    bd = {
        "session": {"image": ""},
        "baseline": {"throughput_tok_s_per_gpu": 2844.2},
        "final": {
            "throughput_tok_s_per_gpu": 2844.2,
            "cumulative_gain_pct_validated": 0.0,
            "revalidation_pending": False,
            "action_path": [],
            "geak_pending": {"status": "awaiting_rebench", "self_reported_gain_pct": 13.79},
        },
    }
    sec = render_final(bd)
    facts = " ".join(sec.key_facts)
    warns = " ".join(sec.warnings)
    assert "AWAITING" in facts and "13.79" in facts or "13.8" in facts
    assert "Validated cumulative gain" not in facts
    assert "audit-only" in warns and "not been" in warns.lower() or "NOT" in warns


# ── 2b material guard: same-config rebench noise must not stamp kernel gain ───


@pytest.mark.asyncio
async def test_2b_no_material_candidate_does_not_promote(tmp_path: Path) -> None:
    """GEAK returned no kernel/head/overlay/patch AND its accepted_config equals
    the pre-KERNEL current_best (pure passthrough). A rebench that beats
    current_best by measurement noise must NOT be recorded as a kernel gain."""
    base, current_best, measured = 8668.5946, 8900.0, 9025.191
    coord = _coord(tmp_path, baseline=base, best_tput=current_best)
    coord.shared_state.current_best["extra_server_args"] = "--max-num-batched-tokens 24576"
    coord.shared_state.current_best["extra_envs"] = {"VLLM_ROCM_USE_AITER": "1"}
    coord.shared_state.optimization_stack = [
        {"action": "explore", "variant_name": "kv-cache-fp8", "tput": current_best}
    ]
    coord.shared_state.resume_pending_revalidation = True
    coord.shared_state.geak_pending = {"status": "awaiting_rebench"}
    # geak_result is non-empty but ships NO material product; accepted_config is
    # the pre-KERNEL current_best config verbatim (passthrough, zero delta).
    coord.shared_state.geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--max-num-batched-tokens 24576", "env": "VLLM_ROCM_USE_AITER=1"},
        "accepted_kernels": [],
        "accepted_heads": [],
        "final_overlay": "",
        "final_patch": "",
    }

    async def _must_not_fallback(**_kwargs):
        raise AssertionError("2a fallback must not run for a no-material drop")

    coord._validate_geak_via_geak_harness = _must_not_fallback  # type: ignore[assignment]

    result = {
        "output_throughput": measured,
        "best_variant": {"fingerprint": "abc"},
        "winners": [],
    }
    await coord._promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="abc"))

    ss = coord.shared_state
    assert ss.current_best["tput"] == pytest.approx(current_best)
    assert ss.cumulative_gain_validated == pytest.approx(0.0)
    assert not any(e.get("action") == "geak_e2e" for e in ss.optimization_stack)
    assert ss.resume_pending_revalidation is False
    assert not ss.geak_pending


@pytest.mark.asyncio
async def test_2b_config_delta_candidate_still_promotes(tmp_path: Path) -> None:
    """GEAK shipped no overlay/patch/kernel list, but its accepted_config adds a
    new flag vs the pre-KERNEL current_best (a kernel enabled via a config
    switch). That is a real GEAK product and must still promote."""
    base, current_best, measured = 8668.5946, 8900.0, 9600.0
    coord = _coord(tmp_path, baseline=base, best_tput=current_best)
    coord.shared_state.current_best["extra_server_args"] = "--max-num-batched-tokens 24576"
    coord.shared_state.current_best["extra_envs"] = {"VLLM_ROCM_USE_AITER": "1"}
    coord.shared_state.optimization_stack = [
        {"action": "explore", "variant_name": "kv-cache-fp8", "tput": current_best}
    ]
    coord.shared_state.resume_pending_revalidation = True
    # accepted_config adds VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1 (a new kernel switch).
    result_blob = {
        "status": "ok",
        "accepted_config": {
            "flags": "--max-num-batched-tokens 24576",
            "env": "VLLM_ROCM_USE_AITER=1 VLLM_ROCM_USE_AITER_FP4_ASM_GEMM=1",
        },
        "accepted_kernels": [],
        "accepted_heads": [],
        "final_overlay": "",
        "final_patch": "",
    }
    coord.shared_state.geak_result = result_blob
    coord._record_geak_candidate(result_blob)

    async def _must_not_fallback(**_kwargs):
        raise AssertionError("2a fallback must not run when 2b validates a real delta")

    coord._validate_geak_via_geak_harness = _must_not_fallback  # type: ignore[assignment]

    result = {
        "output_throughput": measured,
        "best_variant": {"fingerprint": "abc"},
        "winners": [],
    }
    await coord._promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="abc"))

    ss = coord.shared_state
    expected_pct = (measured - base) / base * 100.0
    assert ss.current_best["tput"] == pytest.approx(measured)
    assert ss.cumulative_gain_validated == pytest.approx(expected_pct)
    assert any(e.get("action") == "geak_e2e" for e in ss.optimization_stack)
    assert not ss.geak_pending


@pytest.mark.asyncio
async def test_2b_empty_result_without_prior_geak_e2e_does_not_promote(tmp_path: Path) -> None:
    """A validated 2b decision with an EMPTY geak_result and NO pre-existing
    geak_e2e stack entry has no material to validate: it is same-config noise
    (geak_result lost / never populated), so it must NOT promote."""
    base, current_best, measured = 8668.5946, 8900.0, 9025.191
    coord = _coord(tmp_path, baseline=base, best_tput=current_best)
    coord.shared_state.optimization_stack = [
        {"action": "explore", "variant_name": "kv-cache-fp8", "tput": current_best}
    ]
    coord.shared_state.resume_pending_revalidation = True
    coord.shared_state.geak_pending = {"status": "awaiting_rebench"}
    coord.shared_state.geak_result = {}  # empty: cannot be judged by the helper

    async def _must_not_fallback(**_kwargs):
        raise AssertionError("2a fallback must not run for a no-material drop")

    coord._validate_geak_via_geak_harness = _must_not_fallback  # type: ignore[assignment]

    result = {
        "output_throughput": measured,
        "best_variant": {"fingerprint": "abc"},
        "winners": [],
    }
    await coord._promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="abc"))

    ss = coord.shared_state
    assert ss.current_best["tput"] == pytest.approx(current_best)
    assert ss.cumulative_gain_validated == pytest.approx(0.0)
    assert not any(e.get("action") == "geak_e2e" for e in ss.optimization_stack)
    assert ss.resume_pending_revalidation is False
    assert not ss.geak_pending


# ── material-guard helper unit boundaries ────────────────────────────────────


@pytest.mark.parametrize(
    ("result", "prev_flags", "prev_envs", "expected"),
    [
        # Empty / non-dict -> cannot judge -> True (caller disambiguates).
        ({}, "", {}, True),
        (None, "", {}, True),
        # No product, config identical to prev best -> non-material.
        (
            {"accepted_config": {"flags": "--a 1", "env": "X=1"}, "accepted_kernels": []},
            "--a 1",
            {"X": "1"},
            False,
        ),
        # Env order differs but semantics identical -> non-material.
        (
            {"accepted_config": {"flags": "", "env": "A=1 B=2"}},
            "",
            {"B": "2", "A": "1"},
            False,
        ),
        # accepted_kernels is a list of blank entries -> non-material.
        (
            {"accepted_config": {"flags": "--a 1", "env": ""}, "accepted_kernels": ["", "  "]},
            "--a 1",
            {},
            False,
        ),
        # accepted_kernels has a real entry -> material.
        (
            {"accepted_config": {"flags": "--a 1", "env": ""}, "accepted_kernels": ["fused_rope"]},
            "--a 1",
            {},
            True,
        ),
        # final_overlay is whitespace only -> non-material (config identical).
        (
            {"accepted_config": {"flags": "--a 1", "env": ""}, "final_overlay": "   "},
            "--a 1",
            {},
            False,
        ),
        # accepted_config adds a new env vs prev best -> material.
        (
            {"accepted_config": {"flags": "--a 1", "env": "X=1 NEW=1"}},
            "--a 1",
            {"X": "1"},
            True,
        ),
        # accepted_config MISSING while prev best is non-empty -> non-material
        # (a bare mismatch must not promote and wipe the existing config).
        (
            {"status": "ok", "accepted_kernels": []},
            "--max-num-batched-tokens 24576",
            {"VLLM_ROCM_USE_AITER": "1"},
            False,
        ),
        # accepted_config present but all-empty while prev best is non-empty ->
        # non-material (same wipe hazard).
        (
            {"status": "ok", "accepted_config": {"flags": "", "env": ""}},
            "--max-num-batched-tokens 24576",
            {"VLLM_ROCM_USE_AITER": "1"},
            False,
        ),
    ],
)
def test_geak_result_has_material_boundaries(result, prev_flags, prev_envs, expected) -> None:
    assert _geak_result_has_material(result, prev_best_flags=prev_flags, prev_best_envs=prev_envs) is expected


@pytest.mark.asyncio
async def test_2b_no_material_reverts_provisional_journey_keep(tmp_path: Path) -> None:
    """A passthrough 2b drop must REVERT a provisional kernel_journey KEEP and
    tag it with the no-material reason (not the beat-current_best reason)."""
    base, current_best, measured = 8668.5946, 8900.0, 9025.191
    coord = _coord(tmp_path, baseline=base, best_tput=current_best)
    coord.shared_state.current_best["extra_server_args"] = "--max-num-batched-tokens 24576"
    coord.shared_state.current_best["extra_envs"] = {"VLLM_ROCM_USE_AITER": "1"}
    coord.shared_state.optimization_stack = [
        {"action": "explore", "variant_name": "kv-cache-fp8", "tput": current_best}
    ]
    coord.shared_state.resume_pending_revalidation = True
    coord.shared_state.geak_pending = {"status": "awaiting_rebench"}
    journey_path = tmp_path / "kernel_journey.json"
    journey_path.write_text(
        json.dumps(
            {
                "kernels": [
                    {
                        "kernel_id": "provisional-kernel",
                        "e2e": {
                            "integrated": True,
                            "e2e_gain_pct": 2.0,
                            "validated": True,
                            "decision": "KEEP",
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    # No material product; accepted_config is the pre-KERNEL best verbatim.
    geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--max-num-batched-tokens 24576", "env": "VLLM_ROCM_USE_AITER=1"},
        "accepted_kernels": [],
        "accepted_heads": [],
        "final_overlay": "",
        "final_patch": "",
        "kernel_journey_path": str(journey_path),
    }
    coord.shared_state.geak_result = geak_result
    coord._record_geak_kernel_journey(geak_result)
    provisional_rows = {row["kernel_id"]: row for row in assemble_parts(tmp_path)["kernel_journey"]["kernels"]}
    assert provisional_rows["provisional-kernel"]["e2e"]["decision"] == "KEEP"

    async def _must_not_fallback(**_kwargs):
        raise AssertionError("2a fallback must not run for a no-material drop")

    coord._validate_geak_via_geak_harness = _must_not_fallback  # type: ignore[assignment]

    result = {
        "output_throughput": measured,
        "best_variant": {"fingerprint": "abc"},
        "winners": [],
    }
    await coord._promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="abc"))

    ss = coord.shared_state
    assert ss.current_best["tput"] == pytest.approx(current_best)
    assert not any(e.get("action") == "geak_e2e" for e in ss.optimization_stack)
    assert ss.geak_result["revalidation_status"] == "no_material"
    rejected_rows = {row["kernel_id"]: row for row in assemble_parts(tmp_path)["kernel_journey"]["kernels"]}
    e2e = rejected_rows["provisional-kernel"]["e2e"]
    assert e2e["decision"] == "REVERT"
    assert e2e["validated"] is False
    assert e2e["rejection_reason"] == "geak_no_material_product"


@pytest.mark.asyncio
async def test_2b_empty_result_with_prior_geak_e2e_still_promotes(tmp_path: Path) -> None:
    """Resume revalidation: geak_result was lost (empty) but a geak_e2e stack
    entry already recorded the win. The 2b validated decision must still promote
    (the material was proven in the original KERNEL cycle)."""
    base, current_best, measured = 8668.5946, 8900.0, 9600.0
    coord = _coord(tmp_path, baseline=base, best_tput=current_best)
    coord.shared_state.optimization_stack = [{"action": "geak_e2e", "variant_name": "geak_e2e", "tput": current_best}]
    coord.shared_state.resume_pending_revalidation = True
    coord.shared_state.geak_result = {}  # lost on resume

    async def _must_not_fallback(**_kwargs):
        raise AssertionError("2a fallback must not run when 2b validates a resume win")

    coord._validate_geak_via_geak_harness = _must_not_fallback  # type: ignore[assignment]

    result = {
        "output_throughput": measured,
        "best_variant": {"fingerprint": "abc"},
        "winners": [],
    }
    await coord._promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="abc"))

    ss = coord.shared_state
    expected_pct = (measured - base) / base * 100.0
    assert ss.cumulative_gain_validated == pytest.approx(expected_pct)
    assert ss.resume_pending_revalidation is False


@pytest.mark.asyncio
async def test_2b_resume_reverify_of_promoted_geak_win_still_promotes(tmp_path: Path) -> None:
    """Regression: a resume revalidation of an ALREADY-promoted GEAK win must
    not be judged no_material. On resume geak_result is persisted (non-empty)
    and current_best already holds the GEAK accepted_config, so the fingerprint
    matches by construction; the pre-existing geak_e2e stack entry is the escape
    hatch and must short-circuit the material check."""
    base, measured = 8668.5946, 9800.0
    current_best = 9600.0  # current_best already holds the promoted GEAK win
    coord = _coord(tmp_path, baseline=base, best_tput=current_best)
    # current_best carries the GEAK accepted_config (a later kernel integrate
    # did not change server args), so a real revalidation fingerprint matches.
    coord.shared_state.current_best["extra_server_args"] = "--max-num-batched-tokens 24576"
    coord.shared_state.current_best["extra_envs"] = {"VLLM_ROCM_USE_AITER": "1"}
    coord.shared_state.optimization_stack = [
        {"action": "explore", "variant_name": "kv-cache-fp8", "tput": 8900.0},
        {"action": "geak_e2e", "variant_name": "geak_e2e", "tput": current_best},
        {"action": "integrate_patch", "variant_name": "kernel-x", "tput": current_best},
    ]
    coord.shared_state.resume_pending_revalidation = True
    # geak_result survives the resume (persisted field) and echoes the config.
    geak_result = {
        "status": "ok",
        "accepted_config": {"flags": "--max-num-batched-tokens 24576", "env": "VLLM_ROCM_USE_AITER=1"},
        "accepted_kernels": [],
        "accepted_heads": [],
        "final_overlay": "",
        "final_patch": "",
    }
    coord.shared_state.geak_result = geak_result

    async def _must_not_fallback(**_kwargs):
        raise AssertionError("2a fallback must not run when re-verifying a promoted win")

    coord._validate_geak_via_geak_harness = _must_not_fallback  # type: ignore[assignment]

    result = {
        "output_throughput": measured,
        "best_variant": {"fingerprint": "abc"},
        "winners": [],
    }
    await coord._promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="abc"))

    ss = coord.shared_state
    expected_pct = (measured - base) / base * 100.0
    assert ss.cumulative_gain_validated == pytest.approx(expected_pct)
    assert ss.resume_pending_revalidation is False
    assert ss.geak_result.get("revalidation_status") != "no_material"


# ── B5: a stack entry may only name kernels the overlay was proven to carry ──


def test_promote_with_dead_overlay_leaves_no_kernel_names_in_stack_entry(tmp_path: Path) -> None:
    """A promote whose overlay was proven NOT loaded is a config gain.

    GEAK self-reports ``accepted_kernels`` / ``accepted_heads`` whether or not the
    overlay carrying them survived to the measurement. The per-kernel ledger already
    refuses to credit them without proof; the stack entry is the other reader, and
    ``_geak_contribution`` classifies the dashboard row from those lanes alone. Both
    must make the same call, or a rebench that stripped a dead overlay gets filed
    under ``kernel``.
    """
    base = 2844.209
    measured = 3270.0
    coord = _coord(tmp_path, baseline=base, best_tput=3042.941)
    result = _ok_result(final=3236.489)
    result["accepted_kernels"] = ["c0_triton"]
    result["accepted_heads"] = ["fused_moe_kernel"]

    coord._promote_geak_from_candidate(result, measured_tput=measured, overlay_loaded=False)

    entry = next(e for e in coord.shared_state.optimization_stack if e.get("action") == "geak_e2e")
    # ``_lift_to_current_best`` drops empty values, so "no proof" reads as no lane
    # at all rather than an empty one -- either way there is no name to credit.
    assert not entry.get("accepted_kernels")
    assert not entry.get("accepted_heads")
    assert entry["overlay_loaded"] is False
    # The config lane is untouched, so the row is still a real gain — just not a kernel one.
    assert _geak_contribution(entry) == "config"


def test_promote_with_loaded_overlay_keeps_kernel_names_in_stack_entry(tmp_path: Path) -> None:
    """The mirror case: proof present, so the lanes travel and the row is joint."""
    coord = _coord(tmp_path, baseline=2844.209, best_tput=3042.941)
    result = _ok_result(final=3236.489)
    result["accepted_kernels"] = ["c0_triton"]

    coord._promote_geak_from_candidate(result, measured_tput=3270.0, overlay_loaded=True)

    entry = next(e for e in coord.shared_state.optimization_stack if e.get("action") == "geak_e2e")
    assert entry["accepted_kernels"] == ["c0_triton"]
    assert entry["overlay_loaded"] is True
    # Config gain rode along in the same measurement, so it cannot be decomposed.
    assert _geak_contribution(entry) == "joint"
