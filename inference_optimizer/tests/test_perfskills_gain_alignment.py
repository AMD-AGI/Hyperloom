# Copyright Advanced Micro Devices, Inc. All rights reserved.
"""Alignment / credibility unit tests for the PerfSkills(GEAK) e2e gain path.

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

Run: python3 -m pytest inference_optimizer/tests/test_perfskills_gain_alignment.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from inference_optimizer.breakdown.reporters._renderers.final import render as render_final
from inference_optimizer.orchestrator.coordinator import (
    Coordinator,
    _perfskills_revalidation_decision,
)
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.orchestrator.task_registry import Task


# ── #3: same-harness (2b) revalidation decision ──────────────────────────────


def test_revalidation_validated_when_identity_and_engagement_hold() -> None:
    assert (
        _perfskills_revalidation_decision(
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
        _perfskills_revalidation_decision(
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
        _perfskills_revalidation_decision(
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
        _perfskills_revalidation_decision(
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
        _perfskills_revalidation_decision(
            measured=measured,
            baseline=baseline,
            got_hash="abc",
            expected_hash="abc",
            min_engaged_gain_pct=2.0,
        )
        == "fallback"
    )


# ── #5: provisional promote gain is consistent + not validated ───────────────


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


def test_promote_provisional_gain_matches_current_best_over_baseline(tmp_path: Path) -> None:
    """Provisional gain == (promoted final / baseline) − 1, i.e. cold-to-cold.

    It must NOT use the HOT final (which would overstate and contradict the
    persisted current_best.tput a reader divides by baseline).
    """
    base = 2844.209
    cold_final = 3236.489       # promoted (cold) — becomes current_best.tput
    hot_final = 3299.149        # steady-state; must NOT drive the provisional gain
    coord = _coord(tmp_path, baseline=base, best_tput=3042.941)
    result = {
        "status": "ok",
        "final_throughput_tok_s": cold_final,
        "final_throughput_basis": "cold",
        "throughput_speedup": 1.088,
        "accepted_config": {"flags": "--max-num-batched-tokens 24576", "env": "VLLM_ROCM_USE_AITER=0"},
        "alignment_metrics": {
            "geak_hot_final_tok_s": hot_final,
            "hot_geak_speedup": 1.1329,
            "cold_geak_speedup": 1.088,
            "hot_speedup": 1.0978,
            "cold_speedup": 1.1379,
            "final_basis": "cold",
        },
        "baseline_basis": {"measurement_divergence_pct": 0.5},
    }

    coord._promote_perfskills_result(result)
    ss = coord.shared_state

    # current_best.tput is the promoted (cold) final.
    assert ss.current_best["tput"] == pytest.approx(cold_final)
    # Provisional gain is EXACTLY consistent with the two persisted anchors.
    expected_pct = (cold_final - base) / base * 100.0
    assert ss.cumulative_gain == pytest.approx(expected_pct, abs=1e-6)
    # It equals cold_speedup, and is BELOW the discarded hot-over-cold ratio.
    assert ss.cumulative_gain == pytest.approx((1.1379 - 1.0) * 100.0, abs=0.05)
    assert ss.cumulative_gain < (hot_final - base) / base * 100.0
    # Provenance marks it provisional; validated is NOT stamped here.
    assert ss.cumulative_gain_provenance == "perfskills_cross_harness_provisional"
    assert ss.resume_pending_revalidation is True
    assert ss.cumulative_gain_validated == pytest.approx(0.0)
    # GEAK's own within-harness speedups are stashed for audit cross-check.
    audit = ss.current_best.get("perfskills_alignment") or {}
    assert audit.get("cold_geak_speedup") == pytest.approx(1.088)
    assert audit.get("geak_throughput_speedup") == pytest.approx(1.088)


def test_promote_falls_back_to_final_when_alignment_absent(tmp_path: Path) -> None:
    """Standalone runs (no alignment_metrics) still get a consistent gain."""
    base, final = 100.0, 116.0
    coord = _coord(tmp_path, baseline=base, best_tput=100.0)
    coord._promote_perfskills_result(
        {
            "status": "ok",
            "final_throughput_tok_s": final,
            "accepted_config": {"flags": "", "env": ""},
        }
    )
    assert coord.shared_state.cumulative_gain == pytest.approx(16.0)
    assert coord.shared_state.cumulative_gain_validated == pytest.approx(0.0)


# ── 2a: GEAK-harness fallback validates on GEAK's OWN promoted-basis speedup ──


@pytest.mark.asyncio
async def test_geak_harness_fallback_uses_promoted_basis_speedup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2a validated gain == GEAK's headline (throughput_speedup), not hot A/B."""
    coord = _coord(tmp_path, baseline=2844.209, best_tput=3236.489)
    coord.shared_state.perfskills_result = {
        "status": "ok",
        "throughput_speedup": 1.088,        # GEAK headline (cold basis)
        "final_throughput_basis": "cold",
        "validated_regimes": [{"isl": 1024, "osl": 1024, "conc": 64}],
        "alignment_metrics": {
            "hot_geak_speedup": 1.1329,     # the INFLATED number we must NOT use
            "cold_geak_speedup": 1.088,
            "final_basis": "cold",
        },
    }

    async def _fake_sweep(**_kwargs):
        return {"status": "succeeded"}

    monkeypatch.setattr(
        "inference_optimizer.orchestrator.action_executors._perfskills_sweep.sweep_via_perfskills",
        _fake_sweep,
    )

    out = await coord._validate_perfskills_via_geak_harness(reason="unit")

    assert out["validated"] is True
    ss = coord.shared_state
    # Validated == GEAK's own headline (+8.8%), NOT the hot A/B (+13.29%).
    assert ss.cumulative_gain_validated == pytest.approx(8.8, abs=0.05)
    assert ss.cumulative_gain_validated != pytest.approx(13.29, abs=0.05)
    assert ss.cumulative_gain_provenance == "perfskills_same_harness_geak"
    assert ss.resume_pending_revalidation is False


# ── Fix B: report renders a PROVISIONAL gain honestly (not "+0.00% validated") ─


def _final_breakdown(*, provenance: str, pending: bool, gain_v: float, gain_round: float) -> dict:
    return {
        "session": {"image": ""},
        "baseline": {"throughput_tok_s_per_gpu": 2844.2},
        "final": {
            "throughput_tok_s_per_gpu": 3236.5,
            "cumulative_gain_pct_validated": gain_v,
            "cumulative_gain_pct_per_round_sum": gain_round,
            "cumulative_gain_provenance": provenance,
            "revalidation_pending": pending,
            "validated_at_stack_len": 2,
            "validated_ts": "",
            "action_path": ["perfskills_e2e"],
        },
    }


def test_report_shows_provisional_not_zero_validated() -> None:
    """A cross-harness provisional (validated pending) must not read as +0.00%."""
    sec = render_final(
        _final_breakdown(
            provenance="perfskills_cross_harness_provisional",
            pending=True,
            gain_v=0.0,       # collectors coerces a pending/unstamped validated to 0.0
            gain_round=13.79,
        )
    )
    facts = " ".join(sec.key_facts)
    warns = " ".join(sec.warnings)
    assert "Provisional" in facts
    assert "13.79" in facts or "13.8" in facts
    assert "Validated cumulative gain" not in facts   # must NOT claim validation
    assert "+0.00%" not in facts                       # must NOT read as no-op
    assert "PROVISIONAL" in warns and "cross-harness" in warns


def test_report_shows_validated_when_same_harness_confirmed() -> None:
    """A same-harness validated gain renders as authoritative, no provisional tag."""
    sec = render_final(
        _final_breakdown(
            provenance="perfskills_orch_harness_validated",
            pending=False,
            gain_v=13.5,
            gain_round=13.5,
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
            "perfskills_fallback": True,
            "expected_cfg_hash": expected_hash,
        },
        idempotency_key="reval-1",
    )


@pytest.mark.asyncio
async def test_2b_stamps_validated_from_orchestrator_rebench(tmp_path: Path) -> None:
    """decision==validated → validated == (measured − baseline)/baseline, same harness."""
    base, measured = 2844.209, 3270.0     # ~+14.97%, engaged + identity matches
    coord = _coord(tmp_path, baseline=base, best_tput=3236.489)
    coord.shared_state.optimization_stack = [{"action": "perfskills_e2e", "tput": 3236.489}]
    coord.shared_state.resume_pending_revalidation = True

    # Guard: the GEAK-harness fallback must NOT be taken on the validated path.
    async def _must_not_fallback(**_kwargs):
        raise AssertionError("2a fallback must not run when 2b validates")

    coord._validate_perfskills_via_geak_harness = _must_not_fallback  # type: ignore[assignment]

    result = {
        "output_throughput": measured,
        "best_variant": {"fingerprint": "abc"},
        "winners": [],
    }
    await coord._promote_to_shared_state(
        "explore", result, task=_revalidate_task(expected_hash="abc")
    )

    ss = coord.shared_state
    expected_pct = (measured - base) / base * 100.0
    assert ss.cumulative_gain_validated == pytest.approx(expected_pct, abs=1e-6)
    assert ss.cumulative_gain_provenance == "perfskills_orch_harness_validated"
    assert ss.resume_pending_revalidation is False
    assert ss.cumulative_gain_validated_stack_len == 1


@pytest.mark.asyncio
async def test_2b_identity_mismatch_defers_to_geak_harness(tmp_path: Path) -> None:
    """decision==fallback (config drift) → NO validated stamp; 2a is invoked."""
    base, measured = 2844.209, 3270.0     # engaged, but fingerprint won't match
    coord = _coord(tmp_path, baseline=base, best_tput=3236.489)
    coord.shared_state.optimization_stack = [{"action": "perfskills_e2e", "tput": 3236.489}]
    coord.shared_state.resume_pending_revalidation = True

    called = {"n": 0}

    async def _fallback(**_kwargs):
        called["n"] += 1
        return {"validated": False}

    coord._validate_perfskills_via_geak_harness = _fallback  # type: ignore[assignment]

    result = {
        "output_throughput": measured,
        "best_variant": {"fingerprint": "DRIFTED"},   # != expected "abc"
        "winners": [],
    }
    await coord._promote_to_shared_state(
        "explore", result, task=_revalidate_task(expected_hash="abc")
    )

    ss = coord.shared_state
    # 2b did NOT stamp validated (still 0); it deferred to the GEAK harness (2a).
    assert called["n"] == 1
    assert ss.cumulative_gain_validated == pytest.approx(0.0)
    assert ss.cumulative_gain_provenance != "perfskills_orch_harness_validated"
