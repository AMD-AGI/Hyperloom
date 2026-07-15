# Copyright Advanced Micro Devices, Inc. All rights reserved.
"""Alignment / credibility unit tests for the GEAK(GEAK) e2e gain path.

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

Run: python3 -m pytest inference_optimizer/tests/test_geak_gain_alignment.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown.reporters._renderers.final import render as render_final
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.coordinator_helpers import (
    _geak_revalidation_decision,
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


# ── Shared Coordinator fixture ───────────────────────────────────────────────


def test_geak_legacy_promote_honors_env_gate(monkeypatch) -> None:
    from hyperloom.orchestrator.phases.kernel import KernelPhase

    monkeypatch.delenv("INFERENCE_OPTIMIZER_GEAK_LEGACY_PROMOTE", raising=False)
    coord = Coordinator.__new__(Coordinator)
    coord.shared_state = SharedState()
    phase = KernelPhase(coord)
    assert phase._geak_legacy_promote() is False
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GEAK_LEGACY_PROMOTE", "yes")
    assert phase._geak_legacy_promote() is True


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


def test_geak_legacy_promote_default_off(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INFERENCE_OPTIMIZER_GEAK_LEGACY_PROMOTE", raising=False)
    coord = _coord(tmp_path, baseline=100.0, best_tput=108.0)

    assert coord._geak_legacy_promote() is False


def test_geak_legacy_promote_env_flag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INFERENCE_OPTIMIZER_GEAK_LEGACY_PROMOTE", "yes")
    coord = _coord(tmp_path, baseline=100.0, best_tput=108.0)

    assert coord._geak_legacy_promote() is True


# ── 2a: GEAK-harness fallback validates on GEAK's OWN promoted-basis speedup ──


@pytest.mark.asyncio
async def test_geak_harness_fallback_writes_measured_headline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
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
            "hot_geak_speedup": 1.1329,     # the INFLATED number we must NOT use
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
    assert ss.cumulative_gain_provenance == "geak_same_harness_geak"
    assert ss.resume_pending_revalidation is False
    # Rebench-first writes the headline HERE: current_best.tput == measured, and
    # the geak_e2e stack entry now exists.
    assert ss.current_best["tput"] == pytest.approx(measured)
    assert any(e.get("action") == "geak_e2e" for e in ss.optimization_stack)
    assert not ss.geak_pending  # candidate cleared on promote


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
            "action_path": ["geak_e2e"],
        },
    }


def test_report_shows_provisional_not_zero_validated() -> None:
    """A cross-harness provisional (validated pending) must not read as +0.00%."""
    sec = render_final(
        _final_breakdown(
            provenance="geak_cross_harness_provisional",
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
            provenance="geak_orch_harness_validated",
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
            "geak_fallback": True,
            "expected_cfg_hash": expected_hash,
        },
        idempotency_key="reval-1",
    )


@pytest.mark.asyncio
async def test_2b_stamps_validated_from_orchestrator_rebench(tmp_path: Path) -> None:
    """decision==validated → validated == (measured − baseline)/baseline, same harness."""
    base, measured = 2844.209, 3270.0     # ~+14.97%, engaged + identity matches
    coord = _coord(tmp_path, baseline=base, best_tput=3236.489)
    coord.shared_state.optimization_stack = [{"action": "geak_e2e", "tput": 3236.489}]
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
    await coord._promote_to_shared_state(
        "explore", result, task=_revalidate_task(expected_hash="abc")
    )

    ss = coord.shared_state
    expected_pct = (measured - base) / base * 100.0
    assert ss.cumulative_gain_validated == pytest.approx(expected_pct, abs=1e-6)
    assert ss.cumulative_gain_provenance == "geak_orch_harness_validated"
    assert ss.resume_pending_revalidation is False
    assert ss.cumulative_gain_validated_stack_len == 1


@pytest.mark.asyncio
async def test_2b_identity_mismatch_defers_to_geak_harness(tmp_path: Path) -> None:
    """decision==fallback (config drift) → NO validated stamp; 2a is invoked."""
    base, measured = 2844.209, 3270.0     # engaged, but fingerprint won't match
    coord = _coord(tmp_path, baseline=base, best_tput=3236.489)
    coord.shared_state.optimization_stack = [{"action": "geak_e2e", "tput": 3236.489}]
    coord.shared_state.resume_pending_revalidation = True

    called = {"n": 0}

    async def _fallback(**_kwargs):
        called["n"] += 1
        return {"validated": False}

    coord._validate_geak_via_geak_harness = _fallback  # type: ignore[assignment]

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
    assert ss.cumulative_gain_provenance != "geak_orch_harness_validated"


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
    assert ss.cumulative_gain == pytest.approx(0.0)
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
        provenance="geak_orch_harness_validated",
    )
    ss = coord.shared_state
    expected_pct = (measured - base) / base * 100.0
    # Headline uses the MEASURED tput, not the self-reported 3236.489.
    assert ss.current_best["tput"] == pytest.approx(measured)
    assert ss.current_best["extra_server_args"] == "--max-num-batched-tokens 24576"
    assert ss.current_best["extra_envs"].get("VLLM_ROCM_USE_AITER") == "0"
    assert ss.cumulative_gain_validated == pytest.approx(expected_pct)
    assert ss.cumulative_gain == pytest.approx(expected_pct)
    assert ss.cumulative_gain_provenance == "geak_orch_harness_validated"
    assert ss.resume_pending_revalidation is False
    assert any(e.get("action") == "geak_e2e" for e in ss.optimization_stack)
    assert not ss.geak_pending


def test_report_shows_pending_candidate_excluded_from_headline() -> None:
    """A pending GEAK candidate renders as an audit note + warning and is
    NOT presented as a validated headline gain."""
    bd = {
        "session": {"image": ""},
        "baseline": {"throughput_tok_s_per_gpu": 2844.2},
        "final": {
            "throughput_tok_s_per_gpu": 2844.2,
            "cumulative_gain_pct_validated": 0.0,
            "cumulative_gain_pct_per_round_sum": 0.0,
            "cumulative_gain_provenance": "",
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
