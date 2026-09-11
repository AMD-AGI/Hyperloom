# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Alignment / credibility unit tests for the GEAK e2e gain path."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.assembler import stack_event_parts
from hyperloom.inference_optimizer.breakdown.reporters._renderers.final import render as render_final
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.coordinator_helpers import (
    _geak_result_has_material,
    _geak_revalidation_decision,
    _normalize_geak_overlay_dir,
)
from hyperloom.inference_optimizer.session.session_binding import session_scope
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
    """Rebench-first 2a: the headline is written from the GEAK-harness MEASURED throughput (not a self-reported speedup), and it lifts current_best + optimization_stack the same way the orchestrator (2b) path does."""
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
        return {"status": "succeeded", "promotion_measurement": {"conc": 64, "output_throughput": measured}}

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
    # Rebench-first writes the headline HERE: current_best.tput == measured, and the geak_e2e stack entry now exists.
    assert ss.current_best["tput"] == pytest.approx(measured)
    assert any(e.get("action") == "geak_e2e" for e in ss.optimization_stack)
    assert not ss.geak_pending  # candidate cleared on promote


def _agentx_rebench_coord(tmp_path: Path) -> Coordinator:
    coord = _coord(tmp_path, baseline=100.0, best_tput=150.0)
    state = coord.shared_state
    state.benchmark_mode = "agentx"
    state.framework = "vllm"
    state.baseline_perf = {"output_throughput": 100.0, "total_throughput": 500.0, "e2e_norm_intvty_p90": 5.0}
    state.current_best.update({"input_throughput": 450.0, "total_throughput": 600.0, "e2e_norm_intvty_p90": 6.0})
    state.optimization_stack = [{"action": "explore", "variant_name": "prior-winner", "tput": 150.0}]
    state.cumulative_gain = 20.0
    state.cumulative_gain_validated = 20.0
    state.cumulative_gain_validated_stack_len = 1
    state.cumulative_gain_validated_ts = "2026-09-08T00:00:00Z"
    state.resume_pending_revalidation = True
    state.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": "reval-1"}
    state.geak_result = {
        "status": "ok",
        "throughput_speedup": 2.0,
        "accepted_config": {"flags": "--max-num-batched-tokens 4096", "env": ""},
        "validated_regimes": [{"isl": 1024, "osl": 1024, "conc": 64}],
        # Proposal measurements must never fill missing canonical rebench axes.
        "input_throughput": 9999.0,
        "total_throughput": 10000.0,
        "e2e_norm_intvty_p90": 99.0,
    }
    return coord


@pytest.mark.asyncio
@pytest.mark.parametrize("persisted_mode,ambient", [("agentx", ""), ("agentx", "0"), ("", "1")])
@pytest.mark.parametrize("bench_client", ["auto", "native", "inferencex"])
async def test_agentx_2a_refuses_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, persisted_mode: str, ambient: str, bench_client: str
) -> None:
    monkeypatch.setenv("HYPERLOOM_AGENTX", ambient)
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output")
    coord = _agentx_rebench_coord(tmp_path)
    state = coord.shared_state
    state.benchmark_mode = persisted_mode
    state.geak_result["bench_client"] = bench_client
    before_best = dict(state.current_best)
    before_pending = dict(state.geak_pending)

    async def _must_not_launch(**_kwargs):
        raise AssertionError("GEAK cannot replay the canonical AgentX workload")

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._geak_sweep.sweep_via_geak", _must_not_launch)
    outcome = await coord._validate_geak_via_geak_harness(reason="unit")

    assert outcome == {
        "validated": False,
        "status": "incomparable",
        "reason": "geak_harness_unsupported_canonical_workload",
    }
    assert state.current_best == before_best
    assert state.geak_pending == before_pending
    assert state.resume_pending_revalidation is True
    assert state.cumulative_gain_validated == 20.0
    assert state.cumulative_gain_validated_ts == "2026-09-08T00:00:00Z"


@pytest.mark.asyncio
@pytest.mark.parametrize("bench_client", ["auto", "native", "inferencex"])
async def test_persisted_legacy_mode_allows_existing_geak_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bench_client: str
) -> None:
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.setenv("HYPERLOOM_PERF_METRIC", "output")
    coord = _coord(tmp_path, baseline=100.0, best_tput=150.0)
    coord.shared_state.benchmark_mode = "synthetic"
    coord.shared_state.geak_result = {
        "status": "ok",
        "bench_client": bench_client,
        "throughput_speedup": 2.0,
        "accepted_config": {"flags": "--candidate", "env": ""},
    }
    calls = []

    async def _replay(**kwargs):
        calls.append(kwargs)
        return {"status": "succeeded", "promotion_measurement": {"output_throughput": 200.0}}

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._geak_sweep.sweep_via_geak", _replay)
    outcome = await coord._validate_geak_via_geak_harness(reason="unit")

    assert outcome["validated"] is True
    assert len(calls) == 1
    assert calls[0]["result"]["bench_client"] == bench_client


@pytest.mark.asyncio
@pytest.mark.parametrize("measurement_location", ["flat", "bench_result", "measurement"])
@pytest.mark.parametrize(
    "case",
    [
        "positive",
        "output_drop",
        "missing_axes",
        "missing_output",
        "identity_mismatch",
        "total_regression",
        "lift_refused",
    ],
)
async def test_agentx_2b_uses_current_canonical_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, measurement_location: str
) -> None:
    monkeypatch.delenv("HYPERLOOM_AGENTX", raising=False)
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    coord = _agentx_rebench_coord(tmp_path)
    state = coord.shared_state
    before_best = dict(state.current_best)
    before_stack = list(state.optimization_stack)
    before_proposal = dict(state.geak_result)
    measured = 140.0 if case == "output_drop" else 200.0
    measurement = {
        "fingerprint": "drifted" if case == "identity_mismatch" else "accepted",
        "tput": measured,
        "input_throughput": 800.0 - measured,
        "total_throughput": 800.0,
        "e2e_norm_intvty_p90": 8.0,
    }
    if case == "missing_axes":
        measurement.pop("e2e_norm_intvty_p90")
    elif case == "missing_output":
        measured = None
        measurement.pop("tput")
    elif case == "total_regression":
        measurement.update(input_throughput=350.0, total_throughput=550.0)
    elif case == "lift_refused":
        monkeypatch.setattr(coord, "_promote_geak_from_candidate", lambda *_args, **_kwargs: False)

    async def _must_not_launch(**_kwargs):
        raise AssertionError("Inconclusive AgentX 2b must not launch GEAK 2a")

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._geak_sweep.sweep_via_geak", _must_not_launch)
    variant = measurement
    if measurement_location != "flat":
        variant = {
            "fingerprint": measurement.pop("fingerprint"),
            measurement_location: measurement,
            "total_throughput": 10000.0,
            "e2e_norm_intvty_p90": 99.0,
        }
    result = {"status": "succeeded", "output_throughput": measured, "best_variant": variant, "winners": []}
    await coord._promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="accepted"))

    assert not state.geak_pending
    attempt = state.explore_attempts[-1]
    if case in {"positive", "output_drop"}:
        assert attempt["decision"] == "promoted"
        assert state.current_best["tput"] == measured
        assert state.current_best["input_throughput"] == 800.0 - measured
        assert state.current_best["total_throughput"] == 800.0
        assert state.current_best["e2e_norm_intvty_p90"] == 8.0
        assert state.current_best_measurement["tput"] == measured
        assert state.cumulative_gain_validated == pytest.approx(60.0)
        assert state.cumulative_gain_validated_stack_len == 2
        assert state.resume_pending_revalidation is False
    else:
        assert state.current_best == before_best
        assert state.optimization_stack == before_stack
        assert state.cumulative_gain == 20.0
        assert state.cumulative_gain_validated == 20.0
        assert state.cumulative_gain_validated_stack_len == 1
        assert state.cumulative_gain_validated_ts == "2026-09-08T00:00:00Z"
        assert state.resume_pending_revalidation is True
        if case in {"total_regression", "lift_refused"}:
            assert attempt["decision"] == "no_promote"
            assert attempt["status"] == "no_promote"
            assert state.geak_result["revalidation_status"] == "no_promote"
            assert result["status"] == "no_promote"
        else:
            assert attempt["status"] == "incomparable"
            assert state.geak_result["revalidation_status"] == "fallback_failed"
            assert state.geak_result["revalidation_error"] == "geak_harness_unsupported_canonical_workload"
            assert result["status"] == "incomparable"
    assert {key: state.geak_result[key] for key in before_proposal} == before_proposal


# ── Fix B: report renders a PROVISIONAL gain honestly (not "+0.00% validated") ─


def _final_breakdown(*, pending: bool, gain_v: float) -> dict:
    return {
        "session": {"image": ""},
        "outcome": {
            "baseline": {"throughput_tok_s_per_gpu": 2844.2},
            "final": {
                "throughput_tok_s_per_gpu": 3236.5,
                "gain_pct": gain_v,
                "action_path": ["geak_e2e"],
            },
            "validation": {"validated_at_stack_len": 2, "validated_ts": ""},
        },
        # The candidate's standing is settled at close, after every kernel event
        # has already closed, so the close event is the only place it can live.
        "close": {"geak_candidate": {"revalidation_pending": pending}},
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
        "best_variant": {
            "fingerprint": "abc",
            "workspace": "/runs/rebench",
            "server_log_path": "/runs/rebench/server.log",
            "launch_evidence_path": "/runs/rebench/launch_evidence.json",
            "launch_evidence": {
                "framework": "sglang",
                "actual_server_log_path": "/runs/rebench/server.log",
                "observed_server_identity": {"model_path": "/models/gemma", "tp_size": 1},
            },
        },
        "winners": [],
    }
    await coord._promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="abc"))

    ss = coord.shared_state
    expected_pct = (measured - base) / base * 100.0
    assert ss.cumulative_gain_validated == pytest.approx(expected_pct, abs=1e-6)
    assert ss.resume_pending_revalidation is False
    assert ss.cumulative_gain_validated_stack_len == 1
    assert ss.current_best_measurement["server_log_path"] == "/runs/rebench/server.log"
    assert ss.current_best_measurement["launch_evidence_path"] == "/runs/rebench/launch_evidence.json"
    assert ss.current_best_measurement["observed_server_identity"] == {
        "model_path": "/models/gemma",
        "tp_size": 1,
    }


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
    assert ss.resume_pending_revalidation is True
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
    assert ss.resume_pending_revalidation is True
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
    """`_record_geak_candidate` stores an audit-only pending candidate and leaves current_best / optimization_stack / the gain ledger untouched."""
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
    """`_promote_geak_from_candidate` lifts the headline from a MEASURED tput (never the self-reported number) and clears the pending candidate."""
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
    # ``kernel`` from the task kind alone would credit a lever the measurement
    # never proved.
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

    Those KEEPs land in the per-KERNEL ledger, which feeds ``by_kernel`` and
    ``kernel_lifecycle``. They are deliberately NOT part of the attribution
    total, which is summed from the stack ledger alone.
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


def test_the_route_level_lift_is_claimed_once_from_the_anchor_it_beat(tmp_path: Path) -> None:
    """The route-level lift is recorded once, anchored on the figure it beat.

    The per-kernel KEEPs in the journey do not shrink that claim: in V6 the
    attribution total is summed from the stack ledger alone, so the route row
    owns the whole measured delta and the per-kernel ledger is a disjoint
    account of WHICH kernels rode in on it.
    """
    coord = _coord(tmp_path, baseline=2844.209, best_tput=3000.0)
    result = _ok_result(final=3236.489)
    result["accepted_kernels"] = ["fused_moe"]
    result["kernel_journey_path"] = _journey_with_validated_keeps(tmp_path, [1.05])

    with session_scope(tmp_path):
        coord._promote_geak_from_candidate(result, measured_tput=3400.0, overlay_loaded=True)
        rows = [
            r
            for r in stack_event_parts().get("stack_adoption") or []
            if isinstance(r, dict) and str(r.get("action") or "") == "geak_e2e"
        ]

    # Once, not once per kernel that rode in on it.
    assert len(rows) == 1
    row = rows[0]
    assert row["throughput_after"] == pytest.approx(3400.0)
    # The anchor is current_best, the figure the promotion had to beat.
    assert row["throughput_before"] == pytest.approx(3000.0)
    assert row["local_gain_pct"] == pytest.approx((3400.0 - 3000.0) / 3000.0 * 100.0)
    # ...while the contribution every bucket sums shares the session baseline
    # as its denominator, which is what makes the parts add up to the chain.
    assert row["contribution_pct"] == pytest.approx((3400.0 - 3000.0) / 2844.209 * 100.0)


def test_report_shows_pending_candidate_excluded_from_headline() -> None:
    """A pending GEAK candidate renders as an audit note + warning and is NOT presented as a validated headline gain."""
    bd = {
        "session": {"image": ""},
        "outcome": {
            "baseline": {"throughput_tok_s_per_gpu": 2844.2},
            "final": {"throughput_tok_s_per_gpu": 2844.2, "gain_pct": 0.0, "action_path": []},
            "validation": {},
        },
        "close": {
            "geak_candidate": {
                "status": "awaiting_rebench",
                "self_reported_gain_pct": 13.79,
                "revalidation_pending": False,
            }
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
    """GEAK returned no kernel/head/overlay/patch AND its accepted_config equals the pre-KERNEL current_best (pure passthrough)."""
    base, current_best, measured = 8668.5946, 8900.0, 9025.191
    coord = _coord(tmp_path, baseline=base, best_tput=current_best)
    coord.shared_state.current_best["extra_server_args"] = "--max-num-batched-tokens 24576"
    coord.shared_state.current_best["extra_envs"] = {"VLLM_ROCM_USE_AITER": "1"}
    coord.shared_state.optimization_stack = [
        {"action": "explore", "variant_name": "kv-cache-fp8", "tput": current_best}
    ]
    coord.shared_state.resume_pending_revalidation = True
    coord.shared_state.geak_pending = {"status": "awaiting_rebench"}
    # geak_result is non-empty but ships NO material product; accepted_config is the pre-KERNEL current_best config
    # verbatim (passthrough, zero delta).
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
    """GEAK shipped no overlay/patch/kernel list, but its accepted_config adds a new flag vs the pre-KERNEL current_best (a kernel enabled via a config switch)."""
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
    """A validated 2b decision with an EMPTY geak_result and NO pre-existing geak_e2e stack entry has no material to validate: it is same-config noise (geak_result lost / never populated), so it must NOT promote."""
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
        # accepted_config MISSING while prev best is non-empty -> non-material (a bare mismatch must not promote and
        # wipe the existing config).
        (
            {"status": "ok", "accepted_kernels": []},
            "--max-num-batched-tokens 24576",
            {"VLLM_ROCM_USE_AITER": "1"},
            False,
        ),
        # accepted_config present but all-empty while prev best is non-empty -> non-material (same wipe hazard).
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
async def test_2b_empty_result_with_prior_geak_e2e_still_promotes(tmp_path: Path) -> None:
    """Resume revalidation: geak_result was lost (empty) but a geak_e2e stack entry already recorded the win."""
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
    """Regression: a resume revalidation of an ALREADY-promoted GEAK win must not be judged no_material."""
    base, measured = 8668.5946, 9800.0
    current_best = 9600.0  # current_best already holds the promoted GEAK win
    coord = _coord(tmp_path, baseline=base, best_tput=current_best)
    # current_best carries the GEAK accepted_config (a later kernel integrate did not change server args), so a real
    # revalidation fingerprint matches.
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
    overlay carrying them survived to the measurement. The per-kernel ledger
    refuses to credit them without proof, and the stack entry must agree: a
    rebench that stripped a dead overlay cannot be filed under ``kernel``.
    """
    base = 2844.209
    measured = 3270.0
    coord = _coord(tmp_path, baseline=base, best_tput=3042.941)
    result = _ok_result(final=3236.489)
    result["accepted_kernels"] = ["c0_triton"]
    result["accepted_heads"] = ["fused_moe_kernel"]

    coord._promote_geak_from_candidate(result, measured_tput=measured, overlay_loaded=False)

    entry = next(e for e in coord.shared_state.optimization_stack if e.get("action") == "geak_e2e")
    # ``_lift_to_current_best`` drops empty values, so "no proof" reads as no lane at all rather than an empty one --
    # either way there is no name to credit.
    assert not entry.get("accepted_kernels")
    assert not entry.get("accepted_heads")
    assert entry["overlay_loaded"] is False


def test_promote_with_loaded_overlay_keeps_kernel_names_in_stack_entry(tmp_path: Path) -> None:
    """The mirror case: proof present, so the lanes travel and the row is joint."""
    coord = _coord(tmp_path, baseline=2844.209, best_tput=3042.941)
    result = _ok_result(final=3236.489)
    result["accepted_kernels"] = ["c0_triton"]

    coord._promote_geak_from_candidate(result, measured_tput=3270.0, overlay_loaded=True)

    entry = next(e for e in coord.shared_state.optimization_stack if e.get("action") == "geak_e2e")
    assert entry["accepted_kernels"] == ["c0_triton"]
    assert entry["overlay_loaded"] is True
