# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Alignment / credibility unit tests for the GEAK e2e gain path."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from hyperloom.inference_optimizer.breakdown.recorder.assembler import event_parts
from hyperloom.inference_optimizer.breakdown.recorder.sections import STACK_EVENT_SECTIONS
from hyperloom.inference_optimizer.breakdown.reporters._renderers.final import render as render_final
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.kernel.geak_config import (
    _geak_result_has_artifact,
    _geak_rebench_confirms,
    _normalize_geak_overlay_dir,
)
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task

from ._geak_helpers import resume_pending, set_resume_pending


# ── #3: same-harness (2b) revalidation decision ──────────────────────────────


#: What the report mapper stamps on every AgentX round it accepts as evidence.
_AGENTX_EVIDENCE = {"valid_measurement": True, "diagnostic_only": False, "workload_identity": "agentx-workload"}


@pytest.mark.parametrize(
    ("measured", "baseline", "got_hash", "expected_hash", "current_best", "confirmed"),
    [
        pytest.param(115.0, 100.0, "abc", "abc", None, True, id="identity-and-engagement-hold"),
        pytest.param(115.0, 100.0, "WRONG", "abc", None, False, id="config-identity-mismatch"),
        pytest.param(101.0, 100.0, "abc", "abc", None, False, id="win-collapsed-to-baseline"),
        pytest.param(115.0, 100.0, "", "", None, True, id="no-expected-hash-skips-identity"),
        pytest.param(9623.0, 7380.7, "abc", "abc", 10067.9, False, id="not-beating-current-best"),
        pytest.param(10500.0, 7380.7, "abc", "abc", 10067.9, True, id="beating-current-best"),
        pytest.param(0.0, 100.0, "abc", "abc", None, False, id="no-measurement"),
        pytest.param(115.0, 0.0, "abc", "abc", None, False, id="no-baseline"),
        pytest.param(None, 100.0, "abc", "abc", None, False, id="unmeasured-axis"),
    ],
)
def test_a_rebench_confirms_only_its_own_engaged_config(
    measured, baseline, got_hash, expected_hash, current_best, confirmed
) -> None:
    assert (
        _geak_rebench_confirms(
            measured=measured,
            baseline=baseline,
            got_hash=got_hash,
            expected_hash=expected_hash,
            current_best=current_best,
        )
        is confirmed
    )


def test_normalize_geak_overlay_dir_picks_overlay_subdir(tmp_path: Path) -> None:
    final = tmp_path / "final"
    (final / "overlay").mkdir(parents=True)
    assert _normalize_geak_overlay_dir(str(final)) == str(final / "overlay")


def test_normalize_geak_overlay_dir_keeps_real_overlay(tmp_path: Path) -> None:
    overlay = tmp_path / "final" / "overlay"
    overlay.mkdir(parents=True)
    assert _normalize_geak_overlay_dir(str(overlay)) == str(overlay)


# ── Shared Coordinator fixture ───────────────────────────────────────────────


def _coord(tmp_path: Path, *, baseline: float, best_tput: float) -> Coordinator:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.bus = SimpleNamespace(append_and_seq=AsyncMock(), record_observation=AsyncMock())
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


def _agentx_rebench_coord(tmp_path: Path) -> Coordinator:
    coord = _coord(tmp_path, baseline=100.0, best_tput=150.0)
    state = coord.shared_state
    state.benchmark_mode = "agentx"
    state.framework = "vllm"
    state.baseline_perf = {
        "output_throughput": 100.0,
        "total_throughput": 500.0,
        "e2e_norm_intvty_p90": 5.0,
        "e2e_norm_intvty_p50": 5.0,
        "duration_seconds": 900.0,
        "request_error_rate": 0.0,
        **_AGENTX_EVIDENCE,
    }
    state.current_best.update(
        {
            "input_throughput": 450.0,
            "total_throughput": 600.0,
            "e2e_norm_intvty_p90": 6.0,
            "e2e_norm_intvty_p50": 6.0,
            "duration_seconds": 900.0,
            "request_error_rate": 0.0,
            **_AGENTX_EVIDENCE,
        }
    )
    state.optimization_stack = [{"action": "explore", "variant_name": "prior-winner", "tput": 150.0}]
    state.cumulative_gain = 20.0
    state.cumulative_gain_validated = 20.0
    state.cumulative_gain_validated_stack_len = 1
    state.cumulative_gain_validated_ts = "2026-09-08T00:00:00Z"
    set_resume_pending(state, True)
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
        "e2e_norm_intvty_p50": 99.0,
        "duration_seconds": 900.0,
        "request_error_rate": 0.0,
        **_AGENTX_EVIDENCE,
    }
    return coord


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
        "lift_refused",
    ],
)
async def test_agentx_2b_uses_current_canonical_measurement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str, measurement_location: str
) -> None:
    monkeypatch.delenv("HYPERLOOM_BENCHMARK_CONFIG", raising=False)
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
        "e2e_norm_intvty_p50": 8.0,
        "duration_seconds": 900.0,
        "request_error_rate": 0.0,
        **_AGENTX_EVIDENCE,
    }
    if case == "missing_axes":
        measurement.pop("e2e_norm_intvty_p90")
    elif case == "missing_output":
        measured = None
        measurement.pop("tput")
    elif case == "lift_refused":
        monkeypatch.setattr(coord.phase_kernel, "promote_geak_from_candidate", lambda *_args, **_kwargs: False)

    variant = measurement
    if measurement_location != "flat":
        variant = {
            "fingerprint": measurement.pop("fingerprint"),
            measurement_location: measurement,
            "total_throughput": 10000.0,
            "e2e_norm_intvty_p90": 99.0,
            "e2e_norm_intvty_p50": 99.0,
            "duration_seconds": 900.0,
            "request_error_rate": 0.0,
            **_AGENTX_EVIDENCE,
        }
    result = {"status": "succeeded", "output_throughput": measured, "best_variant": variant, "winners": []}
    await coord.writeback.promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="accepted"))

    assert not state.geak_pending
    attempt = state.explore_attempts[-1]
    # Output throughput guards nothing, so ``output_drop`` promotes like ``positive``.
    if case in {"positive", "output_drop"}:
        assert attempt["decision"] == "promoted"
        assert state.current_best["tput"] == measured
        assert state.current_best["input_throughput"] == 800.0 - measured
        assert state.current_best["total_throughput"] == 800.0
        assert state.current_best["e2e_norm_intvty_p90"] == 8.0
        assert state.current_best_measurement["tput"] == measured
        assert state.cumulative_gain_validated == pytest.approx(60.0)
        assert state.cumulative_gain_validated_stack_len == 2
        assert resume_pending(state) is False
    else:
        assert state.current_best == before_best
        assert state.optimization_stack == before_stack
        assert state.cumulative_gain == 20.0
        assert state.cumulative_gain_validated == 20.0
        assert state.cumulative_gain_validated_stack_len == 1
        assert state.cumulative_gain_validated_ts == "2026-09-08T00:00:00Z"
        assert resume_pending(state) is True
        if case == "missing_output":
            assert attempt["status"] == "failed"
            assert state.geak_result["revalidation_status"] == "failed"
            assert state.geak_result["revalidation_error_class"] == "rebench_no_measurement"
            assert result["status"] == "failed"
        else:
            assert attempt["decision"] == "no_promote"
            assert attempt["status"] == "no_promote"
            assert state.geak_result["revalidation_status"] == "no_promote"
            assert result["status"] == "no_promote"
    assert {key: state.geak_result[key] for key in before_proposal} == before_proposal


# ── Report renders the validated gain as recorded ─


def _final_breakdown(*, pending: bool, gain_v: float) -> dict:
    return {
        "session": {"image": ""},
        "outcome": {
            "baseline": {"output_throughput": 2844.2},
            "final": {
                "output_throughput": 3236.5,
                "gain_pct": gain_v,
                "action_path": ["geak_e2e"],
            },
            "validation": {"validated_at_stack_len": 2, "validated_ts": ""},
        },
        # The candidate's standing is settled at close, after every kernel event
        # has already closed, so the close event is the only place it can live.
        "close": {"geak_candidate": {"revalidation_pending": pending}},
    }


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


def _revalidate_task(*, expected_hash: str, recipe_generation: int | None = None) -> Task:
    params: dict = {
        "source": "stack_revalidate",
        "geak_fallback": True,
        "expected_cfg_hash": expected_hash,
    }
    if recipe_generation is not None:
        params["recipe_generation"] = recipe_generation
    return Task(
        task_id="reval-1",
        kind="explore",
        state="succeeded",
        params=params,
        idempotency_key="reval-1",
    )


@pytest.mark.asyncio
async def test_2b_stamps_validated_from_orchestrator_rebench(tmp_path: Path) -> None:
    """decision==validated → validated == (measured − baseline)/baseline, same harness."""
    base, measured = 2844.209, 3270.0  # ~+14.97%, engaged + identity matches
    coord = _coord(tmp_path, baseline=base, best_tput=3236.489)
    coord.shared_state.optimization_stack = [{"action": "geak_e2e", "variant_name": "geak_e2e", "tput": 3236.489}]
    set_resume_pending(coord.shared_state, True)

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
    await coord.writeback.promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="abc"))

    ss = coord.shared_state
    expected_pct = (measured - base) / base * 100.0
    assert ss.cumulative_gain_validated == pytest.approx(expected_pct, abs=1e-6)
    assert resume_pending(ss) is False
    assert ss.cumulative_gain_validated_stack_len == 1
    assert ss.current_best_measurement["server_log_path"] == "/runs/rebench/server.log"
    assert ss.current_best_measurement["launch_evidence_path"] == "/runs/rebench/launch_evidence.json"
    assert ss.current_best_measurement["observed_server_identity"] == {
        "model_path": "/models/gemma",
        "tp_size": 1,
    }


@pytest.mark.asyncio
async def test_2b_identity_mismatch_is_refused_with_nothing_promoted(tmp_path: Path) -> None:
    """A rebench that returned a measurement of another config did not confirm the candidate."""
    base, measured = 2844.209, 3270.0  # engaged, but fingerprint won't match
    coord = _coord(tmp_path, baseline=base, best_tput=3236.489)
    coord.shared_state.optimization_stack = [{"action": "geak_e2e", "variant_name": "geak_e2e", "tput": 3236.489}]
    set_resume_pending(coord.shared_state, True)
    coord.shared_state.geak_pending = {
        "status": "awaiting_rebench",
        "revalidation_task_id": "reval-1",
    }

    before_best = dict(coord.shared_state.current_best)
    before_stack = list(coord.shared_state.optimization_stack)

    result = {
        "output_throughput": measured,
        "best_variant": {"fingerprint": "DRIFTED"},  # != expected "abc"
        "winners": [],
    }
    await coord.writeback.promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="abc"))

    ss = coord.shared_state
    assert ss.current_best == before_best
    assert ss.optimization_stack == before_stack
    assert ss.cumulative_gain_validated == pytest.approx(0.0)
    assert not ss.geak_pending
    assert resume_pending(ss) is True
    assert ss.geak_result["revalidation_status"] == "no_promote"
    assert result["status"] == "no_promote"


@pytest.mark.asyncio
async def test_2b_no_promote_when_rebench_loses_to_current_best(tmp_path: Path) -> None:
    """A GEAK rebench that beats baseline but loses to current_best is measured, not a KEEP."""
    base, current_best, measured = 7380.7, 10067.9, 9623.0
    coord = _coord(tmp_path, baseline=base, best_tput=current_best)
    coord.shared_state.optimization_stack = [
        {"action": "explore", "variant_name": "kv-cache-fp8", "tput": current_best}
    ]
    set_resume_pending(coord.shared_state, True)
    coord.shared_state.geak_pending = {"status": "awaiting_rebench"}

    result = {
        "output_throughput": measured,
        "best_variant": {"fingerprint": "abc"},
        "winners": [],
    }
    await coord.writeback.promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="abc"))

    ss = coord.shared_state
    assert ss.current_best["tput"] == pytest.approx(current_best)
    assert ss.cumulative_gain_validated == pytest.approx(0.0)
    assert not any(e.get("action") == "geak_e2e" for e in ss.optimization_stack)
    assert resume_pending(ss) is True
    assert not ss.geak_pending
    assert ss.geak_result["revalidation_status"] == "no_promote"


# ── Rebench-first: candidate recorded, headline deferred to measured rebench ──


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["accuracy_drop", "accuracy_unavailable", "intvty_regression"])
@pytest.mark.parametrize("expected_hash", ["abc", ""])
async def test_2b_native_revert_is_conclusive(tmp_path: Path, reason: str, expected_hash: str) -> None:
    coord = _coord(tmp_path, baseline=100.0, best_tput=110.0)
    coord.shared_state.geak_result = {
        **_ok_result(final=150.0),
        "kernel_journey_path": _journey_with_validated_keeps(tmp_path, [1.5]),
    }
    set_resume_pending(coord.shared_state, True)

    result = {
        "status": "succeeded",
        "output_throughput": None,
        "winners": [],
        "per_variant_outcomes": [{"outcome": "REVERT", "reason": reason, "fingerprint": "abc"}],
    }
    await coord.writeback.promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash=expected_hash))

    assert coord.shared_state.current_best["tput"] == 110.0
    assert coord.shared_state.cumulative_gain_validated == 0.0
    assert not coord.shared_state.optimization_stack
    assert not coord.shared_state.geak_pending
    assert resume_pending(coord.shared_state)
    assert coord.shared_state.geak_result["revalidation_status"] == "no_promote"
    assert coord.shared_state.geak_result["revalidation_error"] == reason


@pytest.mark.asyncio
@pytest.mark.parametrize("reason", ["both_axes_regressed (intvty 100.0->50.0 tput 1000.0->900.0)", "new gate wording"])
async def test_structured_native_rejection_is_terminal(tmp_path, reason):
    coord = _coord(tmp_path, baseline=100.0, best_tput=110.0)
    coord.shared_state.geak_result = _ok_result(final=150.0)
    set_resume_pending(coord.shared_state, True)
    coord.shared_state.geak_pending = {"status": "awaiting_rebench", "revalidation_task_id": "reval-1"}
    await coord.writeback.promote_to_shared_state(
        "explore",
        {
            "status": "succeeded",
            "output_throughput": None,
            "winners": [],
            "per_variant_outcomes": [
                {
                    "outcome": "REVERT",
                    "reason": reason,
                    "fingerprint": "abc",
                    "gates": [{"gate": "graded_axes", "passed": False, "reason": reason}],
                }
            ],
        },
        task=_revalidate_task(expected_hash="abc"),
    )
    assert coord.shared_state.current_best["tput"] == 110.0
    assert coord.shared_state.geak_result["revalidation_status"] == "no_promote"
    assert coord.shared_state.geak_result["revalidation_error"] == reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "outcome,reason,fingerprint,settled",
    [("FAILED", "no_measurement", "abc", "failed"), ("REVERT", "accuracy_drop", "different", "no_promote")],
    ids=["no-measurement-stays-retryable", "another-config-reverted-is-refused"],
)
async def test_2b_round_without_a_winner_settles_by_whether_it_measured(
    tmp_path: Path, outcome: str, reason: str, fingerprint: str, settled: str
) -> None:
    coord = _coord(tmp_path, baseline=100.0, best_tput=110.0)
    set_resume_pending(coord.shared_state, True)
    await coord.writeback.promote_to_shared_state(
        "explore",
        {
            "output_throughput": None,
            "winners": [],
            "per_variant_outcomes": [{"outcome": outcome, "reason": reason, "fingerprint": fingerprint}],
        },
        task=_revalidate_task(expected_hash="abc"),
    )
    assert coord.shared_state.current_best["tput"] == 110.0
    assert coord.shared_state.geak_result["revalidation_status"] == settled


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
    coord.phase_kernel._record_geak_candidate(_ok_result(final=3236.489))

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
    """`promote_geak_from_candidate` lifts the headline from a MEASURED tput (never the self-reported number) and clears the pending candidate."""
    base = 2844.209
    measured = 3270.0
    coord = _coord(tmp_path, baseline=base, best_tput=3042.941)
    result = _ok_result(final=3236.489)  # self-reported win
    coord.shared_state.geak_result = result
    coord.phase_kernel._record_geak_candidate(result)
    assert coord.shared_state.geak_pending.get("status") == "awaiting_rebench"
    coord.phase_kernel.promote_geak_from_candidate(
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
    assert resume_pending(ss) is False
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
    coord.phase_kernel.promote_geak_from_candidate(result, measured_tput=3270.0, overlay_loaded=True)

    entry = next(e for e in coord.shared_state.optimization_stack if e.get("action") == "geak_e2e")
    assert entry["lever_kind"] == "kernel"


def _journey_with_validated_keeps(tmp_path: Path, ratios: list[float]) -> str:
    """Write a journey whose KEEPs each carry a validated ``(base,new)`` pair.

    Those KEEPs land in the per-KERNEL ledger. They are deliberately NOT part
    of the attribution total, which is summed from the stack ledger alone.
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
        coord.phase_kernel.promote_geak_from_candidate(result, measured_tput=3400.0, overlay_loaded=True)
        rows = [
            r
            for r in event_parts(STACK_EVENT_SECTIONS).get("stack_adoption") or []
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
    """A pending GEAK candidate renders as an audit note + warning; the validated gain is shown as recorded."""
    bd = {
        "session": {"image": ""},
        "outcome": {
            "baseline": {"output_throughput": 2844.2},
            "final": {"output_throughput": 2844.2, "gain_pct": 0.0, "action_path": []},
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
    assert sec.key_facts == [
        "Final: 2844.20 tok/s.",
        "Delta vs baseline: +0.00 tok/s.",
        "Validated cumulative gain: 0.00%.",
        "GEAK candidate (self-reported +13.79%) is AWAITING a main-flow rebench — excluded from the headline gain "
        "and final stack until a measured rebench validates it.",
    ]
    assert sec.warnings == [
        "A GEAK(GEAK) e2e candidate self-reported a win but has NOT been confirmed by a same-harness main-flow "
        "rebench, so it is intentionally kept out of current_best / action_path / the validated gain. Its "
        "self-reported number is audit-only and must not be presented as the headline result."
    ]


# ── 2b material guard: same-config rebench noise must not stamp kernel gain ───


def test_2b_empty_geak_result_is_not_measured(tmp_path: Path) -> None:
    """A lost or never-populated GEAK result has nothing to measure."""
    coord = _coord(tmp_path, baseline=8668.5946, best_tput=8900.0)
    coord.shared_state.geak_result = {}

    assert coord.writeback.geak_rebench_params(reason="geak_e2e_win") == {
        "skipped": True,
        "reason": "geak_no_material",
    }


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
    set_resume_pending(coord.shared_state, True)
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
    coord.phase_kernel._record_geak_candidate(result_blob)

    result = {
        "output_throughput": measured,
        "best_variant": {"fingerprint": "abc"},
        "winners": [],
    }
    await coord.writeback.promote_to_shared_state("explore", result, task=_revalidate_task(expected_hash="abc"))

    ss = coord.shared_state
    expected_pct = (measured - base) / base * 100.0
    assert ss.current_best["tput"] == pytest.approx(measured)
    assert ss.cumulative_gain_validated == pytest.approx(expected_pct)
    assert any(e.get("action") == "geak_e2e" for e in ss.optimization_stack)
    assert not ss.geak_pending


# ── artifact helper unit boundaries ──────────────────────────────────────────


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        ({"accepted_kernels": ["", "  "]}, False),
        ({"accepted_kernels": ["fused_rope"]}, True),
        ({"accepted_heads": ["attention"]}, True),
        ({"final_overlay": "   "}, False),
        ({"final_patch": "/geak/final.patch"}, True),
    ],
)
def test_geak_result_has_artifact_boundaries(result, expected) -> None:
    assert _geak_result_has_artifact(result) is expected


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

    coord.phase_kernel.promote_geak_from_candidate(result, measured_tput=measured, overlay_loaded=False)

    entry = next(e for e in coord.shared_state.optimization_stack if e.get("action") == "geak_e2e")
    # ``lift_to_current_best`` drops empty values, so "no proof" reads as no lane at all rather than an empty one --
    # either way there is no name to credit.
    assert not entry.get("accepted_kernels")
    assert not entry.get("accepted_heads")
    assert entry["overlay_loaded"] is False


def test_promote_with_loaded_overlay_keeps_kernel_names_in_stack_entry(tmp_path: Path) -> None:
    """The mirror case: proof present, so the lanes travel and the row is joint."""
    coord = _coord(tmp_path, baseline=2844.209, best_tput=3042.941)
    result = _ok_result(final=3236.489)
    result["accepted_kernels"] = ["c0_triton"]

    coord.phase_kernel.promote_geak_from_candidate(result, measured_tput=3270.0, overlay_loaded=True)

    entry = next(e for e in coord.shared_state.optimization_stack if e.get("action") == "geak_e2e")
    assert entry["accepted_kernels"] == ["c0_triton"]
    assert entry["overlay_loaded"] is True
