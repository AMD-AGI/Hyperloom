# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK acceptance follows native retention and the fresh replay's graded axes."""

from __future__ import annotations

from copy import deepcopy

import pytest

from hyperloom.inference_optimizer.breakdown.recorder import assemble_parts
from hyperloom.inference_optimizer.breakdown.recorder.kernel_event import ROUTE_GEAK, make_kernel_recorder
from hyperloom.inference_optimizer.session.sbd_v6 import read_timeline_events
from hyperloom.inference_optimizer.session.session_binding import session_scope
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.coordinator_helpers import _geak_overlay_digest
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.orchestrator.state.task_registry import Task


@pytest.fixture
def promotion(tmp_path, monkeypatch):
    monkeypatch.setenv("HYPERLOOM_AGENTX", "1")
    monkeypatch.delenv("HYPERLOOM_PERF_METRIC", raising=False)
    monkeypatch.setenv("HYPERLOOM_PERF_NOISE_PCT", "2")
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(
        framework="sglang",
        benchmark_mode="agentx",
        baseline_tput=100.0,
        baseline_accuracy=0.8,
        baseline_perf={"total_throughput": 900.0, "intvty_p90": 100.0},
        current_best={"action": "explore", "tput": 110.0, "total_throughput": 1000.0, "intvty_p90": 100.0},
        model_path="/models/test",
        gpu_type="mi355x",
        isl=1024,
        osl=1024,
        conc=64,
        optimization_stack=[{"action": "explore", "variant_name": "anchor", "tput": 110.0}],
        cumulative_gain_validated=100.0 / 9.0,
    )
    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text("# CPU loadability fixture\n")
    result = {
        "status": "ok",
        "throughput_speedup": 1.2,
        "accepted_config": {"flags": "--max-running-requests 64", "env": "SGLANG_USE_AITER=1"},
        "accepted_kernels": [{"short_name": "candidate_kernel", "kind": "authored", "e2e_delta_pct": 3.0}],
        "final_overlay": str(overlay),
        "total_throughput": 1500.0,
        "intvty_p90": 100.0,
        "validated_regimes": [{"isl": 1024, "osl": 1024, "conc": 64}],
    }
    with session_scope(tmp_path):
        recorder = make_kernel_recorder(macro_cycle=0, route=ROUTE_GEAK)
        assert recorder is not None
        recorder.begin(tput_before=110.0)
        coord._kernel_timeline_recorder = recorder
        yield coord, result, recorder


def rebench_task(result):
    return Task(
        task_id="reval-1",
        kind="explore",
        state="succeeded",
        params={
            "source": "resume_stack_revalidate",
            "geak_fallback": True,
            "expected_cfg_hash": "candidate",
            "expected_overlay": result["final_overlay"],
            "expected_overlay_digest": _geak_overlay_digest(result["final_overlay"]),
        },
        idempotency_key="reval-1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lane,claimed_intvty,fresh_intvty,accepted",
    [
        ("direct", 50.0, None, False),
        ("direct", 100.0, None, True),
        ("2a", 100.0, 50.0, False),
        ("2a", 50.0, 100.0, True),
        ("2b", 100.0, 50.0, False),
        ("2b", 50.0, 100.0, True),
        ("2a", 50.0, None, True),
        ("2b", 50.0, None, True),
    ],
)
async def test_geak_acceptance_requires_native_retention(
    promotion, monkeypatch, lane, claimed_intvty, fresh_intvty, accepted
):
    coord, result, recorder = promotion
    state = coord.shared_state
    result["intvty_p90"] = claimed_intvty
    state.geak_result = deepcopy(result)
    coord._record_geak_candidate(result)
    state.resume_pending_revalidation = True
    before_best = deepcopy(state.current_best)
    before_stack = deepcopy(state.optimization_stack)
    before_gain = state.cumulative_gain_validated
    measurement = {"conc": 64, "output_throughput": 120.0, "accuracy": 0.9, "fingerprint": "candidate"}
    if fresh_intvty is not None:
        measurement.update(total_throughput=1200.0, input_throughput=1080.0, intvty_p90=fresh_intvty)
    sweep_calls = []

    async def replay(**kwargs):
        sweep_calls.append(kwargs)
        return {"status": "succeeded", "promotion_measurement": measurement}

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._geak_sweep.sweep_via_geak", replay)
    if lane == "direct":
        returned = coord._promote_geak_from_candidate(result, measured_tput=120.0, overlay_loaded=True)
        assert returned is accepted
    elif lane == "2a":
        returned = await coord._validate_geak_via_geak_harness(reason="retention regression")
        assert returned["validated"] is accepted
        assert len(sweep_calls) == 1
    else:
        await coord._promote_to_shared_state(
            "explore",
            {"output_throughput": 120.0, "best_variant": measurement, "winners": []},
            task=rebench_task(result),
        )
        assert not sweep_calls, "A conclusive retention veto must not trigger another harness"
    assert not state.geak_pending and not state.resume_pending_revalidation
    parts = assemble_parts(coord.session_dir)
    final_ops = [op for op in parts["operations"] if op["kind"] == "kernel_optimizer_run"]
    assert final_ops
    if accepted:
        assert state.current_best["tput"] == 120.0
        assert state.current_best["final_overlay"] == result["final_overlay"]
        assert len(state.optimization_stack) == len(before_stack) + 1
        assert state.kernel_integrate_attempts["candidate_kernel"]["validated"] is True
        expected_total = 1500.0 if lane == "direct" else 1200.0 if fresh_intvty is not None else None
        assert state.current_best.get("total_throughput") == expected_total
        expected_gain = (expected_total - 900.0) / 900.0 * 100.0 if expected_total is not None else 20.0
        assert state.cumulative_gain_validated == pytest.approx(expected_gain)
        assert all(op["status"] == "succeeded" for op in final_ops)
    else:
        assert state.current_best == before_best
        assert state.optimization_stack == before_stack
        assert state.cumulative_gain_validated == before_gain
        assert not state.kernel_integrate_attempts
        assert state.geak_result["revalidation_status"] == "no_promote"
        assert state.geak_result["final_validation"]["reason"] == "graded_comparison_rejected"
        assert all(op["status"] == "failed" for op in final_ops)
        assert not any(op["name"] == "geak_e2e" for op in parts["operations"])
        assert not any(gate["status"] == "passed" for op in final_ops for gate in op.get("gates", []))
    recorder.finish(verdict="adopted" if accepted else "no_gain", tput_after=state.current_best["tput"])
    if lane == "2b":
        event = next(event for event in read_timeline_events(coord.session_dir) if event.get("type") == "kernel")
        attempts = event["ext"]["geak"]["rebench"]["attempts"]
        assert len(attempts) == 1
        assert attempts[0]["decision"] == ("validated" if accepted else "no_promote")
    state.save(coord.session_dir)
    restored = SharedState.load_or_init(coord.session_dir)
    assert restored.current_best == state.current_best
    assert restored.geak_result == state.geak_result


@pytest.mark.asyncio
@pytest.mark.parametrize("via_orchestrator", [False, True])
async def test_fallback_rejection_is_conclusive_and_not_validated(promotion, monkeypatch, via_orchestrator):
    coord, result, _ = promotion
    state = coord.shared_state
    state.geak_result = deepcopy(result)
    coord._record_geak_candidate(result)
    state.resume_pending_revalidation = True
    before = deepcopy(state.current_best)
    before_gain = state.cumulative_gain_validated

    async def replay(**kwargs):
        return {"status": "succeeded", "promotion_measurement": {"output_throughput": 105.0, "accuracy": 0.9}}

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._geak_sweep.sweep_via_geak", replay)
    if via_orchestrator:
        await coord._promote_to_shared_state(
            "explore",
            {"output_throughput": 120.0, "best_variant": {"fingerprint": "different"}, "winners": []},
            task=rebench_task(result),
        )
    else:
        returned = await coord._validate_geak_via_geak_harness(reason="retention regression")
        assert returned == {
            "validated": False,
            "status": "no_promote",
            "reason": "rebench_did_not_beat_current_best",
        }
    assert state.current_best == before
    assert state.cumulative_gain_validated == before_gain
    assert not state.kernel_integrate_attempts
    assert not state.geak_pending and not state.resume_pending_revalidation
    assert state.geak_result["revalidation_status"] == "no_promote"
    assert state.geak_result["final_validation"]["decision"] == "REJECTED"


@pytest.mark.parametrize("effective_flags", ["--mem-fraction-static 0.9", ""])
def test_geak_promotion_retains_fresh_launch_controls(promotion, effective_flags):
    coord, result, _ = promotion
    state = coord.shared_state
    state.current_best.update(
        extra_server_args="--disable-radix-cache --mem-fraction-static 0.7",
        extra_envs={"SGLANG_AITER_MLA_PERSIST": "1", "SGLANG_USE_AITER": "1"},
    )
    measurement = {
        "extra_server_args": effective_flags,
        "effective_extra_server_args": effective_flags,
        "extra_envs": {"SGLANG_USE_AITER": "1"},
        "candidate_extra_server_args": "",
        "candidate_extra_envs": {},
        "args_mode": "replace",
        "remove_args": ["--disable-radix-cache"],
        "unset_envs": ["SGLANG_AITER_MLA_PERSIST"],
        "fingerprint": "fresh-config",
        "recipe_delta": {
            "extra_server_args": "",
            "extra_envs": {},
            "remove_args": [],
            "unset_envs": ["SGLANG_AITER_MLA_PERSIST"],
            "args_mode": "append",
        },
    }

    assert coord._promote_geak_from_candidate(
        result, measured_tput=120.0, measurement_provenance=measurement, overlay_loaded=False
    )
    state.save(coord.session_dir)
    restored = SharedState.load_or_init(coord.session_dir)
    best = restored.current_best
    assert best["extra_server_args"] == effective_flags
    assert best["extra_envs"] == {"SGLANG_USE_AITER": "1"}
    assert best["args_mode"] == "replace"
    assert best["remove_args"] == ["--disable-radix-cache"]
    assert best["unset_envs"] == ["SGLANG_AITER_MLA_PERSIST"]
    entry = restored.optimization_stack[-1]
    assert entry["fingerprint"] == "fresh-config"
    assert entry["recipe_delta"] == measurement["recipe_delta"]


def test_geak_complete_config_survives_direct_promotion(promotion):
    coord, result, _ = promotion
    coord.shared_state.current_best["extra_server_args"] = "--disable-radix-cache"
    result["accepted_config"] = {"flags": "", "env_map": {}, "args_mode": "replace"}
    coord._record_geak_candidate({**result, "final_throughput_tok_s": 120.0})
    assert coord.shared_state.geak_pending["args_mode"] == "replace"
    assert coord._promote_geak_from_candidate(result, measured_tput=120.0, overlay_loaded=False)
    assert coord.shared_state.current_best["extra_server_args"] == ""
    assert coord.shared_state.current_best["args_mode"] == "replace"


@pytest.mark.parametrize("accepted_mode", ["append", "replace"])
def test_geak_replay_without_launch_fields_preserves_stack_controls(promotion, accepted_mode):
    coord, result, _ = promotion
    state = coord.shared_state
    state.current_best.update(
        extra_server_args="--chunked-prefill-size 1024 --mem-fraction-static 0.9",
        extra_envs={"SGLANG_AITER_MLA_PERSIST": "3"},
        args_mode="replace",
        remove_args=["--disable-radix-cache"],
        unset_envs=["SGLANG_AITER_MLA_PERSIST"],
    )
    result["accepted_config"] = {"flags": "--mem-fraction-static 0.95", "env_map": {}, "args_mode": accepted_mode}
    assert coord._promote_geak_from_candidate(
        result, measured_tput=120.0, measurement_provenance={"accuracy": 0.9}, overlay_loaded=False
    )
    best = state.current_best
    expected = (
        "--mem-fraction-static 0.95"
        if accepted_mode == "replace"
        else "--chunked-prefill-size 1024 --mem-fraction-static 0.95"
    )
    assert best["extra_server_args"] == expected
    assert best["args_mode"] == "replace"
    assert best["unset_envs"] == ["SGLANG_AITER_MLA_PERSIST"]
    assert best["extra_envs"] == {"SGLANG_AITER_MLA_PERSIST": "3"}


def test_geak_replay_removal_retains_other_inherited_flags(promotion):
    coord, result, _ = promotion
    recipe = coord.session_dir / "baseline.yaml"
    recipe.write_text("benchmark:\n  envs:\n    EXTRA_SGLANG_ARGS: --disable-radix-cache --mem-fraction-static 0.7\n")
    coord.shared_state.baseline_config_path = str(recipe)
    coord.shared_state.current_best["extra_server_args"] = "--chunked-prefill-size 1024"
    result["accepted_config"] = {"remove_args": ["--disable-radix-cache"]}
    assert coord._promote_geak_from_candidate(result, measured_tput=120.0, overlay_loaded=False)
    best = coord.shared_state.current_best
    assert best["extra_server_args"] == "--mem-fraction-static 0.7 --chunked-prefill-size 1024"
    assert best["args_mode"] == "replace"
