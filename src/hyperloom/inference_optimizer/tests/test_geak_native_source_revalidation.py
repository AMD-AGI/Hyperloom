# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK source products are staged, then kept or reverted by a canonical transactional measurement."""

from __future__ import annotations

import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.actions.executors.benchmark_run.accuracy import AccuracyRoundOutcome
from hyperloom.orchestrator.actions.executors.integrate import _integrate_bench
from hyperloom.orchestrator.actions.executors.integrate._integrate_vocabulary import (
    BENCH_UNMEASURED_CLASS,
    RESULT_BENCHED,
)
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.actions.stop_attribution import SESSION_TIME_EXHAUSTED_CLASS, STOPPED_BY_THE_RUN
from hyperloom.orchestrator.phases.geak_native_revalidation import ORIGIN, source_rebench_params
from hyperloom.orchestrator.phases.geak_rebench import geak_candidate_is_adjudicated
from hyperloom.orchestrator.tests._helpers import init_git_repo, patch_integrate_patch_roots
from .test_geak_revalidation_dispatch import coordinator as coordinator

_INCUMBENT = {"tput": 110.0, "extra_server_args": "--incumbent", "extra_envs": {"SGLANG_USE_AITER": "0"}}
_BUDGET_STOP = STOPPED_BY_THE_RUN[SESSION_TIME_EXHAUSTED_CLASS]
_APPLY_ERRORS = [{"patch": "final.patch", "stderr": "error: patch failed: kernel.py:1"}]


def _candidate(c, tmp_path):
    state = c.shared_state
    state.benchmark_mode = "agentx"
    state.agentx_backend = "inferencex"
    state.baseline_config_path = str(tmp_path / "native.yaml")
    state.baseline_tput = 100.0
    state.current_best = copy.deepcopy(_INCUMBENT)
    directory = tmp_path / "geak-eval"
    directory.mkdir()
    patch = directory / "final.patch"
    patch.write_text("diff --git a/kernel.py b/kernel.py\n--- a/kernel.py\n+++ b/kernel.py\n@@ -1 +1 @@\n-old\n+new\n")
    state.geak_result = {
        "status": "ok",
        "eval_dir": str(directory),
        "final_patch": str(patch),
        "final_throughput_tok_s": 99999.0,
        "accepted_config": {"flags": "--candidate", "env_map": {"SGLANG_USE_AITER": "1"}},
        "accepted_kernels": [{"short_name": "kernel", "e2e_delta_pct": 10, "final_patch": str(patch)}],
    }
    return patch


def _changed_after_staging(c, monkeypatch, change):
    """Run ``change(params)`` on the product once it is staged, before the revalidation reads it."""
    stage = c.writeback.geak_rebench_params

    def staged_then_changed(*, reason):
        params = stage(reason=reason)
        change(params)
        return params

    monkeypatch.setattr(c.writeback, "geak_rebench_params", staged_then_changed)


def test_source_route_retains_configuration_and_stages_exact_patches(coordinator, tmp_path):
    patch = _candidate(coordinator, tmp_path)
    params = coordinator.writeback.geak_rebench_params(reason="unit")
    assert params["origin"] == ORIGIN
    assert params["config_path"] == coordinator.shared_state.baseline_config_path
    assert params["extra_server_args"] == "--candidate"
    assert params["extra_envs"] == {"SGLANG_USE_AITER": "1"}
    assert "--incumbent" in params["base_extra_args"]
    assert Path(params["patches"][0]).read_bytes() == patch.read_bytes()
    assert "final_throughput_tok_s" not in params


@pytest.mark.parametrize(("mode", "backend"), [("synthetic", ""), ("agentx", "mlperf")])
def test_only_an_inferencex_agentx_session_takes_the_source_route(coordinator, tmp_path, mode, backend):
    _candidate(coordinator, tmp_path)
    coordinator.shared_state.benchmark_mode = mode
    coordinator.shared_state.agentx_backend = backend
    assert source_rebench_params(coordinator, reason="unit", result={}, launch={}, overlay="") is None


@pytest.mark.asyncio
async def test_source_candidate_is_measured_as_a_covered_step_with_no_critic_proposal(
    coordinator, tmp_path, monkeypatch
):
    _candidate(coordinator, tmp_path)
    measured = []
    adopted = []
    canonical = {"output_throughput": 120.0, "agentx_epoch": 3}

    async def integrate(ctx):
        measured.append(ctx)
        return {"status": "kept", "bench_result": canonical}

    async def promote(kind, result, *, task):
        assert kind == "integrate_patch"
        assert result["bench_result"] == canonical
        coordinator.shared_state.optimization_stack.append({"task_id": task.task_id, "action": kind})

    coordinator.sub.register_executor("integrate_patch", integrate)
    monkeypatch.setattr(coordinator.writeback, "promote_to_shared_state", promote)
    monkeypatch.setattr(
        coordinator.phase_kernel, "record_geak_adopted_kernels", lambda *args, **kwargs: adopted.append(kwargs)
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.state.shared_state.resolve_graded_comparison",
        lambda *args, **kwargs: SimpleNamespace(comparable=True, candidate=120.0, reference=100.0),
    )
    await coordinator.phase_kernel._revalidate_geak_candidate(reason="unit", handoff_cards=None)
    assert len(measured) == 1
    assert measured[0].task.kind == "integrate_patch"
    # No registry row carries this marker; it is what lets the integration run with no Critic verdict on record.
    assert measured[0].extra["covered_step"] is True
    assert await coordinator.bus.tail(topic="proposal") == []
    assert not coordinator.state.pending_proposals
    assert adopted[0]["measured_tput"] == 120.0
    assert adopted[0]["overlay_loaded"] is False
    assert coordinator.shared_state.geak_result["canonical_revalidation"] == canonical
    assert coordinator.shared_state.geak_pending == {}


async def _settle(coordinator, tmp_path, result):
    """Settle the candidate on ``result`` from the integration; return the product as GEAK reported it."""
    _candidate(coordinator, tmp_path)
    reported = dict(coordinator.shared_state.geak_result)

    async def integrate(ctx):
        return dict(result)

    coordinator.sub.register_executor("integrate_patch", integrate)
    await coordinator.phase_kernel._revalidate_geak_candidate(reason="unit", handoff_cards=None)
    return reported


@pytest.mark.asyncio
async def test_unkept_measurement_settles_no_promote_without_promotion(coordinator, tmp_path):
    measured = {RESULT_BENCHED: True, "status": "reverted", "reason": "canonical replay did not improve"}
    reported = await _settle(coordinator, tmp_path, measured)
    state = coordinator.shared_state
    assert state.geak_pending == {}
    assert state.current_best == _INCUMBENT
    assert state.geak_result["revalidation_status"] == "no_promote"
    assert state.geak_result["revalidation_error"] == "canonical replay did not improve"
    assert not state.optimization_stack
    assert geak_candidate_is_adjudicated(state.geak_result, reported, harness_can_replay=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "error"),
    [
        pytest.param(
            {
                RESULT_BENCHED: True,
                "status": "failed",
                "error_class": _BUDGET_STOP.error_class,
                "error": _BUDGET_STOP.interrupted,
            },
            _BUDGET_STOP.interrupted,
            id="session_budget_stopped_the_bench",
        ),
        pytest.param(
            {
                RESULT_BENCHED: True,
                "status": "failed",
                "error_class": BENCH_UNMEASURED_CLASS,
                "error": "port 8000 is in use",
            },
            "port 8000 is in use",
            id="environment_kept_the_bench_from_measuring",
        ),
        pytest.param(
            {
                "status": "failed",
                "error_class": "integrate_restore_incomplete",
                "error": "previous integration still requires recovery",
            },
            "previous integration still requires recovery",
            id="earlier_integration_left_unrestored",
        ),
        pytest.param(
            {"status": "skipped", "skipped_reason": "multi_node_unsupported", "reason": "not supported multi-node"},
            "not supported multi-node",
            id="multi_node_skip",
        ),
        pytest.param(
            {"status": "apply_failed", "error_class": "git_apply_failed", "error": _APPLY_ERRORS},
            str(_APPLY_ERRORS),
            id="patch_did_not_apply",
        ),
    ],
)
async def test_unmeasured_integration_settles_failed_and_stays_open(coordinator, tmp_path, result, error):
    reported = await _settle(coordinator, tmp_path, result)
    state = coordinator.shared_state
    assert state.geak_result["revalidation_status"] == "failed"
    assert state.geak_result["revalidation_error"] == error
    assert state.geak_pending == {}
    assert state.current_best == _INCUMBENT
    assert not state.optimization_stack
    assert not geak_candidate_is_adjudicated(state.geak_result, reported, harness_can_replay=False)


def _framework_tree(tmp_path, monkeypatch):
    """A git framework tree the staged patch applies to, registered where the integration looks for one."""
    repo = tmp_path / "framework"
    init_git_repo(repo, seed_file="kernel.py", seed_text="old\n")
    patch_integrate_patch_roots(monkeypatch, tmp_path)
    return repo


@pytest.mark.asyncio
async def test_bench_stopped_by_the_session_budget_leaves_the_candidate_open(coordinator, tmp_path, monkeypatch):
    _candidate(coordinator, tmp_path)
    reported = dict(coordinator.shared_state.geak_result)
    repo = _framework_tree(tmp_path, monkeypatch)

    async def stopped_bench(self, bench, **kwargs):
        return {"status": "failed", "error_class": SESSION_TIME_EXHAUSTED_CLASS}, {}

    monkeypatch.setattr(IntegratePatchExecutor, "_bench_patch", stopped_bench)
    coordinator.sub.register_executor("integrate_patch", IntegratePatchExecutor(session_dir=coordinator.session_dir))
    await coordinator.phase_kernel._revalidate_geak_candidate(reason="unit", handoff_cards=None)
    state = coordinator.shared_state
    assert state.geak_result["revalidation_status"] == "failed"
    assert state.geak_result["revalidation_error"] == _BUDGET_STOP.interrupted
    assert state.current_best == _INCUMBENT
    assert (repo / "kernel.py").read_text() == "old\n"
    assert not geak_candidate_is_adjudicated(state.geak_result, reported, harness_can_replay=False)


def _agentx_round(intvty_p50: float) -> dict:
    """One valid AgentX round of the session's workload, equal on every guard axis."""
    return {
        "output_throughput": 110.0,
        "total_throughput": 1100.0,
        "e2e_norm_intvty_p90": 4.0,
        "e2e_norm_intvty_p50": intvty_p50,
        "duration_seconds": 3600.0,
        "request_error_rate": 0.0,
        "valid_measurement": True,
        "diagnostic_only": False,
        "workload_identity": "traces",
    }


@pytest.mark.asyncio
async def test_source_candidate_whose_accuracy_round_yields_no_score_is_reverted(coordinator, tmp_path, monkeypatch):
    """In a session with a baseline accuracy, clearing the performance bar alone does not keep a source product."""
    _candidate(coordinator, tmp_path)
    state = coordinator.shared_state
    state.baseline_accuracy = 0.9
    state.current_best = {**state.current_best, **_agentx_round(10.0)}
    framework = _framework_tree(tmp_path, monkeypatch)
    slot = tmp_path / "variant_00"
    rounds = []
    results = []

    async def bench(self, bench_leg, **kwargs):
        cleared = {"status": "succeeded", "workspace": str(slot / "benchmark_1"), **_agentx_round(13.0)}
        return cleared, {"accuracy_pass": None}

    async def round_without_a_score(slot, launch, state, *, session):
        rounds.append(slot)
        return AccuracyRoundOutcome(slot, "eval_failed")

    executor = IntegratePatchExecutor(session_dir=coordinator.session_dir)

    async def integrate(ctx):
        results.append(await executor(ctx))
        return results[-1]

    monkeypatch.setattr(IntegratePatchExecutor, "_bench_patch", bench)
    monkeypatch.setattr(_integrate_bench, "run_candidate_accuracy_round", round_without_a_score)
    coordinator.sub.register_executor("integrate_patch", integrate)

    await coordinator.phase_kernel._revalidate_geak_candidate(reason="unit", handoff_cards=None)

    assert rounds == [slot]
    assert results[0]["status"] == "accuracy_unavailable_reject"
    assert (framework / "kernel.py").read_text() == "old\n"
    assert not state.optimization_stack
    assert state.geak_result["revalidation_status"] == "no_promote"


@pytest.mark.asyncio
async def test_cancelled_covered_run_reuses_its_task_identity(coordinator, tmp_path, monkeypatch):
    _candidate(coordinator, tmp_path)
    tasks = []

    async def execute(task):
        tasks.append(task)
        if len(tasks) == 1:
            raise asyncio.CancelledError
        return {RESULT_BENCHED: True, "status": "reverted", "reason": "canonical replay did not improve"}

    async def promote(*args, **kwargs):
        pass

    monkeypatch.setattr(coordinator.sub, "execute_covered", execute)
    monkeypatch.setattr(coordinator.writeback, "promote_to_shared_state", promote)
    with pytest.raises(asyncio.CancelledError):
        await coordinator.phase_kernel._revalidate_geak_candidate(reason="unit", handoff_cards=None)
    saved = dict(coordinator.shared_state.geak_pending["native_review"])
    await coordinator.phase_kernel._revalidate_geak_candidate(reason="resume", handoff_cards=None)
    assert len(tasks) == 2
    assert tasks[0].task_id == tasks[1].task_id == saved["task_id"]
    assert await coordinator.bus.tail(topic="proposal") == []
    assert coordinator.shared_state.geak_pending == {}
    assert not coordinator.shared_state.optimization_stack


@pytest.mark.asyncio
async def test_missing_declared_patch_fails_without_proxy_harness(coordinator, tmp_path, monkeypatch):
    patch = _candidate(coordinator, tmp_path)
    patch.unlink()

    async def forbidden(**kwargs):
        raise AssertionError("native source failure must not use GEAK proxy harness")

    monkeypatch.setattr(coordinator.phase_kernel, "_revalidate_on_geak_harness", forbidden)
    await coordinator.phase_kernel._revalidate_geak_candidate(reason="unit", handoff_cards=None)
    assert coordinator.shared_state.geak_pending == {}
    assert coordinator.shared_state.geak_result["revalidation_status"] == "fallback_failed"
    assert "source patch" in coordinator.shared_state.geak_result["revalidation_error"]


@pytest.mark.parametrize("missing", ["combined", "all"])
def test_source_artifacts_are_required_even_with_accepted_config(coordinator, tmp_path, missing):
    patch = _candidate(coordinator, tmp_path)
    coordinator.shared_state.geak_result.pop("final_patch")
    if missing == "all":
        coordinator.shared_state.geak_result["accepted_kernels"][0].pop("final_patch")
        params = coordinator.writeback.geak_rebench_params(reason="unit")
        assert params["reason"] == "geak_native_source_invalid"
    else:
        params = coordinator.writeback.geak_rebench_params(reason="unit")
        assert Path(params["patches"][0]).read_bytes() == patch.read_bytes()


@pytest.mark.asyncio
async def test_patch_changed_after_staging_is_refused_before_execution(coordinator, tmp_path, monkeypatch):
    _candidate(coordinator, tmp_path)

    async def forbidden(task):
        raise AssertionError("changed source must not execute")

    _changed_after_staging(
        coordinator, monkeypatch, lambda params: Path(params["patches"][0]).write_text("patch contents nobody staged")
    )
    monkeypatch.setattr(coordinator.sub, "execute_covered", forbidden)
    await coordinator.phase_kernel._revalidate_geak_candidate(reason="unit", handoff_cards=None)
    assert coordinator.shared_state.geak_pending == {}
    assert coordinator.shared_state.geak_result["revalidation_status"] == "fallback_failed"
    assert "changed or disappeared" in coordinator.shared_state.geak_result["revalidation_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["sitecustomize", "imported_module", "new_file", "deleted_module", "unchanged"])
async def test_staging_binds_entire_overlay_without_manifest(coordinator, tmp_path, monkeypatch, change):
    _candidate(coordinator, tmp_path)

    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text("import kernel_impl\n")
    module = overlay / "kernel_impl.py"
    module.write_text("VALUE = 1\n")
    coordinator.shared_state.geak_result["final_overlay"] = str(overlay)
    executed = []

    def change_overlay(params):
        assert set(params["native_geak_overlay_files"]) == {str(overlay / "sitecustomize.py"), str(module)}
        if change == "sitecustomize":
            (overlay / "sitecustomize.py").write_text("import unstaged_impl\n")
        elif change == "imported_module":
            module.write_text("VALUE = 2\n")
        elif change == "new_file":
            (overlay / "unstaged_impl.py").write_text("VALUE = 3\n")
        elif change == "deleted_module":
            module.unlink()
        else:
            cache = overlay / "__pycache__"
            cache.mkdir()
            (cache / "kernel_impl.cpython-311.pyc").write_bytes(b"interpreter cache")

    async def execute(task):
        executed.append(task)
        return {RESULT_BENCHED: True, "status": "reverted", "reason": "canonical candidate did not improve"}

    _changed_after_staging(coordinator, monkeypatch, change_overlay)
    monkeypatch.setattr(coordinator.sub, "execute_covered", execute)
    await coordinator.phase_kernel._revalidate_geak_candidate(reason="unit", handoff_cards=None)
    assert coordinator.shared_state.geak_pending == {}
    if change == "unchanged":
        assert len(executed) == 1
    else:
        assert executed == []
        assert coordinator.shared_state.geak_result["revalidation_status"] == "fallback_failed"
        assert "staged GEAK overlay changed" in coordinator.shared_state.geak_result["revalidation_error"]


def test_recovery_retains_revalidation_identity_only_for_same_candidate(coordinator, tmp_path):
    _candidate(coordinator, tmp_path)
    result = coordinator.shared_state.geak_result
    saved = {"artifact_digest": "staged", "candidate": dict(result), "task_id": "covered-task"}
    coordinator.shared_state.geak_pending = {
        "native_review": saved,
        "status": "source_rebench_running",
        "revalidation_task_id": "covered-task",
    }
    coordinator.phase_kernel._record_geak_candidate(result)
    assert coordinator.shared_state.geak_pending["native_review"] == saved
    assert coordinator.shared_state.geak_pending["revalidation_task_id"] == "covered-task"
    changed = {**result, "accepted_config": {"flags": "--different"}}
    coordinator.phase_kernel._record_geak_candidate(changed)
    assert "native_review" not in coordinator.shared_state.geak_pending


def test_adjudicated_source_result_reopens_only_when_artifact_changes(coordinator, tmp_path):
    patch = _candidate(coordinator, tmp_path)
    raw = dict(coordinator.shared_state.geak_result)
    params = coordinator.writeback.geak_rebench_params(reason="unit")
    settled = {
        **raw,
        "revalidation_status": "validated",
        "native_source_revalidation": {"params": params, "status": "validated"},
        "canonical_revalidation": {"output_throughput": 120.0},
    }
    assert geak_candidate_is_adjudicated(settled, raw, harness_can_replay=False)
    patch.write_text(patch.read_text().replace("+new", "+changed"))
    assert not geak_candidate_is_adjudicated(settled, raw, harness_can_replay=False)
