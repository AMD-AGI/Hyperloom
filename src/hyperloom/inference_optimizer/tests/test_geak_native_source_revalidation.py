# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK source products are staged, then kept or reverted by a canonical transactional measurement."""

from __future__ import annotations

import asyncio
import copy
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common.perf_metric import GRADED_OUTPUT
from hyperloom.inference_optimizer.session.geak_rebench_verdict import REBENCH_UNMEASURED_GIVEN_UP
from hyperloom.orchestrator.actions.executors.benchmark_run.accuracy import AccuracyRoundOutcome
from hyperloom.orchestrator.actions.executors.integrate import _integrate_bench
from hyperloom.orchestrator.actions.executors.integrate._integrate_vocabulary import RESULT_GRADED
from hyperloom.orchestrator.actions.executors.integrate_patch import IntegratePatchExecutor
from hyperloom.orchestrator.actions.executors.workload._workload_envs import FrameworkScriptMismatchError
from hyperloom.orchestrator.actions.stop_attribution import SESSION_TIME_EXHAUSTED_CLASS, STOPPED_BY_THE_RUN
from hyperloom.orchestrator.phases.geak_native_revalidation import ORIGIN, source_rebench_params
from hyperloom.orchestrator.phases.geak_rebench import REVALIDATION_UNMEASURED, geak_candidate_is_adjudicated
from hyperloom.orchestrator.phases.kernel_geak_promotion import GeakRunnerFacts
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
    assert coordinator.shared_state.geak_result["revalidation_status"] == "validated"
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
    measured = {RESULT_GRADED: True, "status": "reverted", "reason": "canonical replay did not improve"}
    reported = await _settle(coordinator, tmp_path, measured)
    state = coordinator.shared_state
    assert state.geak_pending == {}
    assert state.current_best == _INCUMBENT
    assert state.geak_result["revalidation_status"] == "no_promote"
    assert state.geak_result["revalidation_error"] == "canonical replay did not improve"
    assert not state.optimization_stack
    assert geak_candidate_is_adjudicated(state.geak_result, reported)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("result", "error"),
    [
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
    assert not geak_candidate_is_adjudicated(state.geak_result, reported)


def _framework_tree(tmp_path, monkeypatch):
    """A git framework tree the staged patch applies to, registered where the integration looks for one."""
    repo = tmp_path / "framework"
    init_git_repo(repo, seed_file="kernel.py", seed_text="old\n")
    patch_integrate_patch_roots(monkeypatch, tmp_path)
    return repo


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


async def _measure_with_bench(coordinator, tmp_path, monkeypatch, bench):
    """Measure the candidate through the real integration, with ``bench`` as its bench leg; the product as reported."""
    _candidate(coordinator, tmp_path)
    state = coordinator.shared_state
    state.current_best = {**state.current_best, **_agentx_round(10.0)}
    monkeypatch.setattr(IntegratePatchExecutor, "_bench_patch", bench)
    coordinator.sub.register_executor("integrate_patch", IntegratePatchExecutor(session_dir=coordinator.session_dir))
    reported = dict(state.geak_result)
    await coordinator.phase_kernel._revalidate_geak_candidate(reason="unit", handoff_cards=None)
    return reported


def _bench_raising(error):
    async def bench(self, bench_leg, **kwargs):
        raise error

    return bench


def _bench_returning(round_result):
    async def bench(self, bench_leg, **kwargs):
        return dict(round_result), {}

    return bench


def _failed_round(error_class, error=""):
    return _bench_returning({"status": "failed", "error_class": error_class, "error": error})


_AXES_MISSING = "performance comparison unavailable: candidate_axes_missing"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("bench", "error"),
    [
        pytest.param(_failed_round(SESSION_TIME_EXHAUSTED_CLASS), _BUDGET_STOP.interrupted, id="session_budget_stop"),
        pytest.param(_failed_round("port_in_use", "port 8000 is in use"), "port 8000 is in use", id="environment"),
        pytest.param(
            _bench_raising(OSError(28, "No space left on device")),
            "OSError(28, 'No space left on device')",
            id="bench_raised",
        ),
        pytest.param(
            _bench_raising(FrameworkScriptMismatchError("benchmark_script targets another framework")),
            "benchmark_script targets another framework",
            id="launch_script_mismatch",
        ),
        pytest.param(_failed_round("client_failed", "benchmark client exited 1"), _AXES_MISSING, id="client_failed"),
        pytest.param(_failed_round("server_died", "Segmentation fault"), _AXES_MISSING, id="server_died"),
        pytest.param(_failed_round("a_class_nobody_named"), _AXES_MISSING, id="unknown_error_class"),
        pytest.param(
            _bench_returning({"status": "succeeded", "output_throughput": 130.0, "completed_requests": 10}),
            _AXES_MISSING,
            id="round_without_the_graded_axes",
        ),
    ],
)
async def test_bench_that_graded_nothing_settles_failed_and_stays_open(
    coordinator, tmp_path, monkeypatch, bench, error
):
    repo = _framework_tree(tmp_path, monkeypatch)
    reported = await _measure_with_bench(coordinator, tmp_path, monkeypatch, bench)
    state = coordinator.shared_state
    assert state.geak_result["revalidation_status"] == "failed"
    assert state.geak_result["revalidation_error"] == error
    assert state.current_best["tput"] == _INCUMBENT["tput"]
    assert not state.optimization_stack
    assert (repo / "kernel.py").read_text() == "old\n"
    assert not geak_candidate_is_adjudicated(state.geak_result, reported)


async def _next_kernel_entry(coordinator, reported) -> bool:
    """What the next KERNEL entry does with the product GEAK left on disk; whether it measured it again."""
    geak_dir = coordinator.session_dir / "geak"
    geak_dir.mkdir(exist_ok=True)
    (geak_dir / "result.json").write_text(json.dumps(reported), encoding="utf-8")
    coordinator.phase_kernel._record_geak_kernel_journey = lambda _result: None
    return await coordinator.phase_kernel._promote_crashed_geak_result({}, cards=None)


def _given_up(state) -> bool:
    return state.geak_result.get("revalidation_error_class") == REBENCH_UNMEASURED_GIVEN_UP


class _KernelTimeline:
    """Keeps the rebench conclusions a KERNEL entry records; every other record is accepted and dropped."""

    def __init__(self):
        self.conclusions = []

    def record_geak_rebench_conclusion(self, **conclusion):
        self.conclusions.append(conclusion)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@pytest.mark.asyncio
async def test_candidate_that_keeps_ending_unmeasured_is_given_up_and_not_measured_again(
    coordinator, tmp_path, monkeypatch
):
    _framework_tree(tmp_path, monkeypatch)
    benches = []

    async def bench(self, bench_leg, **kwargs):
        benches.append(bench_leg)
        raise OSError(28, "No space left on device")

    timeline = _KernelTimeline()
    monkeypatch.setattr(coordinator.phase_kernel, "timeline", lambda: timeline)
    reported = await _measure_with_bench(coordinator, tmp_path, monkeypatch, bench)
    state = coordinator.shared_state
    assert not _given_up(state)
    assert timeline.conclusions == []
    assert "dropped unvalidated (rebench_failed)" in state.to_mission_summary()
    assert await _next_kernel_entry(coordinator, reported), "an open candidate is measured again"
    assert state.geak_result["revalidation_status"] == "failed"
    assert state.geak_result[REVALIDATION_UNMEASURED] == 2
    assert _given_up(state)
    assert "dropped unvalidated (rebench_failed, given up unmeasured)" in state.to_mission_summary()
    assert [(c["final_status"], c["final_error_class"]) for c in timeline.conclusions] == [
        ("failed", REBENCH_UNMEASURED_GIVEN_UP)
    ]
    assert "No space left on device" in timeline.conclusions[0]["final_error"]
    assert geak_candidate_is_adjudicated(state.geak_result, reported)
    assert not await _next_kernel_entry(coordinator, reported), "a candidate given up is left alone"
    assert len(benches) == 2
    assert state.current_best["tput"] == _INCUMBENT["tput"]


@pytest.mark.asyncio
async def test_verdict_on_a_candidate_that_once_ended_unmeasured_settles_it(coordinator, tmp_path, monkeypatch):
    _framework_tree(tmp_path, monkeypatch)
    out_of_space = _bench_raising(OSError(28, "No space left on device"))
    reported = await _measure_with_bench(coordinator, tmp_path, monkeypatch, out_of_space)
    state = coordinator.shared_state
    below_the_bar = _bench_returning({"status": "succeeded", **_agentx_round(10.0)})
    monkeypatch.setattr(IntegratePatchExecutor, "_bench_patch", below_the_bar)
    assert await _next_kernel_entry(coordinator, reported)
    assert state.geak_result["revalidation_status"] == "no_promote"
    assert not _given_up(state)
    assert geak_candidate_is_adjudicated(state.geak_result, reported)


@pytest.mark.asyncio
async def test_unmeasured_count_of_one_product_is_not_held_against_the_next(coordinator, tmp_path, monkeypatch):
    _framework_tree(tmp_path, monkeypatch)
    out_of_space = _bench_raising(OSError(28, "No space left on device"))
    reported = await _measure_with_bench(coordinator, tmp_path, monkeypatch, out_of_space)
    state = coordinator.shared_state
    assert state.geak_result[REVALIDATION_UNMEASURED] == 1
    another_run = {**reported, "final_throughput_tok_s": reported["final_throughput_tok_s"] + 1.0}
    assert await _next_kernel_entry(coordinator, another_run)
    assert state.geak_result["revalidation_status"] == "failed"
    assert state.geak_result[REVALIDATION_UNMEASURED] == 1
    assert not _given_up(state)


@pytest.mark.asyncio
async def test_unmeasured_count_follows_a_product_the_geak_runner_hands_back(coordinator, tmp_path, monkeypatch):
    _framework_tree(tmp_path, monkeypatch)
    out_of_space = _bench_raising(OSError(28, "No space left on device"))
    reported = await _measure_with_bench(coordinator, tmp_path, monkeypatch, out_of_space)
    result_path = coordinator.session_dir / "geak" / "result.json"
    result_path.parent.mkdir(exist_ok=True)

    def handed_back(product):
        result_path.write_text(json.dumps(product), encoding="utf-8")
        exited = subprocess.CompletedProcess(args=["geak"], returncode=1, stdout="", stderr="")
        return coordinator.phase_kernel._settle_geak_runner_exit(exited, handoff={}, facts=GeakRunnerFacts())

    assert handed_back(reported)[REVALIDATION_UNMEASURED] == 1
    another_product = {**reported, "final_throughput_tok_s": reported["final_throughput_tok_s"] + 1.0}
    assert REVALIDATION_UNMEASURED not in handed_back(another_product)


@pytest.mark.asyncio
async def test_failed_round_in_a_session_graded_on_output_throughput_stays_open(coordinator, tmp_path, monkeypatch):
    """The output axis reads a round that measured nothing as a comparable loss; it is still no verdict."""
    repo = _framework_tree(tmp_path, monkeypatch)
    coordinator.shared_state.grading = {"objective": GRADED_OUTPUT}
    reported = await _measure_with_bench(coordinator, tmp_path, monkeypatch, _failed_round("server_died"))
    state = coordinator.shared_state
    assert state.geak_result["revalidation_status"] == "failed"
    assert (repo / "kernel.py").read_text() == "old\n"
    assert not geak_candidate_is_adjudicated(state.geak_result, reported)


@pytest.mark.asyncio
async def test_graded_bench_below_the_keep_bar_settles_no_promote(coordinator, tmp_path, monkeypatch):
    repo = _framework_tree(tmp_path, monkeypatch)
    below_the_bar = _bench_returning({"status": "succeeded", **_agentx_round(10.0)})
    reported = await _measure_with_bench(coordinator, tmp_path, monkeypatch, below_the_bar)
    state = coordinator.shared_state
    assert state.geak_result["revalidation_status"] == "no_promote"
    assert "did not clear the interactivity KEEP gate" in state.geak_result["revalidation_error"]
    assert not state.optimization_stack
    assert (repo / "kernel.py").read_text() == "old\n"
    assert geak_candidate_is_adjudicated(state.geak_result, reported)


@pytest.mark.asyncio
async def test_cleared_candidate_whose_keep_cannot_be_committed_stays_open(coordinator, tmp_path, monkeypatch):
    repo = _framework_tree(tmp_path, monkeypatch)
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.parent.mkdir(exist_ok=True)
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    slot = tmp_path / "variant_00"

    async def round_without_a_score(slot, launch, state, *, session):
        return AccuracyRoundOutcome(slot, "eval_failed")

    monkeypatch.setattr(_integrate_bench, "run_candidate_accuracy_round", round_without_a_score)
    cleared = _bench_returning({"status": "succeeded", "workspace": str(slot / "benchmark_1"), **_agentx_round(13.0)})
    reported = await _measure_with_bench(coordinator, tmp_path, monkeypatch, cleared)
    state = coordinator.shared_state
    assert state.geak_result["revalidation_status"] == "failed"
    assert state.geak_result["revalidation_error"] == "git commit failed"
    assert not state.optimization_stack
    assert (repo / "kernel.py").read_text() == "old\n"
    assert not geak_candidate_is_adjudicated(state.geak_result, reported)


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
        return {RESULT_GRADED: True, "status": "reverted", "reason": "canonical replay did not improve"}

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
async def test_missing_declared_patch_settles_unmeasured(coordinator, tmp_path):
    patch = _candidate(coordinator, tmp_path)
    patch.unlink()
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
        return {RESULT_GRADED: True, "status": "reverted", "reason": "canonical candidate did not improve"}

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
    assert geak_candidate_is_adjudicated(settled, raw)
    patch.write_text(patch.read_text().replace("+new", "+changed"))
    assert not geak_candidate_is_adjudicated(settled, raw)
