# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK source products require real review and canonical transactional measurement."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.phases.geak_native_revalidation import ORIGIN
from .test_geak_revalidation_dispatch import coordinator as coordinator


def _candidate(c, tmp_path):
    state = c.shared_state
    state.benchmark_mode = "agentx"
    state.agentx_epoch = 3
    state.agentx_backend = "native"
    state.baseline_config_path = str(tmp_path / "native.yaml")
    state.baseline_tput = 100.0
    state.current_best = {"tput": 110.0, "extra_server_args": "--incumbent", "extra_envs": {"SGLANG_USE_AITER": "0"}}
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


async def _actual_verdict(c, verdict):
    async def ready():
        while not c.state.pending_proposals:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(ready(), 2.0)
    pending = next(iter(c.state.pending_proposals.values()))
    assert pending.action_name == "integrate"
    assert pending.payload["params"]["covered_executor"] == "integrate_patch"
    await c._handle_single_verdict(
        source="critic", pending=pending, verdict=verdict, reasoning="reviewed the staged patch"
    )


def test_source_route_retains_configuration_and_stages_exact_patches(coordinator, tmp_path):
    patch = _candidate(coordinator, tmp_path)
    params = coordinator._geak_rebench_params(reason="unit")
    assert params["origin"] == ORIGIN
    assert params["config_path"] == coordinator.shared_state.baseline_config_path
    assert params["extra_server_args"] == "--candidate"
    assert params["extra_envs"] == {"SGLANG_USE_AITER": "1"}
    assert "--incumbent" in params["base_extra_args"]
    assert Path(params["patches"][0]).read_bytes() == patch.read_bytes()
    assert "final_throughput_tok_s" not in params


@pytest.mark.asyncio
@pytest.mark.parametrize("verdict", ["reject", "needs_review"])
async def test_declined_review_never_runs_or_promotes_proxy(coordinator, tmp_path, monkeypatch, verdict):
    _candidate(coordinator, tmp_path)
    before = dict(coordinator.shared_state.current_best)

    async def forbidden(task):
        raise AssertionError("unapproved source integration must not execute")

    monkeypatch.setattr(coordinator.sub, "execute_covered", forbidden)
    run = asyncio.create_task(coordinator._revalidate_geak_candidate(reason="unit"))
    await _actual_verdict(coordinator, verdict)
    await asyncio.wait_for(run, 2.0)
    assert coordinator.shared_state.geak_pending == {}
    assert coordinator.shared_state.current_best == before
    assert coordinator.shared_state.geak_result["revalidation_status"] == "no_promote"
    assert not coordinator.shared_state.optimization_stack


@pytest.mark.asyncio
async def test_approved_source_uses_covered_integrator_and_canonical_measurement(coordinator, tmp_path, monkeypatch):
    _candidate(coordinator, tmp_path)
    executed = []
    adopted = []
    canonical = {"output_throughput": 120.0, "agentx_epoch": 3}

    async def execute(task):
        assert coordinator.shared_state.get_specialist_patch_verdict(task.params["specialist_task_id"]) == "approve"
        executed.append(task)
        return {"status": "kept", "bench_result": canonical}

    async def promote(kind, result, *, task):
        assert kind == "integrate_patch"
        assert result["bench_result"] == canonical
        coordinator.shared_state.optimization_stack.append({"task_id": task.task_id, "action": kind})

    monkeypatch.setattr(coordinator.sub, "execute_covered", execute)
    monkeypatch.setattr(coordinator, "_promote_to_shared_state", promote)
    monkeypatch.setattr(coordinator, "_record_geak_adopted_kernels", lambda *args, **kwargs: adopted.append(kwargs))
    monkeypatch.setattr(
        "hyperloom.orchestrator.state.shared_state.resolve_graded_comparison",
        lambda *args, **kwargs: SimpleNamespace(comparable=True, candidate=120.0, reference=100.0),
    )
    run = asyncio.create_task(coordinator._revalidate_geak_candidate(reason="unit"))
    await _actual_verdict(coordinator, "approve")
    await asyncio.wait_for(run, 2.0)
    assert len(executed) == 1
    assert executed[0].kind == "integrate_patch"
    assert adopted[0]["measured_tput"] == 120.0
    assert adopted[0]["source_applied"] is True
    assert adopted[0]["overlay_loaded"] is False
    assert coordinator.shared_state.geak_result["canonical_revalidation"] == canonical
    assert coordinator.shared_state.geak_pending == {}


@pytest.mark.asyncio
async def test_cancelled_covered_run_reuses_review_and_task_identity(coordinator, tmp_path, monkeypatch):
    _candidate(coordinator, tmp_path)
    tasks = []

    async def execute(task):
        tasks.append(task)
        if len(tasks) == 1:
            raise asyncio.CancelledError()
        return {"status": "reverted", "reason": "canonical replay did not improve"}

    async def promote(*args, **kwargs):
        pass

    monkeypatch.setattr(coordinator.sub, "execute_covered", execute)
    monkeypatch.setattr(coordinator, "_promote_to_shared_state", promote)
    run = asyncio.create_task(coordinator._revalidate_geak_candidate(reason="unit"))
    await _actual_verdict(coordinator, "approve")
    with pytest.raises(asyncio.CancelledError):
        await run
    saved_review = dict(coordinator.shared_state.geak_pending["native_review"])
    await coordinator._revalidate_geak_candidate(reason="resume")
    assert len(tasks) == 2
    assert tasks[0].task_id == tasks[1].task_id == saved_review["task_id"]
    assert len(coordinator.state.pending_proposals) == 1
    assert coordinator.shared_state.geak_pending == {}
    assert not coordinator.shared_state.optimization_stack


@pytest.mark.asyncio
async def test_missing_declared_patch_fails_without_proxy_harness(coordinator, tmp_path, monkeypatch):
    patch = _candidate(coordinator, tmp_path)
    patch.unlink()

    async def forbidden(**kwargs):
        raise AssertionError("native source failure must not use GEAK proxy harness")

    monkeypatch.setattr(coordinator, "_revalidate_on_geak_harness", forbidden)
    await coordinator._revalidate_geak_candidate(reason="unit")
    assert coordinator.shared_state.geak_pending == {}
    assert coordinator.shared_state.geak_result["revalidation_status"] == "fallback_failed"
    assert "source patch" in coordinator.shared_state.geak_result["revalidation_error"]


@pytest.mark.parametrize("missing", ["combined", "all"])
def test_source_artifacts_are_required_even_with_accepted_config(coordinator, tmp_path, missing):
    patch = _candidate(coordinator, tmp_path)
    coordinator.shared_state.geak_result.pop("final_patch")
    if missing == "all":
        coordinator.shared_state.geak_result["accepted_kernels"][0].pop("final_patch")
        params = coordinator._geak_rebench_params(reason="unit")
        assert params["reason"] == "geak_native_source_invalid"
    else:
        params = coordinator._geak_rebench_params(reason="unit")
        assert Path(params["patches"][0]).read_bytes() == patch.read_bytes()


@pytest.mark.asyncio
async def test_reviewed_patch_mutation_is_rejected_before_execution(coordinator, tmp_path, monkeypatch):
    _candidate(coordinator, tmp_path)
    from hyperloom.orchestrator.phases import geak_native_revalidation as native

    review = native._review

    async def mutate_after_review(c, params):
        outcome = await review(c, params)
        Path(params["patches"][0]).write_text("unreviewed patch contents")
        return outcome

    async def forbidden(task):
        raise AssertionError("changed source must not execute")

    monkeypatch.setattr(native, "_review", mutate_after_review)
    monkeypatch.setattr(coordinator.sub, "execute_covered", forbidden)
    run = asyncio.create_task(coordinator._revalidate_geak_candidate(reason="unit"))
    await _actual_verdict(coordinator, "approve")
    await asyncio.wait_for(run, 2.0)
    assert coordinator.shared_state.geak_pending == {}
    assert coordinator.shared_state.geak_result["revalidation_status"] == "fallback_failed"
    assert "changed or disappeared" in coordinator.shared_state.geak_result["revalidation_error"]


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["sitecustomize", "imported_module", "new_file", "deleted_module", "unchanged"])
async def test_source_review_binds_entire_overlay_without_manifest(coordinator, tmp_path, monkeypatch, change):
    _candidate(coordinator, tmp_path)
    from hyperloom.orchestrator.phases import geak_native_revalidation as native

    overlay = tmp_path / "overlay"
    overlay.mkdir()
    (overlay / "sitecustomize.py").write_text("import kernel_impl\n")
    module = overlay / "kernel_impl.py"
    module.write_text("VALUE = 1\n")
    coordinator.shared_state.geak_result["final_overlay"] = str(overlay)
    executed = []
    review = native._review

    async def mutate_after_review(c, params):
        outcome = await review(c, params)
        assert set(params["native_geak_overlay_files"]) == {str(overlay / "sitecustomize.py"), str(module)}
        if change == "sitecustomize":
            (overlay / "sitecustomize.py").write_text("import unreviewed_impl\n")
        elif change == "imported_module":
            module.write_text("VALUE = 2\n")
        elif change == "new_file":
            (overlay / "unreviewed_impl.py").write_text("VALUE = 3\n")
        elif change == "deleted_module":
            module.unlink()
        else:
            cache = overlay / "__pycache__"
            cache.mkdir()
            (cache / "kernel_impl.cpython-311.pyc").write_bytes(b"interpreter cache")
        return outcome

    async def execute(task):
        executed.append(task)
        return {"status": "reverted", "reason": "canonical candidate did not improve"}

    monkeypatch.setattr(native, "_review", mutate_after_review)
    monkeypatch.setattr(coordinator.sub, "execute_covered", execute)
    run = asyncio.create_task(coordinator._revalidate_geak_candidate(reason="unit"))
    await _actual_verdict(coordinator, "approve")
    await asyncio.wait_for(run, 2.0)
    assert coordinator.shared_state.geak_pending == {}
    if change == "unchanged":
        assert len(executed) == 1
    else:
        assert executed == []
        assert coordinator.shared_state.geak_result["revalidation_status"] == "fallback_failed"
        assert "reviewed GEAK overlay changed" in coordinator.shared_state.geak_result["revalidation_error"]


@pytest.mark.asyncio
async def test_expired_review_budget_clears_pending_without_promotion(coordinator, tmp_path, monkeypatch):
    _candidate(coordinator, tmp_path)
    monkeypatch.setattr(coordinator.shared_state, "remaining_minutes", lambda: 0.0)
    await coordinator._revalidate_geak_candidate(reason="unit")
    assert coordinator.shared_state.geak_pending == {}
    assert coordinator.shared_state.geak_result["native_source_revalidation"]["verdict"] == "timeout"
    assert not coordinator.shared_state.optimization_stack


def test_native_review_recovery_retains_identity_only_for_same_candidate(coordinator, tmp_path):
    _candidate(coordinator, tmp_path)
    result = coordinator.shared_state.geak_result
    saved = {"artifact_digest": "reviewed", "candidate": dict(result), "verdict": "approve", "task_id": "covered-task"}
    coordinator.shared_state.geak_pending = {
        "native_review": saved,
        "status": "source_rebench_running",
        "revalidation_task_id": "covered-task",
    }
    coordinator._record_geak_candidate(result)
    assert coordinator.shared_state.geak_pending["native_review"] == saved
    assert coordinator.shared_state.geak_pending["revalidation_task_id"] == "covered-task"
    changed = {**result, "accepted_config": {"flags": "--different"}}
    coordinator._record_geak_candidate(changed)
    assert "native_review" not in coordinator.shared_state.geak_pending


def test_adjudicated_source_result_reopens_only_when_artifact_changes(coordinator, tmp_path):
    from hyperloom.orchestrator.phases.geak_rebench import geak_candidate_is_adjudicated

    patch = _candidate(coordinator, tmp_path)
    raw = dict(coordinator.shared_state.geak_result)
    params = coordinator._geak_rebench_params(reason="unit")
    settled = {
        **raw,
        "revalidation_status": "validated",
        "native_source_revalidation": {"params": params, "status": "validated"},
        "canonical_revalidation": {"output_throughput": 120.0},
    }
    assert geak_candidate_is_adjudicated(settled, raw, harness_can_replay=False)
    patch.write_text(patch.read_text().replace("+new", "+changed"))
    assert not geak_candidate_is_adjudicated(settled, raw, harness_can_replay=False)
