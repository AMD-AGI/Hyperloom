# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GEAK source products are staged, then kept or reverted by a canonical transactional measurement."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.phases.geak_native_revalidation import ORIGIN, source_rebench_params
from .test_geak_revalidation_dispatch import coordinator as coordinator


def _candidate(c, tmp_path):
    state = c.shared_state
    state.benchmark_mode = "agentx"
    state.agentx_backend = "inferencex"
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


@pytest.mark.asyncio
async def test_unkept_measurement_settles_no_promote_without_promotion(coordinator, tmp_path):
    _candidate(coordinator, tmp_path)
    before = dict(coordinator.shared_state.current_best)

    async def integrate(ctx):
        return {"status": "reverted", "reason": "canonical replay did not improve"}

    coordinator.sub.register_executor("integrate_patch", integrate)
    await coordinator.phase_kernel._revalidate_geak_candidate(reason="unit", handoff_cards=None)
    assert coordinator.shared_state.geak_pending == {}
    assert coordinator.shared_state.current_best == before
    assert coordinator.shared_state.geak_result["revalidation_status"] == "no_promote"
    assert not coordinator.shared_state.optimization_stack


@pytest.mark.asyncio
async def test_cancelled_covered_run_reuses_its_task_identity(coordinator, tmp_path, monkeypatch):
    _candidate(coordinator, tmp_path)
    tasks = []

    async def execute(task):
        tasks.append(task)
        if len(tasks) == 1:
            raise asyncio.CancelledError
        return {"status": "reverted", "reason": "canonical replay did not improve"}

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
        return {"status": "reverted", "reason": "canonical candidate did not improve"}

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
    from hyperloom.orchestrator.phases.geak_rebench import geak_candidate_is_adjudicated

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
