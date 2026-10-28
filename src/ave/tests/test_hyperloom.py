# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Explicit Ave campaigns run through Hyperloom's kernel phase."""

from __future__ import annotations

import asyncio
from concurrent.futures import CancelledError
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from ave.optimizer import AveLoop
from ave.preparation import PreparedKernel
from ave.task import DEFAULT_PROMPT, PromptStep, read_env_prompts
from ave.tests.test_loop import FakeClaude, session_result
from hyperloom_common.git_safety import repo_git
from hyperloom.common.visible_devices import VISIBLE_DEVICE_VARS
from hyperloom.orchestrator.phases import machine_state
from hyperloom.orchestrator.actions.cancel_channel import CancelScope, use_cancel_scope
from hyperloom.orchestrator.actions.executors._server_lifecycle import GpuHandoff
from hyperloom.orchestrator.phases.machine import Transition
from hyperloom.inference_optimizer.cli.bootstrap import LaunchPlatform, _seed_shared_state
from hyperloom.inference_optimizer.cli.executors import _register_executors
from hyperloom.inference_optimizer.cli.parser import _build_parser
from hyperloom.inference_optimizer.breakdown.exporter import build
from hyperloom.inference_optimizer.breakdown.reporters._renderers.capability_summary import capability_rows
from hyperloom.inference_optimizer.breakdown.reporters._renderers.kernel_lifecycle import render as render_kernels
from hyperloom.inference_optimizer.breakdown.reporters.cross_section import build_global_facts
from hyperloom.inference_optimizer.breakdown.workflow_contract import workflow_schema
from hyperloom.inference_optimizer.session.manifest import write_manifest
from hyperloom.inference_optimizer.tests._dispatch_helpers import pump_until_settled
from hyperloom.inference_optimizer.tests.test_kernel_agent_task import coord as coord
from hyperloom.inference_optimizer.tests.test_kernel_integrate_and_report import _fake_workspace, _write_baseline_yaml
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "decision, unfinished_exit, custom_prompts",
    [("KEEP", None, False), ("REVERT", -9, True), ("KEEP", 1, True), ("KEEP", 0, True), ("CANCELLED", None, True)],
)
async def test_hyperloom_cli_selects_ave_and_integrates(
    prepared_run,
    coord,
    monkeypatch,
    decision,
    unfinished_exit,
    custom_prompts,
):
    session = coord.session_dir
    task_dir = prepared_run.result_json.parent / "prepared"
    prompts_file = task_dir / "prompts.json"
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", " AvE " if custom_prompts else "ave")
    monkeypatch.delenv("AVE_PROMPTS_FILE", raising=False)
    if custom_prompts:
        monkeypatch.setenv("AVE_PROMPTS_FILE", str(prompts_file))
    args = _build_parser().parse_args(["optimize", "--model", str(task_dir)])
    monkeypatch.setattr("hyperloom.inference_optimizer.cli.bootstrap.resolve_framework_version", lambda args: "test")
    state = _seed_shared_state(
        session, args, session_id=session.name, framework="sglang", platform=LaunchPlatform({}, {})
    )
    prompts_file.unlink()
    state.phase = "KERNEL_AGENT"
    state.max_minutes = 0
    state.phase_budget_pct = {}
    state.baseline_tput = 800
    state.baseline_config_path = str(session / "benchmark.yaml")
    state.framework_repo_path = str(prepared_run.workspace)
    state.save(session)
    coord.shared_state = state
    _register_executors(coord, session_dir=session)
    _write_baseline_yaml(Path(state.baseline_config_path))
    assert state.kernel_optimizer == "ave"
    assert not coord.phase_kernel.geak_enabled()
    state = SharedState.load_or_init(session)
    coord.shared_state = state
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    assert [step["id"] for step in state.ave_prompts] == (
        ["prepare_layout", "register_tiling"] if custom_prompts else ["optimize"]
    )
    monkeypatch.setattr(coord.phase_kernel, "_maybe_reprofile_for_kernel", AsyncMock())
    monkeypatch.setenv("CLAUDE_CODE_OAUTH_TOKEN", "test-subscription-token")
    monkeypatch.setenv("MAGPIE_PYTHON", "/usr/bin/python3")
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._benchmark_interpreter._resolve_magpie_python",
        lambda _env=None: "/usr/bin/python3",
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.kernel.apply_kernel_patch._clear_python_kernel_caches",
        lambda target: {"status": "ok", "removed": []},
    )
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._aiter_jit.prepare_serving_so_for_csvs",
        lambda envs: {"status": "skipped"},
    )
    free_gpus = Mock(return_value=GpuHandoff(reaped=[], busy=[], holders=[], unverified=""))
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._server_lifecycle.free_gpus_for_handoff",
        free_gpus,
    )
    recorded_keep = AsyncMock()
    monkeypatch.setattr(coord.writeback, "record_integrate_keep", recorded_keep)
    runs = []

    async def select_kernel(*, prompt, options):
        free_gpus.assert_not_called()
        context = json.loads(prompt)
        assert context["steps"] == state.ave_prompts
        assert str(task_dir) in context["workload"]
        assert str(prepared_run.workspace) in context["serving"]
        assert set(options.tools) == {"Read", "Glob", "Grep"}
        assert options.env["CLAUDE_CODE_OAUTH_TOKEN"] == "test-subscription-token"
        assert options.setting_sources == []
        yield session_result(
            "Selection:\n```json\n"
            + json.dumps(
                {
                    "kernel": {
                        "repo_root": str(prepared_run.workspace),
                        "kernel_path": "kernel.py",
                        "source_files": ["kernel.py"],
                        "operator_name": "kernel",
                        "driver_source": prepared_run.driver.read_text(),
                    },
                    "reason": "Selected from the serving workload and trace evidence.",
                }
            )
            + "\n```",
            is_error=False,
        )

    monkeypatch.setattr("claude_agent_sdk.query", select_kernel)
    source = prepared_run.workspace / "kernel.py"
    source.write_text(source.read_text() + "# Existing serving patch\n")
    repo_git(prepared_run.workspace, "add", "kernel.py")
    original_index = repo_git(prepared_run.workspace, "diff", "--no-ext-diff", "--cached").stdout

    async def run_loop(run, *, devices):
        runs.append(run)
        assert run.workspace != prepared_run.workspace
        assert run.base_commit != prepared_run.base_commit
        assert repo_git(prepared_run.workspace, "rev-parse", "HEAD").stdout.strip() == prepared_run.base_commit
        assert repo_git(prepared_run.workspace, "diff", "--no-ext-diff", "--cached").stdout == original_index
        assert "Existing serving patch" in (run.workspace / "kernel.py").read_text()
        assert [step.prompt for step in run.steps] == [step["prompt"] for step in state.ave_prompts]
        assert all(step.files == ("kernel.py",) for step in run.steps)
        proposals = ["VALUE = 7\nLATENCY = 3.0\n"] if custom_prompts else []
        claude = FakeClaude([*proposals, "VALUE = 7\nLATENCY = 1.0\n"])
        monkeypatch.setattr("claude_agent_sdk.query", claude.query)
        result = await AveLoop(run).optimize()
        if unfinished_exit is not None:
            result["status"] = "running"
            run.result_json.write_text(json.dumps(result))
        if decision == "CANCELLED":
            raise CancelledError("Hyperloom stopped the worker")
        return unfinished_exit if unfinished_exit is not None else 0

    monkeypatch.setattr(coord.phase_kernel, "_run_ave_subprocess", run_loop)

    def benchmark(command, **kwargs):
        assert "LATENCY = 1.0" in (prepared_run.workspace / "kernel.py").read_text()
        slot = Path(command[command.index("--output-dir") + 1])
        _fake_workspace(slot, tput=1000 if decision == "KEEP" else 700, accuracy=1.0)
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors.baseline.run_with_session_kill", benchmark)
    await coord.phase_kernel.on_enter_kernel(
        Transition(
            from_phase="FRAMEWORK_AGENT",
            to_phase="KERNEL_AGENT",
            reason="test",
            evidence={},
        )
    )
    queued = [task for task in await coord.tasks.queued() if task.kind == "kernel_agent"]
    assert len(queued) == 1
    assert set(queued[0].requires_lanes) == {"server_lifecycle", "workspace_mutation", "benchmark_lane"}
    await pump_until_settled(coord.dispatcher, timeout=30)
    if decision == "CANCELLED":
        assert (await coord.tasks.get(queued[0].task_id)).state == "cancelled"
        assert coord.shared_state.ave_result == {}
        assert not runs[0].workspace.exists()
        assert json.loads(runs[0].result_json.read_text())["status"] == "cancelled"
        assert (runs[0].result_json.parent / "candidate.patch").is_file()
        assert (runs[0].result_json.parent / "snapshot/kernel.py").read_text() == "VALUE = 7\nLATENCY = 1.0\n"
        assert "LATENCY = 2.0" in (prepared_run.workspace / "kernel.py").read_text()
        assert not any((await coord.locks.lane_holders()).values())
        recorded_keep.assert_not_called()
        return
    result = state.ave_result
    assert result["status"] == (
        "completed" if unfinished_exit is None else "timed_out" if unfinished_exit == -9 else "failed"
    )
    assert json.loads(runs[0].result_json.read_text())["status"] == result["status"]
    assert state.kernel_optimizer == "ave"
    free_gpus.assert_called_once()
    assert "integration" in result, result
    assert result["integration"]["decision"] == decision, result
    assert recorded_keep.await_count == (1 if decision == "KEEP" else 0)
    assert len(runs) == 1
    assert not runs[0].workspace.exists()
    assert Path(result["integrate_params"]["patch_path"]).is_file()
    expected = "1.0" if decision == "KEEP" else "2.0"
    assert f"LATENCY = {expected}" in (prepared_run.workspace / "kernel.py").read_text()
    assert not machine_state.kernel_work_pending(state)
    assert SharedState.load_or_init(session).ave_result == result
    write_manifest(session, session_id=session.name)
    breakdown = build(session)
    from jsonschema import validate

    validate(instance=breakdown, schema=workflow_schema())
    kernel = next(event for event in breakdown["timeline"] if event["type"] == "kernel")
    assert kernel["ext"]["entry"]["route"] == "ave"
    assert kernel["ext"]["attempts"][0]["route"] == "ave"
    assert kernel["ext"]["attempts"][0]["backend"] == "ave"
    assert kernel["ext"]["attempts"][0]["source_kind"] == "ave_kernel"
    assert kernel["ext"]["attempts"][0]["outcome"] == ("adopted" if decision == "KEEP" else "rejected")
    assert kernel["ext"]["attempts"][0]["settled_by"] == "integrate"
    assert kernel["ext"]["attempts"][0]["gain_pct"] == (25.0 if decision == "KEEP" else -12.5)
    assert kernel["ext"]["attempts"][0]["gain_axis"] == "output_throughput"
    assert kernel["ext"]["forge"] is None
    capabilities = capability_rows(breakdown)
    assert capabilities["ave"]["attempts"] == 1
    assert capabilities["ave"]["keeps"] == (1 if decision == "KEEP" else 0)
    assert capabilities["forge"]["attempts"] == 0
    rendered = render_kernels(breakdown)
    assert "Ave touched 1" in rendered.key_facts[0]
    assert "| Ave |" in rendered.markdown_block
    assert f"| ave | {'kept' if decision == 'KEEP' else 'reverted'} |" in rendered.markdown_block
    facts = build_global_facts(breakdown, [rendered])
    assert facts.kernel_pipeline_funnel["attempted"] == 1
    assert facts.kernel_pipeline_funnel["adopted"] == (1 if decision == "KEEP" else 0)
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER")
    await coord.phase_kernel.run_agent(SimpleNamespace(task=SimpleNamespace(params={})))
    assert len(runs) == 1
    coord.phase_kernel._maybe_reprofile_for_kernel.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("gpu_state", ["idle", "busy", "hidden"])
@pytest.mark.parametrize("pin_source", ["process", "recipe"])
async def test_ave_checks_physical_gpus_before_optimizing(prepared_run, coord, monkeypatch, gpu_state, pin_source):
    for name in VISIBLE_DEVICE_VARS:
        monkeypatch.delenv(name, raising=False)
    masks = {"ROCR_VISIBLE_DEVICES": "4,5,6,7", "HIP_VISIBLE_DEVICES": "" if gpu_state == "hidden" else "2,3"}
    if pin_source == "process":
        for name, value in masks.items():
            monkeypatch.setenv(name, value)
    monkeypatch.setenv("TP", "1")
    state = coord.shared_state
    state.kernel_optimizer = "ave"
    state.ave_prompts = [{"id": "optimize", "prompt": DEFAULT_PROMPT}]
    recipe = coord.session_dir / "benchmark.yaml"
    recipe.write_text(json.dumps({"benchmark": {"envs": {"TP": 2, **masks}}}))
    state.baseline_config_path = str(recipe)
    kernel = PreparedKernel(
        prepared_run.repo_root,
        prepared_run.kernel_path,
        prepared_run.source_files,
        prepared_run.operator_name,
        prepared_run.driver.read_text(),
        "Selected from workload evidence",
    )
    monkeypatch.setattr(coord.phase_kernel, "_prepare_ave_kernel", AsyncMock(return_value=kernel))
    monkeypatch.setattr(coord.phase_kernel, "_kernel_rewrite_controller_timeouts", lambda: (75, 105))
    free_gpus = Mock(
        return_value=GpuHandoff(
            reaped=[], busy=[(7, 281000.0, 288000.0)] if gpu_state == "busy" else [], holders=[], unverified=""
        )
    )
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._server_lifecycle.free_gpus_for_handoff", free_gpus)
    launch = Mock(return_value=subprocess.CompletedProcess([], 0, "", ""))
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._subprocess_kill.run_with_session_kill", launch)

    async def optimize(_kernel, *, run_worker, **kwargs):
        await run_worker(prepared_run)
        return {"status": "ok"}

    optimize_mock = AsyncMock(side_effect=optimize)
    monkeypatch.setattr("ave.optimizer.optimize", optimize_mock)
    ctx = SimpleNamespace(task=SimpleNamespace(params={}))

    result = await coord.phase_kernel.run_agent(ctx)

    if gpu_state == "hidden":
        free_gpus.assert_not_called()
    else:
        free_gpus.assert_called_once_with(coord.session_dir, [6, 7], timeout_sec=75)
    assert optimize_mock.await_count == launch.call_count == (1 if gpu_state == "idle" else 0)
    assert result["result"]["status"] == ("ok" if gpu_state == "idle" else "failed")
    if gpu_state == "busy":
        assert result["result"]["error_class"] == "gpus_busy_at_handoff"
        assert "GPU7 281000/288000 MiB" in result["result"]["reason"]
    elif gpu_state == "hidden":
        assert "GPU visibility masks expose no devices" in result["result"]["reason"]
    else:
        env = launch.call_args.kwargs["env"]
        assert env["ROCR_VISIBLE_DEVICES"] == "4,5,6,7"
        assert env["HIP_VISIBLE_DEVICES"] == env["CUDA_VISIBLE_DEVICES"] == "2,3"
        assert env["TP"] == "2"
    assert SharedState.load_or_init(coord.session_dir).ave_result == result["result"]
    assert await coord.phase_kernel.run_agent(ctx) == result
    assert free_gpus.call_count == (0 if gpu_state == "hidden" else 1)


@pytest.mark.asyncio
@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_stopping_ave_reaps_loop_and_its_driver_process(prepared_run, coord, monkeypatch, stop):
    import os
    import signal
    import sys
    import time

    from hyperloom.common.proctree import running

    pid_file = prepared_run.result_json.parent / "driver.pid"
    program = (
        "import subprocess, sys, time\n"
        f"child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], start_new_session=True)\n"
        f"open({str(pid_file)!r}, 'w').write(str(child.pid))\n"
        "time.sleep(60)\n"
    )
    spawn = subprocess.Popen
    processes = []

    def spawn_loop(command, **kwargs):
        assert command[1:4] == ["-m", "ave", "optimize"]
        assert json.loads(Path(command[4]).read_text())["kernel_path"] == prepared_run.kernel_path
        process = spawn([sys.executable, "-c", program], **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", spawn_loop)
    if stop == "timeout":
        object.__setattr__(prepared_run, "deadline_unix", time.time() + 1)
    scope = CancelScope()
    try:
        with use_cancel_scope(scope):
            task = asyncio.create_task(
                coord.phase_kernel._run_ave_subprocess(prepared_run, devices={"gpu_ids": "0", "tp": 1})
            )
        if stop == "cancel":

            async def wait_for_driver():
                while not pid_file.exists():
                    await asyncio.sleep(0.01)

            await asyncio.wait_for(wait_for_driver(), timeout=5)
            scope.cancel(reason="test_stop")
            with pytest.raises(CancelledError):
                await asyncio.wait_for(task, timeout=10)
        else:
            assert await asyncio.wait_for(task, timeout=10) == -9
        assert pid_file.exists()
        assert not running(int(pid_file.read_text()))
        assert not running(processes[0].pid)
    finally:
        scope.cancel(reason="test_cleanup")
        if pid_file.exists() and running(int(pid_file.read_text())):
            os.kill(int(pid_file.read_text()), signal.SIGKILL)
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", [None, "geak", "forge", "ave,forge"])
async def test_other_backends_do_not_prepare_ave(coord, tmp_path, monkeypatch, backend):
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER", raising=False)
    if backend:
        monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", backend)
    monkeypatch.setenv("AVE_PROMPTS_FILE", str(tmp_path / "missing-prompts.json"))
    monkeypatch.setattr("hyperloom.inference_optimizer.cli.bootstrap.resolve_framework_version", lambda args: "test")
    args = _build_parser().parse_args(["optimize", "--model", str(tmp_path)])
    state = _seed_shared_state(
        coord.session_dir,
        args,
        session_id=coord.session_dir.name,
        framework="sglang",
        platform=LaunchPlatform({}, {}),
    )
    coord.shared_state = state
    assert state.kernel_optimizer == ("forge" if backend == "forge" else "geak")
    assert coord.phase_kernel.geak_enabled() == (backend != "forge")
    assert state.ave_prompts == []
    prepare = AsyncMock()
    monkeypatch.setattr(coord.phase_kernel, "_prepare_ave_kernel", prepare)
    monkeypatch.setattr(coord.phase_kernel, "_gemm_tuning_required_before_kernel_opt", lambda: False)
    geak = AsyncMock()
    forge = AsyncMock()
    monkeypatch.setattr(coord.phase_kernel, "_run_geak_kernel_phase", geak)
    monkeypatch.setattr(coord.phase_kernel, "_finish_kernel_entry", forge)
    result = await coord.phase_kernel.run_agent(SimpleNamespace(task=SimpleNamespace(params={})))
    assert result["route"] == ("forge_no_gemm" if backend == "forge" else "geak")
    assert geak.await_count == (0 if backend == "forge" else 1)
    assert forge.await_count == (1 if backend == "forge" else 0)
    prepare.assert_not_called()
    assert not (coord.session_dir / "ave").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selection",
    [
        "no_kernel",
        "wrong_scope",
        "outside_repo",
        "invalid_path",
        "invalid_driver",
        "timeout",
        "provider_failure",
        "sdk_error",
        "missing_sdk",
    ],
)
async def test_ave_preparation_failure_settles_without_another_backend(prepared_run, coord, monkeypatch, selection):
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "ave")
    state = coord.shared_state
    state.kernel_optimizer = "ave"
    state.framework_repo_path = str(prepared_run.workspace)
    state.ave_prompts = [
        {"id": "tile", "files": ["unrelated.py"] if selection == "wrong_scope" else [], "prompt": "Apply tiling"}
    ]
    state.save(coord.session_dir)
    monkeypatch.setattr(coord.phase_kernel, "_maybe_reprofile_for_kernel", AsyncMock())
    monkeypatch.setattr(coord.phase_kernel, "_kernel_rewrite_controller_timeouts", lambda: (60, 60))
    monkeypatch.setattr(coord.phase_kernel, "_gemm_tuning_required_before_kernel_opt", lambda: False)
    geak = AsyncMock()
    forge = AsyncMock()
    monkeypatch.setattr(coord.phase_kernel, "_run_geak_kernel_phase", geak)
    monkeypatch.setattr(coord.phase_kernel, "_finish_kernel_entry", forge)
    free_gpus = Mock()
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors._server_lifecycle.free_gpus_for_handoff", free_gpus)
    selections = []

    async def query(*, prompt, options):
        selections.append(prompt)
        if selection == "timeout":
            raise asyncio.TimeoutError
        if selection == "provider_failure":
            raise RuntimeError("provider unavailable")
        if selection == "sdk_error":
            raise Exception("SDK control request failed")
        if selection == "missing_sdk":
            raise ModuleNotFoundError("No module named 'claude_agent_sdk'")
        kernel = (
            None
            if selection == "no_kernel"
            else {
                "repo_root": str(prepared_run.workspace),
                "kernel_path": "kernel.py",
                "source_files": ["kernel.py"],
                "operator_name": "kernel",
                "driver_source": prepared_run.driver.read_text(),
            }
        )
        if selection == "outside_repo":
            kernel["repo_root"] = str(prepared_run.workspace.parent)
        elif selection == "invalid_path":
            kernel["kernel_path"] = "../outside.py"
        elif selection == "invalid_driver":
            kernel["driver_source"] = "not valid Python !"
        yield session_result(json.dumps({"kernel": kernel, "reason": "No applicable Ave kernel"}), is_error=False)

    monkeypatch.setattr("claude_agent_sdk.query", query)
    worker = AsyncMock()
    monkeypatch.setattr(coord.phase_kernel, "_run_ave_subprocess", worker)
    ctx = SimpleNamespace(task=SimpleNamespace(params={"from_phase": "FRAMEWORK_AGENT"}))
    result = await coord.phase_kernel.run_agent(ctx)
    assert result["route"] == "ave"
    assert state.kernel_optimizer == "ave"
    assert (
        state.ave_result.get("status")
        == {
            "no_kernel": "skipped",
            "wrong_scope": "failed",
            "outside_repo": "failed",
            "invalid_path": "failed",
            "invalid_driver": "failed",
            "timeout": "timed_out",
            "provider_failure": "failed",
            "sdk_error": "failed",
            "missing_sdk": "failed",
        }[selection]
    )
    geak.assert_not_called()
    forge.assert_not_called()
    assert not machine_state.kernel_work_pending(state)
    coord.shared_state = SharedState.load_or_init(coord.session_dir)
    monkeypatch.delenv("KERNEL_OPT_BACKEND_ORDER")
    await coord.phase_kernel.run_agent(ctx)
    assert len(selections) == 1
    coord.phase_kernel._maybe_reprofile_for_kernel.assert_awaited_once()
    worker.assert_not_called()
    free_gpus.assert_not_called()
    assert repo_git(prepared_run.workspace, "status", "--porcelain").stdout == ""


def test_prompt_environment_validates_instructions(tmp_path, monkeypatch):
    prompt_file = tmp_path / "prompts.json"
    prompt_file.write_text(json.dumps({"steps": [{"id": "tile", "prompt": "Apply tiling"}]}))
    monkeypatch.setenv("AVE_PROMPTS_FILE", str(prompt_file))
    assert read_env_prompts() == (PromptStep("tile", "Apply tiling"),)
    prompt_file.write_text(
        json.dumps(
            {
                "steps": [
                    {"id": "tile", "prompt": "Apply tiling"},
                    {"id": "tile", "prompt": "Duplicate ID"},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="AVE_PROMPTS_FILE: duplicate step id"):
        read_env_prompts()
    prompt_file.unlink()
    with pytest.raises(ValueError, match="AVE_PROMPTS_FILE"):
        read_env_prompts()
    monkeypatch.delenv("AVE_PROMPTS_FILE")
    assert read_env_prompts() == (PromptStep("optimize", DEFAULT_PROMPT),)
