# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement revalidation measures and retains the same benchmark script."""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import yaml

from hyperloom.orchestrator.actions.executors import BaselineExecutor, ExploreExecutor
from hyperloom.orchestrator.actions.executors._accuracy_gate import ENABLEMENT_REVALIDATION_REASON
from hyperloom.orchestrator.actions.executors._grid_runner import _build_variant_yaml
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.loop.sub_agent_runner import RunnerContext
from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.fixture
def coordinator(session_dir, monkeypatch):
    for name in (
        "INFERENCEX_PATH",
        "HYPERLOOM_AGENTX",
        "INFERENCE_OPTIMIZER_SERVER_ARGS",
        "INFERENCE_OPTIMIZER_EXTRA_ENV",
        "RUN_EVAL",
        "MODEL_PATH",
        "MAX_MODEL_LEN",
        "TP",
        "CONC",
        "ISL",
        "OSL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("GPU_TYPE", "mi355x")
    monkeypatch.setenv("FRAMEWORK", "sglang")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP", "1")
    monkeypatch.setenv("INFERENCE_OPTIMIZER_LEAK_ROOTS", str(session_dir / "no_leaked_reports"))
    monkeypatch.setattr(
        "hyperloom.orchestrator.actions.executors._ray_serving.maybe_serving_lease", lambda **_kwargs: None
    )
    backends = {
        name: MockBackend(ScriptedPlan(turns=[]), name=name) for name in ("orchestration", "critic", "robustness")
    }
    coord = Coordinator(session_dir, backends=backends)
    yield coord
    coord.db.close()


@pytest.fixture
def cpu_benchmark(tmp_path):
    """Replace the GPU harness with a child that executes the YAML-selected script."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for script, tput in (("sglang_custom.sh", 105.0), ("sglang_mi355x.sh", 140.0)):
        report = {
            "success": True,
            "framework": "sglang",
            "script": script,
            "throughput": {"output_throughput": tput, "completed_requests": 8, "duration_seconds": 1.0},
        }
        (scripts / script).write_text(f"printf '%s\\n' {shlex.quote(json.dumps(report))}\n", encoding="utf-8")
    harness = tmp_path / "cpu_benchmark.py"
    harness.write_text(
        textwrap.dedent(
            """
            import argparse
            import json
            import os
            import subprocess
            from pathlib import Path

            import yaml

            parser = argparse.ArgumentParser()
            parser.add_argument("--benchmark-config", type=Path, required=True)
            parser.add_argument("--output-dir", type=Path, required=True)
            args = parser.parse_args()
            bench = yaml.safe_load(args.benchmark_config.read_text())["benchmark"]
            env = {**os.environ, **{key: str(value) for key, value in bench["envs"].items()}}
            script = Path(__file__).parent / "scripts" / bench["benchmark_script"]
            child = subprocess.run(["bash", str(script)], env=env, capture_output=True, text=True, check=True)
            report = json.loads(child.stdout)
            workspace = args.output_dir / "benchmark_sglang_cpu"
            workspace.mkdir(parents=True)
            (workspace / "benchmark_report.json").write_text(json.dumps(report))
            receipt = {"script": script.name, "runner_type": bench["runner_type"], "report": report}
            (args.output_dir / "child_receipt.json").write_text(json.dumps(receipt))
            result_dir = Path(env["RESULT_DIR"])
            result_dir.mkdir(parents=True, exist_ok=True)
            (result_dir / "results_cpu.json").write_text(
                json.dumps({"results": {"gsm8k": {"exact_match,flexible-extract": 0.95}}})
            )
            """
        ),
        encoding="utf-8",
    )

    def command(*, python_exe, config_path, output_dir):
        return [sys.executable, str(harness), "--benchmark-config", str(config_path), "--output-dir", str(output_dir)]

    return command


@pytest.mark.asyncio
@pytest.mark.parametrize("script_source", ["accepted", "legacy", "generic", "queued_without_override"])
async def test_revalidation_script_matches_measured_anchor_after_resume(
    coordinator, cpu_benchmark, tmp_path, monkeypatch, script_source
):
    config = tmp_path / "accepted.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "framework": "sglang",
                    "model": "/models/test",
                    "run_mode": "local",
                    "benchmark_script": "sglang_custom.sh",
                    "envs": {"TP": 1, "ISL": 256, "OSL": 256, "CONC": 8},
                }
            }
        ),
        encoding="utf-8",
    )
    state = coordinator.shared_state
    state.model_path = "/models/test"
    state.framework = "sglang"
    state.baseline_tput = 100.0
    state.baseline_accuracy = 0.94
    state.baseline_config_path = str(config)
    state.baseline_double_run = False
    state.baseline_benchmark_script = None if script_source == "legacy" else "sglang_custom.sh"
    if script_source == "generic":
        state.baseline_benchmark_script = ""
    state.last_baseline = {
        "decision": "promoted",
        "extras": {
            "fingerprint": {"benchmark_script": "sglang_custom.sh" if script_source == "legacy" else "stale.sh"}
        },
    }
    state.enablement.origin = "eval"
    state.enablement.validation_pending = True
    state.enablement.accuracy_floor = 0.9
    state.enablement.accepted_config_path = str(config)
    state.save(coordinator.session_dir)
    coordinator.shared_state = state = SharedState.load_or_init(coordinator.session_dir)

    if script_source == "queued_without_override":
        # A persisted task created by an older producer may still lack the override.
        task = await coordinator.tasks.create(
            kind="baseline",
            params={"reason": ENABLEMENT_REVALIDATION_REASON, "config_path": str(config)},
            idempotency_key="old-producer-revalidation",
        )
        state.enablement.revalidation_task_id = task.task_id
    else:
        task_id = await coordinator._maybe_enqueue_enablement_baseline_revalidation()
        assert task_id
        task = await coordinator.tasks.get(task_id)

    expected_script = "sglang_custom.sh" if script_source in {"accepted", "legacy"} else "sglang_mi355x.sh"
    expected_tput = 105.0 if expected_script == "sglang_custom.sh" else 140.0
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors.baseline.build_benchmark_command", cpu_benchmark)
    executor = BaselineExecutor(magpie_python=sys.executable, session_dir=coordinator.session_dir, shared_state=state)
    monkeypatch.setattr(executor, "_after_materialize_config", lambda *_args: None)
    monkeypatch.setattr(executor, "_preflight_server_argv", lambda **_kwargs: None)
    result = await executor(
        RunnerContext(task=task, lease=None, extra={"session_dir": coordinator.session_dir, "shared_state": state})
    )
    assert result["status"] == "succeeded", result
    baseline_receipt = json.loads((Path(result["output_dir"]) / "child_receipt.json").read_text())
    measured_yaml = yaml.safe_load(Path(result["materialized_config"]).read_text())["benchmark"]
    assert baseline_receipt["script"] == measured_yaml["benchmark_script"] == expected_script
    assert baseline_receipt["runner_type"] == "mi355x"
    assert result["output_throughput"] == baseline_receipt["report"]["throughput"]["output_throughput"] == expected_tput
    assert result["accuracy"] == 0.95

    await coordinator._promote_to_shared_state("baseline", result, task=task)
    state.save(coordinator.session_dir)
    coordinator.shared_state = state = SharedState.load_or_init(coordinator.session_dir)
    assert state.baseline_tput == expected_tput
    assert state.baseline_accuracy == 0.95
    assert state.baseline_config_path == result["materialized_config"]
    assert state.baseline_benchmark_script == ("sglang_custom.sh" if expected_script == "sglang_custom.sh" else "")
    assert state.enablement.succeeded
    assert not state.enablement.validation_pending

    state.current_best["extra_server_args"] = "--mem-fraction-static 0.9"
    enqueued = await coordinator._enqueue_internal_stack_rebench(reason="revalidated_anchor")
    rebench = await coordinator.tasks.get(enqueued["task_id"])
    child_receipts = []

    async def launch_grid(**kwargs):
        output = tmp_path / "resumed_rebench"
        output.mkdir()
        variant = _build_variant_yaml(
            kwargs["base_yaml_path"],
            kwargs["base_extra_args"],
            kwargs["grid"][0],
            output_subdir=output,
            gpu_type=kwargs["gpu_type"],
            benchmark_script=kwargs["benchmark_script"],
        )
        subprocess.run(
            cpu_benchmark(python_exe=sys.executable, config_path=variant, output_dir=output),
            env={**os.environ, "RESULT_DIR": str(output)},
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        child_receipts.append(json.loads((output / "child_receipt.json").read_text()))
        return []

    monkeypatch.setattr("hyperloom.orchestrator.actions.executors.explore.run_grid", launch_grid)
    monkeypatch.setattr("hyperloom.orchestrator.actions.executors.explore.maybe_serving_lease", lambda **_kwargs: None)
    await ExploreExecutor(session_dir=coordinator.session_dir)._run_explore(
        RunnerContext(task=rebench, lease=None, extra={"shared_state": state})
    )
    assert len(child_receipts) == 1
    assert child_receipts[0]["script"] == expected_script
    assert child_receipts[0]["report"]["throughput"]["output_throughput"] == state.baseline_tput
