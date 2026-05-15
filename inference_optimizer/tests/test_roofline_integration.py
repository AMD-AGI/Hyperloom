"""Regression tests for PMC roofline GPU resolution."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator import roofline_integration as roofline
from inference_optimizer.orchestrator.action_executors import pmc_roofline
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.sub_agent_runner import RunnerContext
from inference_optimizer.orchestrator.task_registry import Task


def test_roofline_analyzer_resolves_gpu_from_env(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PMC_ROOFLINE_GPU_TYPE", raising=False)
    monkeypatch.setenv("GPU_TYPE", "mi300x")
    assert roofline.RooflineAnalyzer().spec.name == "MI300X"

    monkeypatch.setenv("HYPERLOOM_PMC_ROOFLINE_GPU_TYPE", "mi355x")
    assert roofline.RooflineAnalyzer().spec.name == "MI355X"


def test_roofline_gpu_resolution_uses_autodetect_when_env_missing(monkeypatch):
    monkeypatch.delenv("HYPERLOOM_PMC_ROOFLINE_GPU_TYPE", raising=False)
    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.setattr(roofline, "_autodetect_gpu_type", lambda: "mi325x")

    assert roofline.resolve_gpu_type("") == "mi325x"
    assert roofline.RooflineAnalyzer().spec.name == "MI325X"


def test_coordinator_pmc_params_include_session_gpu_type(tmp_path, monkeypatch):
    config_path = tmp_path / "baseline.yaml"
    config_path.write_text(
        """
benchmark:
  framework: sglang
  model: /models/test
  envs:
    TP: "1"
""",
        encoding="utf-8",
    )
    monkeypatch.delenv("HYPERLOOM_PMC_ROOFLINE_GPU_TYPE", raising=False)
    monkeypatch.delenv("GPU_TYPE", raising=False)

    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SimpleNamespace(
        baseline_config_path=str(config_path),
        framework="sglang",
        model_path="/models/test",
        gpu_type="mi300x",
    )

    params = coord._build_pmc_roofline_params()
    assert params is not None
    assert params["gpu_type"] == "mi300x"


@pytest.mark.asyncio
async def test_pmc_roofline_executor_forwards_gpu_type(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_isolated_pmc_roofline(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(pmc_roofline, "run_isolated_pmc_roofline", fake_run_isolated_pmc_roofline)
    task = Task(
        task_id="task123456",
        kind="pmc_roofline",
        state="queued",
        params={
            "allow_direct_gpu": True,
            "server_cmd": ["true"],
            "output_dir": str(tmp_path),
            "gpu_type": "mi355x",
        },
        idempotency_key="pmc-test",
    )

    result = await pmc_roofline.PMCRooflineExecutor()(RunnerContext(task=task, lease=None))

    assert result["status"] == "ok"
    assert captured["gpu_type"] == "mi355x"
