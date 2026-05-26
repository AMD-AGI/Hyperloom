"""Unit tests for ``action_executors.pmc_roofline``.

This driver wraps :func:`run_isolated_pmc_roofline` (which is omitted from
coverage because it shells out to rocprofv3 + sglang). Here we test the
guard rails — Ray-context detection, GPU-visibility env overrides, config
materialisation fallback, and the missing-cmd error paths — without
launching any real subprocess.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors import pmc_roofline as pmc


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _ctx(*, params: dict | None = None, task_id: str = "t-abcdefgh") -> SimpleNamespace:
    return SimpleNamespace(
        task=SimpleNamespace(task_id=task_id, params=params or {}),
        extra={},
    )


@pytest.fixture(autouse=True)
def _clean_pmc_env(monkeypatch):
    for key in (
        "HYPERLOOM_PMC_ROOFLINE_IN_RAY",
        "RAY_ADDRESS",
        "RAY_JOB_ID",
        "RAY_RUNTIME_ENV_CREATE_WORKING_DIR",
        "HYPERLOOM_ALLOW_DIRECT_PMC_ROOFLINE",
    ):
        monkeypatch.delenv(key, raising=False)


# ---------------------------------------------------------------------------
# _truthy / _ray_context_present
# ---------------------------------------------------------------------------

class TestTruthy:
    @pytest.mark.parametrize("value", ["1", "true", "TRUE", "yes", "on"])
    def test_truthy_values(self, value):
        assert pmc._truthy(value) is True

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", "", None, 0])
    def test_falsy_values(self, value):
        assert pmc._truthy(value) is False


class TestRayContextPresent:
    def test_explicit_param(self):
        assert pmc._ray_context_present({"ray_worker": "1"}) is True

    def test_env_signals_presence(self, monkeypatch):
        monkeypatch.setenv("RAY_ADDRESS", "10.0.0.1:6379")
        assert pmc._ray_context_present({}) is True

    def test_absent_without_signals(self):
        assert pmc._ray_context_present({}) is False


# ---------------------------------------------------------------------------
# _find_materialized_config
# ---------------------------------------------------------------------------

class TestFindMaterializedConfig:
    def test_explicit_config_path_returned(self, tmp_path):
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("benchmark: {}\n")
        assert pmc._find_materialized_config({"config_path": str(cfg)}) == cfg

    def test_workspace_lookup(self, tmp_path):
        ws = tmp_path / "ws"
        ws.mkdir()
        target = ws / "baseline_config.yaml"
        target.write_text("benchmark: {}\n")
        assert pmc._find_materialized_config({"workspace": str(ws)}) == target

    def test_returns_none_when_nothing_found(self, tmp_path):
        assert pmc._find_materialized_config({}) is None
        assert pmc._find_materialized_config({"workspace": str(tmp_path)}) is None


# ---------------------------------------------------------------------------
# PMCRooflineExecutor — guard rails
# ---------------------------------------------------------------------------

class TestPMCRooflineExecutor:
    @pytest.mark.asyncio
    async def test_rejects_without_ray_context(self):
        ex = pmc.PMCRooflineExecutor()
        result = await ex(_ctx(params={"server_cmd": ["python", "-V"]}))
        assert result["status"] == "failed"
        assert result["error_class"] == "ray_worker_required"

    @pytest.mark.asyncio
    async def test_allow_direct_gpu_skips_ray_gate(self, monkeypatch):
        called = {}

        def fake_run(**kwargs):
            called["kwargs"] = kwargs
            return {"status": "succeeded", "duration_ms": kwargs.get("duration_ms")}

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.pmc_roofline."
            "run_isolated_pmc_roofline",
            fake_run,
        )
        ex = pmc.PMCRooflineExecutor()
        result = await ex(_ctx(params={
            "allow_direct_gpu": True,
            "server_cmd": ["python", "-V"],
            "health_url": "http://x/health",
            "duration_ms": 100,
        }))
        assert result["status"] == "succeeded"
        assert called["kwargs"]["duration_ms"] == 100

    @pytest.mark.asyncio
    async def test_rejects_overriding_gpu_visibility(self):
        ex = pmc.PMCRooflineExecutor()
        result = await ex(_ctx(params={
            "ray_worker": True,
            "server_cmd": ["python"],
            "extra_envs": {"ROCR_VISIBLE_DEVICES": "0,1"},
        }))
        assert result["status"] == "failed"
        assert result["error_class"] == "ray_gpu_visibility_override"

    @pytest.mark.asyncio
    async def test_missing_server_cmd_when_no_config(self):
        ex = pmc.PMCRooflineExecutor()
        result = await ex(_ctx(params={"ray_worker": True}))
        assert result["status"] == "failed"
        assert result["error_class"] == "missing_server_cmd"

    @pytest.mark.asyncio
    async def test_derives_server_cmd_from_config(self, monkeypatch, tmp_path):
        cfg = tmp_path / "baseline_config.yaml"
        cfg.write_text("benchmark: {}\n")

        def fake_derive(*args, **kwargs):
            return {
                "server_cmd": "python -V",
                "health_url": "http://derived/health",
                "benchmark_cmd": ["python", "-V"],
            }

        captured = {}

        def fake_run(**kwargs):
            captured.update(kwargs)
            return {"status": "succeeded"}

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.pmc_roofline."
            "derive_pmc_roofline_params_from_config",
            fake_derive,
        )
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.pmc_roofline."
            "run_isolated_pmc_roofline",
            fake_run,
        )

        ex = pmc.PMCRooflineExecutor()
        result = await ex(_ctx(params={
            "ray_worker": True,
            "config_path": str(cfg),
        }))
        assert result["status"] == "succeeded"
        # Derived health_url propagated to the run.
        assert captured["health_url"] == "http://derived/health"
