"""Roofline-v2 N20-A: roofline-driven params variant subset selection.

Mirror of test_n20_backends_subset.py — same Option A surface
(`params.variants=['name1','name2']`) wired into ParamsExecutor +
catalogue rendering in prompt_builder for the `params` action.

The params grid is ~3x larger than the backends grid (~28 vs ~10
registered SGLang variants), so the leverage from LLM-driven subset
selection is larger here:
  - full grid: ~30-45min on Qwen3-32B / ~4-5h on R1
  - 3-5 variant subset based on roofline: ~10-20min / ~1-2h
  - savings: ~50-70% of cheap-action wall-clock per round

Same safety properties as backends:
  - no flag VALUE hallucination (LLM picks names from registered grid;
    the registered grid's values are validated by the operator who
    added them)
  - unknown names silently dropped; all-unknown -> bad_param
  - `grid` (Option B, custom variants) wins over `variants` (Option A)
    when both are passed
  - empty/None -> pre-N20 behaviour (run full grid)
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

import pytest

from inference_optimizer.orchestrator.action_executors.params import (
    ParamsExecutor,
    DEFAULT_PARAMS_GRID,
    DEFAULT_VLLM_PARAMS_GRID,
    GridVariant,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return tmp_path


@pytest.fixture
def sglang_config(session_dir) -> Path:
    cfg = session_dir / "config.yaml"
    cfg.write_text(
        "benchmark:\n"
        "  framework: sglang\n"
        "  model: /wekafs/models/Qwen-Qwen3-32B\n"
        "  gpu_type: mi300x\n",
        encoding="utf-8",
    )
    return cfg


@pytest.fixture
def executor(session_dir) -> ParamsExecutor:
    return ParamsExecutor(session_dir=session_dir)


def _make_ctx(*, task_id: str, params: dict) -> SimpleNamespace:
    return SimpleNamespace(
        task=SimpleNamespace(task_id=task_id, params=params),
        extra={},
    )


def _captured_grid_from_run_grid(mock_run_grid: AsyncMock) -> list[GridVariant]:
    assert mock_run_grid.call_args is not None
    grid_arg = mock_run_grid.call_args.kwargs.get("grid")
    if grid_arg is None and mock_run_grid.call_args.args:
        for a in mock_run_grid.call_args.args:
            if isinstance(a, list) and a and isinstance(a[0], GridVariant):
                return a
    return list(grid_arg or [])


# Realistic LLM choice: roofline says cuda-graph misses + KV pressure
# -> pick the cuda_graph_* family + a mem_fraction tweak.
_CUDA_GRAPH_SUBSET = [
    "cuda_graph_max_bs_32", "cuda_graph_max_bs_64", "mem_fraction_0_90",
]


class TestNoSubsetIsBackwardCompat:
    @pytest.mark.asyncio
    async def test_omitting_variants_runs_full_grid(
        self, executor, sglang_config,
    ):
        """params={} -> full DEFAULT_PARAMS_GRID forwarded (cap disabled
        so the assertion sees the raw grid)."""
        from inference_optimizer.orchestrator.action_executors import params as mod
        ctx = _make_ctx(
            task_id="t1",
            params={
                "config_path": str(sglang_config),
                "disable_discovery": True,
                "max_candidates_per_round": 0,
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            await executor(ctx)
        passed = _captured_grid_from_run_grid(rg)
        assert [v.name for v in passed] == [
            v.name for v in DEFAULT_PARAMS_GRID
        ]

    @pytest.mark.asyncio
    async def test_variants_empty_list_runs_full_grid(
        self, executor, sglang_config,
    ):
        from inference_optimizer.orchestrator.action_executors import params as mod
        ctx = _make_ctx(
            task_id="t2",
            params={
                "config_path": str(sglang_config),
                "variants": [],
                "disable_discovery": True,
                "max_candidates_per_round": 0,
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            await executor(ctx)
        passed = _captured_grid_from_run_grid(rg)
        assert len(passed) == len(DEFAULT_PARAMS_GRID)


class TestVariantsSubsetFiltering:
    @pytest.mark.asyncio
    async def test_subset_runs_only_requested_variants(
        self, executor, sglang_config,
    ):
        from inference_optimizer.orchestrator.action_executors import params as mod
        ctx = _make_ctx(
            task_id="t3",
            params={
                "config_path": str(sglang_config),
                "variants": _CUDA_GRAPH_SUBSET,
                "disable_discovery": True,
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            await executor(ctx)
        passed = _captured_grid_from_run_grid(rg)
        assert [v.name for v in passed] == _CUDA_GRAPH_SUBSET

    @pytest.mark.asyncio
    async def test_subset_preserves_extra_args_from_registered_grid(
        self, executor, sglang_config,
    ):
        from inference_optimizer.orchestrator.action_executors import params as mod
        registered = {v.name: v for v in DEFAULT_PARAMS_GRID}
        ctx = _make_ctx(
            task_id="t4",
            params={
                "config_path": str(sglang_config),
                "variants": ["cuda_graph_max_bs_64", "disable_radix_cache"],
                "disable_discovery": True,
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            await executor(ctx)
        passed = _captured_grid_from_run_grid(rg)
        for v in passed:
            r = registered[v.name]
            assert v.extra_sglang_args == r.extra_sglang_args
            assert v.extra_envs == r.extra_envs
            assert v.note == r.note

    @pytest.mark.asyncio
    async def test_subset_silently_drops_unknown_names(
        self, executor, sglang_config,
    ):
        from inference_optimizer.orchestrator.action_executors import params as mod
        ctx = _make_ctx(
            task_id="t5",
            params={
                "config_path": str(sglang_config),
                "variants": [
                    "cuda_graph_max_bs_64", "definitely_not_a_real_variant",
                    "mem_fraction_0_85",
                ],
                "disable_discovery": True,
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            await executor(ctx)
        passed = _captured_grid_from_run_grid(rg)
        assert [v.name for v in passed] == [
            "cuda_graph_max_bs_64", "mem_fraction_0_85",
        ]

    @pytest.mark.asyncio
    async def test_subset_refuses_when_all_names_unknown(
        self, executor, sglang_config,
    ):
        from inference_optimizer.orchestrator.action_executors import params as mod
        ctx = _make_ctx(
            task_id="t6",
            params={
                "config_path": str(sglang_config),
                "variants": ["bogus_one", "bogus_two"],
                "disable_discovery": True,
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            result = await executor(ctx)
        rg.assert_not_called()
        assert result["status"] == "failed"
        assert result["error_class"] == "bad_param"
        assert "available names:" in result["error"]
        # At least one known SGLang params name should appear in the
        # 'available' list so the operator can pick a valid one
        assert "cuda_graph_max_bs_64" in result["error"]

    @pytest.mark.asyncio
    async def test_subset_filters_non_string_entries(
        self, executor, sglang_config,
    ):
        from inference_optimizer.orchestrator.action_executors import params as mod
        ctx = _make_ctx(
            task_id="t7",
            params={
                "config_path": str(sglang_config),
                "variants": [None, 42, "", "  ", "cuda_graph_max_bs_32"],
                "disable_discovery": True,
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            await executor(ctx)
        passed = _captured_grid_from_run_grid(rg)
        assert [v.name for v in passed] == ["cuda_graph_max_bs_32"]


class TestGridWinsOverVariants:
    @pytest.mark.asyncio
    async def test_grid_param_overrides_variants(
        self, executor, sglang_config,
    ):
        from inference_optimizer.orchestrator.action_executors import params as mod
        custom_grid = [
            {"name": "custom_x", "extra_sglang_args": "--custom-flag",
             "extra_envs": {}, "note": "custom"},
        ]
        ctx = _make_ctx(
            task_id="t8",
            params={
                "config_path": str(sglang_config),
                "grid": custom_grid,
                "variants": ["cuda_graph_max_bs_64"],  # ignored
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            await executor(ctx)
        passed = _captured_grid_from_run_grid(rg)
        assert [v.name for v in passed] == ["custom_x"]


class TestVllmGridSubset:
    @pytest.mark.asyncio
    async def test_subset_picks_from_vllm_grid_when_framework_is_vllm(
        self, executor, session_dir,
    ):
        from inference_optimizer.orchestrator.action_executors import params as mod
        cfg = session_dir / "config.yaml"
        cfg.write_text(
            "benchmark:\n"
            "  framework: vllm\n"
            "  model: /wekafs/models/Qwen-Qwen3-32B\n"
            "  gpu_type: mi300x\n",
            encoding="utf-8",
        )
        if not DEFAULT_VLLM_PARAMS_GRID:
            pytest.skip("no vLLM params grid registered")
        vllm_name = DEFAULT_VLLM_PARAMS_GRID[0].name
        ctx = _make_ctx(
            task_id="t9",
            params={
                "config_path": str(cfg),
                "variants": [vllm_name],
                "disable_discovery": True,
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            await executor(ctx)
        passed = _captured_grid_from_run_grid(rg)
        assert [v.name for v in passed] == [vllm_name]


class TestPromptCatalogueRendering:
    def test_params_catalogue_appears_when_params_enabled(self):
        from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
            build_orchestration_prompt,
            FULL_ENABLED_ACTIONS,
        )
        from inference_optimizer.orchestrator.action_registry import ActionRegistry
        reg = ActionRegistry()
        reg.load()
        text = build_orchestration_prompt(
            action_registry=reg,
            enabled_actions=FULL_ENABLED_ACTIONS,
            framework="sglang",
            objective_kind="time_only",
            objective_value=None,
            max_minutes=60,
        )
        assert "PARAMS GRID CATALOGUE (SGLang)" in text
        # All registered SGLang params names must appear in the catalogue
        for v in DEFAULT_PARAMS_GRID:
            assert v.name in text, f"missing variant {v.name!r}"
        # Example block must show the subset syntax
        assert "params={variants:" in text
        assert "cuda_graph_max_bs" in text

    def test_params_catalogue_omitted_when_params_disabled(self):
        from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
            build_orchestration_prompt,
        )
        from inference_optimizer.orchestrator.action_registry import ActionRegistry
        reg = ActionRegistry()
        reg.load()
        # Enable only baseline/report — no params action -> no catalogue
        text = build_orchestration_prompt(
            action_registry=reg,
            enabled_actions=("baseline", "target_analysis", "report"),
            framework="sglang",
            objective_kind="time_only",
            objective_value=None,
            max_minutes=60,
        )
        assert "PARAMS GRID CATALOGUE" not in text
