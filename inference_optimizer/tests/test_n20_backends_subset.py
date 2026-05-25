"""Roofline-v2 N20-A: roofline-driven backends variant subset selection.

Pre-N20 the BackendsExecutor only honoured two grid-control surfaces:
  (a) no override -> run full DEFAULT_BACKENDS_GRID (or vLLM grid),
                      then auto-augment with discovered flags;
  (b) params.grid -> LLM-supplied full custom variants
                     (name + extra_sglang_args + extra_envs); the
                     LLM owns the entire search space.

Empirically (both today's Qwen3 session AND 202 historical reports)
the LLM never used (b) and always fell through to (a) — running ALL
~10 registered variants regardless of what the roofline analysis.md
revealed about hot kernel categories. That wastes ~50% of grid
wall-clock on variants that have no chance of helping (e.g. running
attention backend swaps when the trace shows AllReduce-bound).

N20-A adds a third surface:
  (c) params.variants -> list[str] of registered variant names; the
                         executor narrows the registered grid by
                         name BEFORE auto-discovery augmentation
                         and runs only the named variants.

Safety properties (vs the pre-existing (b) custom-grid surface):
  * No flag hallucination — names must already exist in the
    registered grid. Misspellings fall through (with a debug log)
    instead of launching sglang with garbage args.
  * Auto-discovered flags (newly-added SGLang/vLLM CLI flags found
    via AST scan) are STILL appended — the subset filter applies
    only to the registered grid, not the augmentation pass. So the
    LLM can't accidentally subset away discovery results it didn't
    know existed.
  * Empty/None variants list = pre-N20 behaviour (run full grid).
  * Backward compat: when both `grid` and `variants` are supplied,
    `grid` wins — the LLM took full control via (b) and `variants`
    is silently ignored.

The companion prompt-side change (this same N20-A commit) renders a
BACKENDS GRID CATALOGUE block listing every registered variant name
+ trigger hint so the LLM knows which names are valid.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch, AsyncMock

import pytest

from inference_optimizer.orchestrator.action_executors.backends import (
    BackendsExecutor,
    DEFAULT_BACKENDS_GRID,
    DEFAULT_VLLM_BACKENDS_GRID,
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
    """Minimal Magpie YAML referencing the sglang framework so the
    BackendsExecutor picks DEFAULT_BACKENDS_GRID (not vLLM grid)."""
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
def executor(session_dir) -> BackendsExecutor:
    return BackendsExecutor(session_dir=session_dir)


def _make_ctx(*, task_id: str, params: dict) -> SimpleNamespace:
    """Build a minimal RunnerContext-shaped object for the executor."""
    return SimpleNamespace(
        task=SimpleNamespace(task_id=task_id, params=params),
        extra={},
    )


# Subset of names the LLM might realistically pick for an attention-heavy
# trace; all three live in DEFAULT_BACKENDS_GRID.
_ATTN_SUBSET = ["attn_aiter", "attn_triton", "decode_aiter"]


def _captured_grid_from_run_grid(mock_run_grid: AsyncMock) -> list[GridVariant]:
    """Pull the `grid=` kwarg the executor passed to run_grid."""
    assert mock_run_grid.call_args is not None
    grid_arg = mock_run_grid.call_args.kwargs.get("grid")
    if grid_arg is None and mock_run_grid.call_args.args:
        # positional fallback (run_grid signature has many positionals)
        # — keep this branch defensive in case the signature shifts.
        for a in mock_run_grid.call_args.args:
            if isinstance(a, list) and a and isinstance(a[0], GridVariant):
                return a
    return list(grid_arg or [])


# ===========================================================================
# (1) Backward compat — no subset = old behaviour
# ===========================================================================
class TestNoSubsetIsBackwardCompat:
    @pytest.mark.asyncio
    async def test_omitting_variants_runs_full_grid(
        self, executor, sglang_config, monkeypatch,
    ):
        """params={} + cap disabled -> entire DEFAULT_BACKENDS_GRID
        is passed to run_grid (plus any AST-discovered flags appended
        on top). The executor's default per-round candidate cap (5)
        is orthogonal — set max_candidates_per_round=0 to disable so
        we observe the raw grid the subset filter would see."""
        from inference_optimizer.orchestrator.action_executors import backends as mod
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
        passed_grid = _captured_grid_from_run_grid(rg)
        assert [v.name for v in passed_grid] == [
            v.name for v in DEFAULT_BACKENDS_GRID
        ]

    @pytest.mark.asyncio
    async def test_variants_empty_list_runs_full_grid(
        self, executor, sglang_config,
    ):
        """params.variants=[] is the explicit 'no subset, default
        behaviour' signal — must NOT collapse to zero variants. Same
        cap-disabled trick as above so we see the full grid."""
        from inference_optimizer.orchestrator.action_executors import backends as mod
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
        passed_grid = _captured_grid_from_run_grid(rg)
        assert len(passed_grid) == len(DEFAULT_BACKENDS_GRID)


# ===========================================================================
# (2) Subset filtering — the core N20-A behaviour
# ===========================================================================
class TestVariantsSubsetFiltering:
    @pytest.mark.asyncio
    async def test_subset_runs_only_requested_variants(
        self, executor, sglang_config,
    ):
        """params.variants=[3 names] -> exactly those 3 variants are
        forwarded to run_grid, in the order the LLM requested."""
        from inference_optimizer.orchestrator.action_executors import backends as mod
        ctx = _make_ctx(
            task_id="t3",
            params={
                "config_path": str(sglang_config),
                "variants": _ATTN_SUBSET,
                "disable_discovery": True,
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            await executor(ctx)
        passed_grid = _captured_grid_from_run_grid(rg)
        assert [v.name for v in passed_grid] == _ATTN_SUBSET

    @pytest.mark.asyncio
    async def test_subset_preserves_extra_args_from_registered_grid(
        self, executor, sglang_config,
    ):
        """A subset entry must carry the EXACT extra_sglang_args of
        the registered variant — N20-A is a 'pick by name' filter,
        not a re-render."""
        from inference_optimizer.orchestrator.action_executors import backends as mod
        registered_by_name = {v.name: v for v in DEFAULT_BACKENDS_GRID}
        ctx = _make_ctx(
            task_id="t4",
            params={
                "config_path": str(sglang_config),
                "variants": ["attn_aiter", "custom_ar"],
                "disable_discovery": True,
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            await executor(ctx)
        passed = _captured_grid_from_run_grid(rg)
        for v in passed:
            registered = registered_by_name[v.name]
            assert v.extra_sglang_args == registered.extra_sglang_args
            assert v.extra_envs == registered.extra_envs
            assert v.note == registered.note

    @pytest.mark.asyncio
    async def test_subset_silently_drops_unknown_names(
        self, executor, sglang_config,
    ):
        """LLM passes a typo or a stale name from an earlier session
        -> unknown names are silently dropped (with a debug log), the
        rest still run. Refuses ONLY when ALL names are unknown."""
        from inference_optimizer.orchestrator.action_executors import backends as mod
        ctx = _make_ctx(
            task_id="t5",
            params={
                "config_path": str(sglang_config),
                "variants": ["attn_aiter", "not_a_real_variant", "sched_lpm"],
                "disable_discovery": True,
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            await executor(ctx)
        passed = _captured_grid_from_run_grid(rg)
        assert [v.name for v in passed] == ["attn_aiter", "sched_lpm"]

    @pytest.mark.asyncio
    async def test_subset_refuses_when_all_names_unknown(
        self, executor, sglang_config,
    ):
        """All names unknown -> hard fail with bad_param. Sending an
        empty grid to run_grid would benchmark nothing + silently
        succeed, which is worse than a clear bad_param signal."""
        from inference_optimizer.orchestrator.action_executors import backends as mod
        ctx = _make_ctx(
            task_id="t6",
            params={
                "config_path": str(sglang_config),
                "variants": ["bogus_one", "bogus_two"],
                "disable_discovery": True,
            },
        )
        # run_grid should NOT be called at all when subset resolution fails
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            result = await executor(ctx)
        rg.assert_not_called()
        assert result["status"] == "failed"
        assert result["error_class"] == "bad_param"
        # error message must list the available names so the operator
        # can pick valid ones on the next attempt
        assert "available names:" in result["error"]
        assert "attn_aiter" in result["error"]

    @pytest.mark.asyncio
    async def test_subset_filters_non_string_entries(
        self, executor, sglang_config,
    ):
        """params.variants=[None, 42, '', '  ', 'attn_aiter'] -> the
        executor only respects the well-formed string entries. A
        single valid name should still resolve."""
        from inference_optimizer.orchestrator.action_executors import backends as mod
        ctx = _make_ctx(
            task_id="t7",
            params={
                "config_path": str(sglang_config),
                "variants": [None, 42, "", "  ", "attn_aiter"],
                "disable_discovery": True,
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            await executor(ctx)
        passed = _captured_grid_from_run_grid(rg)
        assert [v.name for v in passed] == ["attn_aiter"]


# ===========================================================================
# (3) Precedence — grid (Option B) wins over variants (Option A)
# ===========================================================================
class TestGridWinsOverVariants:
    @pytest.mark.asyncio
    async def test_grid_param_overrides_variants(
        self, executor, sglang_config,
    ):
        """If both are passed, `grid` (full custom variants from the
        LLM) takes precedence over `variants` (subset of registered
        grid). Documented in the executor docstring."""
        from inference_optimizer.orchestrator.action_executors import backends as mod
        custom_grid = [
            {"name": "custom_x", "extra_sglang_args": "--custom-flag",
             "extra_envs": {}, "note": "custom"},
        ]
        ctx = _make_ctx(
            task_id="t8",
            params={
                "config_path": str(sglang_config),
                "grid": custom_grid,
                "variants": ["attn_aiter", "sched_lpm"],  # ignored
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            await executor(ctx)
        passed = _captured_grid_from_run_grid(rg)
        # Only the LLM-supplied custom variant runs; the registered-grid
        # subset is silently ignored (grid wins).
        assert [v.name for v in passed] == ["custom_x"]


# ===========================================================================
# (4) Subset interaction with AST auto-discovery
# ===========================================================================
class TestSubsetVsDiscovery:
    @pytest.mark.asyncio
    async def test_subset_applies_before_discovery_augmentation(
        self, executor, sglang_config,
    ):
        """The subset filter narrows the REGISTERED grid; the AST
        discovery pass then appends newly-discovered flags ON TOP.
        So the final grid contains:
          - the LLM-requested subset of registered names
          - PLUS any auto-discovered boolean flags
        — discovered flags don't get subset-filtered, because the
        LLM didn't know they existed (couldn't have named them)."""
        from inference_optimizer.orchestrator.action_executors import backends as mod
        # Patch the discovery probe to return a deterministic synthetic
        # set so the test isn't dependent on the live sglang/vllm
        # source tree.
        synthetic_discovered = ["--discovered-flag-a", "--discovered-flag-b"]
        with patch.object(
            mod, "discover_backend_flags",
            return_value=synthetic_discovered,
        ), patch.object(
            mod, "resolve_sglang_server_args_path",
            return_value=(Path("/tmp/fake_server_args.py"), ""),
        ), patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            ctx = _make_ctx(
                task_id="t9",
                params={
                    "config_path": str(sglang_config),
                    "variants": ["attn_aiter"],
                    # discovery enabled by default
                },
            )
            await executor(ctx)
        passed = _captured_grid_from_run_grid(rg)
        names = [v.name for v in passed]
        # The 1 subset-requested registered variant is present
        assert "attn_aiter" in names
        # The 2 auto-discovered flags are appended (subset filter
        # does NOT block discovery output)
        flags_in_grid = " ".join(v.extra_sglang_args or "" for v in passed)
        assert "--discovered-flag-a" in flags_in_grid
        assert "--discovered-flag-b" in flags_in_grid


# ===========================================================================
# (5) vLLM grid path
# ===========================================================================
class TestVllmGridSubset:
    @pytest.mark.asyncio
    async def test_subset_picks_from_vllm_grid_when_framework_is_vllm(
        self, executor, session_dir,
    ):
        from inference_optimizer.orchestrator.action_executors import backends as mod
        cfg = session_dir / "config.yaml"
        cfg.write_text(
            "benchmark:\n"
            "  framework: vllm\n"
            "  model: /wekafs/models/Qwen-Qwen3-32B\n"
            "  gpu_type: mi300x\n",
            encoding="utf-8",
        )
        # Pick a name we know exists only in the vLLM grid
        vllm_name = DEFAULT_VLLM_BACKENDS_GRID[0].name
        ctx = _make_ctx(
            task_id="t10",
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

    @pytest.mark.asyncio
    async def test_sglang_name_invalid_against_vllm_framework(
        self, executor, session_dir,
    ):
        """An SGLang-only variant name (e.g. attn_aiter) is unknown
        when the framework is vLLM — must be silently dropped or
        cause a bad_param when ALL names are SGLang-only."""
        from inference_optimizer.orchestrator.action_executors import backends as mod
        cfg = session_dir / "config.yaml"
        cfg.write_text(
            "benchmark:\n"
            "  framework: vllm\n"
            "  model: /wekafs/models/Qwen-Qwen3-32B\n"
            "  gpu_type: mi300x\n",
            encoding="utf-8",
        )
        ctx = _make_ctx(
            task_id="t11",
            params={
                "config_path": str(cfg),
                "variants": ["attn_aiter", "sched_lpm"],  # SGLang-only
                "disable_discovery": True,
            },
        )
        with patch.object(mod, "run_grid", new=AsyncMock(return_value=[])) as rg:
            result = await executor(ctx)
        rg.assert_not_called()
        assert result["status"] == "failed"
        assert result["error_class"] == "bad_param"


# ===========================================================================
# (6) Prompt rendering of the catalogue
# ===========================================================================
class TestPromptCatalogueRendering:
    def test_backends_catalogue_appears_when_backends_enabled(self):
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
        assert "BACKENDS GRID CATALOGUE (SGLang)" in text
        # All SGLang variant names must appear in the rendered catalogue
        for v in DEFAULT_BACKENDS_GRID:
            assert v.name in text, f"missing variant {v.name!r} in catalogue"
        # The instructional example must be present so the LLM sees
        # the exact propose syntax for `variants`.
        assert "params={variants:" in text
        assert "attn_aiter" in text  # in example body

    def test_backends_catalogue_omitted_when_backends_disabled(self):
        """Dead weight removal — when backends is not in the run's
        enabled set, the catalogue should not be rendered."""
        from inference_optimizer.orchestrator.system_prompts.prompt_builder import (
            build_orchestration_prompt,
        )
        from inference_optimizer.orchestrator.action_registry import ActionRegistry
        reg = ActionRegistry()
        reg.load()
        text = build_orchestration_prompt(
            action_registry=reg,
            enabled_actions=("baseline", "target_analysis", "report"),
            framework="sglang",
            objective_kind="time_only",
            objective_value=None,
            max_minutes=60,
        )
        assert "BACKENDS GRID CATALOGUE" not in text

    def test_vllm_framework_renders_vllm_catalogue(self):
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
            framework="vllm",
            objective_kind="time_only",
            objective_value=None,
            max_minutes=60,
        )
        assert "BACKENDS GRID CATALOGUE (vLLM)" in text
        # vLLM-specific name should be present
        assert DEFAULT_VLLM_BACKENDS_GRID[0].name in text
        # The CATALOGUE table body should not include SGLang-only names.
        # We can't just assert "attn_aiter" not in text because the
        # generic hint section (action catalogue / GRID OVERRIDE)
        # mentions it in an example. Slice the rendered prompt at the
        # catalogue header and check only that block.
        cat_start = text.index("BACKENDS GRID CATALOGUE (vLLM)")
        # Catalogue ends at the next `## ` section header
        cat_body = text[cat_start:]
        next_section = cat_body.find("\n## ", 1)
        if next_section != -1:
            cat_body = cat_body[:next_section]
        # SGLang's `attn_aiter` shouldn't be in the rendered vLLM
        # catalogue rows (vLLM has its own `vllm_attn_aiter_fa` which
        # also contains the substring "attn_aiter" — match the exact
        # row prefix instead of substring).
        for v in DEFAULT_BACKENDS_GRID:
            row_prefix = f"{v.name:28s}"  # name padded to col 28 in renderer
            if v.name == "vllm_attn_aiter_fa":  # belt-and-suspenders
                continue
            assert row_prefix not in cat_body, (
                f"SGLang-only variant {v.name!r} must not appear in "
                "vLLM catalogue body"
            )
