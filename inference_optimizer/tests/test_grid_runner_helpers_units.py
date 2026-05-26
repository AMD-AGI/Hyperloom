"""Helper-level unit tests for ``orchestrator.action_executors._grid_runner``.

The full ``run_grid`` flow is exercised by the existing grid-runner /
baseline-override tests. Here we hit the small parsing + filter
utilities (skip-spec parsing, multi-node invalid variant filter, the
``VariantResult`` ``to_dict`` round-trip, etc.) that current tests miss.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors import _grid_runner as gr
from inference_optimizer.orchestrator.action_executors._grid_runner import (
    GridVariant,
    VariantResult,
    apply_multi_node_invalid_variants,
    apply_single_node_invalid_variants,
    apply_user_skip_list,
    resolve_skip_spec,
    _parse_skip_spec,
)


# ---------------------------------------------------------------------------
# resolve_skip_spec
# ---------------------------------------------------------------------------

class TestResolveSkipSpec:
    def test_returns_empty_when_no_params_and_no_env(self, monkeypatch):
        monkeypatch.delenv("SKIP_VARIANTS", raising=False)
        assert resolve_skip_spec(None) == ""

    def test_env_used_when_params_absent(self, monkeypatch):
        monkeypatch.setenv("SKIP_VARIANTS", "foo,bar")
        assert resolve_skip_spec(None) == "foo,bar"

    def test_params_list_flattened(self):
        assert resolve_skip_spec({"skip_variants": ["a", "b", None]}) == "a,b"

    def test_params_str_passthrough(self):
        assert resolve_skip_spec({"skip_variants": "x"}) == "x"

    def test_params_override_env(self, monkeypatch):
        monkeypatch.setenv("SKIP_VARIANTS", "env-default")
        assert resolve_skip_spec({"skip_variants": "explicit"}) == "explicit"


class TestParseSkipSpec:
    def test_splits_on_commas_and_whitespace(self):
        assert _parse_skip_spec("a, b\nc d") == ["a", "b", "c", "d"]

    def test_empty_input_returns_empty_list(self):
        assert _parse_skip_spec("") == []


# ---------------------------------------------------------------------------
# apply_multi_node_invalid_variants
# ---------------------------------------------------------------------------

class TestMultiNodeInvalidFilter:
    def test_short_circuits_in_single_node(self, monkeypatch):
        # ``is_multi_node()`` is False with the conftest sentinel applied.
        grid = [
            GridVariant(name="a", extra_sglang_args="--cuda-graph-max-bs 8"),
        ]
        kept, dropped = apply_multi_node_invalid_variants(grid)
        assert kept == grid
        assert dropped == []

    def test_drops_undersized_cuda_graph_max_bs_in_multi_node(self, monkeypatch):
        monkeypatch.setattr(gr, "is_multi_node", lambda: True, raising=False)
        # Patch the symbol used by `apply_multi_node_invalid_variants` via
        # the late import inside the function. We instead reach into
        # ``_multi_node_env.is_multi_node`` so the helper sees True.
        from inference_optimizer.orchestrator.action_executors import (
            _multi_node_env as mne,
        )

        monkeypatch.setattr(mne, "is_multi_node", lambda: True)
        monkeypatch.setenv("CONC", "64")
        grid = [
            GridVariant(name="bad", extra_sglang_args="--cuda-graph-max-bs 8"),
            GridVariant(name="ok",  extra_sglang_args="--max-num-seqs 128"),
        ]
        kept, dropped = apply_multi_node_invalid_variants(grid)
        assert [k.name for k in kept] == ["ok"]
        assert [d["name"] for d in dropped] == ["bad"]
        assert "CONC=64" in dropped[0]["reason"]


class TestSingleNodeInvalidFilter:
    def test_drops_multi_node_only_in_single_node(self):
        grid = [
            GridVariant(name="legacy", extra_sglang_args="--foo 1"),
            GridVariant(
                name="mn_only",
                extra_sglang_args="--enable-deepep-moe",
                note="multi_node_only_moe",
            ),
        ]
        kept, dropped = apply_single_node_invalid_variants(grid)
        assert [k.name for k in kept] == ["legacy"]
        assert [d["name"] for d in dropped] == ["mn_only"]

    def test_short_circuits_in_multi_node(self, monkeypatch):
        from inference_optimizer.orchestrator.action_executors import (
            _multi_node_env as mne,
        )

        monkeypatch.setattr(mne, "is_multi_node", lambda: True)
        grid = [
            GridVariant(
                name="mn_only",
                extra_sglang_args="--enable-deepep-moe",
                note="multi_node_only_moe",
            ),
        ]
        kept, dropped = apply_single_node_invalid_variants(grid)
        assert [k.name for k in kept] == ["mn_only"]
        assert dropped == []


# ---------------------------------------------------------------------------
# apply_user_skip_list
# ---------------------------------------------------------------------------

class TestApplyUserSkipList:
    def test_empty_spec_keeps_grid(self):
        grid = [GridVariant(name="a"), GridVariant(name="b")]
        kept, dropped = apply_user_skip_list(grid, skip_spec="")
        assert [k.name for k in kept] == ["a", "b"]
        assert dropped == []

    def test_exact_name_drop(self):
        grid = [GridVariant(name="alpha"), GridVariant(name="beta")]
        kept, dropped = apply_user_skip_list(grid, skip_spec="alpha")
        assert [k.name for k in kept] == ["beta"]
        assert dropped[0]["name"] == "alpha"

    def test_glob_matches_drop(self):
        grid = [
            GridVariant(name="cuda_graph_max_bs_8"),
            GridVariant(name="cuda_graph_max_bs_32"),
            GridVariant(name="schedule_lpm"),
        ]
        kept, dropped = apply_user_skip_list(grid, skip_spec="cuda_graph_*")
        assert [k.name for k in kept] == ["schedule_lpm"]
        assert {d["name"] for d in dropped} == {
            "cuda_graph_max_bs_8", "cuda_graph_max_bs_32",
        }


# ---------------------------------------------------------------------------
# VariantResult.to_dict
# ---------------------------------------------------------------------------

class TestVariantResultToDict:
    def test_succeeded_default_shape(self):
        vr = VariantResult(
            name="v", extra_sglang_args="--foo 1", extra_envs={"A": "1"},
            status="succeeded",
        )
        out = vr.to_dict()
        assert out["status"] == "succeeded"
        # Default ``error_class`` for a success is "" (not None).
        assert out["error_class"] == ""

    def test_failed_round_trip_carries_error_class(self):
        vr = VariantResult(
            name="v", extra_sglang_args="", extra_envs={},
            status="failed", error="boom", error_class="benchmark_report_missing",
        )
        out = vr.to_dict()
        assert out["status"] == "failed"
        assert out["error_class"] == "benchmark_report_missing"
        assert out["error"] == "boom"


# ---------------------------------------------------------------------------
# Sanitisers (re-exposed for tests that don't go through the executor)
# ---------------------------------------------------------------------------

class TestSanitizeScriptName:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("ok.sh", "ok.sh"),
            ("  ok.sh ", "ok.sh"),
            (None, None),
            ("", None),
        ],
    )
    def test_accepts_safe_names(self, raw, expected):
        assert gr.sanitize_script_name(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        ["../danger.sh", "with space.sh", "../sub.sh", "abc.sh; rm -rf /"],
    )
    def test_rejects_unsafe(self, raw):
        with pytest.raises(ValueError):
            gr.sanitize_script_name(raw)


class TestSanitizeResultDir:
    def test_accepts_relative_and_absolute(self):
        assert gr.sanitize_result_dir("runs/x") == "runs/x"
        assert gr.sanitize_result_dir("/workspace/out") == "/workspace/out"

    def test_blank_returns_none(self):
        assert gr.sanitize_result_dir(None) is None
        assert gr.sanitize_result_dir("") is None
        assert gr.sanitize_result_dir("   ") is None

    @pytest.mark.parametrize(
        "raw",
        ["/tmp/with space", "/tmp/leak`whoami`", "/tmp/leak;rm"],
    )
    def test_rejects_unsafe(self, raw):
        with pytest.raises(ValueError):
            gr.sanitize_result_dir(raw)
