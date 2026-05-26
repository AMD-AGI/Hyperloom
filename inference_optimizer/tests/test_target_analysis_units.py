"""Unit tests for ``orchestrator.action_executors.target_analysis``.

The full-flow tests live in ``test_target_analysis_executor`` and exercise
the success path with a stubbed analyzer. Here we cover the small helper
utilities (env coercion, ctx fallbacks) and the failure branches so the
"never fail" guarantee for the runner stays locked.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors import target_analysis as ta


# ---------------------------------------------------------------------------
# env helpers
# ---------------------------------------------------------------------------

class TestEnvHelpers:
    def test_env_int_uses_default_when_missing(self, monkeypatch):
        monkeypatch.delenv("TARGET_INT_TEST", raising=False)
        assert ta._env_int("TARGET_INT_TEST", default=7) == 7

    def test_env_int_parses_valid(self, monkeypatch):
        monkeypatch.setenv("TARGET_INT_TEST", "42")
        assert ta._env_int("TARGET_INT_TEST") == 42

    def test_env_int_falls_back_on_invalid(self, monkeypatch):
        monkeypatch.setenv("TARGET_INT_TEST", "garbage")
        assert ta._env_int("TARGET_INT_TEST", default=3) == 3

    def test_env_str_strips_whitespace(self, monkeypatch):
        monkeypatch.setenv("TARGET_STR_TEST", "  value  ")
        assert ta._env_str("TARGET_STR_TEST") == "value"


# ---------------------------------------------------------------------------
# session_dir resolution
# ---------------------------------------------------------------------------

class _DummySummary:
    status = "ok"
    reason = ""
    warning = ""
    row_count = 3
    best = SimpleNamespace(tput_per_gpu=10.0, conc=4, decode_tp=2)


def _ctx(*, params: dict | None = None, extra: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        task=SimpleNamespace(task_id="ta-t1", kind="target_analysis",
                             params=params or {}),
        extra=extra or {},
    )


class TestResolveSessionDir:
    def test_extra_session_dir_wins(self, tmp_path):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        ctx = _ctx(extra={"session_dir": str(tmp_path)})
        assert ex._resolve_session_dir(ctx) == tmp_path

    def test_params_session_dir_used(self, tmp_path):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        ctx = _ctx(params={"session_dir": str(tmp_path)})
        assert ex._resolve_session_dir(ctx) == tmp_path

    def test_constructor_session_dir_used(self, tmp_path):
        ex = ta.TargetAnalysisExecutor(
            compare_against_gpu="MI300X", session_dir=tmp_path,
        )
        ctx = _ctx()
        assert ex._resolve_session_dir(ctx) == tmp_path

    def test_falls_back_to_paths_session_dir(self, tmp_path, monkeypatch):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        monkeypatch.setattr(
            "inference_optimizer.paths.session_dir", lambda: tmp_path,
        )
        ctx = _ctx()
        assert ex._resolve_session_dir(ctx) == tmp_path

    def test_returns_none_when_fallback_missing(self, monkeypatch):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")

        def boom():
            raise RuntimeError("no session")

        monkeypatch.setattr(
            "inference_optimizer.paths.session_dir", boom,
        )
        assert ex._resolve_session_dir(_ctx()) is None


# ---------------------------------------------------------------------------
# Execution branches
# ---------------------------------------------------------------------------

class TestExecutor:
    @pytest.mark.asyncio
    async def test_skipped_when_no_session_dir(self, monkeypatch):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        monkeypatch.setattr(ex, "_resolve_session_dir", lambda ctx: None)
        result = await ex(_ctx())
        assert result["status"] == "succeeded"
        assert result["baseline_status"] == "skipped"
        assert result["reason"] == "no_session_dir"

    @pytest.mark.asyncio
    async def test_writes_skipped_summary_when_no_gpu(self, tmp_path, monkeypatch):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="")
        monkeypatch.setattr(ex, "_resolve_session_dir", lambda ctx: tmp_path)
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.target_analysis.analyze",
            lambda **kwargs: _DummySummary(),
        )
        result = await ex(_ctx())
        assert result["status"] == "succeeded"
        assert result["baseline_status"] == "ok"
        # Best fields propagate through.
        assert result["best_tput_per_gpu"] == 10.0

    @pytest.mark.asyncio
    async def test_analyzer_crash_is_swallowed(self, tmp_path, monkeypatch):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        monkeypatch.setattr(ex, "_resolve_session_dir", lambda ctx: tmp_path)

        def boom(**_):
            raise RuntimeError("InferenceX 500")

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.target_analysis.analyze",
            boom,
        )
        result = await ex(_ctx())
        assert result["status"] == "succeeded"
        assert result["baseline_status"] == "fetch_error"
        assert "analyzer crashed" in result["note"]

    @pytest.mark.asyncio
    async def test_analyzer_crash_in_no_gpu_branch_is_swallowed(
        self, tmp_path, monkeypatch,
    ):
        ex = ta.TargetAnalysisExecutor(compare_against_gpu="")
        monkeypatch.setattr(ex, "_resolve_session_dir", lambda ctx: tmp_path)

        def boom(**_):
            raise RuntimeError("nope")

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.target_analysis.analyze",
            boom,
        )
        result = await ex(_ctx())
        assert result["baseline_status"] == "fetch_error"

    @pytest.mark.asyncio
    async def test_format_result_uses_summary_without_best(
        self, tmp_path, monkeypatch,
    ):
        class _NoBestSummary:
            status = "no_data"
            reason = "row_count==0"
            warning = "filtered_to_empty"
            row_count = 0
            best = None

        ex = ta.TargetAnalysisExecutor(compare_against_gpu="MI300X")
        monkeypatch.setattr(ex, "_resolve_session_dir", lambda ctx: tmp_path)
        monkeypatch.setattr(
            "inference_optimizer.orchestrator.action_executors.target_analysis.analyze",
            lambda **kwargs: _NoBestSummary(),
        )
        result = await ex(_ctx(params={"model_path": "/m"}))
        assert result["baseline_status"] == "no_data"
        # Best metrics are absent when ``best`` is None.
        assert "best_tput_per_gpu" not in result
