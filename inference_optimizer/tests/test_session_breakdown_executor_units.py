# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for ``orchestrator.action_executors.session_breakdown``."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator.action_executors import session_breakdown as sb


# Helpers
def _ctx(*, params: dict | None = None, extra: dict | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        task=SimpleNamespace(task_id="sb-t1", params=params or {}),
        extra=extra or {},
    )


@pytest.fixture
def session_dir(tmp_path: Path) -> Path:
    (tmp_path / "manifest.json").write_text(json.dumps({"session": "x"}))
    return tmp_path


# _resolve_session_dir
class TestResolveSessionDir:
    def test_extra_session_dir_wins(self, tmp_path):
        ctx = _ctx(extra={"session_dir": str(tmp_path)})
        assert sb.SessionBreakdownExecutor._resolve_session_dir(ctx) == tmp_path

    def test_params_session_dir_used(self, tmp_path):
        ctx = _ctx(params={"session_dir": str(tmp_path)})
        assert sb.SessionBreakdownExecutor._resolve_session_dir(ctx) == tmp_path

    def test_fallback_path_returned_when_manifest_present(
        self, session_dir, monkeypatch,
    ):
        monkeypatch.setattr(
            "inference_optimizer.paths.session_dir", lambda: session_dir,
        )
        ctx = _ctx()
        assert (
            sb.SessionBreakdownExecutor._resolve_session_dir(ctx) == session_dir
        )

    def test_fallback_returns_none_without_manifest(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(
            "inference_optimizer.paths.session_dir", lambda: tmp_path,
        )
        ctx = _ctx()
        assert sb.SessionBreakdownExecutor._resolve_session_dir(ctx) is None


# Execution branches
class TestExecutor:
    @pytest.mark.asyncio
    async def test_failed_when_no_session_dir(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            "inference_optimizer.paths.session_dir", lambda: tmp_path,
        )
        result = await sb.SessionBreakdownExecutor()(_ctx())
        assert result["status"] == "failed"
        assert "session_dir" in result["error"]

    @pytest.mark.asyncio
    async def test_returns_failed_when_writer_raises(self, monkeypatch, session_dir):
        def boom(*args, **kwargs):
            raise RuntimeError("kaboom")

        monkeypatch.setattr(
            "inference_optimizer.breakdown.write_breakdown_json", boom,
        )
        monkeypatch.setattr(
            "inference_optimizer.breakdown.build", lambda *a, **k: {"warnings": []},
        )
        result = await sb.SessionBreakdownExecutor()(
            _ctx(extra={"session_dir": str(session_dir)}),
        )
        assert result["status"] == "failed"
        assert "RuntimeError" in result["error"]

    @pytest.mark.asyncio
    async def test_succeeded_returns_breakdown_metadata(
        self, monkeypatch, session_dir,
    ):
        target = session_dir / "session_breakdown.json"
        target.write_text(json.dumps({"warnings": ["w1"]}))

        def fake_writer(_sd, *, output_path=None):
            return target

        monkeypatch.setattr(
            "inference_optimizer.breakdown.write_breakdown_json", fake_writer,
        )
        monkeypatch.setattr(
            "inference_optimizer.breakdown.build",
            lambda *a, **k: {"warnings": ["w1"]},
        )
        ctx = _ctx(extra={"session_dir": str(session_dir)})
        result = await sb.SessionBreakdownExecutor()(ctx)
        assert result["status"] == "succeeded"
        assert result["breakdown_path"] == str(target)
        assert result["warnings"] == ["w1"]
        assert result["size_bytes"] > 0

    @pytest.mark.asyncio
    async def test_build_exception_yields_empty_warnings(
        self, monkeypatch, session_dir,
    ):
        target = session_dir / "session_breakdown.json"
        target.write_text("{}")

        monkeypatch.setattr(
            "inference_optimizer.breakdown.write_breakdown_json",
            lambda *a, **k: target,
        )

        def raise_build(*args, **kwargs):
            raise RuntimeError("build failure")

        monkeypatch.setattr(
            "inference_optimizer.breakdown.build", raise_build,
        )
        ctx = _ctx(extra={"session_dir": str(session_dir)})
        result = await sb.SessionBreakdownExecutor()(ctx)
        assert result["status"] == "succeeded"
        assert result["warnings"] == []
