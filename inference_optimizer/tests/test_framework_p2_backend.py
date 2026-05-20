"""FrameworkAgentBackend subprocess bridge + P2 real handler tests."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.backends.framework_agent import (
    FrameworkAgentBackend,
)
from inference_optimizer.orchestrator.framework_request_handlers import (
    framework_optimize_handler,
    set_framework_backend,
)


_FA_ROOT = (Path(__file__).resolve().parents[2] / "framework-agent")
_FIXTURES = _FA_ROOT / "tests/agent/fixtures"


@pytest.fixture(autouse=True)
def _reset_framework_backend():
    """Reset the module-level backend singleton between tests so the
    P1 mock branch / P2 real branch don't leak between cases."""
    set_framework_backend(None)
    yield
    set_framework_backend(None)


# ---------------------------------------------------------------------------
# FrameworkAgentBackend.run_optimize -- real subprocess bridge
# ---------------------------------------------------------------------------
def test_backend_run_optimize_with_sglang_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """End-to-end: backend writes task.json -> prepare-task -> AST scan
    -> commit-result on the synthesized envelope. Returns
    OptimizeSuccess with non-empty discovered_flags."""
    monkeypatch.setenv("SGLANG_SOURCE_ROOT", str(_FIXTURES / "mini_sglang"))
    backend = FrameworkAgentBackend(
        framework_agent_root=_FA_ROOT,
        session_dir=tmp_path,
    )
    envelope = backend.run_optimize(
        session_dir=str(tmp_path),
        target_framework="sglang",
        ast_scan_enabled=True,
        ast_frameworks=("sglang",),
    )
    assert envelope["payload_kind"] == "OptimizeSuccess"
    assert envelope["target_framework"] == "sglang"
    # Stage A elapsed ms is recorded.
    assert envelope["stage_a_elapsed_ms"] >= 0
    # AST scan surfaced sglang CLI flags from the fixture.
    flags = envelope["discovered_flags"].get("sglang") or []
    assert "--max-running-requests" in flags
    assert "--cuda-graph-max-bs" in flags


def test_backend_run_optimize_writes_task_and_envelope_to_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("SGLANG_SOURCE_ROOT", str(_FIXTURES / "mini_sglang"))
    backend = FrameworkAgentBackend(
        framework_agent_root=_FA_ROOT,
        session_dir=tmp_path,
    )
    backend.run_optimize(
        session_dir=str(tmp_path),
        target_framework="sglang",
    )
    fw_dir = tmp_path / "runs" / "framework"
    assert fw_dir.is_dir()
    task_dirs = list(fw_dir.iterdir())
    assert len(task_dirs) == 1
    task_dir = task_dirs[0]
    assert task_dir.name.startswith("fw-")
    task = json.loads((task_dir / "task.json").read_text())
    assert task["kind"] == "framework_optimize"
    assert task["target_framework"] == "sglang"
    assert (task_dir / "envelope.json").is_file()


def test_backend_run_optimize_returns_failure_when_source_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """No SGLANG_SOURCE_ROOT env, no /sgl-workspace, no installed
    sglang -> source_not_found OptimizeFailure."""
    monkeypatch.delenv("SGLANG_SOURCE_ROOT", raising=False)
    monkeypatch.delenv("VLLM_SOURCE_ROOT", raising=False)

    # Make container path miss + find_spec miss for sglang.
    import framework_agent.agent.source_resolver as sr

    def fake_find_spec(name: str):
        return None

    monkeypatch.setattr(sr.importlib.util, "find_spec", fake_find_spec)
    monkeypatch.setattr(
        sr,
        "_probe_one",
        lambda fw: None,  # type: ignore[arg-type]
    )

    backend = FrameworkAgentBackend(
        framework_agent_root=_FA_ROOT,
        session_dir=tmp_path,
    )
    envelope = backend.run_optimize(
        session_dir=str(tmp_path),
        target_framework="sglang",
        ast_scan_enabled=True,
        ast_frameworks=("sglang",),
    )
    assert envelope["payload_kind"] == "OptimizeFailure"
    assert envelope["reason"] == "source_not_found"


def test_backend_run_optimize_skips_ast_when_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """ast_scan_enabled=False -> discovered_flags empty, no scan run."""
    backend = FrameworkAgentBackend(
        framework_agent_root=_FA_ROOT,
        session_dir=tmp_path,
    )
    envelope = backend.run_optimize(
        session_dir=str(tmp_path),
        target_framework="sglang",
        ast_scan_enabled=False,
    )
    assert envelope["payload_kind"] == "OptimizeSuccess"
    assert envelope["discovered_flags"] == {}
    assert "(skipped)" in envelope["rationale"]


def test_backend_run_integrate_raises_until_pr_h(tmp_path: Path):
    backend = FrameworkAgentBackend(
        framework_agent_root=_FA_ROOT,
        session_dir=tmp_path,
    )
    with pytest.raises(NotImplementedError, match="PR-H"):
        backend.run_integrate(
            session_dir=str(tmp_path),
            patch_path="/tmp/p.diff",
            patch_id="fw-x",
        )


# ---------------------------------------------------------------------------
# Handler: P1 mock path vs P2 real path selection
# ---------------------------------------------------------------------------
def _run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


def test_handler_falls_back_to_p1_mock_when_no_backend_injected(tmp_path: Path):
    """No set_framework_backend() call -> P1 mock canned envelope."""
    result = _run(framework_optimize_handler(
        {"target_framework": "sglang"},
        session_dir=tmp_path,
    ))
    assert result["payload_kind"] == "OptimizeSuccess"
    assert result["rationale"].startswith("P1 mock")
    assert result["predicted_gain_pct"] == 5.0


def test_handler_invokes_real_backend_when_injected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
):
    """When a real backend is registered, handler delegates to
    run_optimize and returns the synthesized envelope."""
    monkeypatch.setenv("SGLANG_SOURCE_ROOT", str(_FIXTURES / "mini_sglang"))
    backend = FrameworkAgentBackend(
        framework_agent_root=_FA_ROOT,
        session_dir=tmp_path,
    )
    set_framework_backend(backend)
    result = _run(framework_optimize_handler(
        {
            "target_framework": "sglang",
            "ast_scan_enabled": True,
            "ast_frameworks": ("sglang",),
        },
        session_dir=tmp_path,
    ))
    assert result["payload_kind"] == "OptimizeSuccess"
    assert result["status"] == "succeeded"
    # Real path produced flags from the fixture.
    flags = result["discovered_flags"].get("sglang") or []
    assert "--max-running-requests" in flags


def test_handler_returns_failure_envelope_on_backend_exception(
    tmp_path: Path,
):
    """Backend raises -> handler returns OptimizeFailure (not crash)."""

    class _CrashBackend:
        def run_optimize(self, **kwargs: Any) -> dict[str, Any]:
            raise RuntimeError("oops")

    set_framework_backend(_CrashBackend())
    result = _run(framework_optimize_handler(
        {"target_framework": "sglang"},
        session_dir=tmp_path,
    ))
    assert result["status"] == "failed"
    assert result["payload_kind"] == "OptimizeFailure"
    assert result["reason"] == "handler_exception"
    assert "oops" in result["detail"]


def test_backend_run_returns_empty_turn_result(tmp_path: Path):
    """Backend.run() heartbeat path -- never carries real traffic."""
    backend = FrameworkAgentBackend(
        framework_agent_root=_FA_ROOT,
        session_dir=tmp_path,
    )
    result = _run(backend.run("anything"))
    assert result.intents == []
    assert result.raw_text == ""
