# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Pytest hooks and shared helpers for the inference_optimizer tests package."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolate_session_layout_env(monkeypatch, tmp_path_factory):
    """N17 isolation: drop the in-process session-dir pin between tests and point MULTI_NODE_STATE_FILE at a missing sentinel so tests run single-node."""
    monkeypatch.delenv("INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR", raising=False)
    monkeypatch.delenv("INFERENCE_OPTIMIZER_SESSION_LAYOUT", raising=False)
    mn_state_sentinel = tmp_path_factory.mktemp("mn_state") / "missing_state.json"
    monkeypatch.setenv("MULTI_NODE_STATE_FILE", str(mn_state_sentinel))
    monkeypatch.delenv("INFERENCE_OPTIMIZER_NODES", raising=False)


@pytest.fixture(autouse=True)
def _clear_kernel_request_handler_caches():
    """Clear ``lru_cache`` state on env-bound helpers between tests."""
    from inference_optimizer.orchestrator import kernel_request_handlers as krh
    krh._default_geak_budget_minutes.cache_clear()
    krh._default_kernel_batch_parallel.cache_clear()


def _bootstrap_kernel_agent_env() -> None:
    """Point HYPERLOOM_KERNEL_AGENT_ROOT at the in-repo kernel-agent checkout."""
    if os.environ.get("HYPERLOOM_KERNEL_AGENT_ROOT"):
        return
    repo = Path(__file__).resolve().parents[2]
    kernel_agent = repo / "kernel-agent"
    if kernel_agent.is_dir():
        os.environ["HYPERLOOM_KERNEL_AGENT_ROOT"] = str(kernel_agent)


_bootstrap_kernel_agent_env()


def seed_target_analysis_marker(session_dir: Path) -> Path:
    """Write a ``no_target_gpu_configured`` marker JSON at the session dir."""
    from inference_optimizer.session_paths import target_baseline_json
    path = target_baseline_json(session_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({
            "status": "skipped",
            "reason": "no_target_gpu_configured",
            "warning": "compare_against_gpu is empty",
        }),
        encoding="utf-8",
    )
    return path
