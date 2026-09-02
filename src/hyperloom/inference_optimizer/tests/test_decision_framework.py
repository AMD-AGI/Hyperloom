# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decision-framework regression tests for kernel state writeback."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from hyperloom.orchestrator.roles import (
    MockBackend,
    ScriptedPlan,
)
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.inference_optimizer.session.paths import make_session_dir


def _heartbeat() -> Intent:
    return Intent(type=IntentType.SEND_MESSAGE, payload={"topic": "heartbeat", "body_md": "ok"})


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(turns=[], default_intent=_heartbeat())
    return {
        "orchestration": MockBackend(silent, name="o"),
        "critic": MockBackend(silent, name="c"),
        "robustness": MockBackend(silent, name="r"),
    }


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    monkeypatch.setenv("KERNEL_OPT_BACKEND_ORDER", "forge")
    # Pin the kernel-agent root so request handlers resolve from disk.
    kernel_agent_root = Path(__file__).resolve().parents[4] / "src" / "hyperloom" / "agents" / "kernel"
    monkeypatch.setenv("HYPERLOOM_KERNEL_AGENT_ROOT", str(kernel_agent_root))
    # Stub the interpreter resolver to avoid a real Magpie import probe.
    monkeypatch.setenv("MAGPIE_PYTHON", "/usr/bin/python3")
    from hyperloom.orchestrator.actions.executors import _grid_runner

    monkeypatch.setattr(
        _grid_runner,
        "_resolve_magpie_python",
        lambda: "/usr/bin/python3",
    )
    return make_session_dir()


# C — kernel-opt response recorded to SharedState


# D — native-only guard for kernel optimization handler


# E — record_kernel_opt retires kernels stuck in PARTIAL.
def _partial_kernel_opt_result(kernel_id: str, decision: str = "PARTIAL") -> dict[str, Any]:
    return {
        "status": "ok",
        "kernel_id": kernel_id,
        "proposal": {"decision": decision, "reasons": ["no measurable speedup found"]},
        "verification": {
            "compile_passed": True,
            "correctness_passed": False,
            "micro_speedup": 1.0,
            "best_artifact_path": f"/tmp/{kernel_id}.hip",
        },
    }


def test_record_kernel_opt_first_partial_does_not_retire():
    state = SharedState()
    state.record_kernel_opt(_partial_kernel_opt_result("k001"))
    assert "k001" not in state.rejected_kernel_ids
    assert state.kernel_opt_attempts["k001"]["partial_count"] == 1
    assert state.kernel_opt_attempts["k001"]["attempts"] == 1
    assert "rejected_reason" not in state.kernel_opt_attempts["k001"]


def test_record_kernel_opt_retires_after_max_partial_attempts():
    state = SharedState()
    # Default max_partial = 2 → second PARTIAL must retire.
    state.record_kernel_opt(_partial_kernel_opt_result("k001"))
    state.record_kernel_opt(_partial_kernel_opt_result("k001"))
    assert "k001" in state.rejected_kernel_ids
    entry = state.kernel_opt_attempts["k001"]
    assert entry["partial_count"] == 2
    assert entry["attempts"] == 2
    assert "max_partial_attempts_2_without_keep" == entry["rejected_reason"]


def test_record_kernel_opt_revert_retires_immediately():
    state = SharedState()
    state.record_kernel_opt(_partial_kernel_opt_result("k002", decision="REVERT"))
    assert "k002" in state.rejected_kernel_ids
    assert state.kernel_opt_attempts["k002"]["rejected_reason"] == "revert_decision"
    # partial_count is not bumped for REVERT (it's a different terminal state).
    assert state.kernel_opt_attempts["k002"].get("partial_count", 0) == 0


def test_record_kernel_opt_keep_resets_partial_streak():
    state = SharedState()
    state.record_kernel_opt(_partial_kernel_opt_result("k003"))
    assert state.kernel_opt_attempts["k003"]["partial_count"] == 1
    keep = _partial_kernel_opt_result("k003", decision="KEEP")
    state.record_kernel_opt(keep)
    assert state.kernel_opt_attempts["k003"]["partial_count"] == 0
    assert "k003" not in state.rejected_kernel_ids
    assert state.kernel_opt_attempts["k003"]["last_decision"] == "KEEP"


def test_record_kernel_opt_max_partial_env_override(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL", "3")
    state = SharedState()
    state.record_kernel_opt(_partial_kernel_opt_result("k004"))
    state.record_kernel_opt(_partial_kernel_opt_result("k004"))
    assert "k004" not in state.rejected_kernel_ids
    state.record_kernel_opt(_partial_kernel_opt_result("k004"))
    assert "k004" in state.rejected_kernel_ids
    assert state.kernel_opt_attempts["k004"]["rejected_reason"] == "max_partial_attempts_3_without_keep"


def test_record_kernel_opt_history_capped_at_ten():
    state = SharedState()
    for _ in range(15):
        state.record_kernel_opt(_partial_kernel_opt_result("k005"))
    history = state.kernel_opt_attempts["k005"]["history"]
    assert len(history) == 10
    assert state.kernel_opt_attempts["k005"]["attempts"] == 15
    assert state.kernel_opt_attempts["k005"]["partial_count"] == 15
    assert "k005" in state.rejected_kernel_ids


def test_record_kernel_opt_persists_attempts_across_reload(tmp_path):
    state = SharedState()
    state.record_kernel_opt(_partial_kernel_opt_result("k006"))
    state.save(tmp_path)
    reloaded = SharedState.load_or_init(tmp_path)
    assert reloaded.kernel_opt_attempts["k006"]["partial_count"] == 1
    assert reloaded.kernel_opt_attempts["k006"]["attempts"] == 1


def test_record_kernel_opt_prompt_summary_surfaces_history():
    state = SharedState()
    state.record_kernel_opt(_partial_kernel_opt_result("k007"))
    state.record_kernel_opt(_partial_kernel_opt_result("k007"))
    summary = state.to_prompt_summary()
    assert "kernel_id=k007" in summary
    assert "history=attempts=2/partial=2" in summary
    assert "retired=max_partial_attempts_2_without_keep" in summary
