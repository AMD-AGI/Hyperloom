"""Regression tests for grouped kernel dispatch and terminal accounting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.state.shared_state import SharedState


@pytest.mark.asyncio
async def test_backend_sequence_stamps_task_group_identity(tmp_path, monkeypatch):
    candidate = {
        "kernel_id": "k002",
        "source_file": "/repo/kernel.py",
        "task_group": {
            "task_group_id": "tg001",
            "primary_kernel_id": "k002",
            "kernel_ids": ["k001", "k002", "k003", "k004"],
            "rows": [],
        },
    }

    async def fake_ladder(*_args, **_kwargs):
        return (
            {
                "status": "ok",
                "kernel_id": "k002",
                "proposal": {"decision": "REVERT"},
                "verification": {"micro_speedup": 0.0},
            },
            [],
        )

    monkeypatch.setattr(krh, "_backend_order", lambda _payload: ["forge"])
    monkeypatch.setattr(krh, "_run_backend_ladder", fake_ladder)

    result = await krh._run_kernel_backend_sequence(
        {},
        candidate,
        session_dir=tmp_path,
    )

    assert result["task_group_id"] == "tg001"
    assert result["task_group_primary_kernel_id"] == "k002"
    assert result["task_group_kernel_ids"] == ["k001", "k002", "k003", "k004"]


def test_record_kernel_opt_mirrors_terminal_attempt_to_group_members():
    state = SharedState()
    result = {
        "status": "ok",
        "kernel_id": "k002",
        "source_file": "/repo/kernel.py",
        "task_group_id": "tg001",
        "task_group_primary_kernel_id": "k002",
        "task_group_kernel_ids": ["k001", "k002", "k003", "k004"],
        "proposal": {"decision": "REVERT", "reasons": ["no improvement"]},
        "verification": {
            "micro_speedup": 0.0,
            "correctness_passed": False,
        },
        "attempts": [],
    }

    state.record_kernel_opt(result)

    assert set(state.kernel_opt_attempts) == {"k001", "k002", "k003", "k004"}
    for kernel_id in ("k001", "k002", "k003", "k004"):
        entry = state.kernel_opt_attempts[kernel_id]
        assert entry["attempts"] == 1
        assert entry["task_group_id"] == "tg001"
        assert entry["task_group_primary_kernel_id"] == "k002"
        assert kernel_id in state.rejected_kernel_ids


@pytest.mark.asyncio
async def test_single_dispatch_serializes_grouped_candidate(tmp_path, monkeypatch):
    candidate = {
        "kernel_id": "k002",
        "name": "scaled_gemm",
        "source_file": "/repo/kernel.py",
        "reusable_native_kernel": True,
        "input_shapes": [{"shape": "(64,5120) fp8"}],
        "task_group": {
            "task_group_id": "tg001",
            "primary_kernel_id": "k002",
            "kernel_ids": ["k001", "k002"],
            "rows": [],
        },
    }
    captured: dict[str, list[str]] = {}

    async def fake_subprocess(cmd, *, timeout_sec):
        captured["cmd"] = cmd
        return 0, json.dumps({"status": "ok", "kernel_id": "k002"}), ""

    monkeypatch.setattr(krh, "_validate_reusable_native_kernel", lambda _payload: None)
    monkeypatch.setattr(
        krh,
        "_validate_kernel_shape_and_paths",
        lambda _payload, session_dir: None,
    )
    monkeypatch.setattr(krh, "_kernel_agent_root_error", lambda: "")
    monkeypatch.setattr(
        krh,
        "_kernel_agent_tool_path",
        lambda _name: Path("/tools/kernel_optimization.py"),
    )
    monkeypatch.setattr(krh, "_run_subprocess", fake_subprocess)

    result = await krh._run_optimization_single(
        {
            "kernel_id": "k002",
            "session_id": "session",
            "candidate": candidate,
            "source_file": "/repo/kernel.py",
            "backends": "forge",
        },
        session_dir=tmp_path,
    )

    cmd = captured["cmd"]
    option_index = cmd.index("--candidate-json")
    candidate_path = Path(cmd[option_index + 1])
    assert candidate_path.is_file()
    written = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert written["task_group"]["task_group_id"] == "tg001"
    assert result["kernel_id"] == "k002"


@pytest.mark.asyncio
async def test_single_candidate_handler_preserves_task_group(tmp_path, monkeypatch):
    candidate = {
        "kernel_id": "k002",
        "source_file": "/repo/kernel.py",
        "task_group": {
            "task_group_id": "tg001",
            "primary_kernel_id": "k002",
            "kernel_ids": ["k001", "k002"],
            "rows": [],
        },
    }
    captured: dict = {}

    async def fake_single(payload, *, session_dir, timeout_override_sec=None):
        captured.update(payload)
        return {"status": "ok", "kernel_id": payload["kernel_id"]}

    monkeypatch.setattr(
        krh,
        "_validate_trace_analyze_inputs",
        lambda _payload, session_dir: None,
    )
    monkeypatch.setattr(
        krh,
        "_batch_kernel_candidates",
        lambda _payload, session_dir: [candidate],
    )
    monkeypatch.setattr(krh, "_run_optimization_single", fake_single)

    result = await krh.run_optimization_handler(
        {"candidates_path": "/tmp/candidates.json"},
        session_dir=tmp_path,
    )

    assert result["kernel_id"] == "k002"
    assert captured["candidate"]["task_group"]["task_group_id"] == "tg001"
    assert captured["source_file"] == "/repo/kernel.py"
