"""Regression tests for grouped kernel dispatch and terminal accounting."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.phases.machine_state import kernel_work_pending
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


@pytest.mark.asyncio
async def test_batch_exception_preserves_task_group_identity(tmp_path, monkeypatch):
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

    async def fail_sequence(*_args, **_kwargs):
        raise RuntimeError("backend crashed")

    monkeypatch.setattr(krh, "_run_kernel_backend_sequence", fail_sequence)

    result = await krh._run_optimization_batch(
        {},
        [candidate],
        session_dir=tmp_path,
    )

    failed = result["batch_results"][0]
    assert failed["error_class"] == "subtask_exception"
    assert failed["task_group_id"] == "tg001"
    assert failed["task_group_kernel_ids"] == ["k001", "k002"]


def test_record_kernel_opt_keeps_one_keyed_group_ledger():
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

    assert set(state.kernel_opt_attempts) == {"k002"}
    entry = state.kernel_opt_attempts["k002"]
    assert entry["attempts"] == 1
    assert entry["task_group_id"] == "tg001"
    assert entry["task_group_primary_kernel_id"] == "k002"
    assert state.rejected_kernel_ids == ["k002"]


def test_grouped_keep_drains_after_one_source_integration():
    state = SharedState()
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k002",
            "source_file": "/repo/kernel.py",
            "task_group_id": "tg001",
            "task_group_primary_kernel_id": "k002",
            "task_group_kernel_ids": ["k001", "k002"],
            "task_group_shape_case_ids": ["case_001", "case_002"],
            "task_group_shape_case_count": 2,
            "proposal": {"decision": "KEEP"},
            "verification": {
                "micro_speedup": 1.1,
                "correctness_passed": True,
            },
            "attempts": [],
        }
    )

    assert state.pending_keep_kernel_ids() == ["k002"]
    state.optimization_stack.append(
        {
            "action": "integrate",
            "kernel_id": "k002",
            "target_file": "/repo/kernel.py",
        }
    )

    assert state.pending_keep_kernel_ids() == []
    assert kernel_work_pending(state) is False


def test_reused_kernel_id_resets_stale_group_ledger_and_rejection():
    state = SharedState()
    base_result = {
        "status": "ok",
        "kernel_id": "k002",
        "source_file": "/repo/shared.py",
        "task_group_id": "tg001",
        "task_group_primary_kernel_id": "k002",
        "task_group_kernel_ids": ["k002"],
        "verification": {"micro_speedup": 0.0},
        "attempts": [],
    }
    state.record_kernel_opt(
        {
            **base_result,
            "task_group_key": "old-task",
            "proposal": {"decision": "REVERT"},
        }
    )
    assert "k002" in state.rejected_kernel_ids

    state.record_kernel_opt(
        {
            **base_result,
            "task_group_key": "new-task",
            "proposal": {"decision": "KEEP"},
            "verification": {"micro_speedup": 1.1},
        }
    )

    entry = state.kernel_opt_attempts["k002"]
    assert entry["task_group_key"] == "new-task"
    assert entry["attempts"] == 1
    assert len(entry["history"]) == 1
    assert "k002" not in state.rejected_kernel_ids
    assert state.pending_keep_kernel_ids() == ["k002"]


def test_reused_kernel_id_ignores_stale_integration_history():
    state = SharedState()
    state.optimization_stack = [
        {
            "action": "integrate",
            "kernel_id": "k002",
            "task_group_key": "old-task",
            "target_file": "/repo/old.py",
        }
    ]
    state.kernel_integrate_attempts = {
        "old-patch": {
            "kernel_id": "k002",
            "task_group_key": "old-task",
            "target_file": "/repo/old.py",
            "attempt_count": 1,
            "last_decision": "KEEP",
        }
    }
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k002",
            "source_file": "/repo/new.py",
            "task_group_id": "tg001",
            "task_group_key": "new-task",
            "task_group_primary_kernel_id": "k002",
            "task_group_kernel_ids": ["k002"],
            "proposal": {"decision": "KEEP"},
            "verification": {"micro_speedup": 1.1},
            "attempts": [],
        }
    )

    assert state.pending_keep_kernel_ids() == ["k002"]
    assert kernel_work_pending(state) is True


def test_completed_group_does_not_rotate_to_an_untried_sibling(tmp_path):
    candidates_path = tmp_path / "candidates.json"
    task_group = {
        "task_group_id": "tg001",
        "primary_kernel_id": "k002",
        "kernel_ids": ["k001", "k002"],
        "rows": [],
    }
    candidates_path.write_text(
        json.dumps(
            {
                "hot_kernels": [
                    {
                        "kernel_id": "k001",
                        "name": "scaled_gemm",
                        "gpu_pct": 10.0,
                        "source_file": "/repo/kernel.py",
                        "reusable_native_kernel": True,
                    },
                    {
                        "kernel_id": "k002",
                        "name": "scaled_gemm",
                        "gpu_pct": 20.0,
                        "source_file": "/repo/kernel.py",
                        "reusable_native_kernel": True,
                    },
                ],
                "reusable_native_kernel_ids": ["k001", "k002"],
                "task_groups": [task_group],
            }
        )
    )
    state = SharedState.load_or_init(tmp_path)
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k002",
            "source_file": "/repo/kernel.py",
            "task_group_id": "tg001",
            "task_group_primary_kernel_id": "k002",
            "task_group_kernel_ids": ["k001", "k002"],
            "proposal": {"decision": "KEEP"},
            "verification": {
                "micro_speedup": 1.1,
                "correctness_passed": True,
            },
            "attempts": [],
        }
    )
    state.save(tmp_path)

    selected = krh._batch_kernel_candidates(
        {"candidates_path": str(candidates_path)},
        session_dir=tmp_path,
    )

    assert selected == []


def test_reused_ordinal_ids_do_not_suppress_a_different_group(tmp_path):
    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "hot_kernels": [
                    {
                        "kernel_id": "k002",
                        "name": "new_operator",
                        "gpu_pct": 20.0,
                        "source_file": "/repo/shared.py",
                        "reusable_native_kernel": True,
                    }
                ],
                "reusable_native_kernel_ids": ["k002"],
                "task_groups": [
                    {
                        "task_group_id": "tg001",
                        "task_group_key": '["py","new_operator","/repo/shared.py","forward"]',
                        "primary_kernel_id": "k002",
                        "kernel_ids": ["k002"],
                        "rows": [],
                    }
                ],
            }
        )
    )
    state = SharedState.load_or_init(tmp_path)
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k002",
            "source_file": "/repo/shared.py",
            "task_group_id": "tg001",
            "task_group_key": '["py","old_operator","/repo/shared.py","forward"]',
            "task_group_primary_kernel_id": "k002",
            "task_group_kernel_ids": ["k002"],
            "proposal": {"decision": "KEEP"},
            "verification": {"micro_speedup": 1.1},
            "attempts": [],
        }
    )
    state.last_trace_analyze = json.loads(candidates_path.read_text())
    state.save(tmp_path)

    assert state.untried_hot_reusable_kernels(
        min_gpu_pct=0.0,
        top_n=10,
    ) == ["k002"]

    selected = krh._batch_kernel_candidates(
        {"candidates_path": str(candidates_path)},
        session_dir=tmp_path,
    )

    assert [candidate["kernel_id"] for candidate in selected] == ["k002"]


def test_stable_group_key_survives_member_id_reranking(tmp_path):
    candidates_path = tmp_path / "candidates.json"
    task_group_key = '["py","operator","/repo/operator.py","forward"]'
    candidates_payload = {
        "hot_kernels": [
            {
                "kernel_id": "k002",
                "name": "operator",
                "gpu_pct": 20.0,
                "source_file": "/repo/operator.py",
                "reusable_native_kernel": True,
            }
        ],
        "reusable_native_kernel_ids": ["k002"],
        "task_groups": [
            {
                "task_group_id": "tg001",
                "task_group_key": task_group_key,
                "primary_kernel_id": "k002",
                "kernel_ids": ["k002"],
                "rows": [],
            }
        ],
    }
    candidates_path.write_text(json.dumps(candidates_payload))
    state = SharedState.load_or_init(tmp_path)
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k009",
            "source_file": "/repo/operator.py",
            "task_group_id": "tg004",
            "task_group_key": task_group_key,
            "task_group_primary_kernel_id": "k009",
            "task_group_kernel_ids": ["k009"],
            "proposal": {"decision": "KEEP"},
            "verification": {"micro_speedup": 1.1},
            "attempts": [],
        }
    )
    state.last_trace_analyze = candidates_payload
    state.save(tmp_path)

    assert state.untried_hot_reusable_kernels(
        min_gpu_pct=0.0,
        top_n=10,
    ) == []
    assert krh._batch_kernel_candidates(
        {"candidates_path": str(candidates_path)},
        session_dir=tmp_path,
    ) == []


def test_group_ledger_migrates_to_reranked_member_id():
    state = SharedState()
    task_group_key = '["py","operator","/repo/operator.py","forward"]'
    first_result = {
        "status": "ok",
        "kernel_id": "k009",
        "source_file": "/repo/operator.py",
        "task_group_id": "tg004",
        "task_group_key": task_group_key,
        "task_group_primary_kernel_id": "k009",
        "task_group_kernel_ids": ["k009"],
        "proposal": {"decision": "PARTIAL"},
        "verification": {"micro_speedup": 1.0},
        "attempts": [],
    }
    state.record_kernel_opt(first_result)
    state.record_kernel_opt(
        {
            **first_result,
            "kernel_id": "k002",
            "task_group_id": "tg001",
            "task_group_primary_kernel_id": "k002",
            "task_group_kernel_ids": ["k002"],
        }
    )

    assert "k009" not in state.kernel_opt_attempts
    assert state.kernel_opt_attempts["k002"]["attempts"] == 2
    assert len(state.kernel_opt_attempts["k002"]["history"]) == 2


def test_group_ledger_migration_preserves_displaced_task():
    state = SharedState()

    def result(kernel_id: str, task_group_key: str) -> dict:
        return {
            "status": "ok",
            "kernel_id": kernel_id,
            "source_file": f"/repo/{task_group_key}.py",
            "task_group_id": f"tg-{task_group_key}",
            "task_group_key": task_group_key,
            "task_group_primary_kernel_id": kernel_id,
            "task_group_kernel_ids": [kernel_id],
            "proposal": {"decision": "PARTIAL"},
            "verification": {"micro_speedup": 1.0},
            "attempts": [],
        }

    state.record_kernel_opt(result("k001", "task-a"))
    state.record_kernel_opt(result("k002", "task-b"))
    state.record_kernel_opt(result("k002", "task-a"))

    assert state.kernel_opt_attempts["k002"]["task_group_key"] == "task-a"
    assert state.kernel_opt_attempts["k001"]["task_group_key"] == "task-b"
    assert state.kernel_opt_attempts["k002"]["attempts"] == 2
    assert state.kernel_opt_attempts["k001"]["attempts"] == 1


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
    assert result["task_group_id"] == "tg001"
    assert result["task_group_kernel_ids"] == ["k001", "k002"]


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
    assert result["task_group_id"] == "tg001"
    assert result["task_group_kernel_ids"] == ["k001", "k002"]
    assert captured["candidate"]["task_group"]["task_group_id"] == "tg001"
    assert captured["source_file"] == "/repo/kernel.py"
