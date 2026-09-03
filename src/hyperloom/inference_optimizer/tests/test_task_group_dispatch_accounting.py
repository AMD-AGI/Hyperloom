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
    # A grouped REVERT is terminal on the ledger row, not on the shared id set:
    # the synthetic member ids would tombstone the siblings by association.
    assert state.kernel_opt_attempts["k002"]["rejected_reason"] == "revert_decision"
    assert state.kernel_opt_task_attempts["old-task"]["rejected_reason"] == "revert_decision"
    assert state.rejected_kernel_ids == []

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
    assert entry.get("rejected_reason", "") == ""
    assert state.rejected_kernel_ids == []
    assert state.pending_keep_kernel_ids() == ["k002"]


def test_ungrouped_revert_still_tombstones_the_kernel_id():
    """Without a task group there is no sibling to protect, so the id set is terminal."""
    state = SharedState()
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k002",
            "source_file": "/repo/shared.py",
            "task_group_key": "",
            "proposal": {"decision": "REVERT"},
            "verification": {"micro_speedup": 0.0},
            "attempts": [],
        }
    )

    assert state.rejected_kernel_ids == ["k002"]
    assert state.kernel_opt_attempts["k002"]["rejected_reason"] == "revert_decision"


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

    assert (
        state.untried_hot_reusable_kernels(
            min_gpu_pct=0.0,
            top_n=10,
        )
        == []
    )
    assert (
        krh._batch_kernel_candidates(
            {"candidates_path": str(candidates_path)},
            session_dir=tmp_path,
        )
        == []
    )


def test_versioned_group_identity_matches_legacy_ledger_alias(tmp_path):
    candidates_path = tmp_path / "candidates.json"
    legacy_key = '["py","operator","/repo/operator.py","forward"]'
    versioned_key = '{"operation":"operator","source_kind":"py","source_path":"/repo/operator.py","version":2}'
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
                "task_group_key": versioned_key,
                "legacy_task_group_keys": [legacy_key],
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
            "task_group_key": legacy_key,
            "task_group_primary_kernel_id": "k009",
            "task_group_kernel_ids": ["k009"],
            "proposal": {"decision": "KEEP"},
            "verification": {"micro_speedup": 1.1},
            "attempts": [],
        }
    )
    state.save(tmp_path)

    assert (
        krh._batch_kernel_candidates(
            {"candidates_path": str(candidates_path)},
            session_dir=tmp_path,
        )
        == []
    )


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


def test_pending_keep_refreshes_ordinal_after_rerank():
    state = SharedState()
    task_group_key = "stable-task"
    base_result = {
        "status": "ok",
        "source_file": "/repo/operator.py",
        "task_group_id": "tg004",
        "task_group_key": task_group_key,
        "task_group_kernel_ids": ["k009"],
        "proposal": {"decision": "KEEP"},
        "verification": {
            "micro_speedup": 1.2,
            "best_artifact_path": "/artifacts/operator.py",
        },
        "attempts": [],
    }
    state.record_kernel_opt(
        {
            **base_result,
            "kernel_id": "k009",
            "task_group_primary_kernel_id": "k009",
        }
    )
    state.record_kernel_opt(
        {
            **base_result,
            "kernel_id": "k002",
            "task_group_id": "tg001",
            "task_group_primary_kernel_id": "k002",
            "task_group_kernel_ids": ["k002"],
        }
    )

    pending = state.pending_kernel_integration_records()
    assert len(pending) == 1
    assert pending[0]["kernel_id"] == "k002"
    assert state.pending_keep_kernel_ids() == ["k002"]


def test_cross_route_alias_migrates_one_stable_task():
    state = SharedState()
    operator_alias = "operator-v2-without-function"
    base_result = {
        "status": "ok",
        "source_file": "/repo/operator.py",
        "task_group_kernel_ids": ["k009"],
        "proposal": {"decision": "KEEP"},
        "verification": {
            "micro_speedup": 1.2,
            "best_artifact_path": "/artifacts/operator.py",
        },
        "attempts": [],
    }
    state.record_kernel_opt(
        {
            **base_result,
            "kernel_id": "k009",
            "task_group_id": "tg004",
            "task_group_key": "bypass-task-key",
            "legacy_task_group_keys": [operator_alias],
            "identity_route": "bypass",
            "task_group_primary_kernel_id": "k009",
        }
    )
    state.record_kernel_opt(
        {
            **base_result,
            "kernel_id": "k002",
            "task_group_id": "tg001",
            "task_group_key": "skill-task-key",
            "legacy_task_group_keys": [operator_alias],
            "identity_route": "skill",
            "task_group_primary_kernel_id": "k002",
            "task_group_kernel_ids": ["k002"],
        }
    )

    assert set(state.kernel_opt_task_attempts) == {"skill-task-key"}
    assert state.kernel_opt_task_attempts["skill-task-key"]["attempts"] == 2
    pending = state.pending_kernel_integration_records()
    assert len(pending) == 1
    assert pending[0]["kernel_id"] == "k002"


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

    # k002 moved to task-a; the stable entry for task-a now belongs to k002.
    assert state.kernel_opt_attempts["k002"]["task_group_key"] == "task-a"
    assert state.kernel_opt_attempts["k002"]["attempts"] == 2
    # k001 still has its own stable record (task-a was its starting key).
    assert len(state.kernel_opt_task_attempts) >= 2


def test_single_way_ordinal_reuse_preserves_pending_keep(tmp_path):
    state = SharedState.load_or_init(tmp_path)
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k002",
            "source_file": "/repo/old.py",
            "task_group_id": "tg-old",
            "task_group_key": "task-old",
            "task_group_primary_kernel_id": "k002",
            "task_group_kernel_ids": ["k002"],
            "proposal": {"decision": "KEEP"},
            "verification": {
                "micro_speedup": 1.2,
                "best_artifact_path": "/artifacts/old.py",
            },
            "attempts": [],
        }
    )
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k002",
            "source_file": "/repo/new.py",
            "task_group_id": "tg-new",
            "task_group_key": "task-new",
            "task_group_primary_kernel_id": "k002",
            "task_group_kernel_ids": ["k002"],
            "proposal": {"decision": "PARTIAL"},
            "verification": {"micro_speedup": 1.0},
            "attempts": [],
        }
    )
    state.save(tmp_path)

    reloaded = SharedState.load_or_init(tmp_path)
    assert set(reloaded.kernel_opt_task_attempts) == {
        "task-old",
        "task-new",
    }
    pending = reloaded.pending_kernel_integration_records()
    assert len(pending) == 1
    assert pending[0]["task_key"] == "task-old"
    assert pending[0]["artifact_path"] == "/artifacts/old.py"

    resolved, error = krh._resolve_integrate_payload(
        {
            "integration_id": pending[0]["integration_id"],
            "base_tput": 100.0,
        },
        session_dir=tmp_path,
    )

    assert error is None
    assert resolved["kernel_id"] == "k002"
    assert resolved["task_group_key"] == "task-old"
    assert resolved["patch_path"] == "/artifacts/old.py"
    assert resolved["source_file"] == "/repo/old.py"

    reloaded.record_kernel_integrate_result(
        {
            "status": "ok",
            "decision": "KEEP",
            "integration_id": pending[0]["integration_id"],
            "kernel_id": "k002",
            "task_group_key": "task-old",
            "patch_path": "/artifacts/old.py",
            "target_file": "/repo/old.py",
            "gain_pct": 1.5,
        }
    )

    assert reloaded.pending_kernel_integration_records() == []


def test_integrate_rejection_does_not_poison_reused_ordinal(tmp_path):
    state = SharedState.load_or_init(tmp_path)
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k002",
            "source_file": "/repo/old.py",
            "task_group_id": "tg-old",
            "task_group_key": "task-old",
            "task_group_primary_kernel_id": "k002",
            "task_group_kernel_ids": ["k002"],
            "proposal": {"decision": "KEEP"},
            "verification": {
                "micro_speedup": 1.2,
                "best_artifact_path": "/artifacts/old.py",
            },
            "attempts": [],
        }
    )
    pending = state.pending_kernel_integration_records()[0]
    state.record_kernel_integrate_result(
        {
            "status": "ok",
            "decision": "REVERT",
            "integration_id": pending["integration_id"],
            "kernel_id": "k002",
            "task_group_key": "task-old",
            "patch_path": "/artifacts/old.py",
            "target_file": "/repo/old.py",
            "gain_pct": -1.0,
        }
    )
    state.save(tmp_path)
    assert "k002" not in state.rejected_kernel_ids

    candidates_path = tmp_path / "candidates.json"
    candidates_path.write_text(
        json.dumps(
            {
                "hot_kernels": [
                    {
                        "kernel_id": "k002",
                        "name": "new_operator",
                        "gpu_pct": 20.0,
                        "source_file": "/repo/new.py",
                        "reusable_native_kernel": True,
                    }
                ],
                "reusable_native_kernel_ids": ["k002"],
                "task_groups": [
                    {
                        "task_group_id": "tg-new",
                        "task_group_key": "task-new",
                        "primary_kernel_id": "k002",
                        "kernel_ids": ["k002"],
                        "rows": [],
                    }
                ],
            }
        )
    )

    selected = krh._batch_kernel_candidates(
        {"candidates_path": str(candidates_path)},
        session_dir=tmp_path,
    )

    assert [candidate["kernel_id"] for candidate in selected] == ["k002"]


def test_grouped_integrate_revert_clears_kernel_work_pending():
    state = SharedState()
    state.record_kernel_opt(
        {
            "status": "ok",
            "kernel_id": "k002",
            "source_file": "/repo/operator.py",
            "task_group_id": "tg001",
            "task_group_key": "stable-task",
            "task_group_primary_kernel_id": "k002",
            "task_group_kernel_ids": ["k002"],
            "proposal": {"decision": "KEEP"},
            "verification": {
                "micro_speedup": 1.2,
                "best_artifact_path": "/artifacts/operator.py",
            },
            "attempts": [],
        }
    )
    pending = state.pending_kernel_integration_records()[0]

    state.record_kernel_integrate_result(
        {
            "status": "ok",
            "decision": "REVERT",
            "integration_id": pending["integration_id"],
            "kernel_id": "k002",
            "task_group_key": "stable-task",
            "patch_path": "/artifacts/operator.py",
            "target_file": "/repo/operator.py",
            "gain_pct": -1.0,
        }
    )

    assert state.kernel_opt_task_attempts["stable-task"]["integration_status"] == "rejected"
    assert state.kernel_opt_attempts["k002"]["integration_status"] == "rejected"
    assert kernel_work_pending(state) is False


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
        lambda _payload, session_dir, skipped_out=None: [candidate],
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


class TestEnqueueNominatedPatch:
    """A self-nominated fusion sibling becomes a pending integrate record.

    ``enqueue_nominated_patch`` is the fusion-lane analogue of ``_queue_kernel_keep``:
    it writes each nomination sibling as ``status="pending"`` so the shared
    SWEEP-entry drain runs it through the same integrate lane. The record must
    carry the three fusion-specific facts the generic drain cannot infer (env
    flag, keep bar, ``fusion`` action label) and must survive the queue rebuild.
    """

    @staticmethod
    def _patch(name, target, patch_path="", *, env_flag="", micro=1.0, repo="/repo"):
        from hyperloom.orchestrator.kernel.nomination_result import NominatedPatch

        return NominatedPatch(
            kernel_name=name,
            patch_path=patch_path or f"/out/{name}.patch",
            target_file=target,
            kernel_repo=repo,
            snapshot_dir=f"/snap/{name}",
            base_commit="abc123",
            micro_speedup=micro,
            env_flag=env_flag,
        )

    def test_writes_a_pending_record_carrying_the_fusion_facts(self):
        from hyperloom.orchestrator.kernel._kernel_decisions import enqueue_nominated_patch

        state = SharedState()
        record = enqueue_nominated_patch(
            state,
            patch=self._patch("fuse_a", "/repo/a.py", env_flag="ZAYA_FUSED_A ZAYA_EXTRA"),
            keep_threshold_pct=3.0,
        )

        assert record is not None
        assert record["status"] == "pending"
        assert record["source"] == "forge_fusion"
        assert record["action_label"] == "fusion"
        assert record["source_file"] == "/repo/a.py"
        assert record["artifact_path"] == "/out/fuse_a.patch"
        assert record["deploy_repo_root"] == "/repo"
        assert record["fusion_env_flags"] == {"ZAYA_FUSED_A": "1", "ZAYA_EXTRA": "1"}
        assert record["keep_threshold_pct"] == pytest.approx(3.0)
        # Visible to the shared reader the drain uses.
        assert state.pending_kernel_integration_records()[0]["source_file"] == "/repo/a.py"

    def test_a_self_activating_sibling_carries_no_env_flags(self):
        from hyperloom.orchestrator.kernel._kernel_decisions import enqueue_nominated_patch

        state = SharedState()
        record = enqueue_nominated_patch(state, patch=self._patch("cp", "/repo/c.py", env_flag=""))

        assert record["fusion_env_flags"] == {}

    def test_a_sibling_without_an_artifact_or_target_is_refused(self):
        from hyperloom.orchestrator.kernel._kernel_decisions import enqueue_nominated_patch

        state = SharedState()
        assert enqueue_nominated_patch(state, patch=self._patch("x", "/repo/x.py", patch_path="  ")) is None
        assert enqueue_nominated_patch(state, patch=self._patch("y", "")) is None
        assert state.pending_kernel_integrations == {}

    def test_the_record_survives_the_queue_rebuild(self):
        from hyperloom.orchestrator.kernel._kernel_decisions import (
            _ensure_kernel_task_state,
            enqueue_nominated_patch,
        )

        state = SharedState()
        enqueue_nominated_patch(state, patch=self._patch("fuse_a", "/repo/a.py", env_flag="ZAYA_FUSED_A"))
        # A fusion record has no kernel_opt_task_attempts ledger entry; the
        # rebuild must keep it anyway (non-terminal is never evicted).
        _ensure_kernel_task_state(state)

        records = state.pending_kernel_integration_records()
        assert len(records) == 1
        assert records[0]["fusion_env_flags"] == {"ZAYA_FUSED_A": "1"}

    def test_same_source_siblings_collapse_to_the_strongest(self):
        from hyperloom.orchestrator.kernel._kernel_decisions import enqueue_nominated_patch

        state = SharedState()
        enqueue_nominated_patch(state, patch=self._patch("weak", "/repo/a.py", micro=1.1))
        enqueue_nominated_patch(state, patch=self._patch("strong", "/repo/a.py", micro=1.9))

        records = state.pending_kernel_integration_records()
        assert len(records) == 1
        assert records[0]["kernel_id"] == "strong"

    def test_different_source_siblings_stay_independent(self):
        from hyperloom.orchestrator.kernel._kernel_decisions import enqueue_nominated_patch

        state = SharedState()
        enqueue_nominated_patch(state, patch=self._patch("a", "/repo/a.py"))
        enqueue_nominated_patch(state, patch=self._patch("b", "/repo/b.py"))

        assert {r["source_file"] for r in state.pending_kernel_integration_records()} == {
            "/repo/a.py",
            "/repo/b.py",
        }

    def _keep_the_strongest(self, state, source, *, action):
        """KEEP the surviving sibling on ``source`` and lift it onto the stack."""
        (record,) = [r for r in state.pending_kernel_integration_records() if r["source_file"] == source]
        state.record_kernel_integrate_result(
            {
                "status": "complete",
                "decision": "KEEP",
                "kernel_id": record["kernel_id"],
                "integration_id": record["integration_id"],
                "target_file": source,
            }
        )
        state.optimization_stack.append(
            {"action": action, "kernel_id": record["kernel_id"], "target_file": source, "decision": "KEEP"}
        )

    def test_a_kept_fusion_retires_its_same_source_siblings(self):
        """A fusion KEEP overwrites the whole file, so the losing sibling is spent.

        Draining it would re-apply the file over the KEEP and spend another e2e
        measurement on a patch that can no longer be evaluated on its own.
        """
        from hyperloom.orchestrator.kernel._kernel_decisions import enqueue_nominated_patch

        state = SharedState()
        enqueue_nominated_patch(state, patch=self._patch("weak", "/repo/a.py", micro=1.1))
        enqueue_nominated_patch(state, patch=self._patch("strong", "/repo/a.py", micro=1.9))

        self._keep_the_strongest(state, "/repo/a.py", action="fusion")

        assert [r["kernel_id"] for r in state.pending_kernel_integration_records()] == []
        assert state.has_keep_pending_integrate is False
        assert state.next_pending_keep_kernel_id() == ""

    def test_a_kept_fusion_leaves_other_source_files_alone(self):
        """The retirement is per source file, not a blanket drop of the queue."""
        from hyperloom.orchestrator.kernel._kernel_decisions import enqueue_nominated_patch

        state = SharedState()
        enqueue_nominated_patch(state, patch=self._patch("weak", "/repo/a.py", micro=1.1))
        enqueue_nominated_patch(state, patch=self._patch("strong", "/repo/a.py", micro=1.9))
        enqueue_nominated_patch(state, patch=self._patch("elsewhere", "/repo/b.py", micro=1.5))

        self._keep_the_strongest(state, "/repo/a.py", action="fusion")

        assert [r["kernel_id"] for r in state.pending_kernel_integration_records()] == ["elsewhere"]

    def test_a_non_integrating_stack_entry_retires_nothing(self):
        """Only a whole-file kernel overwrite spends a queued patch.

        A framework or explore entry can name the same path without having
        rewritten the kernel, and dropping the queue on it strands real work.
        """
        from hyperloom.orchestrator.kernel._kernel_decisions import enqueue_nominated_patch

        for action in ("explore", "baseline", "specialist", "integrate_patch"):
            state = SharedState()
            enqueue_nominated_patch(state, patch=self._patch("queued", "/repo/a.py", micro=1.4))
            state.optimization_stack.append(
                {"action": action, "kernel_id": "other", "target_file": "/repo/a.py", "decision": "KEEP"}
            )

            assert [r["kernel_id"] for r in state.pending_kernel_integration_records()] == ["queued"], action

    def test_every_integrating_lane_retires_its_same_source_siblings(self):
        """The exclusion follows the whole-file overwrite, not one lane's label."""
        from hyperloom.orchestrator.kernel._kernel_decisions import enqueue_nominated_patch

        for action in ("integrate", "collective", "fusion"):
            state = SharedState()
            enqueue_nominated_patch(state, patch=self._patch("weak", "/repo/a.py", micro=1.1))
            enqueue_nominated_patch(state, patch=self._patch("strong", "/repo/a.py", micro=1.9))

            self._keep_the_strongest(state, "/repo/a.py", action=action)

            assert [r["kernel_id"] for r in state.pending_kernel_integration_records()] == [], action

    def test_the_patch_budget_caps_dispatched_siblings(self, monkeypatch):
        from hyperloom.orchestrator.kernel._kernel_decisions import enqueue_nominated_patch

        monkeypatch.setenv("HL_KERNEL_PATCH_BUDGET", "2")
        state = SharedState()
        for i in range(4):
            enqueue_nominated_patch(state, patch=self._patch(f"k{i}", f"/repo/f{i}.py", micro=1.0 + i))

        # All four are queued (deferred, not dropped); the reader caps dispatch.
        assert len(state.pending_kernel_integrations) == 4
        assert len(state.pending_kernel_integration_records()) == 2

    def test_re_enqueue_is_idempotent_and_refreshes_the_fusion_facts(self):
        from hyperloom.orchestrator.kernel._kernel_decisions import enqueue_nominated_patch

        state = SharedState()
        enqueue_nominated_patch(state, patch=self._patch("a", "/repo/a.py", env_flag="OLD"), keep_threshold_pct=3.0)
        enqueue_nominated_patch(state, patch=self._patch("a", "/repo/a.py", env_flag="NEW"), keep_threshold_pct=5.0)

        assert len(state.pending_kernel_integrations) == 1
        record = next(iter(state.pending_kernel_integrations.values()))
        assert record["fusion_env_flags"] == {"NEW": "1"}
        assert record["keep_threshold_pct"] == pytest.approx(5.0)
