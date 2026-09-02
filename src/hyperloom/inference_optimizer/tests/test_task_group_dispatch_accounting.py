"""Regression tests for grouped kernel dispatch and terminal accounting."""

from __future__ import annotations


from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.phases.machine_state import kernel_work_pending
from hyperloom.orchestrator.state.shared_state import SharedState


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
