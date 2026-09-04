"""Regression tests for grouped kernel dispatch and terminal accounting."""

from __future__ import annotations


import pytest

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
