"""Regression tests for grouped kernel dispatch and terminal accounting."""

from __future__ import annotations


import pytest

from hyperloom.orchestrator.state.shared_state import SharedState


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
