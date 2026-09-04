# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``_fill_integrate_defaults_from_state`` + integrate_handler defaulting (base_tput/config_path/extra_server_args from SharedState)."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel import request_handlers as krh
from hyperloom.orchestrator.state.shared_state import SharedState
from hyperloom.inference_optimizer.session.paths import make_session_dir


@pytest.fixture
def session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


def _seed_state(
    session_dir: Path,
    *,
    baseline_tput: float = 0.0,
    baseline_config_path: str = "",
    current_best_args: str = "",
    current_best_envs: dict[str, str] | None = None,
) -> SharedState:
    state = SharedState.load_or_init(session_dir)
    state.baseline_tput = baseline_tput
    state.baseline_config_path = baseline_config_path
    if current_best_args or current_best_envs:
        state.current_best = {
            "action": "kernel_opt",
            "tput": 900.0,
            "extra_server_args": current_best_args,
            "extra_envs": dict(current_best_envs or {}),
        }
    state.save(session_dir)
    return state


class TestFillIntegrateDefaultsFromState:
    def test_all_three_defaults_fired(self, session_dir):
        _seed_state(
            session_dir,
            baseline_tput=800.0,
            baseline_config_path="/tmp/base.yaml",
            current_best_args="--page-size 16",
        )

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc"},
            session_dir=session_dir,
        )

        # base_tput prefers current_best.tput (900) over the raw baseline (800):
        # the candidate must beat the recipe it stacks onto, not just baseline.
        assert out["base_tput"] == 900.0
        assert out["config_path"] == "/tmp/base.yaml"
        assert out["extra_server_args"] == "--page-size 16"
        assert out["kernel_id"] == "k_abc"

    def test_payload_base_tput_wins(self, session_dir):
        _seed_state(session_dir, baseline_tput=800.0)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc", "base_tput": 999.0},
            session_dir=session_dir,
        )

        assert out["base_tput"] == 999.0

    def test_payload_config_path_wins(self, session_dir):
        _seed_state(
            session_dir,
            baseline_tput=800.0,
            baseline_config_path="/tmp/state.yaml",
        )

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc", "config_path": "/tmp/explicit.yaml"},
            session_dir=session_dir,
        )

        assert out["config_path"] == "/tmp/explicit.yaml"

    def test_payload_extra_args_wins(self, session_dir):
        _seed_state(session_dir, current_best_args="--from-state")

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc", "extra_server_args": "--from-payload"},
            session_dir=session_dir,
        )

        assert out["extra_server_args"] == "--from-payload"

    def test_current_best_envs_are_preserved_and_payload_wins(self, session_dir):
        _seed_state(
            session_dir,
            current_best_envs={
                "VLLM_ROCM_USE_AITER": "1",
                "SHARED": "state",
            },
        )

        out = krh._fill_integrate_defaults_from_state(
            {
                "kernel_id": "k_abc",
                "extra_envs": {
                    "CANDIDATE_ONLY": "1",
                    "SHARED": "candidate",
                },
            },
            session_dir=session_dir,
        )

        assert out["extra_envs"] == {
            "VLLM_ROCM_USE_AITER": "1",
            "CANDIDATE_ONLY": "1",
            "SHARED": "candidate",
        }

    def test_fusion_record_activates_the_env_flag_and_keep_bar(self, session_dir):
        """A fusion sibling's env flag + keep bar are folded in from its record.

        The drain builds a bare ``{kernel_id, integration_id, ...}`` payload; the
        fused path is inert until its env flag is set and fusion keeps its own
        keep bar, so both have to be pulled from the pending record here or
        integrate measures the un-fused path against the wrong threshold.
        """
        from hyperloom.orchestrator.kernel._kernel_decisions import enqueue_nominated_patch
        from hyperloom.orchestrator.kernel.nomination_result import NominatedPatch

        state = _seed_state(session_dir, current_best_envs={"KEEP_ME": "1"})
        record = enqueue_nominated_patch(
            state,
            patch=NominatedPatch(
                kernel_name="fuse_a",
                patch_path="/out/fuse_a.patch",
                target_file="/repo/a.py",
                env_flag="ZAYA_FUSED_A",
            ),
            keep_threshold_pct=3.0,
        )
        state.save(session_dir)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "fuse_a", "integration_id": record["integration_id"]},
            session_dir=session_dir,
        )

        # current_best env survives AND the fused path is now active.
        assert out["extra_envs"] == {"KEEP_ME": "1", "ZAYA_FUSED_A": "1"}
        assert out["keep_threshold_pct"] == pytest.approx(3.0)
        assert out["source"] == "forge_fusion"
        assert out["action_label"] == "fusion"

    def test_a_non_fusion_record_adds_no_fusion_fields(self, session_dir):
        """The fusion folding is opt-in on ``source='forge_fusion'``."""
        _seed_state(session_dir, baseline_tput=800.0)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc"},
            session_dir=session_dir,
        )

        assert "keep_threshold_pct" not in out
        assert out.get("source") != "forge_fusion"

    def test_empty_state_no_op(self, session_dir):
        _seed_state(session_dir)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc"},
            session_dir=session_dir,
        )

        assert "base_tput" not in out or out["base_tput"] in (0.0, 0)
        assert not out.get("config_path")
        assert not out.get("extra_server_args")

    def test_returns_shallow_copy_not_mutating_input(self, session_dir):
        _seed_state(session_dir, baseline_tput=800.0)

        payload = {"kernel_id": "k_abc"}
        out = krh._fill_integrate_defaults_from_state(
            payload,
            session_dir=session_dir,
        )

        assert "base_tput" not in payload
        assert out["base_tput"] == 800.0

    def test_zero_base_tput_in_payload_triggers_fallback(self, session_dir):
        _seed_state(session_dir, baseline_tput=800.0)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc", "base_tput": 0.0},
            session_dir=session_dir,
        )

        assert out["base_tput"] == 800.0

    def test_zero_state_does_not_overwrite_explicit_payload(self, session_dir):
        _seed_state(session_dir, baseline_tput=0.0)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc", "base_tput": 750.0},
            session_dir=session_dir,
        )

        assert out["base_tput"] == 750.0

    def test_base_tput_prefers_current_best_over_baseline(self, session_dir):
        """A candidate must be judged against the CURRENT BEST recipe it stacks
        onto (current_best.tput), not the raw baseline.

        Otherwise a kernel/fusion that beats baseline but regresses vs the
        established best (e.g. a warm-replay recipe) is wrongly KEEP'd instead of
        REVERT'd (observed: forge_fusion adopted at negative gain vs current_best
        while still positive vs baseline, dragging the final recipe down).
        """
        _seed_state(
            session_dir,
            baseline_tput=800.0,
            current_best_args="--page-size 16",  # _seed_state sets current_best.tput=900
        )

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc"},
            session_dir=session_dir,
        )

        assert out["base_tput"] == 900.0  # current_best, NOT baseline 800

    def test_base_tput_falls_back_to_baseline_without_current_best(self, session_dir):
        # No current_best recorded yet (early kernel phase) -> baseline is the
        # only reference available.
        _seed_state(session_dir, baseline_tput=800.0)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k_abc"},
            session_dir=session_dir,
        )

        assert out["base_tput"] == 800.0


class TestBareKernelIdMustNotGuessBetweenSiblings:
    """One ``kernel_id`` can name several pending siblings.

    ``kernel_id`` comes from the patch's ``kernel_name``, so a nomination round
    that authored two patches against different files queues two pending records
    under one ``kernel_id``. Resolving a bare ``{kernel_id}`` payload to whichever
    sibling sorts first would stamp the KEEP/REVERT verdict onto a record the
    caller never named, and ``record_kernel_integrate_result`` cannot catch it
    because this filler hands it a fully-populated ``integration_id``.
    """

    @staticmethod
    def _enqueue(state, *, kernel_name: str, target_file: str, micro_speedup: float) -> dict:
        from hyperloom.orchestrator.kernel._kernel_decisions import enqueue_nominated_patch
        from hyperloom.orchestrator.kernel.nomination_result import NominatedPatch

        record = enqueue_nominated_patch(
            state,
            patch=NominatedPatch(
                kernel_name=kernel_name,
                patch_path=f"/out/{Path(target_file).stem}.patch",
                target_file=target_file,
                micro_speedup=micro_speedup,
            ),
            keep_threshold_pct=3.0,
        )
        assert record is not None
        return record

    def _seed_two_siblings(self, session_dir: Path) -> list[str]:
        state = _seed_state(session_dir, baseline_tput=800.0)
        first = self._enqueue(state, kernel_name="fuse_a", target_file="/repo/a.py", micro_speedup=1.9)
        second = self._enqueue(state, kernel_name="fuse_a", target_file="/repo/b.py", micro_speedup=1.2)
        state.save(session_dir)

        pending = SharedState.load_or_init(session_dir).pending_kernel_integration_records()
        assert [str(record.get("kernel_id")) for record in pending] == ["fuse_a", "fuse_a"]
        return [str(first["integration_id"]), str(second["integration_id"])]

    def test_ambiguous_bare_kernel_id_is_refused_by_name(self, session_dir):
        ids = self._seed_two_siblings(session_dir)

        with pytest.raises(krh.AmbiguousIntegrationTarget) as excinfo:
            krh._fill_integrate_defaults_from_state(
                {"kernel_id": "fuse_a"},
                session_dir=session_dir,
            )

        assert type(excinfo.value) is krh.AmbiguousIntegrationTarget
        assert str(excinfo.value) == (
            "integrate refused: kernel_id='fuse_a' matches 2 pending integration "
            "records; an explicit integration_id is required to bind the "
            f"KEEP/REVERT verdict. candidates={sorted(ids)!r}"
        )

    def test_an_unresolvable_integration_id_does_not_reopen_the_guess(self, session_dir):
        """A named id that matches nothing is not a licence to fall back."""
        ids = self._seed_two_siblings(session_dir)

        with pytest.raises(krh.AmbiguousIntegrationTarget) as excinfo:
            krh._fill_integrate_defaults_from_state(
                {"kernel_id": "fuse_a", "integration_id": "kernel-integration:gone"},
                session_dir=session_dir,
            )

        assert f"candidates={sorted(ids)!r}" in str(excinfo.value)

    def test_an_explicit_integration_id_still_binds_its_own_sibling(self, session_dir):
        ids = self._seed_two_siblings(session_dir)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "fuse_a", "integration_id": ids[1]},
            session_dir=session_dir,
        )

        assert out["integration_id"] == ids[1]
        assert out["kernel_id"] == "fuse_a"

    def test_a_single_pending_sibling_still_resolves_from_a_bare_kernel_id(self, session_dir):
        state = _seed_state(session_dir, baseline_tput=800.0)
        only = self._enqueue(state, kernel_name="fuse_a", target_file="/repo/a.py", micro_speedup=1.9)
        state.save(session_dir)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "fuse_a"},
            session_dir=session_dir,
        )

        assert out["integration_id"] == only["integration_id"]
        assert out["kernel_id"] == "fuse_a"
        assert out["base_tput"] == 800.0

    def test_a_kernel_id_with_no_pending_record_is_unaffected(self, session_dir):
        self._seed_two_siblings(session_dir)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "gemm_tune_fmoe_ck"},
            session_dir=session_dir,
        )

        assert out["kernel_id"] == "gemm_tune_fmoe_ck"
        assert "integration_id" not in out

    @pytest.mark.asyncio
    async def test_integrate_handler_refuses_before_applying_anything(self, session_dir):
        self._seed_two_siblings(session_dir)

        with pytest.raises(krh.AmbiguousIntegrationTarget) as excinfo:
            await krh.integrate_handler(
                {"kernel_id": "fuse_a"},
                session_dir=session_dir,
            )

        assert type(excinfo.value) is krh.AmbiguousIntegrationTarget
        assert "an explicit integration_id is required" in str(excinfo.value)


class TestVendorPlaybookDeployBlocked:
    """A vendor-playbook KEEP (e.g. mori dispatch/combine) must never reach
    apply_kernel_patch: its best_artifact_path is a KernelForge task-bundle
    config copy, not a rewrite of the real installed operator source
    (PR #1191 review finding #1)."""

    def test_backfilled_from_kernel_opt_attempts_ledger(self, session_dir):
        state = SharedState.load_or_init(session_dir)
        state.kernel_opt_attempts = {
            "k010": {
                "vendor_playbook_deploy_blocked": True,
                "vendor_playbook_id": "mori_ep_dispatch_combine",
            }
        }
        state.save(session_dir)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k010"},
            session_dir=session_dir,
        )

        assert out["_vendor_playbook_deploy_blocked"] is True

    def test_backfilled_from_last_kernel_opt_when_ledger_missing(self, session_dir):
        """An LLM-initiated integrate can name a kernel_id that never made it
        into kernel_opt_attempts yet; last_kernel_opt must still catch it."""
        state = SharedState.load_or_init(session_dir)
        state.last_kernel_opt = {
            "kernel_id": "k010",
            "vendor_playbook_deploy_blocked": True,
        }
        state.save(session_dir)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k010"},
            session_dir=session_dir,
        )

        assert out["_vendor_playbook_deploy_blocked"] is True

    def test_normal_kernel_is_not_blocked(self, session_dir):
        state = SharedState.load_or_init(session_dir)
        state.kernel_opt_attempts = {"k001": {"vendor_playbook_deploy_blocked": False}}
        state.save(session_dir)

        out = krh._fill_integrate_defaults_from_state(
            {"kernel_id": "k001"},
            session_dir=session_dir,
        )

        assert "_vendor_playbook_deploy_blocked" not in out

    @pytest.mark.asyncio
    async def test_integrate_handler_refuses_before_touching_filesystem(self, session_dir):
        _seed_state(session_dir, baseline_tput=800.0)
        state = SharedState.load_or_init(session_dir)
        state.kernel_opt_attempts = {
            "k010": {
                "vendor_playbook_deploy_blocked": True,
                "vendor_playbook_id": "mori_ep_dispatch_combine",
                "last_artifact_path": "/tmp/forge/session1/mori_ep_config.py",
            }
        }
        state.save(session_dir)

        result = await krh.integrate_handler(
            {"kernel_id": "k010"},
            session_dir=session_dir,
        )

        assert result["status"] == "failed"
        assert result["error_class"] == "vendor_playbook_not_deployable"
        assert result["decision"] == "NEEDS_REVIEW"


class TestIntegrateRebaselineTimeout:
    def test_explicit_budget_wins(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("benchmark:\n  timeout_seconds: 7200\n")

        assert (
            krh._integrate_rebaseline_timeout_sec(
                {
                    "config_path": str(config),
                    "budget_minutes": 15,
                },
                default_timeout_sec=7800,
            )
            == 900
        )

    def test_benchmark_contract_replaces_legacy_cap(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("benchmark:\n  timeout_seconds: 7200\n")

        assert (
            krh._integrate_rebaseline_timeout_sec(
                {"config_path": str(config)},
                default_timeout_sec=7800,
            )
            == 7200
        )

    def test_shorter_benchmark_contract_is_preserved(self, tmp_path):
        config = tmp_path / "config.yaml"
        config.write_text("benchmark:\n  timeout_seconds: 2400\n")

        assert (
            krh._integrate_rebaseline_timeout_sec(
                {"config_path": str(config)},
                default_timeout_sec=7800,
            )
            == 2400
        )

    def test_executor_default_is_preserved(self):
        assert (
            krh._integrate_rebaseline_timeout_sec(
                {},
                default_timeout_sec=7800,
            )
            == 7800
        )


class TestIntegrateHandlerHonoursStateDefault:
    @pytest.mark.asyncio
    async def test_missing_base_tput_in_payload_still_runs_when_state_has_one(
        self,
        session_dir,
        monkeypatch,
    ):
        """The ``base_tput <= 0`` hard-check must not fire when state has a baseline."""
        _seed_state(session_dir, baseline_tput=800.0)

        result = await krh.integrate_handler(
            {"kernel_id": "k_no_artifact"},
            session_dir=session_dir,
        )

        assert result["status"] == "failed"
        assert result.get("error") != ("integrate_handler requires base_tput > 0 to compute KEEP/REVERT")

    @pytest.mark.asyncio
    async def test_no_base_tput_anywhere_still_fails_with_clear_error(
        self,
        session_dir,
    ):
        result = await krh.integrate_handler(
            {"kernel_id": "k_orphan"},
            session_dir=session_dir,
        )

        assert result["status"] == "failed"
        assert "base_tput" in result["error"]

    @pytest.mark.asyncio
    async def test_env_only_gemm_validation_runs_baseline_with_extra_envs(
        self,
        session_dir,
        monkeypatch,
    ):
        """GEMM tuning validation has no patch; it must still run E2E with envs."""
        _seed_state(session_dir, baseline_tput=1000.0, baseline_config_path="/tmp/base.yaml")
        captured: dict[str, object] = {}

        from hyperloom.orchestrator.actions.executors import baseline as baseline_mod

        class FakeBaselineExecutor:
            default_timeout_sec = baseline_mod.BASELINE_DEFAULT_TIMEOUT_SEC

            def __init__(self, *, session_dir):
                self.session_dir = session_dir

            async def __call__(self, ctx):
                captured["params"] = dict(ctx.task.params)
                return {"output_throughput": 1100.0, "completed_requests": 10}

        monkeypatch.setattr(baseline_mod, "BaselineExecutor", FakeBaselineExecutor)

        result = await krh.integrate_handler(
            {
                "source": "forge_gemm_tuning",
                "kernel_id": "gemm_tune_fmoe_ck",
                "base_tput": 1000.0,
                "config_path": "/tmp/base.yaml",
                "extra_envs": {"AITER_CONFIG_FMOE": "/tmp/fmoe.csv"},
                "budget_minutes": 1,
            },
            session_dir=session_dir,
        )

        assert result["status"] == "ok", result
        assert result["decision"] == "KEEP"
        assert result["new_tput"] == 1100.0
        assert captured["params"]["extra_envs"] == {"AITER_CONFIG_FMOE": "/tmp/fmoe.csv"}
        assert captured["params"]["defer_accuracy_until_after_measure"] is True
        assert captured["params"]["post_measure_accuracy_min_tput"] == pytest.approx(1010.0)


class TestApplybackProvenanceSurvivesTheLedgerFallbacks:
    """An apply-back must arm the strict accuracy gate however it was resolved.

    Provenance used to reach the payload only through the pending record, so
    every KEEP that fell back to a ledger -- which is what the ``source_file``
    dedup forces for the second and later KEEPs against one file -- looked like
    an ordinary kernel patch. That is the one artifact whose correctness was
    proven against a standalone reference only, so it must never be gradeable on
    throughput alone.
    """

    def _seed_attempt_ledger(self, session_dir: Path, *, kernel_id: str) -> None:
        state = SharedState.load_or_init(session_dir)
        state.kernel_opt_attempts = {
            kernel_id: {
                "last_artifact_path": "/tmp/deploy.patch",
                "last_source_file": "/framework/vllm/attention.py",
                "last_framework_applyback": {"artifact_kind": "framework_applyback"},
                "last_integration_validation_status": "pending",
            }
        }
        state.save(session_dir)

    def test_the_attempt_ledger_fallback_carries_provenance(self, session_dir):
        self._seed_attempt_ledger(session_dir, kernel_id="k_applyback")

        out, missing = krh._resolve_integrate_payload(
            {"kernel_id": "k_applyback"},
            session_dir=session_dir,
        )

        assert missing is None, missing
        assert out["artifact_kind"] == "framework_applyback"
        assert out["integration_validation_status"] == "pending"

    def test_the_last_kernel_opt_fallback_carries_provenance(self, session_dir):
        state = SharedState.load_or_init(session_dir)
        state.last_kernel_opt = {
            "kernel_id": "k_applyback",
            "best_artifact_path": "/tmp/deploy.patch",
            "source_file": "/framework/vllm/attention.py",
            "framework_applyback": {"artifact_kind": "framework_applyback"},
            "integration_validation_status": "pending",
        }
        state.save(session_dir)

        out, missing = krh._resolve_integrate_payload(
            {"kernel_id": "k_applyback"},
            session_dir=session_dir,
        )

        assert missing is None, missing
        assert out["artifact_kind"] == "framework_applyback"
        assert out["integration_validation_status"] == "pending"

    def test_an_explicit_payload_value_still_wins(self, session_dir):
        self._seed_attempt_ledger(session_dir, kernel_id="k_applyback")

        out, _missing = krh._resolve_integrate_payload(
            {
                "kernel_id": "k_applyback",
                "integration_validation_status": "passed",
            },
            session_dir=session_dir,
        )

        assert out["integration_validation_status"] == "passed"
