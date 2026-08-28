# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for the unified GEMM-tuning result handling on Coordinator.

Exercises ``_gemm_e2e_candidates`` guard rails for both the forge and GEAK
result shapes, and the ``_handle_gemm_tuning_result`` routing that puts every
backend on the per-tuner E2E path so a promoted gain is always a measurement.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

import hyperloom.inference_optimizer.model_config_utils as mcu_mod
import hyperloom.orchestrator.actions.executors.explore as explore_mod
import hyperloom.orchestrator.kernel.request_handlers as krh_mod
import hyperloom.orchestrator.phases.kernel as kernel_phase_mod
from hyperloom.inference_optimizer.protocol.intent import Intent, IntentType
from hyperloom.inference_optimizer.session.paths import make_session_dir
from hyperloom.orchestrator.loop.coordinator import Coordinator
from hyperloom.orchestrator.phases.kernel import KernelPhase
from hyperloom.orchestrator.roles import MockBackend, ScriptedPlan
from hyperloom.orchestrator.state.shared_state import SharedState


def _journal_entries(session_dir: Path) -> list[dict]:
    """Read the optimization_journal entries written under ``session_dir``."""
    path = session_dir / "reports" / "optimization_journal.json"
    if not path.exists():
        return []
    return list(json.loads(path.read_text(encoding="utf-8")).get("entries") or [])


def _make_integrate(responses):
    """Return an async ``integrate_handler`` double yielding queued responses."""
    calls: list[dict] = []

    async def _fake(payload, *, session_dir):
        calls.append(payload)
        idx = len(calls) - 1
        return responses[idx] if idx < len(responses) else responses[-1]

    _fake.calls = calls
    return _fake


def _coord(tmp_path: Path, **state_kwargs) -> Coordinator:
    coord = Coordinator.__new__(Coordinator)
    coord.session_dir = tmp_path
    coord.shared_state = SharedState(**state_kwargs)
    return coord


def test_syncs_standard_roofline_fallback_into_live_coordinator_state(tmp_path):
    coord = _coord(tmp_path)
    coord.shared_state.save(tmp_path)
    selected_trace = str(tmp_path / "mixed_steady_state.trace.json.gz")
    persisted = SharedState.load_or_init(tmp_path)
    persisted.last_profile_trace = str(tmp_path / "profile.trace.json.gz")
    persisted.last_profile_status = "succeeded"
    persisted.last_profile_args = "--attention-backend AITER"
    persisted.last_profile_workload = {"framework": "vllm", "server_args": "--attention-backend AITER"}
    persisted.last_trace_analyze = {
        "trace_input": persisted.last_profile_trace,
        "steady_state_trace": selected_trace,
        "roofline_snapshot_id": 3,
    }
    persisted.roofline_snapshots = [{"snapshot_id": 3}]
    persisted.baseline_eager_fallback = False
    persisted.save(tmp_path)
    coord.shared_state.baseline_eager_fallback = True

    coord.phase_kernel._sync_profile_state_after_gemm_roofline(
        {
            "shape_capture": {
                "capture_mode": "block_fp8_profile",
                "source_profile_trace": selected_trace,
            }
        }
    )

    assert coord.shared_state.last_profile_trace == persisted.last_profile_trace
    assert coord.shared_state.last_profile_workload == persisted.last_profile_workload
    assert coord.shared_state.last_trace_analyze == persisted.last_trace_analyze
    assert coord.shared_state.roofline_snapshot_id == 3
    assert coord.shared_state.baseline_eager_fallback is False


def test_sync_unions_lifecycle_instead_of_overwriting(tmp_path):
    """Neither the live state's nor the inline Roofline's rows may be dropped."""
    coord = _coord(tmp_path)
    coord.shared_state.record_lifecycle_event(step="explore", status="END", ts="2026-01-01T00:00:00Z")
    coord.shared_state.save(tmp_path)
    # Recorded on the live state only; never persisted before the sync.
    coord.shared_state.record_lifecycle_event(step="live_only", status="START", ts="2026-01-01T00:00:05Z")

    selected_trace = str(tmp_path / "mixed_steady_state.trace.json.gz")
    persisted = SharedState.load_or_init(tmp_path)
    persisted.last_trace_analyze = {"steady_state_trace": selected_trace}
    persisted.record_lifecycle_event(step="profile", status="END", ts="2026-01-01T00:00:03Z")
    persisted.save(tmp_path)

    result = {
        "shape_capture": {
            "capture_mode": "block_fp8_profile",
            "source_profile_trace": selected_trace,
        }
    }
    coord.phase_kernel._sync_profile_state_after_gemm_roofline(result)

    rows = coord.shared_state.lifecycle
    assert [row["step"] for row in rows] == ["explore", "profile", "live_only"]
    assert [row["seq"] for row in rows] == [0, 1, 2]

    # Re-running the merge must not duplicate rows or shuffle seq.
    before = list(rows)
    coord.phase_kernel._sync_profile_state_after_gemm_roofline(result)
    assert coord.shared_state.lifecycle == before


def _silent_backends() -> dict[str, object]:
    silent = ScriptedPlan(
        turns=[],
        default_intent=Intent(
            type=IntentType.SEND_MESSAGE,
            payload={"topic": "heartbeat", "body_md": "ok"},
        ),
    )
    return {
        "orchestration": MockBackend(silent, name="o"),
        "critic": MockBackend(silent, name="c"),
        "robustness": MockBackend(silent, name="r"),
    }


@pytest.fixture
def coord_session_dir(tmp_path, monkeypatch) -> Path:
    monkeypatch.setenv("USER_DATA_PATH", str(tmp_path))
    return make_session_dir()


@pytest.mark.asyncio
async def test_intent_router_gemm_roofline_survives_terminal_lifecycle_save(
    coord_session_dir,
    monkeypatch,
):
    """The lifecycle END save must not clobber the inline Roofline refresh.

    On the intent_router path the terminal lifecycle event persists the live
    state between the handler returning and the GEMM result being handled, so
    the refresh has to be merged back before that save.
    """
    coord = Coordinator(coord_session_dir, backends=_silent_backends())
    try:
        selected_trace = str(coord_session_dir / "mixed_steady_state.trace.json.gz")

        async def fake_handler(payload, *, session_dir):
            # Mirror the handler-owned inline Roofline: it mutates a throwaway
            # SharedState loaded from disk and persists it there.
            state = SharedState.load_or_init(session_dir)
            state.last_profile_trace = str(session_dir / "profile.trace.json.gz")
            state.last_profile_status = "succeeded"
            state.last_profile_workload = {
                "framework": "vllm",
                "server_args": "--attention-backend AITER",
            }
            state.last_trace_analyze = {
                "trace_input": state.last_profile_trace,
                "steady_state_trace": selected_trace,
                "roofline_snapshot_id": 7,
            }
            state.baseline_eager_fallback = False
            state.save(session_dir)
            return {
                "status": "ok",
                "decision": "REVERT",
                "backend": "geak",
                "shape_capture": {
                    "capture_mode": "block_fp8_profile",
                    "source_profile_trace": selected_trace,
                },
            }

        monkeypatch.setitem(
            krh_mod.KERNEL_REQUEST_HANDLERS,
            "run_gemm_tuning",
            fake_handler,
        )
        monkeypatch.setattr(
            coord.dispatcher,
            "_sequence_denial_for_request",
            lambda target, kind: None,
        )
        coord.shared_state.baseline_eager_fallback = True

        await coord._handle_intent(
            "orchestration",
            Intent(
                type=IntentType.REQUEST,
                payload={
                    "target_agent": "kernel_agent",
                    "kind": "run_gemm_tuning",
                    "params": {},
                },
            ),
        )

        assert coord.shared_state.last_trace_analyze.get("steady_state_trace") == selected_trace
        assert coord.shared_state.last_profile_status == "succeeded"
        assert coord.shared_state.roofline_snapshot_id == 7
        assert coord.shared_state.baseline_eager_fallback is False
        # It must also survive on disk so the next run can reuse the trace.
        reloaded = SharedState.load_or_init(coord_session_dir)
        assert reloaded.last_trace_analyze.get("steady_state_trace") == selected_trace
    finally:
        await coord.stop()


class _Bus:
    def __init__(self) -> None:
        self.messages = []

    async def append_and_seq(self, message):
        self.messages.append(message)
        return message


def _collective_campaign(
    tmp_path: Path,
    integration_id: str = "collective-test",
    **overrides,
) -> dict:
    """Build a persisted Collective KEEP campaign for integration tests."""
    campaign = {
        "collective_attempt_id": f"attempt-{integration_id}",
        "integration_id": integration_id,
        "integration_status": "pending",
        "status": "ok",
        "decision": "KEEP",
        "engine": "forge_collective",
        "kept": True,
        "requires_e2e_validation": True,
        "patch": str(tmp_path / "forge.patch"),
        "source_file": "/repo/custom_all_reduce.cuh",
        "kernel_repo": "/repo",
        "snapshot_dir": str(tmp_path / "collective_snapshot"),
    }
    campaign.update(overrides)
    return campaign


def _record_collective_campaign(
    coord: Coordinator,
    tmp_path: Path,
    integration_id: str = "collective-test",
    **overrides,
) -> dict:
    """Persist and return a Collective KEEP campaign."""
    campaign = _collective_campaign(
        tmp_path,
        integration_id,
        **overrides,
    )
    coord.shared_state.record_collective(campaign, tmp_path)
    return campaign


def _collective_recovery_paths(
    tmp_path: Path,
    integration_id: str,
) -> tuple[Path, Path, Path]:
    """Return the backup manifest and checkpoint paths for an integration."""
    identity = hashlib.sha256(integration_id.encode("utf-8")).hexdigest()[:16]
    patch_root = tmp_path / "patches" / f"forge_collective_{identity}"
    manifest = patch_root / "backup" / "forge_collective_x" / "manifest.json"
    return patch_root / "backup", manifest, patch_root / "apply_checkpoint.json"


def _write_collective_recovery(
    tmp_path: Path,
    integration_id: str,
    manifest_status: str,
    *,
    checkpoint: bool = True,
) -> tuple[Path, Path]:
    """Write a Collective apply manifest and optional checkpoint."""
    _, manifest, checkpoint_path = _collective_recovery_paths(
        tmp_path,
        integration_id,
    )
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps({"status": manifest_status}),
        encoding="utf-8",
    )
    if checkpoint:
        checkpoint_path.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "manifest_path": str(manifest),
                }
            ),
            encoding="utf-8",
        )
    return manifest, checkpoint_path


class TestGemmE2eCandidates:
    """Guard rails deciding which tuning results reach the E2E validator."""

    def test_geak_result_yields_the_tuned_dispatch_csv(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=200.0)
        cands = coord._gemm_e2e_candidates(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.1,
                "backend": "geak",
                "tuned_file": "/tuned/gemm.csv",
            }
        )
        assert len(cands) == 1
        assert cands[0]["tuner"] == "a8w8_blockscale_tuned_gemm"
        assert cands[0]["envs"] == {"AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": "/tuned/gemm.csv"}
        assert cands[0]["micro_speedup"] == pytest.approx(1.1)

    @pytest.mark.parametrize(
        "overrides",
        [
            {"status": "failed"},
            {"decision": "REVERT"},
            {"best_speedup": 1.0},
            {"best_speedup": object()},
            {"tuned_file": ""},
        ],
    )
    def test_geak_result_yields_nothing_without_a_usable_keep(self, tmp_path, overrides):
        coord = _coord(tmp_path, baseline_tput=200.0)
        result = {
            "status": "ok",
            "decision": "KEEP",
            "best_speedup": 1.1,
            "backend": "geak",
            "tuned_file": "/tuned/gemm.csv",
        }
        result.update(overrides)
        assert coord._gemm_e2e_candidates(result) == []

    def test_non_dict_result_is_rejected(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=100.0)
        assert coord._gemm_e2e_candidates({}) == []

    def test_forge_result_yields_one_candidate_per_improved_tuner(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=100.0)
        cands = coord._gemm_e2e_candidates(
            {
                "status": "ok",
                "decision": "KEEP",
                "backend": "forge",
                "tuners_run": [
                    {
                        "tuner": "fmoe_ck",
                        "status": "ok",
                        "candidate": True,
                        "env_var": "AITER_CONFIG_FMOE",
                        "env_value": "/cfg/fmoe.csv",
                        "best_micro_speedup": 1.3,
                    },
                    {"tuner": "skipped_one", "status": "ok", "improved_shapes": 0},
                    {"tuner": "failed_one", "status": "failed", "candidate": True},
                ],
            }
        )
        assert [c["tuner"] for c in cands] == ["fmoe_ck"]
        assert cands[0]["envs"] == {"AITER_CONFIG_FMOE": "/cfg/fmoe.csv"}

    def test_forge_result_ignores_a_tuned_file(self, tmp_path):
        """tuned_file is the GEAK shape; forge must come from tuners_run."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        assert (
            coord._gemm_e2e_candidates(
                {
                    "status": "ok",
                    "decision": "KEEP",
                    "best_speedup": 1.2,
                    "backend": "forge",
                    "tuned_file": "/tuned/gemm.csv",
                }
            )
            == []
        )


class TestPromoteFusionIntegrateKeep:
    def test_records_incremental_gain_but_preserves_baseline_total(self, tmp_path):
        coord = _coord(
            tmp_path,
            baseline_tput=100.0,
            cumulative_gain_validated=20.0,
            cumulative_gain_validated_stack_len=1,
            optimization_stack=[
                {
                    "action": "replay_warm_recipe",
                    "tput": 120.0,
                    "gain_pct": 20.0,
                }
            ],
            gain_per_stack_entry=[20.0],
            current_best={
                "action": "replay_warm_recipe",
                "tput": 120.0,
                "extra_envs": {"SGLANG_USE_AITER": "1"},
                "extra_server_args": "--moe-runner-backend aiter",
            },
        )

        KernelPhase(coord)._promote_fusion_integrate_keep(
            {
                "patch": "/tmp/fusion.patch",
                "source_file": "/repo/model.py",
                "kernel_speedup": 3.05,
                "best_pattern": "llm:fused_a+llm:fused_b",
            },
            {
                "status": "ok",
                "decision": "KEEP",
                "new_tput": 180.0,
                # integrate's gain is relative to current_best=120, not the
                # original baseline=100.
                "gain_pct": 50.0,
                "workspace": "/tmp/run",
                "extra_server_args": "--moe-runner-backend aiter",
            },
            extra_envs={"SGLANG_USE_AITER": "1", "ZAYA_FUSED_HYBRID_RESIDUAL": "1"},
        )

        stack = coord.shared_state.optimization_stack
        assert len(stack) == 2
        assert stack[1]["action"] == "fusion"
        assert stack[1]["backend"] == "forge"
        assert stack[1]["engine"] == "forge_fusion"
        assert stack[1]["patch_path"] == "/tmp/fusion.patch"
        assert stack[1]["gain_pct"] == pytest.approx(50.0)
        assert stack[1]["extra_envs"]["SGLANG_USE_AITER"] == "1"
        assert stack[1]["extra_envs"]["ZAYA_FUSED_HYBRID_RESIDUAL"] == "1"
        assert coord.shared_state.current_best["action"] == "fusion"
        assert coord.shared_state.current_best["tput"] == 180.0
        # current_best is a config record; the forge labels live on the entry.
        assert "engine" not in coord.shared_state.current_best
        assert "backend" not in coord.shared_state.current_best
        assert coord.shared_state.cumulative_gain_validated == 80.0
        assert coord.shared_state.gain_per_stack_entry == [20.0, 80.0]
        assert coord.shared_state.cumulative_gain_validated_stack_len == 2

    def test_guard_paths_do_not_promote(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=100.0)
        phase = KernelPhase(coord)

        phase._promote_fusion_integrate_keep("bad", {"decision": "KEEP"})  # type: ignore[arg-type]
        phase._promote_fusion_integrate_keep({}, {"decision": "REVERT"})
        phase._promote_fusion_integrate_keep({}, {"decision": "KEEP", "new_tput": "bad"})
        phase._promote_fusion_integrate_keep({}, {"decision": "KEEP", "new_tput": 0})

        assert coord.shared_state.optimization_stack == []
        assert coord.shared_state.current_best == {}

    def test_dedupes_same_fusion_patch(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=100.0)
        phase = KernelPhase(coord)
        fusion = {"patch": "/tmp/fusion.patch", "source_file": "/repo/model.py"}
        integ = {"decision": "KEEP", "new_tput": 150.0, "gain_pct": 50.0}

        phase._promote_fusion_integrate_keep(fusion, integ)
        phase._promote_fusion_integrate_keep(fusion, integ)

        assert len(coord.shared_state.optimization_stack) == 1
        assert coord.shared_state.optimization_stack[0]["patch_path"] == "/tmp/fusion.patch"
        assert coord.shared_state.current_best["variant_name"] == "forge_fusion:fusion.patch"

    @pytest.mark.asyncio
    async def test_handle_fusion_result_posts_and_integrates_kept_candidate(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        integrated: list[dict] = []

        async def _fake_integrate(result):
            integrated.append(result)

        monkeypatch.setattr(phase, "_integrate_fusion", _fake_integrate)
        result = {
            "status": "ok",
            "kept": True,
            "requires_e2e_validation": True,
            "engine": "forge_fusion",
        }

        await phase._handle_fusion_result(result)

        assert coord.shared_state.last_fusion == result
        assert integrated == [result]
        assert coord.bus.messages[0].payload["kind"] == "run_fusion_done"

    @pytest.mark.asyncio
    async def test_handle_fusion_result_tolerates_non_dict_and_bus_failure(self, tmp_path):
        coord = _coord(tmp_path)

        class BadBus:
            async def append_and_seq(self, *_args, **_kwargs):
                raise RuntimeError("bus down")

        coord.bus = BadBus()
        phase = KernelPhase(coord)

        await phase._handle_fusion_result("not-dict")  # type: ignore[arg-type]

        assert coord.shared_state.last_fusion == {"status": "failed"}

    @pytest.mark.asyncio
    async def test_integrate_fusion_builds_payload_and_records_keep(self, tmp_path, monkeypatch):
        coord = _coord(
            tmp_path,
            baseline_tput=100.0,
            current_best={"extra_envs": {"SGLANG_USE_AITER": "1"}},
        )
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        calls: list[dict] = []

        async def _fake_integrate(payload, *, session_dir):
            assert session_dir == tmp_path
            calls.append(payload)
            return {
                "status": "ok",
                "decision": "KEEP",
                "new_tput": 170.0,
                "gain_pct": 70.0,
                "workspace": str(tmp_path / "integrate"),
            }

        monkeypatch.setattr(krh_mod, "integrate_handler", _fake_integrate)
        monkeypatch.setattr(
            krh_mod,
            "materialize_unified_patch_snapshot",
            lambda **_kwargs: str(tmp_path / "snapshot"),
        )

        await phase._integrate_fusion(
            {
                "patch": str(tmp_path / "fusion.patch"),
                "source_file": "/repo/model.py",
                "kernel_repo": "/repo",
                "env_flags": {"ZAYA_FUSED": "1"},
                "kernel_speedup": 2.5,
                "best_pattern": "llm:fused",
            }
        )

        assert calls[0]["source"] == "forge_fusion"
        assert calls[0]["snapshot_dir"] == str(tmp_path / "snapshot")
        assert calls[0]["extra_envs"] == {"SGLANG_USE_AITER": "1", "ZAYA_FUSED": "1"}
        assert coord.shared_state.last_fusion_integrate["decision"] == "KEEP"
        assert coord.shared_state.current_best["action"] == "fusion"
        assert coord.shared_state.optimization_stack[-1]["engine"] == "forge_fusion"
        assert coord.bus.messages[-1].payload["kind"] == "fusion_integrate_done"

    @pytest.mark.asyncio
    async def test_integrate_fusion_records_snapshot_failure(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path)
        coord.bus = _Bus()
        phase = KernelPhase(coord)

        def _raise_snapshot(**_kwargs):
            raise ValueError("bad patch")

        monkeypatch.setattr(krh_mod, "materialize_unified_patch_snapshot", _raise_snapshot)

        await phase._integrate_fusion(
            {
                "patch": str(tmp_path / "fusion.patch"),
                "source_file": "/repo/model.py",
                "kernel_repo": "/repo",
            }
        )

        assert coord.shared_state.last_fusion_integrate["decision"] == "REVERT"
        assert coord.shared_state.last_fusion_integrate["error_class"] == "ValueError"

    @pytest.mark.asyncio
    async def test_integrate_fusion_skips_missing_patch_or_target(self, tmp_path):
        coord = _coord(tmp_path)
        coord.bus = _Bus()
        phase = KernelPhase(coord)

        await phase._integrate_fusion({"patch": "", "source_file": "/repo/model.py"})
        await phase._integrate_fusion({"patch": "/tmp/fusion.patch", "source_file": ""})

        assert coord.shared_state.last_fusion_integrate == {}

    @pytest.mark.asyncio
    async def test_run_forge_fusion_handles_handler_exception(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path)
        coord.bus = _Bus()
        phase = KernelPhase(coord)

        async def _raise(*_args, **_kwargs):
            raise RuntimeError("fusion boom")

        monkeypatch.setattr(krh_mod, "run_fusion_handler", _raise)

        await phase._run_forge_fusion()

        assert coord.shared_state.last_fusion["decision"] == "REVERT"
        assert coord.shared_state.last_fusion["error_class"] == "RuntimeError"


class TestCollectiveIntegratePromotion:
    """Collective KEEP results must pass through the same E2E adoption gate."""

    @pytest.mark.asyncio
    async def test_collective_only_preempts_default_geak(self, tmp_path, monkeypatch):
        """The directed lane must run before the default GEAK selection."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        phase = KernelPhase(coord)
        calls: list[str] = []

        def _unexpected_geak():
            """Fail if Collective-only consults the default GEAK backend."""
            raise AssertionError("GEAK must not preempt Collective-only")

        async def _reprofile():
            """Record the directed lane's required profile refresh."""
            calls.append("reprofile")

        async def _collective():
            """Record the directed Collective dispatch."""
            calls.append("collective")

        monkeypatch.setenv("HYPERLOOM_COLLECTIVE_ONLY", "1")
        monkeypatch.setattr(phase, "_kernel_enabled", lambda: True)
        monkeypatch.setattr(phase, "_geak_enabled", _unexpected_geak)
        monkeypatch.setattr(phase, "_maybe_reprofile_for_kernel", _reprofile)
        monkeypatch.setattr(
            phase,
            "_maybe_run_collective_before_kernel_opt",
            _collective,
        )

        await phase._on_enter_kernel(from_phase="FRAMEWORK_AGENT")

        assert calls == ["reprofile", "collective"]
        assert coord.shared_state.collective_only_mode is True

    def test_promotes_and_deduplicates_collective_keep(self, tmp_path):
        """An E2E KEEP should become one durable Collective stack entry."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        phase = KernelPhase(coord)
        collective = {
            "integration_id": "collective-promote",
            "patch": "/tmp/collective.patch",
            "source_file": "/repo/custom_all_reduce.cuh",
            "kernel_speedup": 1.2,
            "collective_op": "all_reduce",
            "world_size": 8,
        }
        integrate = {
            "status": "ok",
            "decision": "KEEP",
            "new_tput": 130.0,
            "gain_pct": 30.0,
            "workspace": "/tmp/run",
            "apply_result": {
                "status": "ok",
                "manifest_path": "/tmp/collective-manifest.json",
            },
        }

        phase._promote_collective_integrate_keep(collective, integrate)
        # current_best is a pure config record; the engine that produced the
        # winner is stack-entry provenance.
        assert coord.shared_state.current_best["action"] == "collective"
        assert coord.shared_state.current_best["variant_name"] == "forge_collective"
        assert coord.shared_state.optimization_stack[0]["engine"] == "forge_collective"
        coord.shared_state.current_best = {
            "engine": "later_lane",
            "tput": 150.0,
        }
        coord.shared_state.cumulative_gain_validated = 50.0
        phase._promote_collective_integrate_keep(collective, integrate)

        assert len(coord.shared_state.optimization_stack) == 1
        entry = coord.shared_state.optimization_stack[0]
        assert entry["action"] == "collective"
        assert entry["collective_op"] == "all_reduce"
        assert entry["world_size"] == 8
        assert coord.shared_state.current_best["engine"] == "later_lane"
        assert coord.shared_state.cumulative_gain_validated == pytest.approx(50.0)
        assert coord.shared_state.gain_per_stack_entry == [30.0]

    @pytest.mark.asyncio
    async def test_handle_collective_posts_and_integrates_kept_candidate(self, tmp_path, monkeypatch):
        """The run verdict must be recorded before its E2E integration starts."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        integrated: list[dict] = []

        async def _fake_integrate(result):
            """Capture the candidate handed to the integration gate."""
            integrated.append(result)

        monkeypatch.setattr(phase, "_integrate_collective", _fake_integrate)
        result = {
            "status": "ok",
            "decision": "KEEP",
            "kept": True,
            "requires_e2e_validation": True,
            "engine": "forge_collective",
        }

        await phase._handle_collective_result(result)
        first_attempt_id = coord.shared_state.last_collective["collective_attempt_id"]
        await phase._handle_collective_result(result)

        assert coord.shared_state.last_collective["status"] == "ok"
        assert coord.shared_state.last_collective["patch_cleanup_status"] == "pending"
        assert integrated[0]["patch_cleanup_status"] == "pending"
        assert coord.shared_state.last_collective["collective_attempt_id"] == first_attempt_id
        assert len(coord.shared_state.collective_attempts) == 1
        assert coord.bus.messages[0].payload["kind"] == "run_collective_done"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("finalize_status", ["ok", "partial"])
    async def test_integrate_collective_builds_payload_and_records_keep(self, tmp_path, monkeypatch, finalize_status):
        """Collective integration must use an isolated kernel id and snapshot."""
        coord = _coord(
            tmp_path,
            baseline_tput=100.0,
            current_best={"extra_envs": {"SGLANG_USE_AITER": "1"}},
        )
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        calls: list[dict] = []
        integration_id = "collective-keep"
        manifest_path = tmp_path / "manifest.json"

        async def _fake_integrate(payload, *, session_dir):
            """Return a deterministic E2E KEEP for payload inspection."""
            assert session_dir == tmp_path
            calls.append(payload)
            return {
                "status": "ok",
                "decision": "KEEP",
                "new_tput": 140.0,
                "gain_pct": 40.0,
                "workspace": str(tmp_path / "integrate"),
                "apply_result": {
                    "status": "ok",
                    "manifest_path": str(manifest_path),
                },
                "finalize_result": {
                    "status": "skipped",
                    "reason": "deferred to caller durability checkpoint",
                },
            }

        monkeypatch.setattr(krh_mod, "integrate_handler", _fake_integrate)
        monkeypatch.setattr(
            krh_mod,
            "materialize_unified_patch_snapshot",
            lambda **_kwargs: str(tmp_path / "collective_snapshot"),
        )
        monkeypatch.setattr(
            krh_mod,
            "_maybe_finalize_kernel_patch",
            lambda _apply: {"status": finalize_status},
        )

        campaign = {
            "collective_attempt_id": "collective-attempt-keep",
            "integration_id": integration_id,
            "status": "ok",
            "decision": "KEEP",
            "engine": "forge_collective",
            "kept": True,
            "requires_e2e_validation": True,
            "patch": str(tmp_path / "forge.patch"),
            "source_file": "/repo/custom_all_reduce.cuh",
            "kernel_repo": "/repo",
            "kernel_speedup": 1.2,
            "collective_op": "all_reduce",
            "world_size": 8,
        }
        coord.shared_state.record_collective(campaign, tmp_path)
        await phase._integrate_collective(campaign)

        assert calls[0]["source"] == "forge_collective"
        assert calls[0]["kernel_id"] == "forge_collective"
        assert calls[0]["snapshot_dir"] == str(tmp_path / "collective_snapshot")
        assert calls[0]["extra_envs"] == {"SGLANG_USE_AITER": "1"}
        assert calls[0]["defer_patch_finalize"] is True
        assert calls[0]["backup_root"].endswith("/backup")
        assert calls[0]["apply_checkpoint_path"].endswith("/apply_checkpoint.json")
        assert coord.shared_state.last_collective["integration_decision"] == "KEEP"
        assert coord.shared_state.last_collective["patch_cleanup_status"] == "complete"
        assert coord.shared_state.current_best["action"] == "collective"
        assert coord.bus.messages[-1].payload["kind"] == "collective_integrate_done"

    @pytest.mark.asyncio
    async def test_pending_collective_reuses_applied_manifest(self, tmp_path, monkeypatch):
        """Resume must benchmark an existing apply without overwriting backups."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        integration_id = "resume-collective"
        identity = hashlib.sha256(integration_id.encode("utf-8")).hexdigest()[:16]
        patch_root = tmp_path / "patches" / f"forge_collective_{identity}"
        manifest = patch_root / "backup" / "forge_collective_x" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps({"status": "applied"}),
            encoding="utf-8",
        )
        checkpoint = patch_root / "apply_checkpoint.json"
        checkpoint.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "manifest_path": str(manifest),
                }
            ),
            encoding="utf-8",
        )
        calls: list[dict] = []

        async def _fake_integrate(payload, *, session_dir):
            """Capture the resumed pre-applied payload."""
            assert session_dir == tmp_path
            calls.append(payload)
            return {
                "status": "ok",
                "decision": "REVERT",
                "new_tput": 90.0,
                "gain_pct": -10.0,
                "apply_result": payload["preapplied_apply_result"],
                "revert_result": {"status": "ok"},
            }

        monkeypatch.setattr(krh_mod, "integrate_handler", _fake_integrate)
        monkeypatch.setattr(
            krh_mod,
            "materialize_unified_patch_snapshot",
            lambda **_kwargs: str(tmp_path / "collective_snapshot"),
        )

        campaign = {
            "collective_attempt_id": "collective-attempt-resume",
            "integration_id": integration_id,
            "integration_status": "pending",
            "status": "ok",
            "decision": "KEEP",
            "engine": "forge_collective",
            "kept": True,
            "requires_e2e_validation": True,
            "patch": str(tmp_path / "forge.patch"),
            "source_file": "/repo/custom_all_reduce.cuh",
            "kernel_repo": "/repo",
        }
        coord.shared_state.record_collective(campaign, tmp_path)
        await phase._integrate_collective(campaign)

        assert calls[0]["preapplied_apply_result"]["manifest_path"] == str(manifest)
        assert not checkpoint.exists()

    @pytest.mark.asyncio
    async def test_revert_recovery_does_not_repeat_e2e(self, tmp_path, monkeypatch):
        """An explicit recovery verdict must revert without remeasurement.

        The revert here comes back ``partial``, which under the converged
        lifecycle contract is owed work, not done work: a partial revert can
        leave the patch live on a remote pod, so the row keeps
        ``recovery_required`` and names ``revert`` as the action still owed.
        """
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        integration_id = "revert-collective"
        identity = hashlib.sha256(integration_id.encode("utf-8")).hexdigest()[:16]
        patch_root = tmp_path / "patches" / f"forge_collective_{identity}"
        manifest = patch_root / "backup" / "forge_collective_x" / "manifest.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps({"status": "applied"}),
            encoding="utf-8",
        )
        checkpoint = patch_root / "apply_checkpoint.json"
        checkpoint.write_text(
            json.dumps(
                {
                    "status": "ok",
                    "manifest_path": str(manifest),
                }
            ),
            encoding="utf-8",
        )

        async def _unexpected_integrate(*_args, **_kwargs):
            """Fail if recovery re-enters the E2E measurement path."""
            raise AssertionError("integration must not be repeated")

        monkeypatch.setattr(
            krh_mod,
            "integrate_handler",
            _unexpected_integrate,
        )
        monkeypatch.setattr(
            krh_mod,
            "_maybe_revert_kernel_patch",
            lambda _apply: {"status": "partial"},
        )
        campaign = {
            "collective_attempt_id": "collective-attempt-revert",
            "integration_id": integration_id,
            "integration_status": "recovery_required",
            "integration_recovery_action": "revert",
            "integration_decision": "REVERT",
            "status": "ok",
            "decision": "KEEP",
            "engine": "forge_collective",
            "kept": True,
            "requires_e2e_validation": True,
            "patch": str(tmp_path / "forge.patch"),
            "source_file": "/repo/custom_all_reduce.cuh",
            "kernel_repo": "/repo",
        }
        coord.shared_state.record_collective(campaign, tmp_path)

        await phase._integrate_collective(campaign)

        assert coord.shared_state.last_collective["patch_cleanup_status"] == "recovery_required"
        assert coord.shared_state.last_collective["patch_cleanup_action"] == "revert"
        assert coord.shared_state.last_collective["integration_decision"] == "REVERT"
        # The checkpoint is what a later pass reverts from, so an incomplete
        # revert has to keep it; only a complete cleanup may drop it.
        assert checkpoint.exists()

    @pytest.mark.asyncio
    async def test_run_forge_collective_records_handler_failure(self, tmp_path, monkeypatch):
        """A Collective handler exception must become a durable lane verdict."""
        coord = _coord(tmp_path)
        coord.bus = _Bus()
        phase = KernelPhase(coord)

        async def _raise(*_args, **_kwargs):
            """Raise a deterministic Collective handler failure."""
            raise RuntimeError("collective boom")

        monkeypatch.setattr(krh_mod, "run_collective_handler", _raise)

        await phase._run_forge_collective()

        assert coord.shared_state.last_collective["status"] == "failed"
        assert coord.shared_state.last_collective["decision"] == "REVERT"
        assert coord.shared_state.last_collective["error_class"] == "RuntimeError"
        assert coord.bus.messages[-1].payload["kind"] == "run_collective_done"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "result, error_type",
        [
            (None, TypeError),
            (
                {
                    "status": "ok",
                    "decision": "KEEP",
                    "engine": "forge_collective",
                    "kept": "yes",
                    "requires_e2e_validation": True,
                },
                ValueError,
            ),
            (
                {
                    "status": "ok",
                    "decision": "KEEP",
                    "engine": "forge_collective",
                    "kept": True,
                    "requires_e2e_validation": False,
                },
                ValueError,
            ),
        ],
    )
    async def test_handle_collective_rejects_invalid_handler_contract(self, tmp_path, result, error_type):
        """Invalid handler mappings and E2E flags must fail before recording."""
        coord = _coord(tmp_path)
        coord.bus = _Bus()
        phase = KernelPhase(coord)

        with pytest.raises(error_type):
            await phase._handle_collective_result(result)

        assert coord.shared_state.last_collective == {}

    @pytest.mark.asyncio
    async def test_handle_collective_tolerates_bus_failure(self, tmp_path):
        """A run-result bus failure must not discard the persisted verdict."""
        coord = _coord(tmp_path)
        phase = KernelPhase(coord)

        class _FailingBus:
            async def append_and_seq(self, _message):
                """Raise a deterministic bus failure."""
                raise RuntimeError("bus unavailable")

        coord.bus = _FailingBus()

        await phase._handle_collective_result(
            {
                "status": "failed",
                "decision": "REVERT",
                "engine": "forge_collective",
            }
        )

        assert coord.shared_state.last_collective["status"] == "failed"
        assert coord.shared_state.last_collective["decision"] == "REVERT"

    def test_load_collective_checkpoint_returns_manifest_status(self, tmp_path):
        """A trusted checkpoint must return normalized manifest metadata."""
        backup_root, manifest, checkpoint = _collective_recovery_paths(
            tmp_path,
            "checkpoint-ok",
        )
        manifest.parent.mkdir(parents=True)
        manifest.write_text(
            json.dumps({"status": "applied"}),
            encoding="utf-8",
        )
        checkpoint.write_text(
            json.dumps({"manifest_path": str(manifest)}),
            encoding="utf-8",
        )

        recovered, status = kernel_phase_mod._collective_recovery.load_apply_checkpoint(
            checkpoint,
            backup_root,
        )

        assert recovered["manifest_path"] == str(manifest)
        assert status == "applied"

    @pytest.mark.parametrize(
        "case",
        [
            "checkpoint_not_mapping",
            "manifest_missing",
            "manifest_outside_backup",
            "manifest_not_mapping",
        ],
    )
    def test_load_collective_checkpoint_rejects_untrusted_state(self, tmp_path, case):
        """Malformed or untrusted checkpoint state must be rejected."""
        backup_root, manifest, checkpoint = _collective_recovery_paths(
            tmp_path,
            f"checkpoint-{case}",
        )
        checkpoint.parent.mkdir(parents=True)
        if case == "checkpoint_not_mapping":
            checkpoint.write_text("[]", encoding="utf-8")
        elif case == "manifest_missing":
            checkpoint.write_text(
                json.dumps({"manifest_path": str(manifest)}),
                encoding="utf-8",
            )
        elif case == "manifest_outside_backup":
            outside = tmp_path / "outside-manifest.json"
            outside.write_text("{}", encoding="utf-8")
            checkpoint.write_text(
                json.dumps({"manifest_path": str(outside)}),
                encoding="utf-8",
            )
        else:
            manifest.parent.mkdir(parents=True)
            manifest.write_text("[]", encoding="utf-8")
            checkpoint.write_text(
                json.dumps({"manifest_path": str(manifest)}),
                encoding="utf-8",
            )

        with pytest.raises(ValueError):
            kernel_phase_mod._collective_recovery.load_apply_checkpoint(
                checkpoint,
                backup_root,
            )

    @pytest.mark.asyncio
    async def test_integrate_collective_marks_corrupt_checkpoint_for_recovery(self, tmp_path):
        """A corrupt apply checkpoint must preserve a NEEDS_REVIEW recovery."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        integration_id = "corrupt-checkpoint"
        campaign = _record_collective_campaign(
            coord,
            tmp_path,
            integration_id,
        )
        _, _, checkpoint = _collective_recovery_paths(
            tmp_path,
            integration_id,
        )
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_text("{", encoding="utf-8")

        await phase._integrate_collective(campaign)

        last = coord.shared_state.last_collective
        assert last["integration_decision"] == "NEEDS_REVIEW"
        assert last["patch_cleanup_status"] == "recovery_required"
        assert last["patch_cleanup_action"] == "revert"
        assert last["integration_error_class"] == "collective_apply_checkpoint_invalid"
        assert checkpoint.exists()

    @pytest.mark.asyncio
    async def test_integrate_collective_rejects_multiple_manifests(self, tmp_path):
        """Multiple recovery manifests must require explicit review."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        integration_id = "multiple-manifests"
        campaign = _record_collective_campaign(
            coord,
            tmp_path,
            integration_id,
        )
        backup_root, _, _ = _collective_recovery_paths(
            tmp_path,
            integration_id,
        )
        for name in ("first", "second"):
            manifest = backup_root / name / "manifest.json"
            manifest.parent.mkdir(parents=True)
            manifest.write_text(
                json.dumps({"status": "applied"}),
                encoding="utf-8",
            )

        await phase._integrate_collective(campaign)

        last = coord.shared_state.last_collective
        assert last["integration_decision"] == "NEEDS_REVIEW"
        assert last["patch_cleanup_status"] == "recovery_required"
        assert last["integration_error_class"] == "collective_apply_manifest_ambiguous"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "manifest_status, expected_error, expected_reverts",
        [
            ("reverted", "collective_apply_already_reverted", 0),
            ("prepared", "collective_apply_not_resumable", 1),
            ("failed", "collective_apply_not_resumable", 1),
            ("reverted_partial", "collective_apply_not_resumable", 1),
        ],
    )
    async def test_integrate_collective_recovers_terminal_and_failed_manifests(
        self,
        tmp_path,
        monkeypatch,
        manifest_status,
        expected_error,
        expected_reverts,
    ):
        """Known non-resumable manifests must finish through the revert path."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        integration_id = f"recover-{manifest_status}"
        campaign = _record_collective_campaign(
            coord,
            tmp_path,
            integration_id,
        )
        _write_collective_recovery(
            tmp_path,
            integration_id,
            manifest_status,
        )
        reverts: list[dict] = []

        def _revert(apply_result):
            """Capture recovery reverts and report completion."""
            reverts.append(apply_result)
            return {"status": "ok"}

        monkeypatch.setattr(
            krh_mod,
            "_maybe_revert_kernel_patch",
            _revert,
        )

        await phase._integrate_collective(campaign)

        last = coord.shared_state.last_collective
        assert last["integration_decision"] == "REVERT"
        assert last["patch_cleanup_status"] == "complete"
        assert last["integration_error_class"] == expected_error
        assert len(reverts) == expected_reverts

    @pytest.mark.asyncio
    async def test_integrate_collective_preserves_unknown_manifest_for_review(self, tmp_path, monkeypatch):
        """An unknown manifest status must remain recovery-required."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        integration_id = "unknown-manifest"
        campaign = _record_collective_campaign(
            coord,
            tmp_path,
            integration_id,
        )
        _write_collective_recovery(
            tmp_path,
            integration_id,
            "unexpected",
        )
        monkeypatch.setattr(
            krh_mod,
            "_maybe_revert_kernel_patch",
            lambda _apply: {"status": "ok"},
        )

        await phase._integrate_collective(campaign)

        last = coord.shared_state.last_collective
        assert last["integration_decision"] == "NEEDS_REVIEW"
        assert last["patch_cleanup_status"] == "recovery_required"
        assert last["patch_cleanup_action"] == "revert"
        assert last["integration_error_class"] == "collective_apply_not_resumable"

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "manifest_status, finalize_status",
        [
            ("finalized", "ok"),
            ("finalized_partial", "partial"),
        ],
    )
    async def test_integrate_collective_resumes_finalized_keep(
        self,
        tmp_path,
        monkeypatch,
        manifest_status,
        finalize_status,
    ):
        """A finalized KEEP must promote without repeating finalization."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        integration_id = f"finalize-{manifest_status}"
        campaign = _record_collective_campaign(
            coord,
            tmp_path,
            integration_id,
            integration_status="recovery_required",
            integration_recovery_action="finalize",
            integration_decision="KEEP",
            integration_result_status="ok",
            integration_gain_pct=25.0,
            integration_base_tput=100.0,
            integration_new_tput=125.0,
            integration_workspace=str(tmp_path / "integrate"),
        )
        _write_collective_recovery(
            tmp_path,
            integration_id,
            manifest_status,
        )

        def _unexpected_finalize(_apply):
            """Fail if a finalized manifest is finalized again."""
            raise AssertionError("finalization must not be repeated")

        monkeypatch.setattr(
            krh_mod,
            "_maybe_finalize_kernel_patch",
            _unexpected_finalize,
        )

        await phase._integrate_collective(campaign)

        last = coord.shared_state.last_collective
        assert last["integration_decision"] == "KEEP"
        assert last["patch_cleanup_status"] == "complete"
        assert last["integration_finalize_status"] == finalize_status
        assert coord.shared_state.current_best["tput"] == 125.0

    @pytest.mark.asyncio
    async def test_integrate_collective_reverts_missing_patch(self, tmp_path):
        """A KEEP without a patch must become a complete REVERT."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        campaign = _record_collective_campaign(
            coord,
            tmp_path,
            "missing-patch",
            patch="",
        )

        await phase._integrate_collective(campaign)

        last = coord.shared_state.last_collective
        assert last["integration_decision"] == "REVERT"
        assert last["patch_cleanup_status"] == "complete"
        assert last["integration_error_class"] == "collective_patch_missing"

    @pytest.mark.asyncio
    async def test_integrate_collective_records_snapshot_failure(self, tmp_path, monkeypatch):
        """Snapshot materialization failures must become durable REVERTs."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        campaign = _record_collective_campaign(
            coord,
            tmp_path,
            "snapshot-failure",
            snapshot_dir="",
        )

        def _raise_snapshot(**_kwargs):
            """Raise a deterministic snapshot failure."""
            raise RuntimeError("snapshot failed")

        monkeypatch.setattr(
            krh_mod,
            "materialize_unified_patch_snapshot",
            _raise_snapshot,
        )

        await phase._integrate_collective(campaign)

        last = coord.shared_state.last_collective
        assert last["integration_decision"] == "REVERT"
        assert last["integration_error_class"] == "RuntimeError"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("threshold", ["nan", "-1", "invalid"])
    async def test_integrate_collective_rejects_invalid_keep_threshold(self, tmp_path, monkeypatch, threshold):
        """Invalid KEEP thresholds must fail before the E2E handler runs."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        campaign = _record_collective_campaign(
            coord,
            tmp_path,
            f"threshold-{threshold}",
        )

        async def _unexpected_integrate(*_args, **_kwargs):
            """Fail if an invalid threshold reaches integration."""
            raise AssertionError("integration must not run")

        monkeypatch.setenv("HYPERLOOM_COLLECTIVE_KEEP_PCT", threshold)
        monkeypatch.setattr(
            krh_mod,
            "integrate_handler",
            _unexpected_integrate,
        )

        await phase._integrate_collective(campaign)

        last = coord.shared_state.last_collective
        assert last["integration_decision"] == "REVERT"
        assert last["patch_cleanup_status"] == "complete"
        assert last["integration_error_class"] == "ValueError"

    @pytest.mark.asyncio
    async def test_integrate_collective_rejects_non_mapping_handler_result(self, tmp_path, monkeypatch):
        """A non-mapping E2E result must become a complete REVERT."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        campaign = _record_collective_campaign(
            coord,
            tmp_path,
            "handler-not-mapping",
        )

        async def _invalid_result(*_args, **_kwargs):
            """Return an invalid integration result."""
            return ["invalid"]

        monkeypatch.setattr(
            krh_mod,
            "integrate_handler",
            _invalid_result,
        )

        await phase._integrate_collective(campaign)

        last = coord.shared_state.last_collective
        assert last["integration_decision"] == "REVERT"
        assert last["patch_cleanup_status"] == "complete"
        assert last["integration_error_class"] == "TypeError"

    @pytest.mark.asyncio
    async def test_integrate_collective_marks_invalid_decision_for_review(self, tmp_path, monkeypatch):
        """An unknown E2E decision must require explicit recovery review."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        campaign = _record_collective_campaign(
            coord,
            tmp_path,
            "invalid-decision",
        )

        async def _invalid_decision(*_args, **_kwargs):
            """Return an unsupported integration decision."""
            return {"status": "ok", "decision": "MAYBE"}

        monkeypatch.setattr(
            krh_mod,
            "integrate_handler",
            _invalid_decision,
        )

        await phase._integrate_collective(campaign)

        last = coord.shared_state.last_collective
        assert last["integration_decision"] == "NEEDS_REVIEW"
        assert last["patch_cleanup_status"] == "recovery_required"
        assert last["integration_error_class"] == "collective_integration_decision_invalid"

    @pytest.mark.asyncio
    async def test_integrate_collective_preserves_incomplete_revert(self, tmp_path, monkeypatch):
        """An incomplete revert must retain its recovery action."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        coord.bus = _Bus()
        phase = KernelPhase(coord)
        campaign = _record_collective_campaign(
            coord,
            tmp_path,
            "incomplete-revert",
        )
        manifest = tmp_path / "active-manifest.json"

        async def _revert_result(*_args, **_kwargs):
            """Return a REVERT whose patch cleanup is incomplete."""
            return {
                "status": "ok",
                "decision": "REVERT",
                "apply_result": {
                    "status": "ok",
                    "manifest_path": str(manifest),
                },
                "revert_result": {"status": "failed"},
            }

        monkeypatch.setattr(
            krh_mod,
            "integrate_handler",
            _revert_result,
        )
        monkeypatch.setattr(
            krh_mod,
            "_maybe_revert_kernel_patch",
            lambda _apply: {"status": "failed"},
        )

        await phase._integrate_collective(campaign)

        last = coord.shared_state.last_collective
        assert last["integration_decision"] == "REVERT"
        assert last["patch_cleanup_status"] == "recovery_required"
        assert last["patch_cleanup_action"] == "revert"
        assert last["integration_revert_status"] == "failed"

    @pytest.mark.asyncio
    async def test_integrate_collective_rolls_back_failed_promotion(self, tmp_path, monkeypatch):
        """Promotion failure must restore state and revert the applied patch."""
        coord = _coord(
            tmp_path,
            baseline_tput=100.0,
            current_best={"engine": "existing", "tput": 110.0},
            cumulative_gain_validated=10.0,
        )
        coord.bus = _Bus()
        coord.shared_state.optimization_stack = [{"action": "existing", "variant_name": "existing"}]
        coord.shared_state.gain_per_stack_entry = [10.0]
        phase = KernelPhase(coord)
        campaign = _record_collective_campaign(
            coord,
            tmp_path,
            "promotion-failure",
        )
        manifest = tmp_path / "promotion-manifest.json"

        async def _invalid_keep(*_args, **_kwargs):
            """Return a KEEP with an invalid promoted throughput."""
            return {
                "status": "ok",
                "decision": "KEEP",
                "new_tput": 0.0,
                "gain_pct": 10.0,
                "apply_result": {
                    "status": "ok",
                    "manifest_path": str(manifest),
                },
            }

        monkeypatch.setattr(
            krh_mod,
            "integrate_handler",
            _invalid_keep,
        )
        monkeypatch.setattr(
            krh_mod,
            "_maybe_revert_kernel_patch",
            lambda _apply: {"status": "ok"},
        )

        await phase._integrate_collective(campaign)

        last = coord.shared_state.last_collective
        assert last["integration_decision"] == "REVERT"
        assert last["patch_cleanup_status"] == "complete"
        assert last["integration_error_class"] == "collective_promotion_invalid"
        assert coord.shared_state.current_best["engine"] == "existing"
        assert len(coord.shared_state.optimization_stack) == 1
        assert coord.shared_state.gain_per_stack_entry == [10.0]

    @pytest.mark.asyncio
    async def test_integrate_collective_tolerates_bus_failure(self, tmp_path):
        """An integration bus failure must not discard the terminal verdict."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        phase = KernelPhase(coord)

        class _FailingBus:
            async def append_and_seq(self, _message):
                """Raise a deterministic bus failure."""
                raise RuntimeError("bus unavailable")

        coord.bus = _FailingBus()
        campaign = _record_collective_campaign(
            coord,
            tmp_path,
            "integration-bus-failure",
            patch="",
        )

        await phase._integrate_collective(campaign)

        assert coord.shared_state.last_collective["patch_cleanup_status"] == "complete"
        assert coord.shared_state.last_collective["integration_decision"] == "REVERT"

    def test_promote_collective_ignores_non_keep(self, tmp_path):
        """A non-KEEP integration must not mutate promoted state."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        phase = KernelPhase(coord)

        phase._promote_collective_integrate_keep(
            {"integration_id": "not-keep", "patch": "/tmp/test.patch"},
            {"status": "ok", "decision": "REVERT"},
        )

        assert coord.shared_state.optimization_stack == []

    @pytest.mark.parametrize(
        "field, value, error",
        [
            ("status", "failed", "successful integration"),
            ("apply_result", {}, "apply manifest"),
            ("new_tput", True, "numeric E2E measurements"),
            ("new_tput", 0.0, "new_tput must be positive"),
            ("gain_pct", 0.0, "gain_pct must be positive"),
            ("patch", "", "missing patch_path"),
            ("integration_id", "", "missing integration_id"),
        ],
    )
    def test_promote_collective_rejects_invalid_keep(self, tmp_path, field, value, error):
        """Invalid KEEP promotion inputs must fail before state mutation."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        phase = KernelPhase(coord)
        collective = {
            "integration_id": "promote-guard",
            "patch": "/tmp/collective.patch",
        }
        integrate = {
            "status": "ok",
            "decision": "KEEP",
            "new_tput": 120.0,
            "gain_pct": 20.0,
            "apply_result": {
                "status": "ok",
                "manifest_path": "/tmp/manifest.json",
            },
        }
        target = collective if field in {"patch", "integration_id"} else integrate
        target[field] = value

        with pytest.raises(ValueError, match=error):
            phase._promote_collective_integrate_keep(
                collective,
                integrate,
            )

        assert coord.shared_state.optimization_stack == []

    @pytest.mark.parametrize(
        "collective, integrate",
        [
            ([], {}),
            ({}, []),
        ],
    )
    def test_promote_collective_requires_mapping_inputs(self, tmp_path, collective, integrate):
        """Promotion must reject non-mapping inputs."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        phase = KernelPhase(coord)

        with pytest.raises(TypeError, match="inputs must be mappings"):
            phase._promote_collective_integrate_keep(
                collective,
                integrate,
            )

    @pytest.mark.asyncio
    async def test_collective_stage_resumes_pending_integration(self, tmp_path, monkeypatch):
        """A pending Collective integration must resume before a new campaign."""
        coord = _coord(tmp_path, baseline_tput=100.0)
        phase = KernelPhase(coord)
        campaign = _record_collective_campaign(
            coord,
            tmp_path,
            "pending-stage",
            integration_status="recovery_required",
            integration_recovery_action="revert",
        )
        resumed: list[dict] = []

        async def _resume(result):
            """Capture the pending integration selected for resume."""
            resumed.append(result)

        async def _unexpected_run():
            """Fail if pending recovery launches a new campaign."""
            raise AssertionError("new campaign must not run")

        monkeypatch.setattr(phase, "_integrate_collective", _resume)
        monkeypatch.setattr(phase, "_run_forge_collective", _unexpected_run)
        monkeypatch.setattr(phase, "_collective_only_mode", lambda: False)

        await phase._maybe_run_collective_before_kernel_opt()

        assert resumed == [coord.shared_state.last_collective]
        assert resumed[0]["integration_id"] == campaign["integration_id"]

    @pytest.mark.asyncio
    async def test_collective_only_stage_escalates_after_terminal_lane(self, tmp_path, monkeypatch):
        """Collective-only mode must hand off after its lane is terminal."""
        coord = _coord(tmp_path)
        phase = KernelPhase(coord)

        monkeypatch.setattr(
            phase,
            "_collective_required_before_kernel_opt",
            lambda: False,
        )
        monkeypatch.setattr(phase, "_collective_only_mode", lambda: True)

        await phase._maybe_run_collective_before_kernel_opt()

        assert coord.shared_state.pending_escalate_hint == kernel_phase_mod._phase_state.ESCALATE_HINT_SKIP_TO_SWEEP

    def test_collective_only_mode_validates_state_and_environment(self, tmp_path, monkeypatch):
        """Collective-only gating must accept env truth and reject bad state."""
        coord = _coord(tmp_path)
        phase = KernelPhase(coord)
        monkeypatch.setenv("HYPERLOOM_COLLECTIVE_ONLY", "yes")

        assert phase._collective_only_mode() is True

        coord.shared_state.collective_only_mode = "yes"
        with pytest.raises(ValueError, match="must be boolean"):
            phase._collective_only_mode()


class TestForgeGemmRuntimeConfigMerge:
    def test_merges_candidate_with_aiter_source_configs_when_runtime_cache_is_absent(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, framework="sglang")
        phase = KernelPhase(coord)
        aiter_root = tmp_path / "aiter-source"
        configs_dir = aiter_root / "aiter" / "configs"
        model_configs_dir = configs_dir / "model_configs"
        model_configs_dir.mkdir(parents=True)
        header = "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName\n"
        (configs_dir / "a8w8_blockscale_bpreshuffle_tuned_gemm.csv").write_text(
            header
            + "gfx950,256,16,512,7168,asm,1,1,10.0,base_kernel\n"
            + "gfx950,256,32,512,7168,asm,5,1,12.0,base_duplicate\n",
            encoding="utf-8",
        )
        (model_configs_dir / "qwen3_14b_a8w8_blockscale_bpreshuffle_tuned_gemm.csv").write_text(
            header + "gfx950,256,32,512,7168,asm,2,1,9.0,model_kernel\n",
            encoding="utf-8",
        )
        candidate = tmp_path / "candidate.csv"
        candidate.write_text(
            header
            + "gfx950,256,16,512,7168,asm,3,1,7.0,tuned_kernel\n"
            + "gfx950,256,64,512,7168,asm,4,1,8.0,new_kernel\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("AITER_ROOT_DIR", str(aiter_root))
        monkeypatch.setenv(
            "INFERENCE_OPTIMIZER_AITER_CONFIG_CACHE_DIR",
            str(tmp_path / "missing-runtime-cache"),
        )

        merged_path = phase._merge_gemm_candidate_with_runtime(
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE", str(candidate)
        )

        assert merged_path is not None
        with Path(merged_path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert len(rows) == 3
        assert {(row["M"], row["kernelName"]) for row in rows} == {
            ("16", "tuned_kernel"),
            ("32", "model_kernel"),
            ("64", "new_kernel"),
        }

    def test_merges_fmoe_candidate_by_full_untuned_dispatch_schema(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, framework="sglang")
        phase = KernelPhase(coord)
        aiter_root = tmp_path / "aiter-source"
        configs_dir = aiter_root / "aiter" / "configs"
        configs_dir.mkdir(parents=True)
        key_header = (
            "token,model_dim,inter_dim,expert,topk,act_type,dtype,q_dtype_a,q_dtype_w,q_type,use_g1u1,doweight_stage1"
        )
        tuned_header = f"gfx,cu_num,{key_header},kernelId,us,kernelName\n"
        (configs_dir / "untuned_fmoe.csv").write_text(f"{key_header}\n", encoding="utf-8")
        (configs_dir / "tuned_fmoe.csv").write_text(
            tuned_header + "gfx950,256,64,7168,2048,128,8,Silu,bf16,fp8,fp8,"
            "per_token,1,0,1,10.0,base_per_token\n" + "gfx950,256,64,7168,2048,128,8,Silu,bf16,fp8,fp8,"
            "per_tensor,1,0,2,11.0,base_per_tensor\n",
            encoding="utf-8",
        )
        candidate = tmp_path / "candidate_fmoe.csv"
        candidate.write_text(
            tuned_header + "gfx950,256,64,7168,2048,128,8,Silu,bf16,fp8,fp8,per_token,1,0,3,7.0,tuned_per_token\n",
            encoding="utf-8",
        )
        monkeypatch.setenv("AITER_ROOT_DIR", str(aiter_root))
        monkeypatch.setenv(
            "INFERENCE_OPTIMIZER_AITER_CONFIG_CACHE_DIR",
            str(tmp_path / "missing-runtime-cache"),
        )

        merged_path = phase._merge_gemm_candidate_with_runtime("AITER_CONFIG_FMOE", str(candidate))

        assert merged_path is not None
        with Path(merged_path).open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        assert {(row["q_type"], row["kernelName"]) for row in rows} == {
            ("per_token", "tuned_per_token"),
            ("per_tensor", "base_per_tensor"),
        }

    @pytest.mark.asyncio
    async def test_does_not_e2e_validate_sparse_aiter_candidate_without_base_configs(self, tmp_path, monkeypatch):
        coord = _coord(
            tmp_path,
            framework="sglang",
            baseline_tput=100.0,
            current_best={"tput": 100.0},
        )
        phase = KernelPhase(coord)
        candidate = tmp_path / "candidate.csv"
        candidate.write_text(
            "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName\ngfx950,256,16,512,7168,asm,3,1,7.0,tuned_kernel\n",
            encoding="utf-8",
        )
        fake = _make_integrate([{"decision": "KEEP", "new_tput": 110.0, "gain_pct": 10.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)
        monkeypatch.setattr("importlib.util.find_spec", lambda _name: None)
        monkeypatch.delenv("AITER_ROOT_DIR", raising=False)
        # The merge also probes the baked-in container config dir, which really
        # exists on an aiter image. Without redirecting it the candidate merges
        # against those configs and the "no base configs" premise never holds --
        # so this test passed only where /sgl-workspace/aiter was absent.
        monkeypatch.setattr(
            kernel_phase_mod,
            "_CONTAINER_AITER_CONFIG_DIR",
            tmp_path / "missing-container-configs",
        )
        monkeypatch.setenv(
            "INFERENCE_OPTIMIZER_AITER_CONFIG_CACHE_DIR",
            str(tmp_path / "missing-runtime-cache"),
        )
        # Point the last-resort container config dir at a non-existent path so the
        # "no complete aiter config anywhere" branch is exercised even on a dev
        # box that has the real /sgl-workspace/aiter checkout mounted.
        monkeypatch.setattr(
            "hyperloom.orchestrator.phases.kernel._CONTAINER_AITER_CONFIG_DIR",
            tmp_path / "missing-container-aiter-configs",
        )
        result = {
            "backend": "forge",
            "precision": "bf16",
            "tuners_run": [
                {
                    "status": "ok",
                    "candidate": True,
                    "tuner": "a8w8_blockscale_bpreshuffle",
                    "env_var": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE",
                    "env_value": str(candidate),
                }
            ],
        }

        await phase._validate_gemm_tuning_e2e(result)

        assert fake.calls == []
        assert result["e2e_results"]["reverted"][0]["reason"] == ("complete_aiter_config_unavailable")

    @pytest.mark.asyncio
    async def test_does_not_e2e_validate_missing_aiter_candidate(self, tmp_path, monkeypatch):
        coord = _coord(
            tmp_path,
            framework="sglang",
            baseline_tput=100.0,
            current_best={"tput": 100.0},
        )
        phase = KernelPhase(coord)
        missing_candidate = tmp_path / "missing-candidate.csv"
        fake = _make_integrate([{"decision": "KEEP", "new_tput": 110.0, "gain_pct": 10.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)
        result = {
            "backend": "forge",
            "precision": "fp8",
            "tuners_run": [
                {
                    "status": "ok",
                    "candidate": True,
                    "tuner": "a8w8_blockscale",
                    "env_var": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
                    "env_value": str(missing_candidate),
                }
            ],
        }

        await phase._validate_gemm_tuning_e2e(result)

        assert fake.calls == []
        assert result["e2e_results"]["reverted"][0]["reason"] == ("candidate_artifact_missing")

    @pytest.mark.asyncio
    async def test_integrate_bench_fault_not_recorded_as_zero_gain_revert(self, tmp_path, monkeypatch):
        """A server that never booted is an integrate fault, not a 0% REVERT."""
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        phase = KernelPhase(coord)
        fmoe_candidate = tmp_path / "fmoe.csv"
        dense_candidate = tmp_path / "dense.csv"
        fmoe_candidate.write_text("token,model_dim\n1,2\n", encoding="utf-8")
        dense_candidate.write_text("M,N,K\n1,2,3\n", encoding="utf-8")
        calls: list[dict] = []

        async def _fake_integrate(payload, *, session_dir):
            calls.append(payload)
            if payload["kernel_id"] == "gemm_tune_fmoe_ck":
                return {
                    "status": "failed",
                    "error_class": "bench_exception",
                    "decision": "REVERT",
                    "error": "re-baseline did not succeed",
                }
            return {"status": "ok", "decision": "KEEP", "new_tput": 120.0, "gain_pct": 9.09}

        monkeypatch.setattr(krh_mod, "integrate_handler", _fake_integrate)
        monkeypatch.setattr(explore_mod, "_compute_explore_variant_timeout", lambda **_k: 61)
        monkeypatch.setattr(
            phase,
            "_merge_gemm_candidate_with_runtime",
            lambda _env_var, env_value: env_value,
        )

        result = {
            "backend": "forge",
            "tuners_run": [
                {
                    "status": "ok",
                    "tuner": "fmoe_ck",
                    "improved_shapes": 2,
                    "env_var": "AITER_CONFIG_FMOE",
                    "env_value": str(fmoe_candidate),
                },
                {
                    "status": "ok",
                    "tuner": "dense_bf16",
                    "improved_shapes": 1,
                    "env_var": "AITER_CONFIG_DENSE",
                    "env_value": str(dense_candidate),
                },
            ],
        }

        await phase._validate_gemm_tuning_e2e(result)

        assert len(calls) == 3
        assert result["e2e_results"]["faults"][0]["reason"] == "integrate_fault:bench_exception"
        assert result["e2e_results"]["faults"][0]["fault_attempts"] == 2
        assert result["e2e_results"]["reverted"] == []
        assert result["e2e_results"]["kept"][0]["tuner"] == "dense_bf16"
        assert result["decision"] == "KEEP"

    @pytest.mark.asyncio
    async def test_integrate_fault_retries_once_before_verdict(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        phase = KernelPhase(coord)
        dense_candidate = tmp_path / "dense.csv"
        dense_candidate.write_text("M,N,K\n1,2,3\n", encoding="utf-8")
        calls: list[dict] = []

        async def _fake_integrate(payload, *, session_dir):
            calls.append(payload)
            if len(calls) == 1:
                return {
                    "status": "failed",
                    "error_class": "bench_exception",
                    "decision": "REVERT",
                    "error": "re-baseline did not succeed",
                }
            return {"status": "ok", "decision": "KEEP", "new_tput": 110.0, "gain_pct": 10.0}

        monkeypatch.setattr(krh_mod, "integrate_handler", _fake_integrate)
        monkeypatch.setattr(
            phase,
            "_merge_gemm_candidate_with_runtime",
            lambda _env_var, env_value: env_value,
        )
        result = {
            "backend": "forge",
            "tuners_run": [
                {
                    "status": "ok",
                    "tuner": "dense_bf16",
                    "improved_shapes": 1,
                    "env_var": "AITER_CONFIG_DENSE",
                    "env_value": str(dense_candidate),
                },
            ],
        }

        await phase._validate_gemm_tuning_e2e(result)

        assert len(calls) == 2
        assert result["e2e_results"]["faults"] == []
        assert result["e2e_results"]["kept"][0]["tuner"] == "dense_bf16"
        assert result["decision"] == "KEEP"

    @pytest.mark.asyncio
    async def test_a_stopped_run_leaves_its_tuners_unjudged(self, tmp_path, monkeypatch):
        """A clock that ran out is not a verdict on the tuners it interrupted."""
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        phase = KernelPhase(coord)
        first = tmp_path / "fmoe.csv"
        second = tmp_path / "dense.csv"
        first.write_text("token,model_dim\n1,2\n", encoding="utf-8")
        second.write_text("M,N,K\n1,2,3\n", encoding="utf-8")
        calls: list[dict] = []

        async def _fake_integrate(payload, *, session_dir):
            calls.append(payload)
            return {
                "status": "failed",
                "error_class": "session_time_exhausted",
                "decision": "NEEDS_REVIEW",
            }

        monkeypatch.setattr(krh_mod, "integrate_handler", _fake_integrate)
        monkeypatch.setattr(explore_mod, "_compute_explore_variant_timeout", lambda **_k: 61)
        monkeypatch.setattr(
            phase,
            "_merge_gemm_candidate_with_runtime",
            lambda _env_var, env_value: env_value,
        )

        result = {
            "backend": "forge",
            "tuners_run": [
                {
                    "status": "ok",
                    "tuner": "fmoe_ck",
                    "improved_shapes": 1,
                    "env_var": "AITER_CONFIG_FMOE",
                    "env_value": str(first),
                },
                {
                    "status": "ok",
                    "tuner": "dense_bf16",
                    "improved_shapes": 1,
                    "env_var": "AITER_CONFIG_DENSE",
                    "env_value": str(second),
                },
            ],
        }

        await phase._validate_gemm_tuning_e2e(result)

        assert len(calls) == 1
        assert result["e2e_results"]["kept"] == []
        assert result["e2e_results"]["reverted"] == []
        assert coord.shared_state.optimization_stack == []

    @pytest.mark.asyncio
    async def test_stacks_keeps_and_reverts(self, tmp_path, monkeypatch):
        coord = _coord(
            tmp_path,
            baseline_tput=100.0,
            baseline_runtime_sec=10.0,
            framework="sglang",
            current_best={"action": "warm_replay", "tput": 110.0},
        )
        phase = KernelPhase(coord)
        fmoe_candidate = tmp_path / "fmoe.csv"
        dense_candidate = tmp_path / "dense.csv"
        fmoe_candidate.write_text("token,model_dim\n1,2\n", encoding="utf-8")
        dense_candidate.write_text("M,N,K\n1,2,3\n", encoding="utf-8")
        calls: list[dict] = []
        responses = [
            {"decision": "KEEP", "new_tput": 130.0, "gain_pct": 18.18},
            {"decision": "REVERT", "new_tput": 125.0, "gain_pct": -3.8},
        ]

        async def _fake_integrate(payload, *, session_dir):
            assert session_dir == tmp_path
            calls.append(payload)
            return responses[len(calls) - 1]

        monkeypatch.setattr(krh_mod, "integrate_handler", _fake_integrate)
        monkeypatch.setattr(explore_mod, "_compute_explore_variant_timeout", lambda **_k: 61)
        monkeypatch.setattr(
            phase,
            "_merge_gemm_candidate_with_runtime",
            lambda _env_var, env_value: env_value,
        )

        result = {
            "backend": "forge",
            "precision": "bf16",
            "workspace": str(tmp_path / "gemm"),
            "recommended_env": {"AITER_CONFIG_FMOE": "/raw.csv"},
            "extra_envs": {"AITER_CONFIG_FMOE": "/raw.csv"},
            "tuners_run": [
                {
                    "status": "ok",
                    "tuner": "fmoe_ck",
                    "improved_shapes": 2,
                    "env_var": "AITER_CONFIG_FMOE",
                    "env_value": str(fmoe_candidate),
                    "best_micro_speedup": 1.2,
                },
                {
                    "status": "ok",
                    "tuner": "dense_bf16",
                    "improved_shapes": 1,
                    "env_var": "AITER_CONFIG_DENSE",
                    "env_value": str(dense_candidate),
                    "best_micro_speedup": 1.1,
                },
                {"status": "failed", "tuner": "ignored"},
                {"status": "ok", "tuner": "no_env", "improved_shapes": 1},
            ],
        }

        await phase._validate_gemm_tuning_e2e(result)

        assert [c["kernel_id"] for c in calls] == [
            "gemm_tune_fmoe_ck",
            "gemm_tune_dense_bf16",
        ]
        assert calls[0]["base_tput"] == 110.0
        assert calls[0]["extra_server_args"] == "--moe-runner-backend aiter"
        assert calls[0]["extra_envs"] == {"AITER_CONFIG_FMOE": str(fmoe_candidate)}
        assert calls[0]["budget_minutes"] == 2
        assert calls[1]["base_tput"] == 130.0
        assert calls[1]["extra_envs"] == {
            "AITER_CONFIG_FMOE": str(fmoe_candidate),
            "AITER_CONFIG_DENSE": str(dense_candidate),
        }
        assert coord.shared_state.current_best["variant_name"] == "forge_fmoe_ck"
        assert coord.shared_state.current_best["tput"] == 130.0
        assert coord.shared_state.optimization_stack[0]["variant_name"] == "forge_fmoe_ck"
        assert coord.shared_state.optimization_stack[0]["backend"] == "forge"
        assert result["decision"] == "KEEP"
        assert result["recommended_env"] == {"AITER_CONFIG_FMOE": str(fmoe_candidate)}
        assert result["e2e_results"]["kept"][0]["tuner"] == "fmoe_ck"
        assert result["e2e_results"]["reverted"][0]["tuner"] == "dense_bf16"

    @pytest.mark.asyncio
    async def test_handles_no_candidates_without_rewriting_raw_result(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        phase = KernelPhase(coord)
        monkeypatch.setattr(
            KernelPhase,
            "_ck_blockscale_switch_eligible",
            lambda self, result: False,
        )
        result = {
            "backend": "forge",
            "precision": "bf16",
            "recommended_env": {"AITER_CONFIG": "/raw.csv"},
            "extra_envs": {"AITER_CONFIG": "/raw.csv"},
            "tuners_run": [
                {"status": "failed", "tuner": "bad"},
                {"status": "ok", "tuner": "zero", "improved_shapes": 0},
            ],
        }

        await phase._validate_gemm_tuning_e2e(result)

        assert result["recommended_env"] == {"AITER_CONFIG": "/raw.csv"}
        assert coord.shared_state.optimization_stack == []

    @pytest.mark.asyncio
    async def test_records_integrate_exception_as_fault(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        phase = KernelPhase(coord)
        dense_candidate = tmp_path / "dense.csv"
        dense_candidate.write_text("M,N,K\n1,2,3\n", encoding="utf-8")

        async def _raise_integrate(*_args, **_kwargs):
            raise RuntimeError("integrate failed")

        calls: list[str] = []

        async def _counting_raise(*_args, **_kwargs):
            calls.append("boom")
            raise RuntimeError("integrate failed")

        monkeypatch.setattr(krh_mod, "integrate_handler", _counting_raise)
        monkeypatch.setattr(
            phase,
            "_merge_gemm_candidate_with_runtime",
            lambda _env_var, env_value: env_value,
        )
        result = {
            "backend": "forge",
            "precision": "bf16",
            "tuners_run": [
                {
                    "status": "ok",
                    "tuner": "dense_bf16",
                    "improved_shapes": 1,
                    "env_var": "AITER_CONFIG_DENSE",
                    "env_value": str(dense_candidate),
                },
            ],
        }

        await phase._validate_gemm_tuning_e2e(result)

        assert result["status"] == "failed"
        assert result["micro_decision"] == "integrate_fault"
        assert result["e2e_gain_pct"] is None
        fault = result["e2e_results"]["faults"][0]
        assert fault["reason"] == "integrate_fault:handler_exception"
        assert fault["fault"] is True
        assert fault["fault_attempts"] == 2
        assert len(calls) == 2
        assert result["e2e_results"]["reverted"] == []


class TestBf16DenseFallback:
    def test_fallback_predicate_requires_forge_sglang_fp8_no_candidate(self, tmp_path):
        coord = _coord(tmp_path, framework="sglang")

        assert coord._should_run_bf16_dense_gemm_fallback(
            {
                "backend": "forge",
                "precision": "fp8",
                "framework": "sglang",
                "micro_decision": "no_improvement",
                "tuners_run": [
                    {
                        "status": "no_improvement",
                        "tuner": "a8w8",
                        "improved_shapes": 0,
                    }
                ],
            }
        )

    def test_fallback_predicate_skips_existing_candidate(self, tmp_path):
        coord = _coord(tmp_path, framework="sglang")

        assert not coord._should_run_bf16_dense_gemm_fallback(
            {
                "backend": "forge",
                "precision": "fp8",
                "framework": "sglang",
                "micro_decision": "candidate",
                "recommended_env": {"AITER_CONFIG_GEMM_A8W8": "/tmp/tuned.csv"},
                "tuners_run": [
                    {
                        "status": "ok",
                        "tuner": "a8w8",
                        "improved_shapes": 4,
                        "env_var": "AITER_CONFIG_GEMM_A8W8",
                        "env_value": "/tmp/tuned.csv",
                    }
                ],
            }
        )

    def test_fallback_predicate_skips_candidate_reverted_by_e2e(self, tmp_path):
        coord = _coord(tmp_path, framework="sglang")

        assert not coord._should_run_bf16_dense_gemm_fallback(
            {
                "backend": "forge",
                "precision": "fp8",
                "framework": "sglang",
                "micro_decision": "candidate_no_e2e_gain",
                "e2e_validated": True,
                "e2e_results": {"kept": [], "reverted": [{"tuner": "a8w8"}]},
            }
        )

    def test_fallback_pending_resumes_terminal_fp8_no_candidate(self, tmp_path, monkeypatch):
        coord = _coord(
            tmp_path,
            framework="sglang",
            precision="fp8",
            last_gemm_tuning={
                "status": "ok",
                "decision": "REVERT",
                "backend": "forge",
                "precision": "fp8",
                "framework": "sglang",
                "micro_decision": "no_improvement",
                "tuners_run": [{"status": "no_improvement", "tuner": "a8w8"}],
            },
        )
        monkeypatch.setattr(krh_mod, "_resolve_gemm_tuning_backend", lambda _p: "forge")

        assert coord._bf16_dense_gemm_fallback_pending() is True
        assert coord._gemm_tuning_required_before_kernel_opt() is True

        coord.shared_state.gemm_tuning_attempts.append(
            {
                "status": "complete",
                "decision": "REVERT",
                "backend": "forge",
                "precision": "bf16",
                "workspace": str(tmp_path / "runs/gemm_tuning/kernel_entry_gemm_tuning_bf16_fallback"),
                "tuners_run": [{"tuner": "sglang_dense_bf16"}],
            }
        )

        assert coord._bf16_dense_gemm_fallback_pending() is False
        assert coord._gemm_tuning_required_before_kernel_opt() is False

    @pytest.mark.asyncio
    async def test_kernel_entry_runs_bf16_dense_fallback_after_fp8_no_improvement(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, framework="sglang")
        coord.bus = type(
            "Bus",
            (),
            {"append_and_seq": staticmethod(lambda *_args, **_kwargs: None)},
        )()

        async def _append_and_seq(*_args, **_kwargs):
            return None

        coord.bus.append_and_seq = _append_and_seq
        coord.phase_machine._kernel_enabled = lambda: True
        coord.phase_kernel._geak_enabled = lambda: False
        coord._gemm_tuning_required_before_kernel_opt = lambda: True
        coord.phase_machine._record_phase_entry_evidence = lambda **_kwargs: None
        coord.phase_kernel._kernel_opt_work_remains = lambda: False

        async def _noop(*_args, **_kwargs):
            return None

        coord.phase_kernel._maybe_reprofile_for_kernel = _noop

        calls: list[dict] = []
        responses = [
            {
                "status": "ok",
                "decision": "REVERT",
                "backend": "forge",
                "engine": "forge",
                "precision": "fp8",
                "framework": "sglang",
                "micro_decision": "no_improvement",
                "tuners_run": [
                    {
                        "status": "no_improvement",
                        "tuner": "a8w8",
                        "improved_shapes": 0,
                    }
                ],
            },
            {
                "status": "ok",
                "decision": "REVERT",
                "backend": "forge",
                "engine": "forge",
                "precision": "bf16",
                "framework": "sglang",
                "micro_decision": "no_improvement",
            },
        ]

        async def _fake_run_gemm(payload, *, session_dir):
            assert session_dir == tmp_path
            calls.append(payload)
            return dict(responses[len(calls) - 1])

        monkeypatch.setattr(krh_mod, "run_gemm_tuning_handler", _fake_run_gemm)

        await coord._on_enter_kernel(from_phase="FRAMEWORK_AGENT")

        assert [c["task_id"] for c in calls] == [
            "kernel_entry_gemm_tuning",
            "kernel_entry_gemm_tuning_bf16_fallback",
        ]
        assert calls[1]["precision"] == "bf16"
        assert calls[1]["tuner"] == "sglang_dense_bf16"
        assert coord.shared_state.last_gemm_tuning["precision"] == "bf16"

    @pytest.mark.asyncio
    async def test_kernel_entry_resumes_pending_bf16_fallback_without_rerunning_fp8(self, tmp_path, monkeypatch):
        coord = _coord(
            tmp_path,
            framework="sglang",
            precision="fp8",
            last_gemm_tuning={
                "status": "ok",
                "decision": "REVERT",
                "backend": "forge",
                "precision": "fp8",
                "framework": "sglang",
                "micro_decision": "no_improvement",
                "tuners_run": [{"status": "no_improvement", "tuner": "a8w8"}],
            },
        )
        coord.bus = type("Bus", (), {})()

        async def _append_and_seq(*_args, **_kwargs):
            return None

        async def _noop(*_args, **_kwargs):
            return None

        coord.bus.append_and_seq = _append_and_seq
        coord.phase_machine._kernel_enabled = lambda: True
        coord.phase_kernel._geak_enabled = lambda: False
        coord.phase_machine._record_phase_entry_evidence = lambda **_kwargs: None
        coord.phase_kernel._kernel_opt_work_remains = lambda: False
        coord.phase_kernel._maybe_reprofile_for_kernel = _noop
        monkeypatch.setattr(krh_mod, "_resolve_gemm_tuning_backend", lambda _p: "forge")

        calls: list[dict] = []

        async def _fake_run_gemm(payload, *, session_dir):
            assert session_dir == tmp_path
            calls.append(payload)
            return {
                "status": "ok",
                "decision": "REVERT",
                "backend": "forge",
                "engine": "forge",
                "precision": "bf16",
                "framework": "sglang",
                "micro_decision": "no_improvement",
            }

        monkeypatch.setattr(krh_mod, "run_gemm_tuning_handler", _fake_run_gemm)

        await coord._on_enter_kernel(from_phase="FRAMEWORK_AGENT")

        assert [c["task_id"] for c in calls] == ["kernel_entry_gemm_tuning_bf16_fallback"]
        assert calls[0]["precision"] == "bf16"
        assert coord.shared_state.last_gemm_tuning["task_id"] == ("kernel_entry_gemm_tuning_bf16_fallback")
        assert coord._bf16_dense_gemm_fallback_pending() is False

    @pytest.mark.asyncio
    async def test_bf16_fallback_failure_is_recorded_as_attempt(self, tmp_path):
        coord = _coord(tmp_path, framework="sglang")

        async def _raise(*_args, **_kwargs):
            raise RuntimeError("fallback boom")

        result = await coord._run_bf16_dense_gemm_fallback(_raise)

        assert result["status"] == "failed"
        assert result["decision"] == "REVERT"
        assert result["task_id"] == "kernel_entry_gemm_tuning_bf16_fallback"
        assert result["source"] == "fp8_no_improvement_bf16_fallback"
        assert result["precision"] == "bf16"
        assert coord._is_bf16_dense_gemm_fallback_attempt(result) is True


def _eligible_coord(tmp_path, monkeypatch, **overrides):
    """Coordinator wired for a CK-switch-eligible forge workload.

    forge + sglang + fp8 + gfx942 (mi300x) + block-scale fp8. The block-scale
    probe is forced to ``True`` unless overridden.
    """
    kwargs = dict(
        baseline_tput=100.0,
        framework="sglang",
        precision="fp8",
        gpu_type="mi300x",
        model_path="/models/blockscale-fp8",
    )
    kwargs.update(overrides)
    coord = _coord(tmp_path, **kwargs)
    monkeypatch.setattr(mcu_mod, "_fp8_is_block_scale", lambda _p: True)
    return coord


class TestCkBlockscaleSwitchEligible:
    """``_ck_blockscale_switch_eligible`` gates the CK backend switch to
    forge + sglang + fp8 + gfx942 + block-scale checkpoints."""

    def test_eligible_for_forge_sglang_fp8_mi300x_blockscale(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch)
        assert coord._ck_blockscale_switch_eligible({"backend": "forge"}) is True

    def test_not_eligible_non_forge_backend(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch)
        assert coord._ck_blockscale_switch_eligible({"backend": "geak"}) is False

    def test_not_eligible_non_sglang(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch, framework="vllm")
        assert coord._ck_blockscale_switch_eligible({"backend": "forge"}) is False

    def test_not_eligible_non_fp8(self, tmp_path, monkeypatch):
        # Non-fp8 session precision and no runtime fp8 signal -> not eligible.
        coord = _eligible_coord(tmp_path, monkeypatch, precision="bf16")
        monkeypatch.setattr(krh_mod, "_resolve_forge_precision_and_quant", lambda _s, _p: ("bf16", "auto"))
        assert coord._ck_blockscale_switch_eligible({"backend": "forge"}) is False

    def test_not_eligible_non_gfx942_gpu(self, tmp_path, monkeypatch):
        # mi355x is a known AMD type but NOT in _GFX942_GPU_TYPES.
        coord = _eligible_coord(tmp_path, monkeypatch, gpu_type="mi355x")
        assert coord._ck_blockscale_switch_eligible({"backend": "forge"}) is False

    def test_not_eligible_non_block_scale_fp8(self, tmp_path, monkeypatch):
        # No weight_block_size, so the block-scale probe declines.
        coord = _eligible_coord(tmp_path, monkeypatch)
        monkeypatch.setattr(mcu_mod, "_fp8_is_block_scale", lambda _p: False)
        assert coord._ck_blockscale_switch_eligible({"backend": "forge"}) is False

    def test_eligible_for_runtime_fp8_via_result_precision(self, tmp_path, monkeypatch):
        # Session precision is bf16, but the forge result stamps runtime precision fp8.
        coord = _eligible_coord(tmp_path, monkeypatch, precision="bf16")
        monkeypatch.setattr(krh_mod, "_resolve_forge_precision_and_quant", lambda _s, _p: ("bf16", "auto"))
        assert coord._ck_blockscale_switch_eligible({"backend": "forge", "precision": "fp8"}) is True

    def test_eligible_for_runtime_fp8_via_quantization_arg(self, tmp_path, monkeypatch):
        # Runtime --quantization fp8 is resolved from server args.
        coord = _eligible_coord(tmp_path, monkeypatch, precision="bf16")
        monkeypatch.setattr(krh_mod, "_resolve_forge_precision_and_quant", lambda _s, _p: ("fp8", "auto"))
        assert coord._ck_blockscale_switch_eligible({"backend": "forge"}) is True

    def test_not_eligible_per_token_fp8(self, tmp_path, monkeypatch):
        # Per-channel/per-token fp8 carries no weight_block_size -> declined.
        coord = _eligible_coord(tmp_path, monkeypatch)
        monkeypatch.setattr(mcu_mod, "_fp8_is_block_scale", lambda _p: False)
        assert coord._ck_blockscale_switch_eligible({"backend": "forge"}) is False

    def test_non_dict_result_is_not_eligible(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch)
        assert coord._ck_blockscale_switch_eligible("nope") is False  # type: ignore[arg-type]


class TestCkBlockscaleCandidateInjection:
    """The fp8 block-scale CK switch enters as its own candidate to be measured."""

    def _forge_result(self, **overrides):
        result = {
            "status": "ok",
            "decision": "KEEP",
            "best_speedup": 1.2,
            "backend": "forge",
            "extra_envs": {"AITER_CONFIG": "/cfg/tuned.json"},
        }
        result.update(overrides)
        return result

    def _ck_candidates(self, coord, result):
        return [c for c in coord._gemm_e2e_candidates(result) if c["env_var"] == "SGLANG_FP8_BLOCKSCALE_CK_MAX_M"]

    def test_injects_for_forge_eligible_keep(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch)
        cands = self._ck_candidates(coord, self._forge_result())
        assert len(cands) == 1
        assert cands[0]["envs"] == {"SGLANG_FP8_BLOCKSCALE_CK_MAX_M": "256"}
        assert cands[0]["tuner"] == "ck_blockscale_backend_switch"

    def test_does_not_inject_for_geak_backend(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch)
        result = {
            "status": "ok",
            "decision": "KEEP",
            "best_speedup": 1.2,
            "backend": "geak",
            "tuned_file": "/tuned/gemm.csv",
        }
        assert self._ck_candidates(coord, result) == []
        assert [c["tuner"] for c in coord._gemm_e2e_candidates(result)] == ["a8w8_blockscale_tuned_gemm"]

    def test_does_not_inject_for_bf16_precision(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch, precision="bf16")
        assert self._ck_candidates(coord, self._forge_result()) == []

    def test_does_not_inject_for_non_sglang_framework(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch, framework="vllm")
        assert self._ck_candidates(coord, self._forge_result()) == []

    def test_does_not_inject_for_non_gfx942_gpu(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch, gpu_type="mi355x")
        assert self._ck_candidates(coord, self._forge_result()) == []

    def test_does_not_inject_for_non_block_scale_fp8(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch)
        monkeypatch.setattr(mcu_mod, "_fp8_is_block_scale", lambda _p: False)
        assert self._ck_candidates(coord, self._forge_result()) == []

    def test_does_not_double_inject_when_a_tuner_already_carries_the_switch(self, tmp_path, monkeypatch):
        coord = _eligible_coord(tmp_path, monkeypatch)
        result = self._forge_result(
            tuners_run=[
                {
                    "tuner": "blockscale",
                    "status": "ok",
                    "candidate": True,
                    "env_var": "SGLANG_FP8_BLOCKSCALE_CK_MAX_M",
                    "env_value": "512",
                    "best_micro_speedup": 1.4,
                }
            ]
        )
        cands = self._ck_candidates(coord, result)
        assert len(cands) == 1
        assert cands[0]["env_value"] == "512"


class TestHandleGemmTuningResult:
    @pytest.mark.asyncio
    async def test_forge_requires_e2e_routes_to_validator(self, tmp_path):
        coord = _coord(tmp_path, baseline_tput=100.0)
        called: dict[str, object] = {}

        async def _fake_validate(result):
            called["result"] = result

        coord.phase_kernel._validate_gemm_tuning_e2e = _fake_validate  # type: ignore[assignment]

        await coord._handle_gemm_tuning_result(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.3,
                "backend": "forge",
                "requires_e2e_validation": True,
                "extra_envs": {"AITER_CONFIG": "/cfg/tuned.json"},
            }
        )

        assert "result" in called
        # Validator owns promotion; inline promote must not have run.
        assert coord.shared_state.optimization_stack == []

    @pytest.mark.asyncio
    async def test_forge_e2e_rewrites_latest_attempt_history(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        fake = _make_integrate([{"decision": "REVERT", "new_tput": 90.0, "gain_pct": -10.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        await coord._handle_gemm_tuning_result(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.5,
                "backend": "forge",
                "engine": "forge",
                "requires_e2e_validation": True,
                "recommended_env": {"AITER_DENSE": "/dense.json"},
                "extra_envs": {"AITER_DENSE": "/dense.json"},
                "tuners_run": [
                    {
                        "status": "ok",
                        "improved_shapes": 3,
                        "tuner": "dense_gemm",
                        "env_var": "AITER_DENSE",
                        "env_value": "/dense.json",
                    }
                ],
            }
        )

        attempts = coord.shared_state.gemm_tuning_attempts
        assert len(attempts) == 1
        assert attempts[0]["engine"] == "forge"
        assert attempts[0]["e2e_validated"] is True
        assert attempts[0]["decision"] == "REVERT"
        assert attempts[0]["best_speedup"] == 1.5
        assert coord.shared_state.last_gemm_tuning["decision"] == "REVERT"

    @pytest.mark.asyncio
    async def test_forge_e2e_keep_names_the_artifact_the_stack_recorded(self, tmp_path, monkeypatch):
        """The history row and the stack entry must name the same artifact.

        The breakdown decides ``adopted`` by matching those two strings. Forge
        reports per-tuner envs and never set ``tuned_file``, so the history row
        carried "" and no KEEP could ever match -- measured across 419 real
        attempts, none was reported adopted.
        """
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        fake = _make_integrate([{"decision": "KEEP", "new_tput": 130.0, "gain_pct": 30.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        await coord._handle_gemm_tuning_result(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.5,
                "backend": "forge",
                "engine": "forge",
                "requires_e2e_validation": True,
                "recommended_env": {"AITER_DENSE": "/dense.json"},
                "extra_envs": {"AITER_DENSE": "/dense.json"},
                "tuners_run": [
                    {
                        "status": "ok",
                        "improved_shapes": 3,
                        "tuner": "dense_gemm",
                        "env_var": "AITER_DENSE",
                        "env_value": "/dense.json",
                    }
                ],
            }
        )

        stack = coord.shared_state.optimization_stack
        assert stack, "a KEEP must land on the stack"
        assert stack[-1]["action"] == "gemm_tuning"
        attempts = coord.shared_state.gemm_tuning_attempts
        assert attempts[0]["decision"] == "KEEP"
        assert attempts[0]["tuned_file"], "history row must name the artifact"
        assert attempts[0]["tuned_file"] == stack[-1]["tuned_file"]

    @pytest.mark.asyncio
    async def test_a_second_round_claims_its_own_artifact(self, tmp_path, monkeypatch):
        """Re-tuning the same tuner must not inherit the earlier round's path.

        ``_lift_to_current_best`` skips the stack append when
        ``(action, variant_name)`` already matches, and a GEMM variant is named
        ``<backend>_<tuner>`` -- so after a second macro cycle re-tunes the same
        tuner, the newest stack entry still describes round one. Taking the
        artifact from there would make the second attempt claim the first one's
        file, and with it the first one's gain.
        """
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        # Keep the candidate env value verbatim so each round's path is distinct
        # and the assertion is about provenance, not about merging.
        monkeypatch.setattr(
            KernelPhase,
            "_merge_gemm_candidate_with_runtime",
            lambda _self, _env_var, env_value: env_value,
        )

        def _result(env_value: str) -> dict:
            return {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.5,
                "backend": "forge",
                "engine": "forge",
                "requires_e2e_validation": True,
                "recommended_env": {"AITER_DENSE": env_value},
                "extra_envs": {"AITER_DENSE": env_value},
                "tuners_run": [
                    {
                        "status": "ok",
                        "improved_shapes": 3,
                        "tuner": "dense_gemm",
                        "env_var": "AITER_DENSE",
                        "env_value": env_value,
                    }
                ],
            }

        monkeypatch.setattr(
            krh_mod,
            "integrate_handler",
            _make_integrate([{"decision": "KEEP", "new_tput": 130.0, "gain_pct": 30.0}]),
        )
        await coord._handle_gemm_tuning_result(_result("/round1.json"))

        first_file = coord.shared_state.gemm_tuning_attempts[-1]["tuned_file"]
        assert first_file, "round one must name its artifact"
        stack_len = len(coord.shared_state.optimization_stack)

        monkeypatch.setattr(
            krh_mod,
            "integrate_handler",
            _make_integrate([{"decision": "KEEP", "new_tput": 160.0, "gain_pct": 23.1}]),
        )
        await coord._handle_gemm_tuning_result(_result("/round2.json"))

        # Same (action, variant_name): the append is skipped by design.
        assert len(coord.shared_state.optimization_stack) == stack_len
        second_file = coord.shared_state.gemm_tuning_attempts[-1]["tuned_file"]
        assert second_file and second_file != first_file

    @pytest.mark.asyncio
    async def test_forge_e2e_revert_does_not_claim_an_artifact(self, tmp_path, monkeypatch):
        """A REVERT has nothing on the stack, so it must not name one."""
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        fake = _make_integrate([{"decision": "REVERT", "new_tput": 90.0, "gain_pct": -10.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        await coord._handle_gemm_tuning_result(
            {
                "status": "ok",
                "decision": "KEEP",
                "backend": "forge",
                "engine": "forge",
                "requires_e2e_validation": True,
                "recommended_env": {"AITER_DENSE": "/dense.json"},
                "extra_envs": {"AITER_DENSE": "/dense.json"},
                "tuners_run": [
                    {
                        "status": "ok",
                        "improved_shapes": 3,
                        "tuner": "dense_gemm",
                        "env_var": "AITER_DENSE",
                        "env_value": "/dense.json",
                    }
                ],
            }
        )

        assert coord.shared_state.optimization_stack == []
        assert not coord.shared_state.gemm_tuning_attempts[0].get("tuned_file")

    @pytest.mark.asyncio
    async def test_forge_no_improvement_but_ck_eligible_routes_to_validator(self, tmp_path, monkeypatch):
        # a8w8 tuner reported no_improvement but the CK block-scale switch is
        # eligible → route to the E2E validator, not inline promote.
        coord = _eligible_coord(tmp_path, monkeypatch)
        called: dict[str, object] = {}

        async def _fake_validate(result):
            called["result"] = result

        coord.phase_kernel._validate_gemm_tuning_e2e = _fake_validate  # type: ignore[assignment]

        await coord._handle_gemm_tuning_result(
            {
                "status": "complete",
                "decision": "REVERT",
                "micro_decision": "no_improvement",
                "backend": "forge",
                "requires_e2e_validation": False,
            }
        )

        assert "result" in called
        assert coord.shared_state.optimization_stack == []

    @pytest.mark.asyncio
    async def test_geak_promotes_on_the_measured_tput_not_the_micro_speedup(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        tuned = tmp_path / "gemm.csv"
        tuned.write_text("token,model_dim\n1,2\n", encoding="utf-8")
        fake = _make_integrate([{"decision": "KEEP", "new_tput": 150.0, "gain_pct": 50.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)
        monkeypatch.setattr(
            KernelPhase,
            "_merge_gemm_candidate_with_runtime",
            lambda _self, _env_var, env_value: env_value,
        )

        await coord._handle_gemm_tuning_result(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.4,
                "backend": "geak",
                "tuned_file": str(tuned),
            }
        )

        assert len(fake.calls) == 1
        stack = coord.shared_state.optimization_stack
        assert len(stack) == 1
        assert stack[0]["variant_name"] == "geak_a8w8_blockscale_tuned_gemm"
        assert stack[0]["backend"] == "geak"
        # 150.0 measured, not baseline * best_speedup (140.0).
        assert stack[0]["tput"] == pytest.approx(150.0)
        assert coord.shared_state.current_best["tput"] == pytest.approx(150.0)
        assert coord.shared_state.cumulative_gain_validated == pytest.approx(50.0)

    @pytest.mark.asyncio
    async def test_geak_promotes_nothing_when_the_measurement_reverts(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        tuned = tmp_path / "gemm.csv"
        tuned.write_text("token,model_dim\n1,2\n", encoding="utf-8")
        fake = _make_integrate([{"decision": "REVERT", "new_tput": 90.0, "gain_pct": -10.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)
        monkeypatch.setattr(
            KernelPhase,
            "_merge_gemm_candidate_with_runtime",
            lambda _self, _env_var, env_value: env_value,
        )

        await coord._handle_gemm_tuning_result(
            {
                "status": "ok",
                "decision": "KEEP",
                "best_speedup": 1.4,
                "backend": "geak",
                "tuned_file": str(tuned),
            }
        )

        assert coord.shared_state.optimization_stack == []
        assert not coord.shared_state.current_best


class TestValidateForgeGemmTuningE2E:
    @pytest.mark.asyncio
    async def test_no_candidates_returns_early(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        fake = _make_integrate([{"decision": "KEEP", "new_tput": 200.0, "gain_pct": 100.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        result = {
            "recommended_env": {"X": "1"},
            "extra_envs": {"X": "1"},
            "requires_e2e_validation": True,
            "tuners_run": [
                "not-a-dict",
                {"status": "failed", "improved_shapes": 5, "env_var": "A", "env_value": "1"},
                {"status": "ok", "improved_shapes": 0, "env_var": "B", "env_value": "2"},
                {"status": "ok", "improved_shapes": 3, "env_var": "", "env_value": ""},
            ],
        }
        await coord._validate_gemm_tuning_e2e(result)

        assert fake.calls == []
        assert result["requires_e2e_validation"] is True
        assert "e2e_validated" not in result
        assert coord.shared_state.optimization_stack == []

    @pytest.mark.asyncio
    async def test_vllm_candidate_without_micro_count_validates_full_env_bundle(
        self,
        tmp_path,
        monkeypatch,
    ):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="vllm")
        fake = _make_integrate([{"decision": "KEEP", "new_tput": 112.0, "gain_pct": 12.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        result = {
            "workspace": str(tmp_path),
            "recommended_env": {
                "PYTHONPATH": "/candidate/site",
                "HL_TUNABLEOP_MODE": "candidate",
                "HL_TUNABLEOP_FILE": "/tunableop.csv",
                "PYTORCH_TUNABLEOP_FILENAME": "/tunableop.csv",
            },
            "extra_envs": {},
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok",
                    "candidate": True,
                    "improved_shapes": 0,
                    "tuner": "vllm_dense_tunableop",
                    "env_var": "PYTORCH_TUNABLEOP_FILENAME",
                    "env_value": "/tunableop.csv",
                    "env_vars": {
                        "PYTHONPATH": "/candidate/site",
                        "HL_TUNABLEOP_MODE": "candidate",
                        "HL_TUNABLEOP_FILE": "/tunableop.csv",
                        "PYTORCH_TUNABLEOP_FILENAME": "/tunableop.csv",
                    },
                }
            ],
        }

        await coord._validate_gemm_tuning_e2e(result)

        assert len(fake.calls) == 1
        assert fake.calls[0]["extra_envs"] == {
            "PYTHONPATH": "/candidate/site",
            "HL_TUNABLEOP_MODE": "candidate",
            "HL_TUNABLEOP_FILE": "/tunableop.csv",
            "PYTORCH_TUNABLEOP_FILENAME": "/tunableop.csv",
        }
        assert result["decision"] == "KEEP"
        assert result["e2e_gain_pct"] == pytest.approx(12.0)

    @pytest.mark.asyncio
    async def test_merges_every_aiter_config_in_candidate_env_bundle(
        self,
        tmp_path,
        monkeypatch,
    ):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        phase = KernelPhase(coord)
        dense = tmp_path / "dense.csv"
        moe = tmp_path / "moe.csv"
        dense.write_text("M,N,K\n1,2,3\n", encoding="utf-8")
        moe.write_text("token,model_dim\n1,2\n", encoding="utf-8")
        merged_dense = tmp_path / "merged-dense.csv"
        merged_moe = tmp_path / "merged-moe.csv"
        merge_calls: list[tuple[str, str]] = []

        def _merge(env_var, env_value):
            merge_calls.append((env_var, env_value))
            return str(merged_dense if env_var == "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE" else merged_moe)

        fake = _make_integrate([{"decision": "KEEP", "new_tput": 112.0, "gain_pct": 12.0}])
        monkeypatch.setattr(phase, "_merge_gemm_candidate_with_runtime", _merge)
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)
        result = {
            "workspace": str(tmp_path),
            "tuners_run": [
                {
                    "status": "ok",
                    "candidate": True,
                    "tuner": "combined_aiter",
                    "env_var": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
                    "env_value": str(dense),
                    "env_vars": {
                        "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(dense),
                        "AITER_CONFIG_FMOE": str(moe),
                    },
                }
            ],
        }

        await phase._validate_gemm_tuning_e2e(result)

        assert set(merge_calls) == {
            ("AITER_CONFIG_GEMM_A8W8_BLOCKSCALE", str(dense)),
            ("AITER_CONFIG_FMOE", str(moe)),
        }
        assert fake.calls[0]["extra_envs"] == {
            "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE": str(merged_dense),
            "AITER_CONFIG_FMOE": str(merged_moe),
        }

    @pytest.mark.asyncio
    async def test_keep_stacks_envs_and_rewrites_result(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        fmoe_candidate = tmp_path / "fmoe.json"
        fmoe_candidate.write_text("token,model_dim\n1,2\n", encoding="utf-8")
        fake = _make_integrate(
            [
                {"decision": "KEEP", "new_tput": 120.0, "gain_pct": 20.0},
                {"decision": "KEEP", "new_tput": 132.0, "gain_pct": 10.0},
            ]
        )
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)
        monkeypatch.setattr(
            KernelPhase,
            "_merge_gemm_candidate_with_runtime",
            lambda _self, _env_var, env_value: env_value,
        )

        result = {
            "workspace": str(tmp_path),
            "backend": "forge",
            "recommended_env": {
                "AITER_CONFIG_FMOE": str(fmoe_candidate),
                "AITER_DENSE": "/dense.json",
            },
            "extra_envs": {
                "AITER_CONFIG_FMOE": str(fmoe_candidate),
                "AITER_DENSE": "/dense.json",
            },
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok",
                    "improved_shapes": 5,
                    "tuner": "fmoe_ck",
                    "env_var": "AITER_CONFIG_FMOE",
                    "env_value": str(fmoe_candidate),
                    "best_micro_speedup": 1.2,
                },
                {
                    "status": "ok",
                    "improved_shapes": 3,
                    "tuner": "dense_gemm",
                    "env_var": "AITER_DENSE",
                    "env_value": "/dense.json",
                    "best_micro_speedup": 1.1,
                },
            ],
        }
        await coord._validate_gemm_tuning_e2e(result)

        # fmoe_ck on sglang carries the aiter MoE runner arg; dense does not.
        assert fake.calls[0]["extra_server_args"] == "--moe-runner-backend aiter"
        assert fake.calls[1]["extra_server_args"] == ""
        assert fake.calls[0]["extra_envs"] == {"AITER_CONFIG_FMOE": str(fmoe_candidate)}
        assert fake.calls[1]["extra_envs"] == {
            "AITER_CONFIG_FMOE": str(fmoe_candidate),
            "AITER_DENSE": "/dense.json",
        }
        # base_tput advances after the first KEEP.
        assert fake.calls[1]["base_tput"] == pytest.approx(120.0)

        assert len(coord.shared_state.optimization_stack) == 2
        # Each kept tuner lands as a gemm_tuning KEEP journal event.
        gj = [e for e in _journal_entries(tmp_path) if e.get("kind") == "gemm_tuning"]
        assert [e["throughput_after"] for e in gj] == pytest.approx([120.0, 132.0])
        assert {e["task_id"] for e in gj} == {
            "gemm_tune_e2e_fmoe_ck",
            "gemm_tune_e2e_dense_gemm",
        }
        cb = coord.shared_state.current_best
        assert cb["variant_name"] == "forge_dense_gemm"
        assert cb["tput"] == pytest.approx(132.0)
        assert cb["extra_server_args"] == "--moe-runner-backend aiter"
        # Both tuners' envs accumulate onto current_best, one lift each.
        assert cb["extra_envs"] == {
            "AITER_CONFIG_FMOE": str(fmoe_candidate),
            "AITER_DENSE": "/dense.json",
        }
        assert coord.shared_state.cumulative_gain_validated == pytest.approx(32.0)

        # Result rewritten to the E2E-validated outcome.
        assert result["e2e_validated"] is True
        assert result["requires_e2e_validation"] is False
        assert result["decision"] == "KEEP"
        assert result["status"] == "complete"
        assert result["e2e_gain_pct"] == pytest.approx(32.0)
        assert result["recommended_env"] == {
            "AITER_CONFIG_FMOE": str(fmoe_candidate),
            "AITER_DENSE": "/dense.json",
        }
        # Raw (pre-validation) envs are preserved.
        assert result["recommended_env_raw"] == {
            "AITER_CONFIG_FMOE": str(fmoe_candidate),
            "AITER_DENSE": "/dense.json",
        }

    @pytest.mark.asyncio
    async def test_injects_synthetic_ck_candidate_when_eligible_no_table_candidates(self, tmp_path, monkeypatch):
        # No table candidates, but CK switch is eligible: the synthetic CK
        # candidate is injected, E2E-validated, and stacked under gemm_tuning.
        coord = _eligible_coord(tmp_path, monkeypatch)
        fake = _make_integrate([{"decision": "KEEP", "new_tput": 209.0, "gain_pct": 109.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        result = {
            "workspace": str(tmp_path),
            "backend": "forge",
            "recommended_env": {},
            "extra_envs": {},
            "tuners_run": [
                {
                    "status": "ok",
                    "improved_shapes": 0,
                    "tuner": "a8w8_blockscale",
                    "env_var": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
                    "env_value": "/t.csv",
                },
            ],
        }
        await coord._validate_gemm_tuning_e2e(result)

        assert len(fake.calls) == 1
        assert fake.calls[0]["extra_envs"] == {"SGLANG_FP8_BLOCKSCALE_CK_MAX_M": "256"}
        assert fake.calls[0]["extra_server_args"] == ""

        stack = coord.shared_state.optimization_stack
        assert len(stack) == 1
        assert stack[0]["action"] == "gemm_tuning"
        assert stack[0]["variant_name"] == "forge_ck_blockscale_backend_switch"
        assert stack[0]["extra_envs"]["SGLANG_FP8_BLOCKSCALE_CK_MAX_M"] == "256"
        # Result rewritten to the E2E-validated KEEP outcome.
        assert result["e2e_validated"] is True
        assert result["decision"] == "KEEP"
        assert result["recommended_env"] == {"SGLANG_FP8_BLOCKSCALE_CK_MAX_M": "256"}

    @pytest.mark.asyncio
    async def test_no_synthetic_ck_candidate_when_not_eligible(self, tmp_path, monkeypatch):
        # Not eligible (vllm): no candidates → early return, integrate never called.
        coord = _eligible_coord(tmp_path, monkeypatch, framework="vllm")
        fake = _make_integrate([{"decision": "KEEP", "new_tput": 209.0, "gain_pct": 109.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        result = {
            "backend": "forge",
            "recommended_env": {},
            "extra_envs": {},
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok",
                    "improved_shapes": 0,
                    "tuner": "a8w8_blockscale",
                    "env_var": "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE",
                    "env_value": "/t.csv",
                },
            ],
        }
        await coord._validate_gemm_tuning_e2e(result)

        assert fake.calls == []
        assert coord.shared_state.optimization_stack == []

    @pytest.mark.asyncio
    async def test_keep_only_when_tput_improves(self, tmp_path, monkeypatch):
        # decision==KEEP but new_tput not above running_tput → treated as REVERT.
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        fake = _make_integrate([{"decision": "KEEP", "new_tput": 100.0, "gain_pct": 0.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        result = {
            "recommended_env": {"X": "1"},
            "extra_envs": {"X": "1"},
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok",
                    "improved_shapes": 2,
                    "tuner": "dense",
                    "env_var": "X",
                    "env_value": "1",
                    "best_micro_speedup": 1.05,
                },
            ],
        }
        await coord._validate_gemm_tuning_e2e(result)

        assert coord.shared_state.optimization_stack == []
        assert result["decision"] == "REVERT"
        assert result["micro_decision"] == "candidate_no_e2e_gain"

    @pytest.mark.asyncio
    async def test_all_revert_resets_and_marks_no_gain(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="vllm")
        fake = _make_integrate([{"decision": "REVERT", "new_tput": 90.0, "gain_pct": -10.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)

        result = {
            "recommended_env": {"X": "1"},
            "extra_envs": {"X": "1"},
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok",
                    "improved_shapes": 4,
                    "tuner": "dense",
                    "env_var": "X",
                    "env_value": "1",
                    "best_micro_speedup": 1.05,
                },
            ],
        }
        await coord._validate_gemm_tuning_e2e(result)

        assert coord.shared_state.optimization_stack == []
        assert result["decision"] == "REVERT"
        assert result["micro_decision"] == "candidate_no_e2e_gain"
        assert result["e2e_gain_pct"] == 0.0
        assert result["recommended_env"] == {}
        assert result["requires_e2e_validation"] is False

    @pytest.mark.asyncio
    async def test_integrate_exception_records_fault_not_revert(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")

        async def _boom(payload, *, session_dir):
            raise RuntimeError("integrate crashed")

        monkeypatch.setattr(krh_mod, "integrate_handler", _boom)

        result = {
            "recommended_env": {"X": "1"},
            "extra_envs": {"X": "1"},
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok",
                    "improved_shapes": 2,
                    "tuner": "dense",
                    "env_var": "X",
                    "env_value": "1",
                    "best_micro_speedup": 1.1,
                },
            ],
        }
        await coord._validate_gemm_tuning_e2e(result)

        assert result["status"] == "failed"
        assert result["micro_decision"] == "integrate_fault"
        assert result["e2e_gain_pct"] is None
        faults = result["e2e_results"]["faults"]
        assert len(faults) == 1
        assert faults[0]["reason"] == "integrate_fault:handler_exception"
        assert result["e2e_results"]["reverted"] == []

    @pytest.mark.asyncio
    async def test_timeout_fallback_when_explore_helper_raises(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")

        def _raise(**kwargs):
            raise ValueError("no runtime budget")

        monkeypatch.setattr(explore_mod, "_compute_explore_variant_timeout", _raise)

        captured: dict[str, object] = {}

        async def _fake(payload, *, session_dir):
            captured["budget"] = payload["budget_minutes"]
            return {"decision": "KEEP", "new_tput": 150.0, "gain_pct": 50.0}

        monkeypatch.setattr(krh_mod, "integrate_handler", _fake)

        result = {
            "recommended_env": {"X": "1"},
            "extra_envs": {"X": "1"},
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok",
                    "improved_shapes": 2,
                    "tuner": "dense",
                    "env_var": "X",
                    "env_value": "1",
                    "best_micro_speedup": 1.1,
                },
            ],
        }
        await coord._validate_gemm_tuning_e2e(result)

        # Fallback budget is 15 minutes.
        assert captured["budget"] == 15


class TestForgeGemmE2EApplyGate:
    """A measured gain is only creditable if the artifact was actually used.

    Both checks answer a question throughput cannot: the shape keys never
    resolved (coverage), or the table never reached the server (apply verdict).
    Each is a positive finding, so each blocks the KEEP -- while "cannot tell"
    deliberately does not, because hit lines require AITER_LOG_TUNED_CONFIG=1
    and a scan of 60 production logs found it set in none of them.
    """

    @staticmethod
    def _result():
        return {
            "recommended_env": {"X": "1"},
            "extra_envs": {"X": "1"},
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok",
                    "improved_shapes": 2,
                    "tuner": "dense",
                    "env_var": "X",
                    "env_value": "1",
                    "best_micro_speedup": 1.1,
                },
            ],
        }

    @staticmethod
    def _wire(monkeypatch, *, coverage, verdict):
        fake = _make_integrate([{"decision": "KEEP", "new_tput": 130.0, "gain_pct": 30.0}])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)
        monkeypatch.setattr(KernelPhase, "_gemm_tuned_config_coverage", lambda self, *a, **k: coverage)
        monkeypatch.setattr(KernelPhase, "_gemm_apply_verdict", lambda self, *a, **k: verdict)
        return fake

    @pytest.mark.asyncio
    async def test_unmerged_artifact_blocks_a_measured_keep(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        self._wire(
            monkeypatch,
            coverage=None,
            verdict={
                "verdict": "not_merged",
                "blocks_keep": True,
                "conclusive": True,
                "detail": "1 tuned table(s) absent from the server's merge list",
            },
        )
        result = self._result()

        await coord._validate_gemm_tuning_e2e(result)

        # +30% was measured, and is still refused: the server was running its
        # bundled default table, so the delta is drift, not tuning.
        assert coord.shared_state.optimization_stack == []
        assert coord.shared_state.cumulative_gain_validated == 0.0
        assert result["decision"] == "REVERT"
        reverted = result["e2e_results"]["reverted"]
        assert len(reverted) == 1
        assert "tuned_config_never_applied[not_merged]" in reverted[0]["reason"]
        assert reverted[0]["apply_verdict"]["verdict"] == "not_merged"

    @pytest.mark.asyncio
    async def test_unreachable_shape_keys_block_a_measured_keep(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        self._wire(
            monkeypatch,
            coverage={
                "artifact_applied": False,
                "not_applied_reason": "no_shape_key_matched",
                "requested": 42,
                "covered": 0,
            },
            verdict=None,
        )
        result = self._result()

        await coord._validate_gemm_tuning_e2e(result)

        assert coord.shared_state.optimization_stack == []
        assert result["decision"] == "REVERT"
        reason = result["e2e_results"]["reverted"][0]["reason"]
        assert "tuned_config_never_applied[no_shape_key_matched]" in reason

    @pytest.mark.asyncio
    async def test_both_blockers_are_named(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        self._wire(
            monkeypatch,
            coverage={
                "artifact_applied": False,
                "not_applied_reason": "artifact_table_not_consulted",
                "requested": 7,
                "covered": 0,
            },
            verdict={"verdict": "not_merged", "blocks_keep": True, "conclusive": True},
        )
        result = self._result()

        await coord._validate_gemm_tuning_e2e(result)

        reason = result["e2e_results"]["reverted"][0]["reason"]
        assert "artifact_table_not_consulted+not_merged" in reason

    @pytest.mark.asyncio
    async def test_inconclusive_verdict_does_not_block(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        self._wire(
            monkeypatch,
            coverage={"artifact_applied": True, "coverage_pct": 88.0, "covered": 7, "requested": 8},
            verdict={
                "verdict": "inconclusive_no_hit_logging",
                "blocks_keep": False,
                "conclusive": False,
                "detail": "misses logged but hit logging was off",
            },
        )
        result = self._result()

        await coord._validate_gemm_tuning_e2e(result)

        assert result["decision"] == "KEEP"
        assert len(coord.shared_state.optimization_stack) == 1
        assert coord.shared_state.current_best["tput"] == 130.0

    @pytest.mark.asyncio
    async def test_served_verdict_keeps(self, tmp_path, monkeypatch):
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        self._wire(
            monkeypatch,
            coverage={"artifact_applied": True, "coverage_pct": 100.0, "covered": 8, "requested": 8},
            verdict={
                "verdict": "served",
                "blocks_keep": False,
                "conclusive": True,
                "hits": 512,
            },
        )
        result = self._result()

        await coord._validate_gemm_tuning_e2e(result)

        assert result["decision"] == "KEEP"
        kept = result["e2e_results"]["kept"]
        assert kept[0]["apply_verdict"]["hits"] == 512
        assert coord.shared_state.cumulative_gain_validated == pytest.approx(30.0)

    @pytest.mark.asyncio
    async def test_missing_evidence_leaves_the_decision_alone(self, tmp_path, monkeypatch):
        """No server log at all must not turn into an accusation."""
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        self._wire(monkeypatch, coverage=None, verdict=None)
        result = self._result()

        await coord._validate_gemm_tuning_e2e(result)

        assert result["decision"] == "KEEP"
        assert len(coord.shared_state.optimization_stack) == 1


class TestForgeGemmPairedConfirmation:
    """The promoted gain must say whether it was confirmed against drift.

    ``base_tput`` and ``new_tput`` are measured at different times, so the two
    are a block comparison and any drift between them lands in the result.
    Interleaving separates the two, and when it is not run the number is still
    promoted -- but labelled for what it is.
    """

    @staticmethod
    def _run_e2e(coord, monkeypatch, tputs):
        fake = _make_integrate([{"decision": "KEEP", "new_tput": t, "gain_pct": (t - 100.0)} for t in tputs])
        monkeypatch.setattr(krh_mod, "integrate_handler", fake)
        monkeypatch.setattr(KernelPhase, "_gemm_tuned_config_coverage", lambda self, *a, **k: None)
        monkeypatch.setattr(KernelPhase, "_gemm_apply_verdict", lambda self, *a, **k: None)
        return fake

    @staticmethod
    def _result():
        return {
            "recommended_env": {"X": "1"},
            "extra_envs": {"X": "1"},
            "requires_e2e_validation": True,
            "tuners_run": [
                {
                    "status": "ok",
                    "improved_shapes": 2,
                    "tuner": "dense",
                    "env_var": "X",
                    "env_value": "1",
                    "best_micro_speedup": 1.1,
                },
            ],
        }

    @pytest.mark.asyncio
    async def test_unpaired_by_default_and_labelled_as_such(self, tmp_path, monkeypatch):
        monkeypatch.delenv("HYPERLOOM_GEMM_PAIRED_PAIRS", raising=False)
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        basis: dict[str, str] = {}
        monkeypatch.setattr(
            coord,
            "_update_cumulative_gain_validated",
            lambda tput, **kw: basis.update({"basis": kw.get("measurement_basis", ""), "tput": tput}),
        )
        fake = self._run_e2e(coord, monkeypatch, [130.0])

        await coord._validate_gemm_tuning_e2e(self._result())

        # One integrate call: the confirmation pass did not run.
        assert len(fake.calls) == 1
        assert basis["basis"] == "e2e_rebench_unpaired"
        assert basis["tput"] == 130.0

    @pytest.mark.asyncio
    async def test_paired_confirmation_runs_interleaved_and_labels_the_gain(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_GEMM_PAIRED_PAIRS", "2")
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        basis: dict[str, str] = {}
        monkeypatch.setattr(
            coord,
            "_update_cumulative_gain_validated",
            lambda tput, **kw: basis.update({"basis": kw.get("measurement_basis", "")}),
        )
        # 1 validation call, then A,B,A,B: baseline ~100, candidate ~130.
        fake = self._run_e2e(coord, monkeypatch, [130.0, 100.0, 130.0, 101.0, 131.0])
        result = self._result()

        await coord._validate_gemm_tuning_e2e(result)

        assert len(fake.calls) == 5
        # The confirmation pass alternates env-free and env-carrying runs.
        assert [bool(c["extra_envs"]) for c in fake.calls[1:]] == [False, True, False, True]
        paired = result["paired_confirmation"]
        assert paired["decisive"] is True
        assert paired["reason"] == "candidate_faster"
        assert len(paired["pairs"]) == 2
        assert basis["basis"] == "e2e_paired"

    @pytest.mark.asyncio
    async def test_drifting_pairs_are_not_labelled_confirmed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_GEMM_PAIRED_PAIRS", "2")
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        basis: dict[str, str] = {}
        monkeypatch.setattr(
            coord,
            "_update_cumulative_gain_validated",
            lambda tput, **kw: basis.update({"basis": kw.get("measurement_basis", "")}),
        )
        # The pairs disagree about which side is faster: the machine moved.
        self._run_e2e(coord, monkeypatch, [130.0, 100.0, 130.0, 140.0, 120.0])
        result = self._result()

        await coord._validate_gemm_tuning_e2e(result)

        assert result["paired_confirmation"]["reason"] == "sign_disagreement"
        assert result["paired_confirmation"]["decisive"] is False
        assert basis["basis"] == "e2e_paired_sign_disagreement"

    @pytest.mark.asyncio
    async def test_confirmation_failure_falls_back_to_insufficient_pairs(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_GEMM_PAIRED_PAIRS", "2")
        coord = _coord(tmp_path, baseline_tput=100.0, framework="sglang")
        calls: list[dict] = []

        async def _fake(payload, *, session_dir):
            calls.append(payload)
            if len(calls) == 1:
                return {"decision": "KEEP", "new_tput": 130.0, "gain_pct": 30.0}
            raise RuntimeError("benchmark host went away")

        monkeypatch.setattr(krh_mod, "integrate_handler", _fake)
        monkeypatch.setattr(KernelPhase, "_gemm_tuned_config_coverage", lambda self, *a, **k: None)
        monkeypatch.setattr(KernelPhase, "_gemm_apply_verdict", lambda self, *a, **k: None)
        result = self._result()

        await coord._validate_gemm_tuning_e2e(result)

        # A confirmation that could not run must not revert the artifact, and
        # must not claim to have confirmed anything either.
        assert result["decision"] == "KEEP"
        assert result["paired_confirmation"]["reason"] == "insufficient_pairs"
