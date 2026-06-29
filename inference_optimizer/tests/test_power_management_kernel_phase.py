"""Tests for the KERNEL-phase power-management (roofline-routed settle).

* executor: the roofline-routed settle grid, the public reset /
  re-apply / apply-max-climb-state wrappers, and the median
  kernel-only baseline.
* schema: ``PowerState`` round-trips through ``Recipe``.
* SharedState: ``combo_fingerprint`` stability + the collapsed
  ``power_attribution`` / ``power_settle_sweep_done`` /
  ``power_settle_hold_started_ts`` fields persist.
* Coordinator: run-start reset, MAX climb state at KERNEL entry,
  settle-sweep transition hold + timeout, failed-settle latch, settle
  attribution, and the recipe power_state stamp.

The Coordinator is exercised through ``Coordinator.__new__`` stubs (no
event loop / LLM) — same pattern as ``test_close_phase_sequencer``.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors import (
    _multi_node_env as mne,
)
from inference_optimizer.orchestrator.action_executors import (
    power_management as pm,
)
from inference_optimizer.orchestrator.action_executors.power_management import (
    KERNEL_PM_KEEP_THRESHOLD_PCT,
    PowerManagementExecutor,
    apply_max_climb_state,
    reapply_host_power_state,
    reset_host_power_defaults,
)
from inference_optimizer.orchestrator.coordinator import Coordinator
from inference_optimizer.orchestrator.shared_state import SharedState
from inference_optimizer.recipe_kb.schema import PowerState, Recipe


# ===========================================================================
# Executor: roofline-routed settle resolve grid
# ===========================================================================
class TestSettleResolveGrid:
    """``_resolve_variants`` on the default (no explicit grid) settle path
    builds the roofline-routed grid and returns ``(variants,
    grid_degraded)``."""

    def test_auto_baseline_and_high_always_present(self, tmp_path):
        ex = PowerManagementExecutor(session_dir=tmp_path)
        variants, deg = ex._resolve_variants(
            grid_override=None,
            probed_range=(150, 750),
            floor_w=150,
            ceiling_w=750,
            bound_kind="memory",
            top_sclk_mhz=2400,
            sclk_top_idx=4,
            mclk_levels={"count": 1, "indices": [0], "mhz": [2000]},
        )
        names = [v.name for v in variants]
        assert names[0] == "auto_baseline"
        assert "high" in names
        auto = variants[0]
        # auto_baseline = perflevel auto + ceiling cap + fan100.
        assert auto.perflevel == "auto"
        assert auto.power_cap_w == 750
        assert auto.fan_pct == 100
        assert deg == {}

    def test_determinism_ladder_built_from_top_sclk(self, tmp_path):
        ex = PowerManagementExecutor(session_dir=tmp_path)
        variants, _deg = ex._resolve_variants(
            grid_override=None,
            probed_range=(150, 750),
            floor_w=150,
            ceiling_w=750,
            bound_kind="memory",   # full det ladder (prune is compute-only)
            top_sclk_mhz=2400,
            sclk_top_idx=4,
            mclk_levels={"count": 1, "indices": [0], "mhz": [2000]},
        )
        det = {v.name for v in variants if v.note == "determinism"}
        assert det == {"det_100", "det_95", "det_90", "det_85"}

    def test_memory_axis_capability_gated(self, tmp_path):
        # NOT memory-bound + >= 2 mclk levels → a memory row is emitted.
        ex = PowerManagementExecutor(session_dir=tmp_path)
        variants, deg = ex._resolve_variants(
            grid_override=None,
            probed_range=(150, 750),
            floor_w=150,
            ceiling_w=750,
            bound_kind="compute",
            top_sclk_mhz=2400,
            sclk_top_idx=4,
            mclk_levels={"count": 2, "indices": [0, 1], "mhz": [1000, 2000]},
        )
        mem = [v for v in variants if v.note == "mclk"]
        assert len(mem) == 1
        assert mem[0].sclk_idx == 4   # GFX pinned high
        assert deg == {}

    def test_no_top_sclk_flags_grid_degraded(self, tmp_path):
        ex = PowerManagementExecutor(session_dir=tmp_path)
        variants, deg = ex._resolve_variants(
            grid_override=None,
            probed_range=(150, 750),
            floor_w=150,
            ceiling_w=750,
            bound_kind="memory",
            top_sclk_mhz=None,
            sclk_top_idx=None,
            mclk_levels={"count": 1, "indices": [0], "mhz": [2000]},
        )
        assert [v.name for v in variants] == ["auto_baseline", "high"]
        assert "gfx_determinism" in deg


# ===========================================================================
# Executor: reset / re-apply / max-climb-state wrappers
# ===========================================================================
class TestResetWrapper:
    def test_multi_node_skips(self):
        out = reset_host_power_defaults(is_multi_node=True)
        assert out == {"status": "skipped", "reason": "multi_node"}

    def test_dry_run_skips(self):
        out = reset_host_power_defaults(dry_run=True)
        assert out == {"status": "skipped", "reason": "dry_run"}

    def test_rocm_smi_unavailable_skips(self, monkeypatch):
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: False)
        out = reset_host_power_defaults()
        assert out == {"status": "skipped", "reason": "rocm_smi_unavailable"}

    def test_available_resets(self, monkeypatch):
        called = {"reset": 0}
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: True)
        monkeypatch.setattr(
            pm, "_reset_defaults",
            lambda: called.__setitem__("reset", called["reset"] + 1),
        )
        out = reset_host_power_defaults()
        assert out == {"status": "reset"}
        assert called["reset"] == 1


class TestReapplyWrapper:
    def test_no_snapshot_skips(self):
        assert reapply_host_power_state(None)["reason"] == "no_snapshot"

    def test_no_commands_skips(self):
        assert reapply_host_power_state({"smi_commands": []})["reason"] == (
            "no_commands"
        )

    def test_multi_node_skips(self):
        out = reapply_host_power_state(
            {"smi_commands": ["x"]}, is_multi_node=True,
        )
        assert out["reason"] == "multi_node"

    def test_applies_each_command(self, monkeypatch):
        ran: list[str] = []
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: True)
        monkeypatch.setattr(
            pm, "_run_smi", lambda c, check=False: ran.append(c),
        )
        out = reapply_host_power_state(
            {"smi_commands": ["cmd-a", "cmd-b"]},
        )
        assert out == {"status": "applied", "n": 2}
        assert ran == ["cmd-a", "cmd-b"]


class TestApplyMaxClimbState:
    def test_multi_node_returns_none(self):
        assert apply_max_climb_state(is_multi_node=True) is None

    def test_dry_run_returns_none(self):
        assert apply_max_climb_state(dry_run=True) is None

    def test_rocm_smi_unavailable_returns_none(self, monkeypatch):
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: False)
        assert apply_max_climb_state() is None

    def test_applies_incumbent_stack_and_snapshots(self, monkeypatch):
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: True)
        # Probe the ceiling so the incumbent pins the cap at the max.
        monkeypatch.setattr(
            pm, "_probe_powercap_range", lambda *_a, **_k: (200, 400),
        )
        applied: list[str] = []
        monkeypatch.setattr(
            pm, "_run_smi", lambda c, check=True: applied.append(c),
        )
        monkeypatch.setattr(
            pm, "_probe_current_state",
            lambda devs: {"perflevel": "auto", "powercap_w": 400},
        )
        snap = apply_max_climb_state()
        assert snap is not None
        assert snap["variant_name"] == "climb_max_state"
        # Incumbent = perflevel auto + ceiling cap + fan100. The cap + fan
        # stay maxed; GFX is left on the auto governor (the settle sweep
        # tunes GFX via --setperfdeterminism, and its auto_baseline row
        # reproduces exactly this state).
        assert snap["power_settings"]["perflevel"] == "auto"
        assert snap["power_settings"]["power_cap_w"] == 400
        assert snap["power_settings"]["fan_pct"] == 100
        assert any("--setperflevel auto" in c for c in snap["smi_commands"])
        assert any("--setpoweroverdrive 400" in c for c in snap["smi_commands"])
        assert any("--setfan 100%" in c for c in snap["smi_commands"])

    def test_apply_failure_resets_and_returns_none(self, monkeypatch):
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: True)
        monkeypatch.setattr(
            pm, "_probe_powercap_range", lambda *_a, **_k: (200, 400),
        )
        reset = {"n": 0}
        monkeypatch.setattr(
            pm, "_reset_defaults", lambda: reset.__setitem__("n", reset["n"] + 1),
        )

        def _boom(v):
            raise RuntimeError("setter failed")

        monkeypatch.setattr(pm, "_apply_variant", _boom)
        assert apply_max_climb_state() is None
        assert reset["n"] == 1


# ===========================================================================
# schema: PowerState round-trip
# ===========================================================================
class TestPowerStateSchema:
    def test_round_trip(self):
        ps = PowerState(
            variant_name="perflevel_high",
            power_settings={"perflevel": "high"},
            smi_commands=["rocm-smi --setperflevel high --autorespond yes"],
            device_ids=[0, 1],
            power_gain_pct=4.2,
            kernel_tput=1000.0,
            combined_tput=1042.0,
            ts="2026-06-01T00:00:00+00:00",
        )
        again = PowerState.from_dict(ps.to_dict())
        assert again == ps

    def test_recipe_carries_power_state(self):
        ps = PowerState(variant_name="perflevel_high", power_gain_pct=3.0,
                        smi_commands=["x"])
        r = Recipe(canonical_id="cid", power_state=ps)
        d = r.to_dict()
        assert d["power_state"]["variant_name"] == "perflevel_high"
        back = Recipe.from_dict(d)
        assert back.power_state.power_gain_pct == 3.0
        assert back.power_state.smi_commands == ["x"]

    def test_missing_power_state_defaults_empty(self):
        back = Recipe.from_dict({"canonical_id": "cid"})
        assert back.power_state.is_empty()


# ===========================================================================
# SharedState: combo fingerprint + collapsed fields persist
# ===========================================================================
class TestSharedStateComboFingerprint:
    def test_empty_stack_is_baseline_sentinel(self):
        ss = SharedState()
        assert ss.combo_fingerprint() == "baseline"

    def test_fingerprint_stable_and_ignores_power_rows(self):
        ss = SharedState()
        ss.optimization_stack = [
            {"action": "integrate", "kernel_id": "k001",
             "extra_server_args": "--foo"},
        ]
        fp1 = ss.combo_fingerprint()
        ss.optimization_stack.append(
            {"action": "power_management", "variant_name": "perflevel_high"}
        )
        assert ss.combo_fingerprint() == fp1
        ss.optimization_stack.insert(
            1, {"action": "integrate", "kernel_id": "k002"}
        )
        assert ss.combo_fingerprint() != fp1

    def test_collapsed_fields_persist(self, tmp_path):
        ss = SharedState()
        ss.power_attribution = {
            "kernel_tput": 1000.0, "combined_tput": 1042.0,
            "power_delta_pct": 4.2, "n_reps": 3, "low_confidence": False,
        }
        ss.power_settle_sweep_done = True
        ss.power_settle_hold_started_ts = "2026-06-01T00:00:00+00:00"
        ss.save(tmp_path)
        loaded = SharedState.load_or_init(tmp_path)
        assert loaded.power_attribution["power_delta_pct"] == 4.2
        assert loaded.power_settle_sweep_done is True
        assert loaded.power_settle_hold_started_ts == "2026-06-01T00:00:00+00:00"


# ===========================================================================
# Coordinator stubs
# ===========================================================================
class _StubTaskRow:
    def __init__(self, task_id, kind, params, idempotency_key):
        self.task_id = task_id
        self.kind = kind
        self.state = "queued"
        self.params = params
        self.idempotency_key = idempotency_key


class _StubTaskRegistry:
    def __init__(self):
        self._by_key: dict[str, _StubTaskRow] = {}
        self.created: list[_StubTaskRow] = []

    async def create_or_return_existing(self, *, kind, params, idempotency_key,
                                        **_kw):
        existing = self._by_key.get(idempotency_key)
        if existing is not None:
            return existing, True
        row = _StubTaskRow(uuid.uuid4().hex, kind, dict(params), idempotency_key)
        self._by_key[idempotency_key] = row
        self.created.append(row)
        return row, False


def _coord(tmp_path: Path, state: SharedState | None = None) -> Coordinator:
    c = Coordinator.__new__(Coordinator)
    c.shared_state = state or SharedState()
    c.session_dir = tmp_path
    c.tasks = _StubTaskRegistry()
    c._resumed_from = {"is_resume": False, "rebuilt": False}
    return c


@pytest.fixture(autouse=True)
def _single_node(monkeypatch):
    monkeypatch.setattr(mne, "is_multi_node", lambda: False)


class TestRunStartReset:
    def test_clears_host_state_and_marks_done(self, tmp_path, monkeypatch):
        seen = {"reset": 0}
        monkeypatch.setattr(
            pm, "reset_host_power_defaults",
            lambda **kw: seen.__setitem__("reset", seen["reset"] + 1)
            or {"status": "reset"},
        )
        ss = SharedState()
        ss.host_state_applied = {"variant_name": "stale"}
        c = _coord(tmp_path, ss)
        c._ensure_run_start_power_reset()
        assert ss.host_state_applied is None
        assert seen["reset"] == 1
        c._ensure_run_start_power_reset()
        assert seen["reset"] == 1

    def test_resume_non_kernel_resets_but_preserves_record(self, tmp_path, monkeypatch):
        seen = {"reset": 0}
        monkeypatch.setattr(
            pm, "reset_host_power_defaults",
            lambda **kw: seen.__setitem__("reset", seen["reset"] + 1)
            or {"status": "reset"},
        )
        ss = SharedState()
        ss.phase = "CLOSE"
        ss.host_state_applied = {"variant_name": "winner", "smi_commands": ["x"]}
        c = _coord(tmp_path, ss)
        c._resumed_from = {"is_resume": True, "rebuilt": False}
        c._ensure_run_start_power_reset()
        assert seen["reset"] == 1
        assert ss.host_state_applied == {"variant_name": "winner", "smi_commands": ["x"]}

    def test_resume_into_kernel_reapplies_recorded_state(self, tmp_path, monkeypatch):
        # Resume into KERNEL resets the live GPU then re-applies the
        # recorded host_state_applied (the MAX climb state).
        calls: list[str] = []
        monkeypatch.setattr(
            pm, "reset_host_power_defaults",
            lambda **kw: calls.append("reset") or {"status": "reset"},
        )
        monkeypatch.setattr(
            pm, "reapply_host_power_state",
            lambda snap, **kw: calls.append("apply") or {"status": "applied"},
        )
        ss = SharedState()
        ss.phase = "KERNEL"
        snap = {"variant_name": "climb_max_state",
                "smi_commands": ["rocm-smi --setperflevel high"]}
        ss.host_state_applied = snap
        c = _coord(tmp_path, ss)
        c._resumed_from = {"is_resume": True, "rebuilt": False}
        c._ensure_run_start_power_reset()
        assert calls == ["reset", "apply"]
        assert ss.host_state_applied == snap


class TestMaxClimbStateAtKernelEntry:
    def test_applies_and_records(self, tmp_path, monkeypatch):
        snap = {"variant_name": "climb_max_state",
                "smi_commands": ["rocm-smi --setperflevel high"],
                "power_settings": {"perflevel": "high", "power_cap_w": 400,
                                   "fan_pct": 100}}
        monkeypatch.setattr(pm, "apply_max_climb_state", lambda **kw: snap)
        ss = SharedState()
        c = _coord(tmp_path, ss)
        c._apply_kernel_climb_max_state()
        assert ss.host_state_applied == snap
        assert c._max_climb_state_applied is True
        # Idempotent per instance.
        calls = {"n": 0}
        monkeypatch.setattr(
            pm, "apply_max_climb_state",
            lambda **kw: calls.__setitem__("n", calls["n"] + 1) or snap,
        )
        c._apply_kernel_climb_max_state()
        assert calls["n"] == 0

    def test_multi_node_noop(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mne, "is_multi_node", lambda: True)
        monkeypatch.setattr(
            pm, "apply_max_climb_state",
            lambda **kw: pytest.fail("must not apply on multi-node"),
        )
        ss = SharedState()
        c = _coord(tmp_path, ss)
        c._apply_kernel_climb_max_state()
        assert ss.host_state_applied is None


class TestSettleSweepHold:
    def _kernel_state(self) -> SharedState:
        ss = SharedState()
        ss.baseline_tput = 1000.0
        ss.current_best = {"tput": 1100.0, "extra_server_args": "--foo"}
        ss.optimization_stack = [{"action": "integrate", "kernel_id": "k1"}]
        # The MAX incumbent is the live host state at the plateau.
        ss.host_state_applied = {"variant_name": "climb_max_state",
                                 "smi_commands": ["rocm-smi --setperflevel high"]}
        return ss

    def test_holds_transition_on_plateau(self, tmp_path):
        c = _coord(tmp_path, self._kernel_state())
        held = asyncio.run(c._maybe_hold_kernel_for_power_sweep(
            prior="KERNEL", target="SWEEP", reason="plateau_kernel",
        ))
        assert held is True
        assert len(c.tasks.created) == 1
        params = c.tasks.created[0].params
        # Roofline bottleneck steers Stage 1 (unknown → both rows here).
        assert params["bound_kind"] in ("memory", "compute", "unknown")
        assert params["reason"] == "kernel_settle"
        assert params["measure_kernel_baseline"] is True
        # The MAX incumbent is passed so the executor can keep it
        # on a no-winner round.
        assert params["host_state_applied"]["variant_name"] == "climb_max_state"
        # First defer stamps the hold timestamp.
        assert c.shared_state.power_settle_hold_started_ts

    def test_holds_transition_on_budget_exhaustion(self, tmp_path):
        # The settle PM sweep is mandatory on EVERY KERNEL -> SWEEP exit,
        # including the forced budget-exhausted exit (the no-leverage
        # kernel-phase case that never reaches a real plateau). It must
        # hold the transition and enqueue the settle task just like the
        # plateau exit; the deadline guard bounds the hold.
        c = _coord(tmp_path, self._kernel_state())
        held = asyncio.run(c._maybe_hold_kernel_for_power_sweep(
            prior="KERNEL", target="SWEEP",
            reason="kernel_phase_budget_exhausted",
        ))
        assert held is True
        assert len(c.tasks.created) == 1
        assert c.tasks.created[0].params["reason"] == "kernel_settle"
        assert c.shared_state.power_settle_hold_started_ts

    def test_holds_transition_on_no_more_leverage(self, tmp_path):
        # Same for the scheduling-police skip-to-sweep exit.
        c = _coord(tmp_path, self._kernel_state())
        held = asyncio.run(c._maybe_hold_kernel_for_power_sweep(
            prior="KERNEL", target="SWEEP", reason="no_more_leverage",
        ))
        assert held is True
        assert len(c.tasks.created) == 1
        assert c.tasks.created[0].params["reason"] == "kernel_settle"

    def test_does_not_hold_once_done(self, tmp_path):
        ss = self._kernel_state()
        ss.power_settle_sweep_done = True
        c = _coord(tmp_path, ss)
        held = asyncio.run(c._maybe_hold_kernel_for_power_sweep(
            prior="KERNEL", target="SWEEP", reason="plateau_kernel",
        ))
        assert held is False

    def test_multi_node_sets_done_and_proceeds(self, tmp_path, monkeypatch):
        monkeypatch.setattr(mne, "is_multi_node", lambda: True)
        ss = self._kernel_state()
        c = _coord(tmp_path, ss)
        held = asyncio.run(c._maybe_hold_kernel_for_power_sweep(
            prior="KERNEL", target="SWEEP", reason="plateau_kernel",
        ))
        assert held is False
        assert ss.power_settle_sweep_done is True

    def test_timeout_latches_done_and_proceeds(self, tmp_path):
        # A settle hold that has run past the deadline latches done and
        # proceeds (so a lost / crashed settle task can't wedge KERNEL).
        ss = self._kernel_state()
        old = datetime.now(timezone.utc) - timedelta(seconds=10 ** 9)
        ss.power_settle_hold_started_ts = old.isoformat()
        c = _coord(tmp_path, ss)
        held = asyncio.run(c._maybe_hold_kernel_for_power_sweep(
            prior="KERNEL", target="SWEEP", reason="plateau_kernel",
        ))
        assert held is False
        assert ss.power_settle_sweep_done is True
        # No new task enqueued — we're past the deadline.
        assert c.tasks.created == []


class TestFailedSettleLatch:
    def test_failed_settle_task_releases_hold(self, tmp_path):
        ss = SharedState()
        ss.baseline_tput = 1000.0
        c = _coord(tmp_path, ss)
        task = _StubTaskRow(
            "t1", "power_management",
            {"reason": "kernel_settle"}, "internal-pm-kernel-settle",
        )
        asyncio.run(c._handle_unpromotable_result(
            task, {"status": "failed", "error_class": "rocm_smi_set_failed"},
        ))
        assert ss.power_settle_sweep_done is True


class TestSettleAttribution:
    def test_applied_best_lifts_and_records_flat_attribution(self, tmp_path):
        ss = SharedState()
        ss.baseline_tput = 1000.0
        ss.current_best = {"tput": 1100.0, "extra_server_args": "--foo"}
        c = _coord(tmp_path, ss)
        result = {
            "base_tput": 1100.0,
            "kernel_baseline_tput": 1000.0,   # median vendor-default
            "kernel_baseline_reps": 3,
            "best_variant": {"variant_name": "cap_ceiling_high",
                             "output_throughput": 1155.0},
        }
        asyncio.run(c._record_power_management_attribution(
            result, combined=1155.0, lift_headline=True,
        ))
        assert ss.current_best["tput"] == 1155.0
        assert ss.current_best["extra_server_args"] == "--foo"
        assert ss.cumulative_gain_validated == pytest.approx(15.5)
        attr = ss.power_attribution
        assert attr["kernel_tput"] == 1000.0
        assert attr["combined_tput"] == 1155.0
        # FULL power contribution vs vendor-default kernel-only.
        assert attr["power_delta_pct"] == pytest.approx(15.5)
        assert attr["n_reps"] == 3
        assert attr["low_confidence"] is False

    def test_kept_incumbent_records_without_lift(self, tmp_path):
        # No fresh winner: the MAX incumbent (base_tput) is kept;
        # attribution is recorded but the headline is not re-lifted.
        ss = SharedState()
        ss.baseline_tput = 1000.0
        ss.current_best = {"tput": 1100.0, "extra_server_args": "--foo"}
        c = _coord(tmp_path, ss)
        result = {
            "base_tput": 1100.0,
            "kernel_baseline_tput": 1000.0,
            "kernel_baseline_reps": 3,
            "best_variant": None,
        }
        asyncio.run(c._record_power_management_attribution(
            result, combined=1100.0, lift_headline=False,
        ))
        # Headline unchanged.
        assert ss.current_best["tput"] == 1100.0
        attr = ss.power_attribution
        assert attr["combined_tput"] == 1100.0
        assert attr["power_delta_pct"] == pytest.approx(10.0)

    def test_clamps_negative_delta_and_flags_low_confidence(self, tmp_path):
        # Noise inverted the baseline (median defaults >= combined). The
        # recorded delta clamps to 0 and the entry is flagged.
        ss = SharedState()
        ss.baseline_tput = 1000.0
        ss.current_best = {"tput": 1100.0}
        c = _coord(tmp_path, ss)
        result = {
            "base_tput": 1100.0,
            "kernel_baseline_tput": 1200.0,   # > combined (noise)
            "kernel_baseline_reps": 3,
            "best_variant": {"variant_name": "x", "output_throughput": 1100.0},
        }
        asyncio.run(c._record_power_management_attribution(
            result, combined=1100.0, lift_headline=True,
        ))
        attr = ss.power_attribution
        assert attr["power_delta_pct"] == 0.0
        assert attr["low_confidence"] is True
        # Worse-than-baseline combined must not lower the headline.
        assert ss.current_best["tput"] == 1100.0


class TestRecipePowerStateBuild:
    def test_build_power_state_from_collapsed_state(self, tmp_path):
        ss = SharedState()
        ss.host_state_applied = {
            "variant_name": "cap_ceiling_high",
            "power_settings": {"power_cap_w": 750, "perflevel": "high"},
            "smi_commands": ["rocm-smi --setpoweroverdrive 750"],
            "device_ids": [0],
        }
        ss.power_attribution = {
            "kernel_tput": 1000.0, "combined_tput": 1155.0,
            "power_delta_pct": 15.5, "n_reps": 3, "low_confidence": False,
            "ts": "2026-06-01T00:00:00+00:00",
        }
        c = _coord(tmp_path, ss)
        out = c._build_power_state_for_recipe()
        assert out["variant_name"] == "cap_ceiling_high"
        assert out["power_gain_pct"] == 15.5
        assert out["combined_tput"] == 1155.0
        assert out["smi_commands"] == ["rocm-smi --setpoweroverdrive 750"]
        assert out["n_reps"] == 3
        assert out["low_confidence"] is False

    def test_empty_when_no_power_state(self, tmp_path):
        c = _coord(tmp_path, SharedState())
        assert c._build_power_state_for_recipe() == {}


# ===========================================================================
# Executor: settle-sweep kernel-only baseline (median over reps)
# ===========================================================================
class _FakeVariantResult:
    def __init__(self, tput):
        self.output_throughput = tput


class TestKernelOnlyBaseline:
    def _exec(self, tmp_path):
        return PowerManagementExecutor(session_dir=tmp_path)

    def _call(self, ex, tmp_path, reps=3):
        return asyncio.run(ex._measure_kernel_only_baseline(
            base_yaml_path=tmp_path / "cfg.yaml",
            base_extra_args="",
            output_root=tmp_path,
            variant_timeout_sec=60,
            resolved_model="m",
            resolved_gpu="mi300x",
            override_script=None,
            override_result_dir=None,
            reps=reps,
        ))

    def test_takes_median_over_reps(self, tmp_path, monkeypatch):
        resets = {"n": 0}
        monkeypatch.setattr(
            pm, "_reset_defaults",
            lambda: resets.__setitem__("n", resets["n"] + 1),
        )
        seq = iter([1100.0, 1200.0, 1150.0])
        captured: dict[str, Any] = {}

        async def _fake_run_grid(**kw):
            captured.update(kw)
            return [_FakeVariantResult(next(seq))]

        monkeypatch.setattr(pm, "run_grid", _fake_run_grid)
        tput, n = self._call(self._exec(tmp_path), tmp_path, reps=3)
        assert tput == 1150.0   # median of [1100, 1200, 1150]
        assert n == 3
        # Reset brackets the whole measurement (once before, once after).
        assert resets["n"] == 2
        grid = captured["grid"]
        assert grid[0].extra_server_args == ""
        assert grid[0].extra_envs == {}

    def test_none_on_empty_results(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pm, "_reset_defaults", lambda: None)

        async def _empty(**kw):
            return []

        monkeypatch.setattr(pm, "run_grid", _empty)
        assert self._call(self._exec(tmp_path), tmp_path) == (None, 0)

    def test_skips_zero_throughput_reps(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pm, "_reset_defaults", lambda: None)
        seq = iter([0.0, 1200.0, 0.0])

        async def _mixed(**kw):
            return [_FakeVariantResult(next(seq))]

        monkeypatch.setattr(pm, "run_grid", _mixed)
        tput, n = self._call(self._exec(tmp_path), tmp_path, reps=3)
        assert tput == 1200.0
        assert n == 1

    def test_survives_rep_errors(self, tmp_path, monkeypatch):
        resets = {"n": 0}
        monkeypatch.setattr(
            pm, "_reset_defaults",
            lambda: resets.__setitem__("n", resets["n"] + 1),
        )

        async def _boom(**kw):
            raise RuntimeError("magpie crashed")

        monkeypatch.setattr(pm, "run_grid", _boom)
        assert self._call(self._exec(tmp_path), tmp_path) == (None, 0)
        assert resets["n"] == 2
