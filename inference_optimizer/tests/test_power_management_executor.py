"""Unit tests for :mod:`orchestrator.action_executors.power_management`.

Covers (in roughly this order):

* :func:`_build_variant_from_payload`            — schema validation
* :func:`_enforce_cap_bounds`                    — floor/ceiling rejection
* :func:`_apply_variant_cmds`                    — shell command generation
* :func:`_probe_powercap_range`                  — JSON parse + reduce
* :class:`PowerManagementExecutor` dry-run path  — no rocm-smi, no rebench
* :class:`PowerManagementExecutor` winner select — best-throughput resolution
* :class:`PowerManagementExecutor` reset on fail — `finally` reset semantics

Each subprocess shell-out is patched via ``monkeypatch.setattr`` so the
tests never touch a real GPU.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from inference_optimizer.orchestrator.action_executors import power_management as pm
from inference_optimizer.orchestrator.action_executors.power_management import (
    POWER_CAP_DEFAULT_FLOOR_W,
    PowerManagementExecutor,
    PowerVariant,
    _apply_variant_cmds,
    _build_settle_grid,
    _build_variant_from_payload,
    _enforce_cap_bounds,
    _host_state_is_stale,
    _initial_power_management_search_state,
    _is_contradictory_combo,
    _gfx_high_sclk_idx,
    _parse_mclk_levels_from_clkfrq,
    _parse_mclkrange,
    _parse_sclk_levels_from_clkfrq,
    _parse_sclkrange_top_mhz,
    _probe_mclk_levels,
    _probe_powercap_range,
    _variant_from_accepted,
    power_variant_fingerprint,
)
from inference_optimizer.orchestrator.task_registry import Task


# ---------------------------------------------------------------------------
# Lightweight RunnerContext stand-in (matches the real dataclass surface)
# ---------------------------------------------------------------------------
@dataclass
class _Ctx:
    task: Task
    lease: Any = None
    extra: dict | None = None


def _task(params: dict[str, Any] | None = None, task_id: str = "pm-1") -> Task:
    return Task(
        task_id=task_id,
        kind="power_management",
        state="running",
        params=params or {},
        idempotency_key=task_id,
        requires_lanes=["server_lifecycle", "benchmark_lane"],
    )


# ---------------------------------------------------------------------------
# Variant payload validation
# ---------------------------------------------------------------------------
class TestBuildVariantFromPayload:
    def test_minimal_dict(self):
        v = _build_variant_from_payload({"name": "ok"}, 0)
        assert v.name == "ok"
        assert v.power_cap_w is None
        assert v.perflevel is None
        assert v.devices == ()

    def test_full_payload(self):
        v = _build_variant_from_payload({
            "name": "mix",
            "power_cap_w": 280,
            "perflevel": "manual",
            "sclk_idx": 3,
            "mclk_idx": 2,
            "pcie_idx": 1,
            "perf_deterministic_mhz": 1900,
            "fan_pct": 75,
            "devices": [0, 2],
            "note": "anchor",
        }, 4)
        assert v.power_cap_w == 280
        assert v.perflevel == "manual"
        assert v.sclk_idx == 3
        assert v.devices == (0, 2)
        assert v.fan_pct == 75
        assert v.note == "anchor"

    def test_non_dict_rejected(self):
        with pytest.raises(ValueError, match="must be a dict"):
            _build_variant_from_payload([1, 2], 0)

    def test_blank_name_rejected(self):
        with pytest.raises(ValueError, match="name must be non-empty"):
            _build_variant_from_payload({"name": "   "}, 0)

    def test_int_coercion_failure_rejected(self):
        with pytest.raises(ValueError, match="power_cap_w must be int"):
            _build_variant_from_payload(
                {"name": "x", "power_cap_w": "two hundred"}, 0,
            )

    def test_perflevel_vocab_enforced(self):
        with pytest.raises(ValueError, match="perflevel="):
            _build_variant_from_payload(
                {"name": "x", "perflevel": "warp_speed"}, 0,
            )

    def test_perflevel_case_insensitive(self):
        v = _build_variant_from_payload(
            {"name": "x", "perflevel": "AUTO"}, 0,
        )
        assert v.perflevel == "auto"

    @pytest.mark.parametrize("regressive_level", [
        "low",
        "profile_min_sclk",
        "profile_min_mclk",
    ])
    def test_perf_regressing_perflevels_rejected(self, regressive_level):
        # The executor is a perf optimizer: clock-pinning-DOWN modes
        # have no path to non-negative gain and would just burn a
        # ~40 min Magpie cycle per variant to confirm slower is slower.
        # Rejection happens before any state mutates / any ledger row
        # is written so the fingerprint cache stays clean.
        with pytest.raises(ValueError, match="perflevel="):
            _build_variant_from_payload(
                {"name": "regress", "perflevel": regressive_level}, 0,
            )

    @pytest.mark.parametrize("safe_level", [
        "auto", "high", "manual",
        "profile_standard", "profile_peak", "profile_compute",
    ])
    def test_perf_neutral_or_positive_perflevels_accepted(self, safe_level):
        # The complement of test_perf_regressing_perflevels_rejected:
        # every perflevel left in the vocabulary should validate cleanly
        # (no pins → no auto-inject needed; manual passes trivially).
        v = _build_variant_from_payload(
            {"name": "ok", "perflevel": safe_level}, 0,
        )
        assert v.perflevel == safe_level


class TestManualPerflevelGuard:
    """Pins (`sclk_idx`/`mclk_idx`/`pcie_idx`) require `perflevel='manual'`
    or rocm-smi silently no-ops the setter. The validator either
    auto-injects manual (LLM-friendly) or hard-rejects a contradicting
    perflevel (catches typos / hallucinations BEFORE we burn a bench)."""

    @pytest.mark.parametrize("pin_field, pin_value", [
        ("sclk_idx", 5),
        ("mclk_idx", 2),
        ("pcie_idx", 1),
    ])
    def test_pin_without_perflevel_auto_injects_manual(
        self, pin_field, pin_value,
    ):
        v = _build_variant_from_payload(
            {"name": "pin_only", pin_field: pin_value}, 0,
        )
        assert v.perflevel == "manual", (
            f"{pin_field} pin must auto-inject perflevel=manual or "
            f"rocm-smi --set{pin_field.replace('_idx', '')} is a no-op"
        )
        assert getattr(v, pin_field) == pin_value

    def test_multiple_pins_without_perflevel_auto_inject_once(self):
        v = _build_variant_from_payload(
            {"name": "all_pins", "sclk_idx": 7, "mclk_idx": 3, "pcie_idx": 0},
            0,
        )
        assert v.perflevel == "manual"
        assert (v.sclk_idx, v.mclk_idx, v.pcie_idx) == (7, 3, 0)

    @pytest.mark.parametrize("conflicting_level", [
        "auto", "high",
        "profile_standard", "profile_peak", "profile_compute",
    ])
    def test_pin_with_non_manual_perflevel_rejected(self, conflicting_level):
        # The historical silent-failure mode: LLM (or stale-ledger
        # replay) submits `{perflevel: 'high', sclk_idx: 7}`. rocm-smi
        # would emit a warning and ignore the pin; we'd bench an
        # unmodified GPU and persist a fingerprint claiming the pin
        # was applied. Reject before any work happens.
        with pytest.raises(
            ValueError,
            match=r"cannot be combined with .* pins",
        ):
            _build_variant_from_payload({
                "name": "conflict",
                "perflevel": conflicting_level,
                "sclk_idx": 5,
            }, 0)

    def test_pin_with_explicit_manual_passes(self):
        # The canonical "I know what I'm doing" path: explicit manual
        # + pin. Must validate without auto-inject side effects.
        v = _build_variant_from_payload({
            "name": "explicit_manual",
            "perflevel": "manual",
            "sclk_idx": 7,
        }, 0)
        assert v.perflevel == "manual"
        assert v.sclk_idx == 7

    def test_perfdeterminism_without_perflevel_does_not_auto_inject(self):
        # `--setperfdeterminism` operates independently of perflevel
        # (it's a separate rocm-smi subsystem), so a determinism-only
        # variant must NOT trigger the manual auto-inject. This keeps
        # the Stage-2 determinism ladder rows valid when rebuilt through
        # the validator.
        v = _build_variant_from_payload(
            {"name": "det_only", "perf_deterministic_mhz": 1900}, 0,
        )
        assert v.perflevel is None
        assert v.perf_deterministic_mhz == 1900

    def test_fan_only_does_not_auto_inject(self):
        # Fan control is also independent of perflevel.
        v = _build_variant_from_payload(
            {"name": "fan_only", "fan_pct": 80}, 0,
        )
        assert v.perflevel is None
        assert v.fan_pct == 80

    def test_fan_pct_range(self):
        with pytest.raises(ValueError, match="fan_pct=200"):
            _build_variant_from_payload(
                {"name": "x", "fan_pct": 200}, 0,
            )

    def test_devices_bad_type(self):
        with pytest.raises(ValueError, match="devices must be list"):
            _build_variant_from_payload(
                {"name": "x", "devices": "0,1"}, 0,
            )


# ---------------------------------------------------------------------------
# Cap bounds
# ---------------------------------------------------------------------------
class TestEnforceCapBounds:
    def test_no_cap_passes(self):
        v = PowerVariant(name="x")
        assert _enforce_cap_bounds(v, floor_w=150, ceiling_w=400) is None

    def test_in_range_passes(self):
        v = PowerVariant(name="x", power_cap_w=250)
        assert _enforce_cap_bounds(v, floor_w=150, ceiling_w=400) is None

    def test_below_floor_rejected(self):
        v = PowerVariant(name="x", power_cap_w=100)
        msg = _enforce_cap_bounds(v, floor_w=150, ceiling_w=400)
        assert msg is not None and "below floor" in msg

    def test_above_ceiling_rejected(self):
        v = PowerVariant(name="x", power_cap_w=999)
        msg = _enforce_cap_bounds(v, floor_w=150, ceiling_w=400)
        assert msg is not None and "above ceiling" in msg

    def test_zero_ceiling_disables_check(self):
        v = PowerVariant(name="x", power_cap_w=9999)
        assert _enforce_cap_bounds(v, floor_w=150, ceiling_w=0) is None


# ---------------------------------------------------------------------------
# Roofline-routed settle grid synthesis (the single power tune of a run)
# ---------------------------------------------------------------------------
class TestBuildSettleGrid:
    """`_build_settle_grid` is the single roofline-routed grid the settle
    sweep builds. Throughput-only with the power cap + fan pinned to MAX
    on every row. GFX is tuned via ``--setperfdeterminism`` ONLY (no GFX
    DPM-index pin). Rows: ``auto_baseline`` (incumbent reference) + ``high``
    always; the determinism ladder (det_100 always, det_95/det_90/det_85
    unless compute-bound); the capability-gated memory axis (NOT memory-bound AND
    >= 2 mclk levels). Returns ``(rows, grid_degraded)`` where
    ``grid_degraded`` flags an expected-but-empty ladder."""

    def test_always_emits_auto_baseline_and_high(self):
        rows, _deg = _build_settle_grid(
            cap_w=400, bound_kind=None, top_sclk_mhz=2000,
        )
        assert rows[0].name == "auto_baseline"
        assert rows[0].perflevel == "auto"
        assert rows[0].note == "auto_baseline"
        assert rows[1].name == "high"
        assert rows[1].perflevel == "high"
        for v in (rows[0], rows[1]):
            assert v.power_cap_w == 400
            assert v.fan_pct == 100

    def test_determinism_ladder_full_when_not_compute(self):
        # bound_kind=memory still emits the full det ladder (the prune is
        # compute-only); memory axis is separately skipped here (no mclk).
        rows, deg = _build_settle_grid(
            cap_w=400, bound_kind="memory", top_sclk_mhz=2000,
        )
        det = {v.name: v for v in rows if v.note == "determinism"}
        assert set(det) == {"det_100", "det_95", "det_90", "det_85"}
        assert det["det_100"].perf_deterministic_mhz == 2000
        assert det["det_95"].perf_deterministic_mhz == 1900
        assert det["det_90"].perf_deterministic_mhz == 1800
        assert det["det_85"].perf_deterministic_mhz == 1700
        # Determinism rows carry NO perflevel / sclk pin (just MHz).
        for v in det.values():
            assert v.perflevel is None
            assert v.sclk_idx is None
            assert v.power_cap_w == 400
            assert v.fan_pct == 100
        assert deg == {}

    def test_compute_bound_prunes_det_95_and_90(self):
        rows, deg = _build_settle_grid(
            cap_w=400, bound_kind="compute", top_sclk_mhz=2000,
        )
        det = {v.name for v in rows if v.note == "determinism"}
        assert det == {"det_100"}
        assert deg == {}

    def test_no_top_sclk_flags_grid_degraded(self):
        # det_100 is expected on every box; no probed MHz → 0 det rows
        # AND a structured grid_degraded entry (no silent collapse).
        rows, deg = _build_settle_grid(
            cap_w=400, bound_kind="unknown", top_sclk_mhz=None,
        )
        assert not [v for v in rows if v.note == "determinism"]
        assert "gfx_determinism" in deg
        # auto_baseline + high still present.
        assert {v.name for v in rows} == {"auto_baseline", "high"}

    def test_memory_axis_emitted_when_capable(self):
        # NOT memory-bound + >= 2 mclk levels + a probed sclk top index:
        # GFX pinned high (manual + sclk top) while memory is stepped.
        rows, deg = _build_settle_grid(
            cap_w=400, bound_kind="compute", top_sclk_mhz=2000,
            sclk_top_idx=3,
            mclk_levels={"count": 2, "indices": [0, 1], "mhz": [1000, 2000]},
        )
        mem = [v for v in rows if v.note == "mclk"]
        assert len(mem) == 1  # only the non-top level is stepped to
        m = mem[0]
        assert m.name == "mclk_1000mhz"
        assert m.perflevel == "manual"
        assert m.sclk_idx == 3      # GFX held high
        assert m.mclk_idx == 0      # stepped down
        assert m.power_cap_w == 400
        assert m.fan_pct == 100
        assert deg == {}

    def test_memory_axis_skipped_when_memory_bound(self):
        # Memory-bound → stepping memory down can't help → omit (not a
        # degradation, the gate firing as intended → no grid_degraded).
        rows, deg = _build_settle_grid(
            cap_w=400, bound_kind="memory", top_sclk_mhz=2000,
            sclk_top_idx=3,
            mclk_levels={"count": 2, "indices": [0, 1], "mhz": [1000, 2000]},
        )
        assert not [v for v in rows if v.note == "mclk"]
        assert "memory" not in deg

    def test_memory_axis_skipped_single_level(self):
        # < 2 selectable mclk levels (this MI355X has one) → skipped with
        # reason; NOT a degradation.
        rows, deg = _build_settle_grid(
            cap_w=400, bound_kind="compute", top_sclk_mhz=2000,
            sclk_top_idx=3,
            mclk_levels={"count": 1, "indices": [0], "mhz": [2000]},
        )
        assert not [v for v in rows if v.note == "mclk"]
        assert "memory" not in deg

    def test_memory_axis_capable_but_no_sclk_top_idx_flags_degraded(self):
        # >= 2 levels but no sclk top index → we can't pin GFX high while
        # stepping memory (the row would conflate axes) → flag degraded.
        rows, deg = _build_settle_grid(
            cap_w=400, bound_kind="compute", top_sclk_mhz=2000,
            sclk_top_idx=None,
            mclk_levels={"count": 2, "indices": [0, 1], "mhz": [1000, 2000]},
        )
        assert not [v for v in rows if v.note == "mclk"]
        assert "memory" in deg

    def test_all_rows_carry_cap_and_fan_max(self):
        rows, _deg = _build_settle_grid(
            cap_w=380, fan_pct=100, bound_kind="compute", top_sclk_mhz=2000,
            sclk_top_idx=3,
            mclk_levels={"count": 2, "indices": [0, 1], "mhz": [1000, 2000]},
        )
        for v in rows:
            assert v.power_cap_w == 380
            assert v.fan_pct == 100

    def test_dedups_equal_rounded_det_freqs(self):
        # A tiny top sclk where 95/100% collapse to the same rounded int.
        rows, _deg = _build_settle_grid(
            cap_w=400, bound_kind="unknown", top_sclk_mhz=10,
        )
        det_freqs = [
            v.perf_deterministic_mhz for v in rows if v.note == "determinism"
        ]
        assert len(det_freqs) == len(set(det_freqs))


# ---------------------------------------------------------------------------
# GFX-high sclk index (feeds the GFX-high pin on memory rows)
# ---------------------------------------------------------------------------
class TestGfxHighSclkIdx:
    """`_gfx_high_sclk_idx` selects the sclk DPM index with the HIGHEST
    FREQUENCY (not the largest index) from the ``sclk[N]`` keys of
    ``--showclkfrq --json``. The memory rows pin GFX high via that index
    while stepping memory; because the DPM table is coarse and scrambled,
    picking by frequency (not by index) is what keeps GFX actually high."""

    def test_picks_highest_frequency_index(self):
        data = {"card0": {
            "sclk[0]": "500Mhz", "sclk[1]": "1200Mhz", "sclk[2]": "1900Mhz",
            "mclk[0]": "900Mhz", "mclk[1]": "1300Mhz",
        }}
        # Monotonic table: highest freq == highest index.
        assert _parse_sclk_levels_from_clkfrq(data) == [
            (0, 500), (1, 1200), (2, 1900),
        ]
        assert _gfx_high_sclk_idx(data) == 2

    def test_scrambled_table_picks_by_frequency_not_index(self):
        # Scrambled DPM ordering: the LARGEST index (2) is NOT the
        # highest frequency. The GFX-high pin must be index 1 (2400MHz),
        # not index 2 (158MHz) — that's the whole point of selecting by
        # frequency.
        data = {"card0": {
            "sclk[0]": "500Mhz", "sclk[1]": "2400Mhz", "sclk[2]": "158Mhz",
        }}
        assert _gfx_high_sclk_idx(data) == 1

    def test_takes_max_freq_per_index_across_cards(self):
        data = {
            "card0": {"sclk[0]": "500Mhz", "sclk[1]": "1900Mhz"},
            "card1": {"sclk[0]": "500Mhz", "sclk[1]": "1950Mhz"},
        }
        assert _gfx_high_sclk_idx(data) == 1

    def test_ties_break_to_highest_index(self):
        data = {"card0": {"sclk[0]": "1900Mhz", "sclk[1]": "1900Mhz"}}
        assert _gfx_high_sclk_idx(data) == 1

    def test_none_when_no_sclk_keys(self):
        assert _gfx_high_sclk_idx({"card0": {"mclk[0]": "900Mhz"}}) is None
        assert _gfx_high_sclk_idx({"card0": {"Max GPU Clock": "2100Mhz"}}) is None
        assert _gfx_high_sclk_idx(None) is None
        assert _parse_sclk_levels_from_clkfrq(None) == []


# ---------------------------------------------------------------------------
# Combo-contradiction guard (`_is_contradictory_combo`)
# ---------------------------------------------------------------------------
class TestContradictionFilter:
    """`_is_contradictory_combo` encodes rocm-smi's own semantic
    constraints — the cases where stacking two variants would either
    error at apply time or silently overwrite intent. Returning None
    means the combo is safe to materialise."""

    def test_compatible_pair_passes(self):
        # Different axes, no overlap — the canonical "good combo" case.
        cap = PowerVariant(name="cap", power_cap_w=300)
        perf = PowerVariant(name="perf", perflevel="high")
        assert _is_contradictory_combo((cap, perf)) is None

    def test_conflicting_caps_blocked(self):
        a = PowerVariant(name="a", power_cap_w=300)
        b = PowerVariant(name="b", power_cap_w=350)
        reason = _is_contradictory_combo((a, b))
        assert reason is not None
        assert "power_cap_w" in reason

    def test_same_cap_value_is_allowed(self):
        # Redundant set is fine — last setter wins with identical value.
        a = PowerVariant(name="a", power_cap_w=300)
        b = PowerVariant(name="b", power_cap_w=300, perflevel="high")
        assert _is_contradictory_combo((a, b)) is None

    def test_conflicting_perflevels_blocked(self):
        a = PowerVariant(name="a", perflevel="high")
        b = PowerVariant(name="b", perflevel="manual")
        reason = _is_contradictory_combo((a, b))
        assert reason is not None
        assert "perflevels" in reason

    def test_high_perflevel_blocks_sclk_pin(self):
        # perflevel=high pins clocks to top of DVFS table; an explicit
        # sclk_idx is either redundant or contradictory.
        a = PowerVariant(name="a", perflevel="high")
        b = PowerVariant(name="b", sclk_idx=2)
        reason = _is_contradictory_combo((a, b))
        assert reason is not None
        assert "sclk_idx" in reason or "mclk_idx" in reason

    def test_auto_perflevel_blocks_mclk_pin(self):
        a = PowerVariant(name="a", perflevel="auto")
        b = PowerVariant(name="b", mclk_idx=3)
        reason = _is_contradictory_combo((a, b))
        assert reason is not None

    def test_manual_perflevel_with_clock_pin_is_allowed(self):
        # perflevel=manual is the explicit "clocks set by sclk/mclk"
        # mode; combining with a pin is the intended usage, not a conflict.
        a = PowerVariant(name="a", perflevel="manual")
        b = PowerVariant(name="b", sclk_idx=2)
        assert _is_contradictory_combo((a, b)) is None

    def test_conflicting_determinism_blocked(self):
        a = PowerVariant(name="a", perf_deterministic_mhz=1800)
        b = PowerVariant(name="b", perf_deterministic_mhz=1900)
        reason = _is_contradictory_combo((a, b))
        assert reason is not None
        assert "perf_deterministic_mhz" in reason

    def test_disjoint_device_sets_blocked(self):
        a = PowerVariant(name="a", power_cap_w=300, devices=(0, 1))
        b = PowerVariant(name="b", perflevel="high", devices=(2, 3))
        reason = _is_contradictory_combo((a, b))
        assert reason is not None
        assert "devices" in reason


# ---------------------------------------------------------------------------
# Command generation
# ---------------------------------------------------------------------------
class TestApplyVariantCmds:
    def test_empty_variant_emits_no_commands(self):
        assert _apply_variant_cmds(PowerVariant(name="noop")) == []

    def test_power_cap_uses_setpoweroverdrive(self):
        cmds = _apply_variant_cmds(PowerVariant(name="cap", power_cap_w=275))
        # The upstream rocm-smi Python CLI exposes the cap setter as
        # ``--setpoweroverdrive``; the ``--setpowercap`` alias some docs
        # reference is not actually present on the shipped binary.
        assert any("--setpoweroverdrive 275" in c for c in cmds)
        assert not any("--setpowercap" in c for c in cmds)

    def test_perflevel_present(self):
        cmds = _apply_variant_cmds(PowerVariant(name="pl", perflevel="high"))
        assert any("--setperflevel high" in c for c in cmds)

    def test_devices_emit_per_gpu_flag(self):
        cmds = _apply_variant_cmds(
            PowerVariant(name="d", power_cap_w=200, devices=(0, 2)),
        )
        assert all(" -d 0 -d 2 " in c for c in cmds)

    def test_autorespond_yes_always_appended(self):
        cmds = _apply_variant_cmds(
            PowerVariant(name="x", power_cap_w=200, perflevel="auto"),
        )
        for c in cmds:
            assert c.endswith("--autorespond yes")

    def test_perf_deterministic_zero_skipped(self):
        cmds = _apply_variant_cmds(
            PowerVariant(name="x", perf_deterministic_mhz=0),
        )
        assert not any("--setperfdeterminism" in c for c in cmds)

    def test_perf_deterministic_nonzero_emits(self):
        cmds = _apply_variant_cmds(
            PowerVariant(name="x", perf_deterministic_mhz=1850),
        )
        assert any("--setperfdeterminism 1850" in c for c in cmds)

    def test_fan_pct_emits_percent_suffix(self):
        cmds = _apply_variant_cmds(PowerVariant(name="x", fan_pct=80))
        assert any("--setfan 80%" in c for c in cmds)


# ---------------------------------------------------------------------------
# Probe parsing
# ---------------------------------------------------------------------------
class TestProbePowercapRange:
    def test_parses_max_with_zero_min_sentinel(self, monkeypatch):
        # ``--showmaxpower --json`` emits one ``Max ... Power`` field per
        # device. min_w=0 is the sentinel meaning "no hardware minimum
        # available from this CLI" — the consumer falls back to floor_w.
        payload = """{
          "card0": {"Max Graphics Package Power (W)": "400.0"},
          "card1": {"Max Graphics Package Power (W)": "380.0"}
        }"""
        monkeypatch.setattr(pm, "_run_smi", lambda *a, **k: (0, payload, ""))
        # min-of-maxes (most restrictive ceiling across GPUs).
        assert _probe_powercap_range(()) == (0, 380)

    def test_tolerates_loose_max_power_wording(self, monkeypatch):
        # Cross-CLI-build resilience: some rocm-smi releases word the
        # field slightly differently. The probe filter is loose enough
        # to catch any "max ... power" key.
        payload = """{
          "card0": {"max power (W)": "300"}
        }"""
        monkeypatch.setattr(pm, "_run_smi", lambda *a, **k: (0, payload, ""))
        assert _probe_powercap_range(()) == (0, 300)

    def test_returns_none_on_bad_json(self, monkeypatch):
        monkeypatch.setattr(
            pm, "_run_smi", lambda *a, **k: (0, "not json", ""),
        )
        assert _probe_powercap_range(()) is None

    def test_returns_none_on_run_failure(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("rocm-smi went away")
        monkeypatch.setattr(pm, "_run_smi", boom)
        assert _probe_powercap_range(()) is None

    def test_returns_none_when_payload_has_no_max_field(self, monkeypatch):
        # Hypothetical rocm-smi where --showmaxpower is missing: argparse
        # emits rc=2, _run_smi raises RuntimeError on check=True.
        def boom(*a, **k):
            raise RuntimeError("unrecognized arguments: --showmaxpower")
        monkeypatch.setattr(pm, "_run_smi", boom)
        assert _probe_powercap_range(()) is None


# ---------------------------------------------------------------------------
# Sclk probe (feeds the Stage-2 determinism ladder frequencies)
# ---------------------------------------------------------------------------
class TestProbeTopSclkMhz:
    """``_probe_top_sclk_mhz`` is best-effort by design — when the probe
    fails for ANY reason (missing flag, parse failure, runtime error)
    it returns None and the Stage-2 determinism ladder is skipped (it
    needs a real MHz value to step from). This protects hosts whose
    rocm-smi predates ``--showclkfrq`` from losing the rest of the
    executor."""

    def test_parses_sclk_with_mhz_suffix(self, monkeypatch):
        payload = """{
          "card0": {"sclk[0]": "500Mhz", "sclk[1]": "1200Mhz", "sclk[2]": "1900Mhz"}
        }"""
        monkeypatch.setattr(
            pm, "_run_smi", lambda *a, **k: (0, payload, ""),
        )
        # Top sclk across all reported values.
        assert pm._probe_top_sclk_mhz(()) == 1900

    def test_parses_sclk_with_space_separator(self, monkeypatch):
        payload = """{
          "card0": {"sclk[0]": "500 MHz", "sclk[1]": "1850 MHz"}
        }"""
        monkeypatch.setattr(
            pm, "_run_smi", lambda *a, **k: (0, payload, ""),
        )
        assert pm._probe_top_sclk_mhz(()) == 1850

    def test_parses_newer_max_gpu_clock_shape(self, monkeypatch):
        payload = """{
          "card0": {"Max GPU Clock": "2100Mhz"}
        }"""
        monkeypatch.setattr(
            pm, "_run_smi", lambda *a, **k: (0, payload, ""),
        )
        assert pm._probe_top_sclk_mhz(()) == 2100

    def test_returns_none_on_run_failure(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("unrecognized arguments: --showclkfrq")
        monkeypatch.setattr(pm, "_run_smi", boom)
        assert pm._probe_top_sclk_mhz(()) is None

    def test_returns_none_on_bad_json(self, monkeypatch):
        monkeypatch.setattr(
            pm, "_run_smi", lambda *a, **k: (0, "not json", ""),
        )
        assert pm._probe_top_sclk_mhz(()) is None

    def test_returns_none_when_no_sclk_fields(self, monkeypatch):
        payload = """{"card0": {"unrelated_field": "42"}}"""
        monkeypatch.setattr(
            pm, "_run_smi", lambda *a, **k: (0, payload, ""),
        )
        assert pm._probe_top_sclk_mhz(()) is None


# ---------------------------------------------------------------------------
# Clock-table text parsers (shared with the standalone probe — parity)
# ---------------------------------------------------------------------------
class TestClockTableParsers:
    """`_parse_sclkrange_top_mhz` / `_parse_mclkrange` /
    `_parse_mclk_levels_from_clkfrq` are pure helpers duplicated in the
    standalone probe (parity-tested). Here we lock their behaviour on
    representative rocm-smi shapes."""

    def test_sclkrange_top_picks_largest(self):
        text = "GPU[0]: Valid sclk range: 500Mhz - 2400Mhz"
        assert _parse_sclkrange_top_mhz(text) == 2400

    def test_sclkrange_tolerates_current_marker(self):
        # rocm-smi prints a trailing ``*`` on the active level — the
        # regex consumes only the numeric value + unit, not the marker.
        text = "Valid sclk range: 500Mhz - 2400Mhz *"
        assert _parse_sclkrange_top_mhz(text) == 2400

    def test_sclkrange_empty_returns_none(self):
        assert _parse_sclkrange_top_mhz("no frequencies here") is None

    def test_sclkrange_ghz_normalised(self):
        assert _parse_sclkrange_top_mhz("range: 0.5Ghz - 2.4Ghz") == 2400

    def test_mclkrange_min_max(self):
        assert _parse_mclkrange("Valid mclk range: 900Mhz - 1300Mhz") == (
            900, 1300
        )

    def test_mclkrange_single_freq_returns_none(self):
        assert _parse_mclkrange("only one: 2000Mhz") is None

    def test_mclk_levels_from_clkfrq(self):
        data = {
            "card0": {
                "sclk[0]": "500Mhz", "sclk[1]": "2400Mhz",
                "mclk[0]": "900Mhz", "mclk[1]": "1300Mhz",
            },
        }
        assert _parse_mclk_levels_from_clkfrq(data) == [(0, 900), (1, 1300)]

    def test_mclk_levels_takes_max_per_index_across_cards(self):
        data = {
            "card0": {"mclk[0]": "900Mhz", "mclk[1]": "1200Mhz"},
            "card1": {"mclk[0]": "950Mhz", "mclk[1]": "1300Mhz"},
        }
        assert _parse_mclk_levels_from_clkfrq(data) == [(0, 950), (1, 1300)]

    def test_mclk_levels_empty_on_no_keys(self):
        assert _parse_mclk_levels_from_clkfrq({"card0": {"sclk[0]": "1Mhz"}}) == []
        assert _parse_mclk_levels_from_clkfrq(None) == []


# ---------------------------------------------------------------------------
# Mclk capability probe (drives the memory-axis >= 2-level gate)
# ---------------------------------------------------------------------------
class TestProbeMclkLevels:
    """`_probe_mclk_levels` reads the labeled ``mclk[N]`` section of
    ``--showclkfrq --json`` plus ``--showmclkrange`` and returns the
    selectable-level set the memory-axis capability gate keys on."""

    def test_two_levels_capable(self, monkeypatch):
        clkfrq = {
            "card0": {"mclk[0]": "1000Mhz", "mclk[1]": "2000Mhz"},
        }
        monkeypatch.setattr(
            pm, "_probe_mclkrange_raw",
            lambda *_a, **_k: "Valid mclk range: 1000Mhz - 2000Mhz",
        )
        out = _probe_mclk_levels((), clkfrq_data=clkfrq)
        assert out["count"] == 2
        assert out["indices"] == [0, 1]
        assert out["mhz"] == [1000, 2000]
        assert out["range"] == (1000, 2000)

    def test_single_level_not_capable(self, monkeypatch):
        # This MI355X: one selectable mclk level → gate will skip the axis.
        clkfrq = {"card0": {"mclk[0]": "2000Mhz"}}
        monkeypatch.setattr(pm, "_probe_mclkrange_raw", lambda *_a, **_k: None)
        out = _probe_mclk_levels((), clkfrq_data=clkfrq)
        assert out["count"] == 1
        assert out["range"] is None

    def test_no_mclk_keys_yields_zero(self, monkeypatch):
        monkeypatch.setattr(pm, "_probe_mclkrange_raw", lambda *_a, **_k: None)
        out = _probe_mclk_levels((), clkfrq_data={"card0": {"sclk[0]": "1Mhz"}})
        assert out["count"] == 0
        assert out["indices"] == []


# ---------------------------------------------------------------------------
# Executor — dry-run path
# ---------------------------------------------------------------------------
class TestExecutorDryRun:
    def test_dry_run_with_explicit_grid_returns_resolved_grid(self, tmp_path):
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "dry_run": True,
            "grid": [
                {"name": "a", "power_cap_w": 200},
                {"name": "b", "perflevel": "high"},
            ],
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "succeeded"
        assert result["dry_run"] is True
        assert result["grid_size"] == 2
        assert {v["name"] for v in result["resolved_grid"]} == {"a", "b"}
        assert result["final_state"] == "untouched"
        assert result["winners"] == []

    def test_dry_run_rejects_bad_payload(self, tmp_path):
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {"dry_run": True, "grid": [{"name": "a", "fan_pct": 200}]}
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "failed"
        assert result["error_class"] == "bad_param"
        assert "fan_pct=200" in result["error"]

    def test_dry_run_below_floor_rejected(self, tmp_path):
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "dry_run": True,
            "grid": [{"name": "a", "power_cap_w": 50}],
            "power_cap_floor_w": 150,
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "failed"
        assert result["error_class"] == "bad_param"
        assert "below floor" in result["error"]


# ---------------------------------------------------------------------------
# Executor — multi-node hard refusal
# ---------------------------------------------------------------------------
class TestExecutorMultiNodeRefusal:
    """``rocm-smi`` is node-local; running this action on a multi-node
    RayJob would only apply the chosen state to the head node and
    leave peer workers at defaults. The executor refuses at the
    door with ``error_class='multi_node_unsupported'``, with a
    ``dry_run`` carve-out so the grid-resolution code path stays
    exercisable under simulated multi-node config.
    """

    def test_refuses_multi_node_with_dedicated_error_class(
        self, tmp_path, monkeypatch,
    ):
        # Patch the module-level _multi_node_env.is_multi_node so the
        # executor's late import resolves to True. Patching the import
        # site directly (the executor uses ``from ._multi_node_env
        # import is_multi_node`` inside __call__) means we need to
        # patch the source module, not power_management.
        from inference_optimizer.orchestrator.action_executors import (
            _multi_node_env as mne,
        )
        monkeypatch.setattr(mne, "is_multi_node", lambda: True)

        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "a", "power_cap_w": 250}],
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "failed"
        assert result["error_class"] == "multi_node_unsupported"
        assert "single-node only" in result["error"]
        # The refusal must fire BEFORE any rocm-smi work — the result
        # payload should not carry probe-derived bookkeeping fields
        # (no floor/ceiling/probed_range, no grid_size, no workspace).
        assert "power_floor_w" not in result
        assert "resolved_grid" not in result

    def test_dry_run_bypasses_multi_node_refusal(self, tmp_path, monkeypatch):
        # The dry-run carve-out is what keeps tests + the probe script
        # able to exercise the grid-resolution code under simulated
        # multi-node config without touching the GPU. Asserting the
        # carve-out is intentional, not accidental, so a future
        # tightening of the guard doesn't silently break the harness.
        from inference_optimizer.orchestrator.action_executors import (
            _multi_node_env as mne,
        )
        monkeypatch.setattr(mne, "is_multi_node", lambda: True)

        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "dry_run": True,
            "grid": [{"name": "a", "power_cap_w": 250}],
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "succeeded"
        assert result["dry_run"] is True


# ---------------------------------------------------------------------------
# Executor — probe-driven runtime lift of `power_cap_floor_w`
# ---------------------------------------------------------------------------
class TestExecutorFloorLift:
    """The probed hardware minimum (when one is reported) must win over
    the operator/default soft floor whenever it is stricter. This
    prevents the silent failure mode where an LLM proposes a cap that
    passes the 150 W default floor but is below what the GPU's own
    ``--setpoweroverdrive`` will accept.

    The current upstream ``rocm-smi`` CLI does not expose a
    hardware-minimum reading, so :func:`_probe_powercap_range` returns
    ``min_w=0`` and the lift is dormant in production. These tests
    monkeypatch the probe to return non-zero minima so the lift code
    stays exercised — when a future CLI revision (or an ``amd-smi``
    re-target) reports the minimum, the existing logic picks it up
    automatically.
    """

    def test_probed_min_lifts_floor_and_rejects_variant(
        self, tmp_path, monkeypatch, caplog,
    ):
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: True)
        # Hardware min (220 W) is well above the default 150 W floor and
        # also above the operator's 200 W proposal — the variant must
        # be rejected against the lifted (hardware) floor.
        monkeypatch.setattr(
            pm, "_probe_powercap_range", lambda *_a, **_k: (220, 400),
        )
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "too_low", "power_cap_w": 200}],
            "output_dir": str(tmp_path / "ws"),
        }
        with caplog.at_level("INFO", logger=pm.__name__):
            result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "failed"
        assert result["error_class"] == "bad_param"
        assert "below floor=220" in result["error"]
        # Operator should see the lift in the logs.
        assert any(
            "lifting floor" in rec.message and "220" in rec.message
            for rec in caplog.records
        ), caplog.text

    def test_probed_min_below_operator_floor_keeps_operator_floor(
        self, tmp_path, monkeypatch,
    ):
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: True)
        # Hardware min is LOOSER than the operator floor — the operator
        # wins (we never relax their explicit lower bound).
        monkeypatch.setattr(
            pm, "_probe_powercap_range", lambda *_a, **_k: (100, 400),
        )
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "too_low", "power_cap_w": 175}],
            "power_cap_floor_w": 200,
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "failed"
        assert result["error_class"] == "bad_param"
        assert "below floor=200" in result["error"]

    def test_dry_run_skips_probe_and_keeps_operator_floor(
        self, tmp_path, monkeypatch,
    ):
        """``dry_run=true`` short-circuits before the probe runs, so a
        hardware minimum that would have lifted the floor must NOT be
        consulted (no rocm-smi calls at all). This is the contract
        callers rely on for offline grid validation."""
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: True)
        # If the probe were called, this would lift the floor to 300
        # and reject the 175 W variant. The test asserts it isn't.
        probe_calls: list = []
        def _spy_probe(*a, **k):
            probe_calls.append((a, k))
            return (300, 400)
        monkeypatch.setattr(pm, "_probe_powercap_range", _spy_probe)
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "dry_run": True,
            "grid": [{"name": "ok", "power_cap_w": 175}],
            "power_cap_floor_w": 150,
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "succeeded"
        assert probe_calls == [], "dry_run must not invoke the probe"


# ---------------------------------------------------------------------------
# Executor — roofline-routed settle grid end-to-end wiring
# ---------------------------------------------------------------------------
class TestExecutorSettleGrid:
    """End-to-end coverage of the Coordinator-internal settle path (no
    explicit ``params.grid``): the executor must (a) build the
    roofline-routed grid from the probed ceiling + top sclk + mclk
    capability, (b) bench ``auto_baseline`` N reps (median) as the gate
    reference + each challenger once, (c) promote the highest median
    that clears the noise floor over auto (else keep auto), and (d)
    surface ``bound_kind`` / ``grid_size`` / ``grid_degraded`` /
    ``reference_source`` in the payload."""

    def _stub_bench(
        self, monkeypatch, tput_by_name: dict[str, float],
        *, top_sclk_mhz: int | None = 2000,
        gfx_high_idx: int | None = 3,
        mclk_levels: dict[str, Any] | None = None,
    ):
        """Replace ``_run_one_variant`` with a deterministic stub keyed
        on the variant's name, and stub every rocm-smi probe / apply so
        the test never touches a GPU."""
        async def _fake_run(self_, v, *, base_tput: float, rep=None, **_k):
            tput = tput_by_name.get(v.name, 0.0)
            gain_pct = (
                (tput - base_tput) / base_tput * 100.0
                if base_tput > 0 and tput > 0 else 0.0
            )
            return {
                "variant_name":     v.name,
                "status":           "succeeded" if tput > 0 else "failed",
                "output_throughput": tput,
                "gain_pct":         gain_pct,
                "power_settings":   v.to_dict(),
            }
        monkeypatch.setattr(
            PowerManagementExecutor, "_run_one_variant", _fake_run,
            raising=True,
        )
        monkeypatch.setattr(pm, "_apply_variant", lambda *_a, **_k: [])
        monkeypatch.setattr(pm, "_reset_defaults", lambda: None)
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: True)
        monkeypatch.setattr(
            pm, "_probe_powercap_range", lambda *_a, **_k: (200, 400),
        )
        monkeypatch.setattr(
            pm, "_probe_current_state",
            lambda *_a, **_k: {"powercap_w": None, "perflevel": "auto"},
        )
        # The executor fetches the raw --showclkfrq payload once and
        # threads it into the clock parsers; stub it out so __call__
        # never shells out, then stub the parsers with canned values.
        monkeypatch.setattr(pm, "_probe_clkfrq_raw", lambda *_a, **_k: None)
        monkeypatch.setattr(
            pm, "_probe_top_sclk_mhz", lambda *_a, **_k: top_sclk_mhz,
        )
        monkeypatch.setattr(
            pm, "_gfx_high_sclk_idx", lambda *_a, **_k: gfx_high_idx,
        )
        monkeypatch.setattr(
            pm, "_probe_mclk_levels",
            lambda *_a, **_k: (mclk_levels if mclk_levels is not None
                               else {"count": 1, "indices": [0],
                                     "mhz": [2000], "range": None}),
        )

    def test_det_winner_beats_auto_baseline(self, tmp_path, monkeypatch):
        # unknown roofline + single mclk level → grid = auto_baseline,
        # high, det_100/95/90. det_100 beats the auto reference and wins.
        self._stub_bench(monkeypatch, {
            "auto_baseline": 100.0,   # N-rep median reference
            "high":          105.0,
            "det_90":        110.0,
            "det_95":        120.0,
            "det_100":       130.0,   # overall winner
        })
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "base_tput":  100.0,
            "bound_kind": "unknown",
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "succeeded"
        assert result["bound_kind"] == "unknown"
        # auto_baseline is the gate reference, NOT a challenger.
        assert result["reference_source"] == "auto_baseline"
        assert result["reference_tput"] == pytest.approx(100.0)
        assert result["best_variant"]["variant_name"] == "det_100"
        # Gain measured against the auto_baseline median (100): +30%.
        assert result["best_gain_pct"] == pytest.approx(30.0)
        assert result["final_state"] == "applied_best"
        assert result["grid_degraded"] is None
        # The auto_baseline median seeds attribution directly.
        assert result["kernel_baseline_tput"] == pytest.approx(100.0)

    def test_auto_baseline_runs_n_reps_and_takes_median(
        self, tmp_path, monkeypatch,
    ):
        # The auto_baseline reference row benches ``auto_baseline_reps``
        # times; the median is the gate denominator. Count rep calls.
        rep_calls: list[int | None] = []

        async def _fake_run(self_, v, *, base_tput, rep=None, **_k):
            if v.name == "auto_baseline":
                rep_calls.append(rep)
                # Three reps with a clear median of 100.
                tput = {0: 90.0, 1: 100.0, 2: 130.0}.get(rep or 0, 100.0)
            else:
                tput = 150.0
            return {
                "variant_name": v.name,
                "status": "succeeded",
                "output_throughput": tput,
                "gain_pct": 0.0,
                "power_settings": v.to_dict(),
            }
        self._stub_bench(monkeypatch, {})
        monkeypatch.setattr(
            PowerManagementExecutor, "_run_one_variant", _fake_run,
            raising=True,
        )
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "base_tput":  100.0,
            "bound_kind": "compute",   # prune det rungs → small grid
            "auto_baseline_reps": 3,
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert rep_calls == [0, 1, 2]
        # median(90, 100, 130) == 100 → the gate reference.
        assert result["reference_tput"] == pytest.approx(100.0)
        auto_row = next(
            r for r in result["all_results"]
            if r["variant_name"] == "auto_baseline"
        )
        assert auto_row["output_throughput"] == pytest.approx(100.0)
        assert auto_row["reps"] == 3

    def test_compute_bound_prunes_det_rungs(self, tmp_path, monkeypatch):
        # compute-bound → only det_100 challenger (+ high); det_95/det_90
        # never run.
        self._stub_bench(monkeypatch, {
            "auto_baseline": 100.0,
            "high":          105.0,
            "det_100":       130.0,
        })
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "base_tput":  100.0,
            "bound_kind": "compute",
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        names = {r["variant_name"] for r in result["all_results"]}
        assert names == {"auto_baseline", "high", "det_100"}
        assert result["grid_degraded"] is None

    def test_grid_degraded_when_no_top_sclk(self, tmp_path, monkeypatch):
        # GRID SELF-CHECK: the determinism ladder was expected (det_100
        # on every box) but no top sclk was probed → 0 det rows AND a
        # structured grid_degraded entry surfaced in the result (no
        # silent collapse to auto/high-only).
        self._stub_bench(monkeypatch, {
            "auto_baseline": 100.0,
            "high":          105.0,
        }, top_sclk_mhz=None)
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "base_tput":  100.0,
            "bound_kind": "unknown",
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        names = {r["variant_name"] for r in result["all_results"]}
        assert names == {"auto_baseline", "high"}
        assert result["grid_degraded"] is not None
        assert "gfx_determinism" in result["grid_degraded"]

    def test_memory_axis_runs_when_capable(self, tmp_path, monkeypatch):
        # compute-bound (det rungs pruned) + >= 2 mclk levels → the
        # capability-gated memory row runs (GFX pinned high, memory
        # stepped down).
        self._stub_bench(monkeypatch, {
            "auto_baseline": 100.0,
            "high":          105.0,
            "det_100":       110.0,
            "mclk_1000mhz":  125.0,   # winner
        }, gfx_high_idx=3,
            mclk_levels={"count": 2, "indices": [0, 1],
                         "mhz": [1000, 2000], "range": None})
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "base_tput":  100.0,
            "bound_kind": "compute",
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        names = {r["variant_name"] for r in result["all_results"]}
        assert "mclk_1000mhz" in names
        assert result["best_variant"]["variant_name"] == "mclk_1000mhz"
        assert result["grid_degraded"] is None

    def test_no_challenger_beats_auto_keeps_incumbent(
        self, tmp_path, monkeypatch,
    ):
        # Every challenger is within the noise floor of auto → keep auto:
        # re-apply the auto_baseline state, final_state=kept_incumbent.
        self._stub_bench(monkeypatch, {
            "auto_baseline": 100.0,
            "high":          100.5,   # +0.5% — below the 2% gate
            "det_100":       101.0,   # +1.0% — below the 2% gate
            "det_95":        100.2,
            "det_90":         99.0,
        })
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "base_tput":  100.0,
            "bound_kind": "unknown",
            "keep_threshold_pct": 2.0,
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "no_winners"
        assert result["final_state"] == "kept_incumbent"
        assert result["host_state_applied"]["variant_name"] == "auto_baseline"

    def test_explicit_grid_has_no_auto_baseline(
        self, tmp_path, monkeypatch,
    ):
        # An explicit params.grid is benched as-is against base_tput —
        # no auto_baseline row is synthesised (operator owns the shape).
        self._stub_bench(monkeypatch, {
            "a": 110.0,
            "b": 120.0,
        })
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [
                {"name": "a", "power_cap_w": 300},
                {"name": "b", "perflevel": "high"},
            ],
            "base_tput":  100.0,
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        names = {r["variant_name"] for r in result["all_results"]}
        assert names == {"a", "b"}
        assert result["reference_source"] == "base_tput"
        assert result["best_variant"]["variant_name"] == "b"
        assert result["grid_degraded"] is None


# ---------------------------------------------------------------------------
# Executor — failure paths that don't need a real Magpie subprocess
# ---------------------------------------------------------------------------
class TestExecutorFailurePaths:
    def test_rocm_smi_missing_returns_clean_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: False)
        ex = PowerManagementExecutor(session_dir=tmp_path)
        result = asyncio.run(ex(_Ctx(task=_task({}))))
        assert result["status"] == "failed"
        assert result["error_class"] == "rocm_smi_unavailable"

    def test_no_probe_no_grid_returns_empty_grid(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: True)
        # Probe yields None (rocm-smi missing --showmaxpower, JSON parse
        # failure, missing binary, etc.) AND no explicit grid -> the
        # executor refuses to synthesise a default sweep.
        monkeypatch.setattr(pm, "_probe_powercap_range", lambda *_a, **_k: None)
        ex = PowerManagementExecutor(session_dir=tmp_path)
        result = asyncio.run(ex(_Ctx(task=_task({}))))
        assert result["status"] == "failed"
        assert result["error_class"] == "empty_grid"

    def test_bad_script_name_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: True)
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "a", "power_cap_w": 200}],
            "benchmark_script": "../etc/passwd",
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "failed"
        assert result["error_class"] == "bad_param"

    def test_missing_config_returns_clean_failure(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: True)
        ex = PowerManagementExecutor(
            session_dir=tmp_path,
            default_config_path=tmp_path / "nope.yaml",
        )
        params = {"grid": [{"name": "a", "power_cap_w": 200}]}
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "failed"
        assert result["error_class"] == "missing_config"

    def test_mid_sweep_crash_returns_structured_failure(
        self, tmp_path, monkeypatch,
    ):
        # Mid-sweep crashes used to ``raise`` so SubAgentRunner saw
        # ``SubAgentResult(state='failed', result={})``: the empty
        # ``result`` never reached the Coordinator's power_management
        # audit branch, so ``host_state_applied`` stayed pointing at
        # a winner whose state we'd just blown away with
        # ``_reset_defaults``. The executor now converts the crash
        # into a structured failure with ``final_state=
        # 'reset_after_failure'`` so the Coordinator can clear the
        # cache, AND skips the ledger write so a partial-round
        # update doesn't persist tested fingerprints for variants
        # that didn't actually run to completion.
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: True)
        monkeypatch.setattr(
            pm, "_probe_powercap_range", lambda *_a, **_k: (200, 400),
        )
        monkeypatch.setattr(pm, "_probe_top_sclk_mhz", lambda *_a, **_k: None)
        monkeypatch.setattr(pm, "_apply_variant", lambda v: ["smi cmd"])
        monkeypatch.setattr(pm, "_reset_defaults", lambda: None)
        monkeypatch.setattr(pm, "_snapshot_state", lambda: "")
        # Force the first per-variant rebench to crash. The exception
        # must not propagate out of ``__call__`` — the executor
        # catches it, resets defaults, and builds a failure payload.
        async def _boom(self_, v, **_kw):
            raise RuntimeError("synthetic Magpie crash")
        monkeypatch.setattr(
            PowerManagementExecutor, "_run_one_variant", _boom,
        )
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "cap_a", "power_cap_w": 250}],
            "output_dir": str(tmp_path / "ws"),
        }
        # Create a stub config file so we don't trip missing_config.
        cfg = tmp_path / "ws" / "config.yaml"
        cfg.parent.mkdir(parents=True, exist_ok=True)
        cfg.write_text("benchmark:\n  benchmark_script: x.sh\n")
        params["config_path"] = str(cfg)

        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "failed"
        assert result["error_class"] == "unhandled_exception"
        assert "synthetic Magpie crash" in result["error"]
        assert result["final_state"] == "reset_after_failure"
        # host_state_applied is explicitly None so the Coordinator
        # clears its cache; ledger update is None so the Coordinator
        # skips the merge for this round.
        assert result["host_state_applied"] is None
        assert result["power_management_search_update"] is None


# ---------------------------------------------------------------------------
# Winner selection helper
# ---------------------------------------------------------------------------
class TestWinnerSelection:
    """``_is_winner`` is the per-iteration gate. Each variant runs as
    its own single-row ``run_grid`` (one Magpie subprocess), so the
    structural analog is framework_agent's
    ``delta_pct >= keep_threshold_pct`` — hence the inclusive ``>=``
    and the ``keep_threshold_pct`` parameter name (shared with
    :mod:`explore` and :mod:`framework_agent`). The noise-floor *value*
    is imported from :mod:`_grid_runner` so we agree with explore on
    what counts as Magpie jitter (1.0 % single-node, 2.0 %
    multi-node)."""

    def test_positive_gain_with_base_tput(self):
        ex = PowerManagementExecutor()
        ok = ex._is_winner(
            {"status": "succeeded", "gain_pct": 4.0, "output_throughput": 110},
            reference_tput=100.0,
            keep_threshold_pct=pm.SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT,
        )
        assert ok is True

    def test_below_threshold_loses(self):
        ex = PowerManagementExecutor()
        ok = ex._is_winner(
            {"status": "succeeded", "gain_pct": 0.2, "output_throughput": 100.2},
            reference_tput=100.0,
            keep_threshold_pct=pm.SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT,
        )
        assert ok is False

    def test_exactly_at_threshold_wins(self):
        # Inclusive `>=` to match framework_agent's per-iteration gate
        # (framework_agent.py:832). Each of our iterations is structurally
        # one framework_agent bench-and-check; a variant that exactly
        # clears the noise floor must be treated identically by both.
        ex = PowerManagementExecutor()
        ok = ex._is_winner(
            {"status": "succeeded", "gain_pct": 1.0, "output_throughput": 101.0},
            reference_tput=100.0,
            keep_threshold_pct=pm.SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT,
        )
        assert ok is True

    def test_failed_status_loses(self):
        ex = PowerManagementExecutor()
        ok = ex._is_winner(
            {"status": "failed", "gain_pct": 5.0},
            reference_tput=100.0,
            keep_threshold_pct=pm.SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT,
        )
        assert ok is False

    def test_no_base_tput_accepts_any_positive_tput(self):
        ex = PowerManagementExecutor()
        ok = ex._is_winner(
            {"status": "succeeded", "gain_pct": 0.0, "output_throughput": 99.9},
            reference_tput=0.0,
            keep_threshold_pct=pm.SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT,
        )
        assert ok is True

    def test_explicit_stricter_threshold_rejects_borderline_gain(self):
        # Locks in the gating math for an operator-supplied stricter
        # threshold (the executor itself only consults the single-node
        # default — see the ``SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT``
        # branch in ``PowerManagementExecutor.__call__`` — because the
        # action refuses multi-node entry, but the
        # ``_is_winner`` predicate stays threshold-agnostic so a future
        # caller can still gate at the elevated multi-node noise floor).
        # A 1.5% gain wins at the 1.0% single-node floor and loses
        # against a 2.0% threshold; the comparator is inclusive ``>=``
        # so the threshold value itself is treated as a winner.
        ex = PowerManagementExecutor()
        entry = {"status": "succeeded", "gain_pct": 1.5, "output_throughput": 101.5}
        assert ex._is_winner(
            entry, reference_tput=100.0,
            keep_threshold_pct=pm.SINGLE_NODE_DEFAULT_KEEP_THRESHOLD_PCT,
        ) is True
        assert ex._is_winner(
            entry, reference_tput=100.0, keep_threshold_pct=2.0,
        ) is False

    def test_per_task_keep_threshold_pct_override(self):
        # When the LLM supplies a stricter threshold via task params,
        # a borderline gain that would clear the default falls below.
        ex = PowerManagementExecutor()
        ok = ex._is_winner(
            {"status": "succeeded", "gain_pct": 1.5, "output_throughput": 101.5},
            reference_tput=100.0,
            keep_threshold_pct=5.0,
        )
        assert ok is False


# ---------------------------------------------------------------------------
# Variant fingerprint — stable cross-call identity key
# ---------------------------------------------------------------------------
class TestPowerVariantFingerprint:
    """``power_variant_fingerprint`` is what the cross-call search
    ledger keys on. Renames must NOT change identity (same knobs →
    same fp); any knob change MUST flip the fp; device ordering must
    not matter (sets, not sequences)."""

    def test_same_knobs_different_names_same_fingerprint(self):
        a = PowerVariant(name="cap_80pct_high", power_cap_w=300, perflevel="high")
        b = PowerVariant(name="my_rename", power_cap_w=300, perflevel="high")
        assert power_variant_fingerprint(a.to_dict()) == \
               power_variant_fingerprint(b.to_dict())

    def test_different_caps_different_fingerprints(self):
        a = PowerVariant(name="x", power_cap_w=300, perflevel="high")
        b = PowerVariant(name="x", power_cap_w=350, perflevel="high")
        assert power_variant_fingerprint(a.to_dict()) != \
               power_variant_fingerprint(b.to_dict())

    def test_devices_set_semantics(self):
        # Order doesn't matter; the fingerprint normalizes to a sorted set.
        a = PowerVariant(name="x", power_cap_w=300, devices=(0, 2, 1))
        b = PowerVariant(name="x", power_cap_w=300, devices=(2, 0, 1))
        assert power_variant_fingerprint(a.to_dict()) == \
               power_variant_fingerprint(b.to_dict())

    def test_note_excluded_from_fingerprint(self):
        # ``note`` is metadata for humans, not part of identity.
        a = PowerVariant(name="x", power_cap_w=300, note="round_1")
        b = PowerVariant(name="x", power_cap_w=300, note="round_42")
        assert power_variant_fingerprint(a.to_dict()) == \
               power_variant_fingerprint(b.to_dict())

    def test_accepts_powervariant_instance(self):
        v = PowerVariant(name="x", power_cap_w=300, perflevel="high")
        fp_inst = power_variant_fingerprint(v)
        fp_dict = power_variant_fingerprint(v.to_dict())
        assert fp_inst == fp_dict

    def test_fingerprint_is_16_hex_chars(self):
        v = PowerVariant(name="x", power_cap_w=300)
        fp = power_variant_fingerprint(v.to_dict())
        assert len(fp) == 16
        assert all(c in "0123456789abcdef" for c in fp)


# ---------------------------------------------------------------------------
# Rehydration helper — `accepted` ledger entry → PowerVariant
# ---------------------------------------------------------------------------
class TestVariantFromAccepted:
    def test_round_trips_full_knob_set(self):
        v = PowerVariant(
            name="anchor", power_cap_w=320, perflevel="high",
            perf_deterministic_mhz=1900, fan_pct=85, devices=(0, 1),
        )
        entry = {
            "name": v.name,
            "power_settings": v.to_dict(),
            "fingerprint": power_variant_fingerprint(v.to_dict()),
        }
        out = _variant_from_accepted(entry)
        assert out is not None
        # note is overridden to mark it as a revalidate row.
        assert out.note == "prior_winner_revalidate"
        # All other knobs round-trip.
        assert out.power_cap_w == 320
        assert out.perflevel == "high"
        assert out.perf_deterministic_mhz == 1900
        assert out.fan_pct == 85
        assert out.devices == (0, 1)
        # Fingerprint is stable across rehydration (only note differs,
        # which is excluded from the fp).
        assert power_variant_fingerprint(out.to_dict()) == entry["fingerprint"]

    def test_missing_power_settings_returns_none(self):
        assert _variant_from_accepted({"name": "x"}) is None

    def test_malformed_entry_returns_none(self):
        assert _variant_from_accepted("not a dict") is None  # type: ignore[arg-type]
        assert _variant_from_accepted({}) is None

    def test_synthesises_name_when_missing(self):
        # Caller didn't preserve the display name — fall back to a
        # fingerprint-prefixed synthetic name so audit trails still
        # have something to render.
        entry = {
            "fingerprint": "abcdef1234567890",
            "power_settings": {"power_cap_w": 300},
        }
        out = _variant_from_accepted(entry)
        assert out is not None
        assert out.name.startswith("prior_winner_abcdef12")


# ---------------------------------------------------------------------------
# Host-state staleness check — drives lazy revalidation
# ---------------------------------------------------------------------------
class TestHostStateIsStale:
    def test_matching_state_not_stale(self):
        cached = {"powercap_w": 300, "perflevel": "high"}
        current = {"powercap_w": 300, "perflevel": "high"}
        assert _host_state_is_stale(current=current, cached=cached) is False

    def test_powercap_drift_stale(self):
        # Operator manually tweaked rocm-smi between calls.
        cached = {"powercap_w": 300, "perflevel": "high"}
        current = {"powercap_w": 350, "perflevel": "high"}
        assert _host_state_is_stale(current=current, cached=cached) is True

    def test_perflevel_drift_stale(self):
        cached = {"powercap_w": 300, "perflevel": "high"}
        current = {"powercap_w": 300, "perflevel": "auto"}
        assert _host_state_is_stale(current=current, cached=cached) is True

    def test_missing_cached_field_skipped(self):
        # Partial probe failure shouldn't force re-validation cascade.
        cached = {"powercap_w": 300, "perflevel": None}
        current = {"powercap_w": 300, "perflevel": "auto"}
        assert _host_state_is_stale(current=current, cached=cached) is False

    def test_missing_current_field_skipped(self):
        cached = {"powercap_w": 300, "perflevel": "high"}
        current = {"powercap_w": 300, "perflevel": None}
        assert _host_state_is_stale(current=current, cached=cached) is False

    def test_empty_cache_not_stale(self):
        # No prior state means "fresh start" — caller shouldn't escalate.
        assert _host_state_is_stale(current={"powercap_w": 300}, cached=None) is False
        assert _host_state_is_stale(current={"powercap_w": 300}, cached={}) is False


# ---------------------------------------------------------------------------
# Initial-state ledger shape
# ---------------------------------------------------------------------------
class TestInitialPowerManagementSearchState:
    def test_canonical_shape(self):
        st = _initial_power_management_search_state()
        assert st["schema_version"] == 1
        assert st["accepted"] == []
        assert st["rejected"] == []
        assert st["tested"] == {}
        assert st["name_index"] == {}
        assert st["cursor"] == 0
        assert st["last_round"] == {}


# ---------------------------------------------------------------------------
# Cross-call dedup + re-validation behaviour
# ---------------------------------------------------------------------------
class TestExecutorCrossCallLedger:
    """End-to-end coverage of the search-ledger plumbing: prior tested
    fingerprints get dropped, prior winners can be re-validated, the
    result payload emits a search update + (when applicable) a
    host_state_applied snapshot."""

    def _stub_bench(self, monkeypatch, tput_by_name: dict[str, float]):
        async def _fake_run(self_, v, *, base_tput: float, **_k):
            tput = tput_by_name.get(v.name, 0.0)
            gain_pct = (
                (tput - base_tput) / base_tput * 100.0
                if base_tput > 0 and tput > 0 else 0.0
            )
            return {
                "variant_name":      v.name,
                "status":            "succeeded" if tput > 0 else "failed",
                "output_throughput": tput,
                "gain_pct":          gain_pct,
                "power_settings":    v.to_dict(),
            }
        monkeypatch.setattr(
            PowerManagementExecutor, "_run_one_variant", _fake_run,
            raising=True,
        )
        monkeypatch.setattr(pm, "_apply_variant", lambda *_a, **_k: [
            "rocm-smi --setpoweroverdrive 300 --autorespond yes",
            "rocm-smi --setperflevel high --autorespond yes",
        ])
        monkeypatch.setattr(pm, "_reset_defaults", lambda: None)
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: True)
        monkeypatch.setattr(
            pm, "_probe_powercap_range", lambda *_a, **_k: (200, 400),
        )
        monkeypatch.setattr(
            pm, "_probe_top_sclk_mhz", lambda *_a, **_k: 1900,
        )
        # Default: no drift. Tests override individually.
        monkeypatch.setattr(
            pm, "_probe_current_state",
            lambda *_a, **_k: {"powercap_w": 300, "perflevel": "high"},
        )

    def test_result_includes_search_update_with_round_winners(
        self, tmp_path, monkeypatch,
    ):
        # Single winner round: the result payload must carry a
        # power_management_search_update with the round's tested
        # fingerprints + a non-empty last_round.round_winners.
        self._stub_bench(monkeypatch, {
            "cap_300": 115.0,  # +15% — clears single-node 1% floor
            "perf_h":  100.5,  # +0.5% — below the floor → not a winner
        })
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [
                {"name": "cap_300", "power_cap_w": 300},
                {"name": "perf_h",  "perflevel":  "high"},
            ],
            "base_tput":  100.0,
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        update = result["power_management_search_update"]
        assert update["schema_version"] == 1
        assert update["cursor"] == 2
        # Both fingerprints in tested; only the cap_300 winner in round_winners.
        assert len(update["tested"]) == 2
        winner_fp = power_variant_fingerprint({
            "power_cap_w": 300, "perflevel": None,
            "sclk_idx": None, "mclk_idx": None, "pcie_idx": None,
            "perf_deterministic_mhz": None, "fan_pct": None, "devices": [],
        })
        assert update["last_round"]["round_winners"] == [winner_fp]
        # The non-winner appears in rejected.
        rejected_fps = {r["fingerprint"] for r in update["rejected"]}
        assert winner_fp not in rejected_fps

    def test_already_tested_fingerprints_are_dropped(
        self, tmp_path, monkeypatch,
    ):
        # Prior ledger says cap_300 was already tested. The second
        # call must NOT re-bench it (only the genuinely-new variant
        # runs) AND the dropped fingerprint must surface in
        # ``dropped_variants`` for prompt rendering.
        self._stub_bench(monkeypatch, {
            "perf_h": 115.0,
        })
        prior_fp = power_variant_fingerprint({
            "power_cap_w": 300, "perflevel": None,
            "sclk_idx": None, "mclk_idx": None, "pcie_idx": None,
            "perf_deterministic_mhz": None, "fan_pct": None, "devices": [],
        })
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [
                {"name": "cap_300", "power_cap_w": 300},
                {"name": "perf_h",  "perflevel":  "high"},
            ],
            "base_tput":  100.0,
            "output_dir": str(tmp_path / "ws"),
            "power_management_search": {
                "schema_version": 1,
                "accepted": [],
                "rejected": [],
                "tested": {
                    prior_fp: {
                        "name": "cap_300",
                        "power_settings": {
                            "power_cap_w": 300, "perflevel": None,
                            "devices": [],
                        },
                        "status": "succeeded",
                        "gain_pct": 0.5,
                        "fingerprint": prior_fp,
                    },
                },
                "name_index": {"cap_300": prior_fp},
                "cursor": 1,
            },
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        # Only perf_h ran.
        names_run = {r["variant_name"] for r in result["all_results"]}
        assert names_run == {"perf_h"}
        # Dropped surface lists the deduped one with a reason.
        dropped_names = {d["name"] for d in result["dropped_variants"]}
        assert dropped_names == {"cap_300"}
        assert result["dropped_variants"][0]["reason"] == "already_tested"

    def test_force_retest_bypasses_dedup(self, tmp_path, monkeypatch):
        # Operator wants to re-bench a previously-tested variant
        # (e.g. after a driver upgrade changed hardware behaviour).
        # force_retest=true skips the dedup filter entirely.
        self._stub_bench(monkeypatch, {
            "cap_300": 115.0,
        })
        prior_fp = power_variant_fingerprint({
            "power_cap_w": 300, "perflevel": None,
            "sclk_idx": None, "mclk_idx": None, "pcie_idx": None,
            "perf_deterministic_mhz": None, "fan_pct": None, "devices": [],
        })
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "cap_300", "power_cap_w": 300}],
            "base_tput":  100.0,
            "output_dir": str(tmp_path / "ws"),
            "force_retest": True,
            "power_management_search": {
                "schema_version": 1, "accepted": [], "rejected": [],
                "tested": {
                    prior_fp: {
                        "fingerprint": prior_fp,
                        "power_settings": {"power_cap_w": 300, "devices": []},
                    },
                },
                "name_index": {}, "cursor": 1,
            },
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert {r["variant_name"] for r in result["all_results"]} == {"cap_300"}
        assert result["dropped_variants"] == []

    def test_lazy_revalidate_skipped_when_cache_fresh(
        self, tmp_path, monkeypatch,
    ):
        # Default ``lazy`` mode + cache-fresh probe: prior winners
        # are NOT re-benched (we trust the cache). Only the fresh
        # grid runs.
        self._stub_bench(monkeypatch, {
            "perf_h": 120.0,
        })
        # Cached state matches probe → no drift.
        cached = {"powercap_w": 300, "perflevel": "high"}
        monkeypatch.setattr(
            pm, "_probe_current_state", lambda *_a, **_k: dict(cached),
        )
        prior_fp = power_variant_fingerprint({
            "power_cap_w": 300, "perflevel": "high",
            "sclk_idx": None, "mclk_idx": None, "pcie_idx": None,
            "perf_deterministic_mhz": None, "fan_pct": None, "devices": [],
        })
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "perf_h", "perflevel": "high"}],
            "base_tput":  100.0,
            "output_dir": str(tmp_path / "ws"),
            "host_state_applied": {
                "variant_name": "cap_300_high",
                "measured_state": cached,
            },
            "power_management_search": {
                "schema_version": 1,
                "accepted": [{
                    "name": "cap_300_high",
                    "power_settings": {
                        "power_cap_w": 300, "perflevel": "high",
                        "devices": [],
                    },
                    "fingerprint": prior_fp,
                }],
                "rejected": [], "tested": {}, "name_index": {},
                "cursor": 1,
            },
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        names_run = {r["variant_name"] for r in result["all_results"]}
        # Only the fresh variant ran. Prior winner was NOT revalidated.
        assert names_run == {"perf_h"}
        assert result["cache_stale"] is False
        assert result["revalidate_mode"] == "lazy"

    def test_lazy_revalidate_escalates_on_cache_drift(
        self, tmp_path, monkeypatch,
    ):
        # Live probe diverges from cache → lazy mode re-benches prior
        # winners on top of the fresh grid.
        self._stub_bench(monkeypatch, {
            "cap_300_high": 115.0,  # prior winner re-validated
            "perf_h":       120.0,  # fresh row
        })
        cached = {"powercap_w": 300, "perflevel": "high"}
        # Current probe shows a different cap — operator tweak / reboot.
        monkeypatch.setattr(
            pm, "_probe_current_state",
            lambda *_a, **_k: {"powercap_w": 350, "perflevel": "high"},
        )
        prior_fp = power_variant_fingerprint({
            "power_cap_w": 300, "perflevel": "high",
            "sclk_idx": None, "mclk_idx": None, "pcie_idx": None,
            "perf_deterministic_mhz": None, "fan_pct": None, "devices": [],
        })
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "perf_h", "perflevel": "high"}],
            "base_tput":  100.0,
            "output_dir": str(tmp_path / "ws"),
            "host_state_applied": {
                "variant_name": "cap_300_high",
                "measured_state": cached,
            },
            "power_management_search": {
                "schema_version": 1,
                "accepted": [{
                    "name": "cap_300_high",
                    "power_settings": {
                        "power_cap_w": 300, "perflevel": "high",
                        "devices": [],
                    },
                    "fingerprint": prior_fp,
                }],
                "rejected": [], "tested": {}, "name_index": {},
                "cursor": 1,
            },
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        names_run = [r["variant_name"] for r in result["all_results"]]
        # Prior winner ran first (re-validation), then the fresh row.
        assert names_run == ["cap_300_high", "perf_h"]
        assert result["cache_stale"] is True

    def test_settle_baseline_measured_before_winner_apply(
        self, tmp_path, monkeypatch,
    ):
        # Settle sweep (measure_kernel_baseline=True): the per-variant
        # loop leaves the host at vendor defaults, then the executor
        # measures the kernel-only baseline (median of reps) and ONLY
        # THEN applies the winner once so its state carries into SWEEP.
        self._stub_bench(monkeypatch, {"perf_h": 120.0})
        # Count rocm-smi variant applies. _run_one_variant is stubbed,
        # so it never calls _apply_variant — only the single winner
        # re-apply does (the baseline bench is stubbed below).
        apply_calls = {"n": 0}
        monkeypatch.setattr(pm, "_apply_variant", lambda *_a, **_k: (
            apply_calls.__setitem__("n", apply_calls["n"] + 1)
            or ["rocm-smi --setperflevel high --autorespond yes"]
        ))
        baseline_calls = {"n": 0}

        async def _fake_baseline(self_, **_k):
            baseline_calls["n"] += 1
            # New contract: (median, reps).
            return 90.0, 3

        monkeypatch.setattr(
            PowerManagementExecutor, "_measure_kernel_only_baseline",
            _fake_baseline, raising=True,
        )
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "perf_h", "perflevel": "high"}],
            "base_tput":  100.0,
            "output_dir": str(tmp_path / "ws"),
            "measure_kernel_baseline": True,   # settle-sweep marker
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        # Winner selection unchanged: still measured against base_tput.
        assert result["base_tput"] == 100.0
        assert result["host_state_applied"]["variant_name"] == "perf_h"
        # Vendor-default kernel-only baseline measured exactly once and
        # reported (Coordinator turns it into the FULL power delta).
        assert baseline_calls["n"] == 1
        assert result["kernel_baseline_tput"] == 90.0
        assert result["kernel_baseline_reps"] == 3
        # Winner applied exactly once (after the always-on baseline).
        assert apply_calls["n"] == 1

    def test_settle_baseline_measured_even_when_no_winner(
        self, tmp_path, monkeypatch,
    ):
        # No fresh winner (variant at 0% gain). The vendor-default
        # baseline is STILL measured (always-on) so the keep-incumbent
        # decision has a defaults reference. With no incumbent in params
        # the host strips to defaults.
        self._stub_bench(monkeypatch, {"perf_h": 100.0})  # 0% gain, no win
        baseline_calls = {"n": 0}

        async def _fake_baseline(self_, **_k):
            baseline_calls["n"] += 1
            return 90.0, 3

        monkeypatch.setattr(
            PowerManagementExecutor, "_measure_kernel_only_baseline",
            _fake_baseline, raising=True,
        )
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "perf_h", "perflevel": "high"}],
            "base_tput":  100.0,
            "output_dir": str(tmp_path / "ws"),
            "measure_kernel_baseline": True,
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        # No winner and no incumbent → reset to vendor defaults.
        assert result["host_state_applied"] is None
        # Baseline is always-on now, so it ran even without a winner.
        assert baseline_calls["n"] == 1
        assert result["kernel_baseline_tput"] == 90.0

    def test_always_revalidate_unconditional(self, tmp_path, monkeypatch):
        # ``revalidate_winners='always'`` re-benches prior winners
        # regardless of cache state.
        self._stub_bench(monkeypatch, {
            "cap_300_high": 115.0,
            "perf_h":       120.0,
        })
        # Cache is fresh — but always mode ignores that.
        monkeypatch.setattr(
            pm, "_probe_current_state",
            lambda *_a, **_k: {"powercap_w": 300, "perflevel": "high"},
        )
        prior_fp = power_variant_fingerprint({
            "power_cap_w": 300, "perflevel": "high",
            "sclk_idx": None, "mclk_idx": None, "pcie_idx": None,
            "perf_deterministic_mhz": None, "fan_pct": None, "devices": [],
        })
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "perf_h", "perflevel": "high"}],
            "base_tput":  100.0,
            "output_dir": str(tmp_path / "ws"),
            "revalidate_winners": "always",
            "host_state_applied": {
                "variant_name": "cap_300_high",
                "measured_state": {"powercap_w": 300, "perflevel": "high"},
            },
            "power_management_search": {
                "schema_version": 1,
                "accepted": [{
                    "name": "cap_300_high",
                    "power_settings": {
                        "power_cap_w": 300, "perflevel": "high",
                        "devices": [],
                    },
                    "fingerprint": prior_fp,
                }],
                "rejected": [], "tested": {}, "name_index": {},
                "cursor": 1,
            },
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        names_run = [r["variant_name"] for r in result["all_results"]]
        assert "cap_300_high" in names_run
        assert "perf_h" in names_run
        assert result["revalidate_mode"] == "always"

    def test_never_revalidate_skips_even_on_drift(self, tmp_path, monkeypatch):
        # ``revalidate_winners='never'`` skips re-validation even when
        # the cache is stale. Useful for callers that care about
        # speed over drift detection.
        self._stub_bench(monkeypatch, {
            "perf_h": 120.0,
        })
        monkeypatch.setattr(
            pm, "_probe_current_state",
            lambda *_a, **_k: {"powercap_w": 999, "perflevel": "low"},
        )
        prior_fp = power_variant_fingerprint({
            "power_cap_w": 300, "perflevel": "high",
            "sclk_idx": None, "mclk_idx": None, "pcie_idx": None,
            "perf_deterministic_mhz": None, "fan_pct": None, "devices": [],
        })
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "perf_h", "perflevel": "high"}],
            "base_tput":  100.0,
            "output_dir": str(tmp_path / "ws"),
            "revalidate_winners": "never",
            "host_state_applied": {
                "variant_name": "cap_300_high",
                "measured_state": {"powercap_w": 300, "perflevel": "high"},
            },
            "power_management_search": {
                "schema_version": 1,
                "accepted": [{
                    "name": "cap_300_high",
                    "power_settings": {
                        "power_cap_w": 300, "perflevel": "high",
                        "devices": [],
                    },
                    "fingerprint": prior_fp,
                }],
                "rejected": [], "tested": {}, "name_index": {},
                "cursor": 1,
            },
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        names_run = {r["variant_name"] for r in result["all_results"]}
        # Prior winner NOT in results despite drift.
        assert names_run == {"perf_h"}

    def test_bad_revalidate_mode_rejected(self, tmp_path, monkeypatch):
        monkeypatch.setattr(pm, "_which_rocm_smi", lambda: True)
        monkeypatch.setattr(
            pm, "_probe_powercap_range", lambda *_a, **_k: (200, 400),
        )
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "x", "power_cap_w": 300}],
            "revalidate_winners": "warp_speed",
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "failed"
        assert result["error_class"] == "bad_param"

    def test_host_state_applied_snapshot_on_winner(
        self, tmp_path, monkeypatch,
    ):
        # When a variant wins AND the executor re-applies it, the
        # result payload must include a host_state_applied snapshot
        # carrying the rocm-smi commands + measured state — the
        # Coordinator persists this into SharedState for the final
        # report's host-state block.
        self._stub_bench(monkeypatch, {
            "cap_300": 115.0,
        })
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "cap_300", "power_cap_w": 300}],
            "base_tput":  100.0,
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["final_state"] == "applied_best"
        snap = result["host_state_applied"]
        assert snap is not None
        assert snap["variant_name"] == "cap_300"
        assert snap["power_settings"]["power_cap_w"] == 300
        # Captures the exact rocm-smi commands from _apply_variant.
        assert any("setpoweroverdrive" in c for c in snap["smi_commands"])
        # Probed bounds at apply time are stamped for replay context.
        assert snap["probed_range_w"] == [200, 400]
        # ts is iso8601 (best-effort sanity check on format).
        assert "T" in snap["ts"]

    def test_host_state_applied_none_when_no_winner(
        self, tmp_path, monkeypatch,
    ):
        # No winners → no re-apply → host_state_applied is None so the
        # Coordinator clears any stale cache. The earlier reset_defaults
        # already restored the GPU; the field reflects that.
        self._stub_bench(monkeypatch, {
            "perf_h": 100.5,  # below the noise floor
        })
        ex = PowerManagementExecutor(session_dir=tmp_path)
        params = {
            "grid": [{"name": "perf_h", "perflevel": "high"}],
            "base_tput":  100.0,
            "output_dir": str(tmp_path / "ws"),
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        assert result["status"] == "no_winners"
        assert result["host_state_applied"] is None
        assert result["final_state"] == "reset_to_default"

    def test_prior_winner_below_lifted_floor_skipped(
        self, tmp_path, monkeypatch,
    ):
        # A prior winner whose cap is BELOW the currently-effective
        # floor (e.g. the probe just lifted it after a driver update)
        # must be silently dropped from the re-validation list rather
        # than crashing the executor with a cap-bounds error.
        self._stub_bench(monkeypatch, {
            "perf_h": 115.0,
        })
        # Drift → escalates lazy to revalidate, but our prior winner
        # cap (250W) is below the post-lift floor.
        monkeypatch.setattr(
            pm, "_probe_current_state",
            lambda *_a, **_k: {"powercap_w": 350, "perflevel": "auto"},
        )
        prior_fp = power_variant_fingerprint({
            "power_cap_w": 250, "perflevel": None,
            "sclk_idx": None, "mclk_idx": None, "pcie_idx": None,
            "perf_deterministic_mhz": None, "fan_pct": None, "devices": [],
        })
        ex = PowerManagementExecutor(
            session_dir=tmp_path,
            default_power_cap_floor_w=300,  # lifted floor
        )
        params = {
            "grid": [{"name": "perf_h", "perflevel": "high"}],
            "base_tput":  100.0,
            "output_dir": str(tmp_path / "ws"),
            "host_state_applied": {
                "variant_name": "cap_250",
                "measured_state": {"powercap_w": 300, "perflevel": "high"},
            },
            "power_management_search": {
                "schema_version": 1,
                "accepted": [{
                    "name": "cap_250",
                    "power_settings": {"power_cap_w": 250, "devices": []},
                    "fingerprint": prior_fp,
                }],
                "rejected": [], "tested": {}, "name_index": {},
                "cursor": 1,
            },
        }
        result = asyncio.run(ex(_Ctx(task=_task(params))))
        names_run = {r["variant_name"] for r in result["all_results"]}
        # The prior winner was skipped; only perf_h ran.
        assert names_run == {"perf_h"}
