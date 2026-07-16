# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit coverage for the scriptable/diffusion (xDiT) latency-domain surfacing:

* ``framework_registry.primary_metric_name`` (which field is the result).
* ``collect_roofline_progress`` independent latency ceiling + ``ceiling_kind``.
* ``_normalize_roofline_snapshot`` preserving the latency siblings.
* recorder ``_snapshot_final`` emitting e2el / unit / primary_metric.
* ``SharedState._backfill_scriptable_latency`` deriving e2el from tput.
"""

from __future__ import annotations

from hyperloom.inference_optimizer import framework_registry as fr
from hyperloom.inference_optimizer.breakdown.collectors import roofline as col
from hyperloom.inference_optimizer.breakdown.recorder import instrument as inst


class TestPrimaryMetricName:
    def test_scriptable_uses_e2el(self):
        assert fr.primary_metric_name("xdit") == "e2el_mean_ms"

    def test_serving_uses_throughput(self):
        assert fr.primary_metric_name("sglang") == "throughput_tok_s_per_gpu"
        assert fr.primary_metric_name(None) == "throughput_tok_s_per_gpu"


class TestNormalizeRooflineSnapshotLatencySiblings:
    def test_preserves_latency_and_ceiling_fields(self):
        snap = col._normalize_roofline_snapshot(
            {
                "snapshot_id": 2,
                "ts": "t",
                "e2e_mean_ms": 910.0,
                "roofline_ideal_ms": 360.4,
                "roofline_bound_kind": "compute",
            }
        )
        assert snap["e2e_mean_ms"] == 910.0
        assert snap["roofline_ideal_ms"] == 360.4
        assert snap["roofline_bound_kind"] == "compute"

    def test_missing_latency_fields_are_none(self):
        snap = col._normalize_roofline_snapshot({"snapshot_id": 1, "ts": "t"})
        assert snap["e2e_mean_ms"] is None
        assert snap["roofline_ideal_ms"] is None
        assert snap["roofline_bound_kind"] == "unknown"


class TestCollectRooflineProgressLatencyCeiling:
    def _state(self, snap: dict) -> dict:
        return {
            "framework": "xdit",
            "baseline_tput": 0.168919,
            "cumulative_gain": 550.5,
            "optimization_stack": [
                {"ts": "2026-01-01T00:00:00", "tput": 1.098901, "variant_name": "v", "action": "explore"}
            ],
            "roofline_snapshots": [snap],
        }

    def test_latency_ceiling_surfaced_when_no_tok_s_ceiling(self, tmp_path):
        snap = {
            "snapshot_id": 2,
            "ts": "2026-01-01T00:10:00",
            "theoretical_peak_tok_per_sec": 0.0,
            "e2e_mean_ms": 910.0,
            "roofline_ideal_ms": 360.4,
        }
        out = col.collect_roofline_progress(tmp_path, self._state(snap), {}, [])
        assert out["ceiling_available"] is False  # tok/s side stays off
        assert out["ceiling_kind"] == "latency"
        assert out["latency_ceiling_available"] is True
        assert out["latency_ceiling_ms"] == 360.4
        assert out["achieved_latency_ms"] == 910.0
        assert out["current_best_pct_of_latency_ceiling"] == round(360.4 / 910.0 * 100.0, 4)

    def test_throughput_ceiling_takes_precedence(self, tmp_path):
        snap = {
            "snapshot_id": 1,
            "ts": "2026-01-01T00:10:00",
            "theoretical_peak_tok_per_sec": 2000.0,
        }
        st = self._state(snap)
        st["framework"] = "sglang"
        out = col.collect_roofline_progress(tmp_path, st, {}, [])
        assert out["ceiling_kind"] == "throughput"
        assert out["ceiling_available"] is True
        assert out["latency_ceiling_available"] is False
        assert out["latency_ceiling_ms"] is None

    def test_no_ceiling_when_latency_partial(self, tmp_path):
        # Only e2e present (ideal floor missing) -> no latency ceiling.
        snap = {
            "snapshot_id": 2,
            "ts": "2026-01-01T00:10:00",
            "theoretical_peak_tok_per_sec": 0.0,
            "e2e_mean_ms": 910.0,
            "roofline_ideal_ms": 0.0,
        }
        out = col.collect_roofline_progress(tmp_path, self._state(snap), {}, [])
        assert out["ceiling_kind"] == "none"
        assert out["latency_ceiling_available"] is False


class _Rec:
    def __init__(self):
        self.singletons: dict[str, dict] = {}

    def record_singleton(self, name, payload):
        self.singletons[name] = payload


class _St:
    def __init__(self, **kw):
        self.framework = kw.get("framework", "xdit")
        self.current_best = kw.get("current_best", {})
        self.optimization_stack = kw.get("optimization_stack", [])
        self.cumulative_gain_validated = kw.get("cumulative_gain_validated", 0.0)
        self.cumulative_gain = kw.get("cumulative_gain", 0.0)
        self.cumulative_gain_validated_ts = kw.get("cumulative_gain_validated_ts", "")


class TestSnapshotFinalEmitsLatency:
    def test_scriptable_final_derives_e2el_from_tput(self):
        rec = _Rec()
        st = _St(
            framework="xdit",
            current_best={"action": "explore", "tput": 1.098901},
            optimization_stack=[{"action": "explore"}],
        )
        inst._snapshot_final(rec, st)
        final = rec.singletons["final"]
        assert final["throughput_unit"] == "img/s"
        assert final["primary_metric"] == "e2el_mean_ms"
        # 1000 / 1.098901 ~= 910.0
        assert final["e2el_mean_ms"] == round(1000.0 / 1.098901, 4)

    def test_scriptable_final_prefers_measured_e2el(self):
        rec = _Rec()
        st = _St(
            framework="xdit",
            current_best={"action": "explore", "tput": 1.098901, "e2el_mean_ms": 980.0},
            optimization_stack=[{"action": "explore"}],
        )
        inst._snapshot_final(rec, st)
        assert rec.singletons["final"]["e2el_mean_ms"] == 980.0

    def test_serving_final_has_no_derived_e2el(self):
        rec = _Rec()
        st = _St(
            framework="sglang",
            current_best={"action": "grid", "tput": 123.4},
            optimization_stack=[{"action": "grid"}],
        )
        inst._snapshot_final(rec, st)
        final = rec.singletons["final"]
        assert final["throughput_unit"] == "tok/s"
        assert final["primary_metric"] == "throughput_tok_s_per_gpu"
        assert final["e2el_mean_ms"] is None


class TestBackfillScriptableLatency:
    def _state(self, framework: str, cb: dict):
        from hyperloom.orchestrator.state.shared_state import SharedState

        st = SharedState(session_id="s", model_name="m", model_path="/m")
        st.framework = framework
        st.current_best = cb
        return st

    def test_scriptable_backfills_e2el_from_tput(self):
        st = self._state("xdit", {"action": "explore", "tput": 1.098901})
        st._backfill_scriptable_latency()
        assert st.current_best["e2el_mean_ms"] == round(1000.0 / 1.098901, 4)

    def test_measured_e2el_not_overwritten(self):
        st = self._state("xdit", {"action": "explore", "tput": 1.098901, "e2el_mean_ms": 980.0})
        st._backfill_scriptable_latency()
        assert st.current_best["e2el_mean_ms"] == 980.0

    def test_serving_is_noop(self):
        st = self._state("sglang", {"action": "grid", "tput": 123.4})
        st._backfill_scriptable_latency()
        assert "e2el_mean_ms" not in st.current_best

    def test_non_positive_tput_is_noop(self):
        st = self._state("xdit", {"action": "explore", "tput": 0.0})
        st._backfill_scriptable_latency()
        assert st.current_best.get("e2el_mean_ms") is None
