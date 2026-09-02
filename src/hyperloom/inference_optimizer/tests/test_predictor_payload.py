# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for ``orchestrator.predictor.payload``.

The interesting cases are the ones that fail silently in production: a stack
row that reports accumulated instead of own args, a roofline block that loses
its optional perf-model half, and an evidence sub-block that ships half filled.
"""

from __future__ import annotations

import datetime as dt
import json
from types import SimpleNamespace

import pytest

from hyperloom.orchestrator.predictor import payload as pl


def _state(**overrides):
    """A SharedState stand-in carrying only what the builder reads."""
    base = dict(
        model_name="Qwen-Qwen3-8B",
        model_class="dense",
        gpu_type="mi300x",
        framework="vllm",
        framework_version="0.22.0",
        precision="fp8",
        tp=4,
        ep=1,
        nodes=1,
        model_info={"model_type": "qwen3", "attention_type": "gqa", "num_hidden_layers": 36},
        isl=8192,
        osl=1024,
        conc=64,
        max_model_len=13312,
        phase="FRAMEWORK_AGENT",
        phase_history=[{"reason": "prelude_done"}],
        macro_cycle=0,
        baseline_tput=1820.4,
        current_best={"tput": 1901.7},
        cumulative_gain_validated=4.46,
        optimization_stack=[],
        last_trace_analyze={},
        roofline_snapshots=[],
        phase_started_unix=0.0,
    )
    base.update(overrides)
    return SimpleNamespace(**base)


class TestRequestEnvelope:
    def test_is_json_serialisable(self):
        json.dumps(pl.build_request(_state()))

    def test_carries_the_schema_and_session_id(self):
        req = pl.build_request(_state(), session_id="sess-1")
        assert req["schema"] == pl.REQUEST_SCHEMA
        assert req["session_id"] == "sess-1"

    def test_forwards_identification_and_workload_unrenamed(self):
        req = pl.build_request(_state())
        assert req["identification"]["gpu_type"] == "mi300x"
        assert req["identification"]["model_info"]["num_hidden_layers"] == 36
        # Hyperloom's own spelling: conc, not concurrency.
        assert req["workload"]["conc"] == 64
        assert "concurrency" not in req["workload"]


class TestPhase:
    def test_defaults_to_the_explore_label_not_the_live_phase(self):
        req = pl.build_request(_state())
        assert req["phase"]["phase"] == "EXPLORE"

    def test_label_is_overridable(self):
        req = pl.build_request(_state(), phase_label="FRAMEWORK_AGENT")
        assert req["phase"]["phase"] == "FRAMEWORK_AGENT"

    def test_reason_comes_from_the_newest_transition(self):
        state = _state(phase_history=[{"reason": "prelude_done"}, {"reason": "cycle_reloop"}])
        assert pl.build_request(state)["phase"]["phase_reason"] == "cycle_reloop"

    def test_empty_history_yields_none(self):
        assert pl.build_request(_state(phase_history=[]))["phase"]["phase_reason"] is None


class TestPerformance:
    def test_keep_threshold_is_resolved_not_hardcoded(self):
        """The bar decays with the macro-cycle; 1.0 is only cycle 1."""
        assert pl.build_request(_state(macro_cycle=0))["performance"]["keep_threshold_pct"] == pytest.approx(1.0)
        assert pl.build_request(_state(macro_cycle=1))["performance"]["keep_threshold_pct"] == pytest.approx(0.55)

    def test_stack_reports_own_args_not_the_accumulation(self):
        """``extra_server_args`` accumulates; sending it repeats every prior flag."""
        state = _state(
            optimization_stack=[
                {
                    "candidate_extra_server_args": "--flag-a",
                    "extra_server_args": "--flag-a",
                    "extra_envs": {"A": "1"},
                    "tput": 1900.0,
                },
                {
                    "candidate_extra_server_args": "--flag-b",
                    "extra_server_args": "--flag-a --flag-b",
                    "extra_envs": {"A": "1", "B": "2"},
                    "tput": 1950.0,
                },
            ]
        )
        stack = pl.build_request(state)["performance"]["optimization_stack"]
        assert [s["candidate_extra_server_args"] for s in stack] == ["--flag-a", "--flag-b"]
        assert stack[1]["tput"] == pytest.approx(1950.0)

    def test_current_best_tput_is_the_grading_anchor(self):
        assert pl.build_request(_state())["performance"]["current_best_tput"] == pytest.approx(1901.7)

    def test_absent_measurements_are_none_not_zero(self):
        state = _state(baseline_tput=0.0, current_best={})
        perf = pl.build_request(state)["performance"]
        assert perf["baseline_tput"] is None
        assert perf["current_best_tput"] is None


class TestEvidenceAvailability:
    def test_no_trace_reports_absence_once(self):
        ev = pl.build_request(_state())["evidence"]
        assert ev == {"profile_available": False}

    def test_hot_kernels_alone_count_as_a_profile(self):
        """A trace whose quality gate withheld analysis.md still locates time."""
        state = _state(last_trace_analyze={"hot_kernels_top15": [{"name": "k", "gpu_pct": 5.0}]})
        ev = pl.build_request(state)["evidence"]
        assert ev["profile_available"] is True
        assert ev["hot_kernels"][0]["name"] == "k"

    def test_incomplete_sub_blocks_are_omitted_not_half_filled(self):
        state = _state(last_trace_analyze={"analysis_md_text": "# report with no tables"})
        ev = pl.build_request(state)["evidence"]
        assert ev["profile_available"] is True
        assert "window" not in ev
        assert "operators" not in ev
        assert "roofline" not in ev


class TestRoofline:
    def _snapshot(self, **overrides):
        base = {
            "roofline_mem_ceiling_tok_per_sec": 4210.0,
            "roofline_cmp_ceiling_tok_per_sec": 9880.0,
            "roofline_bound_kind": "memory",
            "achieved_tok_per_sec": 1901.7,
            "gap_to_roofline_pct": 54.8,
        }
        base.update(overrides)
        return base

    def _build(self, snapshot):
        state = _state(
            last_trace_analyze={"hot_kernels_top15": [{"name": "k"}]},
            roofline_snapshots=[snapshot],
        )
        return pl.build_request(state)["evidence"]["roofline"]

    def test_reads_the_newest_snapshot(self):
        block = self._build(self._snapshot())
        assert block["roofline_mem_ceiling_tok_per_sec"] == pytest.approx(4210.0)
        assert block["roofline_bound_kind"] == "memory"

    def test_unknown_bound_kind_becomes_none(self):
        """A label the consumer has no rule for is worse than an absent one."""
        assert self._build(self._snapshot(roofline_bound_kind="unknown"))["roofline_bound_kind"] is None

    def test_perfmodel_fields_are_optional(self):
        """attach_perfmodel_breakdown is best-effort; its absence is a degrade."""
        block = self._build(self._snapshot())
        assert block["hbm_bw_gbps"] is None
        assert block["n_ops_total"] is None

    def test_counts_memory_bound_operators_when_perfmodel_is_present(self):
        snapshot = self._snapshot(
            perfmodel_breakdown={
                "hbm_bw_gbps": 5300.0,
                "peak_achievable_tflops": 1307.0,
                "ops": [{"bound": "memory"}, {"bound": "memory"}, {"bound": "compute"}],
            }
        )
        block = self._build(snapshot)
        assert block["n_ops_total"] == 3
        assert block["n_ops_memory_bound"] == 2
        assert block["hbm_bw_gbps"] == pytest.approx(5300.0)

    def test_block_drops_when_both_ceilings_are_missing(self):
        state = _state(
            last_trace_analyze={"hot_kernels_top15": [{"name": "k"}]},
            roofline_snapshots=[{"roofline_bound_kind": "memory"}],
        )
        assert "roofline" not in pl.build_request(state)["evidence"]


class TestProfileAge:
    def test_derives_seconds_from_the_iso_stamp(self):
        stamped = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=300)
        state = _state(
            last_trace_analyze={"hot_kernels_top15": [{"name": "k"}], "ts": stamped.isoformat()}
        )
        age = pl.build_request(state)["evidence"]["profile_age_sec"]
        assert 295 <= age <= 320

    def test_unparseable_stamp_is_omitted(self):
        state = _state(last_trace_analyze={"hot_kernels_top15": [{"name": "k"}], "ts": "not-a-date"})
        assert "profile_age_sec" not in pl.build_request(state)["evidence"]


class TestHotKernels:
    def test_caps_at_the_shared_top_n(self):
        rows = [{"name": f"k{i}", "gpu_pct": 1.0} for i in range(15)]
        state = _state(last_trace_analyze={"hot_kernels_top15": rows})
        kernels = pl.build_request(state)["evidence"]["hot_kernels"]
        assert len(kernels) == pl.HOT_KERNEL_TOP_N == 8

    def test_projects_hyperloom_spellings(self):
        rows = [
            {
                "name": "torch_gemm",
                "gpu_pct": 14.1,
                "efficiency_percent": 61.0,
                "arithmetic_intensity": 118.4,
                "bound_type": "compute",
                "kernel_category": "GEMM",
                "source_file": "tuned_gemm.py",
            }
        ]
        kernel = pl.build_request(_state(last_trace_analyze={"hot_kernels_top15": rows}))["evidence"][
            "hot_kernels"
        ][0]
        # Not op / efficiency_pct / category.
        assert kernel["name"] == "torch_gemm"
        assert kernel["efficiency_percent"] == pytest.approx(61.0)
        assert kernel["kernel_category"] == "GEMM"
        # Absent frame is explicit, not missing.
        assert kernel["source_line"] is None
        assert kernel["source_function"] is None

    def test_args_and_count_come_from_the_p_item_tables(self):
        analysis_md = "\n".join(
            [
                "## Executive Summary",
                "",
                "| Metric | Value |",
                "|--------|-------|",
                "| Total GPU Time | 100.000 ms |",
                "",
                "### P1: GEMM kernels",
                "",
                "**Data:**",
                "",
                pl.ev.P_ITEM_COLUMNS,
                "|---|---|---|---|---|---|---|---|---|---|---|",
                "| torch_gemm | 38.2 | 14.10% | 9.10 | 1440 | 118.400 | 61.00% "
                "| compute | 16x4096 bf16 | tuned_gemm.py | vllm/ops |",
            ]
        )
        state = _state(
            last_trace_analyze={
                "hot_kernels_top15": [{"name": "torch_gemm", "gpu_pct": 14.1}],
                "analysis_md_text": analysis_md,
            }
        )
        kernel = pl.build_request(state)["evidence"]["hot_kernels"][0]
        assert kernel["args"] == "16x4096 bf16"
        assert kernel["call_count"] == 1440
        assert kernel["time_us"] == pytest.approx(38.2)
