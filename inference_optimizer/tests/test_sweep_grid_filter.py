# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the sweep grid builder's max-model-len filter (drop ISL+OSL > max-model-len up front)."""

from __future__ import annotations

from inference_optimizer.orchestrator.action_executors import sweep


class TestBuildGridMaxModelLenFilter:
    def test_no_filter_when_max_model_len_zero(self):
        runnable, skipped = sweep._build_grid(
            conc_values=[4],
            isl_osl_configs=["1024:1024", "8192:1024", "1024:8192"],
            num_prompts_factor=5,
            base_extra_args="",
            max_model_len=0,
        )
        assert len(runnable) == 3
        assert skipped == []

    def test_filters_combos_exceeding_context_window(self):
        # max_model_len=6144 — the value that made every 8192-token variant abort.
        runnable, skipped = sweep._build_grid(
            conc_values=[4, 16],
            isl_osl_configs=["1024:1024", "8192:1024", "1024:8192"],
            num_prompts_factor=5,
            base_extra_args="",
            max_model_len=6144,
        )
        # Only 1024:1024 (sum 2048 <= 6144) survives, once per conc.
        assert len(runnable) == 2
        assert all(v.extra_envs["ISL"] == "1024" for v in runnable)
        assert all(v.extra_envs["OSL"] == "1024" for v in runnable)
        # 8192:1024 and 1024:8192 dropped for each of the 2 conc values.
        assert len(skipped) == 4
        for rec in skipped:
            assert rec["status"] == "skipped"
            assert "max_model_len=6144" in rec["skip_reason"]
            assert rec["isl"] + rec["osl"] > 6144

    def test_boundary_sum_equal_to_max_is_kept(self):
        # isl+osl == max_model_len must be allowed (the request fits).
        runnable, skipped = sweep._build_grid(
            conc_values=[4],
            isl_osl_configs=["3072:3072"],
            num_prompts_factor=5,
            base_extra_args="",
            max_model_len=6144,
        )
        assert len(runnable) == 1
        assert skipped == []

    def test_malformed_isl_osl_still_skipped_silently(self):
        runnable, skipped = sweep._build_grid(
            conc_values=[4],
            isl_osl_configs=["not:a:number:x", "1024:1024"],
            num_prompts_factor=5,
            base_extra_args="",
            max_model_len=6144,
        )
        # Malformed combo is dropped (not counted as a context-window skip).
        assert len(runnable) == 1
        assert skipped == []


class TestCoerceInt:
    def test_numeric_string(self):
        assert sweep._coerce_int("6144") == 6144

    def test_int_passthrough(self):
        assert sweep._coerce_int(4096) == 4096

    def test_none_and_empty(self):
        assert sweep._coerce_int(None) == 0
        assert sweep._coerce_int("") == 0

    def test_garbage_is_zero(self):
        assert sweep._coerce_int("abc") == 0
