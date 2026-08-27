# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for stage 1 (diagnose): categorization + launch-bound verdict."""

from __future__ import annotations

import gzip
import json

from kernel_agents.fusion import diagnose as dg


class TestCategorizeKernelName:
    def test_compute_bound_categories(self):
        assert dg.categorize_kernel_name("Cijk_Alik_Bljk_MT64x16x256") == "gemm"
        assert dg.categorize_kernel_name("void paged_attention_ll4mi_QKV_mfma16") == "attention"
        assert dg.categorize_kernel_name("_causal_conv1d_update_kernel") == "conv"
        assert dg.categorize_kernel_name("fused_moe_kernel") == "moe"

    def test_launch_bound_categories(self):
        # rmsnorm must win over the "add" substring in a fused add+rmsnorm kernel.
        assert dg.categorize_kernel_name("_ZN5aiter24add_rmsnorm_quant_kernel") == "rmsnorm"
        assert dg.categorize_kernel_name("void rotary_embedding_kernel<bf16>") == "rope"
        assert dg.categorize_kernel_name("vectorized_elementwise silu_kernel") == "activation"
        # cast must win over the "copy" substring in a dtype-conversion copy.
        assert dg.categorize_kernel_name("vectorized_elementwise bfloat16tofloat32_copy") == "cast"
        assert dg.categorize_kernel_name("store_kvcache<1024l>") == "copy"
        assert dg.categorize_kernel_name("vectorized_elementwise CUDAFunctor_add<bf16>") == "add"


class TestDiagnoseFromShares:
    def test_launch_bound_is_candidate(self):
        shares = {"gemm": 0.5, "add": 0.14, "rmsnorm": 0.08, "activation": 0.05, "rope": 0.02}
        d = dg.diagnose_from_shares(shares, busy_fraction_of_wall=0.21)
        assert d.is_candidate
        assert d.launch_bound_share >= 0.25
        assert d.dominant_categories[0] == "add"

    def test_compute_bound_is_annotated_not_vetoed(self):
        """A high GPU-busy fraction no longer rejects the model.

        The busy-of-wall heuristic was calibrated on 5 models, but measured
        counter-examples exist: GEMM-bound Qwen3-14B/32B still gained +6.2% and
        +3.1% end to end from decode fusions. Busy-of-wall is therefore reported
        for ranking and kept visible in the reason, while the downstream
        validate/loop remains the real filter.
        """
        shares = {"gemm": 0.55, "add": 0.25, "rmsnorm": 0.20}
        d = dg.diagnose_from_shares(shares, busy_fraction_of_wall=0.72)
        assert d.is_candidate
        assert "busy" in d.reason
        assert d.busy_fraction_of_wall == 0.72

    def test_below_share_threshold_not_candidate(self):
        # Below the soft launch-bound FLOOR (0.10): almost nothing fusible present.
        shares = {"gemm": 0.72, "attention": 0.20, "rmsnorm": 0.05, "activation": 0.03}
        d = dg.diagnose_from_shares(shares, busy_fraction_of_wall=0.40)
        assert not d.is_candidate
        assert "launch_bound_share" in d.reason

    def test_low_share_but_gpu_idle_is_candidate(self):
        # Calibration regression: GraniteMoE-like case -- LOW launch-bound share
        # (big MoE GEMMs dilute it) but the GPU is mostly idle (dispatch-bound), so
        # it MUST be a candidate. The old share>=0.25 gate wrongly rejected this.
        shares = {"gemm": 0.55, "moe": 0.27, "add": 0.10, "rmsnorm": 0.05, "rope": 0.03}
        d = dg.diagnose_from_shares(shares, busy_fraction_of_wall=0.29)
        assert d.is_candidate
        assert d.launch_bound_share < 0.25

    def test_empty_shares(self):
        d = dg.diagnose_from_shares({}, busy_fraction_of_wall=None)
        assert not d.is_candidate and d.reason == "empty_trace"


class TestLoadTrace:
    @staticmethod
    def _write(path, events, gz=False):
        payload = {"traceEvents": events}
        if gz:
            with gzip.open(path, "wt", encoding="utf-8") as fh:
                json.dump(payload, fh)
        else:
            path.write_text(json.dumps(payload), encoding="utf-8")

    def test_launch_bound_trace_roundtrip(self, tmp_path):
        events = [
            {"cat": "kernel", "name": "Cijk_gemm", "ts": 0, "dur": 40},
            {"cat": "kernel", "name": "add_rmsnorm_quant_kernel", "ts": 100, "dur": 10},
            {"cat": "kernel", "name": "vectorized_elementwise silu", "ts": 200, "dur": 8},
            {"cat": "kernel", "name": "rotary_embedding_kernel", "ts": 300, "dur": 6},
            {"cat": "kernel", "name": "vectorized_elementwise CUDAFunctor_add", "ts": 400, "dur": 6},
            {"cat": "cpu_op", "name": "aten::add", "ts": 0, "dur": 999},  # ignored
        ]
        p = tmp_path / "decode.trace.json"
        self._write(p, events)
        d = dg.diagnose_trace(p, decode_steps=1)
        assert d.is_candidate
        assert d.kernels_per_step == 5
        assert d.category_shares["gemm"] == 40 / 70

    def test_gzip_supported(self, tmp_path):
        p = tmp_path / "d.trace.json.gz"
        self._write(p, [{"cat": "kernel", "name": "rotary_embedding_kernel", "ts": 0, "dur": 5}], gz=True)
        shares, busy, n = dg.load_op_busy_from_kineto_trace(p)
        assert shares == {"rope": 1.0} and n == 1.0

    def test_missing_file_distinct_reason(self, tmp_path):
        # A missing trace must be distinguishable from a present-but-not-fusible
        # trace: reason is trace_unreadable, not empty_trace / launch_bound_share.
        d = dg.diagnose_trace(tmp_path / "nope.json")
        assert not d.is_candidate
        assert d.reason.startswith("trace_unreadable")

    def test_present_but_no_kernels_is_empty_trace(self, tmp_path):
        p = tmp_path / "d.trace.json"
        self._write(p, [{"cat": "cpu_op", "name": "aten::add", "ts": 0, "dur": 5}])
        d = dg.diagnose_trace(p)
        assert not d.is_candidate
        assert d.reason == "empty_trace"

    def test_mul_not_miscounted_as_add(self):
        # A BinaryFunctor multiply must land in ``mul``, not ``add``.
        assert dg.categorize_kernel_name("vectorized_elementwise BinaryFunctor mul<bf16>") == "mul"
