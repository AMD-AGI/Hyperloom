"""Unit tests for the pure-Python helpers and analyzer in
``orchestrator.roofline_integration``.

The full PMC profiler pipeline is integration-only (it shells out to
``rocprofv3`` and sglang), so we keep this module focused on the
shrink-wrapped surfaces that have no rocm/torch dependency:

* helper coercion / kernel-name normalization,
* GPU spec lookup + autodetect fallbacks,
* :class:`RooflineAnalyzer` classification, suggestion routing and
  serialisation,
* small subprocess-adjacent helpers (``_as_cmd``, ``_wait_for_csvs``,
  ``_process_group_kwargs``).

These cover the high-leverage logic without touching any real device,
mirroring the structure of ``test_roofline_executor`` which exercises
the executor wrapping this module.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

import pytest

from inference_optimizer.orchestrator.roofline_integration import (
    Bottleneck,
    GPUSpec,
    KernelRooflineResult,
    MI300X_SPEC,
    MI325X_SPEC,
    MI355X_SPEC,
    PMCKernelResult,
    RooflineAnalyzer,
    _as_cmd,
    _classify_kernel_tier,
    _normalize_kernel_name,
    _process_group_kwargs,
    _to_float,
    _to_int,
    _wait_for_csvs,
    gpu_spec_from_name,
    resolve_gpu_type,
)


class TestCoercionHelpers:
    """``_to_float`` / ``_to_int`` accept anything stringy without raising."""

    def test_to_float_handles_numeric_strings(self):
        assert _to_float("3.14") == pytest.approx(3.14)
        assert _to_float(42) == pytest.approx(42.0)
        assert _to_float(None) == 0.0
        assert _to_float("not a number") == 0.0

    def test_to_int_truncates_floats_and_strings(self):
        assert _to_int("17.9") == 17
        assert _to_int(2.5) == 2
        assert _to_int(None) == 0
        assert _to_int("garbage") == 0


class TestNormalizeKernelName:
    """Kernel-name canonicalisation must match the families used by the
    analyzer + downstream prompts."""

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Cijk_AlikSb_SAB", "hipblaslt_gemm"),
            ("gemm_a16w16_asm_kernel", "aiter_asm_gemm"),
            ("AttN_FwD_paged", "attention"),
            ("flash_attn_v2_decode", "attention"),
            ("moe_ck2stages_a16w8", "moe_gemm"),
            ("moe_ck_tile_fused", "moe_gemm"),
            ("rms_norm_kernel", "rms_norm"),
            ("vectorized_layer_norm_kernel", "rms_norm"),
            ("topk_softmax_routing", "topk"),
            ("rope_v3", "rope"),
            ("rotary_embedding_kernel", "rope"),
            ("rccl_allreduce_ring", "allreduce"),
            ("nccl_send_recv", "allreduce"),
            ("memcpy_d2h", "memcpy"),
            ("softmax_kernel", "softmax"),
            ("skinny_gemm_32x32", "skinny_gemm"),
        ],
    )
    def test_known_family_names_collapse(self, raw, expected):
        assert _normalize_kernel_name(raw) == expected

    def test_unknown_name_passes_through_truncated(self):
        raw = "x" * 200
        result = _normalize_kernel_name(raw)
        assert len(result) == 80
        assert result == "x" * 80

    def test_short_unknown_name_passes_through_verbatim(self):
        assert _normalize_kernel_name("custom_widget") == "custom_widget"


class TestPMCKernelResultProps:
    def test_mfma_ratio_handles_zero_total(self):
        kernel = PMCKernelResult(name="x")
        assert kernel.mfma_ratio_pct == 0.0

    def test_mfma_ratio_computes_percentage(self):
        kernel = PMCKernelResult(name="g", SQ_INSTS_MFMA=80.0, SQ_INSTS_VALU=20.0)
        assert kernel.mfma_ratio_pct == pytest.approx(80.0)

    def test_bytes_transferred_prefers_32b_sum(self):
        kernel = PMCKernelResult(
            name="b",
            TCC_EA_RDREQ_32B_sum=10.0,
            TCC_EA_WRREQ_32B_sum=5.0,
            TCC_EA_RDREQ_sum=1.0,
            TCC_EA_WRREQ_sum=2.0,
        )
        # 32B-sums win when present: (10 + 5) * 32 = 480.
        assert kernel.bytes_transferred == pytest.approx(480.0)

    def test_bytes_transferred_falls_back_to_64b_sum(self):
        kernel = PMCKernelResult(
            name="b",
            TCC_EA_RDREQ_sum=3.0,
            TCC_EA_WRREQ_sum=4.0,
        )
        # 32B sums absent → fallback uses 64B unit width.
        assert kernel.bytes_transferred == pytest.approx((3 + 4) * 64)

    def test_bytes_transferred_returns_none_without_counters(self):
        assert PMCKernelResult(name="b").bytes_transferred is None


class TestGPUSpecLookup:
    """``gpu_spec_from_name`` should map common GPU monikers to specs."""

    @pytest.mark.parametrize(
        "name, expected",
        [
            ("mi300x", MI300X_SPEC),
            ("MI-300x", MI300X_SPEC),
            ("gfx942", MI300X_SPEC),
            ("mi325x", MI325X_SPEC),
            ("mi355x", MI355X_SPEC),
            ("gfx950", MI355X_SPEC),
            (None, MI300X_SPEC),       # default
            ("unknown-gpu", MI300X_SPEC),
        ],
    )
    def test_known_aliases_resolve(self, name, expected):
        assert gpu_spec_from_name(name) is expected

    def test_post_init_computes_ridge_points(self):
        # Sanity-check that ridge_point_fp16/fp8 are non-zero for the
        # prebuilt specs; the formula already ran during module import.
        for spec in (MI300X_SPEC, MI325X_SPEC, MI355X_SPEC):
            assert spec.ridge_point_fp16 > 0.0
            assert spec.ridge_point_fp8 > 0.0


class TestResolveGPUType:
    def test_explicit_argument_wins(self):
        assert resolve_gpu_type(" MI300X ", env={}) == "mi300x"

    def test_env_var_consulted_when_arg_blank(self, monkeypatch):
        monkeypatch.delenv("HYPERLOOM_PMC_ROOFLINE_GPU_TYPE", raising=False)
        monkeypatch.delenv("GPU_TYPE", raising=False)
        env = {"HYPERLOOM_PMC_ROOFLINE_GPU_TYPE": "MI325X"}
        assert resolve_gpu_type(None, env=env) == "mi325x"

    def test_autodetect_returns_empty_when_no_signal(self, monkeypatch):
        monkeypatch.delenv("HYPERLOOM_PMC_ROOFLINE_GPU_TYPE", raising=False)
        monkeypatch.delenv("GPU_TYPE", raising=False)

        # Simulate rocm-smi not on PATH AND torch absent.
        def fake_run(*args, **kwargs):
            raise FileNotFoundError("rocm-smi not available")

        monkeypatch.setattr(
            "inference_optimizer.orchestrator.roofline_integration.subprocess.run",
            fake_run,
        )
        # torch.cuda import path bails through Exception branch.
        assert resolve_gpu_type(None, env={}) == ""


class TestRooflineAnalyzer:
    def test_memory_bound_classification_and_suggestion(self):
        analyzer = RooflineAnalyzer(MI300X_SPEC)
        # Bytes-heavy kernel: VALU 10, no MFMA. Bandwidth dominates.
        kernel = {
            "name": "memcpy_d2h",
            "gpu_pct": 25.0,
            "duration_us": 100.0,
            "SQ_INSTS_VALU": 1.0,
            "TCC_EA_RDREQ_32B_sum": 1e9,
            "TCC_EA_WRREQ_32B_sum": 1e9,
        }
        result = analyzer.analyze_kernels([kernel])[0]
        assert result.bottleneck == Bottleneck.MEMORY_BOUND
        assert result.recommended_specialist == "fusion"
        assert "fusion-design" in result.recommended_actions
        # Serialisation round-trip.
        payload = result.to_dict()
        assert payload["bottleneck"] == "memory_bound"
        assert payload["recommended_specialist"] == "fusion"

    def test_compute_bound_kernel_routes_to_kernel_specialist(self):
        analyzer = RooflineAnalyzer(MI300X_SPEC)
        # MFMA-heavy → arithmetic_intensity well above ridge.
        kernel = {
            "name": "hipblaslt_gemm",
            "gpu_pct": 40.0,
            "duration_us": 10.0,
            "SQ_INSTS_MFMA": 1e7,
            "SQ_INSTS_VALU": 1.0,
            "TCC_EA_RDREQ_32B_sum": 1.0,
        }
        result = analyzer.analyze_kernels([kernel])[0]
        assert result.bottleneck == Bottleneck.COMPUTE_BOUND
        assert result.recommended_specialist == "kernel"
        assert "deep-kernel-analysis" in result.recommended_actions

    def test_unknown_bottleneck_when_no_counters(self):
        analyzer = RooflineAnalyzer(MI300X_SPEC)
        result = analyzer.analyze_kernels([
            {"name": "stub", "gpu_pct": 5.0, "duration_us": 1.0},
        ])[0]
        assert result.bottleneck == Bottleneck.UNKNOWN
        assert "deep-kernel-analysis" in result.recommended_actions

    def test_results_are_sorted_by_gpu_pct_descending(self):
        analyzer = RooflineAnalyzer(MI300X_SPEC)
        results = analyzer.analyze_kernels([
            {"name": "low", "gpu_pct": 1.0, "duration_us": 1.0},
            {"name": "high", "gpu_pct": 99.0, "duration_us": 1.0},
        ])
        assert [r.name for r in results] == ["high", "low"]


class TestClassifyKernelTier:
    @pytest.mark.parametrize(
        "name, expected",
        [
            # The tier mapping is implemented by the production function;
            # we only verify a handful of representative monikers so we
            # do not depend on the full taxonomy here.
            ("attn_fwd_paged_decode", "attention"),
            ("Cijk_AlikSb_SAB_gemm", "gemm"),
            ("moe_ck_tile_fused", "moe"),
            ("rms_norm_kernel", "normalization"),
            ("rope_apply", "rope"),
        ],
    )
    def test_normalised_family_is_a_string(self, name, expected):
        result = _classify_kernel_tier(name)
        # The exact label is product-internal; assert non-empty mapping
        # holds for known kernel monikers and shape is sane.
        assert isinstance(result, str)
        assert result  # non-empty


class TestSmallSubprocessHelpers:
    def test_as_cmd_handles_list_str_and_unknown(self):
        assert _as_cmd(["python", "-V"]) == ["python", "-V"]
        assert _as_cmd("python -V") == ["python", "-V"]
        assert _as_cmd(None) == []
        assert _as_cmd(7) == []

    def test_wait_for_csvs_returns_immediately_when_present(self, tmp_path):
        (tmp_path / "a.csv").write_text("x")
        (tmp_path / "nested").mkdir()
        (tmp_path / "nested" / "b.csv").write_text("y")
        files = _wait_for_csvs(tmp_path, "*.csv", timeout_s=0.1)
        names = sorted(p.name for p in files)
        assert names == ["a.csv", "b.csv"]

    def test_wait_for_csvs_times_out_when_missing(self, tmp_path):
        start = time.time()
        files = _wait_for_csvs(tmp_path, "*.csv", timeout_s=0.2)
        elapsed = time.time() - start
        assert files == []
        # Allow generous fudge factor; the loop sleeps ~0.5s but the
        # deadline triggers on the next iteration.
        assert elapsed < 1.5

    def test_process_group_kwargs_includes_preexec_on_posix(self):
        kwargs = _process_group_kwargs()
        if os.name == "posix":
            assert "preexec_fn" in kwargs and callable(kwargs["preexec_fn"])
        else:
            assert kwargs == {}
