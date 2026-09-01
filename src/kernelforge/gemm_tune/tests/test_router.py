# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for router module."""

import pytest

from kernelforge.gemm_tune import router
from kernelforge.gemm_tune.model_analyzer import ModelProfile
from kernelforge.gemm_tune.router import select_tuners


def _make_profile(is_moe=False, num_experts=0, **kwargs):
    defaults = {
        "model_path": "/fake/model",
        "hidden_size": 4096,
        "intermediate_size": 11008,
        "moe_intermediate_size": 0,
        "num_experts_per_tok": 0,
    }
    defaults.update(kwargs)
    return ModelProfile(is_moe=is_moe, num_experts=num_experts, **defaults)


class TestSelectTuners:
    def test_sglang_moe_bf16(self):
        profile = _make_profile(is_moe=True, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768)
        specs = select_tuners(profile, framework="sglang", precision="bf16", quant_type="none")
        names = [s.name for s in specs if s.should_run]
        assert "fmoe_ck" in names
        assert "sglang_dense_bf16" in names

    def test_sglang_moe_fp8_per_token_skips(self):
        profile = _make_profile(is_moe=True, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768)
        specs = select_tuners(profile, framework="sglang", precision="fp8", quant_type="per_token")
        fmoe = [s for s in specs if s.name == "fmoe_ck"][0]
        assert not fmoe.should_run
        assert "1-stage ASM" in fmoe.skip_reason

    def test_sglang_moe_fp8_blockscale_runs(self):
        profile = _make_profile(is_moe=True, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768)
        specs = select_tuners(profile, framework="sglang", precision="fp8", quant_type="blockscale")
        fmoe = [s for s in specs if s.name == "fmoe_ck"][0]
        assert fmoe.should_run

    def test_sglang_dense_bf16(self):
        profile = _make_profile(is_moe=False)
        specs = select_tuners(profile, framework="sglang", precision="bf16", quant_type="none")
        names = [s.name for s in specs if s.should_run]
        assert "sglang_dense_bf16" in names
        assert "fmoe_ck" not in names

    def test_noncanonical_quant_types_resolve_to_a_dense_tuner(self):
        # Non-canonical quant_type spellings from callers must still select the
        # right dense tuner instead of selecting nothing (-> tuner_not_applicable).
        # gpu_type is pinned so routing does not depend on the host's probed arch.
        profile = _make_profile(is_moe=False)
        for qt, expected in [
            ("w8a8_fp8", "a8w8"),
            ("a8w8_blockscale", "a8w8_blockscale"),
            ("a8w8_bpreshuffle", "a8w8_bpreshuffle"),
            (
                "a8w8_blockscale_bpreshuffle",
                "a8w8_blockscale_bpreshuffle",
            ),
        ]:
            specs = select_tuners(profile, framework="sglang", precision="fp8", quant_type=qt, gpu_type="mi300x")
            names = [s.name for s in specs if s.should_run]
            assert expected in names, f"{qt} -> {names}"

    def test_bpreshuffle_routing_is_arch_conditional(self):
        # Per-token bpreshuffle routes to the dedicated a8w8_bpreshuffle tuner,
        # which writes AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE — the exact config
        # table the gemm_a8w8_bpreshuffle serving op reads.
        profile = _make_profile(is_moe=False)

        specs = select_tuners(profile, framework="sglang", precision="fp8", quant_type="bpreshuffle", gpu_type="mi300x")
        names = [s.name for s in specs if s.should_run]
        assert "a8w8_bpreshuffle" in names, names
        # The blockscale+bpreshuffle tuner (different config table) must NOT be
        # substituted for a per-token bpreshuffle request.
        assert "a8w8_blockscale_bpreshuffle" not in names, names

    def test_bpreshuffle_skips_on_gfx950(self):
        # On gfx950 the CK a8w8_bpreshuffle tuner crashes (FNUZ/OCP dtype
        # mismatch) and the blockscale+bpreshuffle tuner writes a table the
        # per-token serving op never reads, so tuning is skipped with a reason
        # rather than silently producing an unused config.
        profile = _make_profile(is_moe=False)
        specs = select_tuners(profile, framework="sglang", precision="fp8", quant_type="bpreshuffle", gpu_type="mi355x")
        bpre = [s for s in specs if s.name == "a8w8_bpreshuffle"]
        assert bpre, [s.name for s in specs]
        assert not bpre[0].should_run
        assert "gfx950" in bpre[0].skip_reason
        # Must not silently fall back to the mismatched blockscale tuner.
        assert not any(s.name == "a8w8_blockscale_bpreshuffle" and s.should_run for s in specs)

    def test_sglang_dense_fp8_skips_when_no_shapes_obtainable(self):
        # Degenerate config (no dims) + no csv/shapes -> graceful skip, not a
        # hard validation failure (M2).
        profile = _make_profile(is_moe=False, hidden_size=0, intermediate_size=0)
        specs = select_tuners(
            profile, framework="sglang", precision="fp8", quant_type="blockscale", has_untuned_csv=False
        )
        blockscale = [s for s in specs if s.name == "a8w8_blockscale"][0]
        assert not blockscale.should_run
        assert "GEMM shapes" in blockscale.skip_reason

    def test_sglang_dense_fp8_runs_when_config_derivable(self):
        profile = _make_profile(is_moe=False)  # has hidden/intermediate
        specs = select_tuners(
            profile, framework="sglang", precision="fp8", quant_type="blockscale", has_untuned_csv=False
        )
        blockscale = [s for s in specs if s.name == "a8w8_blockscale"][0]
        assert blockscale.should_run

    def test_sglang_dense_fp8_blockscale_runs_without_csv(self):
        # Dense fp8 now derives shapes from config when no CSV is supplied, so it
        # is selected to run instead of being skipped.
        profile = _make_profile(is_moe=False)
        specs = select_tuners(
            profile, framework="sglang", precision="fp8", quant_type="blockscale", has_untuned_csv=False
        )
        blockscale = [s for s in specs if s.name == "a8w8_blockscale"][0]
        assert blockscale.should_run
        assert blockscale.skip_reason is None

    def test_sglang_dense_fp8_blockscale_with_csv(self):
        profile = _make_profile(is_moe=False)
        specs = select_tuners(
            profile, framework="sglang", precision="fp8", quant_type="blockscale", has_untuned_csv=True
        )
        blockscale = [s for s in specs if s.name == "a8w8_blockscale"][0]
        assert blockscale.should_run

    def test_vllm_moe(self):
        profile = _make_profile(is_moe=True, num_experts=64, num_experts_per_tok=4)
        specs = select_tuners(profile, framework="vllm", precision="bf16")
        names = [s.name for s in specs if s.should_run]
        assert "vllm_moe_triton" in names

    def test_vllm_aiter_uses_sglang_tuners(self):
        profile = _make_profile(is_moe=True, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768)
        specs = select_tuners(profile, framework="vllm-aiter", precision="bf16", quant_type="none")
        names = [s.name for s in specs if s.should_run]
        assert "fmoe_ck" in names

    def test_unknown_framework(self):
        profile = _make_profile()
        specs = select_tuners(profile, framework="unknown", precision="bf16")
        assert len(specs) == 0

    def test_priority_order(self):
        profile = _make_profile(is_moe=True, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768)
        specs = select_tuners(profile, framework="sglang", precision="bf16", quant_type="none")
        runnable = [s for s in specs if s.should_run]
        priorities = [s.priority for s in runnable]
        assert priorities == sorted(priorities)


class TestFp4Gfx942Skip:
    """FP4/MXFP4 GEMM is unsupported on gfx942 (aiter requires gfx950).

    The router must skip both the dense a4w4_blockscale tuner and the
    fp4/mxfp4 fmoe_ck MoE path on gfx942 GPUs (mi300x/mi308x/mi325x), while
    still selecting them on gfx950 GPUs (mi355x).
    """

    def test_dense_fp4_skipped_on_gfx942(self):
        profile = _make_profile(is_moe=False)
        specs = select_tuners(
            profile, framework="sglang", precision="fp4", quant_type="fp4", gpu_type="mi300x", has_untuned_csv=True
        )
        a4w4 = [s for s in specs if s.name == "a4w4_blockscale"][0]
        assert not a4w4.should_run
        assert "gfx942" in a4w4.skip_reason
        assert "gfx950" in a4w4.skip_reason

    def test_dense_fp4_runs_on_gfx950(self):
        profile = _make_profile(is_moe=False)
        specs = select_tuners(
            profile, framework="sglang", precision="fp4", quant_type="fp4", gpu_type="mi355x", has_untuned_csv=True
        )
        a4w4 = [s for s in specs if s.name == "a4w4_blockscale"][0]
        assert a4w4.should_run

    def test_moe_mxfp4_skipped_on_gfx942(self):
        profile = _make_profile(is_moe=True, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768)
        specs = select_tuners(profile, framework="sglang", precision="mxfp4", quant_type="mxfp4", gpu_type="mi300x")
        fmoe = [s for s in specs if s.name == "fmoe_ck"][0]
        assert not fmoe.should_run
        assert "gfx942" in fmoe.skip_reason
        assert "gfx950" in fmoe.skip_reason

    def test_moe_mxfp4_runs_on_gfx950(self):
        profile = _make_profile(is_moe=True, num_experts=128, num_experts_per_tok=8, moe_intermediate_size=768)
        specs = select_tuners(profile, framework="sglang", precision="mxfp4", quant_type="mxfp4", gpu_type="mi355x")
        fmoe = [s for s in specs if s.name == "fmoe_ck"][0]
        assert fmoe.should_run

    def test_other_gfx942_skus_skip_fp4(self):
        for gpu_type in ("mi308x", "mi325x"):
            profile = _make_profile(is_moe=False)
            specs = select_tuners(
                profile, framework="sglang", precision="fp4", quant_type="fp4", gpu_type=gpu_type, has_untuned_csv=True
            )
            a4w4 = [s for s in specs if s.name == "a4w4_blockscale"][0]
            assert not a4w4.should_run, f"{gpu_type} should skip fp4"

    def test_fp8_unaffected_on_gfx942(self):
        profile = _make_profile(is_moe=False)
        specs = select_tuners(
            profile,
            framework="sglang",
            precision="fp8",
            quant_type="blockscale",
            gpu_type="mi300x",
            has_untuned_csv=True,
        )
        blockscale = [s for s in specs if s.name == "a8w8_blockscale"][0]
        assert blockscale.should_run


class TestGpuTypeAutoDetect:
    """gpu_type='auto'/'' probes the local host via rocminfo, then gates FP4.

    Detection failure (no rocminfo) must fail open: never skip a tuner on an
    undetectable host.
    """

    def test_auto_detects_gfx942_and_skips_fp4(self, monkeypatch):
        monkeypatch.setattr(router, "_detect_local_gfx_arch", lambda: "gfx942")
        profile = _make_profile(is_moe=False)
        specs = select_tuners(
            profile, framework="sglang", precision="fp4", quant_type="fp4", gpu_type="auto", has_untuned_csv=True
        )
        a4w4 = [s for s in specs if s.name == "a4w4_blockscale"][0]
        assert not a4w4.should_run
        assert "gfx942" in a4w4.skip_reason

    def test_auto_detects_gfx950_and_runs_fp4(self, monkeypatch):
        monkeypatch.setattr(router, "_detect_local_gfx_arch", lambda: "gfx950")
        profile = _make_profile(is_moe=False)
        specs = select_tuners(
            profile, framework="sglang", precision="fp4", quant_type="fp4", gpu_type="auto", has_untuned_csv=True
        )
        a4w4 = [s for s in specs if s.name == "a4w4_blockscale"][0]
        assert a4w4.should_run

    def test_auto_fails_open_when_undetectable(self, monkeypatch):
        monkeypatch.setattr(router, "_detect_local_gfx_arch", lambda: "")
        profile = _make_profile(is_moe=False)
        specs = select_tuners(
            profile, framework="sglang", precision="fp4", quant_type="fp4", gpu_type="auto", has_untuned_csv=True
        )
        a4w4 = [s for s in specs if s.name == "a4w4_blockscale"][0]
        assert a4w4.should_run

    def test_resolve_gfx_arch_auto_and_empty(self, monkeypatch):
        monkeypatch.setattr(router, "_detect_local_gfx_arch", lambda: "gfx942")
        assert router._resolve_gfx_arch("auto") == "gfx942"
        assert router._resolve_gfx_arch("") == "gfx942"
        assert router._resolve_gfx_arch("  AUTO ") == "gfx942"
        # explicit values bypass detection
        assert router._resolve_gfx_arch("mi355x") == "gfx950"
        assert router._resolve_gfx_arch("gfx942") == "gfx942"

    def test_detect_local_gfx_arch_parses_rocminfo(self, monkeypatch):
        sample = "  Name:                    AMD EPYC\n  Name:                    gfx942\n"

        class _Completed:
            stdout = sample

        monkeypatch.setattr(router.subprocess, "run", lambda *a, **k: _Completed())
        assert router._detect_local_gfx_arch() == "gfx942"

    def test_detect_local_gfx_arch_failopen_on_missing_binary(self, monkeypatch):
        def _raise(*a, **k):
            raise FileNotFoundError("rocminfo")

        monkeypatch.setattr(router.subprocess, "run", _raise)
        assert router._detect_local_gfx_arch() == ""

    @pytest.mark.parametrize(
        ("detected", "expected"),
        [("gfx942", "mi300x"), ("gfx950", "mi355x")],
    )
    def test_public_resolver_detects_once_and_canonicalizes(
        self,
        monkeypatch,
        detected,
        expected,
    ):
        calls = []

        def detect():
            calls.append(True)
            return detected

        monkeypatch.setattr(router, "_detect_local_gfx_arch", detect)

        assert router.resolve_gpu_type("auto") == expected
        assert calls == [True]

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("gfx942", "mi300x"),
            ("MI300X", "mi300x"),
            ("AMD Instinct MI300X", "mi300x"),
            ("mi308x", "mi300x"),
            ("mi325x", "mi300x"),
            ("gfx950", "mi355x"),
            ("MI355X", "mi355x"),
            ("AMD-Instinct-MI355X", "mi355x"),
        ],
    )
    def test_public_resolver_canonicalizes_explicit_known_values(
        self,
        raw,
        expected,
    ):
        assert router.resolve_gpu_type(raw) == expected

    def test_public_resolver_fails_closed_when_detection_fails(self, monkeypatch):
        monkeypatch.setattr(router, "_detect_local_gfx_arch", lambda: "")

        with pytest.raises(ValueError, match="--gpu-type"):
            router.resolve_gpu_type("auto")

    def test_auto_resolves_each_architecture_to_its_own_canonical_name(self, monkeypatch):
        detected = iter(("gfx942", "gfx950"))
        monkeypatch.setattr(router, "_detect_local_gfx_arch", lambda: next(detected))

        gpu_types = [router.resolve_gpu_type("auto") for _ in range(2)]

        assert gpu_types == ["mi300x", "mi355x"]
        assert all("auto" not in value for value in gpu_types)
