# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``orchestrator.roofline_ceiling``.

Pins three contracts:

1. **Formula correctness against published numbers.** The decode-only
   memory-bound peak matches ITK Research's worked B200 + 70 B FP8
   example (≈ 114 tok/s) and the MI300X scaling of that example.
2. **Graceful degrade.** Unknown ``gpu_type``, empty ``model_path``,
   missing HF index, and ``concurrency=0`` all return ``0.0`` (never
   raise, never produce inf / NaN).
3. **HF metadata parsing.** ``load_model_meta`` reads ``total_size`` /
   layer / KV / dtype from a minimal HF dir laid out under ``tmp_path``.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator.roofline_ceiling import (
    HW_SPECS,
    ModelMeta,
    RooflineBreakdown,
    _resolve_dtype_bytes,
    _resolve_peak_tflops,
    compute_compute_bound_ceiling_tok_per_sec,
    compute_kv_bytes_per_token,
    compute_peak_from_state,
    compute_roofline_breakdown_from_state,
    compute_theoretical_peak_output_tok_per_sec,
    load_model_meta,
)


# ---------------------------------------------------------------------------
# Formula correctness against published numbers.
# ---------------------------------------------------------------------------
class TestPeakFormulaAgainstPublishedNumbers:
    """Worked examples from ITK Research and arXiv 2402.16363.

    The cited B200 + 70 B FP8 example uses ``HBM_BW / weight_bytes``
    with KV-cache treated as negligible relative to weights. We
    replicate that by injecting a synthetic spec into ``HW_SPECS`` and
    setting ``num_layers=0`` so the KV term is zero.
    """

    def test_itk_b200_70b_fp8_matches_cited_114_tok_s(self, monkeypatch):
        # ITK Research: B200 (8 TB/s HBM) + 70 GB FP8 weights.
        # 8e12 / 70e9 ≈ 114.28 tok/s. Inject B200 spec for this test only.
        monkeypatch.setitem(
            HW_SPECS, "b200_test", {"hbm_gb": 192.0, "hbm_bw_gbps": 8000.0}
        )
        peak = compute_theoretical_peak_output_tok_per_sec(
            gpu_type="b200_test",
            num_gpus=1,
            weight_bytes=70_000_000_000,
            num_layers=0,
            num_kv_heads=0,
            head_dim=0,
            kv_dtype_bytes=1.0,
            isl=0,
            osl=0,
            concurrency=1,
        )
        assert peak == pytest.approx(114.28, rel=0.01)

    def test_itk_b200_70b_fp4_doubles_to_228_tok_s(self, monkeypatch):
        """FP8 → FP4 halves bytes/param so peak doubles, per ITK Research."""
        monkeypatch.setitem(
            HW_SPECS, "b200_test", {"hbm_gb": 192.0, "hbm_bw_gbps": 8000.0}
        )
        peak_fp4 = compute_theoretical_peak_output_tok_per_sec(
            gpu_type="b200_test",
            num_gpus=1,
            weight_bytes=35_000_000_000,
            num_layers=0,
            num_kv_heads=0,
            head_dim=0,
            kv_dtype_bytes=0.5,
            isl=0,
            osl=0,
            concurrency=1,
        )
        assert peak_fp4 == pytest.approx(228.57, rel=0.01)


# ---------------------------------------------------------------------------
# MI300X realistic sanity (Llama-70B-style dense model).
# ---------------------------------------------------------------------------
class TestMI300XRealistic:
    """Llama-3-70B BF16 on 1×MI300X: literature places single-stream
    decode at 30-40 tok/s (5.3 TB/s / 140 GB ≈ 37.8)."""

    def test_llama70b_bf16_single_mi300x(self):
        peak = compute_theoretical_peak_output_tok_per_sec(
            gpu_type="mi300x",
            num_gpus=1,
            weight_bytes=140_000_000_000,
            num_layers=80,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype_bytes=2.0,
            isl=2048,
            osl=512,
            concurrency=1,
        )
        assert 25.0 < peak < 45.0

    def test_concurrency_scales_throughput_in_weight_dominated_regime(self):
        """With small isl/osl, weight reads dominate; batching N requests
        amortizes weight reads N× so peak should be near-linearly higher."""
        kwargs = dict(
            gpu_type="mi300x",
            num_gpus=8,
            weight_bytes=140_000_000_000,
            num_layers=80,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype_bytes=2.0,
            isl=128,
            osl=64,
        )
        p1 = compute_theoretical_peak_output_tok_per_sec(
            **kwargs, concurrency=1
        )
        p20 = compute_theoretical_peak_output_tok_per_sec(
            **kwargs, concurrency=20
        )
        # Weight reads amortize: 20× concurrency should give >10× peak
        # (not exactly 20× because KV term doesn't amortize).
        assert p20 / max(p1, 1e-9) > 10.0


# ---------------------------------------------------------------------------
# Graceful degrade.
# ---------------------------------------------------------------------------
class TestGracefulDegrade:
    def test_unknown_gpu_type_returns_zero(self):
        peak = compute_theoretical_peak_output_tok_per_sec(
            gpu_type="rtx4090",
            num_gpus=1,
            weight_bytes=10**9,
            num_layers=12,
            num_kv_heads=8,
            head_dim=64,
            kv_dtype_bytes=2.0,
            isl=128,
            osl=128,
            concurrency=1,
        )
        assert peak == 0.0

    def test_empty_gpu_type_returns_zero(self):
        peak = compute_theoretical_peak_output_tok_per_sec(
            gpu_type="",
            num_gpus=1,
            weight_bytes=10**9,
            num_layers=12,
            num_kv_heads=8,
            head_dim=64,
            kv_dtype_bytes=2.0,
            isl=128,
            osl=128,
            concurrency=1,
        )
        assert peak == 0.0

    def test_zero_concurrency_clamps_to_one(self):
        peak = compute_theoretical_peak_output_tok_per_sec(
            gpu_type="mi300x",
            num_gpus=1,
            weight_bytes=10**9,
            num_layers=12,
            num_kv_heads=8,
            head_dim=64,
            kv_dtype_bytes=2.0,
            isl=128,
            osl=128,
            concurrency=0,
        )
        assert peak > 0.0

    def test_zero_weight_and_kv_returns_zero(self):
        # Pathological all-zero inputs must not raise / divide-by-zero.
        peak = compute_theoretical_peak_output_tok_per_sec(
            gpu_type="mi300x",
            num_gpus=1,
            weight_bytes=0,
            num_layers=0,
            num_kv_heads=0,
            head_dim=0,
            kv_dtype_bytes=0.0,
            isl=0,
            osl=0,
            concurrency=1,
        )
        assert peak == 0.0


# ---------------------------------------------------------------------------
# Dtype + KV helpers.
# ---------------------------------------------------------------------------
class TestResolveDtypeBytes:
    @pytest.mark.parametrize(
        "tag,expected",
        [
            ("bfloat16", 2.0),
            ("BF16", 2.0),
            ("float16", 2.0),
            ("float32", 4.0),
            ("float8_e4m3fn", 1.0),
            ("float8_e5m2", 1.0),
            ("fp8", 1.0),
            ("fp4", 0.5),
            ("", 2.0),
            (None, 2.0),
            ("unknown_dtype", 2.0),
        ],
    )
    def test_canonical_mapping(self, tag, expected):
        assert _resolve_dtype_bytes(tag) == expected


class TestComputeKVBytesPerToken:
    def test_factor_two_for_k_plus_v(self):
        # 2 (K+V) × 80 layers × 8 KV heads × 128 head_dim × 2 bytes
        #   = 327_680 bytes per token, summed over all layers
        assert compute_kv_bytes_per_token(
            num_layers=80,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype_bytes=2.0,
        ) == 327_680

    def test_fp8_kv_halves_volume(self):
        assert compute_kv_bytes_per_token(
            num_layers=80,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype_bytes=1.0,
        ) == 163_840


# ---------------------------------------------------------------------------
# HF metadata extraction.
# ---------------------------------------------------------------------------
def _write_synthetic_model(
    model_dir: Path,
    *,
    total_size: int,
    num_layers: int = 80,
    num_kv_heads: int | None = 8,
    hidden_size: int = 8192,
    num_attention_heads: int = 64,
    torch_dtype: str = "bfloat16",
    head_dim: int | None = None,
    num_experts: int | None = None,
    num_experts_per_tok: int | None = None,
    moe_intermediate_size: int | None = None,
    n_routed_experts: int | None = None,
    num_local_experts: int | None = None,
    quant_method: str | None = None,
    dtype: str | None = None,
) -> None:
    """Lay down a minimal HF-shaped model dir.

    Pass ``num_kv_heads=None`` to omit ``num_key_value_heads`` (MHA
    branch). Pass ``num_experts`` / ``num_experts_per_tok`` /
    ``moe_intermediate_size`` to emit an MoE config (Qwen3-A3B style).
    Pass ``n_routed_experts`` to emit a DeepSeek-V3-style alias instead
    of ``num_experts``. Pass ``num_local_experts`` to emit a gpt-oss-
    style alias (GptOssForCausalLM). Pass ``quant_method``
    (e.g. ``"fp8"``) to emit a ``quantization_config`` block. Pass
    ``dtype`` to write the DeepSeek-style activation-dtype field
    alongside / instead of the standard ``torch_dtype``.
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    config: dict = {
        "num_hidden_layers": num_layers,
        "num_attention_heads": num_attention_heads,
        "hidden_size": hidden_size,
        "torch_dtype": torch_dtype,
    }
    if num_kv_heads is not None:
        config["num_key_value_heads"] = num_kv_heads
    if head_dim is not None:
        config["head_dim"] = head_dim
    if num_experts is not None:
        config["num_experts"] = num_experts
    if num_experts_per_tok is not None:
        config["num_experts_per_tok"] = num_experts_per_tok
    if moe_intermediate_size is not None:
        config["moe_intermediate_size"] = moe_intermediate_size
    if n_routed_experts is not None:
        config["n_routed_experts"] = n_routed_experts
    if num_local_experts is not None:
        config["num_local_experts"] = num_local_experts
    if quant_method is not None:
        config["quantization_config"] = {
            "quant_method": quant_method,
            "weight_block_size": [128, 128],
            "activation_scheme": "dynamic",
        }
    if dtype is not None:
        config["dtype"] = dtype
    (model_dir / "config.json").write_text(json.dumps(config))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": {}})
    )


class TestLoadModelMeta:
    def test_reads_total_size_and_geometry(self, tmp_path):
        _write_synthetic_model(
            tmp_path / "m",
            total_size=70_000_000_000,
            num_layers=80,
            num_kv_heads=8,
            hidden_size=8192,
            num_attention_heads=64,
            torch_dtype="float8_e4m3fn",
        )
        meta = load_model_meta(tmp_path / "m")
        assert meta is not None
        assert isinstance(meta, ModelMeta)
        assert meta.weight_bytes == 70_000_000_000
        assert meta.num_layers == 80
        assert meta.num_kv_heads == 8
        assert meta.head_dim == 128
        assert meta.weight_dtype_bytes == 1.0

    def test_mha_fallback_when_num_kv_heads_absent(self, tmp_path):
        _write_synthetic_model(
            tmp_path / "m",
            total_size=1_000_000_000,
            num_layers=12,
            num_kv_heads=None,
            num_attention_heads=16,
            hidden_size=2048,
        )
        meta = load_model_meta(tmp_path / "m")
        assert meta is not None
        assert meta.num_kv_heads == 16
        assert meta.head_dim == 128

    def test_head_dim_directly_from_config(self, tmp_path):
        _write_synthetic_model(
            tmp_path / "m",
            total_size=1_000_000_000,
            num_attention_heads=64,
            hidden_size=8192,
            head_dim=200,
        )
        meta = load_model_meta(tmp_path / "m")
        assert meta is not None
        assert meta.head_dim == 200

    def test_missing_safetensors_index_uses_safetensor_file_sizes(self, tmp_path):
        d = tmp_path / "m"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({
                "num_hidden_layers": 12,
                "num_key_value_heads": 4,
                "num_attention_heads": 8,
                "hidden_size": 1024,
                "torch_dtype": "bfloat16",
            })
        )
        (d / "model-00001-of-00002.safetensors").write_bytes(b"x" * 13)
        (d / "model-00002-of-00002.safetensors").write_bytes(b"y" * 17)

        meta = load_model_meta(d)
        assert meta is not None
        assert meta.weight_bytes == 30
        assert meta.num_layers == 12
        assert meta.num_kv_heads == 4
        assert meta.head_dim == 128

    def test_missing_safetensors_index_and_files_returns_none(self, tmp_path):
        d = tmp_path / "m"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"num_hidden_layers": 12, "torch_dtype": "bfloat16"})
        )
        assert load_model_meta(d) is None

    def test_safetensors_index_without_total_size_uses_file_sizes(self, tmp_path):
        d = tmp_path / "m"
        _write_synthetic_model(d, total_size=1_000_000_000)
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {}, "weight_map": {}})
        )
        (d / "model-00001-of-00001.safetensors").write_bytes(b"z" * 23)

        meta = load_model_meta(d)
        assert meta is not None
        assert meta.weight_bytes == 23

    def test_missing_safetensors_uses_pytorch_bin_file_sizes(self, tmp_path):
        d = tmp_path / "m"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({
                "num_hidden_layers": 12,
                "num_key_value_heads": 4,
                "num_attention_heads": 8,
                "hidden_size": 1024,
                "torch_dtype": "float16",
            })
        )
        (d / "pytorch_model-00001-of-00002.bin").write_bytes(b"a" * 11)
        (d / "pytorch_model-00002-of-00002.bin").write_bytes(b"b" * 19)

        meta = load_model_meta(d)
        assert meta is not None
        assert meta.weight_bytes == 30
        assert meta.weight_dtype_bytes == 2.0

    def test_missing_config_returns_none(self, tmp_path):
        d = tmp_path / "m"
        d.mkdir()
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 1_000_000_000}})
        )
        assert load_model_meta(d) is None

    def test_nonexistent_model_path_returns_none(self, tmp_path):
        assert load_model_meta(tmp_path / "does_not_exist") is None

    def test_empty_model_path_returns_none(self):
        assert load_model_meta("") is None

    def test_precision_hint_used_when_torch_dtype_missing(self, tmp_path):
        d = tmp_path / "m"
        d.mkdir()
        (d / "config.json").write_text(json.dumps({
            "num_hidden_layers": 80,
            "num_key_value_heads": 8,
            "num_attention_heads": 64,
            "hidden_size": 8192,
        }))
        (d / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 70_000_000_000}})
        )
        meta = load_model_meta(d, precision_hint="fp8")
        assert meta is not None
        assert meta.weight_dtype_bytes == 1.0


# ---------------------------------------------------------------------------
# State-driven entry point.
# ---------------------------------------------------------------------------
class TestComputePeakFromState:
    def test_happy_path_yields_positive(self, tmp_path):
        _write_synthetic_model(
            tmp_path / "m",
            total_size=140_000_000_000,
            num_layers=80,
            num_kv_heads=8,
            hidden_size=8192,
            num_attention_heads=64,
            torch_dtype="bfloat16",
        )
        state = SimpleNamespace(
            model_path=str(tmp_path / "m"),
            gpu_type="mi300x",
            tp=8,
            precision="bf16",
            conc=20,
            isl=2048,
            osl=512,
        )
        peak = compute_peak_from_state(state)
        assert peak > 0.0
        assert peak < 1e6

    def test_missing_model_path_returns_zero(self):
        state = SimpleNamespace(
            model_path="/no/such/dir",
            gpu_type="mi300x",
            tp=8,
            precision="bf16",
            conc=20,
            isl=2048,
            osl=512,
        )
        assert compute_peak_from_state(state) == 0.0

    def test_unknown_gpu_returns_zero(self, tmp_path):
        _write_synthetic_model(tmp_path / "m", total_size=10**9)
        state = SimpleNamespace(
            model_path=str(tmp_path / "m"),
            gpu_type="rtx4090",
            tp=1,
            precision="bf16",
            conc=1,
            isl=128,
            osl=128,
        )
        assert compute_peak_from_state(state) == 0.0


# ---------------------------------------------------------------------------
# MoE active weight bytes (PR: MoE-aware decode ceiling).
# ---------------------------------------------------------------------------
def _write_qwen3_moe_model(model_dir: Path, *, total_size: int) -> None:
    """Lay down a Qwen3-30B-A3B-shaped MoE HF dir for ceiling tests."""
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps({
        "architectures": ["Qwen3MoeForCausalLM"],
        "model_type": "qwen3_moe",
        "num_hidden_layers": 48,
        "num_attention_heads": 32,
        "num_key_value_heads": 4,
        "hidden_size": 2048,
        "head_dim": 128,
        "intermediate_size": 6144,
        "moe_intermediate_size": 768,
        "num_experts": 128,
        "num_experts_per_tok": 8,
        "torch_dtype": "bfloat16",
    }))
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": {}})
    )


class TestMoEActiveWeightBytes:
    """MoE models route a small subset of experts per token; the
    decode-roofline divisor must use the active subset or the ceiling
    drops below measured throughput (within_roofline_pct > 100%)."""

    def test_qwen3_30b_a3b_active_is_small_fraction_of_total(self, tmp_path):
        """Qwen3-30B-A3B: 128 experts, 8 active. Active weight bytes
        should land in the ~8-15% range of total (experts dominate the
        weight budget; non-expert + 8/128 × experts is the active set)."""
        _write_qwen3_moe_model(tmp_path / "m", total_size=61_064_245_248)
        meta = load_model_meta(tmp_path / "m")
        assert meta is not None
        assert meta.weight_bytes == 61_064_245_248
        assert 0 < meta.active_weight_bytes < meta.weight_bytes
        ratio = meta.active_weight_bytes / meta.weight_bytes
        assert 0.05 < ratio < 0.20, (
            f"Qwen3-30B-A3B active ratio {ratio:.3f} outside expected window"
        )

    def test_dense_model_active_equals_total(self, tmp_path):
        """Dense (no num_experts / num_experts_per_tok) → active = total,
        preserves legacy behaviour for Qwen3-8B / Llama-70B style configs."""
        _write_synthetic_model(
            tmp_path / "m",
            total_size=16_381_470_720,
            num_layers=36,
            num_kv_heads=8,
            num_attention_heads=32,
            hidden_size=4096,
        )
        meta = load_model_meta(tmp_path / "m")
        assert meta is not None
        assert meta.active_weight_bytes == meta.weight_bytes

    def test_moe_ceiling_higher_than_dense_equivalent(self, tmp_path):
        """Same total bytes, but MoE active routing → ceiling several
        × higher than naive dense ceiling. Pins the property the PR
        actually delivers (fixes 'within_roofline_pct > 100%' on MoE)."""
        _write_qwen3_moe_model(tmp_path / "m", total_size=61_064_245_248)
        meta = load_model_meta(tmp_path / "m")
        assert meta is not None
        kwargs = dict(
            gpu_type="mi355x",
            num_gpus=1,
            num_layers=meta.num_layers,
            num_kv_heads=meta.num_kv_heads,
            head_dim=meta.head_dim,
            kv_dtype_bytes=meta.weight_dtype_bytes,
            isl=1024,
            osl=1024,
            concurrency=8,
        )
        dense_peak = compute_theoretical_peak_output_tok_per_sec(
            weight_bytes=meta.weight_bytes, **kwargs,
        )
        moe_peak = compute_theoretical_peak_output_tok_per_sec(
            weight_bytes=meta.weight_bytes,
            active_weight_bytes=meta.active_weight_bytes,
            **kwargs,
        )
        assert moe_peak > dense_peak * 2.0, (
            f"MoE ceiling {moe_peak:.0f} should be much higher than "
            f"dense-treated ceiling {dense_peak:.0f}"
        )

    def test_active_weight_bytes_zero_falls_back_to_weight_bytes(self):
        """Backward-compat: callers that don't know about MoE leave
        active_weight_bytes=0; the function must behave as before."""
        without = compute_theoretical_peak_output_tok_per_sec(
            gpu_type="mi300x",
            num_gpus=1,
            weight_bytes=140_000_000_000,
            num_layers=80,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype_bytes=2.0,
            isl=2048,
            osl=512,
            concurrency=1,
        )
        with_zero = compute_theoretical_peak_output_tok_per_sec(
            gpu_type="mi300x",
            num_gpus=1,
            weight_bytes=140_000_000_000,
            active_weight_bytes=0,
            num_layers=80,
            num_kv_heads=8,
            head_dim=128,
            kv_dtype_bytes=2.0,
            isl=2048,
            osl=512,
            concurrency=1,
        )
        assert without == with_zero

    def test_moe_geometry_overshoots_safe_degrade(self, tmp_path):
        """If computed expert bytes >= safetensors total_size (config /
        quantization mismatch), helper must clamp to weight_bytes
        instead of producing negative non_expert_bytes."""
        model_dir = tmp_path / "m"
        model_dir.mkdir()
        # huge num_experts but tiny total_size → expert_bytes > total
        (model_dir / "config.json").write_text(json.dumps({
            "num_hidden_layers": 4,
            "num_attention_heads": 32,
            "num_key_value_heads": 4,
            "hidden_size": 4096,
            "moe_intermediate_size": 4096,
            "num_experts": 256,
            "num_experts_per_tok": 2,
            "torch_dtype": "bfloat16",
        }))
        (model_dir / "model.safetensors.index.json").write_text(
            json.dumps({"metadata": {"total_size": 100_000_000}})
        )
        meta = load_model_meta(model_dir)
        assert meta is not None
        # Safe degrade: active equals total (no inflation, no negatives).
        assert meta.active_weight_bytes == meta.weight_bytes


# ---------------------------------------------------------------------------
# HW_SPECS table sanity.
# ---------------------------------------------------------------------------
class TestResolveEffectiveConcurrency:
    """Concurrency fallback chain (PR-A): state.conc -> baseline yaml
    envs.CONC -> 1. The yaml path covers the SharedState-default-8 vs
    ci-config-CONC=64 mismatch the e2e exposed on Qwen3-30B-A3B."""

    def test_state_conc_wins_when_positive(self):
        from inference_optimizer.orchestrator.roofline_ceiling import (
            _resolve_effective_concurrency,
        )
        state = SimpleNamespace(conc=32, last_baseline={})
        assert _resolve_effective_concurrency(state) == 32

    def test_falls_back_to_baseline_yaml_envs_conc(self, tmp_path):
        from inference_optimizer.orchestrator.roofline_ceiling import (
            _resolve_effective_concurrency,
        )
        yaml_path = tmp_path / "baseline_config.with_envs.yaml"
        yaml_path.write_text(
            "benchmark:\n"
            "  envs:\n"
            "    CONC: 64\n"
            "    ISL: 256\n",
            encoding="utf-8",
        )
        state = SimpleNamespace(
            conc=0,  # SharedState default — drives the fallback
            last_baseline={
                "extras": {"materialized_config": str(yaml_path)},
            },
        )
        assert _resolve_effective_concurrency(state) == 64

    def test_baseline_yaml_conc_is_authoritative_over_stale_state_conc(
        self, tmp_path,
    ):
        """P0 fix (priority inverted): the materialized baseline yaml's
        ``CONC`` is the ground truth the Magpie subprocess actually ran
        with, so it wins over ``state.conc`` even when the latter is
        positive. This reproduces the e2e bug where ``state.conc`` stayed
        at the SharedState default 8 while the run actually used CONC=64
        (session 095726Z): the old code returned 8 and under-counted the
        ceiling ~8x; the fix returns 64."""
        from inference_optimizer.orchestrator.roofline_ceiling import (
            _resolve_effective_concurrency,
        )
        yaml_path = tmp_path / "baseline_config.with_envs.yaml"
        yaml_path.write_text(
            "benchmark:\n  envs:\n    CONC: 64\n",
            encoding="utf-8",
        )
        state = SimpleNamespace(
            conc=8,  # stale SharedState default
            last_baseline={
                "extras": {"materialized_config": str(yaml_path)},
            },
        )
        assert _resolve_effective_concurrency(state) == 64

    def test_missing_yaml_falls_back_to_one(self):
        from inference_optimizer.orchestrator.roofline_ceiling import (
            _resolve_effective_concurrency,
        )
        state = SimpleNamespace(
            conc=0,
            last_baseline={
                "extras": {"materialized_config": "/no/such/file.yaml"},
            },
        )
        assert _resolve_effective_concurrency(state) == 1

    def test_malformed_yaml_falls_back_to_one(self, tmp_path):
        from inference_optimizer.orchestrator.roofline_ceiling import (
            _resolve_effective_concurrency,
        )
        bad = tmp_path / "broken.yaml"
        bad.write_text("not: [valid yaml at all", encoding="utf-8")
        state = SimpleNamespace(
            conc=0,
            last_baseline={"extras": {"materialized_config": str(bad)}},
        )
        assert _resolve_effective_concurrency(state) == 1

    def test_yaml_without_conc_falls_back_to_one(self, tmp_path):
        from inference_optimizer.orchestrator.roofline_ceiling import (
            _resolve_effective_concurrency,
        )
        yaml_path = tmp_path / "no_conc.yaml"
        yaml_path.write_text(
            "benchmark:\n  envs:\n    ISL: 256\n",
            encoding="utf-8",
        )
        state = SimpleNamespace(
            conc=0,
            last_baseline={"extras": {"materialized_config": str(yaml_path)}},
        )
        assert _resolve_effective_concurrency(state) == 1

    def test_no_last_baseline_falls_back_to_one(self):
        from inference_optimizer.orchestrator.roofline_ceiling import (
            _resolve_effective_concurrency,
        )
        state = SimpleNamespace(conc=0, last_baseline=None)
        assert _resolve_effective_concurrency(state) == 1

    def test_compute_peak_from_state_uses_yaml_fallback(self, tmp_path):
        """End-to-end: state.conc=0 + yaml envs.CONC=64 →
        peak computed with batch=64 (not 1, not 8)."""
        _write_synthetic_model(
            tmp_path / "m",
            total_size=140_000_000_000,
            num_layers=80, num_kv_heads=8,
            hidden_size=8192, num_attention_heads=64,
            torch_dtype="bfloat16",
        )
        yaml_path = tmp_path / "bl.yaml"
        yaml_path.write_text(
            "benchmark:\n  envs:\n    CONC: 64\n",
            encoding="utf-8",
        )
        state = SimpleNamespace(
            model_path=str(tmp_path / "m"),
            gpu_type="mi300x",
            tp=1,
            precision="bf16",
            conc=0,  # default — forces yaml fallback
            isl=2048,
            osl=512,
            last_baseline={"extras": {"materialized_config": str(yaml_path)}},
        )
        peak_with_yaml = compute_peak_from_state(state)
        # Drop the yaml so resolution falls through to state.conc=1 (yaml is
        # authoritative under the P0 fix, so it would otherwise still win).
        state.last_baseline = {}
        state.conc = 1
        peak_with_conc_1 = compute_peak_from_state(state)
        # batch=64 amortizes weight reads 64×, so the yaml-resolved peak
        # must be substantially higher than the conc=1 peak.
        assert peak_with_yaml > 10 * peak_with_conc_1


class TestMoEBatchSaturation:
    """P1 fix: the MoE weight-read term grows with batch as the union of
    activated experts saturates toward all experts. A constant
    ``active_weight_bytes`` (B=1 routing) over-amortizes at high batch and
    inflates the ceiling (within% collapses below the real value)."""

    _COMMON = dict(
        gpu_type="mi355x", num_gpus=1, weight_bytes=60_000_000_000,
        num_layers=48, num_kv_heads=4, head_dim=128, kv_dtype_bytes=2.0,
        isl=256, osl=256,
    )

    def test_saturates_to_dense_at_high_batch(self):
        # 128 experts, top-8 → activated_fraction = 1-(1-8/128)^B (coupon
        # union). It asymptotes to 1.0 (all experts read == dense) as B
        # grows; at B=512, (0.9375)^512≈4e-15 so the union is dense to
        # machine precision. (At mid batch it stays strictly below dense —
        # see TestMoEUnionUpperBound — which is the bug fix.)
        moe = compute_theoretical_peak_output_tok_per_sec(
            **self._COMMON, concurrency=512,
            num_experts=128, experts_per_tok=8,
            expert_weight_bytes=53_000_000_000,
        )
        dense = compute_theoretical_peak_output_tok_per_sec(
            **self._COMMON, concurrency=512,  # no expert fields -> weight_bytes
        )
        assert moe == pytest.approx(dense, rel=1e-6)

    def test_saturation_lowers_ceiling_vs_constant_active(self):
        # Old behaviour: constant active_weight_bytes regardless of batch.
        old = compute_theoretical_peak_output_tok_per_sec(
            **self._COMMON, concurrency=64,
            active_weight_bytes=10_000_000_000,
        )
        # Fixed: expert union saturates at B=64 -> effective weight ~ full.
        new = compute_theoretical_peak_output_tok_per_sec(
            **self._COMMON, concurrency=64,
            num_experts=128, experts_per_tok=8,
            expert_weight_bytes=53_000_000_000,
        )
        # Larger effective weight -> lower ceiling -> within% rises to a
        # sensible value instead of collapsing.
        assert new < old

    def test_batch1_matches_active(self):
        # At B=1 the saturated weight == non_expert + (k/n)*expert == active.
        non_expert = 60_000_000_000 - 53_000_000_000
        active = non_expert + int((8 / 128) * 53_000_000_000)
        sat = compute_theoretical_peak_output_tok_per_sec(
            **self._COMMON, concurrency=1,
            num_experts=128, experts_per_tok=8,
            expert_weight_bytes=53_000_000_000,
        )
        constant_active = compute_theoretical_peak_output_tok_per_sec(
            **self._COMMON, concurrency=1, active_weight_bytes=active,
        )
        assert sat == pytest.approx(constant_active, rel=1e-3)


class TestMoEUnionUpperBound:
    """The decode ceiling must stay an upper bound on real throughput at
    every batch. Anchored to real InferenceX data: Qwen3-30B-A3B on
    1×MI300X (TP=1, ISL=OSL=1024) measured output_throughput=1754.16 tok/s
    at conc=16. The linear ``min(1, B*k/n)`` expert-union estimate declares
    all 128 experts active at B=16 (16*8/128=1.0), inflating the per-token
    weight read and dropping the ceiling below the measured value
    (within%>100%). The expected union fraction under uniform routing is
    ``1-(1-k/n)^B`` (coupon/occupancy), which keeps the ceiling above the
    measurement."""

    # Real Qwen3-30B-A3B geometry (config.json) + safetensors total_size.
    _NUM_LAYERS = 48
    _NUM_KV_HEADS = 4
    _HEAD_DIM = 128
    _HIDDEN = 2048
    _MOE_INTER = 768
    _NUM_EXPERTS = 128
    _EXPERTS_PER_TOK = 8
    _DTYPE_BYTES = 2.0
    _WEIGHT_BYTES = 61_064_245_248
    _EXPERT_WEIGHT_BYTES = (
        _NUM_LAYERS * _NUM_EXPERTS * 3 * _HIDDEN * _MOE_INTER * int(_DTYPE_BYTES)
    )
    _MEASURED_CONC16_TOK_S = 1754.16  # InferenceX sweep, real measurement

    def _ceiling(self, concurrency: int) -> float:
        return compute_theoretical_peak_output_tok_per_sec(
            gpu_type="mi300x", num_gpus=1,
            weight_bytes=self._WEIGHT_BYTES,
            num_experts=self._NUM_EXPERTS,
            experts_per_tok=self._EXPERTS_PER_TOK,
            expert_weight_bytes=self._EXPERT_WEIGHT_BYTES,
            num_layers=self._NUM_LAYERS,
            num_kv_heads=self._NUM_KV_HEADS,
            head_dim=self._HEAD_DIM,
            kv_dtype_bytes=self._DTYPE_BYTES,
            isl=1024, osl=1024, concurrency=concurrency,
        )

    def test_ceiling_is_upper_bound_at_mid_batch(self):
        # conc=16: real 1754 tok/s must not exceed the theoretical ceiling.
        # Linear union-bound gives ~1336 (<1754) and FAILS; the coupon
        # union gives ~1980 (>1754) and PASSES.
        ceiling = self._ceiling(16)
        assert ceiling >= self._MEASURED_CONC16_TOK_S, (
            f"ceiling {ceiling:.0f} < measured "
            f"{self._MEASURED_CONC16_TOK_S} at conc=16 — roofline upper "
            "bound violated (expert union over-saturated)"
        )

    def test_union_fraction_below_one_at_mid_batch(self):
        # At B=16 the union of activated experts is ~64% (coupon), not the
        # 100% the linear bound assumes; back it out from the ceiling and
        # check it sits strictly below the dense (all-experts) ceiling.
        dense = compute_theoretical_peak_output_tok_per_sec(
            gpu_type="mi300x", num_gpus=1,
            weight_bytes=self._WEIGHT_BYTES,
            num_layers=self._NUM_LAYERS,
            num_kv_heads=self._NUM_KV_HEADS,
            head_dim=self._HEAD_DIM,
            kv_dtype_bytes=self._DTYPE_BYTES,
            isl=1024, osl=1024, concurrency=16,
        )
        # Un-saturated union → smaller effective weight → higher ceiling.
        assert self._ceiling(16) > dense


class TestHWSpecsTable:
    def test_mi_series_present(self):
        for key in ("mi300x", "mi325x", "mi355x"):
            assert key in HW_SPECS, f"missing {key} in HW_SPECS"
            spec = HW_SPECS[key]
            assert spec["hbm_bw_gbps"] > 0
            assert spec["hbm_gb"] > 0

    def test_bw_monotonic_across_generations(self):
        # MI300X (HBM3) < MI325X (HBM3e) < MI355X (HBM3e refresh).
        assert (
            HW_SPECS["mi300x"]["hbm_bw_gbps"]
            < HW_SPECS["mi325x"]["hbm_bw_gbps"]
            < HW_SPECS["mi355x"]["hbm_bw_gbps"]
        )

    def test_peak_tflops_present_for_supported_precisions(self):
        # Every entry must carry a peak_tflops dict with at least bf16.
        for key in ("mi300x", "mi325x", "mi355x"):
            tbl = HW_SPECS[key].get("peak_tflops")
            assert isinstance(tbl, dict)
            assert tbl.get("bf16", 0) > 0
            assert tbl.get("fp8", 0) > 0

    def test_mi355x_doubles_mi300x_bf16(self):
        # CDNA4 ≈ 2× CDNA3 matrix peak at the same precision.
        m3 = HW_SPECS["mi300x"]["peak_tflops"]["bf16"]
        m5 = HW_SPECS["mi355x"]["peak_tflops"]["bf16"]
        assert 1.8 < m5 / m3 < 2.2

    def test_mi325x_compute_equals_mi300x(self):
        # MI325X reuses the CDNA3 die — same matrix peak, larger HBM only.
        assert (
            HW_SPECS["mi300x"]["peak_tflops"]["bf16"]
            == HW_SPECS["mi325x"]["peak_tflops"]["bf16"]
        )


# ---------------------------------------------------------------------------
# Two-sided roofline: T_cmp formula + RooflineBreakdown classification.
# ---------------------------------------------------------------------------
class TestResolvePeakTFLOPS:
    """``_resolve_peak_tflops`` lookup with safe degrade on miss."""

    def test_hits_known_mi355x_bf16(self):
        assert _resolve_peak_tflops("mi355x", "bf16") == 2516.6

    def test_alias_bfloat16_same_as_bf16(self):
        assert (
            _resolve_peak_tflops("mi355x", "bfloat16")
            == _resolve_peak_tflops("mi355x", "bf16")
        )

    def test_mi300x_fp8_dense(self):
        # Dense FP8 (not sparse-doubled).
        assert _resolve_peak_tflops("mi300x", "fp8") == 2614.9

    def test_unknown_gpu_zero(self):
        assert _resolve_peak_tflops("nvidia_h100", "bf16") == 0.0

    def test_unknown_precision_zero(self):
        # MI300X does not support MXFP4 — table miss must degrade to 0.
        assert _resolve_peak_tflops("mi300x", "mxfp4") == 0.0

    def test_empty_inputs_zero(self):
        assert _resolve_peak_tflops("", "bf16") == 0.0
        assert _resolve_peak_tflops("mi355x", "") == 0.0


class TestComputeBoundCeiling:
    """T_cmp = (F_peak * G * dtype_bytes) / (2 * active_weight_bytes_B1).

    The divisor is the **B=1 active** weight, not a batch-saturated value
    — per-token compute is batch-invariant even when memory traffic is
    not (see helper docstring for the physical justification).
    """

    def test_matches_hand_calculation_mi355x_bf16_a3b_like(self):
        # 095726Z-style A3B: 2516.6 TFLOPS * 1 GPU * 2 bytes / (2 * 6.7 GB)
        # = 5033.2e12 / 13.4e9 ≈ 375 600 tok/s.
        cmp = compute_compute_bound_ceiling_tok_per_sec(
            gpu_type="mi355x", num_gpus=1, precision_tag="bf16",
            active_weight_bytes=6_700_000_000,
            weight_bytes=61_000_000_000,
            weight_dtype_bytes=2.0,
        )
        assert cmp == pytest.approx(375_611.94, rel=1e-3)

    def test_scales_linearly_with_num_gpus(self):
        common = dict(
            gpu_type="mi355x", precision_tag="bf16",
            active_weight_bytes=10_000_000_000,
            weight_bytes=10_000_000_000, weight_dtype_bytes=2.0,
        )
        one = compute_compute_bound_ceiling_tok_per_sec(num_gpus=1, **common)
        eight = compute_compute_bound_ceiling_tok_per_sec(num_gpus=8, **common)
        assert eight == pytest.approx(8 * one, rel=1e-9)

    def test_dense_uses_weight_bytes_when_active_missing(self):
        # active_weight_bytes=0 must fall back to weight_bytes (dense).
        cmp_zero = compute_compute_bound_ceiling_tok_per_sec(
            gpu_type="mi355x", num_gpus=1, precision_tag="bf16",
            active_weight_bytes=0,
            weight_bytes=10_000_000_000, weight_dtype_bytes=2.0,
        )
        cmp_explicit = compute_compute_bound_ceiling_tok_per_sec(
            gpu_type="mi355x", num_gpus=1, precision_tag="bf16",
            active_weight_bytes=10_000_000_000,
            weight_bytes=10_000_000_000, weight_dtype_bytes=2.0,
        )
        assert cmp_zero == pytest.approx(cmp_explicit, rel=1e-9)

    def test_moe_uses_active_b1_not_batch_saturated(self):
        # Anti-regression: if a future refactor reroutes T_cmp through a
        # batch-saturated effective_weight, the divisor would jump ~10x
        # on this MoE shape (active=6.7G vs weight=61G) and T_cmp would
        # drop ~10x. Lock the helper to active_weight_bytes.
        cmp_active = compute_compute_bound_ceiling_tok_per_sec(
            gpu_type="mi355x", num_gpus=1, precision_tag="bf16",
            active_weight_bytes=6_700_000_000,
            weight_bytes=61_000_000_000, weight_dtype_bytes=2.0,
        )
        cmp_full = compute_compute_bound_ceiling_tok_per_sec(
            gpu_type="mi355x", num_gpus=1, precision_tag="bf16",
            active_weight_bytes=61_000_000_000,
            weight_bytes=61_000_000_000, weight_dtype_bytes=2.0,
        )
        assert cmp_active > 8 * cmp_full

    def test_zero_on_unknown_gpu(self):
        assert compute_compute_bound_ceiling_tok_per_sec(
            gpu_type="h100", num_gpus=1, precision_tag="bf16",
            active_weight_bytes=10_000_000_000,
            weight_bytes=10_000_000_000, weight_dtype_bytes=2.0,
        ) == 0.0

    def test_zero_on_unknown_precision(self):
        # MXFP4 on MI300X — not in HW_SPECS, must degrade.
        assert compute_compute_bound_ceiling_tok_per_sec(
            gpu_type="mi300x", num_gpus=1, precision_tag="mxfp4",
            active_weight_bytes=10_000_000_000,
            weight_bytes=10_000_000_000, weight_dtype_bytes=0.5,
        ) == 0.0

    def test_zero_on_zero_dtype_bytes(self):
        assert compute_compute_bound_ceiling_tok_per_sec(
            gpu_type="mi355x", num_gpus=1, precision_tag="bf16",
            active_weight_bytes=10_000_000_000,
            weight_bytes=10_000_000_000, weight_dtype_bytes=0.0,
        ) == 0.0


class TestRooflineBreakdownClassification:
    """``compute_roofline_breakdown_from_state`` integration: routes the
    correct ``bound_kind`` for every (mem, cmp) ordering and degrades
    safely when one or both ceilings are unavailable."""

    def _mock_state_and_helpers(self, monkeypatch, mem_val, cmp_val):
        """Stub out ``load_model_meta`` + both ceilings so we control
        the (mem, cmp) ordering directly without standing up a
        synthetic HF model."""
        from inference_optimizer.orchestrator import roofline_ceiling
        meta = ModelMeta(
            weight_bytes=10_000_000_000, num_layers=48, num_kv_heads=4,
            head_dim=128, weight_dtype_bytes=2.0,
            active_weight_bytes=5_000_000_000,
        )
        monkeypatch.setattr(
            roofline_ceiling, "load_model_meta", lambda *a, **kw: meta
        )
        monkeypatch.setattr(
            roofline_ceiling,
            "compute_theoretical_peak_output_tok_per_sec",
            lambda **kw: mem_val,
        )
        monkeypatch.setattr(
            roofline_ceiling,
            "compute_compute_bound_ceiling_tok_per_sec",
            lambda **kw: cmp_val,
        )
        return SimpleNamespace(
            model_path="/fake", gpu_type="mi355x", tp=1,
            precision="bf16", conc=8, isl=256, osl=256,
            last_baseline={},
        )

    def test_memory_bound_when_cmp_higher(self, monkeypatch):
        state = self._mock_state_and_helpers(monkeypatch, 8000.0, 40_000.0)
        br = compute_roofline_breakdown_from_state(state)
        assert br.mem_tok_per_sec == 8000.0
        assert br.cmp_tok_per_sec == 40_000.0
        assert br.peak_tok_per_sec == 8000.0
        assert br.bound_kind == "memory"

    def test_compute_bound_when_cmp_lower(self, monkeypatch):
        state = self._mock_state_and_helpers(monkeypatch, 8000.0, 2000.0)
        br = compute_roofline_breakdown_from_state(state)
        assert br.peak_tok_per_sec == 2000.0
        assert br.bound_kind == "compute"

    def test_unknown_when_both_zero(self, monkeypatch):
        state = self._mock_state_and_helpers(monkeypatch, 0.0, 0.0)
        br = compute_roofline_breakdown_from_state(state)
        assert br.peak_tok_per_sec == 0.0
        assert br.bound_kind == "unknown"

    def test_degrades_to_memory_when_cmp_unavailable(self, monkeypatch):
        # T_cmp == 0 typically means precision missing from HW_SPECS;
        # the result must keep T_mem visible as the ceiling and label
        # the side memory-bound (matches pre-PR behaviour).
        state = self._mock_state_and_helpers(monkeypatch, 8000.0, 0.0)
        br = compute_roofline_breakdown_from_state(state)
        assert br.peak_tok_per_sec == 8000.0
        assert br.cmp_tok_per_sec == 0.0
        assert br.bound_kind == "memory"

    def test_compute_only_when_mem_zero(self, monkeypatch):
        state = self._mock_state_and_helpers(monkeypatch, 0.0, 2000.0)
        br = compute_roofline_breakdown_from_state(state)
        assert br.peak_tok_per_sec == 2000.0
        assert br.bound_kind == "compute"

    def test_backward_compat_compute_peak_from_state_returns_min(
        self, monkeypatch,
    ):
        # Old API must keep returning a scalar float == breakdown.peak.
        state = self._mock_state_and_helpers(monkeypatch, 8000.0, 2000.0)
        assert compute_peak_from_state(state) == 2000.0

    def test_returns_empty_breakdown_on_missing_model(self, monkeypatch):
        from inference_optimizer.orchestrator import roofline_ceiling
        monkeypatch.setattr(
            roofline_ceiling, "load_model_meta", lambda *a, **kw: None
        )
        state = SimpleNamespace(model_path="", gpu_type="mi355x", tp=1)
        br = compute_roofline_breakdown_from_state(state)
        assert br.peak_tok_per_sec == 0.0
        assert br.bound_kind == "unknown"


class TestPhysicalInterpretation095726Z:
    """End-to-end anchor for the formula change using real session
    parameters from 20260528T095726Z (Qwen3-30B-A3B, MI355X, bf16,
    yaml CONC=64, ISL=OSL=256, achieved output_throughput=6244.3 tok/s).

    Pins the within% interpretation: decode-stage MoE at conc=64 stays
    memory-bound, so adding T_cmp must NOT change T_peak — the within%
    that PR-9749520 anchored at 77.4 % is preserved end-to-end."""

    _A3B_META = dict(
        gpu_type="mi355x", num_gpus=1, precision_tag="bf16",
        weight_bytes=61_000_000_000,
        active_weight_bytes=6_700_000_000,
        weight_dtype_bytes=2.0,
    )
    _ACHIEVED_TOK_S = 6244.3

    def test_t_cmp_is_far_above_t_mem_at_decode(self):
        cmp = compute_compute_bound_ceiling_tok_per_sec(**self._A3B_META)
        mem = compute_theoretical_peak_output_tok_per_sec(
            gpu_type="mi355x", num_gpus=1,
            weight_bytes=61_000_000_000,
            active_weight_bytes=6_700_000_000,
            num_experts=128, experts_per_tok=8,
            expert_weight_bytes=58_000_000_000,
            num_layers=48, num_kv_heads=4, head_dim=128,
            kv_dtype_bytes=2.0, isl=256, osl=256, concurrency=64,
        )
        # T_cmp ≈ 375 612 ; T_mem ≈ 8 065 ; ratio ~ 46×.
        assert cmp / mem > 10

    def test_breakdown_stays_memory_bound_with_cmp_visible(self, tmp_path):
        # Stand up a minimal HF-like model dir so load_model_meta works
        # without monkeypatching, exercising the full code path.
        _write_synthetic_model(
            tmp_path / "a3b",
            total_size=61_000_000_000,
            num_layers=48, num_kv_heads=4,
            hidden_size=2048, num_attention_heads=32,
            torch_dtype="bfloat16",
            num_experts=128, num_experts_per_tok=8,
            moe_intermediate_size=768,
        )
        yaml_path = tmp_path / "bl.yaml"
        yaml_path.write_text(
            "benchmark:\n  envs:\n    CONC: 64\n", encoding="utf-8",
        )
        state = SimpleNamespace(
            model_path=str(tmp_path / "a3b"),
            gpu_type="mi355x", tp=1, precision="bf16",
            conc=8,  # stale SharedState default; yaml CONC=64 wins (P0).
            isl=256, osl=256,
            last_baseline={"extras": {"materialized_config": str(yaml_path)}},
        )
        br = compute_roofline_breakdown_from_state(state)
        # T_cmp must be present (formula change requirement) AND must
        # NOT cap the ceiling at decode — bound stays memory.
        assert br.cmp_tok_per_sec > br.mem_tok_per_sec
        # PerfModel (FusedMoE coupon formula) now drives peak_tok_per_sec;
        # it agrees with the legacy T_mem ceiling within < 1% here, so the
        # within% anchor remains valid. The peak must stay > measured.
        assert br.bound_kind == "memory"
        assert br.peak_tok_per_sec >= self._ACHIEVED_TOK_S
        # within% recomputed from achieved must still match the
        # PR-9749520 anchor (~77 %, fixed by P0+P1 ceiling correctness).
        within_pct = 100.0 * self._ACHIEVED_TOK_S / br.peak_tok_per_sec
        assert 70.0 < within_pct < 85.0


class TestDeepSeekV3ConfigAliases:
    """DeepSeek-V3-derived models (DeepSeek V3 / GigaChat3.1-A1.8B …)
    use HF config aliases the original Qwen-A3B path didn't cover:

      * ``n_routed_experts`` instead of ``num_experts``
      * ``quantization_config.quant_method = "fp8"`` with bf16
        activation dtype recorded in the legacy ``dtype`` field.

    Without the aliases the helper treats these models as dense and
    keeps the un-shrunken ``weight_bytes`` divisor (over-counts active
    params ~10x at decode), AND mis-reads the activation dtype as the
    weight dtype (over-counts byte size ~2x). This test pins both
    routes so DeepSeek-V3 family ceilings stay accurate."""

    def test_n_routed_experts_alias_equivalent_to_num_experts(self, tmp_path):
        # Two model dirs, identical geometry, differ only in the field
        # name used for the routed-expert count. The breakdown must be
        # identical — anti-regression for the alias resolver.
        # ``total_size`` is the bf16-equivalent (~26 GB so the expert
        # pool ~19.6 GB fits inside; real GigaChat ships as fp8 ≈12.3 GB
        # but this test isolates the alias logic from the dtype path).
        common = dict(
            total_size=26_000_000_000,
            num_layers=26, num_kv_heads=32,
            hidden_size=1536, num_attention_heads=32,
            torch_dtype="bfloat16",
            num_experts_per_tok=4,
            moe_intermediate_size=1280,
        )
        _write_synthetic_model(
            tmp_path / "qwen_alias", num_experts=64, **common
        )
        _write_synthetic_model(
            tmp_path / "ds_alias", n_routed_experts=64, **common
        )
        qwen = load_model_meta(str(tmp_path / "qwen_alias"))
        ds = load_model_meta(str(tmp_path / "ds_alias"))
        assert qwen is not None and ds is not None
        # Identical MoE decomposition for both alias variants.
        assert qwen.num_experts == ds.num_experts == 64
        assert qwen.experts_per_tok == ds.experts_per_tok == 4
        assert qwen.expert_weight_bytes == ds.expert_weight_bytes
        assert qwen.active_weight_bytes == ds.active_weight_bytes

    def test_fp8_quantization_config_drives_weight_dtype(self, tmp_path):
        # GigaChat3.1-A1.8B fingerprint: quant_method=fp8 + dtype=bfloat16
        # (the bf16 is the *activation* dtype, weights are block fp8).
        # Without the quant_method short-circuit ``load_model_meta``
        # picks up the bf16 dtype and over-counts weight bytes 2x.
        _write_synthetic_model(
            tmp_path / "gigachat",
            total_size=12_000_000_000,
            num_layers=26, num_kv_heads=32,
            hidden_size=1536, num_attention_heads=32,
            torch_dtype="bfloat16",  # HF-standard activation dtype
            dtype="bfloat16",  # DeepSeek-V3 redundancy
            quant_method="fp8",  # block-fp8 weight quant
            n_routed_experts=64,
            num_experts_per_tok=4,
            moe_intermediate_size=1280,
        )
        meta = load_model_meta(str(tmp_path / "gigachat"))
        assert meta is not None
        # fp8 -> 1 byte per param (not 2 from the bf16 activation dtype).
        assert meta.weight_dtype_bytes == 1.0
        # MoE decomposition still fires via n_routed_experts.
        assert meta.num_experts == 64
        assert meta.experts_per_tok == 4
        assert meta.expert_weight_bytes > 0

    def test_quant_method_overrides_torch_dtype(self, tmp_path):
        # quant_method must win even when torch_dtype is set to
        # something inconsistent — the safetensors data is fp8 once
        # the quant config says so.
        _write_synthetic_model(
            tmp_path / "fp8_model",
            total_size=12_000_000_000,
            num_layers=4, num_kv_heads=8,
            hidden_size=1024, num_attention_heads=16,
            torch_dtype="bfloat16",  # would yield 2.0 if read directly
            quant_method="fp8",
        )
        meta = load_model_meta(str(tmp_path / "fp8_model"))
        assert meta is not None
        assert meta.weight_dtype_bytes == 1.0  # fp8 wins

    def test_dtype_field_fallback_when_torch_dtype_missing(self, tmp_path):
        # No torch_dtype, no quant_method — the DeepSeek-style ``dtype``
        # field should still be consulted before the precision_hint.
        _write_synthetic_model(
            tmp_path / "ds_no_qd",
            total_size=4_000_000_000,
            num_layers=4, num_kv_heads=4,
            hidden_size=512, num_attention_heads=8,
            torch_dtype="bfloat16",
        )
        # Strip torch_dtype manually so we exercise the dtype fallback.
        import json as _json
        cfg_path = tmp_path / "ds_no_qd" / "config.json"
        cfg = _json.loads(cfg_path.read_text())
        cfg.pop("torch_dtype", None)
        cfg["dtype"] = "float16"
        cfg_path.write_text(_json.dumps(cfg))
        meta = load_model_meta(str(tmp_path / "ds_no_qd"))
        assert meta is not None
        assert meta.weight_dtype_bytes == 2.0  # fp16

    def test_mxfp4_alias_in_dtype_table(self):
        # HW_SPECS uses ``mxfp4`` as a key for MI355X; the byte-size
        # table must agree (0.5 byte/param), otherwise the active-
        # params arithmetic in compute_compute_bound_ceiling_tok_per_sec
        # would silently fall back to bf16.
        assert _resolve_dtype_bytes("mxfp4") == 0.5

    def test_num_local_experts_alias_equivalent_to_num_experts(self, tmp_path):
        # gpt-oss family (GptOssForCausalLM) writes the routed-expert
        # count under ``num_local_experts``. Without the alias the helper
        # degraded to dense on gpt-oss-120b and inflated the active-
        # weight divisor ~13x (full 65 GB instead of routed 5 GB),
        # collapsing within% to ~28% on a memory-bound decode that
        # should sit ~70-80%. Two synthetic configs identical except
        # for the field name must produce identical ModelMeta.
        # bf16-equiv total_size has to be larger than the routed pool
        # (128 experts × 36 layers × 3 × 2880 × 2880 × 2 bytes ≈ 229 GB),
        # otherwise ``_compute_expert_decomposition`` triggers its
        # "expert pool ≥ weight_bytes" safe-degrade and the test would
        # falsely fail. Real gpt-oss-120b ships as mxfp4 (~65 GB on
        # disk); the bf16 equivalent is ~260 GB, which we use here so
        # the MoE decomposition path executes.
        common = dict(
            total_size=260_000_000_000,
            num_layers=36, num_kv_heads=8,
            hidden_size=2880, num_attention_heads=64,
            torch_dtype="bfloat16",
            num_experts_per_tok=4,
            moe_intermediate_size=2880,
        )
        _write_synthetic_model(
            tmp_path / "qwen_alias", num_experts=128, **common
        )
        _write_synthetic_model(
            tmp_path / "gptoss_alias", num_local_experts=128, **common
        )
        qwen = load_model_meta(str(tmp_path / "qwen_alias"))
        gptoss = load_model_meta(str(tmp_path / "gptoss_alias"))
        assert qwen is not None and gptoss is not None
        assert qwen.num_experts == gptoss.num_experts == 128
        assert qwen.experts_per_tok == gptoss.experts_per_tok == 4
        assert qwen.expert_weight_bytes == gptoss.expert_weight_bytes
        assert qwen.active_weight_bytes == gptoss.active_weight_bytes
        # And both must be < weight_bytes (MoE shrinks the divisor).
        assert qwen.active_weight_bytes < qwen.weight_bytes


# ---------------------------------------------------------------------------
# PerfModel bottom-up breakdown (Phase 2 / 3).
# ---------------------------------------------------------------------------

class TestPerfModelBreakdown:
    """Smoke tests for compute_roofline_from_perfmodel and related helpers."""

    def _make_meta(self) -> "ModelMeta":
        """Minimal dense Llama-style ModelMeta."""
        from inference_optimizer.orchestrator.roofline_ceiling import ModelMeta
        return ModelMeta(
            weight_bytes=int(13e9),   # ~13 GB weight
            num_layers=32,
            num_kv_heads=8,
            head_dim=128,
            weight_dtype_bytes=2.0,
            active_weight_bytes=int(13e9),
            hidden_size=4096,
            intermediate_size=11008,
            vocab_size=32000,
            num_attention_heads=32,
        )

    def test_returns_none_for_unknown_gpu(self):
        from inference_optimizer.orchestrator.roofline_ceiling import (
            compute_roofline_from_perfmodel,
        )
        meta = self._make_meta()
        result = compute_roofline_from_perfmodel(
            meta=meta, gpu_type="unknown_gpu_xyz",
            concurrency=8, isl=1024, osl=512,
        )
        assert result is None

    def test_returns_none_when_meta_missing_hidden(self):
        from inference_optimizer.orchestrator.roofline_ceiling import (
            ModelMeta, compute_roofline_from_perfmodel,
        )
        meta = ModelMeta(
            weight_bytes=int(13e9), num_layers=32,
            num_kv_heads=8, head_dim=128, weight_dtype_bytes=2.0,
            # hidden_size=0 -> insufficient
        )
        result = compute_roofline_from_perfmodel(
            meta=meta, gpu_type="mi300x",
            concurrency=8, isl=1024, osl=512,
        )
        assert result is None

    def test_hw_specs_achievable_coverage(self):
        from inference_optimizer.orchestrator.roofline_ceiling import (
            HW_SPECS_ACHIEVABLE, _resolve_achievable_tflops,
        )
        for gpu in ("mi300x", "mi325x", "mi355x"):
            assert gpu in HW_SPECS_ACHIEVABLE
        # MI300X bf16 achievable must be less than vendor-quoted (708 < 1307)
        assert _resolve_achievable_tflops("mi300x", "bf16") == 708.0
        assert _resolve_achievable_tflops("mi355x", "bf16") == 1686.0
        assert _resolve_achievable_tflops("mi300x", "fp8") == 1273.0

    def test_perfmodel_breakdown_when_traceLens_available(self):
        """When model metadata is complete the result is a valid PerfModelBreakdown."""
        from inference_optimizer.orchestrator.roofline_ceiling import (
            compute_roofline_from_perfmodel,
            PerfModelBreakdown,
        )

        meta = self._make_meta()
        result = compute_roofline_from_perfmodel(
            meta=meta, gpu_type="mi300x",
            concurrency=16, isl=1024, osl=1024,
        )
        assert result is not None
        assert isinstance(result, PerfModelBreakdown)
        # Decode ceiling must be positive
        assert result.decode_tok_per_s > 0
        # Prefill is compute-bound for S=1024, so also > 0
        assert result.prefill_tok_per_s > 0
        # Must have ops (at least q/k/v/o + sdpa)
        assert len(result.ops) >= 5
        # Achievable peaks must match MI300X bf16
        assert result.peak_achievable_tflops == pytest.approx(708.0 * 1)
        assert result.hbm_bw_gbps == pytest.approx(5300.0)
        # All ops have non-negative pct_time summing to ~1
        assert all(0.0 <= op.pct_time <= 1.0 + 1e-6 for op in result.ops)
        total_pct = sum(op.pct_time for op in result.ops)
        assert total_pct == pytest.approx(1.0, abs=1e-6)
        # bound_kind must be one of the valid values
        assert result.bound_kind in ("memory", "compute", "unknown")

    def test_v2_breakdown_returns_tuple(self):
        """compute_roofline_breakdown_from_state_v2 returns (legacy, pm_or_None)."""
        from types import SimpleNamespace
        from inference_optimizer.orchestrator.roofline_ceiling import (
            compute_roofline_breakdown_from_state_v2,
            RooflineBreakdown,
        )
        state = SimpleNamespace(
            model_path="", gpu_type="mi300x", tp=1,
            conc=8, isl=256, osl=256, precision="bf16",
            last_baseline={},
        )
        legacy, pm_bd = compute_roofline_breakdown_from_state_v2(state)
        assert isinstance(legacy, RooflineBreakdown)
        # pm_bd is None because model_path is empty (no config.json)
        assert pm_bd is None


# ---------------------------------------------------------------------------
# PerfModel MoE formula correctness
# ---------------------------------------------------------------------------

class TestPerfModelMoE:
    """Verify that compute_roofline_from_perfmodel uses moe_intermediate_size
    for MoE models and gives a valid upper bound on measured throughput."""

    def _make_qwen3_a3b_meta(self) -> "ModelMeta":
        """Qwen3-30B-A3B geometry: 128 experts, top-8, moe_inter=768."""
        from inference_optimizer.orchestrator.roofline_ceiling import ModelMeta
        return ModelMeta(
            weight_bytes=61_064_245_248,
            num_layers=48,
            num_kv_heads=4,
            head_dim=128,
            weight_dtype_bytes=2.0,
            active_weight_bytes=6_700_000_000,
            num_experts=128,
            experts_per_tok=8,
            expert_weight_bytes=int(48 * 128 * 3 * 2048 * 768 * 2),
            hidden_size=2048,
            intermediate_size=6144,  # present in config but not used for MoE FFN
            moe_intermediate_size=768,
            vocab_size=151936,
            num_attention_heads=32,
        )

    def test_moe_ffn_uses_moe_intermediate_size(self):
        """With moe_intermediate_size set, PerfModel uses FusedMoE formula
        (coupon E_active) not the per-token fixed-topk formula."""
        from inference_optimizer.orchestrator.roofline_ceiling import (
            compute_roofline_from_perfmodel,
        )
        meta = self._make_qwen3_a3b_meta()
        result = compute_roofline_from_perfmodel(
            meta=meta, gpu_type="mi300x",
            concurrency=1, isl=1024, osl=1024,
        )
        assert result is not None
        # At batch=1: E_active ≈ topk = 8 (coupon formula == B=1 estimate),
        # so decode is memory-bound (weight IO dominates).
        assert result.bound_kind == "memory"
        # The "moe_fused" op should appear in the op breakdown
        moe_ops = [op for op in result.ops if op.name == "moe_fused"]
        assert len(moe_ops) == 1

    def test_moe_ceiling_is_upper_bound_conc16(self):
        """PerfModel (FusedMoE coupon formula) ceiling for Qwen3-30B-A3B at conc=16
        must be >= 1754 tok/s (real InferenceX measurement on MI300X)."""
        from inference_optimizer.orchestrator.roofline_ceiling import (
            compute_roofline_from_perfmodel,
        )
        meta = self._make_qwen3_a3b_meta()
        result = compute_roofline_from_perfmodel(
            meta=meta, gpu_type="mi300x",
            concurrency=16, isl=1024, osl=1024,
        )
        assert result is not None
        assert result.decode_tok_per_s >= 1754.16, (
            f"PerfModel ceiling {result.decode_tok_per_s:.1f} < measured 1754.16 tok/s"
        )

    def test_moe_ceiling_is_upper_bound_conc64(self):
        """At conc=64, FusedMoE coupon formula should give a tighter (lower) ceiling
        than the B=1 per-token estimate because E_active grows toward num_experts."""
        from inference_optimizer.orchestrator.roofline_ceiling import (
            ModelMeta, compute_roofline_from_perfmodel,
        )
        meta = self._make_qwen3_a3b_meta()
        result_1 = compute_roofline_from_perfmodel(
            meta=meta, gpu_type="mi300x", concurrency=1, isl=512, osl=512,
        )
        result_64 = compute_roofline_from_perfmodel(
            meta=meta, gpu_type="mi300x", concurrency=64, isl=512, osl=512,
        )
        assert result_1 is not None and result_64 is not None
        # At conc=64, more expert weights are loaded (E_active > topk),
        # so bytes grow → ceiling drops vs conc=1.
        assert result_64.decode_tok_per_s < result_1.decode_tok_per_s * 64, (
            "conc=64 ceiling should not be 64× higher than conc=1 (MoE weight saturation)"
        )


class TestPerfModelTransparentReplacement:
    """compute_roofline_breakdown_from_state must use PerfModel peak when
    model config is complete, and fall back to legacy when it is not."""

    def test_uses_perfmodel_peak_when_config_available(self, tmp_path):
        """For a known GPU + complete config, peak comes from PerfModel."""
        from types import SimpleNamespace
        from inference_optimizer.orchestrator.roofline_ceiling import (
            compute_roofline_breakdown_from_state,
            compute_roofline_from_perfmodel,
            load_model_meta,
        )
        _write_synthetic_model(
            tmp_path / "m",
            total_size=int(13e9),
            num_layers=32,
            num_kv_heads=8,
            hidden_size=4096,
            num_attention_heads=32,
            torch_dtype="bfloat16",
        )
        state = SimpleNamespace(
            model_path=str(tmp_path / "m"),
            gpu_type="mi300x", tp=1, precision="bf16",
            conc=8, isl=256, osl=256,
            last_baseline={},
        )
        br = compute_roofline_breakdown_from_state(state)
        meta = load_model_meta(str(tmp_path / "m"))
        pm_bd = compute_roofline_from_perfmodel(
            meta=meta, gpu_type="mi300x", concurrency=8,
            isl=256, osl=256, precision_tag="bf16",
        )
        assert pm_bd is not None
        assert br.peak_tok_per_sec == pytest.approx(pm_bd.decode_tok_per_s, rel=1e-6)
        assert br.bound_kind == pm_bd.bound_kind

    def test_moe_uses_perfmodel_peak(self, tmp_path):
        """MoE models now also use PerfModel (FusedMoE coupon formula).
        For Qwen3-30B-A3B at conc=64, the FusedMoE ceiling must be above
        the measured 6244 tok/s (from TestPhysicalInterpretation095726Z)."""
        from types import SimpleNamespace
        from inference_optimizer.orchestrator.roofline_ceiling import (
            compute_roofline_breakdown_from_state,
            compute_roofline_from_perfmodel,
            load_model_meta,
        )
        _write_qwen3_moe_model(tmp_path / "a3b", total_size=61_000_000_000)
        yaml_path = tmp_path / "bl.yaml"
        yaml_path.write_text("benchmark:\n  envs:\n    CONC: 64\n", encoding="utf-8")
        state = SimpleNamespace(
            model_path=str(tmp_path / "a3b"),
            gpu_type="mi355x", tp=1, precision="bf16",
            conc=8, isl=256, osl=256,
            last_baseline={"extras": {"materialized_config": str(yaml_path)}},
        )
        br = compute_roofline_breakdown_from_state(state)
        meta = load_model_meta(str(tmp_path / "a3b"))
        pm_bd = compute_roofline_from_perfmodel(
            meta=meta, gpu_type="mi355x", concurrency=64,
            isl=256, osl=256, precision_tag="bf16",
        )
        assert pm_bd is not None
        assert br.peak_tok_per_sec == pytest.approx(pm_bd.decode_tok_per_s, rel=1e-6)
        # FusedMoE ceiling must remain above the measured 6244.3 tok/s
        assert br.peak_tok_per_sec >= 6244.3, (
            f"MoE PerfModel ceiling {br.peak_tok_per_sec:.1f} < measured 6244.3 tok/s"
        )

    def test_falls_back_to_legacy_for_unknown_gpu(self, tmp_path):
        """For an unknown GPU (PerfModel returns None), legacy ceiling is used."""
        from types import SimpleNamespace
        from inference_optimizer.orchestrator.roofline_ceiling import (
            compute_roofline_breakdown_from_state,
        )
        _write_synthetic_model(
            tmp_path / "m",
            total_size=int(13e9),
            num_layers=32,
            num_kv_heads=8,
            hidden_size=4096,
            num_attention_heads=32,
            torch_dtype="bfloat16",
        )
        state = SimpleNamespace(
            model_path=str(tmp_path / "m"),
            gpu_type="unknown_gpu_xyz", tp=1, precision="bf16",
            conc=8, isl=256, osl=256,
            last_baseline={},
        )
        # Legacy ceiling for unknown GPU also returns 0 (no HW spec), so just
        # verify the function returns without raising.
        br = compute_roofline_breakdown_from_state(state)
        assert br.peak_tok_per_sec >= 0.0

