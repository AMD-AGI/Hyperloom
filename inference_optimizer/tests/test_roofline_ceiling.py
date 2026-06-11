# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Tests for ``orchestrator.roofline_ceiling`` (formula correctness, graceful degrade, HF metadata parsing)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from inference_optimizer.orchestrator.roofline_ceiling import (
    HW_SPECS,
    ModelMeta,
    _resolve_dtype_bytes,
    _resolve_peak_tflops,
    apply_runtime_dtype,
    compute_compute_bound_ceiling_tok_per_sec,
    compute_kv_bytes_per_token,
    compute_peak_from_state,
    compute_roofline_breakdown_from_state,
    compute_theoretical_peak_output_tok_per_sec,
    load_model_meta,
    resolve_runtime_dtype,
    resolve_runtime_workload,
)


# Formula correctness against published numbers.
class TestPeakFormulaAgainstPublishedNumbers:
    """Worked examples from ITK Research and arXiv 2402.16363 (KV-cache treated as negligible via num_layers=0)."""

    def test_itk_b200_70b_fp8_matches_cited_114_tok_s(self, monkeypatch):
        # ITK Research B200: 8e12 / 70e9 ≈ 114.28 tok/s. Inject the B200 spec for this test only.
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


# MI300X realistic sanity (Llama-70B-style dense model).
class TestMI300XRealistic:
    """Llama-3-70B BF16 on 1×MI300X: single-stream decode ≈ 30-40 tok/s (5.3 TB/s / 140 GB ≈ 37.8)."""

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
        """With small isl/osl, batching N requests amortizes weight reads N× so peak is near-linearly higher."""
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
        # 20× concurrency gives >10× peak (KV term doesn't amortize, so not exactly 20×).
        assert p20 / max(p1, 1e-9) > 10.0


# Graceful degrade.
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


# Dtype + KV helpers.
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
        # 2 (K+V) × 80 layers × 8 KV heads × 128 head_dim × 2 bytes = 327_680 bytes/token.
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


# HF metadata extraction.
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
    """Lay down a minimal HF-shaped model dir; optional kwargs emit MHA / MoE / quant / alias variants."""
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


# State-driven entry point.
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


# MoE active weight bytes (PR: MoE-aware decode ceiling).
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
    """MoE models route a subset of experts per token; the decode divisor must use the active subset (else within_roofline_pct > 100%)."""

    def test_qwen3_30b_a3b_active_is_small_fraction_of_total(self, tmp_path):
        """Qwen3-30B-A3B: 128 experts, 8 active → active weight bytes land in the ~8-15% range of total."""
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
        """Dense (no num_experts) → active = total."""
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
        """Same total bytes, MoE active routing → ceiling several × higher than naive dense (fixes within_roofline_pct > 100%)."""
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
        """Backward-compat: active_weight_bytes=0 behaves as before (uses weight_bytes)."""
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
        """When computed expert bytes >= total_size, the helper clamps to weight_bytes (no negative non_expert_bytes)."""
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


# HW_SPECS table sanity.
class TestResolveEffectiveConcurrency:
    """Concurrency fallback chain (PR-A): state.conc -> baseline yaml envs.CONC -> 1."""

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
        """P0 fix: the materialized baseline yaml's ``CONC`` wins over ``state.conc`` (session 095726Z: state stayed at default 8 while the run used 64)."""
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
        """End-to-end: state.conc=0 + yaml envs.CONC=64 → peak computed with batch=64."""
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
        # Drop the yaml so resolution falls through to state.conc=1.
        state.last_baseline = {}
        state.conc = 1
        peak_with_conc_1 = compute_peak_from_state(state)
        # batch=64 amortizes weight reads, so the yaml-resolved peak is much higher.
        assert peak_with_yaml > 10 * peak_with_conc_1


class TestMoEBatchSaturation:
    """P1 fix: the MoE weight-read term grows with batch as activated experts saturate toward all experts (a constant active_weight_bytes over-amortizes at high batch)."""

    _COMMON = dict(
        gpu_type="mi355x", num_gpus=1, weight_bytes=60_000_000_000,
        num_layers=48, num_kv_heads=4, head_dim=128, kv_dtype_bytes=2.0,
        isl=256, osl=256,
    )

    def test_saturates_to_dense_at_high_batch(self):
        # 128 experts top-8: activated_fraction = 1-(1-8/128)^B (coupon union) asymptotes to dense as B grows (~dense at B=512); strictly below dense at mid batch (see TestMoEUnionUpperBound).
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
        # Larger effective weight -> lower ceiling -> within% rises to a sensible value.
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
    """Decode ceiling must stay an upper bound on real throughput at every batch (Qwen3-30B-A3B 1xMI300X measured 1754.16 tok/s at conc=16); the coupon union ``1-(1-k/n)^B`` keeps the ceiling above the measurement where the linear ``min(1,B*k/n)`` bound under-estimates it."""

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


# HW_SPECS table sanity
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


# Two-sided roofline: T_cmp formula + RooflineBreakdown classification.
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
    """T_cmp = (F_peak * G * dtype_bytes) / (2 * active_weight_bytes_B1); the divisor is the B=1 active weight (per-token compute is batch-invariant)."""

    def test_matches_hand_calculation_mi355x_bf16_a3b_like(self):
        # A3B: 2516.6 TFLOPS * 2 bytes / (2 * 6.7 GB) ≈ 375 600 tok/s.
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
        # Anti-regression: T_cmp must divide by active_weight_bytes (6.7G), not a batch-saturated 61G.
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
    """``compute_roofline_breakdown_from_state`` routes the correct ``bound_kind`` for every (mem, cmp) ordering and degrades safely."""

    def _mock_state_and_helpers(self, monkeypatch, mem_val, cmp_val):
        """Stub ``load_model_meta`` + both ceilings to control the (mem, cmp) ordering directly."""
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
        # T_cmp == 0 (precision missing from HW_SPECS) → keep T_mem as the ceiling, label memory-bound.
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

    def test_quantized_legacy_fallback_uses_activation_kv_dtype(self, monkeypatch):
        from inference_optimizer.orchestrator import roofline_ceiling
        meta = ModelMeta(
            weight_bytes=10_000_000_000, num_layers=48, num_kv_heads=4,
            head_dim=128, weight_dtype_bytes=1.0,
            active_weight_bytes=5_000_000_000,
        )
        captured: dict[str, float] = {}

        def _fake_mem(**kw):
            captured["kv_dtype_bytes"] = kw["kv_dtype_bytes"]
            return 8000.0

        monkeypatch.setattr(roofline_ceiling, "load_model_meta", lambda *a, **kw: meta)
        monkeypatch.setattr(
            roofline_ceiling,
            "compute_theoretical_peak_output_tok_per_sec",
            _fake_mem,
        )
        monkeypatch.setattr(
            roofline_ceiling,
            "compute_compute_bound_ceiling_tok_per_sec",
            lambda **kw: 40_000.0,
        )
        state = SimpleNamespace(
            model_path="/fake", gpu_type="mi355x", tp=1,
            precision="fp8", conc=8, isl=256, osl=256,
            last_baseline={},
        )
        br = compute_roofline_breakdown_from_state(state)

        assert br.peak_tok_per_sec == 8000.0
        assert captured["kv_dtype_bytes"] == 2.0

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
    """End-to-end anchor (session 095726Z, Qwen3-30B-A3B/MI355X/bf16/CONC=64): decode-stage MoE stays memory-bound so adding T_cmp doesn't change T_peak (within% ~77%)."""

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
        # Real HF-like model dir so load_model_meta exercises the full code path.
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
        # T_cmp must be present but must NOT cap the ceiling at decode (bound stays memory).
        assert br.cmp_tok_per_sec > br.mem_tok_per_sec
        # PerfModel (FusedMoE coupon formula) now drives peak_tok_per_sec;
        # it agrees with the legacy T_mem ceiling within < 1% here, so the
        # within% anchor remains valid. The peak must stay > measured.
        assert br.bound_kind == "memory"
        assert br.peak_tok_per_sec >= self._ACHIEVED_TOK_S
        # within% recomputed from achieved must still match the PR-9749520 anchor (~77%).
        within_pct = 100.0 * self._ACHIEVED_TOK_S / br.peak_tok_per_sec
        assert 70.0 < within_pct < 85.0


class TestDeepSeekV3ConfigAliases:
    """DeepSeek-V3-derived models use HF aliases (``n_routed_experts``, ``quant_method=fp8`` + bf16 ``dtype``) that must be resolved so ceilings stay accurate."""

    def test_n_routed_experts_alias_equivalent_to_num_experts(self, tmp_path):
        # Identical geometry differing only in the routed-expert field name must give an identical breakdown.
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
        # quant_method=fp8 + dtype=bfloat16 (activation): without the quant short-circuit, load_model_meta over-counts weight bytes 2x.
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
        # quant_method wins even when torch_dtype is inconsistent.
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
        # With no torch_dtype/quant_method, the DeepSeek-style ``dtype`` field is consulted before the precision_hint.
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
        # ``mxfp4`` must map to 0.5 byte/param to match the HW_SPECS key (else the ceiling arithmetic falls back to bf16).
        assert _resolve_dtype_bytes("mxfp4") == 0.5

    def test_num_local_experts_alias_equivalent_to_num_experts(self, tmp_path):
        # gpt-oss (GptOssForCausalLM) writes the routed-expert count under ``num_local_experts``; the alias must resolve so it isn't treated as dense.
        # total_size is the ~260 GB bf16-equivalent (above the routed pool) so the MoE decomposition path executes.
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
        assert br.mem_tok_per_sec == pytest.approx(pm_bd.decode_mem_tok_per_s, rel=1e-6)
        assert br.cmp_tok_per_sec == pytest.approx(pm_bd.decode_cmp_tok_per_s, rel=1e-6)
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
        assert br.mem_tok_per_sec == pytest.approx(pm_bd.decode_mem_tok_per_s, rel=1e-6)
        assert br.cmp_tok_per_sec == pytest.approx(pm_bd.decode_cmp_tok_per_s, rel=1e-6)
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


class TestRuntimeDtypeResolution:
    """Roofline ceiling must use the dtype the run actually read, not the
    on-disk ``torch_dtype``. Covers the ``within_roofline_pct > 100`` bug
    where a float32 checkpoint served with ``--quantization fp8`` produced
    a ceiling below the measured throughput."""

    def _dense_fp32_state(self, tmp_path, extra_args: str):
        # float32 dense checkpoint (4 B/param on disk), served fp8 at runtime.
        # Non-empty args model an accepted optimized variant, so current_best
        # carries a tput (the arm the ceiling is compared against).
        _write_synthetic_model(
            tmp_path / "m",
            total_size=32_000_000_000,
            num_layers=48,
            num_kv_heads=8,
            hidden_size=5120,
            num_attention_heads=40,
            torch_dtype="float32",
        )
        current_best = {"extra_server_args": extra_args}
        if extra_args:
            current_best["tput"] = 1000.0
        return SimpleNamespace(
            model_path=str(tmp_path / "m"),
            gpu_type="mi300x", tp=1, precision="fp8",
            conc=64, isl=1024, osl=1024,
            last_baseline={},
            current_best=current_best,
        )

    def test_quantization_fp8_arg_scales_weight_to_one_byte(self, tmp_path):
        state = self._dense_fp32_state(tmp_path, "--quantization fp8")
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        rt = resolve_runtime_dtype(state, meta)
        assert rt.weight_dtype_bytes == 1.0
        assert rt.quantization == "fp8"
        assert rt.source == "server_args_quantization"
        assert rt.compute_precision_tag == "fp8"

    def test_weight_bytes_rescaled_by_runtime_dtype(self, tmp_path):
        state = self._dense_fp32_state(tmp_path, "--quantization fp8")
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        rt = resolve_runtime_dtype(state, meta)
        scaled = apply_runtime_dtype(meta, rt)
        # float32 (4 B) -> fp8 (1 B): weight bytes drop 4x.
        assert scaled.weight_dtype_bytes == 1.0
        assert scaled.weight_bytes == meta.weight_bytes // 4

    def test_fp8_ceiling_exceeds_no_quant_ceiling(self, tmp_path):
        """Runtime fp8 ceiling must be strictly higher than the no-quant
        (bf16-weight) ceiling, so within% drops back below 100."""
        st_fp8 = self._dense_fp32_state(tmp_path, "--quantization fp8")
        st_cfg = self._dense_fp32_state(tmp_path, "")  # no quant -> bf16 weight
        peak_fp8 = compute_roofline_breakdown_from_state(st_fp8).peak_tok_per_sec
        peak_cfg = compute_roofline_breakdown_from_state(st_cfg).peak_tok_per_sec
        assert peak_fp8 > peak_cfg > 0

    def test_precision_fp8_without_quant_does_not_shrink_weight(self, tmp_path):
        """A workload tagged precision=fp8 whose run did NOT pass
        --quantization must NOT be modeled as fp8 weights: the server
        serves bf16, so the weight term stays 2 B (regression for the
        baseline within% under-report)."""
        state = self._dense_fp32_state(tmp_path, "")  # no quant args
        assert state.precision == "fp8"  # workload tag still fp8
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        rt = resolve_runtime_dtype(state, meta)
        # Weight floored at bf16 (fp32 downcast), NOT fp8.
        assert rt.weight_dtype_bytes == 2.0
        assert rt.quantization == "none"
        assert rt.source == "config_torch_dtype"

    def test_baseline_arm_ignores_optimized_quant_args(self, tmp_path):
        """When achieved comes from the baseline arm (no current_best tput),
        an optimized current_best's --quantization fp8 must NOT leak onto
        the baseline ceiling."""
        state = self._dense_fp32_state(tmp_path, "")
        # current_best has fp8 args but no tput -> not the achieved arm.
        state.current_best = {"extra_server_args": "--quantization fp8"}
        state.last_baseline = {}
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        rt = resolve_runtime_dtype(state, meta)
        assert rt.quantization == "none"
        assert rt.weight_dtype_bytes == 2.0

    def test_no_quantization_keeps_config_dtype(self, tmp_path):
        # bf16 checkpoint, no quant args -> weight stays 2 B.
        _write_synthetic_model(
            tmp_path / "m",
            total_size=16_000_000_000,
            num_layers=48, num_kv_heads=8,
            hidden_size=5120, num_attention_heads=40,
            torch_dtype="bfloat16",
        )
        state = SimpleNamespace(
            model_path=str(tmp_path / "m"),
            gpu_type="mi300x", tp=1, precision="",
            conc=64, isl=1024, osl=1024,
            last_baseline={}, current_best={},
        )
        meta = load_model_meta(state.model_path)
        rt = resolve_runtime_dtype(state, meta)
        assert rt.weight_dtype_bytes == 2.0
        assert rt.quantization == "none"

    def test_prequantized_moe_fp8_not_double_scaled(self, tmp_path):
        # On-disk fp8 MoE checkpoint: total_size already reflects fp8.
        _write_synthetic_model(
            tmp_path / "m",
            total_size=480_000_000_000,
            num_layers=62, num_kv_heads=8,
            hidden_size=6144, num_attention_heads=48,
            num_experts=256, num_experts_per_tok=8,
            moe_intermediate_size=1536,
            quant_method="fp8",
        )
        state = SimpleNamespace(
            model_path=str(tmp_path / "m"),
            gpu_type="mi300x", tp=4, precision="fp8",
            conc=64, isl=1024, osl=1024,
            last_baseline={}, current_best={},
        )
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        assert meta.weight_dtype_bytes == 1.0  # quant_method already fp8
        rt = resolve_runtime_dtype(state, meta)
        assert rt.source == "quantization_config"
        scaled = apply_runtime_dtype(meta, rt)
        # No double-scaling: weight bytes unchanged.
        assert scaled.weight_bytes == meta.weight_bytes

    def test_dtype_arg_without_quant_sets_weight_dtype(self, tmp_path):
        state = self._dense_fp32_state(tmp_path, "--dtype bfloat16")
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        rt = resolve_runtime_dtype(state, meta)
        assert rt.weight_dtype_bytes == 2.0
        assert rt.source == "server_args_dtype"
        assert rt.activation_dtype_bytes == 2.0

    def test_activation_dtype_floored_at_bf16(self, tmp_path):
        # Even with fp8 weights, activations stay >= 2 B.
        state = self._dense_fp32_state(tmp_path, "--quantization fp8")
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        rt = resolve_runtime_dtype(state, meta)
        assert rt.activation_dtype_bytes >= 2.0

    def test_eq_form_arg_parsing(self, tmp_path):
        # --quantization=fp8 (= form) parses identically to space form.
        state = self._dense_fp32_state(tmp_path, "--quantization=fp8")
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        rt = resolve_runtime_dtype(state, meta)
        assert rt.weight_dtype_bytes == 1.0
        assert rt.quantization == "fp8"

    def _baseline_yaml(
        self,
        tmp_path,
        env_key: str,
        args: str,
        *,
        model: str = "",
        precision: str = "fp8",
        tp: int = 1,
        conc: int = 64,
        isl: int = 1024,
        osl: int = 1024,
        framework: str = "sglang",
        runner_type: str = "mi300x",
    ) -> str:
        """Write a materialized baseline yaml carrying ``EXTRA_*_ARGS`` and
        return its path (the shape the executor stamps post-baseline)."""
        import yaml as _yaml  # type: ignore[reportMissingModuleSource]
        envs = {
            "TP": tp,
            "CONC": conc,
            "ISL": isl,
            "OSL": osl,
            env_key: args,
        }
        benchmark = {
            "framework": framework,
            "precision": precision,
            "runner_type": runner_type,
            "envs": envs,
        }
        if model:
            benchmark["model"] = model
        cfg = {"benchmark": benchmark}
        path = tmp_path / "baseline.yaml"
        path.write_text(_yaml.safe_dump(cfg), encoding="utf-8")
        return str(path)

    def test_baseline_yaml_args_resolved_when_current_best_empty(self, tmp_path):
        """Baseline-only run: --quantization fp8 lives only in the yaml's
        EXTRA_SGLANG_ARGS (current_best carries no extra_server_args), so
        dtype resolution must read it from the materialized config."""
        cfg_path = self._baseline_yaml(
            tmp_path, "EXTRA_SGLANG_ARGS", "--quantization fp8",
        )
        state = self._dense_fp32_state(tmp_path, "")  # no top-level args
        state.current_best = {"action": "baseline", "tput": 100.0}
        state.last_baseline = {"extras": {"materialized_config": cfg_path}}
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        rt = resolve_runtime_dtype(state, meta)
        assert rt.weight_dtype_bytes == 1.0
        assert rt.quantization == "fp8"
        assert rt.source == "server_args_quantization"

    def test_baseline_yaml_args_vllm_env_key(self, tmp_path):
        # The vllm framework routes flags through EXTRA_VLLM_ARGS.
        cfg_path = self._baseline_yaml(
            tmp_path, "EXTRA_VLLM_ARGS", "--quantization fp8",
        )
        state = self._dense_fp32_state(tmp_path, "")
        state.current_best = {}
        state.last_baseline = {"extras": {"materialized_config": cfg_path}}
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        rt = resolve_runtime_dtype(state, meta)
        assert rt.weight_dtype_bytes == 1.0

    def test_top_level_args_win_over_baseline_yaml(self, tmp_path):
        """An optimized current_best (real extra_server_args) takes
        precedence over the baseline yaml fallback."""
        cfg_path = self._baseline_yaml(
            tmp_path, "EXTRA_SGLANG_ARGS", "--dtype bfloat16",
        )
        state = self._dense_fp32_state(tmp_path, "--quantization fp8")
        state.last_baseline = {"extras": {"materialized_config": cfg_path}}
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        rt = resolve_runtime_dtype(state, meta)
        # current_best's fp8 wins; the yaml's bf16 is never consulted.
        assert rt.weight_dtype_bytes == 1.0
        assert rt.quantization == "fp8"

    def test_optimized_overlay_overrides_same_baseline_flag(self, tmp_path):
        """When both baseline and current_best set the same flag, the
        optimized overlay must win."""
        cfg_path = self._baseline_yaml(
            tmp_path, "EXTRA_SGLANG_ARGS", "--dtype float16",
        )
        state = self._dense_fp32_state(tmp_path, "--dtype bfloat16")
        state.last_baseline = {"extras": {"materialized_config": cfg_path}}
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        rt = resolve_runtime_dtype(state, meta)
        assert rt.source == "server_args_dtype"
        assert rt.weight_dtype_tag == "bfloat16"

    def test_optimized_extra_envs_server_args_are_resolved(self, tmp_path):
        """Accepted env-only configs can carry server args in EXTRA_*_ARGS."""
        state = self._dense_fp32_state(tmp_path, "")
        state.current_best = {
            "tput": 1000.0,
            "extra_envs": {"EXTRA_SGLANG_ARGS": "--quantization fp8"},
        }
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        rt = resolve_runtime_dtype(state, meta)
        assert rt.weight_dtype_bytes == 1.0
        assert rt.quantization == "fp8"
        assert rt.source == "server_args_quantization"

    def test_optimized_extra_envs_override_top_level_server_args(self, tmp_path):
        """materialize_config_with_envs applies extra_envs after
        extra_server_args, so an EXTRA_*_ARGS env wins for dtype resolution."""
        state = self._dense_fp32_state(tmp_path, "--dtype bfloat16")
        state.current_best["extra_envs"] = {"EXTRA_SGLANG_ARGS": "--quantization fp8"}
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        rt = resolve_runtime_dtype(state, meta)
        assert rt.weight_dtype_bytes == 1.0
        assert rt.quantization == "fp8"

    def test_optimized_extra_envs_replace_baseline_server_args(self, tmp_path):
        """extra_envs.EXTRA_*_ARGS replaces the YAML server-args env."""
        cfg_path = self._baseline_yaml(
            tmp_path, "EXTRA_SGLANG_ARGS", "--quantization fp8",
        )
        state = self._dense_fp32_state(tmp_path, "")
        state.current_best = {
            "tput": 1000.0,
            "extra_envs": {"EXTRA_SGLANG_ARGS": "--dtype bfloat16"},
        }
        state.last_baseline = {"extras": {"materialized_config": cfg_path}}
        meta = load_model_meta(state.model_path, precision_hint="fp8")
        rt = resolve_runtime_dtype(state, meta)
        assert rt.source == "server_args_dtype"
        assert rt.weight_dtype_tag == "bfloat16"
        assert rt.quantization == "none"

    def test_runtime_workload_uses_baseline_yaml_fields(self, tmp_path):
        state = self._dense_fp32_state(tmp_path, "")
        runtime_model = state.model_path
        state.model_path = "/wrong/model"
        state.tp = 99
        state.conc = 8
        state.isl = 1
        state.osl = 1
        cfg_path = self._baseline_yaml(
            tmp_path,
            "EXTRA_SGLANG_ARGS",
            "--dtype bfloat16",
            model=runtime_model,
            precision="bf16",
            tp=4,
            conc=64,
            isl=1024,
            osl=2048,
            framework="sglang",
        )
        state.last_baseline = {"extras": {"materialized_config": cfg_path}}

        runtime = resolve_runtime_workload(state)
        assert runtime.model_path == runtime_model
        assert runtime.precision == "bf16"
        assert runtime.framework == "sglang"
        assert runtime.tp == 4
        assert runtime.concurrency == 64
        assert runtime.isl == 1024
        assert runtime.osl == 2048
        assert runtime.server_args == "--dtype bfloat16"

    def test_runtime_workload_uses_real_gpu_over_magpie_runner(self, tmp_path):
        """MI325X runs via Magpie's mi300x runner but roofline uses MI325X."""
        state = self._dense_fp32_state(tmp_path, "")
        state.gpu_type = "mi325x"
        cfg_path = self._baseline_yaml(
            tmp_path,
            "EXTRA_SGLANG_ARGS",
            "--dtype bfloat16",
            runner_type="mi300x",
        )
        state.last_baseline = {"extras": {"materialized_config": cfg_path}}

        runtime = resolve_runtime_workload(state)
        assert runtime.gpu_type == "mi325x"

    def test_ceiling_uses_baseline_yaml_model_when_state_is_stale(self, tmp_path):
        state = self._dense_fp32_state(tmp_path, "")
        runtime_model = state.model_path
        state.model_path = "/wrong/model"
        state.tp = 0
        state.conc = 0
        state.isl = 0
        state.osl = 0
        cfg_path = self._baseline_yaml(
            tmp_path,
            "EXTRA_SGLANG_ARGS",
            "--dtype bfloat16",
            model=runtime_model,
            precision="bf16",
            tp=1,
            conc=32,
            isl=512,
            osl=512,
        )
        state.last_baseline = {"extras": {"materialized_config": cfg_path}}

        bd = compute_roofline_breakdown_from_state(state)
        assert bd.peak_tok_per_sec > 0
        assert bd.bound_kind in {"memory", "compute"}

    def test_missing_baseline_yaml_degrades_safely(self, tmp_path):
        # Unreadable materialized_config -> no crash, falls back to config.
        state = self._dense_fp32_state(tmp_path, "")
        state.precision = ""
        state.current_best = {}
        state.last_baseline = {"extras": {"materialized_config": "/no/such.yaml"}}
        meta = load_model_meta(state.model_path)
        rt = resolve_runtime_dtype(state, meta)
        assert rt.quantization == "none"


class TestArmPinnedPrecision:
    """``arm`` pins ceiling precision: baseline keeps baseline dtype even after a fp8 current_best is promoted."""

    def _bf16_baseline_with_fp8_best(self, tmp_path):
        _write_synthetic_model(
            tmp_path / "m",
            total_size=140_000_000_000,
            num_layers=80, num_kv_heads=8,
            hidden_size=8192, num_attention_heads=64,
            torch_dtype="bfloat16",
        )
        # Baseline yaml carries NO quantization (bf16 weights).
        yaml_path = tmp_path / "bl.yaml"
        yaml_path.write_text(
            "benchmark:\n  envs:\n    CONC: 32\n",
            encoding="utf-8",
        )
        return SimpleNamespace(
            model_path=str(tmp_path / "m"),
            gpu_type="mi300x",
            tp=1,
            precision="bf16",
            conc=32,
            isl=2048,
            osl=512,
            last_baseline={"extras": {"materialized_config": str(yaml_path)}},
            # Promoted optimized arm quantizes weights to fp8.
            current_best={
                "tput": 2000.0,
                "extra_server_args": "--quantization fp8",
            },
        )

    def test_baseline_arm_ignores_current_best_fp8(self, tmp_path):
        state = self._bf16_baseline_with_fp8_best(tmp_path)
        meta = load_model_meta(state.model_path)
        rt = resolve_runtime_dtype(state, meta, arm="baseline")
        # Baseline yaml had no --quantization -> weights stay bf16 (2 bytes).
        assert rt.weight_dtype_bytes == 2.0
        assert rt.quantization == "none"

    def test_current_best_arm_picks_up_fp8(self, tmp_path):
        state = self._bf16_baseline_with_fp8_best(tmp_path)
        meta = load_model_meta(state.model_path)
        rt = resolve_runtime_dtype(state, meta, arm="current_best")
        assert rt.weight_dtype_bytes == 1.0
        assert rt.quantization == "fp8"

    def test_baseline_arm_ceiling_below_fp8_arm(self, tmp_path):
        state = self._bf16_baseline_with_fp8_best(tmp_path)
        baseline_peak = compute_roofline_breakdown_from_state(
            state, arm="baseline",
        ).peak_tok_per_sec
        best_peak = compute_roofline_breakdown_from_state(
            state, arm="current_best",
        ).peak_tok_per_sec
        # fp8 halves weight IO, so the best-arm ceiling must be higher.
        assert baseline_peak > 0
        assert best_peak > baseline_peak
