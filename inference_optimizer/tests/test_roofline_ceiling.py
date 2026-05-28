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
    _resolve_dtype_bytes,
    compute_kv_bytes_per_token,
    compute_peak_from_state,
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
) -> None:
    """Lay down a minimal HF-shaped model dir.

    Pass ``num_kv_heads=None`` to omit ``num_key_value_heads`` from
    config.json (covers the MHA fallback branch).
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

    def test_missing_safetensors_index_returns_none(self, tmp_path):
        d = tmp_path / "m"
        d.mkdir()
        (d / "config.json").write_text(
            json.dumps({"num_hidden_layers": 12, "torch_dtype": "bfloat16"})
        )
        assert load_model_meta(d) is None

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
# HW_SPECS table sanity.
# ---------------------------------------------------------------------------
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
