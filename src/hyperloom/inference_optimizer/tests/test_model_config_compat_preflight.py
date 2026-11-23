# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the model-config compatibility preflight."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import pytest

from hyperloom.inference_optimizer.cli import model_gate as cli
from hyperloom.inference_optimizer import gpu_types
from hyperloom.inference_optimizer.cli import model_compat


def _write_config(model_dir: Path, *, with_tokenizer: bool = True, **fields) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text(json.dumps(fields), encoding="utf-8")
    # Most config-compat tests are unrelated to the tokenizer-artifact check; ship a tokenizer by default so they
    # exercise only the field they target.
    if with_tokenizer:
        (model_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")


def _write_corrupt_config(model_dir: Path) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{not valid json", encoding="utf-8")


def _args(model: str, *, gpu_type: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(model=model, isl=1024, osl=1024, gpu_type=gpu_type)


def _seed_state(session_dir: Path, monkeypatch):
    monkeypatch.setenv(
        "INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR",
        str(session_dir),
    )
    from hyperloom.orchestrator.state.shared_state import SharedState

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "reports").mkdir(parents=True, exist_ok=True)
    SharedState(session_id="t", model_name="m", model_path="m").save(session_dir)


@pytest.fixture(autouse=True)
def _default_non_amd_gpu(monkeypatch):
    """Keep config checks hermetic unless a test passes gpu_type explicitly."""
    monkeypatch.delenv("GPU_TYPE", raising=False)
    monkeypatch.setattr(gpu_types, "_autodetect_gpu_type", lambda: None)


# _detect_incompatible_model_config
def test_detect_reports_the_first_matching_detector(tmp_path):
    """First match wins: on AMD a bitsandbytes checkpoint reports its quant format, not the missing tokenizer a later detector also sees."""
    m = tmp_path / "bnb_no_tokenizer"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="llama",
        max_position_embeddings=4096,
        quantization_config={"quant_method": "bitsandbytes"},
    )
    reason = model_compat._detect_incompatible_model_config(str(m), gpu_type="mi300x")
    assert reason is not None
    assert "bitsandbytes" in reason
    assert "tokenizer" not in reason


def test_detect_healthy_config_returns_none(tmp_path):
    m = tmp_path / "ok"
    _write_config(m, model_type="llama", max_position_embeddings=4096)
    assert model_compat._detect_incompatible_model_config(str(m)) is None


def test_detect_diffusers_pipeline_blocks_without_config_json(tmp_path):
    m = tmp_path / "flux"
    m.mkdir()
    (m / "model_index.json").write_text(
        json.dumps({"_class_name": "FluxPipeline"}),
        encoding="utf-8",
    )

    reason = model_compat._detect_incompatible_model_config(str(m))

    assert reason is not None
    assert "Diffusers pipeline" in reason
    assert "causal LM" in reason


def test_detect_missing_tokenizer_blocks(tmp_path):
    # Gensyn-Swarm fine-tune class: weights + config, no tokenizer artifacts.
    m = tmp_path / "no_tok"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="qwen2",
        max_position_embeddings=32768,
    )
    reason = model_compat._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "tokenizer" in reason.lower()


def test_detect_missing_tokenizer_skipped_for_scriptable_xdit(tmp_path):
    """xDiT (scriptable) is a server-less image workload that never loads a HF tokenizer, so a missing-tokenizer config.json must NOT block it."""
    m = tmp_path / "xdit_no_tok"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="qwen2",
        max_position_embeddings=32768,
    )
    assert model_compat._detect_incompatible_model_config(str(m), framework="xdit") is None


def test_detect_missing_tokenizer_still_blocks_serving_framework(tmp_path):
    """Regression guard: the skip is scoped to scriptable — an explicit serving framework (sglang) must still block a missing-tokenizer checkpoint."""
    m = tmp_path / "sglang_no_tok"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="qwen2",
        max_position_embeddings=32768,
    )
    reason = model_compat._detect_incompatible_model_config(str(m), framework="sglang")
    assert reason is not None
    assert "tokenizer" in reason.lower()


def test_detect_with_tokenizer_ok(tmp_path):
    m = tmp_path / "with_tok"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="qwen2",
        max_position_embeddings=32768,
    )
    (m / "tokenizer.json").write_text("{}", encoding="utf-8")
    assert model_compat._detect_incompatible_model_config(str(m)) is None


def test_detect_missing_tokenizer_but_auto_map_ok(tmp_path):
    # A custom AutoTokenizer in auto_map can supply the tokenizer at load time.
    m = tmp_path / "auto_tok"
    _write_config(
        m,
        with_tokenizer=False,
        model_type="custom",
        max_position_embeddings=4096,
        auto_map={"AutoTokenizer": ["x.TokClass", None]},
    )
    assert model_compat._detect_incompatible_model_config(str(m)) is None


def test_detect_corrupt_config_blocks(tmp_path):
    m = tmp_path / "corrupt"
    _write_corrupt_config(m)
    reason = model_compat._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "unparseable" in reason


def test_detect_absent_config_not_blocked(tmp_path):
    m = tmp_path / "no_config"
    m.mkdir(parents=True, exist_ok=True)
    assert model_compat._detect_incompatible_model_config(str(m)) is None


def test_detect_dual_chunk_blocks_on_amd(tmp_path):
    m = tmp_path / "dual_chunk_amd"
    _write_config(
        m,
        model_type="qwen2",
        max_position_embeddings=1010000,
        dual_chunk_attention_config={"chunk_size": 262144},
    )
    reason = model_compat._detect_incompatible_model_config(str(m), gpu_type="mi300x")
    assert reason is not None
    assert "dual_chunk" in reason


def test_detect_dual_chunk_not_blocked_off_amd(tmp_path):
    m = tmp_path / "dual_chunk_non_amd"
    _write_config(
        m,
        model_type="qwen2",
        max_position_embeddings=1010000,
        dual_chunk_attention_config={"chunk_size": 262144},
    )
    assert model_compat._detect_incompatible_model_config(str(m)) is None


def _write_quant_config(model_dir: Path, payload: dict) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "hf_quant_config.json").write_text(
        json.dumps(payload),
        encoding="utf-8",
    )


def test_detect_modelopt_fp8_blocks_on_amd(tmp_path):
    """ModelOpt FP8 (declared in hf_quant_config.json) has no ROCm loader."""
    m = tmp_path / "modelopt_fp8"
    _write_config(m, model_type="llama", max_position_embeddings=8192)
    _write_quant_config(
        m,
        {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "FP8", "kv_cache_quant_algo": "FP8"},
        },
    )
    reason = model_compat._detect_incompatible_model_config(str(m), gpu_type="mi300x")
    assert reason is not None
    assert "modelopt" in reason.lower() or "FP8" in reason


def test_detect_nvfp4_blocks_on_amd(tmp_path):
    m = tmp_path / "nvfp4"
    _write_config(m, model_type="llama", max_position_embeddings=8192)
    _write_quant_config(
        m,
        {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "NVFP4"},
        },
    )
    reason = model_compat._detect_incompatible_model_config(str(m), gpu_type="mi300x")
    assert reason is not None
    assert "NVFP4" in reason or "nvfp4" in reason.lower()


def test_detect_bitsandbytes_blocks_on_amd(tmp_path):
    """bitsandbytes declared in config.json.quantization_config; CUDA-only kernels."""
    m = tmp_path / "bnb"
    _write_config(
        m,
        model_type="llama",
        max_position_embeddings=8192,
        quantization_config={"quant_method": "bitsandbytes"},
    )
    reason = model_compat._detect_incompatible_model_config(str(m), gpu_type="mi300x")
    assert reason is not None
    assert "bitsandbytes" in reason.lower()


def test_detect_modelopt_fp8_not_blocked_off_amd(tmp_path):
    """The same checkpoint can still run on a vendor (NVIDIA) engine."""
    m = tmp_path / "modelopt_fp8_nv"
    _write_config(m, model_type="llama", max_position_embeddings=8192)
    _write_quant_config(
        m,
        {
            "producer": {"name": "modelopt"},
            "quantization": {"quant_algo": "FP8"},
        },
    )
    assert model_compat._detect_incompatible_model_config(str(m)) is None


def test_detect_amd_native_fp8_not_blocked(tmp_path):
    """AMD Quark / compressed-tensors FP8 is ROCm-native; must NOT be blocked."""
    m = tmp_path / "quark_fp8"
    _write_config(
        m,
        model_type="llama",
        max_position_embeddings=8192,
        quantization_config={"quant_method": "fp8"},
    )
    assert model_compat._detect_incompatible_model_config(str(m), gpu_type="mi300x") is None


def _write_safetensors_header(model_dir: Path, tensors: dict) -> None:
    header = json.dumps(tensors).encode("utf-8")
    with (model_dir / "model.safetensors").open("wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)


def test_detect_vocab_shape_mismatch_blocks(tmp_path):
    m = tmp_path / "qwen_vocab_mismatch"
    _write_config(
        m,
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        max_position_embeddings=4096,
        vocab_size=152064,
    )
    _write_safetensors_header(
        m,
        {
            "model.embed_tokens.weight": {
                "dtype": "BF16",
                "shape": [151936, 1536],
                "data_offsets": [0, 0],
            },
            "lm_head.weight": {
                "dtype": "BF16",
                "shape": [151936, 1536],
                "data_offsets": [0, 0],
            },
        },
    )
    reason = model_compat._detect_incompatible_model_config(str(m))
    assert reason is not None
    assert "vocab_size=152064" in reason
    assert "151936" in reason


def test_detect_vocab_shape_match_not_blocked(tmp_path):
    # config vocab_size matches the weight output dim -> must NOT block.
    m = tmp_path / "qwen_vocab_ok"
    _write_config(
        m,
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        max_position_embeddings=4096,
        vocab_size=151936,
    )
    _write_safetensors_header(
        m,
        {
            "model.embed_tokens.weight": {
                "dtype": "BF16",
                "shape": [151936, 1536],
                "data_offsets": [0, 0],
            },
        },
    )
    assert model_compat._detect_incompatible_model_config(str(m)) is None


def test_detect_vocab_shape_padded_not_blocked(tmp_path):
    # actual > config vocab_size -> commonly a padded embedding (rounded up to an alignment / TP boundary while config
    # keeps the unpadded value).
    m = tmp_path / "qwen_vocab_padded"
    _write_config(
        m,
        model_type="qwen2",
        architectures=["Qwen2ForCausalLM"],
        max_position_embeddings=4096,
        vocab_size=151936,
    )
    _write_safetensors_header(
        m,
        {
            "model.embed_tokens.weight": {
                "dtype": "BF16",
                "shape": [152064, 1536],  # padded up from 151936
                "data_offsets": [0, 0],
            },
            "lm_head.weight": {
                "dtype": "BF16",
                "shape": [152064, 1536],
                "data_offsets": [0, 0],
            },
        },
    )
    assert model_compat._detect_incompatible_model_config(str(m)) is None


def test_read_safetensors_header_parses_and_rejects(tmp_path):
    # Valid header round-trips; truncated / oversized headers return None.
    good = tmp_path / "model.safetensors"
    _write_safetensors_header(tmp_path, {"x.weight": {"shape": [4, 4]}})
    assert model_compat._read_safetensors_header(good) == {"x.weight": {"shape": [4, 4]}}

    truncated = tmp_path / "trunc.safetensors"
    truncated.write_bytes(struct.pack("<Q", 999) + b"{")  # claims 999, has 1
    assert model_compat._read_safetensors_header(truncated) is None

    short = tmp_path / "short.safetensors"
    short.write_bytes(b"\x01\x02")  # < 8-byte length prefix
    assert model_compat._read_safetensors_header(short) is None


# Gemma2 detection helpers (model_config_utils)
@pytest.mark.parametrize(
    "name,expected",
    [
        ("gemma2", True),
        ("gemma2-9b", True),
        ("google-gemma-2-9b-it", True),
        ("gemma_2_2b", True),
        ("llama-gemma2-test", True),
        ("gemma-3-12b", False),
        ("gemma3-12b", False),
        ("gemma25", False),
        ("notgemma2", False),
        ("mygemma2", False),
        ("gemma12-model", False),
        ("", False),
    ],
)
def test_path_looks_like_gemma2(name, expected):
    from hyperloom.inference_optimizer import model_config_utils as mcu

    assert mcu._path_looks_like_gemma2(name) is expected


def test_model_is_gemma2_falls_back_to_path_on_residual_config(tmp_path):
    # config.json present but empty (no model_type/architectures) -> the path heuristic decides; a gemma-2 path is
    # still detected.
    from hyperloom.inference_optimizer import model_config_utils as mcu

    m = tmp_path / "google-gemma-2-9b-it"
    m.mkdir()
    (m / "config.json").write_text("{}", encoding="utf-8")
    assert mcu.model_is_gemma2(str(m)) is True


def test_model_is_gemma2_trusts_identified_non_gemma_config(tmp_path):
    # config clearly identifies llama; even a gemma-2 path must NOT override it.
    from hyperloom.inference_optimizer import model_config_utils as mcu

    m = tmp_path / "gemma-2-distill-llama"
    m.mkdir()
    (m / "config.json").write_text(
        json.dumps(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
            }
        ),
        encoding="utf-8",
    )
    assert mcu.model_is_gemma2(str(m)) is False


def test_model_is_gemma2_detects_nested_text_config(tmp_path):
    from hyperloom.inference_optimizer import model_config_utils as mcu

    m = tmp_path / "wrapper"
    m.mkdir()
    (m / "config.json").write_text(
        json.dumps(
            {
                "model_type": "wrapper",
                "text_config": {"model_type": "gemma2"},
            }
        ),
        encoding="utf-8",
    )
    assert mcu.model_is_gemma2(str(m)) is True


# _preflight_model_config_compat — persistence + return contract
def test_preflight_blocks_and_persists(tmp_path, monkeypatch):
    model = tmp_path / "bad"
    _write_corrupt_config(model)
    sd = tmp_path / "session"
    _seed_state(sd, monkeypatch)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is True
    final = json.loads((sd / "reports" / "final.json").read_text())
    assert final["stop_reason"] == "model_config_incompatible"
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "model_config_incompatible"
    breakdown = json.loads((sd / "session_breakdown.json").read_text())
    assert breakdown["outcome"]["stop_reason"] == "model_config_incompatible"


def test_preflight_passes_for_healthy_model(tmp_path, monkeypatch):
    model = tmp_path / "good"
    _write_config(model, model_type="llama", max_position_embeddings=8192)
    sd = tmp_path / "session_ok"
    _seed_state(sd, monkeypatch)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is False
    assert not (sd / "reports" / "final.json").exists()


def test_preflight_runs_amd_only_detectors_for_args_gpu_type(
    tmp_path,
    monkeypatch,
):
    model = tmp_path / "bnb"
    _write_config(
        model,
        model_type="llama",
        max_position_embeddings=4096,
        quantization_config={"quant_method": "bitsandbytes"},
    )
    sd = tmp_path / "session_amd_quant"
    _seed_state(sd, monkeypatch)

    assert (
        cli._preflight_model_config_compat(
            _args(str(model), gpu_type="mi300x"),
            sd,
        )
        is True
    )
    final = json.loads((sd / "reports" / "final.json").read_text())
    assert final["stop_reason"] == "model_config_incompatible"


def test_preflight_persists_under_strict_env(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_STRICT_STOP_REASON", "1")
    model = tmp_path / "bad_strict"
    _write_corrupt_config(model)
    sd = tmp_path / "session_strict"
    _seed_state(sd, monkeypatch)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is True
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "model_config_incompatible"


# Private / third-party quantization formats (paroquant, MLX, mxtq, GGUF)
@pytest.mark.parametrize(
    ("config", "files", "expected"),
    [
        pytest.param(
            {
                "model_type": "qwen3_5",
                "max_position_embeddings": 32768,
                "quantization_config": {"quant_method": "paroquant", "bits": 4, "group_size": 128, "krot": 8},
            },
            (),
            "paroquant",
            id="paroquant",
        ),
        # MTPLX / MLX 8-bit affine: mode set, no quant_method.
        pytest.param(
            {
                "model_type": "qwen3_5",
                "max_position_embeddings": 32768,
                "quantization_config": {"bits": 8, "group_size": 64, "mode": "affine"},
            },
            (),
            "MLX",
            id="mlx_affine",
        ),
        # bits/group_size but no quant_method/mode; sglang raises "Unknown quantization method: ''" in engine init.
        pytest.param(
            {
                "model_type": "llama",
                "max_position_embeddings": 32768,
                "quantization_config": {"group_size": 64, "bits": 8},
            },
            (),
            "quant_method",
            id="no_method",
        ),
        pytest.param(
            {
                "model_type": "qwen3_5_moe",
                "max_position_embeddings": 32768,
                "quantization_config": {"weight_format": "mxtq", "method": "affine", "group_size": 64},
            },
            (),
            "mxtq",
            id="mxtq",
        ),
        pytest.param(
            {
                "model_type": "llama",
                "architectures": ["LlamaForCausalLM"],
                "max_position_embeddings": 4096,
                "quantization_config": {"quant_method": ""},
            },
            (),
            "quant_method is empty",
            id="empty_method",
        ),
        pytest.param(
            {"model_type": "qwen3_5_moe", "max_position_embeddings": 32768},
            ("model-TQ3_4S.gguf",),
            "GGUF",
            id="gguf_only",
        ),
    ],
)
def test_private_quant_blocks(tmp_path, config, files, expected):
    m = tmp_path / "model"
    _write_config(m, **config)
    for name in files:
        (m / name).write_text("dummy", encoding="utf-8")
    reason = model_compat._detect_incompatible_model_config(str(m))
    assert reason is not None and expected in reason


def test_private_quant_mlx_weights_index_blocks(tmp_path):
    # JANG/MLX often declares no quant config; the tell is '.biases'/'.scales'.
    m = tmp_path / "mlx_weights"
    _write_config(m, model_type="qwen3_5", max_position_embeddings=32768)
    (m / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "lm_head.biases": "model-00001.safetensors",
                    "lm_head.scales": "model-00001.safetensors",
                    "model.embed_tokens.weight": "model-00001.safetensors",
                }
            }
        ),
        encoding="utf-8",
    )
    reason = model_compat._detect_incompatible_model_config(str(m))
    assert reason is not None and "MLX" in reason


def test_peft_adapter_only_checkpoint_blocks(tmp_path):
    m = tmp_path / "adapter_only"
    _write_config(m, model_type="qwen2", max_position_embeddings=32768)
    (m / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen2-7B"}),
        encoding="utf-8",
    )
    (m / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "base_model.model.lm_head.lora_A.default.weight": "adapter_model.safetensors",
                    "base_model.model.lm_head.lora_B.default.weight": "adapter_model.safetensors",
                    "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": ("adapter_model.safetensors"),
                }
            }
        ),
        encoding="utf-8",
    )

    reason = model_compat._detect_incompatible_model_config(str(m))

    assert reason is not None
    assert "PEFT/LoRA adapter" in reason
    assert "base_model.model.lm_head.base_layer.weight" in reason


def test_merged_peft_like_checkpoint_with_base_weights_not_blocked(tmp_path):
    m = tmp_path / "merged_adapter"
    _write_config(m, model_type="qwen2", max_position_embeddings=32768)
    (m / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001.safetensors",
                    "lm_head.weight": "model-00001.safetensors",
                    "base_model.model.layers.0.self_attn.q_proj.lora_A.default.weight": ("model-00001.safetensors"),
                }
            }
        ),
        encoding="utf-8",
    )

    assert model_compat._detect_incompatible_model_config(str(m)) is None


@pytest.mark.parametrize("method", ["fp8", "awq", "gptq", "compressed-tensors"])
def test_standard_quant_not_blocked(tmp_path, method):
    m = tmp_path / "std"
    _write_config(
        m,
        model_type="qwen3",
        max_position_embeddings=32768,
        quantization_config={"quant_method": method},
    )
    assert model_compat._detect_incompatible_model_config(str(m)) is None


def test_gguf_with_safetensors_not_blocked(tmp_path):
    # A normal safetensors model that merely also ships a .gguf must still pass.
    m = tmp_path / "gguf_plus_st"
    _write_config(m, model_type="qwen3", max_position_embeddings=32768)
    (m / "model.gguf").write_text("dummy", encoding="utf-8")
    (m / "model.safetensors").write_text("dummy", encoding="utf-8")
    assert model_compat._detect_incompatible_model_config(str(m)) is None


def test_gguf_with_auxiliary_bin_still_blocks(tmp_path):
    # An arbitrary .bin next to GGUF is not an HF-loadable weight file.
    m = tmp_path / "gguf_plus_aux_bin"
    _write_config(m, model_type="qwen3", max_position_embeddings=32768)
    (m / "model.gguf").write_text("dummy", encoding="utf-8")
    (m / "tokenizer_cache.bin").write_text("dummy", encoding="utf-8")
    reason = model_compat._detect_incompatible_model_config(str(m))
    assert reason is not None and "GGUF" in reason


def test_gguf_with_pytorch_model_bin_not_blocked(tmp_path):
    # HF PyTorch weights are loadable by the default framework loaders.
    m = tmp_path / "gguf_plus_hf_bin"
    _write_config(m, model_type="qwen3", max_position_embeddings=32768)
    (m / "model.gguf").write_text("dummy", encoding="utf-8")
    (m / "pytorch_model.bin").write_text("dummy", encoding="utf-8")
    assert model_compat._detect_incompatible_model_config(str(m)) is None


# Langfuse parity on fail-fast — pre-flight gates exit before the normal Langfuse flush point, so each must push the
# breakdown to Langfuse itself.
def _spy_langfuse_emit(monkeypatch) -> dict[str, list]:
    calls: dict[str, list] = {"flush": [], "patch": [], "record": []}
    from hyperloom.inference_optimizer import breakdown as bd
    from hyperloom.inference_optimizer.trace import langfuse_emitter as lfe

    monkeypatch.setattr(lfe, "flush_session", lambda sd: calls["flush"].append(Path(sd)))
    monkeypatch.setattr(bd, "patch_breakdown_langfuse", lambda sd: calls["patch"].append(Path(sd)))
    monkeypatch.setattr(
        lfe,
        "record_session_breakdown",
        lambda sd, *a, **k: calls["record"].append(Path(sd)),
    )
    return calls


def test_model_config_fail_fast_emits_to_langfuse(tmp_path, monkeypatch):
    model = tmp_path / "bad_lf"
    _write_corrupt_config(model)
    sd = tmp_path / "session_lf"
    _seed_state(sd, monkeypatch)
    calls = _spy_langfuse_emit(monkeypatch)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is True
    # written to disk AND pushed to Langfuse in flush -> patch -> record order.
    assert (sd / "session_breakdown.json").exists()
    assert calls["flush"] == [sd]
    assert calls["patch"] == [sd]
    assert calls["record"] == [sd]


def test_unsupported_model_fail_fast_emits_to_langfuse(tmp_path, monkeypatch):
    model = tmp_path / "vlm_lf"
    _write_config(
        model,
        model_type="llava",
        architectures=["LlavaForConditionalGeneration"],
        max_position_embeddings=4096,
    )
    sd = tmp_path / "session_vlm_lf"
    _seed_state(sd, monkeypatch)
    calls = _spy_langfuse_emit(monkeypatch)

    assert cli._preflight_unsupported_model_arch(_args(str(model)), sd) is True
    assert calls["record"] == [sd]


def test_context_window_fail_fast_emits_to_langfuse(tmp_path, monkeypatch):
    model = tmp_path / "ctx_lf"
    _write_config(model, model_type="llama", max_position_embeddings=128)
    sd = tmp_path / "session_ctx_lf"
    _seed_state(sd, monkeypatch)
    calls = _spy_langfuse_emit(monkeypatch)

    # ISL+OSL (1024+1024) + headroom >> 128 -> fail fast.
    assert cli._preflight_context_window(_args(str(model)), sd) is True
    assert calls["record"] == [sd]


def test_emit_to_langfuse_is_best_effort(tmp_path, monkeypatch):
    # A Langfuse outage must never turn a clean fail-fast into a crash, nor mask the persisted stop reason.
    model = tmp_path / "bad_raise"
    _write_corrupt_config(model)
    sd = tmp_path / "session_raise"
    _seed_state(sd, monkeypatch)

    from hyperloom.inference_optimizer.trace import langfuse_emitter as lfe

    def _boom(*a, **k):
        raise OSError("langfuse down")

    monkeypatch.setattr(lfe, "flush_session", _boom)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is True
    state = json.loads((sd / "state.json").read_text())
    assert state["stop_reason"] == "model_config_incompatible"


def test_healthy_model_does_not_emit_to_langfuse(tmp_path, monkeypatch):
    # A passing pre-flight must not touch Langfuse — the normal end-of-session path owns that for runs that actually
    # start.
    model = tmp_path / "good_lf"
    _write_config(model, model_type="llama", max_position_embeddings=8192)
    sd = tmp_path / "session_good_lf"
    _seed_state(sd, monkeypatch)
    calls = _spy_langfuse_emit(monkeypatch)

    assert cli._preflight_model_config_compat(_args(str(model)), sd) is False
    assert calls["record"] == []


def test_declared_standard_quant_with_scales_index_not_blocked(tmp_path):
    # AWQ/GPTQ/compressed-tensors legitimately ship '.scales'/'.biases' tensors; a declared supported quant_method
    # must NOT be misread as MLX (the weight- index tell only applies to checkpoints with NO quant_method declared).
    for method in ("awq", "gptq", "compressed-tensors"):
        m = tmp_path / f"std_scales_{method.replace('-', '_')}"
        _write_config(
            m,
            model_type="qwen3",
            max_position_embeddings=32768,
            quantization_config={"quant_method": method},
        )
        (m / "model.safetensors.index.json").write_text(
            json.dumps(
                {
                    "weight_map": {
                        "model.layers.0.self_attn.q_proj.scales": "model-00001.safetensors",
                        "model.layers.0.self_attn.q_proj.biases": "model-00001.safetensors",
                        "model.embed_tokens.weight": "model-00001.safetensors",
                    }
                }
            ),
            encoding="utf-8",
        )
        assert model_compat._detect_incompatible_model_config(str(m)) is None, method


def test_run_compat_detector_resolves_repo_id_before_dispatch(tmp_path, monkeypatch):
    """A repo-id must be resolved to its local cache dir before the waterfall so disk-reading detectors receive a real directory instead of the bare repo-id (which Path().is_dir() would reject, silently skipping the check)."""
    local_dir = tmp_path / "cache" / "models--org--repo"
    local_dir.mkdir(parents=True)
    monkeypatch.setattr(
        model_compat,
        "resolve_local_model_dir",
        lambda mp: local_dir if mp == "org/repo" else None,
    )
    seen: dict[str, str] = {}

    def _fake_detector(model_path):
        seen["model_path"] = model_path
        return

    spec = model_compat.DetectorSpec("t", _fake_detector, args=("model_path",))
    model_compat._run_compat_detector(spec, model_path="org/repo", data={})
    assert seen["model_path"] == str(local_dir)


def test_run_compat_detector_falls_back_to_raw_when_unresolvable(monkeypatch):
    """An unresolvable model path (e.g. an uncached repo-id) falls back to the raw string so behaviour is unchanged from before the resolver."""
    monkeypatch.setattr(model_compat, "resolve_local_model_dir", lambda mp: None)
    seen: dict[str, str] = {}

    def _fake_detector(model_path):
        seen["model_path"] = model_path
        return

    spec = model_compat.DetectorSpec("t", _fake_detector, args=("model_path",))
    model_compat._run_compat_detector(spec, model_path="/raw/model/path", data={})
    assert seen["model_path"] == "/raw/model/path"
