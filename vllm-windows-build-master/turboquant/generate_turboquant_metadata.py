#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import regex as re
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

try:
    from transformers import AutoModelForImageTextToText
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForImageTextToText = None

try:
    from transformers import AutoModelForVision2Seq
except ImportError:  # pragma: no cover - depends on transformers version
    AutoModelForVision2Seq = None

from vllm.transformers_utils.config import get_config
from vllm.v1.attention.ops.turboquant_kv_cache import get_turboquant_outlier_count
from vllm.v1.attention.ops.turboquant_metadata import (
    TurboQuantCalibrationMetadata,
    TurboQuantLayerMetadata,
    TurboQuantMetadata,
    TurboQuantTensorMetadata,
    save_turboquant_metadata,
)

PROJECTION_PATTERN = re.compile(
    r"(^|.*\.)layers\.(?P<layer>\d+)\.self_attn\.(?P<proj>[kv]_proj)$"
)


def _load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    if not prompts:
        raise ValueError("The calibration prompt file is empty.")
    return prompts


def _derive_model_shape(
    model: str,
    *,
    trust_remote_code: bool = False,
) -> tuple[int, int, int, tuple[str, ...] | None, dict | None]:
    config = get_config(model, trust_remote_code=trust_remote_code)
    text_config = getattr(config, "text_config", None)
    source = text_config if text_config is not None else config

    head_size = getattr(source, "head_dim", None)
    hidden_size = getattr(source, "hidden_size", None)
    num_attention_heads = getattr(source, "num_attention_heads", None)
    num_kv_heads = getattr(source, "num_key_value_heads", num_attention_heads)
    num_hidden_layers = getattr(source, "num_hidden_layers", None)
    if head_size is None and (hidden_size is None or num_attention_heads is None):
        raise ValueError("Unable to derive TurboQuant head_size from the model config.")
    if num_hidden_layers is None:
        raise ValueError(
            "Unable to derive TurboQuant metadata shape from the model config."
        )
    if head_size is None:
        assert hidden_size is not None
        assert num_attention_heads is not None
        head_size = hidden_size // num_attention_heads
    layer_types = getattr(source, "layer_types", None)
    if layer_types is not None:
        layer_types = tuple(layer_types)
    quantization_config = getattr(config, "quantization_config", None)
    if quantization_config is None and text_config is not None:
        quantization_config = getattr(text_config, "quantization_config", None)
    return (
        head_size,
        num_kv_heads,
        num_hidden_layers,
        layer_types,
        quantization_config,
    )


def _is_quantized_model(quantization_config: dict | None) -> bool:
    return isinstance(quantization_config, dict) and len(quantization_config) > 0


def _validate_calibration_model_choice(
    *,
    target_model: str,
    calibration_model: str,
    quantization_config: dict | None,
) -> None:
    if calibration_model != target_model:
        return
    if not _is_quantized_model(quantization_config):
        return

    quant_method = quantization_config.get("quant_method", "unknown")
    raise ValueError(
        "TurboQuant calibration should not run directly on a quantized target "
        f"checkpoint ({quant_method}). Pass `--calibration-model` pointing to "
        "the original non-quantized model so activation statistics are collected "
        "from real weights instead of a partially reconstructed fallback model."
    )


def _resolve_layer_indices(
    num_hidden_layers: int,
    layer_types: tuple[str, ...] | None,
) -> list[int]:
    if layer_types is None:
        return list(range(num_hidden_layers))
    if len(layer_types) != num_hidden_layers:
        raise ValueError(
            "Model config layer_types length does not match num_hidden_layers."
        )
    return [
        layer_idx
        for layer_idx, layer_type in enumerate(layer_types)
        if layer_type == "full_attention"
    ]


def _resolve_torch_dtype(dtype: str) -> torch.dtype | str:
    if dtype == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[dtype]
    except KeyError as e:
        raise ValueError(f"Unsupported calibration dtype: {dtype}") from e


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _load_calibration_model(
    model_name_or_path: str,
    *,
    torch_dtype: torch.dtype | str,
    trust_remote_code: bool,
):
    model_loaders = [
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoModelForVision2Seq,
        AutoModel,
    ]
    errors: list[str] = []

    for loader in model_loaders:
        if loader is None:
            continue
        try:
            return loader.from_pretrained(
                model_name_or_path,
                torch_dtype=torch_dtype,
                trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
        except (KeyError, ValueError) as e:
            errors.append(f"{loader.__name__}: {e}")

    raise ValueError(
        "Unable to load the calibration model with the installed transformers "
        "build. This usually means the checkpoint architecture is newer than "
        "your local transformers version. Upgrade transformers, then rerun. "
        f"Loader errors: {' | '.join(errors)}"
    )


def _ensure_padding_token(tokenizer) -> None:
    if tokenizer.pad_token_id is not None:
        return
    if tokenizer.eos_token_id is None:
        raise ValueError(
            "Tokenizer must define pad_token_id or eos_token_id for calibration."
        )
    tokenizer.pad_token = tokenizer.eos_token


def _discover_projection_modules(
    model: torch.nn.Module,
    required_layer_indices: list[int],
) -> dict[tuple[int, str], torch.nn.Module]:
    modules: dict[tuple[int, str], torch.nn.Module] = {}
    for module_name, module in model.named_modules():
        match = PROJECTION_PATTERN.match(module_name)
        if match is None:
            continue
        layer_idx = int(match.group("layer"))
        if layer_idx not in required_layer_indices:
            continue
        tensor_tag = "key" if match.group("proj") == "k_proj" else "value"
        modules[(layer_idx, tensor_tag)] = module

    missing = [
        (layer_idx, tensor_tag)
        for layer_idx in required_layer_indices
        for tensor_tag in ("key", "value")
        if (layer_idx, tensor_tag) not in modules
    ]
    if missing:
        formatted = ", ".join(
            f"layer {layer_idx} {tensor_tag}" for layer_idx, tensor_tag in missing
        )
        raise ValueError(
            "Unable to locate required TurboQuant calibration projection modules: "
            f"{formatted}."
        )
    return modules


def _select_high_precision_indices(
    channel_scores: torch.Tensor,
    outlier_count: int,
) -> tuple[tuple[int, ...], ...]:
    if channel_scores.ndim != 2:
        raise ValueError(
            "TurboQuant channel scores must have shape [num_kv_heads, head_size]."
        )
    tie_break = torch.linspace(
        0.0,
        1e-9,
        channel_scores.shape[-1],
        dtype=channel_scores.dtype,
        device=channel_scores.device,
    )
    selected = torch.topk(
        channel_scores + tie_break.unsqueeze(0),
        k=outlier_count,
        dim=-1,
    ).indices
    selected = torch.sort(selected, dim=-1).values.cpu()
    return tuple(tuple(int(index) for index in head.tolist()) for head in selected)


def _build_tensor_metadata_from_scores(
    channel_scores: torch.Tensor,
    outlier_count: int,
) -> TurboQuantTensorMetadata:
    return TurboQuantTensorMetadata(
        high_precision_indices=_select_high_precision_indices(
            channel_scores, outlier_count
        )
    )


class _ActivationAccumulator:
    def __init__(self, num_kv_heads: int, head_size: int) -> None:
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.current_attention_mask: torch.Tensor | None = None
        self.channel_scores: dict[tuple[int, str], torch.Tensor] = {}

    def set_attention_mask(self, attention_mask: torch.Tensor) -> None:
        self.current_attention_mask = attention_mask

    def clear_attention_mask(self) -> None:
        self.current_attention_mask = None

    def hook(
        self,
        layer_idx: int,
        tensor_tag: str,
    ):
        def _hook(_module, _inputs, output):
            projected = output[0] if isinstance(output, tuple) else output
            if projected.ndim != 3:
                raise ValueError(
                    "TurboQuant calibration expects projection outputs with shape "
                    "[batch, seq, hidden]."
                )
            expected_hidden = self.num_kv_heads * self.head_size
            if projected.shape[-1] != expected_hidden:
                raise ValueError(
                    "TurboQuant calibration projection output size mismatch: "
                    f"expected {expected_hidden}, got {projected.shape[-1]}."
                )
            if self.current_attention_mask is None:
                raise RuntimeError("Calibration attention mask was not initialized.")
            flat_mask = self.current_attention_mask.reshape(-1).to(torch.bool)
            flat_projected = (
                projected.detach()
                .to(torch.float32)
                .reshape(-1, self.num_kv_heads, self.head_size)
            )
            if flat_projected.shape[0] != flat_mask.numel():
                raise ValueError(
                    "Projection output shape does not match calibration attention_mask."
                )
            valid = flat_projected[flat_mask]
            channel_scores = valid.square().sum(dim=0).cpu()
            key = (layer_idx, tensor_tag)
            existing = self.channel_scores.get(key)
            self.channel_scores[key] = (
                channel_scores if existing is None else existing + channel_scores
            )

        return _hook


@torch.inference_mode()
def _collect_activation_channel_scores(
    *,
    model_name_or_path: str,
    prompts: list[str],
    required_layer_indices: list[int],
    num_kv_heads: int,
    head_size: int,
    batch_size: int,
    max_seq_len: int,
    dtype: str,
    device: str,
    trust_remote_code: bool,
) -> tuple[dict[tuple[int, str], torch.Tensor], int]:
    model_device = _resolve_device(device)
    model_torch_dtype = _resolve_torch_dtype(dtype)
    model = _load_calibration_model(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        torch_dtype=model_torch_dtype,
    )
    model.to(model_device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    _ensure_padding_token(tokenizer)

    modules = _discover_projection_modules(model, required_layer_indices)
    accumulator = _ActivationAccumulator(num_kv_heads=num_kv_heads, head_size=head_size)
    handles = [
        module.register_forward_hook(accumulator.hook(layer_idx, tensor_tag))
        for (layer_idx, tensor_tag), module in sorted(modules.items())
    ]

    observed_tokens = 0
    try:
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start : start + batch_size]
            encoded = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_len,
            )
            attention_mask = encoded["attention_mask"].to(model_device)
            observed_tokens += int(attention_mask.sum().item())
            accumulator.set_attention_mask(attention_mask)
            model(
                input_ids=encoded["input_ids"].to(model_device),
                attention_mask=attention_mask,
                use_cache=False,
            )
            accumulator.clear_attention_mask()
    finally:
        for handle in handles:
            handle.remove()

    if observed_tokens <= 0:
        raise ValueError("TurboQuant calibration observed zero tokens.")
    return accumulator.channel_scores, observed_tokens


def _build_calibrated_metadata(
    *,
    recipe: str,
    head_size: int,
    model_name: str,
    num_hidden_layers: int,
    layer_types: tuple[str, ...] | None,
    layer_pattern: str,
    num_kv_heads: int,
    calibration_scores: dict[tuple[int, str], torch.Tensor],
    calibration_metadata: TurboQuantCalibrationMetadata,
) -> TurboQuantMetadata:
    outlier_count = get_turboquant_outlier_count(head_size, recipe)
    layers: dict[str, TurboQuantLayerMetadata] = {}
    for layer_idx in _resolve_layer_indices(num_hidden_layers, layer_types):
        key_scores = calibration_scores.get((layer_idx, "key"))
        value_scores = calibration_scores.get((layer_idx, "value"))
        if key_scores is None or value_scores is None:
            raise ValueError(
                f"Missing calibration scores for TurboQuant layer {layer_idx}."
            )
        for tensor_tag, scores in (("key", key_scores), ("value", value_scores)):
            if scores.shape != (num_kv_heads, head_size):
                raise ValueError(
                    f"Unexpected {tensor_tag} calibration score shape for layer "
                    f"{layer_idx}: {tuple(scores.shape)}."
                )
        layer_name = layer_pattern.format(i=layer_idx)
        layers[layer_name] = TurboQuantLayerMetadata(
            key=_build_tensor_metadata_from_scores(key_scores, outlier_count),
            value=_build_tensor_metadata_from_scores(value_scores, outlier_count),
        )

    return TurboQuantMetadata(
        recipe=recipe,
        head_size=head_size,
        model_name=model_name,
        layers=layers,
        calibration=calibration_metadata,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate an activation-calibrated TurboQuant metadata artifact for "
            "mixed-bit KV cache modes."
        )
    )
    parser.add_argument(
        "--model", required=True, help="Local model path or HF model id."
    )
    parser.add_argument(
        "--calibration-model",
        help=(
            "Optional local model path or HF model id used only for calibration. "
            "Required when --model points to a quantized checkpoint."
        ),
    )
    parser.add_argument(
        "--kv-cache-dtype",
        choices=("turboquant25", "turboquant35"),
        required=True,
    )
    parser.add_argument(
        "--prompts-file",
        required=True,
        help="Text file with one calibration prompt per line.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to the generated turboquant_kv.json file.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Allow remote model config/model code when loading for calibration.",
    )
    parser.add_argument(
        "--layer-pattern",
        default="model.layers.{i}.self_attn.attn",
        help="Layer-name pattern used to populate metadata keys.",
    )
    parser.add_argument(
        "--max-prompts",
        type=int,
        default=128,
        help="Maximum number of prompts to use from the prompts file.",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum calibration sequence length after tokenization.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Number of prompts to process per calibration forward pass.",
    )
    parser.add_argument(
        "--dtype",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
        help="Model dtype to use during calibration.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Calibration device, for example 'auto', 'cpu', or 'cuda:0'.",
    )
    args = parser.parse_args()

    if args.max_prompts <= 0:
        raise ValueError("--max-prompts must be positive.")
    if args.max_seq_len <= 0:
        raise ValueError("--max-seq-len must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    prompts = _load_prompts(args.prompts_file)[: args.max_prompts]
    prompts_sha256 = hashlib.sha256("\n".join(prompts).encode("utf-8")).hexdigest()
    (
        head_size,
        num_kv_heads,
        num_hidden_layers,
        layer_types,
        quantization_config,
    ) = _derive_model_shape(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    calibration_model = args.calibration_model or args.model
    _validate_calibration_model_choice(
        target_model=args.model,
        calibration_model=calibration_model,
        quantization_config=quantization_config,
    )
    required_layer_indices = _resolve_layer_indices(num_hidden_layers, layer_types)
    calibration_scores, num_observed_tokens = _collect_activation_channel_scores(
        model_name_or_path=calibration_model,
        prompts=prompts,
        required_layer_indices=required_layer_indices,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        dtype=args.dtype,
        device=args.device,
        trust_remote_code=args.trust_remote_code,
    )
    metadata = _build_calibrated_metadata(
        recipe=args.kv_cache_dtype,
        head_size=head_size,
        model_name=args.model,
        num_hidden_layers=num_hidden_layers,
        layer_types=layer_types,
        layer_pattern=args.layer_pattern,
        num_kv_heads=num_kv_heads,
        calibration_scores=calibration_scores,
        calibration_metadata=TurboQuantCalibrationMetadata(
            method="activation_energy_v1",
            objective="sum_squared_activation",
            num_prompts=len(prompts),
            max_seq_len=args.max_seq_len,
            batch_size=args.batch_size,
            num_observed_tokens=num_observed_tokens,
            dtype=args.dtype,
            device=str(_resolve_device(args.device)),
            prompts_sha256=prompts_sha256,
        ),
    )
    save_turboquant_metadata(metadata, Path(args.output))


if __name__ == "__main__":
    main()
