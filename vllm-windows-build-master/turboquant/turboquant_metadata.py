# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import torch

TURBOQUANT_METADATA_VERSION = 1
TURBOQUANT_TRANSFORM_VERSION = "structured_hadamard_v1"
TURBOQUANT_CODEBOOK_VERSION = "lloyd_beta_v1"
TURBOQUANT_OUTLIER_RATIOS = {
    "turboquant25": 0.25,
    "turboquant35": 0.50,
}
TURBOQUANT_GROUP_ALIGNMENT = 16


def _get_turboquant_outlier_count(head_size: int, kv_cache_dtype: str) -> int:
    if head_size % TURBOQUANT_GROUP_ALIGNMENT != 0:
        raise ValueError(
            "TurboQuant KV cache requires head_size to be a multiple of 16."
        )
    ratio = TURBOQUANT_OUTLIER_RATIOS[kv_cache_dtype]
    aligned = int(
        round(head_size * ratio / TURBOQUANT_GROUP_ALIGNMENT)
        * TURBOQUANT_GROUP_ALIGNMENT
    )
    if aligned <= 0 or aligned >= head_size:
        raise ValueError(
            f"Unsupported TurboQuant head_size {head_size} for {kv_cache_dtype}."
        )
    return aligned


@dataclass(frozen=True)
class TurboQuantTensorMetadata:
    high_precision_indices: tuple[tuple[int, ...], ...]

    def get_group_indices(
        self,
        device: torch.device,
        head_size: int,
        kv_cache_dtype: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        high_cpu, low_cpu = _cached_group_indices(
            self.high_precision_indices,
            head_size,
            kv_cache_dtype,
        )
        if device.type == "cpu":
            return high_cpu, low_cpu
        return high_cpu.to(device=device), low_cpu.to(device=device)

    def to_json(self) -> list[list[int]]:
        return [list(indices) for indices in self.high_precision_indices]


@cache
def _cached_group_indices(
    high_precision_indices: tuple[tuple[int, ...], ...],
    head_size: int,
    kv_cache_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    outlier_count = _get_turboquant_outlier_count(head_size, kv_cache_dtype)
    if len(high_precision_indices) == 0:
        raise ValueError("TurboQuant metadata must contain at least one KV head.")

    all_idx = torch.arange(head_size, dtype=torch.int64)
    high_groups: list[torch.Tensor] = []
    low_groups: list[torch.Tensor] = []
    for head_idx, high_indices in enumerate(high_precision_indices):
        if len(high_indices) != outlier_count:
            raise ValueError(
                "TurboQuant metadata high-precision group size mismatch for "
                f"head {head_idx}: expected {outlier_count}, got "
                f"{len(high_indices)}."
            )
        high = torch.tensor(high_indices, dtype=torch.int64)
        if torch.any(high[:-1] >= high[1:]):
            raise ValueError(
                "TurboQuant metadata high-precision indices must be strictly "
                "sorted."
            )
        if high.min().item() < 0 or high.max().item() >= head_size:
            raise ValueError(
                "TurboQuant metadata high-precision indices are out of range."
            )
        unique = torch.unique(high)
        if unique.numel() != high.numel():
            raise ValueError(
                "TurboQuant metadata high-precision indices must be unique."
            )
        low_mask = torch.ones(head_size, dtype=torch.bool)
        low_mask.scatter_(0, high, False)
        low = all_idx[low_mask]
        high_groups.append(high)
        low_groups.append(low)

    return (
        torch.stack(high_groups, dim=0).contiguous(),
        torch.stack(low_groups, dim=0).contiguous(),
    )


@dataclass(frozen=True)
class TurboQuantLayerMetadata:
    key: TurboQuantTensorMetadata
    value: TurboQuantTensorMetadata

    def to_json(self) -> dict[str, list[list[int]]]:
        return {
            "key_high_precision_indices": self.key.to_json(),
            "value_high_precision_indices": self.value.to_json(),
        }


@dataclass(frozen=True)
class TurboQuantCalibrationMetadata:
    method: str
    objective: str
    num_prompts: int
    max_seq_len: int
    batch_size: int
    num_observed_tokens: int
    dtype: str
    device: str
    prompts_sha256: str

    def to_json(self) -> dict[str, object]:
        return {
            "method": self.method,
            "objective": self.objective,
            "num_prompts": self.num_prompts,
            "max_seq_len": self.max_seq_len,
            "batch_size": self.batch_size,
            "num_observed_tokens": self.num_observed_tokens,
            "dtype": self.dtype,
            "device": self.device,
            "prompts_sha256": self.prompts_sha256,
        }


@dataclass(frozen=True)
class TurboQuantMetadata:
    recipe: str
    head_size: int
    model_name: str | None
    layers: dict[str, TurboQuantLayerMetadata]
    calibration: TurboQuantCalibrationMetadata | None = None
    version: int = TURBOQUANT_METADATA_VERSION
    transform_version: str = TURBOQUANT_TRANSFORM_VERSION
    codebook_version: str = TURBOQUANT_CODEBOOK_VERSION

    def get_layer(self, layer_name: str) -> TurboQuantLayerMetadata:
        candidate_names = _turboquant_layer_name_candidates(layer_name)
        for candidate_name in candidate_names:
            layer = self.layers.get(candidate_name)
            if layer is not None:
                return layer

        raise KeyError(
            "TurboQuant metadata does not contain layer "
            f"{layer_name!r}. Tried aliases: {', '.join(repr(name) for name in candidate_names)}."
        )

    def to_json(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "version": self.version,
            "recipe": self.recipe,
            "head_size": self.head_size,
            "model_name": self.model_name,
            "transform_version": self.transform_version,
            "codebook_version": self.codebook_version,
            "layers": {
                layer_name: layer_metadata.to_json()
                for layer_name, layer_metadata in self.layers.items()
            },
        }
        if self.calibration is not None:
            payload["calibration"] = self.calibration.to_json()
        return payload


def _parse_tensor_metadata(
    payload: object,
    field_name: str,
) -> TurboQuantTensorMetadata:
    if not isinstance(payload, list):
        raise ValueError(f"TurboQuant metadata field {field_name!r} must be a list.")
    high_precision_indices: list[tuple[int, ...]] = []
    for head_payload in payload:
        if not isinstance(head_payload, list) or not all(
            isinstance(index, int) for index in head_payload
        ):
            raise ValueError(
                f"TurboQuant metadata field {field_name!r} must contain integer lists."
            )
        high_precision_indices.append(tuple(head_payload))
    return TurboQuantTensorMetadata(
        high_precision_indices=tuple(high_precision_indices)
    )


def turboquant_metadata_from_json(payload: dict[str, object]) -> TurboQuantMetadata:
    version = int(payload.get("version", TURBOQUANT_METADATA_VERSION))
    if version != TURBOQUANT_METADATA_VERSION:
        raise ValueError(
            f"Unsupported TurboQuant metadata version {version}. Expected "
            f"{TURBOQUANT_METADATA_VERSION}."
        )

    recipe = payload.get("recipe")
    if not isinstance(recipe, str):
        raise ValueError("TurboQuant metadata must define a string recipe.")
    head_size = payload.get("head_size")
    if not isinstance(head_size, int):
        raise ValueError("TurboQuant metadata must define an integer head_size.")
    model_name = payload.get("model_name")
    if model_name is not None and not isinstance(model_name, str):
        raise ValueError("TurboQuant metadata model_name must be a string or null.")

    layers_payload = payload.get("layers")
    if not isinstance(layers_payload, dict):
        raise ValueError("TurboQuant metadata must define an object-valued layers.")

    layers: dict[str, TurboQuantLayerMetadata] = {}
    for layer_name, layer_payload in layers_payload.items():
        if not isinstance(layer_name, str) or not isinstance(layer_payload, dict):
            raise ValueError("TurboQuant metadata layers must be object-valued.")
        layers[layer_name] = TurboQuantLayerMetadata(
            key=_parse_tensor_metadata(
                layer_payload.get("key_high_precision_indices"),
                "key_high_precision_indices",
            ),
            value=_parse_tensor_metadata(
                layer_payload.get("value_high_precision_indices"),
                "value_high_precision_indices",
            ),
        )

    calibration_payload = payload.get("calibration")
    calibration: TurboQuantCalibrationMetadata | None = None
    if calibration_payload is not None:
        if not isinstance(calibration_payload, dict):
            raise ValueError(
                "TurboQuant metadata calibration field must be an object or null."
            )
        calibration = TurboQuantCalibrationMetadata(
            method=str(calibration_payload.get("method", "")),
            objective=str(calibration_payload.get("objective", "")),
            num_prompts=int(calibration_payload.get("num_prompts", 0)),
            max_seq_len=int(calibration_payload.get("max_seq_len", 0)),
            batch_size=int(calibration_payload.get("batch_size", 0)),
            num_observed_tokens=int(calibration_payload.get("num_observed_tokens", 0)),
            dtype=str(calibration_payload.get("dtype", "")),
            device=str(calibration_payload.get("device", "")),
            prompts_sha256=str(calibration_payload.get("prompts_sha256", "")),
        )

    return TurboQuantMetadata(
        recipe=recipe,
        head_size=head_size,
        model_name=model_name,
        calibration=calibration,
        transform_version=str(
            payload.get("transform_version", TURBOQUANT_TRANSFORM_VERSION)
        ),
        codebook_version=str(
            payload.get("codebook_version", TURBOQUANT_CODEBOOK_VERSION)
        ),
        layers=layers,
        version=version,
    )


@cache
def load_turboquant_metadata(path: str) -> TurboQuantMetadata:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("TurboQuant metadata root must be a JSON object.")
    return turboquant_metadata_from_json(payload)


def save_turboquant_metadata(metadata: TurboQuantMetadata, path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata.to_json(), f, indent=2, sort_keys=True)
        f.write("\n")


def discover_turboquant_metadata_path(
    model_name_or_path: str | None,
    explicit_path: str | None,
) -> str | None:
    if explicit_path is not None:
        return explicit_path
    if model_name_or_path is None:
        return None

    model_path = Path(model_name_or_path)
    if model_path.is_file():
        model_path = model_path.parent
    elif not model_path.is_dir():
        return None

    metadata_path = model_path / "turboquant_kv.json"
    if metadata_path.is_file():
        return str(metadata_path)
    return None


def build_default_turboquant_metadata(
    *,
    recipe: str,
    head_size: int,
    num_kv_heads: int,
    layer_names: list[str],
    model_name: str | None = None,
) -> TurboQuantMetadata:
    outlier_count = _get_turboquant_outlier_count(head_size, recipe)
    default_high = tuple(tuple(range(outlier_count)) for _ in range(num_kv_heads))
    layer_metadata = TurboQuantLayerMetadata(
        key=TurboQuantTensorMetadata(default_high),
        value=TurboQuantTensorMetadata(default_high),
    )
    return TurboQuantMetadata(
        recipe=recipe,
        head_size=head_size,
        model_name=model_name,
        layers={layer_name: layer_metadata for layer_name in layer_names},
    )


def _turboquant_layer_name_candidates(layer_name: str) -> tuple[str, ...]:
    candidates: list[str] = []

    def add(name: str) -> None:
        if name not in candidates:
            candidates.append(name)

    add(layer_name)

    if layer_name.endswith(".attn"):
        add(layer_name.removesuffix(".attn"))

    if layer_name.startswith("language_model."):
        stripped = layer_name.removeprefix("language_model.")
        add(stripped)
        if stripped.endswith(".attn"):
            add(stripped.removesuffix(".attn"))

    return tuple(candidates)
