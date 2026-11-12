# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import torch

TURBOQUANT_KV_CACHE_BITS = {
    "turboquant25": 2.5,
    "turboquant35": 3.5,
}
TURBOQUANT_OUTLIER_RATIOS = {
    "turboquant25": 0.25,
    "turboquant35": 0.50,
}
TURBOQUANT_GROUP_BITS = {
    "turboquant25": (3, 2),
    "turboquant35": (4, 3),
}
TURBOQUANT_VECTOR_NORM_BYTES = 2
TURBOQUANT_RESIDUAL_NORM_BYTES = 2
TURBOQUANT_NORM_BYTES = TURBOQUANT_VECTOR_NORM_BYTES + TURBOQUANT_RESIDUAL_NORM_BYTES
TURBOQUANT_GROUP_ALIGNMENT = 16
TURBOQUANT_SEED = 20250428
TURBOQUANT_QJL_SEED_OFFSET = 10_000
TURBOQUANT_QJL_SCALE = math.sqrt(math.pi / 2.0)
TURBOQUANT_CODEBOOK_GRID_POINTS = 32768
TURBOQUANT_CODEBOOK_EPS = 1e-6


@dataclass(frozen=True)
class TurboQuantGroupLayout:
    dim: int
    bits: int
    mse_bits: int
    mse_payload_bytes: int
    qjl_payload_bytes: int
    qjl_offset: int
    vector_norm_offset: int
    residual_norm_offset: int
    packed_bytes: int


@dataclass(frozen=True)
class TurboQuantLayout:
    groups: tuple[TurboQuantGroupLayout, TurboQuantGroupLayout]
    packed_dim: int


def is_turboquant_kv_cache(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype in TURBOQUANT_KV_CACHE_BITS


def get_turboquant_bits(kv_cache_dtype: str) -> float:
    try:
        return TURBOQUANT_KV_CACHE_BITS[kv_cache_dtype]
    except KeyError as e:
        raise ValueError(
            f"Unsupported TurboQuant KV cache dtype: {kv_cache_dtype}"
        ) from e


def _canonical_turboquant_dtype(bits_or_dtype: float | int | str) -> str:
    if isinstance(bits_or_dtype, str):
        if not is_turboquant_kv_cache(bits_or_dtype):
            raise ValueError(f"Unsupported TurboQuant KV cache dtype: {bits_or_dtype}")
        return bits_or_dtype

    bits = float(bits_or_dtype)
    if bits == 2.5:
        return "turboquant25"
    if bits == 3.5:
        return "turboquant35"
    raise ValueError(f"Unsupported TurboQuant bit-width: {bits}")


@cache
def _bit_layout(
    device_type: str,
    device_index: int | None,
    head_size: int,
    bits: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(device_type, device_index)
    bit_offsets = torch.arange(bits, dtype=torch.int64, device=device)
    bit_offsets = bit_offsets.expand(head_size, bits).reshape(-1)
    flat_positions = torch.arange(head_size * bits, dtype=torch.int64, device=device)
    return (
        bit_offsets,
        flat_positions // 8,
        flat_positions % 8,
    )


@cache
def _hadamard_block_sizes(dim: int) -> tuple[int, ...]:
    sizes: list[int] = []
    remaining = dim
    while remaining > 0:
        block = 1 << (remaining.bit_length() - 1)
        sizes.append(block)
        remaining -= block
    return tuple(sizes)


def _fwht_pow2(x: torch.Tensor) -> torch.Tensor:
    orig_shape = x.shape
    size = orig_shape[-1]
    out = x.reshape(-1, size)
    block = 1
    while block < size:
        out = out.reshape(out.shape[0], -1, block * 2)
        left = out[..., :block]
        right = out[..., block : 2 * block]
        out = torch.cat((left + right, left - right), dim=-1)
        out = out.reshape(-1, size)
        block *= 2
    return out.reshape(orig_shape)


@cache
def _structured_signs_cached(
    device_type: str,
    device_index: int | None,
    dim: int,
    seed: int,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    if device.type == "cuda":
        with torch.accelerator.device_index(device.index):
            return torch.where(
                torch.randint(0, 2, (dim,), generator=generator, device=device) > 0,
                torch.ones(dim, dtype=torch.float32, device=device),
                -torch.ones(dim, dtype=torch.float32, device=device),
            )
    return torch.where(
        torch.randint(0, 2, (dim,), generator=generator, device=device) > 0,
        torch.ones(dim, dtype=torch.float32, device=device),
        -torch.ones(dim, dtype=torch.float32, device=device),
    )


def _apply_block_hadamard(
    x: torch.Tensor,
    signs: torch.Tensor,
    *,
    normalized: bool,
    inverse: bool,
) -> torch.Tensor:
    outputs: list[torch.Tensor] = []
    cursor = 0
    for block_size in _hadamard_block_sizes(x.shape[-1]):
        block = x[..., cursor : cursor + block_size]
        block_signs = signs[cursor : cursor + block_size]
        if inverse:
            block = _fwht_pow2(block)
            block = block * block_signs
        else:
            block = block * block_signs
            block = _fwht_pow2(block)
        if normalized:
            block = block / math.sqrt(block_size)
        outputs.append(block)
        cursor += block_size
    return torch.cat(outputs, dim=-1)


def _apply_mse_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return _apply_block_hadamard(x, signs, normalized=True, inverse=False)


def _apply_mse_inverse_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return _apply_block_hadamard(x, signs, normalized=True, inverse=True)


def _apply_qjl_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return _apply_block_hadamard(x, signs, normalized=False, inverse=False)


def _apply_qjl_inverse_transform(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    return _apply_block_hadamard(x, signs, normalized=False, inverse=True)


def _beta_coordinate_pdf(x: torch.Tensor, dim: int) -> torch.Tensor:
    exponent = 0.5 * (dim - 3)
    log_norm = (
        torch.lgamma(torch.tensor(dim / 2.0, dtype=torch.float64))
        - 0.5 * math.log(math.pi)
        - torch.lgamma(torch.tensor((dim - 1) / 2.0, dtype=torch.float64))
    )
    base = (1.0 - x.square()).clamp_min(TURBOQUANT_CODEBOOK_EPS)
    return torch.exp(log_norm + exponent * torch.log(base))


@cache
def _dimension_aware_codebook(
    dim: int,
    bits: int,
) -> torch.Tensor:
    levels = 1 << bits
    grid = torch.linspace(
        -1.0 + TURBOQUANT_CODEBOOK_EPS,
        1.0 - TURBOQUANT_CODEBOOK_EPS,
        TURBOQUANT_CODEBOOK_GRID_POINTS,
        dtype=torch.float64,
    )
    weights = _beta_coordinate_pdf(grid, dim)
    centroids = torch.linspace(
        -1.0 + 1.0 / (levels + 1),
        1.0 - 1.0 / (levels + 1),
        levels,
        dtype=torch.float64,
    )

    for _ in range(200):
        bounds = torch.empty(levels + 1, dtype=torch.float64)
        bounds[0] = -1.0
        bounds[-1] = 1.0
        bounds[1:-1] = 0.5 * (centroids[:-1] + centroids[1:])
        assignments = torch.bucketize(grid, bounds[1:-1])

        masses = torch.zeros(levels, dtype=torch.float64)
        sums = torch.zeros(levels, dtype=torch.float64)
        masses.scatter_add_(0, assignments, weights)
        sums.scatter_add_(0, assignments, weights * grid)
        new_centroids = sums / masses.clamp_min(1e-18)
        if torch.max(torch.abs(new_centroids - centroids)) < 1e-10:
            centroids = new_centroids
            break
        centroids = new_centroids

    return centroids.to(torch.float32)


def get_turboquant_outlier_count(head_size: int, kv_cache_dtype: str) -> int:
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


def get_turboquant_group_dims(
    head_size: int,
    kv_cache_dtype: str,
) -> tuple[int, int]:
    outlier_count = get_turboquant_outlier_count(head_size, kv_cache_dtype)
    return outlier_count, head_size - outlier_count


@cache
def _layout_cached(
    kv_cache_dtype: str,
    head_size: int,
) -> TurboQuantLayout:
    group_dims = get_turboquant_group_dims(head_size, kv_cache_dtype)
    group_bits = TURBOQUANT_GROUP_BITS[kv_cache_dtype]
    groups: list[TurboQuantGroupLayout] = []
    cursor = 0
    for dim, bits in zip(group_dims, group_bits, strict=True):
        mse_bits = bits - 1
        mse_payload_bytes = (dim * mse_bits + 7) // 8
        qjl_payload_bytes = (dim + 7) // 8
        qjl_offset = cursor + mse_payload_bytes
        vector_norm_offset = qjl_offset + qjl_payload_bytes
        residual_norm_offset = vector_norm_offset + TURBOQUANT_VECTOR_NORM_BYTES
        packed_bytes = (
            mse_payload_bytes
            + qjl_payload_bytes
            + TURBOQUANT_VECTOR_NORM_BYTES
            + TURBOQUANT_RESIDUAL_NORM_BYTES
        )
        groups.append(
            TurboQuantGroupLayout(
                dim=dim,
                bits=bits,
                mse_bits=mse_bits,
                mse_payload_bytes=mse_payload_bytes,
                qjl_payload_bytes=qjl_payload_bytes,
                qjl_offset=qjl_offset,
                vector_norm_offset=vector_norm_offset,
                residual_norm_offset=residual_norm_offset,
                packed_bytes=packed_bytes,
            )
        )
        cursor += packed_bytes
    return TurboQuantLayout(groups=tuple(groups), packed_dim=cursor)


def get_turboquant_layout(
    kv_cache_dtype: str,
    head_size: int,
) -> TurboQuantLayout:
    return _layout_cached(kv_cache_dtype, head_size)


def get_turboquant_packed_dim(
    head_size: int,
    bits_or_dtype: float | int | str,
) -> int:
    kv_cache_dtype = _canonical_turboquant_dtype(bits_or_dtype)
    return get_turboquant_layout(kv_cache_dtype, head_size).packed_dim


def get_turboquant_rotation(
    device: torch.device,
    dim: int,
    seed_offset: int = 0,
) -> torch.Tensor:
    return _structured_signs_cached(
        device.type,
        device.index,
        dim,
        TURBOQUANT_SEED + seed_offset + dim,
    )


def get_turboquant_qjl_matrix(
    device: torch.device,
    dim: int,
    seed_offset: int = 0,
) -> torch.Tensor:
    return _structured_signs_cached(
        device.type,
        device.index,
        dim,
        TURBOQUANT_SEED + TURBOQUANT_QJL_SEED_OFFSET + seed_offset + dim,
    )


@cache
def _transform_matrix_cached(
    device_type: str,
    device_index: int | None,
    dim: int,
    seed_offset: int,
    kind: str,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    identity = torch.eye(dim, dtype=torch.float32, device=device)
    if kind == "mse_forward":
        return _apply_mse_transform(
            identity, get_turboquant_rotation(device, dim, seed_offset)
        )
    if kind == "mse_inverse":
        return _apply_mse_inverse_transform(
            identity, get_turboquant_rotation(device, dim, seed_offset)
        )
    if kind == "qjl_forward":
        return _apply_qjl_transform(
            identity, get_turboquant_qjl_matrix(device, dim, seed_offset)
        )
    if kind == "qjl_inverse":
        return _apply_qjl_inverse_transform(
            identity, get_turboquant_qjl_matrix(device, dim, seed_offset)
        )
    raise ValueError(f"Unsupported TurboQuant transform kind: {kind}")


def get_turboquant_mse_transform_matrix(
    device: torch.device,
    dim: int,
    seed_offset: int = 0,
) -> torch.Tensor:
    """Return the [dim, dim] MSE forward-transform matrix for the given group.

    Materialises the structured Hadamard transform F such that
    ``F @ v`` gives the rotated unit vector used for centroid assignment.
    Result is cached per (device, dim, seed_offset).
    """
    return _transform_matrix_cached(
        device.type,
        device.index,
        dim,
        seed_offset,
        "mse_forward",
    )


def get_turboquant_mse_inverse_transform_matrix(
    device: torch.device,
    dim: int,
    seed_offset: int = 0,
) -> torch.Tensor:
    """Return the [dim, dim] MSE inverse-transform matrix for the given group.

    Materialises F^T (the transpose/inverse of the MSE forward transform).
    Used in the decode postprocess kernel to convert accumulator values back
    to the original coordinate space.
    Result is cached per (device, dim, seed_offset).
    """
    return _transform_matrix_cached(
        device.type,
        device.index,
        dim,
        seed_offset,
        "mse_inverse",
    )


def get_turboquant_qjl_transform_matrix(
    device: torch.device,
    dim: int,
    seed_offset: int = 0,
) -> torch.Tensor:
    """Return the [dim, dim] QJL forward-transform matrix for the given group.

    Materialises the structured Hadamard QJL projection matrix G such that
    ``sign(G @ residual)`` gives the 1-bit QJL codes.
    Result is cached per (device, dim, seed_offset).
    """
    return _transform_matrix_cached(
        device.type,
        device.index,
        dim,
        seed_offset,
        "qjl_forward",
    )


def get_turboquant_qjl_inverse_transform_matrix(
    device: torch.device,
    dim: int,
    seed_offset: int = 0,
) -> torch.Tensor:
    """Return the [dim, dim] QJL inverse-transform matrix for the given group.

    Materialises G^T (the transpose/inverse of the QJL forward transform).
    Used in the decode postprocess kernel to convert QJL accumulator values
    back to the original coordinate space.
    Result is cached per (device, dim, seed_offset).
    """
    return _transform_matrix_cached(
        device.type,
        device.index,
        dim,
        seed_offset,
        "qjl_inverse",
    )


@cache
def _mse_to_qjl_matrix_cached(
    device_type: str,
    device_index: int | None,
    dim: int,
    mse_seed_offset: int,
    qjl_seed_offset: int,
) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    mse_inverse = get_turboquant_mse_inverse_transform_matrix(
        device, dim, mse_seed_offset
    )
    qjl_forward = get_turboquant_qjl_transform_matrix(device, dim, qjl_seed_offset)
    return torch.matmul(mse_inverse, qjl_forward)


def get_turboquant_mse_to_qjl_matrix(
    device: torch.device,
    dim: int,
    mse_seed_offset: int = 0,
    qjl_seed_offset: int = 0,
) -> torch.Tensor:
    """Return the [dim, dim] composed MSE-inverse × QJL-forward matrix.

    Pre-computes ``F^T @ G`` (the MSE inverse transform followed by the QJL
    forward transform).  Used in the fused KV update kernel to project the MSE
    centroid approximation into the QJL space for residual computation, without
    materialising the intermediate MSE-inverse vector.
    Result is cached per (device, dim, mse_seed_offset, qjl_seed_offset).
    """
    return _mse_to_qjl_matrix_cached(
        device.type,
        device.index,
        dim,
        mse_seed_offset,
        qjl_seed_offset,
    )


def get_turboquant_centroids(
    device: torch.device,
    dim: int,
    bits: int,
) -> torch.Tensor:
    return _dimension_aware_codebook(dim, bits).to(device=device)


def get_turboquant_mse_codebook_bits(
    kv_cache_dtype: str,
    _head_size: int,
) -> tuple[int, ...]:
    high_bits, low_bits = TURBOQUANT_GROUP_BITS[kv_cache_dtype]
    return tuple(sorted({high_bits - 1, low_bits - 1}))


def pack_turboquant_indices(indices: torch.Tensor, bits: int) -> torch.Tensor:
    head_size = indices.shape[-1]
    packed_bytes = (head_size * bits + 7) // 8
    if packed_bytes == 0:
        return torch.empty(
            (*indices.shape[:-1], 0),
            dtype=torch.uint8,
            device=indices.device,
        )

    bit_offsets, byte_index, bit_index = _bit_layout(
        indices.device.type,
        indices.device.index,
        head_size,
        bits,
    )
    bit_offsets = bit_offsets.reshape(*((1,) * (indices.ndim - 1)), head_size, bits)
    bit_index = bit_index.reshape(*((1,) * (indices.ndim - 1)), head_size, bits)
    contrib = (
        ((indices.unsqueeze(-1).to(torch.int32) >> bit_offsets) & 1) << bit_index
    ).to(torch.int32)
    contrib = contrib.reshape(*indices.shape[:-1], head_size * bits)
    packed = torch.zeros(
        (*indices.shape[:-1], packed_bytes),
        dtype=torch.int32,
        device=indices.device,
    )
    scatter_index = byte_index.reshape(
        *((1,) * (indices.ndim - 1)),
        head_size * bits,
    ).expand_as(contrib)
    packed.scatter_add_(-1, scatter_index, contrib)
    return packed.to(torch.uint8)


def unpack_turboquant_indices(
    packed: torch.Tensor,
    head_size: int,
    bits: int,
) -> torch.Tensor:
    if bits == 0:
        return torch.zeros(
            (*packed.shape[:-1], head_size),
            dtype=torch.uint8,
            device=packed.device,
        )

    bit_offsets, byte_index, bit_index = _bit_layout(
        packed.device.type,
        packed.device.index,
        head_size,
        bits,
    )
    gather_index = byte_index.reshape(
        *((1,) * (packed.ndim - 1)),
        head_size * bits,
    ).expand(*packed.shape[:-1], head_size * bits)
    selected = packed.to(torch.int32).gather(-1, gather_index)
    bits_view = ((selected >> bit_index) & 1).reshape(
        *packed.shape[:-1], head_size, bits
    )
    bit_offsets = bit_offsets.reshape(*((1,) * (packed.ndim - 1)), head_size, bits)
    values = (bits_view << bit_offsets).sum(dim=-1)
    return values.to(torch.uint8)


def _norms_to_bytes(norms: torch.Tensor, byte_width: int) -> torch.Tensor:
    norm_half = norms.to(torch.float16).contiguous()
    return norm_half.reshape(-1).view(torch.uint8).reshape(*norm_half.shape, byte_width)


def _bytes_to_norms(norm_bytes: torch.Tensor, byte_width: int) -> torch.Tensor:
    raw = norm_bytes.contiguous().reshape(-1, byte_width).view(torch.float16)
    return raw.reshape(*norm_bytes.shape[:-1], 1).to(torch.float32)


def build_turboquant_outlier_masks(
    x: torch.Tensor,
    kv_cache_dtype: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    outlier_count, _ = get_turboquant_group_dims(x.shape[-1], kv_cache_dtype)
    x_fp32 = x.to(torch.float32)
    score = x_fp32.reshape(-1, x_fp32.shape[-2], x_fp32.shape[-1]).square().mean(dim=0)
    outlier_idx = torch.topk(score, k=outlier_count, dim=-1).indices
    outlier_idx = torch.sort(outlier_idx, dim=-1).values

    all_idx = torch.arange(x.shape[-1], device=x.device, dtype=torch.int64)
    all_idx = all_idx.unsqueeze(0).expand(score.shape[0], -1)
    regular_mask = torch.ones_like(all_idx, dtype=torch.bool)
    regular_mask.scatter_(1, outlier_idx, False)
    regular_idx = all_idx[regular_mask].reshape(score.shape[0], -1)
    return outlier_idx, regular_idx


def _gather_group(
    x: torch.Tensor,
    group_indices: torch.Tensor,
) -> torch.Tensor:
    return torch.gather(
        x,
        dim=-1,
        index=group_indices.unsqueeze(0).expand(x.shape[0], -1, -1),
    )


def apply_turboquant_query_transforms(
    query: torch.Tensor,
    group_indices: tuple[torch.Tensor, torch.Tensor],
    rotations: tuple[torch.Tensor, torch.Tensor],
    qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    kv_head_for_query_head: torch.Tensor | None = None,
    per_query_group_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    query_fp32 = query.to(torch.float32)
    if per_query_group_indices is None:
        assert kv_head_for_query_head is not None
        gathered_indices = tuple(
            group.index_select(0, kv_head_for_query_head) for group in group_indices
        )
    else:
        gathered_indices = per_query_group_indices
    gathered_groups = tuple(
        _gather_group(query_fp32, group) for group in gathered_indices
    )
    q_rot = tuple(
        _apply_mse_transform(group_tensor, rotation)
        for group_tensor, rotation in zip(gathered_groups, rotations, strict=True)
    )
    q_qjl = tuple(
        _apply_qjl_transform(group_tensor, qjl_matrix)
        * (TURBOQUANT_QJL_SCALE / group_tensor.shape[-1])
        for group_tensor, qjl_matrix in zip(gathered_groups, qjl_matrices, strict=True)
    )
    return q_rot, q_qjl


def scatter_turboquant_output(
    head_size: int,
    dtype: torch.dtype,
    group_outputs: tuple[torch.Tensor, torch.Tensor],
    group_indices: tuple[torch.Tensor, torch.Tensor],
    kv_head_for_query_head: torch.Tensor | None = None,
    per_query_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if out is None:
        output = torch.zeros(
            (*group_outputs[0].shape[:-1], head_size),
            dtype=torch.float32,
            device=group_outputs[0].device,
        )
    else:
        output = out
        output.zero_()

    if per_query_indices is None:
        assert kv_head_for_query_head is not None
        per_query_indices = tuple(
            group.index_select(0, kv_head_for_query_head) for group in group_indices
        )
    for group_output, indices in zip(group_outputs, per_query_indices, strict=True):
        src = group_output.to(dtype=output.dtype) if out is not None else group_output
        output.scatter_add_(
            -1,
            indices.unsqueeze(0).expand(group_output.shape[0], -1, -1),
            src,
        )
    if out is not None:
        return out
    return output.to(dtype=dtype)


def quantize_turboquant_vectors(
    x: torch.Tensor,
    kv_cache_dtype: str,
    rotations: tuple[torch.Tensor, torch.Tensor],
    qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    centroids: dict[int, torch.Tensor],
    group_indices: tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    layout = get_turboquant_layout(kv_cache_dtype, x.shape[-1])
    groups = tuple(
        _gather_group(x.to(torch.float32), indices) for indices in group_indices
    )
    packed_groups: list[torch.Tensor] = []

    for group_x, group_layout, rotation, qjl_matrix in zip(
        groups,
        layout.groups,
        rotations,
        qjl_matrices,
        strict=True,
    ):
        vector_norms = group_x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        unit = group_x / vector_norms
        rotated = _apply_mse_transform(unit, rotation)
        indices = torch.zeros_like(rotated, dtype=torch.uint8)
        rotated_hat = torch.zeros_like(rotated, dtype=torch.float32)

        if group_layout.mse_bits > 0:
            mse_centroids = centroids[group_layout.mse_bits]
            mse_indices = torch.abs(rotated.unsqueeze(-1) - mse_centroids).argmin(
                dim=-1
            )
            indices = mse_indices.to(torch.uint8)
            rotated_hat = mse_centroids[mse_indices.long()]

        mse_hat = _apply_mse_inverse_transform(rotated_hat, rotation)
        residual = unit - mse_hat
        residual_norms = residual.norm(dim=-1, keepdim=True)
        qjl_bits = (_apply_qjl_transform(residual, qjl_matrix) >= 0).to(torch.uint8)
        packed_groups.append(
            torch.cat(
                (
                    pack_turboquant_indices(indices, group_layout.mse_bits),
                    pack_turboquant_indices(qjl_bits, 1),
                    _norms_to_bytes(
                        vector_norms.squeeze(-1),
                        TURBOQUANT_VECTOR_NORM_BYTES,
                    ),
                    _norms_to_bytes(
                        residual_norms.squeeze(-1),
                        TURBOQUANT_RESIDUAL_NORM_BYTES,
                    ),
                ),
                dim=-1,
            )
        )

    return torch.cat(packed_groups, dim=-1)


def dequantize_turboquant_vectors(
    packed: torch.Tensor,
    kv_cache_dtype: str,
    head_size: int,
    rotations: tuple[torch.Tensor, torch.Tensor],
    qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    centroids: dict[int, torch.Tensor],
    group_indices: tuple[torch.Tensor, torch.Tensor],
    dtype: torch.dtype,
) -> torch.Tensor:
    layout = get_turboquant_layout(kv_cache_dtype, head_size)
    group_outputs: list[torch.Tensor] = []
    cursor = 0
    for group_layout, rotation, qjl_matrix in zip(
        layout.groups,
        rotations,
        qjl_matrices,
        strict=True,
    ):
        group_packed = packed[..., cursor : cursor + group_layout.packed_bytes]
        cursor += group_layout.packed_bytes
        group_cursor = 0
        mse_indices = unpack_turboquant_indices(
            group_packed[
                ..., group_cursor : group_cursor + group_layout.mse_payload_bytes
            ],
            group_layout.dim,
            group_layout.mse_bits,
        )
        group_cursor += group_layout.mse_payload_bytes
        qjl_bits = unpack_turboquant_indices(
            group_packed[
                ..., group_cursor : group_cursor + group_layout.qjl_payload_bytes
            ],
            group_layout.dim,
            1,
        )
        group_cursor += group_layout.qjl_payload_bytes
        vector_norms = _bytes_to_norms(
            group_packed[
                ..., group_cursor : group_cursor + TURBOQUANT_VECTOR_NORM_BYTES
            ],
            TURBOQUANT_VECTOR_NORM_BYTES,
        )
        group_cursor += TURBOQUANT_VECTOR_NORM_BYTES
        residual_norms = _bytes_to_norms(
            group_packed[
                ..., group_cursor : group_cursor + TURBOQUANT_RESIDUAL_NORM_BYTES
            ],
            TURBOQUANT_RESIDUAL_NORM_BYTES,
        )

        rotated_hat = torch.zeros(
            (*group_packed.shape[:-1], group_layout.dim),
            dtype=torch.float32,
            device=packed.device,
        )
        if group_layout.mse_bits > 0:
            rotated_hat = centroids[group_layout.mse_bits][mse_indices.long()]

        mse_hat = _apply_mse_inverse_transform(rotated_hat, rotation)
        qjl_signs = qjl_bits.to(torch.float32).mul_(2.0).sub_(1.0)
        qjl_hat = _apply_qjl_inverse_transform(qjl_signs, qjl_matrix) * (
            TURBOQUANT_QJL_SCALE / group_layout.dim
        )
        group_outputs.append((mse_hat + qjl_hat * residual_norms) * vector_norms)

    if packed.shape[-2] != group_indices[0].shape[0]:
        num_heads = packed.shape[-2]
        kv_head_for_query_head = torch.arange(
            num_heads,
            device=packed.device,
            dtype=torch.int64,
        )
        kv_head_for_query_head %= group_indices[0].shape[0]
    else:
        kv_head_for_query_head = torch.arange(
            packed.shape[-2],
            device=packed.device,
            dtype=torch.int64,
        )
    return scatter_turboquant_output(
        head_size=head_size,
        dtype=dtype,
        group_outputs=(group_outputs[0], group_outputs[1]),
        group_indices=group_indices,
        kv_head_for_query_head=kv_head_for_query_head,
    )
