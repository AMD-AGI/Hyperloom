# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fused Triton kernel for TurboQuant quantize-and-store KV cache update.

This module replaces the previous two-step approach (Python-side quantise then
Triton scatter) with a single kernel launch that reads raw fp16 KV vectors,
quantises them on-device, and writes the packed uint8 result directly into the
paged cache.

Encoding per token/head/group
------------------------------
1. Compute the L2 norm of the grouped input slice (||v||).
2. Project the unit vector through the structured Hadamard MSE transform and
   find the nearest centroid for each output coordinate.  Encode centroid
   indices as packed multi-bit integers (MSE_BITS per coordinate).
3. Compute the residual between the unit vector and its MSE approximation.
4. Project the residual through the QJL Hadamard transform.  The sign of each
   output coordinate is stored as a packed 1-bit integer.
5. Write ||v|| and ||residual|| as float16 into the two norm fields.

Layout on-disk (per group, contiguous bytes)
---------------------------------------------
  [mse_payload_bytes]  packed centroid indices (MSE_BITS per coordinate)
  [qjl_payload_bytes]  packed QJL sign bits (1 bit per coordinate)
  [2 bytes]            float16 vector norm
  [2 bytes]            float16 residual norm

Kernel grid: (T, H)
  Axis 0 – one program per token (slot_mapping).
  Axis 1 – one program per KV head.
  Each program processes both group0 and group1 for its (token, head) pair.

TILE (default 32)
  Inner loops tile over the group dimension in chunks of TILE elements.
  Increasing TILE improves register reuse but raises register pressure.
"""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.turboquant_kv_cache import TurboQuantLayout


def _require_gb10_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("TurboQuant Triton KV update requires CUDA tensors.")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 1):
        raise ValueError("TurboQuant KV cache requires NVIDIA GB10 / SM121.")


@triton.jit
def _pack_tile_bits(
    byte_accum,
    byte_positions,
    values,
    offs,
    tile_start: tl.constexpr,
    bits: tl.constexpr,
    dim: tl.constexpr,
    TILE: tl.constexpr,
):
    if bits == 0:
        return byte_accum

    for local_idx in range(TILE):
        dim_idx = tile_start + local_idx
        if dim_idx < dim:
            value = tl.sum(
                tl.where(offs == local_idx, values.to(tl.int32), 0),
                axis=0,
            )
            for bit in range(bits):
                bit_position = dim_idx * bits + bit
                byte_idx = bit_position // 8
                bit_offset = bit_position % 8
                bit_value = ((value >> bit) & 1) << bit_offset
                byte_accum += tl.where(byte_positions == byte_idx, bit_value, 0)
    return byte_accum


@triton.jit
def _store_half_as_bytes(
    cache_ptr,
    token_base,
    stride_cache_d: tl.constexpr,
    byte_offset: tl.constexpr,
    value,
):
    value_half = value.to(tl.float16)
    value_bits = value_half.to(tl.uint16, bitcast=True)
    lo = (value_bits & 0xFF).to(tl.uint8)
    hi = ((value_bits >> 8) & 0xFF).to(tl.uint8)
    tl.store(cache_ptr + token_base + byte_offset * stride_cache_d, lo)
    tl.store(cache_ptr + token_base + (byte_offset + 1) * stride_cache_d, hi)


@triton.jit
def _compute_group_norm_sq(
    x_ptr,
    x_token_base,
    group_idx_ptr,
    group_idx_stride_0,
    group_idx_stride_1,
    x_stride_d,
    head_idx,
    offs,
    TILE: tl.constexpr,
    DIM: tl.constexpr,
):
    norm_sq = 0.0
    NUM_TILES: tl.constexpr = (DIM + TILE - 1) // TILE
    for in_tile_idx in range(NUM_TILES):
        in_start = in_tile_idx * TILE
        in_idx = in_start + offs
        in_mask = in_idx < DIM
        gathered_idx = tl.load(
            group_idx_ptr + head_idx * group_idx_stride_0 + in_idx * group_idx_stride_1,
            mask=in_mask,
            other=0,
        ).to(tl.int64)
        group_tile = tl.load(
            x_ptr + x_token_base + gathered_idx * x_stride_d,
            mask=in_mask,
            other=0.0,
        ).to(tl.float32)
        norm_sq += tl.sum(group_tile * group_tile, axis=0)
    return norm_sq


@triton.jit
def _compute_projected_tile(
    x_ptr,
    x_token_base,
    group_idx_ptr,
    group_idx_stride_0,
    group_idx_stride_1,
    matrix_ptr,
    matrix_stride_0,
    matrix_stride_1,
    x_stride_d,
    head_idx,
    norm,
    offs,
    out_start: tl.constexpr,
    TILE: tl.constexpr,
    DIM: tl.constexpr,
):
    out_idx = out_start + offs
    out_mask = out_idx < DIM
    projected = tl.zeros([TILE], dtype=tl.float32)
    NUM_TILES: tl.constexpr = (DIM + TILE - 1) // TILE
    for in_tile_idx in range(NUM_TILES):
        in_start = in_tile_idx * TILE
        in_idx = in_start + offs
        in_mask = in_idx < DIM
        gathered_idx = tl.load(
            group_idx_ptr + head_idx * group_idx_stride_0 + in_idx * group_idx_stride_1,
            mask=in_mask,
            other=0,
        ).to(tl.int64)
        group_tile = tl.load(
            x_ptr + x_token_base + gathered_idx * x_stride_d,
            mask=in_mask,
            other=0.0,
        ).to(tl.float32)
        unit_tile = group_tile / norm
        matrix_tile = tl.load(
            matrix_ptr
            + in_idx[:, None] * matrix_stride_0
            + out_idx[None, :] * matrix_stride_1,
            mask=in_mask[:, None] & out_mask[None, :],
            other=0.0,
        )
        projected += tl.sum(unit_tile[:, None] * matrix_tile, axis=0)
    return projected


@triton.jit
def _select_centroid_tile(
    projected_tile,
    centroids_ptr,
    offs,
    out_start: tl.constexpr,
    DIM: tl.constexpr,
    LEVELS: tl.constexpr,
    TILE: tl.constexpr,
):
    out_mask = (out_start + offs) < DIM
    centroid_tile = tl.zeros([TILE], dtype=tl.float32)
    centroid_indices = tl.zeros([TILE], dtype=tl.int32)
    first_centroid = tl.load(centroids_ptr)
    best_dist = tl.abs(projected_tile - first_centroid)
    centroid_tile = tl.where(out_mask, first_centroid, 0.0)
    for level in range(1, LEVELS):
        centroid = tl.load(centroids_ptr + level)
        dist = tl.abs(projected_tile - centroid)
        better = dist < best_dist
        best_dist = tl.where(better, dist, best_dist)
        centroid_tile = tl.where(better, centroid, centroid_tile)
        centroid_indices = tl.where(
            better,
            tl.full([offs.shape[0]], level, dtype=tl.int32),
            centroid_indices,
        )
    centroid_indices = tl.where(out_mask, centroid_indices, 0)
    centroid_tile = tl.where(out_mask, centroid_tile, 0.0)
    return centroid_tile, centroid_indices


@triton.jit
def _quantize_group(
    x_ptr,
    cache_ptr,
    token_base,
    x_token_base,
    group_idx_ptr,
    group_idx_stride_0,
    group_idx_stride_1,
    mse_matrix_ptr,
    mse_matrix_stride_0,
    mse_matrix_stride_1,
    qjl_matrix_ptr,
    qjl_matrix_stride_0,
    qjl_matrix_stride_1,
    mse_to_qjl_ptr,
    mse_to_qjl_stride_0,
    mse_to_qjl_stride_1,
    centroids_ptr,
    x_stride_d,
    stride_cache_3,
    head_idx,
    offs,
    TILE: tl.constexpr,
    DIM: tl.constexpr,
    MSE_BITS: tl.constexpr,
    MSE_PAYLOAD_BYTES: tl.constexpr,
    MSE_ACC_BYTES: tl.constexpr,
    QJL_PAYLOAD_BYTES: tl.constexpr,
    QJL_ACC_BYTES: tl.constexpr,
    BASE_OFFSET: tl.constexpr,
    QJL_OFFSET: tl.constexpr,
    VECTOR_NORM_OFFSET: tl.constexpr,
    RESIDUAL_NORM_OFFSET: tl.constexpr,
    LEVELS: tl.constexpr,
):
    norm_sq = _compute_group_norm_sq(
        x_ptr,
        x_token_base,
        group_idx_ptr,
        group_idx_stride_0,
        group_idx_stride_1,
        x_stride_d,
        head_idx,
        offs,
        TILE=TILE,
        DIM=DIM,
    )
    norm = tl.sqrt(tl.maximum(norm_sq, 1e-24))
    unit_norm_sq = norm_sq / (norm * norm)

    mse_byte_positions = tl.arange(0, MSE_ACC_BYTES)
    mse_bytes = tl.zeros([MSE_ACC_BYTES], dtype=tl.int32)
    dot_rot_hat = 0.0
    rot_hat_norm_sq = 0.0

    NUM_TILES: tl.constexpr = (DIM + TILE - 1) // TILE
    for mse_tile_idx in range(NUM_TILES):
        mse_start = mse_tile_idx * TILE
        projected_tile = _compute_projected_tile(
            x_ptr,
            x_token_base,
            group_idx_ptr,
            group_idx_stride_0,
            group_idx_stride_1,
            mse_matrix_ptr,
            mse_matrix_stride_0,
            mse_matrix_stride_1,
            x_stride_d,
            head_idx,
            norm,
            offs,
            out_start=mse_start,
            TILE=TILE,
            DIM=DIM,
        )
        centroid_tile, centroid_indices = _select_centroid_tile(
            projected_tile,
            centroids_ptr,
            offs,
            out_start=mse_start,
            DIM=DIM,
            LEVELS=LEVELS,
            TILE=TILE,
        )
        mse_bytes = _pack_tile_bits(
            mse_bytes,
            mse_byte_positions,
            centroid_indices,
            offs,
            tile_start=mse_start,
            bits=MSE_BITS,
            dim=DIM,
            TILE=TILE,
        )
        dot_rot_hat += tl.sum(projected_tile * centroid_tile, axis=0)
        rot_hat_norm_sq += tl.sum(centroid_tile * centroid_tile, axis=0)

    qjl_byte_positions = tl.arange(0, QJL_ACC_BYTES)
    qjl_bytes = tl.zeros([QJL_ACC_BYTES], dtype=tl.int32)
    for qjl_tile_idx in range(NUM_TILES):
        qjl_start = qjl_tile_idx * TILE
        qjl_tile = _compute_projected_tile(
            x_ptr,
            x_token_base,
            group_idx_ptr,
            group_idx_stride_0,
            group_idx_stride_1,
            qjl_matrix_ptr,
            qjl_matrix_stride_0,
            qjl_matrix_stride_1,
            x_stride_d,
            head_idx,
            norm,
            offs,
            out_start=qjl_start,
            TILE=TILE,
            DIM=DIM,
        )
        qjl_idx = qjl_start + offs
        qjl_mask = qjl_idx < DIM
        for mse_tile_idx in range(NUM_TILES):
            mse_start = mse_tile_idx * TILE
            mse_idx = mse_start + offs
            mse_mask = mse_idx < DIM
            projected_tile = _compute_projected_tile(
                x_ptr,
                x_token_base,
                group_idx_ptr,
                group_idx_stride_0,
                group_idx_stride_1,
                mse_matrix_ptr,
                mse_matrix_stride_0,
                mse_matrix_stride_1,
                x_stride_d,
                head_idx,
                norm,
                offs,
                out_start=mse_start,
                TILE=TILE,
                DIM=DIM,
            )
            centroid_tile, _ = _select_centroid_tile(
                projected_tile,
                centroids_ptr,
                offs,
                out_start=mse_start,
                DIM=DIM,
                LEVELS=LEVELS,
                TILE=TILE,
            )
            matrix_tile = tl.load(
                mse_to_qjl_ptr
                + mse_idx[:, None] * mse_to_qjl_stride_0
                + qjl_idx[None, :] * mse_to_qjl_stride_1,
                mask=mse_mask[:, None] & qjl_mask[None, :],
                other=0.0,
            )
            qjl_tile -= tl.sum(centroid_tile[:, None] * matrix_tile, axis=0)

        qjl_bits = tl.where(qjl_mask & (qjl_tile >= 0.0), 1, 0).to(tl.int32)
        qjl_bytes = _pack_tile_bits(
            qjl_bytes,
            qjl_byte_positions,
            qjl_bits,
            offs,
            tile_start=qjl_start,
            bits=1,
            dim=DIM,
            TILE=TILE,
        )

    residual_norm_sq = tl.maximum(
        unit_norm_sq + rot_hat_norm_sq - 2.0 * dot_rot_hat, 0.0
    )
    residual_norm = tl.sqrt(residual_norm_sq)

    tl.store(
        cache_ptr + token_base + (BASE_OFFSET + mse_byte_positions) * stride_cache_3,
        mse_bytes.to(tl.uint8),
        mask=mse_byte_positions < MSE_PAYLOAD_BYTES,
    )
    tl.store(
        cache_ptr + token_base + (QJL_OFFSET + qjl_byte_positions) * stride_cache_3,
        qjl_bytes.to(tl.uint8),
        mask=qjl_byte_positions < QJL_PAYLOAD_BYTES,
    )
    _store_half_as_bytes(
        cache_ptr,
        token_base,
        stride_cache_3,
        VECTOR_NORM_OFFSET,
        norm,
    )
    _store_half_as_bytes(
        cache_ptr,
        token_base,
        stride_cache_3,
        RESIDUAL_NORM_OFFSET,
        residual_norm,
    )


@triton.jit
def _turboquant_quantize_store_kernel(
    x_ptr,
    cache_ptr,
    slot_mapping_ptr,
    group0_idx_ptr,
    group1_idx_ptr,
    mse_matrix_0_ptr,
    mse_matrix_1_ptr,
    qjl_matrix_0_ptr,
    qjl_matrix_1_ptr,
    mse_to_qjl_0_ptr,
    mse_to_qjl_1_ptr,
    centroids_0_ptr,
    centroids_1_ptr,
    x_stride_t,
    x_stride_h,
    x_stride_d,
    group0_idx_stride_0,
    group0_idx_stride_1,
    group1_idx_stride_0,
    group1_idx_stride_1,
    mse_matrix_0_stride_0,
    mse_matrix_0_stride_1,
    mse_matrix_1_stride_0,
    mse_matrix_1_stride_1,
    qjl_matrix_0_stride_0,
    qjl_matrix_0_stride_1,
    qjl_matrix_1_stride_0,
    qjl_matrix_1_stride_1,
    mse_to_qjl_0_stride_0,
    mse_to_qjl_0_stride_1,
    mse_to_qjl_1_stride_0,
    mse_to_qjl_1_stride_1,
    stride_cache_0,
    stride_cache_1,
    stride_cache_2,
    stride_cache_3,
    block_size: tl.constexpr,
    TILE: tl.constexpr,
    G0_DIM: tl.constexpr,
    G0_MSE_BITS: tl.constexpr,
    G0_MSE_PAYLOAD_BYTES: tl.constexpr,
    G0_MSE_ACC_BYTES: tl.constexpr,
    G0_QJL_PAYLOAD_BYTES: tl.constexpr,
    G0_QJL_ACC_BYTES: tl.constexpr,
    G0_BASE_OFFSET: tl.constexpr,
    G0_QJL_OFFSET: tl.constexpr,
    G0_VECTOR_NORM_OFFSET: tl.constexpr,
    G0_RESIDUAL_NORM_OFFSET: tl.constexpr,
    G0_LEVELS: tl.constexpr,
    G1_DIM: tl.constexpr,
    G1_MSE_BITS: tl.constexpr,
    G1_MSE_PAYLOAD_BYTES: tl.constexpr,
    G1_MSE_ACC_BYTES: tl.constexpr,
    G1_QJL_PAYLOAD_BYTES: tl.constexpr,
    G1_QJL_ACC_BYTES: tl.constexpr,
    G1_BASE_OFFSET: tl.constexpr,
    G1_QJL_OFFSET: tl.constexpr,
    G1_VECTOR_NORM_OFFSET: tl.constexpr,
    G1_RESIDUAL_NORM_OFFSET: tl.constexpr,
    G1_LEVELS: tl.constexpr,
):
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    slot_idx = tl.load(slot_mapping_ptr + token_idx).to(tl.int64)
    if slot_idx < 0:
        return

    block_idx = slot_idx // block_size
    block_offset = slot_idx % block_size
    token_base = (
        block_idx * stride_cache_0
        + block_offset * stride_cache_1
        + head_idx * stride_cache_2
    )
    x_token_base = token_idx * x_stride_t + head_idx * x_stride_h
    offs = tl.arange(0, TILE)

    _quantize_group(
        x_ptr,
        cache_ptr,
        token_base,
        x_token_base,
        group0_idx_ptr,
        group0_idx_stride_0,
        group0_idx_stride_1,
        mse_matrix_0_ptr,
        mse_matrix_0_stride_0,
        mse_matrix_0_stride_1,
        qjl_matrix_0_ptr,
        qjl_matrix_0_stride_0,
        qjl_matrix_0_stride_1,
        mse_to_qjl_0_ptr,
        mse_to_qjl_0_stride_0,
        mse_to_qjl_0_stride_1,
        centroids_0_ptr,
        x_stride_d,
        stride_cache_3,
        head_idx,
        offs,
        TILE=TILE,
        DIM=G0_DIM,
        MSE_BITS=G0_MSE_BITS,
        MSE_PAYLOAD_BYTES=G0_MSE_PAYLOAD_BYTES,
        MSE_ACC_BYTES=G0_MSE_ACC_BYTES,
        QJL_PAYLOAD_BYTES=G0_QJL_PAYLOAD_BYTES,
        QJL_ACC_BYTES=G0_QJL_ACC_BYTES,
        BASE_OFFSET=G0_BASE_OFFSET,
        QJL_OFFSET=G0_QJL_OFFSET,
        VECTOR_NORM_OFFSET=G0_VECTOR_NORM_OFFSET,
        RESIDUAL_NORM_OFFSET=G0_RESIDUAL_NORM_OFFSET,
        LEVELS=G0_LEVELS,
    )
    _quantize_group(
        x_ptr,
        cache_ptr,
        token_base,
        x_token_base,
        group1_idx_ptr,
        group1_idx_stride_0,
        group1_idx_stride_1,
        mse_matrix_1_ptr,
        mse_matrix_1_stride_0,
        mse_matrix_1_stride_1,
        qjl_matrix_1_ptr,
        qjl_matrix_1_stride_0,
        qjl_matrix_1_stride_1,
        mse_to_qjl_1_ptr,
        mse_to_qjl_1_stride_0,
        mse_to_qjl_1_stride_1,
        centroids_1_ptr,
        x_stride_d,
        stride_cache_3,
        head_idx,
        offs,
        TILE=TILE,
        DIM=G1_DIM,
        MSE_BITS=G1_MSE_BITS,
        MSE_PAYLOAD_BYTES=G1_MSE_PAYLOAD_BYTES,
        MSE_ACC_BYTES=G1_MSE_ACC_BYTES,
        QJL_PAYLOAD_BYTES=G1_QJL_PAYLOAD_BYTES,
        QJL_ACC_BYTES=G1_QJL_ACC_BYTES,
        BASE_OFFSET=G1_BASE_OFFSET,
        QJL_OFFSET=G1_QJL_OFFSET,
        VECTOR_NORM_OFFSET=G1_VECTOR_NORM_OFFSET,
        RESIDUAL_NORM_OFFSET=G1_RESIDUAL_NORM_OFFSET,
        LEVELS=G1_LEVELS,
    )


def turboquant_write_packed_kv(
    x: torch.Tensor,
    cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    layout: TurboQuantLayout,
    group_indices: tuple[torch.Tensor, torch.Tensor],
    mse_transform_matrices: tuple[torch.Tensor, torch.Tensor],
    qjl_transform_matrices: tuple[torch.Tensor, torch.Tensor],
    mse_to_qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    centroids: dict[int, torch.Tensor],
) -> None:
    """Quantize raw KV vectors and write them directly into the paged cache."""
    if x.numel() == 0:
        return

    _require_gb10_cuda(x.device)
    if cache.dtype != torch.uint8:
        raise ValueError("TurboQuant KV cache update expects uint8 cache storage.")
    if x.ndim != 3:
        raise ValueError(f"Expected input shape [T, H, D], got {x.shape}")
    if cache.ndim != 4:
        raise ValueError(
            "Expected cache shape [num_blocks, block_size, num_kv_heads, packed_dim], "
            f"got {cache.shape}"
        )
    if cache.shape[2] != x.shape[1]:
        raise ValueError("TurboQuant cache head count does not match the input.")
    if cache.shape[3] != layout.packed_dim:
        raise ValueError("TurboQuant cache packed_dim does not match the layout.")
    if slot_mapping.ndim != 1 or slot_mapping.shape[0] != x.shape[0]:
        raise ValueError("slot_mapping must be a 1D tensor aligned with input tokens.")
    if (
        group_indices[0].shape[0] != x.shape[1]
        or group_indices[1].shape[0] != x.shape[1]
    ):
        raise ValueError("TurboQuant group metadata must match the KV head count.")

    group0, group1 = layout.groups
    tile = 32
    grid = (x.shape[0], x.shape[1])
    _turboquant_quantize_store_kernel[grid](
        x_ptr=x,
        cache_ptr=cache,
        slot_mapping_ptr=slot_mapping,
        group0_idx_ptr=group_indices[0],
        group1_idx_ptr=group_indices[1],
        mse_matrix_0_ptr=mse_transform_matrices[0],
        mse_matrix_1_ptr=mse_transform_matrices[1],
        qjl_matrix_0_ptr=qjl_transform_matrices[0],
        qjl_matrix_1_ptr=qjl_transform_matrices[1],
        mse_to_qjl_0_ptr=mse_to_qjl_matrices[0],
        mse_to_qjl_1_ptr=mse_to_qjl_matrices[1],
        centroids_0_ptr=centroids[group0.mse_bits],
        centroids_1_ptr=centroids[group1.mse_bits],
        x_stride_t=x.stride(0),
        x_stride_h=x.stride(1),
        x_stride_d=x.stride(2),
        group0_idx_stride_0=group_indices[0].stride(0),
        group0_idx_stride_1=group_indices[0].stride(1),
        group1_idx_stride_0=group_indices[1].stride(0),
        group1_idx_stride_1=group_indices[1].stride(1),
        mse_matrix_0_stride_0=mse_transform_matrices[0].stride(0),
        mse_matrix_0_stride_1=mse_transform_matrices[0].stride(1),
        mse_matrix_1_stride_0=mse_transform_matrices[1].stride(0),
        mse_matrix_1_stride_1=mse_transform_matrices[1].stride(1),
        qjl_matrix_0_stride_0=qjl_transform_matrices[0].stride(0),
        qjl_matrix_0_stride_1=qjl_transform_matrices[0].stride(1),
        qjl_matrix_1_stride_0=qjl_transform_matrices[1].stride(0),
        qjl_matrix_1_stride_1=qjl_transform_matrices[1].stride(1),
        mse_to_qjl_0_stride_0=mse_to_qjl_matrices[0].stride(0),
        mse_to_qjl_0_stride_1=mse_to_qjl_matrices[0].stride(1),
        mse_to_qjl_1_stride_0=mse_to_qjl_matrices[1].stride(0),
        mse_to_qjl_1_stride_1=mse_to_qjl_matrices[1].stride(1),
        stride_cache_0=cache.stride(0),
        stride_cache_1=cache.stride(1),
        stride_cache_2=cache.stride(2),
        stride_cache_3=cache.stride(3),
        block_size=cache.shape[1],
        TILE=tile,
        G0_DIM=group0.dim,
        G0_MSE_BITS=group0.mse_bits,
        G0_MSE_PAYLOAD_BYTES=group0.mse_payload_bytes,
        G0_MSE_ACC_BYTES=triton.next_power_of_2(group0.mse_payload_bytes),
        G0_QJL_PAYLOAD_BYTES=group0.qjl_payload_bytes,
        G0_QJL_ACC_BYTES=triton.next_power_of_2(group0.qjl_payload_bytes),
        G0_BASE_OFFSET=0,
        G0_QJL_OFFSET=group0.qjl_offset,
        G0_VECTOR_NORM_OFFSET=group0.vector_norm_offset,
        G0_RESIDUAL_NORM_OFFSET=group0.residual_norm_offset,
        G0_LEVELS=1 << group0.mse_bits,
        G1_DIM=group1.dim,
        G1_MSE_BITS=group1.mse_bits,
        G1_MSE_PAYLOAD_BYTES=group1.mse_payload_bytes,
        G1_MSE_ACC_BYTES=triton.next_power_of_2(group1.mse_payload_bytes),
        G1_QJL_PAYLOAD_BYTES=group1.qjl_payload_bytes,
        G1_QJL_ACC_BYTES=triton.next_power_of_2(group1.qjl_payload_bytes),
        G1_BASE_OFFSET=group0.packed_bytes,
        G1_QJL_OFFSET=group1.qjl_offset,
        G1_VECTOR_NORM_OFFSET=group1.vector_norm_offset,
        G1_RESIDUAL_NORM_OFFSET=group1.residual_norm_offset,
        G1_LEVELS=1 << group1.mse_bits,
        num_warps=4,
        num_stages=2,
    )
