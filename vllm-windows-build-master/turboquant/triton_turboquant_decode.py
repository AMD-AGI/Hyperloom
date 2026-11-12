# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from functools import cache

import torch

from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.turboquant_kv_cache import (
    TURBOQUANT_QJL_SCALE,
    apply_turboquant_query_transforms,
    get_turboquant_layout,
    get_turboquant_mse_inverse_transform_matrix,
    get_turboquant_qjl_inverse_transform_matrix,
)


@cache
def _norm_lut(device_type: str, device_index: int | None) -> torch.Tensor:
    device = torch.device(device_type, device_index)
    values = torch.arange(1 << 16, dtype=torch.int32, device=device)
    return values.to(torch.int16).view(torch.float16).to(torch.float32)


def get_turboquant_norm_lut(device: torch.device) -> torch.Tensor:
    return _norm_lut(device.type, device.index)


def _require_gb10_cuda(device: torch.device) -> None:
    if device.type != "cuda":
        raise ValueError("TurboQuant Triton decode requires CUDA tensors.")
    capability = torch.cuda.get_device_capability(device)
    if capability != (12, 1):
        raise ValueError("TurboQuant KV cache requires NVIDIA GB10 / SM121.")


@triton.jit
def _apply_softcap(logits, softcap):
    scaled = logits / softcap
    return softcap * (2 * tl.sigmoid(2 * scaled) - 1)


@triton.jit
def _load_half_from_bytes(
    cache_ptr,
    token_base,
    lut_ptr,
    byte_offset: tl.constexpr,
    stride_cache_d: tl.constexpr,
):
    lo = tl.load(
        cache_ptr + token_base + byte_offset * stride_cache_d,
        mask=True,
        other=0,
    ).to(tl.int32)
    hi = tl.load(
        cache_ptr + token_base + (byte_offset + 1) * stride_cache_d,
        mask=True,
        other=0,
    ).to(tl.int32)
    return tl.load(lut_ptr + lo + (hi << 8), mask=True, other=0.0)


@triton.jit
def _unpack_fixed_indices(
    cache_ptr,
    token_base,
    offs_d,
    stride_cache_d: tl.constexpr,
    bits: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_SIZE_PADDED: tl.constexpr,
    base_offset: tl.constexpr,
    mask_d,
):
    indices = tl.zeros([BLOCK_N, HEAD_SIZE_PADDED], dtype=tl.int32)
    if bits == 0:
        return indices

    for bit in range(bits):
        bit_position = offs_d[None, :] * bits + bit
        byte_offset = base_offset + bit_position // 8
        bit_offset = bit_position % 8
        byte = tl.load(
            cache_ptr + token_base[:, None] + byte_offset * stride_cache_d,
            mask=mask_d,
            other=0,
        ).to(tl.int32)
        indices += ((byte >> bit_offset) & 1) << bit
    return indices


@triton.jit
def _unpack_signs(
    cache_ptr,
    token_base,
    offs_d,
    stride_cache_d: tl.constexpr,
    byte_offset: tl.constexpr,
    mask_d,
):
    bit_position = offs_d[None, :]
    qjl_byte_offset = byte_offset + bit_position // 8
    qjl_bit_offset = bit_position % 8
    byte = tl.load(
        cache_ptr + token_base[:, None] + qjl_byte_offset * stride_cache_d,
        mask=mask_d,
        other=0,
    ).to(tl.int32)
    bits = (byte >> qjl_bit_offset) & 1
    return bits.to(tl.float32) * 2.0 - 1.0


@triton.jit
def _turboquant_decode_kernel(
    q_rot_0_ptr,
    q_qjl_0_ptr,
    q_rot_1_ptr,
    q_qjl_1_ptr,
    key_cache_ptr,
    value_cache_ptr,
    block_table_ptr,
    token_seq_ids_ptr,
    token_kv_lens_ptr,
    token_query_positions_ptr,
    acc_mse_0_ptr,
    acc_qjl_0_ptr,
    acc_mse_1_ptr,
    acc_qjl_1_ptr,
    lse_ptr,
    norm_lut_ptr,
    centroids_0_ptr,
    centroids_1_ptr,
    sink_ptr,
    mm_prefix_range_ptr,
    softmax_scale,
    softcap,
    q0_stride_0,
    q0_stride_1,
    q1_stride_0,
    q1_stride_1,
    stride_k_cache_0,
    stride_k_cache_1,
    stride_k_cache_2,
    stride_k_cache_3,
    stride_v_cache_0,
    stride_v_cache_1,
    stride_v_cache_2,
    stride_v_cache_3,
    block_table_stride,
    acc0_stride_0,
    acc0_stride_1,
    acc1_stride_0,
    acc1_stride_1,
    lse_stride_0,
    lse_stride_1,
    kv_group_num: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    USE_SOFTCAP: tl.constexpr,
    USE_SINKS: tl.constexpr,
    USE_MM_PREFIX: tl.constexpr,
    MAX_MM_RANGES: tl.constexpr,
    SLIDING_WINDOW: tl.constexpr,
    RETURN_LSE: tl.constexpr,
    G0_DIM: tl.constexpr,
    G0_PADDED: tl.constexpr,
    G0_MSE_BITS: tl.constexpr,
    G0_GROUP_OFFSET: tl.constexpr,
    G0_QJL_OFFSET: tl.constexpr,
    G0_VECTOR_NORM_OFFSET: tl.constexpr,
    G0_RESIDUAL_NORM_OFFSET: tl.constexpr,
    G0_QJL_SCALE: tl.constexpr,
    G1_DIM: tl.constexpr,
    G1_PADDED: tl.constexpr,
    G1_MSE_BITS: tl.constexpr,
    G1_GROUP_OFFSET: tl.constexpr,
    G1_QJL_OFFSET: tl.constexpr,
    G1_VECTOR_NORM_OFFSET: tl.constexpr,
    G1_RESIDUAL_NORM_OFFSET: tl.constexpr,
    G1_QJL_SCALE: tl.constexpr,
):
    cur_token = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_kv_head = cur_head // kv_group_num

    seq_idx = tl.load(token_seq_ids_ptr + cur_token)
    seq_len = tl.load(token_kv_lens_ptr + cur_token)
    query_pos = tl.load(token_query_positions_ptr + cur_token)

    offs_d0 = tl.arange(0, G0_PADDED)
    mask_d0 = offs_d0 < G0_DIM
    offs_d1 = tl.arange(0, G1_PADDED)
    mask_d1 = offs_d1 < G1_DIM

    q_rot_0 = tl.load(
        q_rot_0_ptr + cur_token * q0_stride_0 + cur_head * q0_stride_1 + offs_d0,
        mask=mask_d0,
        other=0.0,
    )
    q_qjl_0 = tl.load(
        q_qjl_0_ptr + cur_token * q0_stride_0 + cur_head * q0_stride_1 + offs_d0,
        mask=mask_d0,
        other=0.0,
    )
    q_rot_1 = tl.load(
        q_rot_1_ptr + cur_token * q1_stride_0 + cur_head * q1_stride_1 + offs_d1,
        mask=mask_d1,
        other=0.0,
    )
    q_qjl_1 = tl.load(
        q_qjl_1_ptr + cur_token * q1_stride_0 + cur_head * q1_stride_1 + offs_d1,
        mask=mask_d1,
        other=0.0,
    )

    if USE_SINKS:
        e_max = tl.load(sink_ptr + cur_head).to(tl.float32)
        e_sum = tl.where(e_max > float("-inf"), 1.0, 0.0)
    else:
        e_max = -float("inf")
        e_sum = 0.0
    acc_mse_0 = tl.zeros([G0_PADDED], dtype=tl.float32)
    acc_qjl_0 = tl.zeros([G0_PADDED], dtype=tl.float32)
    acc_mse_1 = tl.zeros([G1_PADDED], dtype=tl.float32)
    acc_qjl_1 = tl.zeros([G1_PADDED], dtype=tl.float32)

    for start_n in range(0, seq_len, BLOCK_N):
        offs_n = start_n + tl.arange(0, BLOCK_N)
        mask_n = offs_n < seq_len
        block_ids = tl.load(
            block_table_ptr + seq_idx * block_table_stride + offs_n // BLOCK_SIZE,
            mask=mask_n,
            other=0,
        )
        block_offsets = offs_n % BLOCK_SIZE

        key_token_base = (
            block_ids * stride_k_cache_0
            + block_offsets * stride_k_cache_1
            + cur_kv_head * stride_k_cache_2
        )
        value_token_base = (
            block_ids * stride_v_cache_0
            + block_offsets * stride_v_cache_1
            + cur_kv_head * stride_v_cache_2
        )

        key_qjl_signs_0 = _unpack_signs(
            key_cache_ptr,
            key_token_base,
            offs_d0,
            stride_k_cache_3,
            G0_QJL_OFFSET,
            mask_n[:, None] & mask_d0[None, :],
        )
        key_vector_norm_0 = _load_half_from_bytes(
            key_cache_ptr,
            key_token_base,
            norm_lut_ptr,
            G0_VECTOR_NORM_OFFSET,
            stride_k_cache_3,
        )
        key_residual_norm_0 = _load_half_from_bytes(
            key_cache_ptr,
            key_token_base,
            norm_lut_ptr,
            G0_RESIDUAL_NORM_OFFSET,
            stride_k_cache_3,
        )
        key_logits = (
            key_vector_norm_0
            * key_residual_norm_0
            * tl.sum(
                key_qjl_signs_0 * q_qjl_0[None, :],
                axis=1,
            )
        )
        if G0_MSE_BITS > 0:
            key_indices_0 = _unpack_fixed_indices(
                key_cache_ptr,
                key_token_base,
                offs_d0,
                stride_k_cache_3,
                G0_MSE_BITS,
                BLOCK_N,
                G0_PADDED,
                G0_GROUP_OFFSET,
                mask_n[:, None] & mask_d0[None, :],
            )
            key_centroids_0 = tl.load(
                centroids_0_ptr + key_indices_0,
                mask=mask_n[:, None] & mask_d0[None, :],
                other=0.0,
            )
            key_logits += key_vector_norm_0 * tl.sum(
                key_centroids_0 * q_rot_0[None, :],
                axis=1,
            )

        key_qjl_signs_1 = _unpack_signs(
            key_cache_ptr,
            key_token_base,
            offs_d1,
            stride_k_cache_3,
            G1_QJL_OFFSET,
            mask_n[:, None] & mask_d1[None, :],
        )
        key_vector_norm_1 = _load_half_from_bytes(
            key_cache_ptr,
            key_token_base,
            norm_lut_ptr,
            G1_VECTOR_NORM_OFFSET,
            stride_k_cache_3,
        )
        key_residual_norm_1 = _load_half_from_bytes(
            key_cache_ptr,
            key_token_base,
            norm_lut_ptr,
            G1_RESIDUAL_NORM_OFFSET,
            stride_k_cache_3,
        )
        key_logits += (
            key_vector_norm_1
            * key_residual_norm_1
            * tl.sum(
                key_qjl_signs_1 * q_qjl_1[None, :],
                axis=1,
            )
        )
        if G1_MSE_BITS > 0:
            key_indices_1 = _unpack_fixed_indices(
                key_cache_ptr,
                key_token_base,
                offs_d1,
                stride_k_cache_3,
                G1_MSE_BITS,
                BLOCK_N,
                G1_PADDED,
                G1_GROUP_OFFSET,
                mask_n[:, None] & mask_d1[None, :],
            )
            key_centroids_1 = tl.load(
                centroids_1_ptr + key_indices_1,
                mask=mask_n[:, None] & mask_d1[None, :],
                other=0.0,
            )
            key_logits += key_vector_norm_1 * tl.sum(
                key_centroids_1 * q_rot_1[None, :],
                axis=1,
            )

        logits = key_logits * softmax_scale
        if USE_SOFTCAP:
            logits = _apply_softcap(logits, softcap)
        valid_mask = mask_n
        if CAUSAL:
            valid_mask = valid_mask & (offs_n <= query_pos)
            if SLIDING_WINDOW > 0:
                valid_mask = valid_mask & ((query_pos - offs_n) < SLIDING_WINDOW)
        if USE_MM_PREFIX:
            for i in range(MAX_MM_RANGES):
                range_start = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2
                )
                range_end = tl.load(
                    mm_prefix_range_ptr + seq_idx * MAX_MM_RANGES * 2 + i * 2 + 1
                )
                is_valid = range_start < range_end
                q_in_range = (
                    (query_pos >= range_start)
                    & (query_pos <= range_end)
                    & is_valid
                )
                k_in_range = (
                    (offs_n >= range_start)
                    & (offs_n <= range_end)
                    & is_valid
                )
                valid_mask = valid_mask | (q_in_range & k_in_range)
        logits = tl.where(valid_mask, logits, float("-inf"))

        n_e_max = tl.maximum(tl.max(logits, axis=0), e_max)
        re_scale = tl.exp(e_max - n_e_max)
        p = tl.exp(logits - n_e_max)

        value_vector_norm_0 = _load_half_from_bytes(
            value_cache_ptr,
            value_token_base,
            norm_lut_ptr,
            G0_VECTOR_NORM_OFFSET,
            stride_v_cache_3,
        )
        value_residual_norm_0 = _load_half_from_bytes(
            value_cache_ptr,
            value_token_base,
            norm_lut_ptr,
            G0_RESIDUAL_NORM_OFFSET,
            stride_v_cache_3,
        )
        value_vector_norm_1 = _load_half_from_bytes(
            value_cache_ptr,
            value_token_base,
            norm_lut_ptr,
            G1_VECTOR_NORM_OFFSET,
            stride_v_cache_3,
        )
        value_residual_norm_1 = _load_half_from_bytes(
            value_cache_ptr,
            value_token_base,
            norm_lut_ptr,
            G1_RESIDUAL_NORM_OFFSET,
            stride_v_cache_3,
        )

        acc_mse_0 *= re_scale
        acc_qjl_0 *= re_scale
        acc_mse_1 *= re_scale
        acc_qjl_1 *= re_scale

        if G0_MSE_BITS > 0:
            value_indices_0 = _unpack_fixed_indices(
                value_cache_ptr,
                value_token_base,
                offs_d0,
                stride_v_cache_3,
                G0_MSE_BITS,
                BLOCK_N,
                G0_PADDED,
                G0_GROUP_OFFSET,
                mask_n[:, None] & mask_d0[None, :],
            )
            value_centroids_0 = tl.load(
                centroids_0_ptr + value_indices_0,
                mask=mask_n[:, None] & mask_d0[None, :],
                other=0.0,
            )
            acc_mse_0 += tl.sum(
                (p * value_vector_norm_0)[:, None] * value_centroids_0,
                axis=0,
            )

        value_qjl_signs_0 = _unpack_signs(
            value_cache_ptr,
            value_token_base,
            offs_d0,
            stride_v_cache_3,
            G0_QJL_OFFSET,
            mask_n[:, None] & mask_d0[None, :],
        )
        acc_qjl_0 += tl.sum(
            (p * value_vector_norm_0 * value_residual_norm_0 * G0_QJL_SCALE)[:, None]
            * value_qjl_signs_0,
            axis=0,
        )

        if G1_MSE_BITS > 0:
            value_indices_1 = _unpack_fixed_indices(
                value_cache_ptr,
                value_token_base,
                offs_d1,
                stride_v_cache_3,
                G1_MSE_BITS,
                BLOCK_N,
                G1_PADDED,
                G1_GROUP_OFFSET,
                mask_n[:, None] & mask_d1[None, :],
            )
            value_centroids_1 = tl.load(
                centroids_1_ptr + value_indices_1,
                mask=mask_n[:, None] & mask_d1[None, :],
                other=0.0,
            )
            acc_mse_1 += tl.sum(
                (p * value_vector_norm_1)[:, None] * value_centroids_1,
                axis=0,
            )

        value_qjl_signs_1 = _unpack_signs(
            value_cache_ptr,
            value_token_base,
            offs_d1,
            stride_v_cache_3,
            G1_QJL_OFFSET,
            mask_n[:, None] & mask_d1[None, :],
        )
        acc_qjl_1 += tl.sum(
            (p * value_vector_norm_1 * value_residual_norm_1 * G1_QJL_SCALE)[:, None]
            * value_qjl_signs_1,
            axis=0,
        )

        e_sum = e_sum * re_scale + tl.sum(p, axis=0)
        e_max = n_e_max

    acc_mse_0 = acc_mse_0 / e_sum
    acc_qjl_0 = acc_qjl_0 / e_sum
    acc_mse_1 = acc_mse_1 / e_sum
    acc_qjl_1 = acc_qjl_1 / e_sum

    tl.store(
        acc_mse_0_ptr + cur_token * acc0_stride_0 + cur_head * acc0_stride_1 + offs_d0,
        acc_mse_0,
        mask=mask_d0,
    )
    tl.store(
        acc_qjl_0_ptr + cur_token * acc0_stride_0 + cur_head * acc0_stride_1 + offs_d0,
        acc_qjl_0,
        mask=mask_d0,
    )
    tl.store(
        acc_mse_1_ptr + cur_token * acc1_stride_0 + cur_head * acc1_stride_1 + offs_d1,
        acc_mse_1,
        mask=mask_d1,
    )
    tl.store(
        acc_qjl_1_ptr + cur_token * acc1_stride_0 + cur_head * acc1_stride_1 + offs_d1,
        acc_qjl_1,
        mask=mask_d1,
    )
    if RETURN_LSE:
        tl.store(
            lse_ptr + cur_head * lse_stride_0 + cur_token * lse_stride_1,
            tl.log(e_sum) + e_max,
        )


@triton.jit
def _turboquant_decode_postprocess_group_kernel(
    acc_mse_ptr,
    acc_qjl_ptr,
    mse_inverse_ptr,
    qjl_inverse_ptr,
    group_indices_ptr,
    out_ptr,
    acc_stride_0,
    acc_stride_1,
    mse_inverse_stride_0,
    mse_inverse_stride_1,
    qjl_inverse_stride_0,
    qjl_inverse_stride_1,
    group_indices_stride_0,
    out_stride_0,
    out_stride_1,
    out_stride_2,
    DIM: tl.constexpr,
    DIM_PADDED: tl.constexpr,
    BLOCK_OUT: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    cur_token = tl.program_id(0)
    cur_head = tl.program_id(1)
    cur_tile = tl.program_id(2)

    offs_out = cur_tile * BLOCK_OUT + tl.arange(0, BLOCK_OUT)
    mask_out = offs_out < DIM
    reconstructed = tl.zeros([BLOCK_OUT], dtype=tl.float32)

    for start_k in range(0, DIM_PADDED, BLOCK_K):
        offs_k = start_k + tl.arange(0, BLOCK_K)
        mask_k = offs_k < DIM
        acc_mse = tl.load(
            acc_mse_ptr + cur_token * acc_stride_0 + cur_head * acc_stride_1 + offs_k,
            mask=mask_k,
            other=0.0,
        )
        acc_qjl = tl.load(
            acc_qjl_ptr + cur_token * acc_stride_0 + cur_head * acc_stride_1 + offs_k,
            mask=mask_k,
            other=0.0,
        )
        mse_inverse = tl.load(
            mse_inverse_ptr
            + offs_k[:, None] * mse_inverse_stride_0
            + offs_out[None, :] * mse_inverse_stride_1,
            mask=mask_k[:, None] & mask_out[None, :],
            other=0.0,
        )
        qjl_inverse = tl.load(
            qjl_inverse_ptr
            + offs_k[:, None] * qjl_inverse_stride_0
            + offs_out[None, :] * qjl_inverse_stride_1,
            mask=mask_k[:, None] & mask_out[None, :],
            other=0.0,
        )
        reconstructed += tl.sum(acc_mse[:, None] * mse_inverse, axis=0)
        reconstructed += tl.sum(acc_qjl[:, None] * qjl_inverse, axis=0)

    out_indices = tl.load(
        group_indices_ptr + cur_head * group_indices_stride_0 + offs_out,
        mask=mask_out,
        other=0,
    )
    tl.store(
        out_ptr
        + cur_token * out_stride_0
        + cur_head * out_stride_1
        + out_indices * out_stride_2,
        reconstructed,
        mask=mask_out,
    )


def turboquant_decode_attention_fwd(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    key_group_indices: tuple[torch.Tensor, torch.Tensor],
    value_group_indices: tuple[torch.Tensor, torch.Tensor],
    key_rotations: tuple[torch.Tensor, torch.Tensor],
    key_qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    value_rotations: tuple[torch.Tensor, torch.Tensor],
    value_qjl_matrices: tuple[torch.Tensor, torch.Tensor],
    centroids: dict[int, torch.Tensor],
    norm_lut: torch.Tensor,
    softmax_scale: float,
    kv_cache_dtype: str,
    token_seq_ids: torch.Tensor | None = None,
    token_kv_lens: torch.Tensor | None = None,
    token_query_positions: torch.Tensor | None = None,
    kv_head_for_query_head: torch.Tensor | None = None,
    key_query_group_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    value_query_group_indices: tuple[torch.Tensor, torch.Tensor] | None = None,
    value_mse_inverse_matrices: tuple[torch.Tensor, torch.Tensor] | None = None,
    value_qjl_inverse_matrices: tuple[torch.Tensor, torch.Tensor] | None = None,
    causal: bool = True,
    sliding_window: tuple[int, int] = (-1, -1),
    sinks: torch.Tensor | None = None,
    mm_prefix_range: torch.Tensor | None = None,
    logits_soft_cap: float = 0.0,
    output_lse: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    if query.ndim != 3:
        raise ValueError(f"Expected query shape [T, H, D], got {query.shape}")
    _require_gb10_cuda(query.device)

    layout = get_turboquant_layout(kv_cache_dtype, query.shape[-1])
    kv_group_num = query.shape[1] // key_cache.shape[2]
    if kv_head_for_query_head is None:
        kv_head_for_query_head = (
            torch.arange(query.shape[1], device=query.device, dtype=torch.int64)
            // kv_group_num
        )

    q_rot, q_qjl = apply_turboquant_query_transforms(
        query=query,
        group_indices=key_group_indices,
        rotations=key_rotations,
        qjl_matrices=key_qjl_matrices,
        kv_head_for_query_head=kv_head_for_query_head,
        per_query_group_indices=key_query_group_indices,
    )

    if token_seq_ids is None or token_kv_lens is None or token_query_positions is None:
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        token_seq_ids = torch.repeat_interleave(
            torch.arange(seq_lens.shape[0], device=query.device, dtype=torch.int32),
            query_lens,
        )
        token_offsets = torch.arange(
            query.shape[0], device=query.device, dtype=torch.int32
        ) - query_start_loc.index_select(0, token_seq_ids.to(torch.int64))
        token_kv_lens = seq_lens.index_select(0, token_seq_ids.to(torch.int64)).to(
            torch.int32
        )
        if causal:
            token_query_positions = (
                token_kv_lens
                - query_lens.index_select(0, token_seq_ids.to(torch.int64))
                + token_offsets
            ).to(torch.int32)
        else:
            token_query_positions = torch.zeros_like(token_offsets, dtype=torch.int32)
    if value_query_group_indices is None:
        assert kv_head_for_query_head is not None
        value_query_group_indices = tuple(
            group.index_select(0, kv_head_for_query_head)
            for group in value_group_indices
        )

    group0 = layout.groups[0]
    group1 = layout.groups[1]
    g0_pad = triton.next_power_of_2(group0.dim)
    g1_pad = triton.next_power_of_2(group1.dim)
    block_n = 8 if query.shape[-1] >= 256 else 16

    acc_mse_0 = torch.empty_like(q_rot[0], dtype=torch.float32)
    acc_qjl_0 = torch.empty_like(q_rot[0], dtype=torch.float32)
    acc_mse_1 = torch.empty_like(q_rot[1], dtype=torch.float32)
    acc_qjl_1 = torch.empty_like(q_rot[1], dtype=torch.float32)
    lse = (
        output_lse
        if output_lse is not None
        else torch.empty(
            (query.shape[1], query.shape[0]),
            dtype=torch.float32,
            device=query.device,
        )
    )
    sink_tensor = (
        sinks
        if sinks is not None
        else torch.empty(1, dtype=torch.float32, device=query.device)
    )
    mm_prefix_tensor = (
        mm_prefix_range
        if mm_prefix_range is not None
        else torch.empty(1, dtype=torch.int32, device=query.device)
    )
    max_mm_ranges = mm_prefix_range.shape[1] if mm_prefix_range is not None else 1

    _turboquant_decode_kernel[(query.shape[0], query.shape[1])](
        q_rot_0_ptr=q_rot[0],
        q_qjl_0_ptr=q_qjl[0],
        q_rot_1_ptr=q_rot[1],
        q_qjl_1_ptr=q_qjl[1],
        key_cache_ptr=key_cache,
        value_cache_ptr=value_cache,
        block_table_ptr=block_table,
        token_seq_ids_ptr=token_seq_ids,
        token_kv_lens_ptr=token_kv_lens,
        token_query_positions_ptr=token_query_positions,
        acc_mse_0_ptr=acc_mse_0,
        acc_qjl_0_ptr=acc_qjl_0,
        acc_mse_1_ptr=acc_mse_1,
        acc_qjl_1_ptr=acc_qjl_1,
        lse_ptr=lse,
        norm_lut_ptr=norm_lut,
        centroids_0_ptr=centroids[group0.mse_bits],
        centroids_1_ptr=centroids[group1.mse_bits],
        sink_ptr=sink_tensor,
        mm_prefix_range_ptr=mm_prefix_tensor,
        softmax_scale=softmax_scale,
        softcap=logits_soft_cap,
        q0_stride_0=q_rot[0].stride(0),
        q0_stride_1=q_rot[0].stride(1),
        q1_stride_0=q_rot[1].stride(0),
        q1_stride_1=q_rot[1].stride(1),
        stride_k_cache_0=key_cache.stride(0),
        stride_k_cache_1=key_cache.stride(1),
        stride_k_cache_2=key_cache.stride(2),
        stride_k_cache_3=key_cache.stride(3),
        stride_v_cache_0=value_cache.stride(0),
        stride_v_cache_1=value_cache.stride(1),
        stride_v_cache_2=value_cache.stride(2),
        stride_v_cache_3=value_cache.stride(3),
        block_table_stride=block_table.stride(0),
        acc0_stride_0=acc_mse_0.stride(0),
        acc0_stride_1=acc_mse_0.stride(1),
        acc1_stride_0=acc_mse_1.stride(0),
        acc1_stride_1=acc_mse_1.stride(1),
        lse_stride_0=lse.stride(0),
        lse_stride_1=lse.stride(1),
        kv_group_num=kv_group_num,
        BLOCK_SIZE=key_cache.shape[1],
        BLOCK_N=block_n,
        CAUSAL=causal,
        USE_SOFTCAP=logits_soft_cap > 0,
        USE_SINKS=sinks is not None,
        USE_MM_PREFIX=mm_prefix_range is not None,
        MAX_MM_RANGES=max_mm_ranges,
        SLIDING_WINDOW=(sliding_window[0] + 1 if sliding_window[0] >= 0 else 0),
        RETURN_LSE=output_lse is not None,
        G0_DIM=group0.dim,
        G0_PADDED=g0_pad,
        G0_MSE_BITS=group0.mse_bits,
        G0_GROUP_OFFSET=0,
        G0_QJL_OFFSET=group0.qjl_offset,
        G0_VECTOR_NORM_OFFSET=group0.vector_norm_offset,
        G0_RESIDUAL_NORM_OFFSET=group0.residual_norm_offset,
        G0_QJL_SCALE=TURBOQUANT_QJL_SCALE / group0.dim,
        G1_DIM=group1.dim,
        G1_PADDED=g1_pad,
        G1_MSE_BITS=group1.mse_bits,
        G1_GROUP_OFFSET=group0.packed_bytes,
        G1_QJL_OFFSET=group1.qjl_offset,
        G1_VECTOR_NORM_OFFSET=group1.vector_norm_offset,
        G1_RESIDUAL_NORM_OFFSET=group1.residual_norm_offset,
        G1_QJL_SCALE=TURBOQUANT_QJL_SCALE / group1.dim,
        num_warps=4,
        num_stages=2,
    )

    if value_mse_inverse_matrices is None:
        value_mse_inverse_matrices = (
            get_turboquant_mse_inverse_transform_matrix(
                query.device, group0.dim, seed_offset=101
            ),
            get_turboquant_mse_inverse_transform_matrix(
                query.device, group1.dim, seed_offset=211
            ),
        )
    if value_qjl_inverse_matrices is None:
        value_qjl_inverse_matrices = (
            get_turboquant_qjl_inverse_transform_matrix(
                query.device, group0.dim, seed_offset=307
            ),
            get_turboquant_qjl_inverse_transform_matrix(
                query.device, group1.dim, seed_offset=401
            ),
        )

    output = torch.empty_like(query) if out is None else out
    block_out = 32
    block_k = 32
    for acc_mse, acc_qjl, mse_inverse, qjl_inverse, indices, group, dim_pad in zip(
        (acc_mse_0, acc_mse_1),
        (acc_qjl_0, acc_qjl_1),
        value_mse_inverse_matrices,
        value_qjl_inverse_matrices,
        value_query_group_indices,
        layout.groups,
        (g0_pad, g1_pad),
        strict=True,
    ):
        grid = (query.shape[0], query.shape[1], triton.cdiv(group.dim, block_out))
        _turboquant_decode_postprocess_group_kernel[grid](
            acc_mse_ptr=acc_mse,
            acc_qjl_ptr=acc_qjl,
            mse_inverse_ptr=mse_inverse,
            qjl_inverse_ptr=qjl_inverse,
            group_indices_ptr=indices,
            out_ptr=output,
            acc_stride_0=acc_mse.stride(0),
            acc_stride_1=acc_mse.stride(1),
            mse_inverse_stride_0=mse_inverse.stride(0),
            mse_inverse_stride_1=mse_inverse.stride(1),
            qjl_inverse_stride_0=qjl_inverse.stride(0),
            qjl_inverse_stride_1=qjl_inverse.stride(1),
            group_indices_stride_0=indices.stride(0),
            out_stride_0=output.stride(0),
            out_stride_1=output.stride(1),
            out_stride_2=output.stride(2),
            DIM=group.dim,
            DIM_PADDED=dim_pad,
            BLOCK_OUT=block_out,
            BLOCK_K=block_k,
            num_warps=4,
            num_stages=2,
        )

    return output
