"""Real Multi-TurboQuant KV cache integration for vLLM v0.19.0.

The KV cache is stored as packed uint8 bytes. Each cache slot has
``head_size`` bytes (since dtype=uint8 with the model's original
``head_size`` element count). The TQ encoder packs each (head, token)
into ``packed_dim`` bytes; the remaining ``head_size - packed_dim``
bytes per slot are unused but kept so the cache shape stays compatible
with the standard attention kernel layout.

Memory savings:
    For Qwen3-14B (head_size=128, fp16 → 256 bytes/slot vs uint8 → 128
    bytes/slot), this gives an immediate 50% KV cache reduction. The
    actual packed_dim (54-70 bytes for the 6 supported methods) is
    smaller still; recovering the rest would need a custom kernel that
    reads packed_dim-wide slots directly.

Decode strategy:
    During attention forward, we identify the unique blocks referenced
    by the current batch's block_table, decode only those blocks back
    to fp16 K/V, build a *compact* temporary fp16 cache, remap the
    block_table to the compact indices, and run the standard
    ``unified_attention`` Triton kernel on the temp cache. This keeps
    the persistent cache small (the savings stay) while reusing all of
    vLLM's existing fast attention machinery.
"""

from __future__ import annotations

import torch

# Map vLLM kv_cache_dtype string → multi_turboquant CacheMethod value
_TQ_DTYPE_MAP: dict[str, str] = {
    "isoquant3": "iso3",
    "isoquant4": "iso4",
    "planarquant3": "planar3",
    "planarquant4": "planar4",
    "turboquant25": "turbo2",
    "turboquant35": "turbo3",
}

_TQ_DTYPES: frozenset[str] = frozenset(_TQ_DTYPE_MAP)


def is_tq_dtype(kv_cache_dtype: str) -> bool:
    return kv_cache_dtype in _TQ_DTYPES


# ──────────────────────────────────────────────────────────────────────
# Method instance + group-indices cache (avoid re-instantiating)
# ──────────────────────────────────────────────────────────────────────

# Per-dtype: (method instance, group_indices_for_head_size)
_METHOD_CACHE: dict[str, object] = {}
_GROUP_INDICES_CACHE: dict[tuple[str, int, str], tuple] = {}


def _get_method(kv_cache_dtype: str):
    if kv_cache_dtype in _METHOD_CACHE:
        return _METHOD_CACHE[kv_cache_dtype]
    from multi_turboquant import CacheMethod, get_method
    name = _TQ_DTYPE_MAP[kv_cache_dtype]
    method = get_method(CacheMethod(name))
    _METHOD_CACHE[kv_cache_dtype] = method
    return method


def _get_fixed_group_indices(
    kv_cache_dtype: str,
    head_size: int,
    num_kv_heads: int,
    device: torch.device,
):
    """Compute & cache deterministic outlier/regular indices for TurboQuant.

    TurboQuant variants need a per-head split of dimensions into
    "outlier" (higher precision) and "regular" (lower precision)
    groups. We use the simple fixed split (first ``oc`` dims as
    outliers, rest as low) so encode and decode see the same indices,
    since we don't preserve metadata across the cache write/read
    boundary. Calibrated metadata could be plugged in here later.
    """
    if not kv_cache_dtype.startswith("turboquant"):
        return None

    cache_key = (kv_cache_dtype, head_size, num_kv_heads, str(device))
    if cache_key in _GROUP_INDICES_CACHE:
        return _GROUP_INDICES_CACHE[cache_key]

    from multi_turboquant.methods.turboquant import get_outlier_count
    method_name = _TQ_DTYPE_MAP[kv_cache_dtype]
    oc = get_outlier_count(head_size, method_name)
    high = (
        torch.arange(oc, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(num_kv_heads, -1)
        .contiguous()
    )
    low = (
        torch.arange(oc, head_size, device=device, dtype=torch.long)
        .unsqueeze(0)
        .expand(num_kv_heads, -1)
        .contiguous()
    )
    gi = (high, low)
    _GROUP_INDICES_CACHE[cache_key] = gi
    return gi


def get_packed_dim(kv_cache_dtype: str, head_size: int) -> int:
    """Number of packed bytes per (head, token) for this method."""
    method = _get_method(kv_cache_dtype)
    return method.packed_dim(head_size)


# ──────────────────────────────────────────────────────────────────────
# Cache write: encode K/V → scatter packed bytes into uint8 cache
# ──────────────────────────────────────────────────────────────────────

def tq_write_kv_cache(
    key: torch.Tensor,        # [num_tokens, num_kv_heads, head_size] fp16/bf16
    value: torch.Tensor,      # [num_tokens, num_kv_heads, head_size] fp16/bf16
    key_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size] uint8
    value_cache: torch.Tensor,  # [num_blocks, block_size, num_kv_heads, head_size] uint8
    slot_mapping: torch.Tensor,  # [num_tokens] flat slot indices
    kv_cache_dtype: str,
) -> None:
    """Encode K/V with the configured TQ method and write packed bytes.

    The first ``packed_dim`` bytes of each slot store the encoded data;
    the remaining bytes are unused (zero-initialised at allocation).
    """
    method = _get_method(kv_cache_dtype)
    head_size = key.shape[-1]
    num_kv_heads = key.shape[-2]
    packed_dim = method.packed_dim(head_size)
    block_size = key_cache.shape[1]

    # For TurboQuant we need fixed group_indices so encode and decode
    # see the same dimension split (we don't preserve per-tensor metadata
    # across the cache write/read boundary).
    if kv_cache_dtype.startswith("turboquant"):
        gi = _get_fixed_group_indices(
            kv_cache_dtype, head_size, num_kv_heads, key.device,
        )
        method._group_indices = gi  # _TurboQuantBase honours this

    # Encode all tokens at once (vectorised over tokens × heads).
    key_compressed = method.encode(key)   # CompressedKV
    value_compressed = method.encode(value)
    # Each .data is shape [num_tokens, num_kv_heads, packed_dim] uint8
    key_packed: torch.Tensor = key_compressed.data
    value_packed: torch.Tensor = value_compressed.data

    # Filter out padding tokens (slot < 0).
    valid_mask = slot_mapping >= 0
    if not bool(valid_mask.all()):
        valid_slots = slot_mapping[valid_mask]
        key_packed = key_packed[valid_mask]
        value_packed = value_packed[valid_mask]
    else:
        valid_slots = slot_mapping

    block_indices = (valid_slots // block_size).long()
    block_offsets = (valid_slots % block_size).long()

    # Vectorised scatter: write the packed bytes into the first
    # ``packed_dim`` columns of each (block, offset) slot.
    key_cache[block_indices, block_offsets, :, :packed_dim] = key_packed
    value_cache[block_indices, block_offsets, :, :packed_dim] = value_packed


# ──────────────────────────────────────────────────────────────────────
# Cache read: decode active blocks into a compact fp16 cache
# ──────────────────────────────────────────────────────────────────────

def tq_decode_active_blocks(
    key_cache: torch.Tensor,    # [num_blocks, block_size, num_kv_heads, head_size] uint8
    value_cache: torch.Tensor,  # same shape uint8
    block_table: torch.Tensor,  # [batch, max_blocks_per_seq] int32, -1 = unused
    seq_lens: torch.Tensor,     # [batch] int32
    kv_cache_dtype: str,
    target_dtype: torch.dtype,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decode the blocks referenced by the current batch.

    Returns:
        compact_key_cache: [num_active_blocks, block_size, num_kv_heads,
            head_size] in target_dtype.
        compact_value_cache: same shape.
        new_block_table: [batch, max_blocks_per_seq] int32, indices into
            the compact caches (entries that were originally -1 stay -1).
    """
    method = _get_method(kv_cache_dtype)
    head_size = key_cache.shape[-1]
    num_kv_heads = key_cache.shape[2]
    packed_dim = method.packed_dim(head_size)
    num_blocks = key_cache.shape[0]
    device = key_cache.device

    # Apply the same fixed group_indices as the write path so the
    # TurboQuant decode can find them on the method instance.
    if kv_cache_dtype.startswith("turboquant"):
        gi = _get_fixed_group_indices(
            kv_cache_dtype, head_size, num_kv_heads, device,
        )
        method._group_indices = gi
    else:
        gi = None

    # Determine how many blocks each request actually uses.
    # block_table[b, i] is valid for i < ceil(seq_lens[b] / block_size).
    blocks_per_seq = (seq_lens + block_size - 1) // block_size
    max_used = int(blocks_per_seq.max().item()) if blocks_per_seq.numel() else 0

    if max_used == 0:
        empty = torch.empty(
            (0, block_size, key_cache.shape[2], head_size),
            dtype=target_dtype, device=device,
        )
        return empty, empty, block_table

    # Used block ids (one per (batch, position-within-seq) pair).
    # We mask out positions beyond each seq's actual block count.
    used_table = block_table[:, :max_used].long()
    seq_block_idx = torch.arange(max_used, device=device).unsqueeze(0)
    used_mask = seq_block_idx < blocks_per_seq.unsqueeze(1)
    flat_used = used_table[used_mask]

    if flat_used.numel() == 0:
        empty = torch.empty(
            (0, block_size, key_cache.shape[2], head_size),
            dtype=target_dtype, device=device,
        )
        return empty, empty, block_table

    # Unique active blocks. We need a deterministic ordering so the
    # remap below is consistent.
    unique_active = torch.unique(flat_used, sorted=True)

    # Gather just those blocks.
    # active_packed shape: [num_active, block_size, num_kv_heads, head_size]
    active_key_packed = key_cache[unique_active][..., :packed_dim].contiguous()
    active_value_packed = value_cache[unique_active][..., :packed_dim].contiguous()
    num_active_blocks = active_key_packed.shape[0]

    # Multi-TurboQuant decode functions operate on 3D tensors
    # [batch, num_kv_heads, packed_dim]. Flatten the (block, slot)
    # dims for decode, then reshape back.
    flat_key_packed = active_key_packed.reshape(
        num_active_blocks * block_size, num_kv_heads, packed_dim,
    )
    flat_value_packed = active_value_packed.reshape(
        num_active_blocks * block_size, num_kv_heads, packed_dim,
    )

    from multi_turboquant.methods.base import CompressedKV
    from multi_turboquant.config import CacheMethod

    cm = CacheMethod(_TQ_DTYPE_MAP[kv_cache_dtype])
    decode_metadata = {"group_indices": gi} if gi is not None else None
    flat_key_decoded = method.decode(
        CompressedKV(
            data=flat_key_packed, head_dim=head_size, method=cm,
            metadata=decode_metadata,
        ),
        dtype=target_dtype,
    )
    flat_value_decoded = method.decode(
        CompressedKV(
            data=flat_value_packed, head_dim=head_size, method=cm,
            metadata=decode_metadata,
        ),
        dtype=target_dtype,
    )

    key_decoded = flat_key_decoded.reshape(
        num_active_blocks, block_size, num_kv_heads, head_size,
    )
    value_decoded = flat_value_decoded.reshape(
        num_active_blocks, block_size, num_kv_heads, head_size,
    )

    # Build the compact remapped block table. Entries originally <0
    # stay <0; valid entries are mapped to their compact index.
    remap = torch.full((num_blocks,), -1, dtype=torch.int64, device=device)
    remap[unique_active] = torch.arange(
        unique_active.numel(), dtype=torch.int64, device=device,
    )
    new_block_table = block_table.clone().to(torch.int64)
    valid_mask = block_table >= 0
    new_block_table[valid_mask] = remap[block_table[valid_mask].long()]
    new_block_table = new_block_table.to(block_table.dtype)

    return key_decoded, value_decoded, new_block_table
