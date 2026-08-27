# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Token coverage generation for GEMM tuning."""

from __future__ import annotations


# Default token batch sizes for MoE tuning (covers prefill + decode)
_DEFAULT_TOKENS = [4, 8, 16, 32, 48, 64, 96, 128, 256, 512]
_HIGH_CONC_TOKENS = [768, 1024]
_VERY_HIGH_CONC_TOKENS = [1536, 2048, 4096, 8192]

# sglang CUDAGraph capture batch sizes (server_args default list)
_SGLANG_CUDAGRAPH_BS = [
    1,
    2,
    4,
    8,
    12,
    16,
    24,
    32,
    40,
    48,
    56,
    64,
    72,
    80,
    88,
    96,
    104,
    112,
    120,
    128,
    136,
    144,
    152,
    160,
    168,
    176,
    184,
    192,
    200,
    208,
    216,
    224,
    232,
    240,
    248,
    256,
    272,
    288,
    304,
    320,
    336,
    352,
    368,
    384,
    400,
    416,
    432,
    448,
    464,
    480,
    496,
    512,
]


def compute_token_coverage(
    conc: int = 0,
    explicit_tokens: list[int] | None = None,
) -> list[int]:
    """Compute which token (batch) sizes to tune.

    If explicit_tokens is provided, use those directly. Otherwise generate
    a coverage set based on the target concurrency.

    Args:
        conc: Target serving concurrency. Higher values add larger batch sizes.
        explicit_tokens: If provided, overrides automatic generation.

    Returns:
        Sorted list of deduplicated token sizes.
    """
    if explicit_tokens:
        return sorted(set(explicit_tokens))

    tokens = list(_DEFAULT_TOKENS)
    if conc >= 128:
        tokens.extend(_HIGH_CONC_TOKENS)
    if conc >= 512:
        tokens.extend(_VERY_HIGH_CONC_TOKENS)
    return sorted(set(tokens))


def compute_dense_gemm_shapes(
    hidden_size: int,
    intermediate_size: int,
    tokens: list[int],
    tp: int = 1,
) -> list[tuple[int, int, int]]:
    """Compute (M, N, K) dense GEMM shapes from model config.

    For a standard transformer MLP (gate_proj/up_proj: hidden->inter, down_proj: inter->hidden):
      - gate/up: (M=batch, N=intermediate/tp, K=hidden)
      - down:    (M=batch, N=hidden, K=intermediate/tp)

    Args:
        hidden_size: Model hidden dimension.
        intermediate_size: MLP intermediate dimension.
        tokens: Batch sizes (M dimension).
        tp: Tensor parallel degree.

    Returns:
        Deduplicated (M, N, K) tuples.
    """
    n_inter = intermediate_size // tp
    n_hidden = hidden_size  # hidden is not TP-split for output proj

    shapes = set()
    for m in tokens:
        # gate_proj / up_proj
        shapes.add((m, n_inter, hidden_size))
        # down_proj
        shapes.add((m, n_hidden, n_inter))
    return sorted(shapes)


def compute_vllm_moe_batch_sizes(
    conc: int = 0,
    explicit_tokens: list[int] | None = None,
) -> list[int]:
    """Batch sizes for vLLM MoE Triton sweep.

    vLLM's fused_moe uses M (tokens routed per expert) not total batch.
    Typical: 1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192.
    """
    if explicit_tokens:
        return sorted(set(explicit_tokens))

    sizes = [1, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    if conc < 64:
        sizes = [s for s in sizes if s <= 2048]
    return sizes


def compute_sglang_cudagraph_m_values(
    conc: int = 64,
    max_bs: int = 512,
) -> list[int]:
    """Compute M values matching sglang CUDAGraph capture batch sizes.

    sglang captures CUDAGraphs at specific batch sizes. GEMM tuning must
    cover these exact M values for the tuned config to be used at runtime.

    Args:
        conc: Target serving concurrency. Determines how far up the BS list to tune.
        max_bs: Maximum batch size to include. Higher = more shapes but longer tuning.

    Returns:
        Sorted list of batch sizes that will be captured as CUDAGraphs.
    """
    effective_max = min(max_bs, max(conc * 4, 128))
    return [bs for bs in _SGLANG_CUDAGRAPH_BS if bs <= effective_max]
