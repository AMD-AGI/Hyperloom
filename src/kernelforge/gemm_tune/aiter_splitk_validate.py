# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Per-shape production split-K support, by trial-dispatch.

The trial is per-op: a limit measured on one op's kernel says nothing about
another's, so :data:`_TRIAL_OPS` registers the production callable per op and an
unregistered op answers ``None`` (caller keeps its static cap). Answering with
the wrong kernel's limit would license a splitK the real kernel rejects (crash)
or tighten one it accepts (silent loss). ``a8w8_blockscale_bpreshuffle`` trials
its asm kernel only: its ck/cktile wrappers take no ``splitK`` at all, and
``_SPLITK_FORWARDING_LIBTYPES`` in ``tuners._aiter_dense_common`` caps those
rows to 0.
"""

from __future__ import annotations

import os

_BLOCK_N = _BLOCK_K = 128


def _resolve_device(gpu_ids: str = "") -> str:
    """Torch device for the trial, honoring the tuner's assigned ``gpu_ids``."""
    first = next((g.strip() for g in gpu_ids.split(",") if g.strip()), "")
    if not first:
        return "cuda"
    visible = (
        os.environ.get("HIP_VISIBLE_DEVICES")
        or os.environ.get("CUDA_VISIBLE_DEVICES")
        or os.environ.get("ROCR_VISIBLE_DEVICES")
    )
    if visible:
        ids = [v.strip() for v in visible.split(",") if v.strip()]
        if first in ids:
            return f"cuda:{ids.index(first)}"
        # Assigned card is not among the visible set: cannot target it here, so fall back to the default device rather
        # than raising on an invalid index.
        return "cuda"
    return f"cuda:{first}"


def _operands(m: int, n: int, k: int, device: str):
    """Random fp8 blockscale operands (x, w, x_scale, w_scale, bf16 out) for one trial dispatch."""
    import torch
    from aiter import dtypes

    sn = (n + _BLOCK_N - 1) // _BLOCK_N
    sk_dim = (k + _BLOCK_K - 1) // _BLOCK_K
    x = (torch.rand((m, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    w = (torch.rand((n, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    xs = torch.rand([m, sk_dim], dtype=dtypes.fp32, device=device)
    ws = torch.rand([sn, sk_dim], dtype=dtypes.fp32, device=device)
    out = torch.empty(m, n, dtype=dtypes.bf16, device=device)
    return x, w, xs, ws, out


def _supports_a8w8_blockscale(m: int, n: int, k: int, split_k: int, device: str = "cuda") -> bool:
    """True if the production a8w8_blockscale CK kernel dispatches (m,n,k,split_k)."""
    import torch
    import aiter

    x, w, xs, ws, out = _operands(m, n, k, device)
    aiter.gemm_a8w8_blockscale_ck(x, w, xs, ws, out, splitK=split_k)
    torch.cuda.synchronize()
    return True


def _supports_a8w8_blockscale_bpreshuffle(m: int, n: int, k: int, split_k: int, device: str = "cuda") -> bool:
    """True if the production a8w8_blockscale_bpreshuffle asm kernel dispatches (m,n,k,split_k).

    A tuned asm row passes its splitK explicitly, and the launcher then rejects
    any splitK whose 128-aligned K partition yields a different split count.
    Operand contents do not matter: the check runs on shapes before launch.
    """
    import torch
    import aiter

    x, w, xs, ws, out = _operands(m, n, k, device)
    aiter.gemm_a8w8_blockscale_bpreshuffle_asm(x, w, out, xs, ws, splitK=split_k)
    torch.cuda.synchronize()
    return True


#: Ops whose production split-K limit this module can measure, keyed by the
#: tuner ``script_key``. Absent => no trial, caller keeps its static cap.
_TRIAL_OPS = {
    "a8w8_blockscale": _supports_a8w8_blockscale,
    "a8w8_blockscale_bpreshuffle": _supports_a8w8_blockscale_bpreshuffle,
}

#: Highest splitK the trial scans per op. aiter's tuner generates asm candidates
#: up to splitK=8 (``get_valid_asm_splitK_list(K, 8)``), so a lower scan would
#: rewrite valid picks.
_SCAN_CEILING = {"a8w8_blockscale_bpreshuffle": 8}
_DEFAULT_SCAN_CEILING = 6


def max_supported_splitk(
    mnk: tuple[int, int, int],
    ceiling: int = 6,
    device: str = "cuda",
    op: str = "a8w8_blockscale",
) -> int | None:
    """Max splitK in ``0..ceiling`` the production kernel accepts for (m,n,k).

    ``None`` when ``op`` has no registered trial, or when the splitK=0 control
    fails, so the caller keeps its static fallback rather than trusting an
    unvalidated trial.
    """
    probe = _TRIAL_OPS.get(op)
    if probe is None:
        return None
    m, n, k = mnk
    try:
        if not probe(m, n, k, 0, device=device):
            return None
    except (ImportError, OSError, RuntimeError):
        return None
    best = 0
    for sk in range(1, max(0, ceiling) + 1):
        try:
            ok = probe(m, n, k, sk, device=device)
        except RuntimeError:
            ok = False
        if not ok:
            break
        best = sk
    return best


def make_support_fn(ceiling: int | None = None, gpu_ids: str = "", op: str = "a8w8_blockscale"):
    """Return an (m,n,k)->int|None callable memoized per shape for reuse as the ``support_fn`` of ``_cap_splitk_to_serve_safe``.

    ``op`` selects the production kernel to trial; an unregistered op answers
    ``None`` for every shape so the caller falls back to its static cap instead
    of being handed another op's limit. ``ceiling`` defaults to the op's scan
    ceiling.
    """
    if op not in _TRIAL_OPS:
        return lambda m, n, k: None
    if ceiling is None:
        ceiling = _SCAN_CEILING.get(op, _DEFAULT_SCAN_CEILING)

    cache: dict[tuple[int, int, int], int | None] = {}
    device = _resolve_device(gpu_ids)

    def _fn(m: int, n: int, k: int):
        key = (m, n, k)
        if key not in cache:
            cache[key] = max_supported_splitk(key, ceiling=ceiling, device=device, op=op)
        return cache[key]

    return _fn
