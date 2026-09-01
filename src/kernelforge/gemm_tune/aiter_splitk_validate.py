# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Per-shape production split-K support, by trial-dispatch.

The aiter *tuner* benchmarks split-K values the *production* kernel
(``gemm_a8w8_blockscale_ck``) cannot dispatch; serving such a config raises
"This GEMM is not supported!" and crashes engine init. The real supported max is
NOT a constant -- it varies per (M,N,K) (empirically 2 or 3 on gfx950). This
module finds it by trial: build valid fp8 tensors for the shape and call the
production kernel with increasing splitK, catching the failure. The splitK=0
control (KBatch=1) must pass; if it does not, tensors/GPU are the problem, not
split-K, so we return ``None`` (caller falls back to a static cap).

GPU-dependent (imports torch + aiter); all imports are lazy and guarded so this
module is importable (and its pure logic testable) without a GPU.
"""

from __future__ import annotations

import os

_BLOCK_N = _BLOCK_K = 128


def _resolve_device(gpu_ids: str = "") -> str:
    """Torch device for the trial, honoring the tuner's assigned ``gpu_ids``.

    The trial runs IN-PROCESS (not a subprocess that inherits CUDA_VISIBLE_DEVICES),
    so a bare ``device="cuda"`` always lands on the first visible card -- wrong on a
    multi-tenant node whose assigned GPU is not index 0, which then makes a spurious
    dispatch failure wrongly delete valid split-K rows. Pick the first assigned id;
    if the parent already restricts visible devices, map the physical id to its
    local torch index, otherwise use it directly.
    """
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
        # Assigned card is not among the visible set: cannot target it here, so
        # fall back to the default device rather than raising on an invalid index.
        return "cuda"
    return f"cuda:{first}"


def _supports(m: int, n: int, k: int, split_k: int, device: str = "cuda") -> bool:
    """True if the production a8w8_blockscale CK kernel dispatches (m,n,k,split_k)."""
    import torch  # noqa: PLC0415
    import aiter  # noqa: PLC0415
    from aiter import dtypes  # noqa: PLC0415

    sn = (n + _BLOCK_N - 1) // _BLOCK_N
    sk_dim = (k + _BLOCK_K - 1) // _BLOCK_K
    x = (torch.rand((m, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    w = (torch.rand((n, k), dtype=dtypes.fp16, device=device) / 10).to(dtypes.fp8)
    xs = torch.rand([m, sk_dim], dtype=dtypes.fp32, device=device)
    ws = torch.rand([sn, sk_dim], dtype=dtypes.fp32, device=device)
    out = torch.empty(m, n, dtype=dtypes.bf16, device=device)
    aiter.gemm_a8w8_blockscale_ck(x, w, xs, ws, out, splitK=split_k)
    torch.cuda.synchronize()
    return True


def max_supported_splitk(m: int, n: int, k: int, ceiling: int = 6, device: str = "cuda") -> int | None:
    """Max splitK in ``0..ceiling`` the production kernel accepts for (m,n,k).

    Returns ``None`` when the splitK=0 control fails (no GPU / aiter not
    importable / tensor mismatch) so the caller keeps its static fallback rather
    than trusting an unvalidated trial. Support is contiguous from 0 (KBatch grows
    as 2**splitK), so the scan stops at the first unsupported value.
    """
    try:
        if not _supports(m, n, k, 0, device=device):
            return None
    except Exception:  # noqa: BLE001 — no GPU / import error / tensor issue
        return None
    best = 0
    for sk in range(1, max(0, ceiling) + 1):
        try:
            ok = _supports(m, n, k, sk, device=device)
        except Exception:  # noqa: BLE001 — treat a hard error as "unsupported"
            ok = False
        if not ok:
            break
        best = sk
    return best


def make_support_fn(ceiling: int = 6, gpu_ids: str = ""):
    """Return an (m,n,k)->int|None callable memoized per shape for reuse as the
    ``support_fn`` of ``_cap_splitk_to_serve_safe``. A ``None`` control result is
    cached too so a GPU-less environment trials at most once per shape. ``gpu_ids``
    (the tuner's assigned cards) pins the trial to the correct GPU on a shared
    node instead of always using device 0."""
    cache: dict[tuple[int, int, int], int | None] = {}
    device = _resolve_device(gpu_ids)

    def _fn(m: int, n: int, k: int):
        key = (m, n, k)
        if key not in cache:
            cache[key] = max_supported_splitk(m, n, k, ceiling=ceiling, device=device)
        return cache[key]

    return _fn
