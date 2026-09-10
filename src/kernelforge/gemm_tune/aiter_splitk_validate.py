# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Per-shape production split-K support, by trial-dispatch.

The trial is per-op: a limit measured on one op's kernel says nothing about
another's, so :data:`_TRIAL_OPS` registers the production callable per op and an
unregistered op answers ``None`` (caller keeps its static cap). Answering with
the wrong kernel's limit would license a splitK the real kernel rejects (crash)
or tighten one it accepts (silent loss). ``a8w8_blockscale_bpreshuffle`` is
absent on purpose: its ck/cktile wrappers take no ``splitK`` at all, so
``_SPLITK_FORWARDING_LIBTYPES`` in ``tuners._aiter_dense_common`` is the gate
that matters there.
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


def _supports_a8w8_blockscale(m: int, n: int, k: int, split_k: int, device: str = "cuda") -> bool:
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


#: Ops whose production split-K limit this module can measure, keyed by the
#: tuner ``script_key``. Absent => no trial, caller keeps its static cap.
_TRIAL_OPS = {
    "a8w8_blockscale": _supports_a8w8_blockscale,
}


def supported_ops() -> frozenset[str]:
    """Op keys :func:`make_support_fn` can build a real trial for."""
    return frozenset(_TRIAL_OPS)


def max_supported_splitk(
    m: int,
    n: int,
    k: int,
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
    try:
        if not probe(m, n, k, 0, device=device):
            return None
    except Exception:  # noqa: BLE001 — no GPU / import error / tensor issue
        return None
    best = 0
    for sk in range(1, max(0, ceiling) + 1):
        try:
            ok = probe(m, n, k, sk, device=device)
        except Exception:  # noqa: BLE001 — treat a hard error as "unsupported"
            ok = False
        if not ok:
            break
        best = sk
    return best


def make_support_fn(ceiling: int = 6, gpu_ids: str = "", op: str = "a8w8_blockscale"):
    """Return an (m,n,k)->int|None callable memoized per shape for reuse as the ``support_fn`` of ``_cap_splitk_to_serve_safe``.

    ``op`` selects the production kernel to trial; an unregistered op answers
    ``None`` for every shape so the caller falls back to its static cap instead
    of being handed another op's limit.
    """
    if op not in _TRIAL_OPS:
        return lambda m, n, k: None

    cache: dict[tuple[int, int, int], int | None] = {}
    device = _resolve_device(gpu_ids)

    def _fn(m: int, n: int, k: int):
        key = (m, n, k)
        if key not in cache:
            cache[key] = max_supported_splitk(m, n, k, ceiling=ceiling, device=device, op=op)
        return cache[key]

    return _fn
