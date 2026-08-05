# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Framework-facing entry points for installed KernelForge kernel packs.

Every entry point is *fallback-first*: it returns ``None`` for anything it is
not certain about, and the patched framework call site runs its original code in
that case. The guards, in the order they fire:

1. ``$HYPERLOOM_FORGE_KERNEL_PACKS`` unset  -> off (default; strict no-op).
2. No installed / preflight-passing pack for the op.
3. Tensor shape, dtype, layout or device outside the pack's contract.
4. ``(N, dtype)`` not in the machine-local preflight allowlist.
5. Inside a CUDA/HIP graph capture with the shape not already built. Building
   runs the FlyDSL JIT (host compilation + allocations), which must never
   happen mid-capture; the pre-capture warmup passes populate the cache, and a
   shape first seen during capture simply falls back.
6. First launch of a new ``(M, N, dtype)`` disagrees with the framework
   reference -> blacklist the shape and fall back for the rest of the process.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from ._packs import Pack
from ._packs import dtype_tag
from ._packs import enabled_pack_names
from ._packs import packs_for_op
from ._packs import verify_enabled

log = logging.getLogger(__name__)

OP_ROWWISE_SOFTMAX = "rowwise_softmax"

#: Minimum SNR (dB) the first launch of a new build key must clear. Well under
#: the ~140 dB an f32 row-wise softmax actually delivers, but high enough to
#: catch a miscompile or a wrong-shape launch.
_FIRST_USE_MIN_SNR_DB = 30.0

_STATS: dict[str, int] = {}
_STATS_LOCK = threading.Lock()
_VERIFIED_KEYS: set[tuple[str, int, int, str]] = set()


def _bump(counter: str) -> None:
    with _STATS_LOCK:
        _STATS[counter] = _STATS.get(counter, 0) + 1


def stats() -> dict[str, int]:
    """Snapshot of the dispatch counters (hits, and every fallback reason).

    Exposed so a Hyperloom benchmark run can assert the kernel actually fired
    rather than silently falling back for the whole run.
    """
    with _STATS_LOCK:
        return dict(_STATS)


def is_enabled() -> bool:
    """Whether any pack is switched on for this process."""
    return bool(enabled_pack_names())


def enabled_packs() -> tuple[str, ...]:
    """Pack names requested via ``$HYPERLOOM_FORGE_KERNEL_PACKS``."""
    return enabled_pack_names()


def _is_capturing() -> bool:
    """True while the current stream is capturing a CUDA/HIP graph."""
    try:
        import torch

        probe = getattr(torch.cuda, "is_current_stream_capturing", None)
        return bool(probe()) if probe is not None else False
    except Exception:  # noqa: BLE001 - probe must never break the caller
        return False


def rowwise_softmax(x: Any) -> Any | None:
    """``torch.softmax(x, dim=-1)`` via a KernelForge pack, or ``None``.

    Args:
        x: A 2-D, contiguous, CUDA/HIP tensor. Rows are reduced independently.

    Returns:
        A new tensor with the same shape and dtype as ``x``, or ``None`` when
        any guard rejects the call. ``None`` means "run the original code";
        it is never an error.
    """
    packs = packs_for_op(OP_ROWWISE_SOFTMAX)
    if not packs:
        _bump("skip_no_pack")
        return None

    import torch

    if not isinstance(x, torch.Tensor) or x.dim() != 2 or not x.is_cuda:
        _bump("skip_bad_tensor")
        return None
    if not x.is_contiguous():
        _bump("skip_noncontiguous")
        return None
    tag = dtype_tag(x.dtype)
    if tag is None:
        _bump("skip_dtype")
        return None

    m, n = int(x.shape[0]), int(x.shape[1])
    if m <= 0:
        _bump("skip_empty")
        return None

    for pack in packs:
        if not pack.supports(n, tag):
            continue
        out = _run_rowwise_softmax(pack, x, m, n, tag)
        if out is not None:
            return out
    _bump("skip_unverified_shape")
    return None


def _run_rowwise_softmax(pack: Pack, x: Any, m: int, n: int, tag: str) -> Any | None:
    import torch

    capturing = _is_capturing()
    key = (pack.name, m, n, tag)

    launcher = pack.build(m, n, tag) if not capturing else pack.build_if_cached(m, n, tag)
    if launcher is None:
        _bump("skip_capture_cold" if capturing else "skip_build_failed")
        return None

    try:
        import flydsl.expr as fx

        out = torch.empty_like(x)
        # The stream handle has to be taken per call: during graph capture torch
        # runs on a side stream, and a handle cached from the default stream
        # would enqueue outside the capture (producing an empty graph).
        launcher(x, out, m, stream=fx.Stream(torch.cuda.current_stream().cuda_stream))
    except Exception as e:  # noqa: BLE001 - any launch failure => fall back
        log.warning(
            "forge_kernels: pack %r launch failed for %s (%s: %s); blacklisting",
            pack.name,
            (m, n, tag),
            type(e).__name__,
            e,
        )
        pack.blacklist(m, n, tag)
        _bump("skip_launch_failed")
        return None

    if verify_enabled() and not capturing and key not in _VERIFIED_KEYS:
        if not _first_use_matches_reference(pack, x, out, m, n, tag, key):
            return None

    _bump("hit")
    return out


def _first_use_matches_reference(
    pack: Pack,
    x: Any,
    out: Any,
    m: int,
    n: int,
    tag: str,
    key: tuple[str, int, int, str],
) -> bool:
    """Score the first launch of a build key against ``torch.softmax``.

    Costs one reference softmax per new shape, once. It is the last line of
    defence between a KernelForge artifact and a served token, and it is what
    turns "the kernel compiled" into "the kernel is right on this machine".
    """
    import torch

    try:
        reference = torch.softmax(x.float(), dim=-1)
        noise = (reference - out.float()).pow(2).mean().item()
        signal = reference.pow(2).mean().item()
        snr_db = float("inf") if noise <= 0.0 else 10.0 * torch.log10(torch.tensor(signal / noise)).item()
    except Exception as e:  # noqa: BLE001 - a broken check must not break serving
        log.warning(
            "forge_kernels: pack %r first-use check errored for %s (%s); blacklisting",
            pack.name,
            (m, n, tag),
            type(e).__name__,
        )
        pack.blacklist(m, n, tag)
        _bump("skip_verify_error")
        return False

    if snr_db < _FIRST_USE_MIN_SNR_DB:
        log.error(
            "forge_kernels: pack %r produced %.1f dB SNR for %s (need >= %.1f); "
            "blacklisting the shape and falling back to the framework op",
            pack.name,
            snr_db,
            (m, n, tag),
            _FIRST_USE_MIN_SNR_DB,
        )
        pack.blacklist(m, n, tag)
        _bump("skip_verify_failed")
        return False

    with _STATS_LOCK:
        _VERIFIED_KEYS.add(key)
    log.info(
        "forge_kernels: pack %r verified %s at first use (%.1f dB SNR)",
        pack.name,
        (m, n, tag),
        snr_db,
    )
    _bump("verified_new_shape")
    return True
