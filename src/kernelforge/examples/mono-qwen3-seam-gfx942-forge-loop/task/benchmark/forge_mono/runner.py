# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Host side of the seam launch: one compiled kernel and scratch region per row count of one TP
rank, the epoch they share, and the one launch a call makes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import flydsl.compiler as flyc
import torch
from sgl_amd_mono.common import guard
from sgl_amd_mono.common.epoch import EPOCH_MARKS
from sgl_amd_mono.minimax_m31.residency import active_workgroups_per_cu

from qwen3_seam import seam

# a wait's deadline: far past a healthy launch, short enough that a stalled rank ends the run soon
DEFAULT_DEADLINE_S = 6.0


def _writable_flydsl_cache() -> None:
    """AITER points FlyDSL at its bundled read-only cache; compiled launches need a writable one."""
    current = os.environ.get("FLYDSL_RUNTIME_CACHE_DIR")
    if current and os.access(current, os.W_OK):
        return
    path = Path.home() / ".flydsl" / "cache"
    path.mkdir(parents=True, exist_ok=True)
    os.environ["FLYDSL_RUNTIME_CACHE_DIR"] = str(path)


class SeamRunner:
    """``sym``: this rank's peer buffer (``seam.peer_bytes(tp)``); ``peers``: every rank's, int64 on
    the device. The launches move the epoch on themselves, so ranks that run the same launches agree
    on it. A test's fault injection (``SGLANG_AMD_MONO_FAULT_INJECT``) is built into every width."""

    def __init__(
        self,
        tp: int,
        rank: int,
        sym: int,
        peers: torch.Tensor,
        device: torch.device,
        widths: tuple[int, ...] = tuple(range(1, seam.MAX_TOKENS + 1)),
    ) -> None:
        _writable_flydsl_cache()
        self.tp, self.rank, self.sym, self.peers = tp, rank, sym, peers
        self.widths = tuple(widths)
        fault = guard.fault_injection(rank)
        site, after = fault if fault is not None else (0, 0)
        self.keys = {s: seam.SeamBuild(tokens=s, tp=tp, fault_site=site, fault_after=after) for s in self.widths}
        self.epoch = torch.zeros(seam.EPOCH_WORDS, dtype=torch.int32, device=device)
        self.set_deadline(DEFAULT_DEADLINE_S)
        self.scratch = {s: torch.zeros(seam.scratch_bytes(s), dtype=torch.uint8, device=device) for s in self.widths}
        self._launch = {s: seam.build_seam(key) for s, key in self.keys.items()}
        self._compiled: dict[int, Any] = {}
        # each width's resident CTAs per CU, as the HIP driver reports for its code object
        self.residency: dict[int, int] = {}

    def set_deadline(self, seconds: float) -> None:
        """Every wait's deadline, as the guard block's clock ticks."""
        ticks = guard.deadline_ticks(seconds)
        lo = ticks & 0xFFFFFFFF
        self.epoch[guard.GUARD_DEADLINE] = lo - (1 << 32) if lo >= 1 << 31 else lo
        self.epoch[guard.GUARD_DEADLINE + 1] = ticks >> 32

    def set_epoch(self, value: int) -> None:
        """Start from epoch ``value`` (mod 2**32), as if a launch had just moved it there."""
        word = value & 0xFFFFFFFF
        word = word - (1 << 32) if word >= 1 << 31 else word
        self.epoch[0] = word
        self.epoch[EPOCH_MARKS : EPOCH_MARKS + seam.BLOCKS] = word

    def error_words(self) -> list[int]:
        """The guard's sticky words: a timed-out wait's site + 1, a failure seen elsewhere, a fault."""
        return [int(self.epoch[i]) for i in guard.ERROR_WORDS]

    def _args(self, s, attn, residual, res_out, x, w, mode):
        return (
            attn.data_ptr(),
            residual.data_ptr(),
            res_out.data_ptr(),
            x.data_ptr(),
            w.wo.data_ptr(),
            w.wos.data_ptr(),
            w.norm_w.data_ptr(),
            self.scratch[s].data_ptr(),
            self.sym,
            self.peers.data_ptr(),
            self.rank,
            self.epoch.data_ptr(),
            mode,
            torch.cuda.current_stream(),
        )

    def warm(self, weights, compute_units: int) -> None:
        """Compile and load every width (a launch that returns at once), refusing a width whose grid
        cannot be resident on ``compute_units`` CUs."""
        dummy = torch.zeros(1, seam.HIDDEN, dtype=torch.bfloat16, device=self.epoch.device)
        for s in self.widths:
            if s not in self._compiled:
                self._compiled[s] = flyc.compile(
                    self._launch[s], *self._args(s, dummy, dummy, dummy, dummy, weights, 0)
                )
            name = seam.kernel_name(self.keys[s])
            per_cu = active_workgroups_per_cu(self._compiled[s], name, seam.THREADS)
            self.residency[s] = per_cu
            if per_cu * compute_units < seam.BLOCKS:
                raise RuntimeError(
                    f"{name}: {per_cu} resident CTA(s) per CU x {compute_units} CUs < its {seam.BLOCKS}-CTA grid"
                )
        torch.cuda.synchronize()

    def __call__(self, attn: torch.Tensor, residual: torch.Tensor, weights) -> tuple[torch.Tensor, torch.Tensor]:
        """``attn``: this rank's attention output [S, HIDDEN / tp]; ``residual``: the residual stream
        [S, HIDDEN], the same on every rank. Returns (x, res_out): the post-attention norm's output
        and the residual after the attention block, the same on every rank."""
        s = attn.shape[0]
        if s not in self._compiled:
            raise KeyError(f"width {s} is not warmed; this runner holds {sorted(self._compiled)}")
        if attn.dtype != torch.bfloat16 or tuple(attn.shape) != (s, seam.HIDDEN // self.tp) or not attn.is_contiguous():
            raise ValueError(
                f"attn must be a contiguous bf16 [{s}, {seam.HIDDEN // self.tp}], got {attn.dtype} {tuple(attn.shape)}"
            )
        if (
            residual.dtype != torch.bfloat16
            or tuple(residual.shape) != (s, seam.HIDDEN)
            or not residual.is_contiguous()
        ):
            raise ValueError(f"residual must be a contiguous bf16 [{s}, {seam.HIDDEN}]")
        x = torch.empty_like(residual)
        res_out = torch.empty_like(residual)
        self._compiled[s](*self._args(s, attn, residual, res_out, x, weights, 1))
        return x, res_out
