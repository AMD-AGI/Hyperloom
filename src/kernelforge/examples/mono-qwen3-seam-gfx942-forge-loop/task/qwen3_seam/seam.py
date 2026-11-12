# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Qwen3-14B-FP8's attention seam on gfx942: one persistent launch from the attention output to
the post-attention RMSNorm, a decoder layer and a TP rank at a time (TP 2 or 4).

``BLOCKS`` x ``THREADS`` CTAs over S rows (1 .. 16):

    push    (160 tasks of 32 columns): this rank's o_proj partial as SGLang computes it -- the
            rows quantized per 1x128 like ``aiter_per1x128_quant`` (e4m3fnuz, scale amax / 240,
            its correctly rounded inverse, clamped, RNE), then the 128x128 block-FP8 GEMM like
            aiter's Triton ``gemm_a8w8_blockscale`` (``mfma_f32_16x16x32_fp8_fp8``, a K block's
            32-wide steps in its kpack=2 operand order, each block's sum times its row's scale
            fused with the weight block's scale into the accumulator) -- as bf16 pairs to every
            rank's ATTN region (system scope)
    reduce  (the same 160 tasks): the ranks' partials summed in rank order in f32 and rounded
            to bf16 (the all-reduce), + residual in f32 -> ``res_out`` rounded, the f32 sum kept
            in SRES for the norm; the task's sum of squares of each row -> SSQ (device scope)
    norm    (a row a CTA, from CTA 240): the row's 160 sums of squares in task order -> rstd;
            ``x = sum * rstd * norm_w``, as ``aiter.rmsnorm2d_fwd_with_add`` takes the unrounded sum

Every wait is a guarded mailbox poll of ``sgl_amd_mono.common``. A launch's tag is its epoch + 1,
never 0, which a cleared region holds. The mailbox regions (ATTN in the peers, SSQ here) alternate
halves with the epoch's parity, so the one pair of epochs that share a tag, 2**32 - 1 and 0, never
share a region. No rank finishes a launch before every rank has pushed its partials for it, so no
rank runs two launches ahead and no ATTN pair is overwritten before it is read.
"""

import ast
import contextlib
import hashlib
from dataclasses import dataclass
from pathlib import Path

import flydsl.compiler as flyc
import flydsl.expr as fx
import numpy as np
from aiter.ops.flydsl.kernels import buffer_ops as bo
from flydsl.expr import gpu, range_constexpr, rocdl
from flydsl.expr import math as fmath
from flydsl.expr.typing import Int32, Int64, T

from sgl_amd_mono.common import guard
from sgl_amd_mono.common.epoch import EPOCH_MARKS, epoch_end
from sgl_amd_mono.common.ops import (
    CM_DEV,
    bf16_pair,
    bf16_round,
    bf2_f32,
    div_rn,
    fp8_pack4,
    hw_rcp,
    hw_rsq,
    ld_i32,
    peer_bases,
    rsrc,
    sum_partials,
    uniform,
    xshfl,
)
from sgl_amd_mono.common.plan import (
    BLOCKS,
    PAIR_BYTES,
    THREADS,
    WAVES,
    first_task,
    kernel_sources,
    key_tuple,
    pair_layout,
)
from sgl_amd_mono.common.sync import Mailbox, preg, publish, sreg

HIDDEN = 5120
GROUP = 128  # the activation quantization group and the weight block edge
MAX_TOKENS = 16
EPS = 1e-6
FP8_MAX = 240.0  # e4m3fnuz
# aiter's (float)(1. / max): the inverse taken in double, rounded once to f32
INV_FP8_MAX = float(np.float32(1.0 / FP8_MAX))
COLS = 32  # columns of a push / reduce task
TILES = COLS // 16  # 16-column MFMA tiles of a task, a wave each
PAIRS = COLS // 2  # bf16 pairs of a task's row
SEAM_TASKS = HIDDEN // COLS  # 160, CTAs 0 .. 159
NORM0 = 240  # the norm's rows on CTAs 240 .. 255
EPOCH_WORDS = EPOCH_MARKS + BLOCKS
_STREAM = fx.Stream(None)


def _sources() -> str:
    """The digest of the code the launch traces: this package's modules, and the runtime and aiter
    FlyDSL modules they import (``plan.kernel_sources``), as FlyDSL keys a kernel on its own
    functions' source only."""
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        source = path.read_bytes()
        h.update(source)
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module, *(f"{node.module}.{a.name}" for a in node.names)]
            else:
                continue
            for name in names:
                # not a module of a traced package (an attribute, or numpy)
                with contextlib.suppress(FileNotFoundError):
                    h.update(kernel_sources(name).encode())
    return h.hexdigest()[:16]


SOURCES = _sources()


@dataclass(frozen=True)
class SeamBuild:
    tokens: int
    tp: int = 4
    # SGLANG_AMD_MONO_FAULT_INJECT (``guard.fault_injection``): the wait that stalls
    fault_site: int = 0
    fault_after: int = 0


def attn_k(tp: int) -> int:
    """A rank's o_proj input width: its heads x 128."""
    return HIDDEN // tp


def xq_row(tp: int) -> int:
    """LDS words of a row's FP8 activations, four a word, 16 bytes off the banks."""
    return attn_k(tp) // 4 + 4


def peer_half_bytes(tp: int) -> int:
    """One parity's ATTN region: every rank's bf16-pair partial of every row."""
    return tp * MAX_TOKENS * (HIDDEN // 2) * PAIR_BYTES


def flags_offset(tp: int) -> int:
    """The guard's peer flags, past the two parity halves."""
    return 2 * peer_half_bytes(tp)


def peer_bytes(tp: int) -> int:
    return flags_offset(tp) + guard.FLAG_BYTES


def ssq_half_bytes(s: int) -> int:
    return s * SEAM_TASKS * PAIR_BYTES


def scratch_layout(s: int) -> dict[str, tuple[int, int]]:
    """SSQ (two parity halves), then SRES (the rows' f32 sums)."""
    out = pair_layout([("ssq", 2 * s * SEAM_TASKS)], align=256)
    off = -(-max(o + n for o, n in out.values()) // 256) * 256
    out["sres"] = (off, s * HIDDEN * 4)
    return out


def scratch_bytes(s: int) -> int:
    return max(o + n for o, n in scratch_layout(s).values())


def kernel_name(key: SeamBuild) -> str:
    name = f"qwen3_seam_s{key.tokens}_tp{key.tp}"
    if key.fault_site:
        name += f"_fault{key.fault_site}_{key.fault_after}"
    return name


def attn_pair(src, t, task, q):
    """The ATTN pair of (source rank, row t, the task's pair q)."""
    return (src * MAX_TOKENS + t) * (HIDDEN // 2) + task * PAIRS + q


def tag_of(ep):
    tag = ep + 1
    return (tag == 0).select(fx.Int32(1), tag)


# ---------------------------------------------------------------- push
@flyc.jit
def quantize_rows(c):
    """Every row's FP8 activations -> LDS XQ, their per-1x128 dequant scales -> LDS XS.

    A group of 128 is 32 lanes of four elements, two groups a wave; with an odd group
    count the upper half of the last wave repeats the last group."""
    s, kb_n, k, lane, wave, a = c["S"], c["kb"], c["k"], c["lane"], c["wave"], c["args"]
    half, h = lane // 32, lane % 32
    groups = s * kb_n
    for g0 in range(2 * wave, groups, 2 * WAVES):
        g = fx.min(g0 + half, groups - 1)
        t, kb = g // kb_n, g % kb_n
        w = fx.Vector(bo.buffer_load(rsrc(a["attn"]), (t * k + kb * GROUP + 4 * h) // 2, vec_width=2, dtype=T.i32))
        vals = [bf2_f32(w[0])[0], bf2_f32(w[0])[1], bf2_f32(w[1])[0], bf2_f32(w[1])[1]]
        amax = fx.Float32(1e-10)
        for v in vals:
            amax = fx.max(amax, fmath.absf(v))
        for off in (16, 8, 4, 2, 1):
            amax = fx.max(amax, xshfl(amax, off))
        scale = amax * fx.Float32(INV_FP8_MAX)
        inv = div_rn(fx.Float32(1.0), scale, hw_rcp(scale))
        q8 = [fx.min(fx.max(v * inv, fx.Float32(-FP8_MAX)), fx.Float32(FP8_MAX)) for v in vals]
        fx.ptr_store(fp8_pack4(*q8), c["xq"] + (t * c["row"] + kb * (GROUP // 4) + h))
        if h == 0:
            fx.ptr_store(scale, c["xs"] + (t * kb_n + kb))


@flyc.jit
def oproj_tiles(c, task):
    """Waves 0 .. TILES-1: the task's 16-column tiles of this rank's o_proj partial -> LDS PART."""
    lane, wave, k, kb_n, a = c["lane"], c["wave"], c["k"], c["kb"], c["args"]
    if wave < TILES:
        v2i = fx.Vector.make_type(2, fx.Int32)
        l16, g = lane % 16, lane // 16
        n0 = task * COLS + wave * 16
        rw = rsrc(a["wo"])
        acc = [fx.Float32(0.0) for _ in range(4)]
        for kb in range_constexpr(kb_n):
            tile = fx.Vector.filled(4, 0.0, fx.Float32)
            for st in range_constexpr(GROUP // 32):
                # Triton's kpack=2 operand: a lane holds 16 consecutive K of each 64, its low
                # eight in step 2j and its high eight in step 2j + 1
                kk = kb * GROUP + (st // 2) * 64 + g * 16 + (st % 2) * 8
                a64 = fx.ptr_load(c["xq"] + (l16 * c["row"] + kk // 4), result_type=v2i).bitcast(fx.Int64)[0]
                b64 = fx.Vector(bo.buffer_load(rw, ((n0 + l16) * k + kk) // 4, vec_width=2, dtype=T.i32)).bitcast(
                    fx.Int64
                )[0]
                tile = fx.Vector(rocdl.mfma_f32_16x16x32_fp8_fp8(T.vec(4, T.f32), [a64, b64, tile, 0, 0, 0]))
            ws = fx.Float32(bo.buffer_load(rsrc(a["wos"]), (n0 // GROUP) * kb_n + kb, vec_width=1, dtype=T.f32))
            for i in range_constexpr(4):
                a_s = fx.ptr_load(c["xs"] + ((4 * g + i) * kb_n + kb))
                acc[i] = fmath.fma(tile[i] * a_s, ws, acc[i])
        # MFMA D: lane l holds rows 4 (l / 16) + i of column l % 16
        for i in range_constexpr(4):
            fx.ptr_store(acc[i], c["part"] + ((4 * g + i) * COLS + wave * 16 + l16))


@flyc.jit
def stage_push(c, task):
    """Columns 32 task ..: this rank's o_proj partial, as bf16 pairs to every rank's ATTN."""
    quantize_rows(c)
    gpu.barrier()
    oproj_tiles(c, task)
    gpu.barrier()
    s, tid = c["S"], c["tid"]
    if tid < s * PAIRS:
        t, q = tid // PAIRS, tid % PAIRS
        v0 = fx.ptr_load(c["part"] + (t * COLS + 2 * q))
        v1 = fx.ptr_load(c["part"] + (t * COLS + 2 * q + 1))
        at = 2 * attn_pair(c["rank"], t, task, q)
        for p in range_constexpr(c["tp"]):
            c["mb"].put_bf(c["peer"][p], at, [v0, v1])


# ---------------------------------------------------------------- reduce and norm
@flyc.jit
def stage_reduce(c, task):
    """Columns 32 task ..: the ranks' partials in rank order (the all-reduce's bf16), + residual
    -> ``res_out`` and SRES; each row's sum of squares over the task's columns -> SSQ."""
    s, tid, a = c["S"], c["tid"], c["args"]
    sq = fx.Float32(0.0)
    if tid < s * PAIRS:
        t, q = tid // PAIRS, tid % PAIRS
        acc0, acc1 = sum_partials(c["mb"].poll, c["own"], lambda src: attn_pair(src, t, task, q), c["tp"])
        col = task * COLS + 2 * q
        r0, r1 = bf2_f32(ld_i32(a["res_in"], (t * HIDDEN + col) // 2))
        v0 = bf16_round(acc0) + r0
        v1 = bf16_round(acc1) + r1
        bo.buffer_store(bf16_pair(v0, v1).bitcast(fx.Int32), rsrc(a["res_out"]), (t * HIDDEN + col) // 2)
        bo.buffer_store(
            fx.Vector.from_elements([v0.bitcast(fx.Int32), v1.bitcast(fx.Int32)], fx.Int32),
            rsrc(c["sres"].value),
            t * HIDDEN + col,
            cache_modifier=CM_DEV,
        )
        sq = v0 * v0 + v1 * v1
    for off in (8, 4, 2, 1):
        sq = sq + xshfl(sq, off)
    publish(c["mb"].put, c["ssq"], (tid // PAIRS) * SEAM_TASKS + task, sq, (tid < s * PAIRS) & (tid % PAIRS == 0))


@flyc.jit
def block_sum(c, v):
    """``v`` summed over the CTA: each wave's xor butterfly, then the waves in order."""
    lane, wave, red = c["lane"], c["wave"], c["red"]
    for off in (32, 16, 8, 4, 2, 1):
        v = v + xshfl(v, off)
    if lane == 0:
        fx.ptr_store(v, red + wave)
    gpu.barrier()
    total = fx.Float32(0.0)
    for w in range_constexpr(WAVES):
        total = total + fx.ptr_load(red + w)
    return total


@flyc.jit
def stage_norm(c, t):
    """Row t: its 160 sums of squares -> rstd; ``x = sum * rstd * norm_w``."""
    tid, a = c["tid"], c["args"]
    part = fx.Float32(0.0)
    if tid < SEAM_TASKS:
        part = c["mb"].poll([(c["ssq"], t * SEAM_TASKS + tid, 1)])[0][0].bitcast(fx.Float32)
    gpu.barrier()
    rstd = hw_rsq(block_sum(c, part) * fx.Float32(1.0 / HIDDEN) + fx.Float32(EPS))
    for j in range_constexpr(HIDDEN // 2 // THREADS):
        cp = tid + THREADS * j
        sv = fx.Vector(
            bo.buffer_load(rsrc(c["sres"].value), t * HIDDEN + 2 * cp, vec_width=2, dtype=T.i32, cache_modifier=CM_DEV)
        )
        w0, w1 = bf2_f32(ld_i32(a["norm_w"], cp))
        x0 = sv[0].bitcast(fx.Float32) * rstd * w0
        x1 = sv[1].bitcast(fx.Float32) * rstd * w1
        bo.buffer_store(bf16_pair(x0, x1).bitcast(fx.Int32), rsrc(a["x_out"]), t * (HIDDEN // 2) + cp)


# ---------------------------------------------------------------- the launch
@flyc.jit
def run_seam(c, bid):
    for task in range(first_task(bid, 0), SEAM_TASKS, BLOCKS):
        stage_push(c, task)
    for task in range(first_task(bid, 0), SEAM_TASKS, BLOCKS):
        stage_reduce(c, task)
    for t in range(first_task(bid, NORM0), c["S"], BLOCKS):
        stage_norm(c, t)


@flyc.jit
def finish(c, epoch, ep):
    """Every CTA marks the launch done after its work; CTA 0 waits for every mark and moves the
    epoch on (``common.epoch``)."""
    rocdl.s_waitcnt(vmcnt=0)
    gpu.barrier()
    if c["tid"] == 0:
        bo.buffer_store(ep + 1, rsrc(epoch), EPOCH_MARKS + c["bid"], cache_modifier=CM_DEV)
    epoch_end({"bid": c["bid"], "tid": c["tid"]}, epoch, ep)


def build_seam(key: SeamBuild):
    s, tp = key.tokens, key.tp
    assert 1 <= s <= MAX_TOKENS and tp in (2, 4), key
    k, kb_n, row = attn_k(tp), attn_k(tp) // GROUP, xq_row(tp)
    layout = scratch_layout(s)
    half, ssq_half = peer_half_bytes(tp), ssq_half_bytes(s)
    keyed = key_tuple(key, SOURCES)

    @fx.struct
    class Lds:
        xq: fx.Array[fx.Int32, MAX_TOKENS * row, 16]
        xs: fx.Array[fx.Float32, MAX_TOKENS * kb_n, 16]
        part: fx.Array[fx.Float32, MAX_TOKENS * COLS, 16]
        red: fx.Array[fx.Float32, WAVES, 16]

    @flyc.kernel(name=kernel_name(key), known_block_size=[THREADS, 1, 1])
    def kseam(
        attn: Int64,
        res_in: Int64,
        res_out: Int64,
        x_out: Int64,
        wo: Int64,
        wos: Int64,
        norm_w: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        mode: Int32,
    ):
        _ = keyed
        with guard.scope(
            guard.Guard(
                epoch,
                own=sym,
                peers=peers,
                rank=rank,
                npes=tp,
                flags_offset=flags_offset(tp),
                fault_site=key.fault_site or None,
                fault_after=key.fault_after,
            )
        ):
            if mode != 0:
                tid = fx.thread_idx.x
                bid = fx.block_idx.x
                lds = fx.SharedAllocator().allocate(Lds).peek()
                ep = uniform(ld_i32(epoch, 0, CM_DEV))
                odd = fx.Int64(ep & 1)
                par = odd * fx.Int64(half)
                c = {
                    "S": s,
                    "tp": tp,
                    "k": k,
                    "kb": kb_n,
                    "row": row,
                    "tid": tid,
                    "bid": bid,
                    "lane": tid % 64,
                    "wave": tid // 64,
                    "rank": rank,
                    "mb": Mailbox(tag_of(ep)),
                    "xq": lds.xq.ptr,
                    "xs": lds.xs.ptr,
                    "part": lds.part.ptr,
                    "red": lds.red.ptr,
                    "own": preg(sym + par, 0, "attn"),
                    "peer": [preg(b + par, 0, "attn") for b in peer_bases(peers, tp)],
                    "ssq": sreg(scratch + odd * fx.Int64(ssq_half), layout["ssq"][0], "ssq"),
                    "sres": sreg(scratch, layout["sres"][0], "sres"),
                    "args": {
                        "attn": attn,
                        "res_in": res_in,
                        "res_out": res_out,
                        "x_out": x_out,
                        "wo": wo,
                        "wos": wos,
                        "norm_w": norm_w,
                    },
                }
                run_seam(c, bid)
                finish(c, epoch, ep)

    @flyc.jit
    def launch(
        attn: Int64,
        res_in: Int64,
        res_out: Int64,
        x_out: Int64,
        wo: Int64,
        wos: Int64,
        norm_w: Int64,
        scratch: Int64,
        sym: Int64,
        peers: Int64,
        rank: Int32,
        epoch: Int64,
        mode: Int32,
        stream: fx.Stream = _STREAM,
    ):
        _ = keyed
        kseam(attn, res_in, res_out, x_out, wo, wos, norm_w, scratch, sym, peers, rank, epoch, mode).launch(
            grid=(BLOCKS,), block=(THREADS,), stream=stream
        )

    return launch


__all__ = [
    "BLOCKS",
    "EPOCH_WORDS",
    "HIDDEN",
    "MAX_TOKENS",
    "THREADS",
    "SeamBuild",
    "build_seam",
    "kernel_name",
    "peer_bytes",
    "scratch_bytes",
]
