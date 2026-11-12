# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""
v5_splitk_2pass: SPLIT-K two-pass W4A16 dequant-GEMV (M=1) for gfx1100 / RDNA3.

STRATEGY
--------
Two deterministic passes (no atomics):

  Pass 1 (partials kernel): grid = (ceil(N/BLOCK_N), K_SPLITS).
    Each program owns one (N-tile, K-slice). It walks ONLY the groups that fall in
    its K-slice, applies scales/zeros there, and writes a fp32 partial sum to
    partials[k_split, n]. This raises occupancy at small N: instead of one program
    per N-tile serially chewing all of K, we now have K_SPLITS x N-tiles programs,
    enough to fill all 84 CUs even for tiny N.

  Pass 2 (combine kernel): grid = (ceil(N/BLOCK_N2),).
    Each program loads the [K_SPLITS] fp32 partials for its N columns and sums
    them -> final c[n] in fp16. Cheap; partials array is tiny (K_SPLITS x N fp32).

Unpack is reshape-FREE: we loop the 8 nibble positions, shift+mask each int32 row
and accumulate a[k]*q directly into the fp32 accumulator, avoiding tl.reshape
overhead that dominated small-N tiles. K_SPLITS and BLOCK_N are autotuned; the
weight read (the bandwidth-bound term) happens exactly once across both passes.
"""

import torch
import triton
import triton.language as tl

G = 128


# ---------------------------------------------------------------------------
# Pass 1: per (N-tile, K-slice) -> fp32 partial sums, scales/zeros applied.
# ---------------------------------------------------------------------------
def _pass1_configs():
    cfgs = []
    for block_n in (16, 32, 64, 128, 256):
        for ksplit in (2, 4, 8, 16):
            for nw in (1, 2, 4):
                for wpe in (0, 1, 2):
                    cfgs.append(
                        triton.Config(
                            {"BLOCK_N": block_n, "K_SPLITS": ksplit, "waves_per_eu": wpe},
                            num_warps=nw,
                            num_stages=2,
                        )
                    )
    return cfgs


@triton.autotune(configs=_pass1_configs(), key=["K", "N"])
@triton.jit
def _gemv_splitk_pass1(
    a_ptr, qw_ptr, s_ptr, z_ptr, part_ptr,
    K, N,
    BLOCK_N: tl.constexpr,
    K_SPLITS: tl.constexpr,
    GROUP: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    nmask = n < N

    ROWS: tl.constexpr = GROUP // 8          # int32 rows per group
    n_groups = K // GROUP                     # total groups along K

    # Contiguous slice of groups assigned to this K-split.
    # ceil division so all groups are covered with no overlap.
    groups_per_split = (n_groups + K_SPLITS - 1) // K_SPLITS
    g_start = pid_k * groups_per_split
    g_end = tl.minimum(g_start + groups_per_split, n_groups)

    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    for g in range(g_start, g_end):
        k0 = g * GROUP
        row0 = k0 // 8
        rows = row0 + tl.arange(0, ROWS)                       # [ROWS] int32 rows

        # [ROWS, BLOCK_N] packed weights, contiguous in N (coalesced).
        qw = tl.load(
            qw_ptr + rows[:, None] * N + n[None, :],
            mask=nmask[None, :],
            other=0,
        )

        # reshape-FREE unpack: loop the 8 nibble positions within each int32 row.
        contrib = tl.zeros((BLOCK_N,), dtype=tl.float32)
        asum = tl.zeros((1,), dtype=tl.float32)
        for nib in tl.static_range(8):
            # global k for (row r, nibble nib) = (row0 + r) * 8 + nib
            kk = k0 + tl.arange(0, ROWS) * 8 + nib            # [ROWS]
            av = tl.load(a_ptr + kk).to(tl.float32)            # [ROWS]
            q = ((qw >> (nib * 4)) & 0xF).to(tl.float32)       # [ROWS, BLOCK_N]
            contrib += tl.sum(av[:, None] * q, axis=0)         # [BLOCK_N]
            asum += tl.sum(av)

        s = tl.load(s_ptr + g * N + n, mask=nmask, other=0.0).to(tl.float32)
        z = tl.load(z_ptr + g * N + n, mask=nmask, other=0.0).to(tl.float32)
        acc += (contrib - z * asum) * s

    # partials laid out [K_SPLITS, N]
    tl.store(part_ptr + pid_k * N + n, acc, mask=nmask)


# ---------------------------------------------------------------------------
# Pass 2: sum partials over K_SPLITS -> final fp16 output.
# ---------------------------------------------------------------------------
def _pass2_configs():
    return [
        triton.Config({"BLOCK_N2": bn, "waves_per_eu": wpe}, num_warps=nw, num_stages=1)
        for bn in (64, 128, 256, 512)
        for nw in (1, 2, 4)
        for wpe in (0, 1, 2)
    ]


@triton.autotune(configs=_pass2_configs(), key=["N", "K_SPLITS"])
@triton.jit
def _gemv_splitk_pass2(
    part_ptr, c_ptr,
    N, K_SPLITS,
    BLOCK_N2: tl.constexpr,
):
    pid = tl.program_id(0)
    n = pid * BLOCK_N2 + tl.arange(0, BLOCK_N2)
    nmask = n < N

    acc = tl.zeros((BLOCK_N2,), dtype=tl.float32)
    for ks in range(0, K_SPLITS):
        p = tl.load(part_ptr + ks * N + n, mask=nmask, other=0.0)
        acc += p

    tl.store(c_ptr + n, acc.to(c_ptr.type.element_ty), mask=nmask)


def gemv(a, w_q, w_s, w_zp, group_size):
    """M=1 W4A16 dequant-GEMV. a:[1,K] fp16, w_q:[K//8,N] int32, w_s/w_zp:[K//G,N].
    Returns c:[1,N] fp16."""
    assert group_size == G, "this variant assumes group_size == 128"
    K8, N = w_q.shape
    K = K8 * 8
    assert K % G == 0

    a = a.contiguous()
    w_q = w_q.contiguous()
    w_s = w_s.contiguous()
    w_zp = w_zp.contiguous()

    c = torch.empty((1, N), dtype=torch.float16, device=a.device)

    # Worst-case K_SPLITS in the autotune set so the partials buffer is large enough.
    MAX_K_SPLITS = 16
    partials = torch.empty((MAX_K_SPLITS, N), dtype=torch.float32, device=a.device)

    def grid1(meta):
        return (triton.cdiv(N, meta["BLOCK_N"]), meta["K_SPLITS"])

    _gemv_splitk_pass1[grid1](
        a, w_q, w_s, w_zp, partials,
        K, N,
        GROUP=G,
    )

    # The K_SPLITS actually used by pass1's chosen config:
    chosen_ksplits = _gemv_splitk_pass1.best_config.kwargs["K_SPLITS"]

    def grid2(meta):
        return (triton.cdiv(N, meta["BLOCK_N2"]),)

    _gemv_splitk_pass2[grid2](
        partials, c,
        N, chosen_ksplits,
    )
    return c


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def ref_dequant_matmul(a, w_q, w_s, w_zp, group_size):  # ground truth
    K8, N = w_q.shape
    K = K8 * 8
    shifts = (torch.arange(8, device=w_q.device, dtype=torch.int32) * 4)
    q = ((w_q.unsqueeze(1) >> shifts.view(1, 8, 1)) & 0xF).reshape(K, N).float()  # k=row*8+nibble
    gidx = torch.arange(K, device=w_q.device) // group_size
    w = (q - w_zp.float()[gidx]) * w_s.float()[gidx]
    return a.float() @ w


if __name__ == "__main__":
    torch.manual_seed(0)
    dev = "cuda"
    shapes = [(5120, 5120), (5120, 7168), (13824, 5120), (5120, 27648)]
    max_rel = 0.0
    try:
        for (K, N) in shapes:
            G_ = 128
            a = torch.randn((1, K), dtype=torch.float16, device=dev)
            w_q = torch.randint(
                -(2**31), 2**31, (K // 8, N), dtype=torch.int32, device=dev
            )
            w_s = (torch.randn((K // G_, N), device=dev).abs() * 0.01 + 0.005).to(torch.float16)
            w_zp = torch.randint(0, 16, (K // G_, N), dtype=torch.uint8, device=dev)

            c = gemv(a, w_q, w_s, w_zp, G_)
            ref = ref_dequant_matmul(a, w_q, w_s, w_zp, G_).to(torch.float32)
            got = c.to(torch.float32)
            rel = (got - ref).abs().max().item() / (ref.abs().max().item() + 1e-6)
            max_rel = max(max_rel, rel)
            print(f"  (K={K},N={N}) rel={rel:.3e}")
        print(f"VARIANT_OK rel={max_rel:.3e}")
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"VARIANT_ERROR {e}")
