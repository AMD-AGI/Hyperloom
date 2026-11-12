# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 CLVwp contributors
"""Direct check of the per-(head, channel) fp8 K scale kernels (issue #28 option 2).

Runs the patched Triton kernels exactly as the engine calls them, on/off, and
compares against a torch reference. Bit-exact expected: same ops, same order.

  python kv_pc_kernel_check.py    (needs one GPU; prints KERNEL_CHECK_OK)
"""
import os

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch

from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
    triton_reshape_and_cache_flash,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

DEV = "cuda"
FP8 = torch.float8_e4m3fn
torch.manual_seed(0)


def to_fp8(x):  # Triton store semantics: clamp to 448, implicit e4m3 cast
    return x.clamp(-448.0, 448.0).to(FP8)


def ref_store(key, pc):
    s = torch.ones((), device=key.device) if pc is None else None
    if pc is None:
        return to_fp8(key)
    return to_fp8(key / pc)


def check_store(layout: str) -> None:
    T, Hk, D, BS, NB = 10, 4, 128, 16, 4
    x = 8
    key = torch.randn(T, Hk, D, device=DEV) * 30
    value = torch.randn(T, Hk, D, device=DEV)
    slot = torch.arange(T, device=DEV)
    pc = (torch.rand(Hk, D, device=DEV) * 0.05 + 0.01)
    k_scale = torch.ones(1, device=DEV)
    v_scale = torch.ones(1, device=DEV)
    ref = ref_store(key, pc if layout == "flat" else pc)

    if layout == "flat":       # [num_blocks, block_size, Hk, D]
        kc0 = torch.zeros(NB, BS, Hk, D, device=DEV, dtype=FP8)
        vc0 = torch.zeros(NB, BS, Hk, D, device=DEV, dtype=FP8)
        caches = (kc0, vc0)
    else:                       # head-major [num_blocks, Hk, D//x, BS, x]
        kc0 = torch.zeros(NB, Hk, D // x, BS, x, device=DEV, dtype=FP8)
        vc0 = torch.zeros(NB, Hk, D, BS, device=DEV, dtype=FP8)
        caches = (kc0, vc0)

    for use_pc in (False, True):
        kc, vc = [c.clone() for c in caches]
        triton_reshape_and_cache_flash(
            key, value, kc, vc, slot, "fp8", k_scale, v_scale,
            k_scale_channel=pc if use_pc else None,
        )
        got = kc
        if layout == "head-major":  # unpack [NB, Hk, D//x, BS, x] -> [T, Hk, D]
            got = kc.permute(0, 3, 1, 2, 4).reshape(NB, BS, Hk, D)
        got = got[slot.to(torch.int64) // BS, (slot % BS).to(torch.int64)]
        want = ref if use_pc else ref_store(key, None)
        assert torch.equal(got.float(), want.float()), \
            f"store mismatch layout={layout} use_pc={use_pc}"
    print(f"store {layout}: OK (on/off)")


def check_read() -> None:
    L, Hq, Hk, D, BS, NB = 64, 8, 4, 128, 16, 8
    q = torch.randn(L, Hq, D, device=DEV)
    k16 = torch.randn(L, Hk, D, device=DEV) * 20
    v16 = torch.randn(L, Hk, D, device=DEV)
    pc = torch.rand(Hk, D, device=DEV) * 0.05 + 0.01
    k_descale = torch.ones(1, 1, device=DEV)

    for use_pc in (False, True):
        scale = pc if use_pc else torch.ones_like(pc)
        kf = to_fp8(k16 / scale)                 # what the store leaves
        kc = torch.zeros(NB, BS, Hk, D, device=DEV, dtype=FP8)
        vc = torch.zeros(NB, BS, Hk, D, device=DEV, dtype=torch.float16)
        for t in range(L):                       # scatter into page blocks
            b, off = divmod(t, BS)
            kc[b, off] = kf[t]
            vc[b, off] = v16[t]
        out = torch.zeros(L, Hq, D, device=DEV)
        unified_attention(
            q=q, k=kc, v=vc, out=out,
            cu_seqlens_q=torch.tensor([0, L], device=DEV, dtype=torch.int32),
            max_seqlen_q=L, seqused_k=torch.tensor([L], device=DEV),
            max_seqlen_k=L, softmax_scale=D ** -0.5, causal=True,
            window_size=(-1, -1), block_table=torch.arange(NB, device=DEV)[None],
            softcap=0.0, q_descale=None, k_descale=k_descale, v_descale=k_descale,
            k_descale_channel=pc if use_pc else None,
        )
        # torch reference: descale (scalar 1.0 * pc), then causal SDPA per kv head
        kd = (kf.float() * scale).to(q.dtype)
        G = Hq // Hk
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.permute(1, 0, 2).float(),
            kd.repeat_interleave(G, dim=1).permute(1, 0, 2),
            v16.repeat_interleave(G, dim=1).permute(1, 0, 2),
            is_causal=True,
        ).permute(1, 0, 2).reshape(L, Hq, D)
        err = (out.float() - ref).abs().max().item()
        assert err < 2e-2, f"read mismatch use_pc={use_pc}: max abs err {err}"
    print("read 2d: OK (on/off)")


if __name__ == "__main__":
    check_store("flat")
    check_store("head-major")
    check_read()
    print("KERNEL_CHECK_OK")
