# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""S5 de-risk PART B: Triton kernel_paged_attention_2d (the REAL ROCM_ATTN decode baseline) on the v0 paged
KV layout. Same shapes as mb_native.py (1 decode token, head 256, GQA 16/8, block 16). Import vllm (no native
pyd -> no _C conflict). Prints us/call. Compare directly with mb_native.py's NATIVE numbers."""
import os, time
os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
import torch, triton
from vllm.v1.attention.ops.chunked_prefill_paged_decode import kernel_paged_attention_2d
print("kpa2d imported")

dev, dt = "cuda", torch.float16
NQ, NKV, HD, BS = 16, 8, 256, 16
X = 16 // dt.itemsize
QPK = NQ // NKV
scale = 1.0 / (HD ** 0.5)
W = int(os.environ.get("MB_SW", "0"))

def bench(fn, it=300, wu=60):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / it * 1e6

for SEQ in (512, 1024, 2048, 4096):
    NBLK = (SEQ + BS - 1) // BS
    q = torch.randn(1, NQ, HD, device=dev, dtype=dt)
    out = torch.empty(1, NQ, HD, device=dev, dtype=dt)
    kc = torch.randn(NBLK, NKV, HD // X, BS, X, device=dev, dtype=dt)
    vc = torch.randn(NBLK, NKV, HD, BS, device=dev, dtype=dt)
    btab = torch.arange(NBLK, device=dev, dtype=torch.int32).view(1, NBLK)
    slens = torch.tensor([SEQ], device=dev, dtype=torch.int32)
    qsl = torch.tensor([0, 1], device=dev, dtype=torch.int32)  # query_start_loc [num_seqs+1]
    def run():
        kernel_paged_attention_2d[(1, NKV)](
            output_ptr=out, query_ptr=q, key_cache_ptr=kc, value_cache_ptr=vc,
            sink_ptr=None, block_tables_ptr=btab, seq_lens_ptr=slens, alibi_slopes_ptr=None,
            scale=scale, k_scale=1.0, v_scale=1.0, out_scale_inv=1.0,
            num_query_heads=NQ, num_queries_per_kv=QPK,
            num_queries_per_kv_padded=triton.next_power_of_2(QPK),
            block_table_stride=btab.stride(0), query_stride_0=q.stride(0), query_stride_1=q.stride(1),
            output_stride_0=out.stride(0), output_stride_1=out.stride(1),
            BLOCK_SIZE=BS, PHYSICAL_BLOCK_SIZE=vc.shape[3], HEAD_SIZE=HD,
            HEAD_SIZE_PADDED=triton.next_power_of_2(HD), USE_ALIBI_SLOPES=False, SLIDING_WINDOW=W, x=X,
            stride_k_cache_0=kc.stride(0), stride_k_cache_1=kc.stride(1), stride_k_cache_2=kc.stride(2),
            stride_k_cache_3=kc.stride(3), stride_k_cache_4=kc.stride(4),
            stride_v_cache_0=vc.stride(0), stride_v_cache_1=vc.stride(1), stride_v_cache_2=vc.stride(2),
            stride_v_cache_3=vc.stride(3),
            filter_by_query_len=True, query_start_len_ptr=qsl, USE_SINKS=False, USE_FP8=False)
    try:
        run(); t = bench(run)
        print(f"KPA2D seq={SEQ:5d}: {t:7.1f} us")
    except Exception as e:
        print(f"KPA2D seq={SEQ:5d} FAILED: {repr(e)[:220]}")
print("MB_KPA2D_DONE")
