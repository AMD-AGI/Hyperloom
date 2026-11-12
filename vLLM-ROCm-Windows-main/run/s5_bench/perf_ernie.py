# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""ERNIE-4.5-21B-A3B (compressed-tensors W4A16 gs32 MoE, head_dim 128 uniform, no sliding) on Win ROCm.
Fits VRAM with big headroom (~11GB weights on 20GB) -> NO WDDM shared-memory spill -> reliable tok/s.
head 128 uniform => S5 works GLOBALLY (attention_backend=ROCM_ATTN, no per-layer routing).
S5_ROUTE=1 -> ROCM_ATTN + native paged_attention_v1 decode swap; =0 -> TRITON_ATTN baseline.
OPT=1 enables the MoE-decode GEMV + wvSplitK dense (56.8-style opt path). Reports free-VRAM to prove no spill."""
import os, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
OPT = os.environ.get("OPT", "0") == "1"
if OPT:
    os.environ.setdefault("VLLM_WIN_MOE_DECODE", "1")
    os.environ.setdefault("VLLM_WIN_ROCM_C", "1")
    os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "1")
else:
    os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")

ROUTE = os.environ.get("S5_ROUTE", "0") == "1"
if ROUTE:
    os.environ["VLLM_WIN_ATTN_NATIVE"] = "1"
    os.environ["VLLM_WIN_CACHE_NATIVE"] = "1"
FLASH = os.environ.get("FLASH", "0") == "1"
if FLASH:
    os.environ["VLLM_WIN_FLASH_ATTN"] = "1"  # swap unified_attention -> native flash kernel (pure decode)

import torch
from vllm import LLM, SamplingParams

MODEL = os.environ.get("G_MODEL", "cyankiwi/ERNIE-4.5-21B-A3B-Thinking-AWQ-4bit")
MAXLEN = int(os.environ.get("G_MAXLEN", "4096"))
GPUUTIL = float(os.environ.get("G_GPUUTIL", "0.85"))
CG = os.environ.get("G_CUDAGRAPH", "1") == "1"
GIB = 2 ** 30

if ROUTE:
    try:
        import vllm_windows_rocm.cops as _C5
        _C5.maybe_patch_s5_decode()
    except Exception as e:
        print("S5 cops patch failed:", repr(e))
if FLASH:
    try:
        import vllm_windows_rocm.cops as _C5
        _C5.maybe_patch_flash_decode()
    except Exception as e:
        print("flash patch failed:", repr(e))

free0, total = torch.cuda.mem_get_info()
KVDTYPE = os.environ.get("G_KVDTYPE", "auto")
BACKEND = "ROCM_ATTN" if ROUTE else "TRITON_ATTN"
extra = (dict(enforce_eager=False, compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"})
         if CG else dict(enforce_eager=True))
if KVDTYPE.startswith("kvarn_"):
    BACKEND = None  # per-layer backend selection: KVARN for the quantized layers
    extra["block_size"] = int(os.environ.get("G_BLOCKSIZE", "128"))
_nb = os.environ.get("G_NUMBLOCKS")
if _nb:
    extra["num_gpu_blocks_override"] = int(_nb)  # cap KV blocks (bypass KVarN's buggy profiled sizing)
print(f"== ERNIE | S5={ROUTE} FLASH={FLASH} OPT={OPT} kv={KVDTYPE} backend={BACKEND} cg={CG} | free {free0/GIB:.1f}/{total/GIB:.1f} GiB maxlen={MAXLEN} util={GPUUTIL}")
t0 = time.perf_counter()
llm = LLM(model=MODEL, dtype=os.environ.get("G_DTYPE", "bfloat16"), attention_backend=BACKEND,
    tensor_parallel_size=1, gpu_memory_utilization=GPUUTIL, max_model_len=MAXLEN,
    kv_cache_dtype=KVDTYPE, trust_remote_code=False, **extra)  # vLLM native ernie45_moe.py (config auto_map .py not downloaded)
print(f"engine init: {time.perf_counter()-t0:.1f}s")
free1, _ = torch.cuda.mem_get_info()
print(f"VRAM used after init: {(total-free1)/GIB:.2f} GiB | FREE {free1/GIB:.2f} GiB  (>2 GiB free => no spill)")

prompt = "Write a detailed essay about the history and future of GPU computing. " * 40  # ~450 tok ctx
llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=8))
torch.cuda.synchronize()
best = 0.0
for _ in range(3):
    t0 = time.perf_counter()
    out = llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=128, ignore_eos=True))
    torch.cuda.synchronize()
    best = max(best, len(out[0].outputs[0].token_ids) / (time.perf_counter() - t0))
free2, _ = torch.cuda.mem_get_info()
print(f"S5_ROUTE={ROUTE} FLASH={FLASH} OPT={OPT}  decode tok/s (best of 3): {best:.1f}  | FREE VRAM now {free2/GIB:.2f} GiB")
try:
    import vllm_windows_rocm.cops as _C5
    if ROUTE:
        print(f"S5 native paged_attention_v1 calls: {_C5._S5_NATIVE_CALLS[0]}")
    if FLASH:
        print(f"FLASH paged_attention_flash calls: {_C5._FLASH_CALLS[0]}")
except Exception:
    pass
co = llm.generate(["Question: What is the capital of France? Give a one-line answer."],
                  SamplingParams(temperature=0.0, max_tokens=64))
print("COHERENCE:", repr(co[0].outputs[0].text[:220]))
print("PERF_ERNIE_OK")
