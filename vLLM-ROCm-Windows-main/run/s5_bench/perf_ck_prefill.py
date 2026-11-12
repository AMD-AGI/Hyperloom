# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""CK ck_tile FMHA-varlen PREFILL lever on Win ROCm gfx1100. Measures TTFT (prefill latency) at several
prompt lengths -- the axis CK actually moves (compute-bound WMMA attention), NOT single-stream decode.
CK=1 -> VLLM_WIN_CK_PREFILL + TritonAttentionImpl.forward routes pure-prefill attention to CK varlen;
CK=0 -> stock Triton unified_attention. Same model/prompts both runs => the TTFT delta is the CK win.
Chunking disabled (max_num_batched_tokens high) so CK handles the whole prompt in one forward.
Model must be head_size 128, no sliding-window/softcap (ERNIE-4.5 / Qwen2.5-7B qualify)."""
import os, time
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")

CK = os.environ.get("CK", "0") == "1"
if CK:
    os.environ["VLLM_WIN_CK_PREFILL"] = "1"
OPT = os.environ.get("OPT", "0") == "1"
if OPT:
    os.environ.setdefault("VLLM_WIN_MOE_DECODE", "1")
    os.environ.setdefault("VLLM_WIN_ROCM_C", "1")
    os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "1")
else:
    os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")

import torch
from vllm import LLM, SamplingParams

MODEL = os.environ.get("G_MODEL", "cyankiwi/ERNIE-4.5-21B-A3B-Thinking-AWQ-4bit")
MAXLEN = int(os.environ.get("G_MAXLEN", "8192"))
GPUUTIL = float(os.environ.get("G_GPUUTIL", "0.85"))
GIB = 2 ** 30

if CK:
    try:
        import vllm_windows_rocm.cops as _C5
        _C5.maybe_patch_ck_prefill()
    except Exception as e:
        print("CK prefill patch failed:", repr(e))

free0, total = torch.cuda.mem_get_info()
# prefill is eager under FULL_DECODE_ONLY cudagraph; chunking + prefix-cache OFF so every prompt is one
# full prefill forward (query_len==seq_len) -> the CK gate's max_query_len==max_seq_len holds.
extra = dict(enforce_eager=False, compilation_config={"mode": 0, "cudagraph_mode": "FULL_DECODE_ONLY"},
             max_num_batched_tokens=max(MAXLEN, 8192),
             enable_prefix_caching=False, enable_chunked_prefill=False)
print(f"== CK-PREFILL | CK={CK} OPT={OPT} model={MODEL.split('/')[-1]} | free {free0/GIB:.1f}/{total/GIB:.1f} GiB maxlen={MAXLEN} util={GPUUTIL}")
t0 = time.perf_counter()
llm = LLM(model=MODEL, dtype=os.environ.get("G_DTYPE", "bfloat16"), attention_backend="TRITON_ATTN",
    tensor_parallel_size=1, gpu_memory_utilization=GPUUTIL, max_model_len=MAXLEN,
    kv_cache_dtype="auto", trust_remote_code=False, **extra)
print(f"engine init: {time.perf_counter()-t0:.1f}s")
free1, _ = torch.cuda.mem_get_info()
print(f"VRAM used after init: {(total-free1)/GIB:.2f} GiB | FREE {free1/GIB:.2f} GiB")

tok = llm.get_tokenizer()
UNIT = "Write a detailed technical essay about the history and future of GPU computing and parallel algorithms. "
def make_prompt(target_tok):
    p = UNIT
    while len(tok(p).input_ids) < target_tok:
        p += UNIT
    return p

sp1 = SamplingParams(temperature=0.0, max_tokens=1)   # TTFT ~= prefill latency
llm.generate([make_prompt(256)], sp1)                 # warm up kernels/autotune
torch.cuda.synchronize()

def ck_calls():
    try:
        import vllm_windows_rocm.cops as _C5
        return _C5._CK_CALLS[0]
    except Exception:
        return -1

print(f"tag       ptok   TTFT_ms(best/3)   CK_calls_delta")
TARGETS = tuple(int(x) for x in os.environ.get("G_TARGETS", "2048,4096,6144,8192,10240").split(","))
for target in TARGETS:
    if target > MAXLEN - 64:
        continue
    prompt = make_prompt(target)
    ptok = len(tok(prompt).input_ids)
    best = 1e9
    c0 = ck_calls()
    for _ in range(3):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        llm.generate([prompt], sp1)
        torch.cuda.synchronize()
        best = min(best, (time.perf_counter() - t0) * 1000.0)
    dc = ck_calls() - c0
    print(f"CK={int(CK)}      {ptok:5d}   {best:8.1f}          {dc}")

free2, _ = torch.cuda.mem_get_info()
print(f"total CK forward calls: {ck_calls()} | FREE VRAM now {free2/GIB:.2f} GiB")
co = llm.generate(["Question: What is the capital of France? One line."], SamplingParams(temperature=0.0, max_tokens=32))
print("COHERENCE:", repr(co[0].outputs[0].text[:180]))
print("PERF_CK_PREFILL_OK")
