# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Profile a Qwythos decode to find the bottleneck (CPU/eager-overhead vs GPU kernels).

Run from run/. Reports CPU vs CUDA(HIP) time split and the top ops by GPU time.
"""
import os
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# VLLM_PROFILE_COMPILE: 0 = eager (default), or CompilationMode 1/2/3 for inductor.
_COMPILE = int(os.environ.get("VLLM_PROFILE_COMPILE", "0"))
if not _COMPILE:
    os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("VLLM_BENCH_MODEL", "sahilchachra/Qwythos-9B-Claude-Mythos-5-1M-AWQ")
_kw = dict(model=MODEL, dtype="float16", attention_backend="TRITON_ATTN",
           tensor_parallel_size=1, gpu_memory_utilization=0.92, max_model_len=4096,
           trust_remote_code=True)
if _COMPILE:
    # inductor fusion, NO cudagraph (so the profiler sees the fused kernels individually).
    _kw["enforce_eager"] = False
    _kw["compilation_config"] = {"mode": _COMPILE, "cudagraph_mode": "NONE"}
else:
    _kw["enforce_eager"] = True
llm = LLM(**_kw)

prompts = ["Explain in detail how a modern GPU executes a matrix multiplication."]
llm.generate(prompts, SamplingParams(temperature=0, max_tokens=8))  # warmup
torch.cuda.synchronize()

# timed decode
t0 = time.perf_counter()
out = llm.generate(prompts, SamplingParams(temperature=0, max_tokens=64, ignore_eos=True))
torch.cuda.synchronize()
dt = time.perf_counter() - t0
gen = len(out[0].outputs[0].token_ids)
print(f"DECODE {gen} tok in {dt:.2f}s -> {gen/dt:.1f} tok/s")

acts = [ProfilerActivity.CPU]
try:
    acts.append(ProfilerActivity.CUDA)
except Exception:
    pass

with profile(activities=acts) as prof:
    llm.generate(prompts, SamplingParams(temperature=0, max_tokens=48, ignore_eos=True))
    torch.cuda.synchronize()

ka = prof.key_averages()
tot_cpu = sum(getattr(e, "self_cpu_time_total", 0) for e in ka)
tot_cuda = sum(getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0)) for e in ka)
print(f"=== total self CPU {tot_cpu/1000:.1f} ms | total self CUDA {tot_cuda/1000:.1f} ms ===")
try:
    print(ka.table(sort_by="self_cuda_time_total", row_limit=25))
except Exception:
    print(ka.table(sort_by="self_device_time_total", row_limit=25))
print("PROFILE_OK")
