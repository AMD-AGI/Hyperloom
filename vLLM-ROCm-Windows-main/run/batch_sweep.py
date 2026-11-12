# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Batch-scaling sweep: load Qwythos once (inductor mode=1 + FULL_DECODE_ONLY) and measure
aggregate + per-request decode tok/s at increasing concurrency. Shows whether vLLM's value is
throughput (high batch) vs single-stream latency (batch=1, where llama.cpp competes)."""
import os
import time

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# compile enabled -> do NOT disable dynamo

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("VLLM_BENCH_MODEL", "sahilchachra/Qwythos-9B-Claude-Mythos-5-1M-AWQ")
MAXTOK = int(os.environ.get("VLLM_BENCH_MAXTOK", "128"))
BATCHES = [int(x) for x in os.environ.get("VLLM_SWEEP_BATCHES", "1,2,4,8,16,32").split(",")]
GIB = 2 ** 30
_, total = torch.cuda.mem_get_info()

llm = LLM(model=MODEL, dtype="float16", attention_backend="TRITON_ATTN",
          tensor_parallel_size=1, gpu_memory_utilization=0.92, max_model_len=8192,
          trust_remote_code=True, enforce_eager=False,
          compilation_config={"mode": 1, "cudagraph_mode": "FULL_DECODE_ONLY"})

base = "Write a detailed essay about the history and future of GPU computing, part "
sp = SamplingParams(temperature=0.0, max_tokens=MAXTOK, ignore_eos=True)

# warm up all cudagraph batch sizes we will test
for b in BATCHES:
    llm.generate([base + str(i) for i in range(b)], SamplingParams(temperature=0, max_tokens=4))
torch.cuda.synchronize()

print("=" * 64)
print(f"BATCH SWEEP  model={MODEL}  decode_tok={MAXTOK}")
print(f"{'batch':>6} {'wall_s':>8} {'agg_tok/s':>10} {'tok/s/req':>10}")
results = []
for b in BATCHES:
    prompts = [base + str(i) for i in range(b)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = llm.generate(prompts, sp)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    gen = sum(len(o.outputs[0].token_ids) for o in out)
    agg = gen / dt
    print(f"{b:>6} {dt:>8.2f} {agg:>10.1f} {agg / b:>10.1f}")
    results.append((b, agg, agg / b))

free2, _ = torch.cuda.mem_get_info()
print(f"VRAM used at end: {(total - free2) / GIB:.2f} GiB")
print("SWEEP_OK")
