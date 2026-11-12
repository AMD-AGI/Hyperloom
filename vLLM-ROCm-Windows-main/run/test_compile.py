# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Ground-truth probe: enable vLLM's inductor compile (VLLM_COMPILE=3) and capture the exact
failure (expected: torch.distributed.tensor / DTensor import on USE_DISTRIBUTED=0).

Deliberately does NOT set TORCHDYNAMO_DISABLE, so dynamo+inductor run."""
import os
import traceback

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
# NOTE: TORCHDYNAMO_DISABLE intentionally NOT set.

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("VLLM_BENCH_MODEL", "sahilchachra/Qwythos-9B-Claude-Mythos-5-1M-AWQ")
MODE = int(os.environ.get("VLLM_COMPILE_MODE", "3"))  # 1=STOCK 2=TRACE_ONCE 3=VLLM_COMPILE
CGMODE = os.environ.get("VLLM_COMPILE_CGMODE", "NONE")  # cudagraph_mode
print(f"== compile test: mode={MODE} cudagraph={CGMODE} model={MODEL}")

try:
    llm = LLM(
        model=MODEL, dtype="float16", attention_backend="TRITON_ATTN",
        tensor_parallel_size=1, gpu_memory_utilization=0.92, max_model_len=4096,
        trust_remote_code=True, enforce_eager=False,
        kv_cache_dtype=os.environ.get("VLLM_COMPILE_KVDTYPE", "auto"),
        compilation_config={"mode": MODE, "cudagraph_mode": CGMODE},
    )
    p = "The capital of France is Paris. Explain in two sentences why it became the capital."
    out = llm.generate([p], SamplingParams(temperature=0, max_tokens=64))
    print("GEN:", repr(out[0].outputs[0].text[:300]))
    print("COMPILE_OK")
except Exception:
    print("==== COMPILE FAILED, full traceback ====")
    traceback.print_exc()
    print("COMPILE_FAIL")
