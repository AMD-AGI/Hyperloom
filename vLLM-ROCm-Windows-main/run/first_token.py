# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Phase-1 milestone: first token from vLLM on native Windows + ROCm (gfx1100).

Kept in run/ (NOT the project root) so sys.path[0] doesn't shadow the editable `vllm`
package with the cloned `vllm/` source directory.

Eager mode, TORCH_SDPA attention, single GPU, no custom kernels (empty build) — the goal is
correct token generation, not speed.
"""
import os

# Must be set before importing vllm.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # run engine in-process
os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")  # no _rocm_C wvSplitK/LLMM1 kernels -> torch.linear
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")  # AITER is CDNA-only / unavailable
os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")

import torch  # noqa: E402

print("torch", torch.__version__, "| cuda_avail", torch.cuda.is_available(),
      "| dev", torch.cuda.get_device_name(0))

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("VLLM_TEST_MODEL", "facebook/opt-125m")
print("Loading model:", MODEL)

llm = LLM(
    model=MODEL,
    enforce_eager=True,
    dtype="float16",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.5,
    max_model_len=512,
    attention_backend="TRITON_ATTN",
)

prompts = ["Hello, my name is"]
out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=16))
print("=" * 60)
for o in out:
    print("PROMPT:", repr(o.prompt))
    print("OUTPUT:", repr(o.outputs[0].text))
print("=" * 60)
print("FIRST_TOKEN_OK")
