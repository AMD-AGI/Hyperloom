# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Tier-A precision gate: greedy-decode a fixed prompt set and save the token sequences, so two
runs (e.g. fp16 KV vs fp8 KV, or before/after a speed change) can be compared for drift.

Run twice with different VLLM_PC_TAG / VLLM_BENCH_KVDTYPE, then compare the saved JSONs.
"""
import json
import os
import sys

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D) and not os.path.isfile(os.path.join(_D, "tools", "winrocm_paths.py")):
    _D = os.path.dirname(_D)
sys.path.insert(0, os.path.join(_D, "tools"))
import winrocm_paths as wp  # noqa: E402

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ROCM_USE_SKINNY_GEMM", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("VLLM_BENCH_MODEL", "sahilchachra/Qwythos-9B-Claude-Mythos-5-1M-AWQ")
KVDTYPE = os.environ.get("VLLM_BENCH_KVDTYPE", "auto")
TAG = os.environ.get("VLLM_PC_TAG", KVDTYPE)
OUT = os.environ.get("VLLM_PC_OUT") or os.path.join(wp.repo_root(), "run", "precision_out")
os.makedirs(OUT, exist_ok=True)

PROMPTS = [
    "Explain how a transformer attention layer works.",
    "Write a Python function that returns the nth Fibonacci number.",
    "The three primary colors are",
    "Summarize the causes of the First World War in two sentences.",
    "Translate to French: The weather is nice today and I want to go for a walk.",
    "List the planets of the solar system in order from the sun.",
    "What is the derivative of x^3 + 2x with respect to x?",
    "Describe the plot of Romeo and Juliet in three sentences.",
    "Give step-by-step instructions to make a cup of tea.",
    "Why is the sky blue? Answer concisely.",
]

llm = LLM(model=MODEL, dtype="float16", attention_backend="TRITON_ATTN",
          tensor_parallel_size=1, gpu_memory_utilization=0.92, max_model_len=4096,
          kv_cache_dtype=KVDTYPE, trust_remote_code=True, enforce_eager=False,
          compilation_config={"mode": 1, "cudagraph_mode": "FULL_DECODE_ONLY"})

sp = SamplingParams(temperature=0.0, max_tokens=48)
outs = llm.generate(PROMPTS, sp)
data = {"tag": TAG, "kv": KVDTYPE, "seqs": [list(o.outputs[0].token_ids) for o in outs]}
path = os.path.join(OUT, f"pc_{TAG}.json")
with open(path, "w") as f:
    json.dump(data, f)
print("WROTE", path)

# If a golden (auto) run exists, compare.
golden = os.path.join(OUT, "pc_auto.json")
if os.path.exists(golden) and TAG != "auto":
    g = json.load(open(golden))["seqs"]
    tot = match = first_div = 0
    diverged = 0
    for gi, ci in zip(g, data["seqs"]):
        n = min(len(gi), len(ci))
        for k in range(n):
            tot += 1
            if gi[k] == ci[k]:
                match += 1
        if gi[:n] != ci[:n]:
            diverged += 1
    print(f"TOKEN_AGREEMENT {match}/{tot} = {100*match/max(tot,1):.1f}% | prompts_diverged {diverged}/{len(g)}")
print("PC_OK")
