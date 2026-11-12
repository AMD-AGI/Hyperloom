# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: 2026 CLVwp contributors
"""KV-cache dtype bench: decode speed and accuracy across context lengths.

Compares kv_cache_dtype configs (fp16 "auto", "fp8", fp8 with calculated
scales) on the same prompts at several context lengths, measuring:

  - prefill time (max_tokens=1 call)
  - decode speed over a fixed token budget (prefix caching disabled, the
    decode rate is (max_tokens-1) / (total - prefill) from a second call)
  - output drift vs a saved fp16 reference (token agreement, first divergence)

Results are appended as JSON lines to VLLM_KV_OUT (one line per
(ctx_len, dtype) pair) so separate invocations can be aggregated later.

Environment:
  VLLM_KV_MODEL     model (default Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4)
  VLLM_KV_DTYPE     kv_cache_dtype for this run (default auto)
  VLLM_KV_SCALES    1 -> pass calculate_kv_scales=True (calibrated fp8)
  VLLM_KV_CTXS      comma list of context targets (default 128,2048,8192,16384)
  VLLM_KV_MAXTOK    decode tokens per measurement (default 128)
  VLLM_KV_BACKEND   attention backend (default TRITON_ATTN)
  VLLM_KV_UTIL      gpu_memory_utilization (default 0.6)
  VLLM_KV_TRUST     1 -> trust_remote_code=True (default 0)
  VLLM_KV_BLOCK_SIZE block_size for the LLM (e.g. 128 for KVarN; default: vLLM's)
  VLLM_KV_OUT       JSONL output path (default <temp dir>/kv_results.jsonl)
  VLLM_KV_REF       fp16 reference JSON written by the auto run and read by
                    later runs for the drift check (default <temp dir>/kv_ref.json)
  VLLM_KV_SAVE_REF  1 -> write VLLM_KV_REF from this run's outputs
"""

import gc
import json
import math
import os
import tempfile
import time

# In-process engine, like every other run/ script: the plugin's config-time hooks (the
# static KV-scale override among them) are applied in the process that builds the
# VllmConfig, and a spawned engine-core child would start without them.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, SamplingParams

MODEL = os.environ.get("VLLM_KV_MODEL", "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4")
DTYPE = os.environ.get("VLLM_KV_DTYPE", "auto")
SCALES = os.environ.get("VLLM_KV_SCALES", "0") == "1"
CTXS = [int(c) for c in os.environ.get("VLLM_KV_CTXS", "128,2048,8192,16384").split(",")]
MAXTOK = int(os.environ.get("VLLM_KV_MAXTOK", "128"))
BACKEND = os.environ.get("VLLM_KV_BACKEND", "TRITON_ATTN")
OUT = os.environ.get("VLLM_KV_OUT", os.path.join(tempfile.gettempdir(), "kv_results.jsonl"))
REF = os.environ.get("VLLM_KV_REF", os.path.join(tempfile.gettempdir(), "kv_ref.json"))
UTIL = float(os.environ.get("VLLM_KV_UTIL", "0.6"))
TRUST = os.environ.get("VLLM_KV_TRUST", "0") == "1"
SAVE_REF = os.environ.get("VLLM_KV_SAVE_REF", "0") == "1"

GIB = 1024 ** 3

FILLER = (
    "The history of GPU computing begins in the late 1990s when fixed function "
    "graphics pipelines gave way to programmable shaders. Researchers noticed "
    "that vertex and pixel shaders could be tricked into running general "
    "arithmetic, and the GPGPU movement was born. In 2006 CUDA exposed a C "
    "language interface to the chip, and deep learning arrived a decade later "
    "to consume every flop the industry could produce. Matrix multiply units, "
    "wavelevel scheduling, high bandwidth memory and quantized arithmetic now "
    "dominate accelerator design. "
)
QUESTION = "\n\nSummarize the key developments mentioned above in one sentence."


def build_prompt(tokenizer, target_tokens: int) -> str:
    """Repeat the filler until the prompt reaches roughly target_tokens."""
    reps = max(1, math.ceil(target_tokens / 85)) + 1
    text = FILLER * reps + QUESTION
    ids = tokenizer.encode(text)
    if len(ids) > target_tokens:
        head = tokenizer.decode(ids[: max(0, target_tokens - 16)])
        text = head + QUESTION
    return text


def measure(llm, tokenizer, ctx: int) -> dict:
    prompt = build_prompt(tokenizer, ctx)
    n_in = len(tokenizer.encode(prompt))

    # Two prefill-only runs, keep the faster: the first call at a new length can still
    # include Triton JIT/autotune work, and decode below is derived by subtracting it.
    prefill = float("inf")
    for _ in range(2):
        t0 = time.perf_counter()
        llm.generate([prompt], SamplingParams(max_tokens=1, ignore_eos=True))
        prefill = min(prefill, time.perf_counter() - t0)

    t0 = time.perf_counter()
    out = llm.generate([prompt], SamplingParams(max_tokens=MAXTOK, ignore_eos=True))[0]
    total_t = time.perf_counter() - t0
    decode_t = total_t - prefill
    n_out = len(out.outputs[0].token_ids)
    tok_s = (n_out - 1) / decode_t if decode_t > 0 else float("nan")

    return {
        "ctx_target": ctx, "ctx_actual": n_in, "dtype": DTYPE, "scales": SCALES,
        "prefill_s": round(prefill, 3), "total_s": round(total_t, 3),
        "decode_tok_s": round(tok_s, 1), "n_out": n_out,
        "text": out.outputs[0].text,
    }


def main() -> None:
    import torch

    free0, total = torch.cuda.mem_get_info()
    print(f"== KV bench | dtype={DTYPE} scales={SCALES} backend={BACKEND} "
          f"| GPU {total/GIB:.1f} GiB, free {free0/GIB:.1f} GiB")
    print(f"== ctxs={CTXS} maxtok={MAXTOK}")

    kwargs = dict(
        model=MODEL, dtype="float16", attention_backend=BACKEND,
        tensor_parallel_size=1, gpu_memory_utilization=UTIL,
        max_model_len=max(CTXS) + 256, kv_cache_dtype=DTYPE,
        enable_prefix_caching=False, enforce_eager=True,
        trust_remote_code=TRUST,
    )
    if SCALES:
        kwargs["calculate_kv_scales"] = True
    if os.environ.get("VLLM_KV_NOCHUNK", "0") == "1":
        kwargs["enable_chunked_prefill"] = False
        # vLLM refuses max_num_batched_tokens < max_model_len without chunking;
        # a single-shot prefill needs the whole prompt in one batch.
        kwargs["max_num_batched_tokens"] = kwargs["max_model_len"]
    # KVarN's fp16 tail pool is sized from max_num_batched_tokens (pool_slots:
    # 2*max_num_seqs + prefill_blocks + 8 per layer) and is not counted by
    # gpu_memory_utilization: the 8192 default OOMs a 16 GiB card on 7-9B
    # models before the first long-context prefill completes.
    batched = os.environ.get("VLLM_KV_BATCHED_TOKENS")
    if batched:
        kwargs["max_num_batched_tokens"] = int(batched)
    seqs = os.environ.get("VLLM_KV_MAX_SEQS")
    if seqs:
        kwargs["max_num_seqs"] = int(seqs)
    blk = os.environ.get("VLLM_KV_BLOCK_SIZE")
    if blk:
        kwargs["block_size"] = int(blk)
    llm = LLM(**kwargs)
    tok = llm.get_tokenizer()
    # Warm-up: JIT/autotune every kernel once on a short prompt so the first measured
    # length does not pay the compile time (it showed up as a bogus decode rate).
    llm.generate([FILLER + QUESTION], SamplingParams(max_tokens=4, ignore_eos=True))

    rows = [measure(llm, tok, c) for c in CTXS]

    ref = {}
    if os.path.exists(REF):
        with open(REF, encoding="utf-8") as f:
            ref = json.load(f)

    print("-" * 60)
    for r in rows:
        line = (f"ctx={r['ctx_actual']:>6} prefill={r['prefill_s']:>6.3f}s "
                f"decode={r['decode_tok_s']:>6.1f} tok/s")
        key = str(r["ctx_actual"])
        if key in ref:
            rtxt, ttxt = ref[key], r["text"]
            a, b = tok.encode(rtxt), tok.encode(ttxt)
            same = sum(1 for x, y in zip(a, b) if x == y)
            first = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y),
                         min(len(a), len(b)))
            agree = same / max(len(a), len(b)) if a or b else 1.0
            r["agree_frac"] = round(agree, 3)
            r["first_divergence"] = first
            line += f" | agree={agree:.0%} first_diff_tok={first}"
        print(line)
        with open(OUT, "a", encoding="utf-8") as f:
            f.write(json.dumps(r) + "\n")

    if SAVE_REF:
        with open(REF, "w", encoding="utf-8") as f:
            json.dump({str(r["ctx_actual"]): r["text"] for r in rows}, f)
        print(f"reference saved -> {REF}")

    free1, _ = torch.cuda.mem_get_info()
    print(f"VRAM used at end: {(total-free1)/GIB:.2f} GiB")
    print("KV_BENCH_OK")


if __name__ == "__main__":
    main()
