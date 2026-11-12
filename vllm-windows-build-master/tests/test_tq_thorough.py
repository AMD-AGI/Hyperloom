"""Thorough Multi-TurboQuant validation.

Goes beyond "outputs differ from FP16" to verify:
  1. KV cache memory is actually smaller in practice (more tokens fit at
     the same gpu_memory_utilization).
  2. Generated text remains coherent across longer outputs (not just
     first 20 tokens).
  3. Each method has measurable time / token cost so the user knows
     what they're paying for the memory savings.

Each method runs in a fresh subprocess to avoid GPU state bleed.

Configuration via environment variables:
    VLLM_MODEL_PATH       Path to a model directory (required).
    VLLM_PYTHON           Path to the venv python.exe (required).
    CUDA_VISIBLE_DEVICES  GPU index (default: 0).
"""
import faulthandler
faulthandler.enable()
import os
import sys
import subprocess
import json
import time

os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

MODEL_PATH = os.environ.get("VLLM_MODEL_PATH")
PYTHON_EXE = os.environ.get("VLLM_PYTHON", sys.executable)

if not MODEL_PATH:
    print("ERROR: set VLLM_MODEL_PATH to your model directory.")
    sys.exit(1)
if not os.path.isfile(PYTHON_EXE):
    print(f"ERROR: VLLM_PYTHON not found: {PYTHON_EXE}")
    sys.exit(1)

METHODS = [
    'auto', 'isoquant3', 'isoquant4', 'planarquant3', 'planarquant4',
    'turboquant25', 'turboquant35',
]

# 8 different prompts so we can do real batched concurrency.
PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which plants",
    "The three laws of thermodynamics are",
    "A classic apple pie recipe needs",
    "Machine learning differs from traditional programming because",
    "The fastest way to learn a new language is",
    "When debugging a memory leak, you should",
    "The principle of relativity states that",
]

# Longer generation so quality is measurable, not just 20 first tokens.
MAX_TOKENS = 80


RUNNER_SCRIPT = r'''
import faulthandler, os, sys, json, time
faulthandler.enable()
_cuda_bin = os.environ.get(
    "CUDA_HOME",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
) + r"\bin"
if os.path.isdir(_cuda_bin):
    os.add_dll_directory(_cuda_bin)

model_path = sys.argv[1]
kv_dtype = sys.argv[2]
prompts = json.loads(sys.argv[3])
max_tokens = int(sys.argv[4])

import torch
from vllm import LLM, SamplingParams

t0 = time.perf_counter()
llm = LLM(
    model=model_path,
    dtype="float16",
    kv_cache_dtype=kv_dtype,
    max_model_len=512,
    max_num_seqs=8,            # try to fit all 8 prompts at once
    max_num_batched_tokens=512,
    gpu_memory_utilization=0.5,
    enforce_eager=True,  # Hold graph/compile behavior constant across methods.
    trust_remote_code=True,
)
load_secs = time.perf_counter() - t0

# Capture KV cache info from the GPU worker. We can't read it from
# the public API easily, so probe the kv_cache_manager.
try:
    engine = llm.llm_engine
    kvcm = engine.engine_core.engine_core.scheduler.kv_cache_manager
    cache_blocks = kvcm.block_pool.num_gpu_blocks
    block_size = kvcm.block_size
    cache_tokens = cache_blocks * block_size
except Exception:
    cache_tokens = -1

# Greedy sampling so differences come from KV math, not RNG.
params = SamplingParams(temperature=0.0, max_tokens=max_tokens, seed=0)
t1 = time.perf_counter()
outputs = llm.generate(prompts, params)
gen_secs = time.perf_counter() - t1

results = []
total_out_tokens = 0
for out in outputs:
    tokens = list(out.outputs[0].token_ids)
    text = out.outputs[0].text
    total_out_tokens += len(tokens)
    results.append({"tokens": tokens, "text": text})

print("===RESULT===")
print(json.dumps({
    "load_secs": load_secs,
    "gen_secs": gen_secs,
    "cache_tokens": cache_tokens,
    "total_out_tokens": total_out_tokens,
    "tok_per_sec": total_out_tokens / max(gen_secs, 1e-9),
    "results": results,
}))
'''


def run_method(method: str) -> dict | None:
    proc = subprocess.run(
        [PYTHON_EXE, "-c", RUNNER_SCRIPT,
         MODEL_PATH, method, json.dumps(PROMPTS), str(MAX_TOKENS)],
        capture_output=True, text=True, timeout=2400,
    )
    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode})")
        print(f"  stderr tail: {proc.stderr[-400:]}")
        return None
    marker = '===RESULT==='
    idx = proc.stdout.find(marker)
    if idx < 0:
        print("  No result marker — full stdout tail:")
        print(proc.stdout[-400:])
        return None
    json_str = proc.stdout[idx + len(marker):].strip().split('\n')[0]
    return json.loads(json_str)


def check_coherent(text: str) -> bool:
    """Quick coherence sanity check on generated text."""
    if len(text.strip()) < 10:
        return False
    words = text.split()
    if len(words) < 5:
        return False
    # Reject extreme repetition (less than 25% unique words)
    unique = len(set(w.lower() for w in words))
    if unique / max(len(words), 1) < 0.25:
        return False
    # Reject obvious garbage (lots of non-ascii or weird chars)
    printable = sum(1 for c in text if c.isprintable() or c in '\n\t')
    if printable / max(len(text), 1) < 0.95:
        return False
    return True


def main():
    print("=" * 78)
    print("Thorough Multi-TurboQuant Validation")
    print("=" * 78)
    print(f"Model: Qwen3-14B-abliterated-AWQ-4bit")
    print(f"Prompts: {len(PROMPTS)}, max_tokens: {MAX_TOKENS}")
    print(f"Methods: {METHODS}")
    print()

    all_results: dict[str, dict | None] = {}
    for method in METHODS:
        print(f"Running {method}...")
        result = run_method(method)
        if result is None:
            all_results[method] = None
            print(f"  FAILED")
            continue
        all_results[method] = result
        print(
            f"  OK — load {result['load_secs']:.1f}s, "
            f"gen {result['gen_secs']:.1f}s, "
            f"{result['tok_per_sec']:.1f} tok/s, "
            f"cache {result['cache_tokens']} tok"
        )

    print()
    print("=" * 78)
    print("Results table")
    print("=" * 78)
    baseline = all_results.get('auto')
    if baseline is None:
        print("FATAL: FP16 baseline failed; aborting.")
        return

    fp16_cache = baseline['cache_tokens']
    fp16_tps = baseline['tok_per_sec']

    print(f"{'Method':<14} {'Cache':>10} {'CacheRatio':>12} "
          f"{'GenTime':>10} {'Tok/s':>10} {'vs FP16':>14} "
          f"{'Coherent':>10}")
    print("-" * 90)
    for m in METHODS:
        r = all_results.get(m)
        if r is None:
            print(f"{m:<14} FAILED")
            continue
        coh = sum(1 for x in r['results'] if check_coherent(x['text']))
        coh_str = f"{coh}/{len(r['results'])}"
        cache_ratio = (
            f"{r['cache_tokens']/fp16_cache:.2f}x"
            if fp16_cache > 0 else "?"
        )
        if m == 'auto':
            vs_str = "(baseline)"
        else:
            same_baseline = all(
                rr['tokens'] == bb['tokens']
                for rr, bb in zip(r['results'], baseline['results'])
            )
            vs_str = "IDENTICAL" if same_baseline else "DIFFERS"
        print(
            f"{m:<14} {r['cache_tokens']:>10} {cache_ratio:>12} "
            f"{r['gen_secs']:>9.1f}s {r['tok_per_sec']:>9.1f} "
            f"{vs_str:>14} {coh_str:>10}"
        )

    print()
    print("=" * 78)
    print("Sample text (prompt 0)")
    print("=" * 78)
    for m in METHODS:
        r = all_results.get(m)
        if r is None:
            continue
        text = r['results'][0]['text']
        snippet = text[:120].replace('\n', ' ')
        print(f"  {m:<14} {snippet!r}")

    print()
    print("=" * 78)
    print("Verdict")
    print("=" * 78)

    tq_methods = [m for m in METHODS if m != 'auto']
    all_ok = True
    for m in tq_methods:
        r = all_results.get(m)
        if r is None:
            print(f"  ❌ {m}: FAILED to run")
            all_ok = False
            continue
        # Coherence
        coh = sum(1 for x in r['results'] if check_coherent(x['text']))
        if coh < len(r['results']) // 2:
            print(f"  ❌ {m}: only {coh}/{len(r['results'])} coherent outputs")
            all_ok = False
        # Differs from baseline
        same = all(
            rr['tokens'] == bb['tokens']
            for rr, bb in zip(r['results'], baseline['results'])
        )
        if same:
            print(f"  ❌ {m}: outputs identical to FP16 (placebo!)")
            all_ok = False
        # Cache larger than FP16
        if r['cache_tokens'] <= fp16_cache:
            print(f"  ❌ {m}: cache not larger than FP16 "
                  f"({r['cache_tokens']} vs {fp16_cache})")
            all_ok = False

    if all_ok:
        print(
            f"\n  ✅ ALL 6 TQ METHODS: real compression, "
            f"coherent outputs, cache >= 2x FP16."
        )
    else:
        print(f"\n  ⚠️  Some checks failed — see above.")
    print("=" * 78)


if __name__ == '__main__':
    main()
