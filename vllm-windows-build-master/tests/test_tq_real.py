"""REAL Multi-TurboQuant validation.

Runs FP16 baseline + all 6 TQ methods with deterministic sampling and
verifies each TQ method produces output DIFFERENT from FP16 (proving the
quantization noise is actually entering the pipeline) and DIFFERENT from
the other methods (proving each has its own numerical signature).

Configuration via environment variables:
    VLLM_MODEL_PATH       Path to a model directory (required).
    VLLM_PYTHON           Path to the venv python.exe (required for
                          subprocess isolation between methods).
    CUDA_VISIBLE_DEVICES  GPU index (default: 0).

Example:
    set VLLM_MODEL_PATH=E:\\models\\Qwen3-14B-AWQ-4bit
    set VLLM_PYTHON=E:\\vllm-windows-build\\venv\\Scripts\\python.exe
    python tests\\test_tq_real.py
"""
import faulthandler
faulthandler.enable()
import os
import sys
import subprocess
import json

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
    "auto", "isoquant3", "isoquant4", "planarquant3", "planarquant4",
    "turboquant25", "turboquant35",
]

# Deterministic sampling so output diffs reflect KV math, not RNG.
PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which plants",
]

RUNNER_SCRIPT = r'''
import faulthandler, os, sys, json
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

from vllm import LLM, SamplingParams
llm = LLM(
    model=model_path,
    dtype="float16",
    kv_cache_dtype=kv_dtype,
    max_model_len=512,
    max_num_seqs=4,
    max_num_batched_tokens=512,
    gpu_memory_utilization=0.5,
    enforce_eager=True,  # Hold graph/compile behavior constant across methods.
    trust_remote_code=True,
)

params = SamplingParams(temperature=0.0, max_tokens=20, seed=0)
outputs = llm.generate(prompts, params)

results = []
for out in outputs:
    results.append({
        "tokens": list(out.outputs[0].token_ids),
        "text": out.outputs[0].text,
    })

print("===RESULT===")
print(json.dumps(results))
'''


def run_method(method: str) -> list[dict] | None:
    proc = subprocess.run(
        [PYTHON_EXE, "-c", RUNNER_SCRIPT,
         MODEL_PATH, method, json.dumps(PROMPTS)],
        capture_output=True, text=True, timeout=1800,
    )
    if proc.returncode != 0:
        print(f"  FAILED (exit {proc.returncode})")
        print(f"  stderr tail: {proc.stderr[-500:]}")
        return None
    marker = "===RESULT==="
    idx = proc.stdout.find(marker)
    if idx < 0:
        print("  No result marker found")
        print(f"  stdout tail: {proc.stdout[-500:]}")
        return None
    json_str = proc.stdout[idx + len(marker):].strip().split("\n")[0]
    return json.loads(json_str)


def main():
    print("=" * 70)
    print("Real Multi-TurboQuant Validation")
    print("=" * 70)
    print(f"Model:   {MODEL_PATH}")
    print(f"Python:  {PYTHON_EXE}")
    print(f"Prompts: {len(PROMPTS)}")
    print(f"Methods: {METHODS}")
    print()

    all_results: dict[str, list[dict] | None] = {}
    for method in METHODS:
        print(f"Running {method}...")
        result = run_method(method)
        all_results[method] = result
        if result is None:
            print("  FAILED")
        else:
            print(f"  OK ({len(result)} outputs)")

    print()
    print("=" * 70)
    print("Analysis")
    print("=" * 70)

    baseline = all_results["auto"]
    if baseline is None:
        print("FATAL: baseline failed")
        return

    print("\nBaseline (auto/fp16) outputs:")
    for i, r in enumerate(baseline):
        print(f"  [{i}] {r['text'][:70]!r}")

    print()
    print(f"{'Method':<15} {'Differs from FP16':<22} {'Differs from others'}")
    print("-" * 70)

    tq_methods = [m for m in METHODS if m != "auto"]
    for method in tq_methods:
        result = all_results[method]
        if result is None:
            print(f"{method:<15} FAILED")
            continue

        same_as_fp16 = all(
            r["tokens"] == b["tokens"]
            for r, b in zip(result, baseline)
        )
        differs_fp16 = "DIFFERS" if not same_as_fp16 else "IDENTICAL (placebo!)"

        unique_among_tq = True
        for other in tq_methods:
            if other == method or all_results[other] is None:
                continue
            all_same = all(
                r["tokens"] == o["tokens"]
                for r, o in zip(result, all_results[other])
            )
            if all_same:
                unique_among_tq = False
                break
        differs_other = "UNIQUE" if unique_among_tq else "matches another TQ"
        print(f"{method:<15} {differs_fp16:<22} {differs_other}")

    print()
    print("Sample outputs (first prompt):")
    for method in METHODS:
        r = all_results[method]
        if r is None:
            print(f"  {method:<15} FAILED")
        else:
            print(f"  {method:<15} {r[0]['text'][:65]!r}")

    print()
    print("=" * 70)
    tq_ok = [all_results[m] is not None for m in tq_methods]
    tq_differ = [
        all_results[m] is not None and
        not all(r["tokens"] == b["tokens"]
                for r, b in zip(all_results[m], baseline))
        for m in tq_methods
    ]
    if all(tq_ok) and all(tq_differ):
        print("RESULT: All TQ methods run AND produce output different from FP16.")
        print("        Compression math is affecting inference.")
    elif all(tq_ok):
        print("RESULT: All TQ methods run but some produce IDENTICAL output to FP16.")
        print("        Those methods are acting as placebos.")
    else:
        print("RESULT: Some TQ methods failed to run.")
    print("=" * 70)


if __name__ == "__main__":
    main()
