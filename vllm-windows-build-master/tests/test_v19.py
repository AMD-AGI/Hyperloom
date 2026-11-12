"""Smoke test: vLLM 0.25.1 on Windows with normal FP16 KV cache.

Configuration via environment variables (set before running):
    VLLM_MODEL_PATH       Path to a Qwen3 / Llama / similar model.
                          Default: None (you must set it).
    CUDA_VISIBLE_DEVICES  GPU index to pin to (default: 0).
    VLLM_ENFORCE_EAGER    Set to 1 only for graph/compile troubleshooting.

Example:
    set VLLM_MODEL_PATH=E:\\models\\Qwen3-14B-AWQ-4bit
    python tests\\test_v19.py
"""
import faulthandler
faulthandler.enable()
import os
import sys

# Set sane defaults for Windows
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

# Add CUDA DLL search path. Edit if your CUDA install lives elsewhere.
_CUDA_BIN = os.environ.get(
    "CUDA_HOME",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
) + r"\bin"
if os.path.isdir(_CUDA_BIN):
    os.add_dll_directory(_CUDA_BIN)

MODEL = os.environ.get("VLLM_MODEL_PATH")
ENFORCE_EAGER = os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1"
if not MODEL:
    print("ERROR: set VLLM_MODEL_PATH to your model directory.")
    sys.exit(1)

print("=" * 60)
print("vLLM 0.25.1 Windows smoke test (FP16 KV cache)")
print("=" * 60)
print(f"Model: {MODEL}")
print()

from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL,
    dtype="float16",
    kv_cache_dtype="auto",
    max_model_len=int(os.environ.get("VLLM_MAX_MODEL_LEN", "2048")),
    gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.5")),
    enforce_eager=ENFORCE_EAGER,
    trust_remote_code=True,
)

print("\nModel loaded successfully")

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=32,
    seed=0,
)

prompts = [
    "The capital of France is",
    "Write a haiku about programming:",
]

outputs = llm.generate(prompts, sampling_params)

print("\n" + "=" * 60)
print("Results:")
print("=" * 60)
for output in outputs:
    print(f"\nPrompt: {output.prompt!r}")
    print(f"Generated: {output.outputs[0].text!r}")

print("\n" + "=" * 60)
print("TEST PASSED")
print("=" * 60)
