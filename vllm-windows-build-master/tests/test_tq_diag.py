"""Faulthandler-guarded diagnostic for Multi-TurboQuant methods.

Dumps a stack trace and exits if generate() stalls past 90 seconds, so
you can distinguish a real hang from a slow-but-terminating PyTorch
fallback decode.

Usage:
    set VLLM_MODEL_PATH=E:\\models\\Qwen3-14B-abliterated-AWQ-4bit
    set VLLM_PYTHON=E:\\vllm-windows-build\\venv\\Scripts\\python.exe
    set TQ_METHOD=isoquant3
    %VLLM_PYTHON% tests\\test_tq_diag.py

Set TQ_METHOD to one of: turboquant25, turboquant35, isoquant3,
isoquant4, planarquant3, planarquant4.
"""
import faulthandler
import os
import sys

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

CUDA_BIN = os.environ.get(
    'CUDA_BIN',
    r'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin',
)
if os.path.isdir(CUDA_BIN):
    os.add_dll_directory(CUDA_BIN)

TORCH_LIB = os.environ.get('TORCH_LIB')
if TORCH_LIB and os.path.isdir(TORCH_LIB):
    os.add_dll_directory(TORCH_LIB)

faulthandler.enable()
HANG_TIMEOUT = int(os.environ.get('TQ_HANG_TIMEOUT', '90'))
faulthandler.dump_traceback_later(HANG_TIMEOUT, repeat=False, exit=True)

TQ_METHOD = os.environ.get('TQ_METHOD', 'isoquant3')
MODEL = os.environ.get('VLLM_MODEL_PATH')
if not MODEL:
    sys.exit('Set VLLM_MODEL_PATH to your model directory')

METHOD_PRESETS = {
    'turboquant25': 'max_compression',
    'turboquant35': 'speed',
    'isoquant3': 'no_calibration_symmetric',
    'isoquant4': 'no_calibration_quality',
    'planarquant3': 'k_only_planar',
    'planarquant4': 'k_only_planar',
}
if TQ_METHOD not in METHOD_PRESETS:
    sys.exit(f'Unknown TQ_METHOD={TQ_METHOD!r}. Choices: {sorted(METHOD_PRESETS)}')

from multi_turboquant import get_preset
from multi_turboquant.integration import patch_vllm

preset_name = METHOD_PRESETS[TQ_METHOD]
patch_vllm(get_preset(preset_name))
print(f'[DIAG] patched for {TQ_METHOD} preset={preset_name}', flush=True)

from vllm import LLM, SamplingParams

llm = LLM(
    model=MODEL,
    dtype='float16',
    kv_cache_dtype=TQ_METHOD,
    max_model_len=512,
    gpu_memory_utilization=0.5,
    enforce_eager=True,
    trust_remote_code=True,
)
print(f'[DIAG] model loaded for {TQ_METHOD}', flush=True)

params = SamplingParams(temperature=0, max_tokens=5)
print(f'[DIAG] calling generate() for {TQ_METHOD}...', flush=True)
outputs = llm.generate(['Hello'], params)
print(f'[DIAG] generate returned!', flush=True)
for out in outputs:
    print(f'[DIAG] output: {out.outputs[0].text!r}', flush=True)
print(f'[DIAG] PASSED for {TQ_METHOD}')
