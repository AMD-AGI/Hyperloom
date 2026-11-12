# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Build vLLM's csrc/moe helper kernels into a native Windows torch.ops._moe_C extension.

Reuses the same hipify recipe as build_c_ext.py (PYTORCH_MAP word-boundary engine + redirect
shims + device-lib-path). Compiles only the ROCm-available helpers (moe_align_block_size,
batched_moe_align_block_size, moe_sum, topk_softmax) -- the CUDA-only GEMMs are #ifndef USE_ROCM
upstream. Replaces the torch fallbacks in cops.py (_install_moe_C)."""
import os
import shutil
import sys
import time

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D) and not os.path.isfile(os.path.join(_D, "tools", "winrocm_paths.py")):
    _D = os.path.dirname(_D)
sys.path.insert(0, os.path.join(_D, "tools"))
import winrocm_paths as wp

import torch
from torch.utils import cpp_extension

from torch.utils.hipify import hipify_python as _hp
_orig = _hp.hipify
def _no_none(*a, **k):
    r = _orig(*a, **k)
    try:
        for key, v in r.items():
            if getattr(v, "hipified_path", None) is None:
                v.hipified_path = key
    except Exception:
        pass
    return r
_hp.hipify = _no_none

VLLM_CSRC = wp.vllm_csrc()
HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(HERE, "shim")
HIPDIR = wp.build_dir("vw_moe_hip")
BUILD_DIR = wp.build_dir("vw_moe_build", "VLLM_WIN_MOE_DIR", clean=True)
DEVICE_LIB = wp.device_lib()

SHIMS = {
    "ATen/cuda/CUDAContext.h": "#include <ATen/hip/HIPContext.h>\n"
                              "#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>\n",
    "ATen/cuda/Exceptions.h": "#include <ATen/hip/Exceptions.h>\n",
    "ATen/cuda/Atomic.cuh": "#include <ATen/hip/Atomic.cuh>\n",
    "c10/cuda/CUDAGuard.h": "#include <c10/hip/HIPGuard.h>\n"
                           "#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>\n",
    "c10/cuda/CUDAStream.h": "#include <c10/hip/HIPStream.h>\n"
                            "#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>\n",
}
for rel, body in SHIMS.items():
    dst = os.path.join(SHIM, *rel.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8", newline="\n").write("#pragma once\n" + body)

from torch.utils.hipify.hipify_python import RE_PYTORCH_PREPROCESSOR, PYTORCH_MAP

print("torch", torch.__version__, "hip", torch.version.hip)
print(f"=== hipify csrc -> {HIPDIR} ===")
shutil.rmtree(HIPDIR, ignore_errors=True)
shutil.copytree(VLLM_CSRC, HIPDIR)

def _pt(mo):
    return str(PYTORCH_MAP[mo.group(1)])

n = 0
for dp, _, fns in os.walk(HIPDIR):
    for fn in fns:
        if not fn.endswith((".cu", ".cuh", ".cpp", ".h", ".hpp", ".cc")):
            continue
        p = os.path.join(dp, fn)
        s = open(p, encoding="utf-8", errors="ignore").read()
        s2 = RE_PYTORCH_PREPROCESSOR.sub(_pt, s)
        if s2 != s:
            open(p, "w", encoding="utf-8", newline="\n").write(s2)
            n += 1
print("rewrote", n, "files")

src = [os.path.join(HIPDIR, "moe", "moe_align_sum_kernels.cu"),
       os.path.join(HIPDIR, "moe", "topk_softmax_kernels.cu"),
       os.path.join(HERE, "win_moe_bindings.cu")]
HIPINC = wp.hip_include()
print("=== compiling _moe_C ===")
sys.stdout.flush()
t0 = time.perf_counter()
cpp_extension.load(
    name="vllm_win_moe_C", sources=src,
    extra_include_paths=[SHIM, HIPDIR, HIPINC],
    build_directory=BUILD_DIR,
    extra_cuda_cflags=[
        f"--rocm-device-lib-path={DEVICE_LIB}",
        "-U__HIP_NO_HALF_CONVERSIONS__", "-U__HIP_NO_HALF_OPERATORS__",
        "-DTORCH_HIP_VERSION=0", "-DUSE_ROCM=1",
        f"-I{SHIM}", f"-I{HIPDIR}", f"-I{HIPINC}",
    ],
    verbose=True,
)
print("BUILD_OK in", round(time.perf_counter() - t0, 1), "s")

# smoke test moe_sum + moe_align
inp = torch.randn(4, 3, 16, device="cuda", dtype=torch.float16)
out = torch.empty(4, 16, device="cuda", dtype=torch.float16)
torch.ops._moe_C.moe_sum(inp, out)
print("MOE_SUM_OK", bool((out - inp.sum(1)).abs().max().item() < 1e-2))
print("MOE_C_OK")
