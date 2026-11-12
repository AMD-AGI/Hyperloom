# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Build vLLM's csrc/rocm/skinny_gemms.cu (LLMM1 + wvSplitK) into a native Windows torch.ops._rocm_C
extension for gfx1100 (RDNA3). Same proven recipe as build_c_ext.py: hipify csrc with torch's own
PYTORCH_MAP engine + ATen/c10 cuda->hip shim headers, then cpp_extension.load. skinny_gemms.cu already
has a __HIP__GFX1X__ (gfx11/12) code path (v_dot2_f32_f16 + mov_dpp reductions), so it compiles for
RDNA3; the gfx9 MFMA and gfx950/gfx12 fp8 paths are #if-compiled-out. Only LLMM1 + wvSplitK are bound
(win_rocm_bindings.cu). Enables vLLM's rocm_unquantized_gemm skinny path (VLLM_ROCM_USE_SKINNY_GEMM=1)
for the M=1 dense-MLP / attention-proj GEMVs.
"""
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
HIPDIR = wp.build_dir("vw_rocmc_hip")
BUILD_DIR = wp.build_dir("vw_rocmc_build", "VLLM_WIN_ROCM_C_DIR", clean=True)
DEVICE_LIB = wp.device_lib()

# reuse the same include-shim safety net as build_c_ext.py
SHIMS = {
    "ATen/cuda/CUDAContext.h": "#include <ATen/hip/HIPContext.h>\n"
                              "#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>\n",
    "ATen/cuda/Exceptions.h": "#include <ATen/hip/Exceptions.h>\n",
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

src = [os.path.join(HIPDIR, "rocm", "skinny_gemms.cu"),
       os.path.join(HERE, "win_rocm_bindings.cu")]
print("=== compiling vllm_win_rocm_C ===")
sys.stdout.flush()
t0 = time.perf_counter()
cpp_extension.load(
    name="vllm_win_rocm_C", sources=src,
    extra_include_paths=[SHIM, HIPDIR],
    build_directory=BUILD_DIR,
    extra_cuda_cflags=[
        f"--rocm-device-lib-path={DEVICE_LIB}",
        "-U__HIP_NO_HALF_CONVERSIONS__", "-U__HIP_NO_HALF_OPERATORS__",
        "-DTORCH_HIP_VERSION=0",
        "-DUSE_ROCM=1",
        f"-I{SHIM}", f"-I{HIPDIR}",
    ],
    extra_ldflags=["/LIBPATH:" + wp.hip_lib(), "hipblas.lib", "rocblas.lib", "amdhip64.lib"],
    verbose=True,
)
print("BUILD_OK in", round(time.perf_counter() - t0, 1), "s")

# ---- correctness + quick bench vs F.linear on the real dense-MLP / attn shapes ----
import torch.nn.functional as F
cu = 84
def _bench(fn, it=100, wu=30):
    for _ in range(wu): fn()
    torch.cuda.synchronize(); t = time.perf_counter()
    for _ in range(it): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t) / it
for (N, K) in [(4224, 2816), (2816, 2112), (8192, 2816), (2816, 4096)]:
    W = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1, K, device="cuda", dtype=torch.bfloat16)
    ref = F.linear(x, W)
    out = torch.ops._rocm_C.wvSplitK(W, x, None, cu)
    rel = (out.float() - ref.float()).abs().max().item() / (ref.float().abs().max().item() + 1e-6)
    t_w = _bench(lambda: torch.ops._rocm_C.wvSplitK(W, x, None, cu))
    t_l = _bench(lambda: F.linear(x, W))
    print(f"wvSplitK N={N} K={K}: rel={rel:.2e} | wvSplitK {t_w*1e6:6.1f}us | F.linear {t_l*1e6:6.1f}us | "
          f"speedup {t_l/t_w:.2f}x")
print("ROCM_C_OK")
