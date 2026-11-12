# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Compile-spike: build csrc/attention/{paged_attention_v1.cu,paged_attention_v2.cu} (generic wave32
paged attention) into torch.ops._C.paged_attention_v1/v2 on native Windows ROCm (gfx1100). Same recipe
as build_c_ext.py. Goal here is: does it COMPILE for gfx1100? block_size 128 + sliding-window mask come
next if it does. head_size 256 is already instantiated."""
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
HIPDIR = wp.build_dir("vw_attn_hip")
BUILD_DIR = wp.build_dir("vw_attn_build", "VLLM_WIN_ATTN_DIR", clean=True)
DEVICE_LIB = wp.device_lib()

SHIMS = {
    "ATen/cuda/CUDAContext.h": "#include <ATen/hip/HIPContext.h>\n"
                              "#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>\n",
    "ATen/cuda/Exceptions.h": "#include <ATen/hip/Exceptions.h>\n",
    "c10/cuda/CUDAGuard.h": "#include <c10/hip/HIPGuard.h>\n"
                           "#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>\n",
    "c10/cuda/CUDAStream.h": "#include <c10/hip/HIPStream.h>\n"
                            "#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>\n",
    "c10/cuda/CUDAException.h": "#include <c10/hip/HIPException.h>\n",
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

src = [os.path.join(HIPDIR, "attention", "paged_attention_v1.cu"),
       os.path.join(HIPDIR, "attention", "paged_attention_v2.cu"),
       os.path.join(HERE, "win_attn_bindings.cu")]
print("=== compiling vllm_win_attn_C ===")
sys.stdout.flush()
t0 = time.perf_counter()
cpp_extension.load(
    name="vllm_win_attn_C", sources=src,
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
print("has v1:", hasattr(torch.ops._C, "paged_attention_v1"),
      "| has v2:", hasattr(torch.ops._C, "paged_attention_v2"))
print("ATTN_C_OK")
