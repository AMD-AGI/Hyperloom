# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Build paged_attention_flash.cu (self-contained flash-layout decode paged-attention, my own HIP -> no
csrc hipify needed) into vllm_win_attn_flash_C.pyd -> torch.ops._C.paged_attention_flash. Same HIP recipe
as build_attn_c.py."""
import os, sys, time, torch

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D) and not os.path.isfile(os.path.join(_D, "tools", "winrocm_paths.py")):
    _D = os.path.dirname(_D)
sys.path.insert(0, os.path.join(_D, "tools"))
import winrocm_paths as wp

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

HERE = os.path.dirname(os.path.abspath(__file__))
SHIM = os.path.join(HERE, "shim")
BUILD_DIR = wp.build_dir("vw_attnflash_build", "VLLM_WIN_FLASH_DIR", clean=True)
DEVICE_LIB = wp.device_lib()

# CUDA->HIP header shims (ATen/cuda/CUDAContext.h etc. pull in cuda_runtime_api.h which is absent on ROCm)
SHIMS = {
    "ATen/cuda/CUDAContext.h": "#include <ATen/hip/HIPContext.h>\n"
                              "#include <ATen/hip/impl/HIPStreamMasqueradingAsCUDA.h>\n",
    "c10/cuda/CUDAGuard.h": "#include <c10/hip/HIPGuard.h>\n"
                           "#include <ATen/hip/impl/HIPGuardImplMasqueradingAsCUDA.h>\n",
}
for rel, body in SHIMS.items():
    dst = os.path.join(SHIM, *rel.split("/"))
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    open(dst, "w", encoding="utf-8", newline="\n").write("#pragma once\n" + body)
print("torch", torch.__version__, "hip", torch.version.hip)
t0 = time.perf_counter()
cpp_extension.load(
    name="vllm_win_attn_flash_C",
    sources=[os.path.join(HERE, "paged_attention_flash.cu")],
    build_directory=BUILD_DIR,
    extra_include_paths=[SHIM],
    extra_cuda_cflags=[
        f"--rocm-device-lib-path={DEVICE_LIB}",
        "-U__HIP_NO_HALF_CONVERSIONS__", "-U__HIP_NO_HALF_OPERATORS__",
        "-DTORCH_HIP_VERSION=0", "-DUSE_ROCM=1", "-O3", f"-I{SHIM}",
    ],
    extra_ldflags=["/LIBPATH:" + wp.hip_lib(), "amdhip64.lib"],
    verbose=True,
)
print("BUILD_OK in", round(time.perf_counter() - t0, 1), "s")
print("has flash:", hasattr(torch.ops._C, "paged_attention_flash"))
print("ATTN_FLASH_OK")
