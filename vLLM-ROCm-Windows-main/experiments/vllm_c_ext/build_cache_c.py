# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Build csrc/cache_kernels.cu (reshape_and_cache_flash) into torch.ops._C_cache_ops on native Windows
ROCm (gfx1100). Same proven recipe as build_c_ext.py. Only reshape_and_cache_flash is bound; the rest of
the file (MLA/cp_gather/swap_blocks) compiles but is unbound."""
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
HIPDIR = wp.build_dir("vw_cache_hip")
BUILD_DIR = wp.build_dir("vw_cache_build", "VLLM_WIN_CACHE_DIR", clean=True)
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

# The fp8 convert/gather/upconvert host fns after reshape_and_cache_flash use
# vllm::fp8::scaled_vec_conversion<bf16_8_t> which the AMD fp8 quant_utils path lacks. We only bind
# reshape_and_cache_flash. TRUNCATE the file at the next top-level `namespace vllm {` after it (the
# convert_fp8 block) -- a clean boundary that keeps includes + kernels + macros + reshape_and_cache(_flash)
# and drops convert_fp8/gather/cp_gather/indexer/concat_mla_q. (A mid-namespace #if 0 unbalances braces.)
_ck = os.path.join(HIPDIR, "cache_kernels.cu")
_s = open(_ck, encoding="utf-8").read()
_i = _s.find("void reshape_and_cache_flash(")
_j = _s.find("namespace vllm {", _i)
if _i != -1 and _j != -1:
    open(_ck, "w", encoding="utf-8", newline="\n").write(_s[:_j])
    print(f"truncated cache_kernels.cu at offset {_j} (dropped convert_fp8/gather/fp8-upconvert)")

src = [os.path.join(HIPDIR, "cache_kernels.cu"),
       os.path.join(HERE, "win_cache_bindings.cu")]
print("=== compiling vllm_win_cache_C ===")
sys.stdout.flush()
t0 = time.perf_counter()
cpp_extension.load(
    name="vllm_win_cache_C", sources=src,
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

# ---- correctness: native reshape_and_cache_flash vs a torch reference ----
import torch
NT, NH, HD, BS, NB = 6, 8, 128, 16, 32
key = torch.randn(NT, NH, HD, device="cuda", dtype=torch.float16)
val = torch.randn(NT, NH, HD, device="cuda", dtype=torch.float16)
# flash layout: key_cache [num_blocks, block_size, num_heads, head_size]
kc = torch.zeros(NB, BS, NH, HD, device="cuda", dtype=torch.float16)
vc = torch.zeros_like(kc)
slot = torch.arange(NT, device="cuda", dtype=torch.int64) + 3  # arbitrary slots
one = torch.tensor(1.0, device="cuda", dtype=torch.float32)
torch.ops._C_cache_ops.reshape_and_cache_flash(key, val, kc, vc, slot, "auto", one, one)
ok = True
for t in range(NT):
    b, o = int(slot[t]) // BS, int(slot[t]) % BS
    if not torch.allclose(kc[b, o], key[t], atol=1e-3) or not torch.allclose(vc[b, o], val[t], atol=1e-3):
        ok = False
print("RESHAPE_CACHE_OK", ok)
print("CACHE_C_OK")
