# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Phase-0+ kernel hardening: build & validate real HIP kernels on gfx1100.

Run with MSVC env active (see experiments/phase0_hip_ext/README.md), e.g.:
    cmd /c "\"..\\..\\tools\\winrocm_env.bat\" ^
        && python build_and_test.py"
"""
import os
import sys
import time

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D) and not os.path.isfile(os.path.join(_D, "tools", "winrocm_paths.py")):
    _D = os.path.dirname(_D)
sys.path.insert(0, os.path.join(_D, "tools"))
import winrocm_paths as wp

import torch
from torch.utils import cpp_extension

# Workaround for torch 2.10+rocm7.13 cpp_extension hipify None bug (see phase0_hip_ext/README.md).
from torch.utils.hipify import hipify_python as _hp

_orig_hipify = _hp.hipify


def _hipify_no_none(*a, **k):
    res = _orig_hipify(*a, **k)
    try:
        for key, v in res.items():
            if getattr(v, "hipified_path", None) is None:
                v.hipified_path = key
    except Exception as e:
        print("hipify-patch warning:", repr(e))
    return res


_hp.hipify = _hipify_no_none

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = wp.build_dir("vw_p0kbuild", clean=True)
DEVICE_LIB = wp.device_lib()

print("torch", torch.__version__, "hip", torch.version.hip)
sys.stdout.flush()

t0 = time.time()
ext = cpp_extension.load(
    name="phase0_kernels",
    sources=[os.path.join(HERE, "ops.cu"), os.path.join(HERE, "bindings.cu")],
    build_directory=BUILD_DIR,
    extra_cuda_cflags=[
        f"--rocm-device-lib-path={DEVICE_LIB}",
        # torch sets -D__HIP_NO_HALF_CONVERSIONS__=1 / -D__HIP_NO_HALF_OPERATORS__=1, which
        # strip the __half(float) ctor that rocWMMA's headers need. Re-enable (last flag wins).
        "-U__HIP_NO_HALF_CONVERSIONS__",
        "-U__HIP_NO_HALF_OPERATORS__",
    ],
    verbose=True,
)
print("BUILD_OK in", round(time.time() - t0, 1), "s")

# --- test RMSNorm ---
torch.manual_seed(0)
rows, hidden, eps = 32, 4096, 1e-6
x = torch.randn(rows, hidden, device="cuda", dtype=torch.float32)
w = torch.randn(hidden, device="cuda", dtype=torch.float32)
out = ext.rmsnorm(x, w, eps)
ref = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w
rms_err = (out - ref).abs().max().item()
print("RMSNORM_OK", bool(rms_err < 1e-3), "max_err", rms_err)

# --- test WMMA GEMM ---
M = N = K = 256
a = torch.randn(M, K, device="cuda", dtype=torch.float16)
b = torch.randn(K, N, device="cuda", dtype=torch.float16)
c = ext.wmma_gemm(a, b)
ref2 = a.float() @ b.float()
abs_err = (c - ref2).abs().max().item()
rel = abs_err / (ref2.abs().max().item() + 1e-6)
print("WMMA_GEMM_OK", bool(rel < 0.02), "max_abs_err", round(abs_err, 4), "rel", round(rel, 5))

# --- test INT8 WMMA GEMM (iu8 -> i32) ---
ai = torch.randint(-4, 5, (128, 128), device="cuda", dtype=torch.int8)
bi = torch.randint(-4, 5, (128, 128), device="cuda", dtype=torch.int8)
ci = ext.wmma_igemm(ai, bi)
refi = (ai.float() @ bi.float()).round().to(torch.int32)
idiff = (ci - refi).abs().max().item()
print("WMMA_IGEMM_OK", bool(idiff == 0), "max_abs_diff", idiff)
