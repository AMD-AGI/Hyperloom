# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Phase-0 Gate A — build & import a native multi-file HIP .pyd on Windows gfx1100.

PASSED 2026-06-29 on RX 7900 XT: hipcc/clang 22 compiled kernel.cu + bindings.cu for
gfx1100-1103, MSVC link.exe linked against torch 2.10+rocm7.13, the .pyd imported and
the kernel ran with max_err 0.0.

Run from an environment where MSVC cl/link are on PATH (call vcvars64.bat first), e.g.:

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

# --- Workaround: torch 2.10+rocm7.13 cpp_extension hipify None bug -------------
# _jit_compile() builds: sources = [hipify_result[s].hipified_path if s in result else s]
# with NO None guard. hipify returns hipified_path=None for files it leaves unchanged,
# and that None then crashes object_file_path(). Replace None with the original path
# (a valid .cu; torch's _is_cuda_file accepts both .cu and .hip).
from torch.utils.hipify import hipify_python as _hp

_orig_hipify = _hp.hipify


def _hipify_no_none(*args, **kwargs):
    res = _orig_hipify(*args, **kwargs)
    try:
        for k, v in res.items():
            if getattr(v, "hipified_path", None) is None:
                v.hipified_path = k
    except Exception as e:  # pragma: no cover
        print("hipify-patch warning:", repr(e))
    return res


_hp.hipify = _hipify_no_none
# ------------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
BUILD_DIR = wp.build_dir("vw_p0build", clean=True)  # short path avoids Windows long-path build issues
DEVICE_LIB = wp.device_lib()  # HIP SDK device bitcode location

print("=== env ===")
print("torch", torch.__version__, "hip", torch.version.hip)
print("cpp_ext ROCM_HOME", cpp_extension.ROCM_HOME, "IS_HIP", cpp_extension.IS_HIP_EXTENSION)
sys.stdout.flush()

sources = [os.path.join(HERE, "kernel.cu"), os.path.join(HERE, "bindings.cu")]
print("=== building multi-file HIP extension:", [os.path.basename(s) for s in sources], "===")
sys.stdout.flush()

t0 = time.time()
ext = cpp_extension.load(
    name="phase0_hip_ext",
    sources=sources,
    build_directory=BUILD_DIR,
    extra_cuda_cflags=[f"--rocm-device-lib-path={DEVICE_LIB}"],
    verbose=True,
)
print("BUILD_OK in", round(time.time() - t0, 1), "s")

a = torch.randn(4096, device="cuda", dtype=torch.float32)
b = torch.randn(4096, device="cuda", dtype=torch.float32)
c = ext.hip_add(a, b)
torch.cuda.synchronize()
err = (c - (a + b)).abs().max().item()
print("HIP_EXT_RESULT_OK", bool(err < 1e-4), "max_err", err)
