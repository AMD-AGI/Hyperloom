# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Build the HIP M=1 W4 GEMV and microbench it cache-cold vs the torch reference, at the 14B shapes.
Compares to the Triton kernel's cold bandwidth (o 192, qkv 263, down 279, gate 535 GB/s)."""
import os
import sys
import time

_D = os.path.dirname(os.path.abspath(__file__))
while _D != os.path.dirname(_D) and not os.path.isfile(os.path.join(_D, "tools", "winrocm_paths.py")):
    _D = os.path.dirname(_D)
sys.path.insert(0, os.path.join(_D, "tools"))
import winrocm_paths as wp

# short scratch dir for the JIT build (was set by build_run_hip.bat)
os.environ.setdefault("TORCH_EXTENSIONS_DIR", wp.build_dir("vw_hipgemv_build", "VLLM_WIN_HIPGEMV_DIR"))

import torch
from torch.utils import cpp_extension

# hipify on this build returns hipified_path=None for extra_files (Windows path bug) -> cpp_extension
# crashes in object_file_path. Guard: fall back to the original path. (Same fix as build_c_ext.py.)
from torch.utils.hipify import hipify_python as _hp
_orig_hipify = _hp.hipify
def _no_none(*a, **k):
    r = _orig_hipify(*a, **k)
    try:
        for key, v in r.items():
            if getattr(v, "hipified_path", None) is None:
                v.hipified_path = key
    except Exception:
        pass
    return r
_hp.hipify = _no_none

HERE = os.path.dirname(os.path.abspath(__file__))
DEVICE_LIB = wp.device_lib()
G = 128

mod = cpp_extension.load(
    name="gemv_w4_hip",
    sources=[os.path.join(HERE, "gemv_w4.cu")],
    extra_cuda_cflags=[
        f"--rocm-device-lib-path={DEVICE_LIB}",
        "-U__HIP_NO_HALF_CONVERSIONS__", "-U__HIP_NO_HALF_OPERATORS__",
        "-DUSE_ROCM=1", "-DTORCH_HIP_VERSION=0", "-O3",
    ],
    verbose=True,
)
print("HIP_BUILD_OK")


def ref(a, wq, s, z):
    K8, N = wq.shape
    K = K8 * 8
    sh = (torch.arange(8, device=wq.device, dtype=torch.int32) * 4)
    q = ((wq.unsqueeze(1) >> sh.view(1, 8, 1)) & 0xF).reshape(K, N).float()
    gidx = torch.arange(K, device=wq.device) // G
    return a.float() @ ((q - z.float()[gidx]) * s.float()[gidx])


SHAPES = {"o K5120 N5120": (5120, 5120), "qkv K5120 N7168": (5120, 7168),
          "down K13824 N5120": (13824, 5120), "gate K5120 N27648": (5120, 27648)}

print("HIP M=1 W4 GEMV, cache-cold (% of 800 GB/s DRAM):")
for name, (K, N) in SHAPES.items():
    wbytes = K * N * 0.5
    ncopies = max(3, int(160e6 // wbytes) + 1)
    a = torch.randn(1, K, device="cuda", dtype=torch.float16) * 0.1
    copies = [(torch.randint(0, 2**31 - 1, (K // 8, N), device="cuda", dtype=torch.int32),
               torch.randn(K // G, N, device="cuda", dtype=torch.float16) * 0.01,
               torch.randint(0, 16, (K // G, N), device="cuda", dtype=torch.uint8)) for _ in range(ncopies)]
    wq, s, z = copies[0]
    r = ref(a, wq, s, z)
    best = None
    for sg in (8, 16, 32, 64):
        try:
            c = torch.ops.vllm_win_hip.gemv_w4(a, wq, s, z, G, sg)
            rel = (c.float() - r).abs().max().item() / (r.abs().max().item() + 1e-6)
            if rel > 1e-2:
                best = best or ("WRONG", rel, sg)
                continue
            for i in range(ncopies):
                torch.ops.vllm_win_hip.gemv_w4(a, *copies[i % ncopies], G, sg)
            torch.cuda.synchronize()
            it = ncopies * 8
            t0 = time.perf_counter()
            for i in range(it):
                torch.ops.vllm_win_hip.gemv_w4(a, *copies[i % ncopies], G, sg)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / it
            gbps = wbytes / dt / 1e9
            if best is None or (isinstance(best[0], float) and gbps > best[0]):
                best = (gbps, dt, sg, rel)
        except Exception as e:
            best = best or ("ERR", str(e)[:50], sg)
    if best and isinstance(best[0], float):
        gbps, dt, sg, rel = best
        print(f"  {name:22} {dt*1e6:7.1f}us {gbps:6.1f} GB/s  sg={sg} rel={rel:.0e} ({gbps/800*100:.0f}%)")
    else:
        print(f"  {name:22} {best}")
print("HIP_GEMV_DONE")
