# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 ThePie88 (https://github.com/ThePie88/vLLM-ROCm-Windows)
"""Build ck_fmha_C.pyd (torch.ops-free pybind): ck_fmha_fwd(q,k,v,o,scale,causal) wrapping the CK ck_tile
FMHA d128-fp16 causal + non-causal instances for gfx1100. All sources compiled as .cu (hipcc) so the CK
device headers + the kernel instances build. Recipe = build_attn_flash_c.py + CK flags + the memcpy patch."""
import os, shutil, sys, time, torch
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

CK = wp.ck_root()
FMHA = os.path.join(CK, "example", "ck_tile", "01_fmha")
GEN = wp.ck_gen_dir()
HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "build_src")
BUILD_DIR = wp.build_dir("vw_ckfmha_build", "VLLM_WIN_CKFMHA_DIR", clean=True)
os.makedirs(SRC, exist_ok=True)

INST_C = "fmha_fwd_d128_fp16_batch_b128x64x32x128x32x128_r8x1x1_r8x1x1_w16x16x16_w16x16x16_o6_qr_hpad_vr_psskddv_nlogits_nbias_mask_nlse_ndropout_nskip_nqscale_ntrload_nsink_gfx11.cpp"
INST_NC = "fmha_fwd_d128_fp16_batch_b128x64x32x128x32x128_r8x1x1_r8x1x1_w16x16x16_w16x16x16_o6_qr_hpad_vr_psskddv_nlogits_nbias_nmask_nlse_ndropout_nskip_nqscale_ntrload_nsink_gfx11.cpp"
# all sources as .cu so cpp_extension compiles them with hipcc (device + CK headers)
shutil.copy(os.path.join(HERE, "ck_fmha_ext.cpp"), os.path.join(SRC, "ck_fmha_ext.cu"))
shutil.copy(os.path.join(GEN, INST_C), os.path.join(SRC, "inst_c.cu"))
shutil.copy(os.path.join(GEN, INST_NC), os.path.join(SRC, "inst_nc.cu"))
sources = [os.path.join(SRC, f) for f in ("ck_fmha_ext.cu", "inst_c.cu", "inst_nc.cu")]

print("torch", torch.__version__, "hip", torch.version.hip)
t0 = time.perf_counter()
cpp_extension.load(
    name="ck_fmha_C", sources=sources, build_directory=BUILD_DIR,
    extra_include_paths=[os.path.join(CK, "include"), FMHA],
    extra_cuda_cflags=[
        f"--offload-arch={wp.offload_arch()}", "-std=c++17", "-O3",
        f"-I{os.path.join(CK, 'include')}", f"-I{FMHA}",
        "--rocm-path=" + wp.hip_root(), "--rocm-device-lib-path=" + wp.device_lib(),
        "-Wno-undefined-func-template", "-Wno-float-equal",
        "-DCK_TILE_FMHA_FWD_FAST_EXP2=1", "-fgpu-flush-denormals-to-zero",
        "-DCK_TILE_FMHA_FWD_SPLITKV_API=0", "-DCK_TILE_FMHA_FWD_APPENDKV_API=0",
        "-DCK_TILE_FMHA_FWD_PAGEDKV_API=0", "-DCK_TILE_FMHA_FWD_BATCH_PREFILL_API=0",
    ],
    extra_ldflags=["/LIBPATH:" + wp.hip_lib(), "amdhip64.lib"],
    verbose=True,
)
print("BUILD_OK in", round(time.perf_counter() - t0, 1), "s")
import ck_fmha_C
print("has ck_fmha_fwd:", hasattr(ck_fmha_C, "ck_fmha_fwd"))
print("CK_FMHA_EXT_OK")
