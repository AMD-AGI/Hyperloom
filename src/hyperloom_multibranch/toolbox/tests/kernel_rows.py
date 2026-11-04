# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rows of ``a8w8_blockscale_bpreshuffle`` tables as several branches might tune them, and the install's own declaration
of the variable and its untuned sibling.

The rows are synthetic: GEMM shapes of public model configurations, aiter's own kernel instance names, and made-up
times and error ratios, with the ``tflops`` and ``bw`` columns left at 0. An integration table (the head) is merged with
the branches' tables. Here each table is kept to a few rows, in the groups such tables fall into:

- ``SHARED``: rows the head, alpha and gamma all hold, identical.
- ``ALPHA_NEW``: rows only alpha tuned. alpha is the head and these, so it holds every row of the head.
- ``BETA_*``: beta's. Two of its shapes the head also tuned, to other rows, timed slower than the head's
  (``BETA_SHARED`` against ``BETA_HEAD``); ``BETA_NEW`` are shapes only it tuned.
- ``GAMMA_*``: gamma's. One shape the head also tuned, with another kernel name (``GAMMA_SHARED`` against
  ``GAMMA_HEAD``); ``GAMMA_NEW`` are shapes only it tuned.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "ALPHA_NEW",
    "BETA_HEAD",
    "BETA_NEW",
    "BETA_SHARED",
    "CORE",
    "ENV",
    "GAMMA_HEAD",
    "GAMMA_NEW",
    "GAMMA_SHARED",
    "HEADER",
    "SHARED",
    "UNTUNED",
    "csv_text",
    "install",
]

ENV = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"
#: The parts of aiter's ``aiter/jit/core.py`` that matter here: where the variable names its default and its tuned name.
CORE = f"""
import os

AITER_ROOT_DIR = "/unused"
{ENV} = os.getenv(
    "{ENV}",
    f"{{AITER_ROOT_DIR}}/aiter/configs/a8w8_blockscale_bpreshuffle_tuned_gemm.csv",
)


class AITER_CONFIG:
    @property
    def {ENV}_FILE(self):
        return self.get_config_file(
            "{ENV}",
            {ENV},
            "a8w8_blockscale_bpreshuffle_tuned_gemm",
        )
"""
#: The install's ``a8w8_blockscale_bpreshuffle_untuned_gemm.csv``: the header is all aiter reads of it.
UNTUNED = "M,N,K\n"
HEADER = "gfx,cu_num,M,N,K,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio"
SHARED = (
    "gfx950,256,16,512,7168,asm,6,7,8.0,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E,0,0,0.003",
    "gfx950,256,20480,512,7168,ck,13,2,110.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
    "gfx950,256,20480,576,7168,ck,15,2,160.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.009",
    "gfx950,256,128,4096,1280,ck,6,3,6.5,a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1,0,0,0.0",
    "gfx950,256,20480,7168,256,ck,16,0,105.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
)
ALPHA_NEW = (
    "gfx950,256,3072,5120,5120,ck,13,0,100.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
    "gfx950,256,6144,5120,17408,ck,13,0,100.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
    "gfx950,256,10240,5120,5120,ck,13,0,100.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
    "gfx950,256,12288,5120,17408,ck,13,0,100.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
)
BETA_HEAD = (
    "gfx950,256,64,34816,5120,ck,17,0,20.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
    "gfx950,256,64,5120,17408,ck,7,0,20.0,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,0,0,0.0",
)
BETA_SHARED = (
    "gfx950,256,64,34816,5120,ck,17,0,49.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
    "gfx950,256,64,5120,17408,ck,12,0,39.0,a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
)
BETA_NEW = (
    "gfx950,256,1,34816,5120,ck,5,0,36.0,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1,0,0,0.0",
    "gfx950,256,16,34816,5120,ck,7,0,38.0,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,0,0,0.0",
    "gfx950,256,1,5120,17408,ck,7,0,38.5,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,0,0,0.0",
    "gfx950,256,16,5120,17408,ck,7,0,37.5,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,0,0,0.0",
)
GAMMA_HEAD = (
    "gfx950,256,20480,7168,2304,ck,13,2,400.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
)
GAMMA_SHARED = (
    "gfx950,256,20480,7168,2304,ck,13,2,400.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
)
GAMMA_NEW = (
    "gfx950,256,1024,5120,5120,ck,14,0,100.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
    "gfx950,256,1024,34816,5120,ck,14,0,100.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
    "gfx950,256,2048,5120,17408,ck,14,0,100.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
)


def csv_text(*groups: tuple[str, ...]) -> bytes:
    """A table of the rows in ``groups``, in the order given, as a branch would put it."""
    return "".join(f"{line}\n" for line in (HEADER, *(row for group in groups for row in group))).encode()


def install(root: Path) -> Path:
    """A stand-in for the image's aiter in ``root``: the ``configs`` directory (returned), ``jit/core.py`` beside it and
    ``csrc`` (where the tuner scripts are) around the package."""
    configs = root / "aiter" / "configs"
    (root / "aiter" / "jit").mkdir(parents=True)
    (root / "csrc").mkdir()
    configs.mkdir(parents=True)
    (root / "aiter" / "jit" / "core.py").write_text(CORE, encoding="utf-8")
    (configs / "a8w8_blockscale_bpreshuffle_untuned_gemm.csv").write_text(UNTUNED, encoding="utf-8")
    return configs
