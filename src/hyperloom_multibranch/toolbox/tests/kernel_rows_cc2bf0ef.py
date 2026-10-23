# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Rows of the ``a8w8_blockscale_bpreshuffle`` tables of the no-KB run ``cc2bf0ef`` (Qwen3-14B-FP8), copied as the
branches tuned them (``MB-blobs``), and the install's own declaration of the variable and its untuned sibling.

The run's integration table (the head, 74 rows) was merged with four branch tables and every merge was refused by
shape. Here each table is kept to a few of its rows, in the groups the run's tables fall into:

- ``SHARED``: rows the head, ``8d915e2b`` (b09) and ``d3d78d03`` (b06) all hold, identical.
- ``B09_NEW``: rows only b09 tuned. b09 is the head and these, so it holds every row of the head.
- ``B01_*``: ``a5398873`` (b01, b07). Two of its shapes the head also tuned, to other rows (``B01_SHARED`` against
  ``B01_HEAD``); ``B01_NEW`` are shapes only it tuned.
- ``B06_*``: ``d3d78d03`` (b06). One shape the head also tuned, with another kernel name (``B06_SHARED`` against
  ``B06_HEAD``); ``B06_NEW`` are shapes only it tuned.
"""

from __future__ import annotations

from pathlib import Path

__all__ = [
    "B01_HEAD",
    "B01_NEW",
    "B01_SHARED",
    "B06_HEAD",
    "B06_NEW",
    "B06_SHARED",
    "B09_NEW",
    "CORE",
    "ENV",
    "HEADER",
    "SHARED",
    "UNTUNED",
    "csv_text",
    "install",
]

ENV = "AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE"
#: Lines 160-170 and 328-340 of the image's ``aiter/jit/core.py``: where the variable names its default and its tuned name.
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
    "gfx950,256,16,512,7168,asm,6,7,7.9691,_ZN5aiter42fp8gemm_bf16_blockscale_BpreShuffle_32x128E,14.74,476.98,0.0029",
    "gfx950,256,20480,512,7168,ck,13,2,111.424,a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,1349.12,1538.65,0.0",
    "gfx950,256,20480,576,7168,ck,15,2,164.23,a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,1029.74,1062.67,0.0089",
    "gfx950,256,128,4096,1280,ck,6,3,6.4459,a8w8_blockscale_bpreshuffle_1x128x128_256x16x128x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_8_1x2_intrawave_v1,208.22,1001.46,0.0",
    "gfx950,256,20480,7168,256,ck,16,0,104.9379,a8w8_blockscale_bpreshuffle_1x128x128_256x64x128x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1,716.25,2865.31,0.0",
)
B09_NEW = (
    "gfx950,256,3072,5120,5120,ck,13,0,100.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
    "gfx950,256,6144,5120,17408,ck,13,0,100.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
    "gfx950,256,10240,5120,5120,ck,13,0,100.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
    "gfx950,256,12288,5120,17408,ck,13,0,100.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
)
B01_HEAD = (
    "gfx950,256,64,34816,5120,ck,17,0,20.0,a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1,0,0,0.0",
    "gfx950,256,64,5120,17408,ck,7,0,20.0,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,0,0,0.0",
)
B01_SHARED = (
    "gfx950,256,64,34816,5120,ck,17,0,49.1618,a8w8_blockscale_bpreshuffle_1x128x128_256x64x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1,464.12,3723.26,0.0",
    "gfx950,256,64,5120,17408,ck,12,0,39.1064,a8w8_blockscale_bpreshuffle_1x128x128_256x32x64x256_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1,291.73,2324.39,0.0",
)
B01_NEW = (
    "gfx950,256,1,34816,5120,ck,5,0,36.6625,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x128_8x16_16x16_16x16x1_8x32x1_1x16x1x16_4_1x1_intrawave_v1,9.72,4864.17,0.0",
    "gfx950,256,16,34816,5120,ck,7,0,38.1202,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,149.64,4707.58,0.0",
    "gfx950,256,1,5120,17408,ck,7,0,38.4946,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,4.63,2316.08,0.0",
    "gfx950,256,16,5120,17408,ck,7,0,37.3518,a8w8_blockscale_bpreshuffle_1x128x128_256x16x64x256_16x16_16x16_16x16x1_16x16x1_1x16x1x16_4_1x1_intrawave_v1,76.36,2398.05,0.0",
)
B06_HEAD = (
    "gfx950,256,20480,7168,2304,ck,13,2,395.8895,a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_8x32x1_8x32x1_1x32x1x8_8_2x1_intrawave_v1,1708.7,902.53,0.0",
)
B06_SHARED = (
    "gfx950,256,20480,7168,2304,ck,13,2,395.8895,a8w8_blockscale_bpreshuffle_1x128x128_256x64x256x128_16x16_16x16_16x16x1_16x16x1_1x32x1x8_8_2x1_intrawave_v1,1708.7,902.53,0.0",
)
B06_NEW = (
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
