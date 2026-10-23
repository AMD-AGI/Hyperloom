# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lines of aiter's own tuned tables, copied as the image ships them (``aiter/configs``, a gfx950 image).

A table is its header and a few of its rows; an untuned sibling is only its header, which is all aiter reads of it.
"""

from __future__ import annotations

__all__ = [
    "A8W8_HEADER",
    "A8W8_ROWS",
    "A8W8_UNTUNED",
    "BF16_HEADER",
    "BF16_ROWS",
    "BF16_UNTUNED",
    "csv_text",
]

A8W8_UNTUNED = "M,N,K,q_dtype_w\n"
A8W8_HEADER = "gfx,cu_num,M,N,K,q_dtype_w,libtype,kernelId,splitK,us,kernelName,tflops,bw,errRatio"
#: One shape, 128 x 1536 x 5120 int8, tuned on two gfx942 chips that differ in ``cu_num``, and two others of gfx950.
A8W8_ROWS = (
    "gfx942,80,128,1536,5120,torch.int8,asm,3,3,14.8921,_ZN5aiter41I8gemm_bf16_perTokenI8_BpreShuffle_64x128E,135.19,598.5,0.1017",
    "gfx942,304,128,1536,5120,torch.int8,asm,0,1,17.9602,_ZN5aiter41I8gemm_bf16_perTokenI8_BpreShuffle_16x128E,112.1,496.26,0.0",
    "gfx942,304,64,1536,5120,torch.int8,asm,0,1,16.7759,_ZN5aiter41I8gemm_bf16_perTokenI8_BpreShuffle_16x128E,60.0,500.04,0.0",
    "gfx950,256,1024,1280,8192,torch.float8_e4m3fn,cktile,216,0,27.0331,a8w8_bpreshuffle_cktile_0x0x8x4x1x0x0x0x0x3_48x64x256_1x4x1_16x16x128_default,794.39,795.17,0.0",
)

BF16_UNTUNED = "M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle\n"
BF16_HEADER = (
    "gfx,cu_num,M,N,K,bias,dtype,outdtype,scaleAB,bpreshuffle,libtype,solidx,splitK,us,kernelName,err_ratio,tflops,bw"
)
BF16_ROWS = (
    "gfx950,256,64,256,5120,False,torch.bfloat16,torch.float32,False,False,asm,5,13,7.9191,_ZN5aiter39bf16gemm_fp32bf16_tn_64x64_splitk_cleanE,0.0,21.19,417.92",
    "gfx950,256,80,256,5120,False,torch.bfloat16,torch.float32,False,False,asm,7,13,7.9676,_ZN5aiter39bf16gemm_fp32bf16_tn_80x64_splitk_cleanE,0.0,26.32,436.97",
    "gfx950,256,128,256,5120,False,torch.bfloat16,torch.float32,False,False,asm,5,10,9.0259,_ZN5aiter39bf16gemm_fp32bf16_tn_64x64_splitk_cleanE,0.0,37.18,442.91",
    "gfx950,256,192,256,5120,False,torch.bfloat16,torch.float32,False,False,asm,8,10,9.8147,_ZN5aiter39bf16gemm_fp32bf16_tn_96x64_splitk_cleanE,0.0,51.28,477.43",
)


def csv_text(header: str, *rows: str) -> str:
    return "".join(f"{line}\n" for line in (header, *rows))
