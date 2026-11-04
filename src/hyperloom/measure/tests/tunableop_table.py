# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A TunableOp results file and the name PyTorch reads it under, for the tests of a table loaded through a variable."""

from __future__ import annotations

from pathlib import Path

#: The variable PyTorch's TunableOp reads the results file from.
TUNABLEOP_ENV = "PYTORCH_TUNABLEOP_FILENAME"
#: Cards of a node: a test reads the file of device 0 up to the last of the cards it runs on.
NODE_CARDS = 8
#: The head of a TunableOp results file in the form PyTorch's TunableOp writes, for GEMMs at a decode batch size; the
#: versions, solution ids, shapes and times are made up.
TUNABLEOP_TABLE = (
    "Validator,PT_VERSION,2.0.0\n"
    "Validator,HIP_VERSION,700\n"
    "Validator,HIPBLASLT_VERSION,100000-0000000\n"
    "Validator,GCN_ARCH_NAME,gfx950:sramecc+:xnack-\n"
    "Validator,ROCBLAS_VERSION,5.0.0.0000000\n"
    "GemmTunableOp_BFloat16_TN,tn_1024_64_1024_ld_1024_1024_1024,Gemm_Hipblaslt_1,0.02\n"
    "GemmTunableOp_BFloat16_TN,tn_2048_64_1024_ld_1024_1024_2048,Gemm_Rocblas_2,0.01\n"
    "GemmTunableOp_BFloat16_TN,tn_4096_64_1024_ld_1024_1024_4096,Gemm_Rocblas_3,0.03\n"
).encode()


def tunableop_reads(value: str, ordinal: int) -> Path:
    """The file TunableOp reads for ``PYTORCH_TUNABLEOP_FILENAME=value`` on device ``ordinal``.

    PyTorch's ``TuningContext::SetFilename``: the ordinal replaces a ``%d``; without one it goes in before the last dot of
    the whole path, or at the end when there is none. On PyTorch 2.9.1, ``t.csv`` is read as ``t0.csv`` on
    device 0 and ``t1.csv`` on device 1, ``t`` as ``t0``, and ``d.x/t`` as ``d0.x/t``.
    """
    if "%d" in value:
        return Path(value.replace("%d", str(ordinal), 1))
    dot = value.rfind(".")
    return Path(f"{value}{ordinal}" if dot < 0 else f"{value[:dot]}{ordinal}{value[dot:]}")
