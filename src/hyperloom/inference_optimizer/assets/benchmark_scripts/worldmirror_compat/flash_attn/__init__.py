# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""flash-attn compatible surface backed by torch SDPA.

ROCm ships no flash-attn wheel, but HY-World-2.0 imports it unconditionally on
the bf16/fp16 attention path. This package satisfies that import and routes the
call to scaled_dot_product_attention, which on ROCm already dispatches to the
aiter/CK fused attention kernels.

Only reachable when the real package is absent: worldmirror_<runner>.sh probes
for flash_attn first and appends this directory to PYTHONPATH on a miss.
"""

from .flash_attn_interface import flash_attn_func, flash_attn_qkvpacked_func

__all__ = ["flash_attn_func", "flash_attn_qkvpacked_func"]
__version__ = "0.0.0+hyperloom.sdpa-shim"
