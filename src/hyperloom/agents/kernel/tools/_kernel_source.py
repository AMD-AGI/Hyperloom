###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Source resolution shared by the compute and bypass analysis paths.

TraceLens owns path-finding: :func:`resolve_source_verdict` is the one call both
routes make, wrapping TraceLens' ``resolve_kernel_source`` and returning its
``ResolveResult`` straight through. Only the Triton-vs-native routing decision
is HL's, and it lives here so both callers route identically.
"""

from __future__ import annotations

from TraceLens.TraceUtils.kernel_source import ResolveResult, resolve_kernel_source

__all__ = ["ResolveResult", "resolve_source_verdict"]

#: Match "triton" in the symbol, library, or launcher path, not the ``.py`` suffix: native kernels dispatch through ``.py`` too.
_TRITON_MARKER = "triton"


def resolve_source_verdict(
    symbol: str,
    *,
    kernel_file: str = "",
    op_name: str = "",
    library: str = "",
) -> ResolveResult:
    """Resolve one device symbol to its source via TraceLens.

    A genuine Triton kernel takes the Triton route with its launcher; a native
    kernel passes just the symbol so TraceLens' native gate can classify
    Tensile/CK/MIOpen instead of the Triton path accepting the ``.py`` dispatcher.
    """
    is_triton = (
        _TRITON_MARKER in symbol.lower() or _TRITON_MARKER in library.lower() or _TRITON_MARKER in kernel_file.lower()
    )
    return resolve_kernel_source(
        symbol,
        kernel_file=kernel_file if is_triton else "",
        is_triton=is_triton,
        op_name=op_name,
    )
