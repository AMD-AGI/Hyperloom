###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Source resolution for the analysis routes.

TraceLens owns path-finding: :func:`resolve_source_verdict` is the one call the
routes make, wrapping TraceLens' ``resolve_kernel_source`` and returning its
``ResolveResult`` straight through. TraceLens also owns the native-vs-Triton
routing decision; HL just forwards the symbol and its launcher and lets
TraceLens classify.

TraceLens' ``kernel_source`` is an independent path-identifier (source path
mapping only, not TraceLens' analysis layer). Importing this module imports
TraceLens: an importable TraceLens is a hard requirement here.
"""

from __future__ import annotations

from TraceLens.TraceUtils.kernel_source import ResolveResult, resolve_kernel_source

__all__ = ["ResolveResult", "resolve_source_verdict"]


def resolve_source_verdict(
    symbol: str,
    *,
    kernel_file: str = "",
    op_name: str = "",
) -> ResolveResult:
    """Resolve one device symbol to its source via TraceLens.

    HL forwards the symbol and its launcher; TraceLens owns the native-vs-Triton
    routing decision (it falls back to native when a ``.py`` launcher has no real
    ``@triton.jit``/``@gluon.jit`` def).
    """
    return resolve_kernel_source(symbol, kernel_file=kernel_file, op_name=op_name)
