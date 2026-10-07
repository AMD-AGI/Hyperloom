###############################################################################
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# See LICENSE for license information.
###############################################################################

"""Source resolution shared by the compute and bypass analysis paths.

TraceLens owns path-finding: :func:`resolve_source_verdict` is the one call both
routes make, wrapping TraceLens' ``resolve_kernel_source`` and returning its
``ResolveResult`` straight through. TraceLens also owns the native-vs-Triton
routing decision; HL just forwards the symbol and its launcher and lets
TraceLens classify.

TraceLens' ``kernel_source`` is an independent path-identifier (source path
mapping only, not TraceLens' analysis layer). Both routes resolve source through
it, bypass included, so importing this module imports TraceLens: an importable
TraceLens is a hard requirement here. The bypass analysis layer is otherwise
TraceLens-free (it reads raw Kineto and builds its own rows) and leans on
TraceLens solely for this source path mapping.
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
