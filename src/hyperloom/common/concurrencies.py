# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ``C1,C2,...`` concurrency list the single-lineage and multi-branch commands take, parsed one way. Stdlib-only."""

from __future__ import annotations

import argparse


def parse_concurrencies(value: str) -> tuple[int, ...]:
    """``C1,C2,...`` as the distinct concurrencies it names, ascending; each must be positive. An ``argparse`` type."""
    concs = tuple(sorted({int(item) for item in value.split(",")}))
    if concs[0] < 1:
        raise argparse.ArgumentTypeError(f"concurrencies must be positive, got {value}")
    return concs


__all__ = ["parse_concurrencies"]
