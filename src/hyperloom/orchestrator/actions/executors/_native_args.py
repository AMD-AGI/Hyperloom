# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compose argv tokens for the native launch contract without shell rewriting."""

from __future__ import annotations

import shlex
from collections.abc import Sequence


def compose_native_args(inherited: Sequence[str], extra: str, *, remove: Sequence[str], replace: bool) -> list[str]:
    """Retain JSON/space operands and apply removals to the intended layer."""
    proposed = shlex.split(extra)
    before = [] if replace else list(inherited)
    if replace:
        before, proposed = proposed, []
    kept: list[str] = []
    skipping = False
    for token in before:
        if token.startswith("--"):
            skipping = token.partition("=")[0] in remove
        if not skipping:
            kept.append(token)
    return kept + proposed
