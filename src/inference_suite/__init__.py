# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``inference_suite`` -- resolves a serving framework's base launch and runs one inference round on it.

It builds only on ``hyperloom_common`` and third-party libraries; ``.importlinter`` holds the
contracts that keep it so, and names the modules other packages may import.

A host resolves a session's base in two steps: :func:`fetch_sources` reads every source once, the only step that
reaches the network or runs a source's own code, and :func:`resolve_base` reads that :class:`SourceSnapshot` alone.
"""

from __future__ import annotations

from inference_suite.contracts.base_launch import LaunchQuery
from inference_suite.recipe.resolve import Miss, SourceSnapshot, resolve_base
from inference_suite.recipe.sources import fetch_sources

__all__ = ["LaunchQuery", "Miss", "SourceSnapshot", "fetch_sources", "resolve_base"]
