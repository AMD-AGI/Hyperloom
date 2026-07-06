# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Hyperloom — unified ``hyperloom.*`` namespace (tree-reform scaffold).

This package is the target single namespace for the tree-reform migration
(see ``tree-reform.MD``). During the transition it exists side-by-side with the
legacy top-level packages (``inference_optimizer``, ``robustness_agent`` …);
modules are moved in here incrementally, one migration step at a time.

Phase P2.0 intentionally ships an empty scaffold: no behavior lives here yet.
"""

from __future__ import annotations

__all__: list[str] = []
