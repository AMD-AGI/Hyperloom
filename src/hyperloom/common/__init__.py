# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""``hyperloom.common`` — zero-dependency shared library (tree-reform scaffold).

Target home for the deduplicated, first-party-dependency-free building blocks
described in ``tree-reform.MD`` §7 (``io`` / ``env`` / ``payload_aliases`` /
``jsonio`` / ``timeutil`` / ``gain_math`` / ``paths`` / ``subprocess_bridge`` /
``llm/`` …). Constraint: this package may only import the stdlib (plus ``httpx``
for the future ``llm`` submodule) and must never import ``orchestrator`` /
``inference_optimizer`` / ``agents`` — keeping the dependency graph acyclic.

Phase P2.0 intentionally ships an empty scaffold: extraction happens in P2.1.
"""

from __future__ import annotations

__all__: list[str] = []
