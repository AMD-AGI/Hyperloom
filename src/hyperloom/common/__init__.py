# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""``hyperloom.common`` — zero-dependency shared library.

Home for deduplicated, first-party-dependency-free building blocks. Constraint:
this package may only import the stdlib (plus ``httpx`` for the ``llm``
submodule) and must never import ``orchestrator`` / ``inference_optimizer`` /
``agents`` — keeping the dependency graph acyclic.

Intentionally has no ``paths.py`` module: the kernel-agent ``tools/_paths.py``
mirror is a standalone-script copy that must not depend on ``hyperloom.common``,
and no other package's path helpers are safe-to-merge duplicates.
"""

from __future__ import annotations

__all__: list[str] = []
