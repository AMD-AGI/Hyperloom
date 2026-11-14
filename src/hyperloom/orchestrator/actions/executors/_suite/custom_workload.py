# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What a custom single-file round adds to a scriptable one.

A custom campaign optimizes one operator-declared module, so its round enforces that
declaration before any budget is spent.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from hyperloom.common.optimization_scope import optimization_scope
from hyperloom.orchestrator.framework.optimization_scope import check_patch_scope

log = logging.getLogger(__name__)

CUSTOM_FRAMEWORK = "custom"


def is_custom(framework: str) -> bool:
    return str(framework or "").strip().lower() == CUSTOM_FRAMEWORK


def enforce_custom_scope(framework: str) -> None:
    """Fail a custom round whose declared boundary cannot be enforced.

    Checked here, where ``FRAMEWORK_REPO_PATH`` is already exported, so an
    out-of-boundary campaign fails before spending any budget rather than at the first
    patch apply.

    Raises:
        ValueError: When the declaration is unreadable or the boundary does not hold.
    """
    if not is_custom(framework) or optimization_scope() is None:
        return
    check_patch_scope(Path(os.environ.get("FRAMEWORK_REPO_PATH", "")), [])


__all__ = ["CUSTOM_FRAMEWORK", "enforce_custom_scope", "is_custom"]
