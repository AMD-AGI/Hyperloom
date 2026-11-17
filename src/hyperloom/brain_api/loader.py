# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Loading a brain by module name. The body reaches a brain only through here, and only by string."""

from __future__ import annotations

import importlib
from typing import Any, Mapping

from hyperloom.brain_api.brain import Brain, BrainContractError, validate_brain
from hyperloom.brain_api.points import points_of

DEFAULT_MODULE = "hyperloom.brain"
FACTORY = "create_brain"


def load_brain(module: str = DEFAULT_MODULE, config: Mapping[str, Any] | None = None) -> Brain:
    """Import ``module``, build its brain with ``config``, and check it against the contract and its eval points.

    Raises:
        BrainContractError: if the module cannot be imported, has no factory, or builds a brain that fails the check.
    """
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        raise BrainContractError(f"cannot import brain module {module!r}: {exc}") from exc
    factory = getattr(mod, FACTORY, None)
    if not callable(factory):
        raise BrainContractError(f"brain module {module!r} has no {FACTORY}()")
    brain = validate_brain(factory(dict(config or {})))
    points_of(brain)
    return brain


def create(context: Any) -> Brain:
    """The plug factory of the ``brain`` socket; ``config.module`` names the brain, the rest is its configuration."""
    config = dict(context.config)
    module = str(config.pop("module", DEFAULT_MODULE))
    return load_brain(module, config)
