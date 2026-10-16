# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The optional arguments an injected Implementer callback receives only by declaring them."""

from __future__ import annotations

import inspect
import logging
from collections.abc import Callable, Mapping
from typing import Any

log = logging.getLogger(__name__)


def declared_extras(agent_fn: Callable[..., Any], extras: Mapping[str, Any]) -> dict[str, Any]:
    """The entries of ``extras`` that ``agent_fn`` names as parameters; none when its signature cannot be read.

    A callback opts into each optional argument by naming it, so one written before an
    argument existed is still called the way it was written for.
    """
    try:
        params = inspect.signature(agent_fn).parameters
    except (ValueError, TypeError):
        log.debug("could not introspect agent_fn signature", exc_info=True)
        return {}
    return {name: value for name, value in extras.items() if name in params}
