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
    """The entries of ``extras`` that ``agent_fn`` takes; none when its signature cannot be read.

    A callback takes the entries it names. One that accepts any keyword takes them all: it has declared that it
    forwards what it is given, and withholding an entry from it would drop that entry without a trace.
    """
    try:
        params = inspect.signature(agent_fn).parameters
    except (ValueError, TypeError):
        log.debug("could not introspect agent_fn signature", exc_info=True)
        return {}
    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in params.values()):
        return dict(extras)
    return {name: value for name, value in extras.items() if name in params}
