# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Workload-shape tags a session stamps on the recipe KB entries it writes."""

from __future__ import annotations

import os
from typing import Any

from hyperloom.common.coerce import optional_positive_int

#: SharedState fields that describe the served workload's parallelism and traffic shape.
WORKLOAD_SHAPE_FIELDS: tuple[str, ...] = ("tp", "ep", "conc", "isl", "osl", "max_model_len")


def workload_shape_tags(state: Any) -> dict[str, Any]:
    """Return the session's workload-shape tags, omitting unset values.

    ``ep`` falls back to the ``EP`` environment variable for callers that never
    set it on the state.
    """
    tags: dict[str, Any] = {}
    for field in WORKLOAD_SHAPE_FIELDS:
        value = getattr(state, field, None)
        if value not in (None, "", 0):
            tags[field] = value
    if "ep" not in tags:
        ep = optional_positive_int(os.environ.get("EP"))
        if ep is not None:
            tags["ep"] = ep
    return tags
