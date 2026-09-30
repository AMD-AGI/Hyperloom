# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU-lease TTL resolution for Coordinator-internal specialist dispatches."""

from __future__ import annotations

import logging
from typing import Any

from .collaborator import CoordinatorCollaborator

log = logging.getLogger(__name__)


class GpuLanes(CoordinatorCollaborator):
    """Resolves lease TTL for Coordinator-internal specialist dispatches."""

    def _framework_authoring_lanes_ttl(self, params: dict[str, Any], *, base_ttl_sec: int) -> tuple[list[str], int]:
        """Compute lanes and lease TTL for an internally-dispatched specialist.

        Delegates lane selection to ``specialist_lanes``; adjusts the TTL when
        a GPU lease is required.
        """
        from .specialists.profile import requires_gpu, specialist_lanes

        base_lanes = ["research_lane"]
        lanes = specialist_lanes(params, base_lanes)
        ttl = int(base_ttl_sec or 0)
        if requires_gpu(params):
            ttl = self._gpu_lease_ttl_sec(ttl, params=params)
        return lanes, ttl
