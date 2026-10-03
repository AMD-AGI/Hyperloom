# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Lane and lease-TTL resolution for Coordinator-internal specialist dispatches."""

from __future__ import annotations

from typing import Any

from .collaborator import CoordinatorCollaborator
from .specialists.profile import requires_gpu, specialist_lanes


class GpuLanes(CoordinatorCollaborator):
    """Resolves lanes and lease TTL for Coordinator-internal specialist dispatches."""

    def framework_authoring_lanes_ttl(self, params: dict[str, Any], *, base_ttl_sec: int) -> tuple[list[str], int]:
        """Resolve lanes + lease TTL; a GPU specialist's TTL follows its GPU lease."""
        ttl = int(base_ttl_sec or 0)
        if requires_gpu(params):
            ttl = self._coord.dispatcher.gpu_lease_ttl_sec(ttl, params=params)
        return specialist_lanes(params, ["research_lane"]), ttl
