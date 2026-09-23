# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU-lease params and lane resolution for a Coordinator-dispatched specialist."""

from __future__ import annotations

import logging
from typing import Any

from hyperloom.common.env import is_truthy

from .collaborator import CoordinatorCollaborator

log = logging.getLogger(__name__)


class GpuLanes(CoordinatorCollaborator):
    """Resolves GPU params and lane leases for Coordinator-internal dispatches."""

    def _framework_gpu_params(self) -> dict[str, Any]:
        """Return the ``{needs_gpu, gpu_count}`` params for framework authoring."""
        try:
            from .actions.executors._multi_node_env import is_multi_node

            if is_multi_node():
                return {}
        except Exception:  # noqa: BLE001 — treat probe failure as single-node
            # Loud enough to notice: this swallowed a wrong-depth relative import once, which read as "single-node"
            # and handed a multi-node run the whole machine.
            log.warning("gpu_lanes: multi-node probe failed; assuming single-node", exc_info=True)
        cap = int(getattr(self.framework_gpu_pool, "capacity", 0) or 0)
        if cap <= 0:
            return {}
        return {"needs_gpu": True, "gpu_count": cap}

    def _framework_authoring_lanes_ttl(self, params: dict[str, Any], *, base_ttl_sec: int) -> tuple[list[str], int]:
        """Resolve lanes + lease TTL for an internally-dispatched framework specialist."""
        lanes = ["research_lane"]
        ttl = int(base_ttl_sec or 0)
        if is_truthy(params.get("needs_gpu")):
            lanes.append("gpu_research_lane")
            try:
                ttl = self._gpu_lease_ttl_sec(ttl, params=params)
            except Exception:  # noqa: BLE001 — fall back to the base TTL
                log.exception(
                    "framework GPU: gpu_research_lane TTL re-source failed; using base TTL",
                )
        return lanes, ttl
