# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""GPU-lease params and lane resolution for a Coordinator-dispatched specialist.

Neither optimisation arm owns this: the source arm dispatches authoring
specialists, enablement dispatches repair ones, and both need the same
whole-machine pool and the same lane tiering. It lived on the source arm only
because that is where the first caller was, which left the enablement package
reaching back into a phase handler at runtime for it.
"""

from __future__ import annotations

import logging
from typing import Any

from .collaborator import CoordinatorCollaborator

log = logging.getLogger(__name__)


class GpuLanes(CoordinatorCollaborator):
    """Resolves GPU params and lane leases for Coordinator-internal dispatches."""

    @staticmethod
    def _coerce_needs_gpu(value: Any) -> bool:
        """Coerce a params ``needs_gpu`` value (bool | str) to bool.

        Matches the truthy set used by ``intent_router`` / the dispatcher so a
        JSON-string ``"true"`` and a real ``True`` route identically.

        Args:
            value: The raw ``needs_gpu`` params value.

        Returns:
            bool: Whether the specialist requests a GPU lease.
        """
        if isinstance(value, str):
            return value.strip().lower() in ("1", "true", "yes", "on")
        return bool(value)

    def _framework_gpu_params(self) -> dict[str, Any]:
        """Return the ``{needs_gpu, gpu_count}`` params for framework authoring.

        ``gpu_count`` defaults to the whole-machine pool capacity.

        Returns:
            dict: ``{"needs_gpu": True, "gpu_count": <n>}`` on single-node hosts
            with a non-empty whole-machine pool, else ``{}`` (authoring falls
            back to the research-lane-only path).
        """
        try:
            from .actions.executors._multi_node_env import is_multi_node

            if is_multi_node():
                return {}
        except Exception:  # noqa: BLE001 — treat probe failure as single-node
            # Loud enough to notice: this swallowed a wrong-depth relative
            # import once, which read as "single-node" and handed a multi-node
            # run the whole machine.
            log.warning("gpu_lanes: multi-node probe failed; assuming single-node", exc_info=True)
        cap = int(getattr(self.framework_gpu_pool, "capacity", 0) or 0)
        if cap <= 0:
            return {}
        return {"needs_gpu": True, "gpu_count": cap}

    def _framework_authoring_lanes_ttl(self, params: dict[str, Any], *, base_ttl_sec: int) -> tuple[list[str], int]:
        """Resolve lanes + lease TTL for an internally-dispatched framework specialist.

        When ``needs_gpu`` is set the task acquires the cap-1
        ``gpu_research_lane`` (in addition to ``research_lane``) and its lease
        TTL is re-sourced from the GPU wall budget.

        Args:
            params: The specialist params (checked for ``needs_gpu``).
            base_ttl_sec: The default lane lease TTL (raised, never lowered, for
                a GPU task).

        Returns:
            ``(lanes, ttl_sec)`` — ``["research_lane"]`` (+ ``gpu_research_lane``
            when GPU) and the (possibly budget-raised) lease TTL.
        """
        lanes = ["research_lane"]
        ttl = int(base_ttl_sec or 0)
        if self._coerce_needs_gpu(params.get("needs_gpu")):
            lanes.append("gpu_research_lane")
            try:
                ttl = self._gpu_lease_ttl_sec(ttl, params=params)
            except Exception:  # noqa: BLE001 — fall back to the base TTL
                log.exception(
                    "framework GPU: gpu_research_lane TTL re-source failed; using base TTL",
                )
        return lanes, ttl
