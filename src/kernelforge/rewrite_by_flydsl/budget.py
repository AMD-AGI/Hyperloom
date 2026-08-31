# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Central wall-clock policy for the FlyDSL rewrite pipeline."""

from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class RewriteBudgetPolicy:
    """Named reserves and ceilings shared by every rewrite stage."""

    applyback_reserve_sec: int = 20 * 60
    applyback_host_validation_reserve_sec: int = 5 * 60
    applyback_min_agent_sec: int = 60
    applyback_post_agent_reserve_sec: int = 30
    applyback_agent_finalization_reserve_sec: int = 2 * 60
    applyback_shell_command_max_sec: int = 5 * 60
    import_validation_timeout_sec: int = 60
    driver_preflight_reserve_sec: int = 60
    reference_preflight_timeout_sec: int = 10 * 60
    candidate_probe_timeout_sec: int = 5 * 60

    @property
    def applyback_start_min_remaining_sec(self) -> int:
        return self.applyback_host_validation_reserve_sec + self.applyback_min_agent_sec

    def search_stop_unix(self, deadline_unix: float) -> float:
        return deadline_unix - self.applyback_reserve_sec

    def remaining_seconds(self, deadline_unix: float | None) -> float:
        if not deadline_unix or deadline_unix <= 0:
            return float("inf")
        return max(0.0, deadline_unix - time.time())

    def can_start_applyback(self, deadline_unix: float | None) -> bool:
        return self.remaining_seconds(deadline_unix) > self.applyback_start_min_remaining_sec

    def agent_timeout_sec(
        self,
        *,
        deadline_unix: float | None,
        configured_timeout_sec: int,
        attempts_left: int,
    ) -> int:
        remaining = self.remaining_seconds(deadline_unix)
        if not math.isfinite(remaining):
            return max(1, int(configured_timeout_sec))
        available = remaining - self.applyback_host_validation_reserve_sec
        return max(
            1,
            int(
                min(
                    float(configured_timeout_sec),
                    available / max(1, int(attempts_left)),
                )
            ),
        )

    def host_validation_timeout_sec(
        self,
        deadline_unix: float | None,
    ) -> int:
        remaining = self.remaining_seconds(deadline_unix)
        if not math.isfinite(remaining):
            return self.applyback_host_validation_reserve_sec
        return max(1, int(remaining - self.applyback_post_agent_reserve_sec))

    def to_dict(self) -> dict[str, int]:
        return asdict(self)


DEFAULT_REWRITE_BUDGET = RewriteBudgetPolicy()
