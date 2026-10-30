# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node-only: the numeric settings a per-round server restart reads, resolved before it starts."""

from __future__ import annotations

import os
from dataclasses import dataclass

from hyperloom.common.env import env_int
from hyperloom.inference_optimizer import multi_node

# Default /health poll timeout; override per-run via HYPERLOOM_MN_HEALTH_WAIT_S.
DEFAULT_HEALTH_TIMEOUT_S = 900  # 15 min.


@dataclass(frozen=True)
class RestartSettings:
    """Every numeric ``HYPERLOOM_MN_*`` setting one restart and its readiness waits read."""

    poll_timeout_s: int
    health_wait_s: int
    health_wait_explicit: bool
    health_wait_aiter_s: int
    health_poll_s: int
    worker_port: int
    worker_wedge_grace_s: int
    worker_crash_grace_s: int
    completion_probe_count: int
    completion_probe_tokens: int
    completion_probe_min_tokens: int
    models_empty_grace_s: int
    completions_grace_s: int
    published_ready_s: int
    published_ready_ok_streak: int
    published_dns_skip_after: int

    @classmethod
    def from_env(cls) -> RestartSettings:
        """Read every setting; raises EnvValueError naming the first one that is set but is not an integer."""
        # Local import to keep httpx out of the single-node import path.
        from hyperloom.inference_optimizer.multi_node.cli import resolve_poll_timeout_s

        return cls(
            poll_timeout_s=resolve_poll_timeout_s(DEFAULT_HEALTH_TIMEOUT_S),
            health_wait_s=env_int("HYPERLOOM_MN_HEALTH_WAIT_S", DEFAULT_HEALTH_TIMEOUT_S),
            health_wait_explicit=bool((os.environ.get("HYPERLOOM_MN_HEALTH_WAIT_S") or "").strip()),
            health_wait_aiter_s=env_int("HYPERLOOM_MN_HEALTH_WAIT_AITER_S", 1800),
            health_poll_s=env_int("HYPERLOOM_MN_HEALTH_POLL_S", 10),
            worker_port=env_int("HYPERLOOM_MN_WORKER_PORT", 30000),
            worker_wedge_grace_s=env_int("HYPERLOOM_MN_WORKER_WEDGE_GRACE_S", 420),
            worker_crash_grace_s=env_int("HYPERLOOM_MN_WORKER_CRASH_GRACE_S", 45),
            completion_probe_count=env_int("HYPERLOOM_MN_COMPLETION_PROBE_COUNT", 2),
            completion_probe_tokens=multi_node.completion_probe_tokens(),
            completion_probe_min_tokens=multi_node.completion_probe_min_tokens(),
            models_empty_grace_s=env_int("HYPERLOOM_MN_MODELS_EMPTY_GRACE_S", 600),
            completions_grace_s=env_int("HYPERLOOM_MN_COMPLETIONS_GRACE_S", 480),
            published_ready_s=env_int("HYPERLOOM_MN_PUBLISHED_READY_S", 300),
            # One poll is the smallest streak that is still a gate, so the two poll counts floor at 1, not 0.
            published_ready_ok_streak=max(1, env_int("HYPERLOOM_MN_PUBLISHED_READY_OK_STREAK", 2)),
            published_dns_skip_after=max(1, env_int("HYPERLOOM_MN_PUBLISHED_DNS_SKIP_AFTER", 6)),
        )
