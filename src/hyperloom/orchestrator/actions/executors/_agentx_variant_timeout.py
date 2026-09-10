# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""AgentX variant timeout pricing shared by phase and executor code."""

from __future__ import annotations

from typing import Any


def agentx_variant_timeout_sec(cap: int, *, shared_state: Any = None, conc: int | None = None) -> int:
    """Raise a variant's hard cap to what an AgentX round actually needs."""
    from ._workload_envs import agentx_active, agentx_env_for_conc

    if not agentx_active(shared_state):
        return cap
    from .baseline import agentx_baseline_timeout_sec

    return max(cap, agentx_baseline_timeout_sec(agentx_env_for_conc(conc)))
