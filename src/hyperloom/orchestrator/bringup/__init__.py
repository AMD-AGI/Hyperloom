# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Boot-observation layer: classify a bring-up attempt, pin the trees it ran against."""

from __future__ import annotations

from hyperloom.orchestrator.bringup.argv_preflight import (
    ARGV_INVALID,
    argv_invalid_observation,
    check_server_argv,
    is_argv_invalid,
)
from hyperloom.orchestrator.bringup.env_preflight import (
    ENV_FAULT,
    EnvVerdict,
    check_environment,
    is_env_fault,
)
from hyperloom.orchestrator.bringup.ladder import observation_summary
from hyperloom.orchestrator.bringup.observe import (
    observe_bringup,
    recorded_verdict,
    session_root,
    verdict_of,
)
from hyperloom.orchestrator.bringup.persist import (
    DEGRADED_NO_PATH,
    DEGRADED_UNREADABLE,
    load_boot_observation,
    write_boot_observation,
)
from hyperloom.orchestrator.bringup.trees import resolve_trees, write_trees

__all__ = [
    "ARGV_INVALID",
    "DEGRADED_NO_PATH",
    "DEGRADED_UNREADABLE",
    "ENV_FAULT",
    "EnvVerdict",
    "argv_invalid_observation",
    "check_environment",
    "check_server_argv",
    "is_argv_invalid",
    "is_env_fault",
    "load_boot_observation",
    "observation_summary",
    "observe_bringup",
    "recorded_verdict",
    "resolve_trees",
    "session_root",
    "verdict_of",
    "write_boot_observation",
    "write_trees",
]
