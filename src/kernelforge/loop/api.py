# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The forge-loop interface: what code outside the loop and its orchestrator may use.

Other workflows and Hyperloom run forge-loop as a subprocess and read the result it publishes, or run it in process
with :func:`run_forge_loop`, and reuse a few pieces the loop owns. They reach all of it through this module, never
through the loop's internals.
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from collections.abc import Mapping
from typing import Any

from kernelforge.loop.bootstrap.errors import (
    ForgeLoopError,
    ForgeLoopUsageError,
    InvalidOptionValue,
    TaskPreparationFailed,
)
from kernelforge.loop.bootstrap.request import ForgeLoopRequest
from kernelforge.loop.bootstrap.run import run_forge_loop
from kernelforge.loop.gate_settings import GateMeasurement
from kernelforge.loop.path_ownership import (
    ATTEMPT_ROOT_DIR,
    PRODUCER_PATH_PATTERNS,
    is_producer_owned_path,
    runtime_gitignore_globs,
)
from kernelforge.loop.reporting import BestResultPublisher
from kernelforge.orchestrator.agent import ImplementerSetup, ImplementerTask, SessionOptions, make_agent_fn

log = logging.getLogger(__name__)

#: Brackets the JSON result forge-loop prints last, so a caller can find it in output that is otherwise free-form.
RESULT_SENTINEL = "__FORGE_RESULT__"


def forge_loop_command(*arguments: str) -> list[str]:
    """The argv that runs ``kernelforge forge-loop`` with this process's interpreter and package."""
    program = (
        [sys.executable, "-m", "kernelforge.cli"] if sys.executable else [shutil.which("kernelforge") or "kernelforge"]
    )
    return [*program, "forge-loop", *arguments]


def format_forge_loop_result(result: Mapping[str, Any]) -> str:
    """The line forge-loop prints to publish ``result``."""
    return f"{RESULT_SENTINEL}{json.dumps(result)}{RESULT_SENTINEL}"


def parse_forge_loop_result(output: str) -> dict[str, Any] | None:
    """The last result ``output`` published; ``None`` when it published none that is a JSON object.

    A missing result is silent: the caller decides what that means. A malformed one is logged, so it is not mistaken
    for a run that published nothing.
    """
    parts = output.split(RESULT_SENTINEL)
    if len(parts) < 3:
        return None
    try:
        payload = json.loads(parts[-2])
    except ValueError as error:
        log.warning("forge-loop result is not valid JSON: %s", error)
        return None
    if not isinstance(payload, dict):
        log.warning("forge-loop result is a JSON %s, not an object", type(payload).__name__)
        return None
    return payload


__all__ = [
    "ATTEMPT_ROOT_DIR",
    "BestResultPublisher",
    "ForgeLoopError",
    "ForgeLoopRequest",
    "ForgeLoopUsageError",
    "InvalidOptionValue",
    "GateMeasurement",
    "ImplementerSetup",
    "ImplementerTask",
    "PRODUCER_PATH_PATTERNS",
    "RESULT_SENTINEL",
    "SessionOptions",
    "TaskPreparationFailed",
    "format_forge_loop_result",
    "forge_loop_command",
    "is_producer_owned_path",
    "make_agent_fn",
    "parse_forge_loop_result",
    "run_forge_loop",
    "runtime_gitignore_globs",
]
