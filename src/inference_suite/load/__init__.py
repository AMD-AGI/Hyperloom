# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The round's load: the fixed-shape InferenceX client, the AgentX replay and the MLPerf agentic harness against the
server, and the server-less scriptable workload.

Each runs as a child of the round under its gates, writes its output below the run_dir's ``client/``, and ends in
one of three ways: a stop gate tripped, the child exited non-zero, or it wrote the result document the round reads.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from inference_suite.contracts.config import ServerConfig
from inference_suite.contracts.report import FailedStep
from inference_suite.runtime.watch import Exited, Stop


def inherited_env(server: ServerConfig) -> dict[str, str]:
    """The host environment the round's children may see: the names ``server.env.inherit`` allows that are set."""
    return {name: os.environ[name] for name in server.env_inherit if name in os.environ}


def exit_stop(ended: Exited | Stop, what: str, stderr: Path) -> Stop | None:
    """Why the child's end fails the round: the gate that stopped it, or its non-zero exit."""
    if isinstance(ended, Stop):
        return ended
    if ended.returncode == 0:
        return None
    return Stop(FailedStep.CLIENT_FAILED, f"{what} exited with code {ended.returncode}; see {stderr}")


def read_result(path: Path, what: str) -> Mapping[str, object] | Stop:
    """The JSON object ``what`` wrote to ``path``, or why there is none to read."""
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return Stop(FailedStep.CLIENT_OUTPUT_MISSING, f"{what} wrote no {path}")
    except ValueError as exc:
        return Stop(FailedStep.CLIENT_OUTPUT_MISSING, f"{what} wrote {path} that is not JSON: {exc}")
    if not isinstance(document, Mapping):
        return Stop(FailedStep.CLIENT_OUTPUT_MISSING, f"{what} wrote {path} holding {type(document).__name__}")
    return document


__all__ = ["exit_stop", "inherited_env", "read_result"]
