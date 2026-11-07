# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The InferenceX server script a recipe boots through, and the env names it pins."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.common.agentx_workload import is_agentx_client_script
from hyperloom.common.benchmark_config import BenchmarkPoint
from hyperloom_common.coerce import first_text

log = logging.getLogger(__name__)

# ``export NAME=value`` with no ``${NAME:-...}`` guard: the recipe's own value
# wins over anything the caller exported under that name.
_UNGUARDED_EXPORT_RE = re.compile(r"^[^\S\n]*export\s+([A-Za-z_][A-Za-z0-9_]*)=(?!\"?\$\{?\1[:-])", re.MULTILINE)


def resolve_launch_server_script(point: BenchmarkPoint) -> str:
    """Path of the script that boots the server, or ``""`` when unresolvable.

    ``benchmark_script`` names a server launcher on every non-AgentX run. The
    MLPerf switch pins its client there instead, whose server inference_suite
    boots, so for that one recipe shape the launcher is the framework's builtin:
    the ``AGENTX_SERVER_SCRIPT`` override, else ``{framework}_{gpu}.sh``, in
    ``<checkout>/benchmarks/`` with no recursive search. Recipe-recorded values
    beat the ambient env because the recipe is the record of what actually ran.
    """
    envs = point.envs
    script = Path(point.benchmark_script.strip()).name
    if not script:
        return ""

    if is_agentx_client_script(script):
        script = first_text(envs.get("AGENTX_SERVER_SCRIPT"), os.environ.get("AGENTX_SERVER_SCRIPT"))
        if not script:
            framework = first_text(point.framework, envs.get("FRAMEWORK")).lower()
            if not framework:
                return ""
            gpu = first_text(
                envs.get("GPU_TYPE"),
                envs.get("RUNNER_TYPE"),
                point.runner_type,
                os.environ.get("GPU_TYPE"),
                os.environ.get("RUNNER_TYPE"),
                "mi300x",
            ).lower()
            script = f"{framework}_{gpu}.sh"

    for root in (
        point.inferencex_path.strip(),
        os.environ.get("INFERENCEX_PATH", "").strip(),
    ):
        if not root:
            continue
        benchmarks = Path(root) / "benchmarks"
        candidate = benchmarks / script
        # The builtin sources benchmark_lib.sh from its own directory and dies
        # without it, so a half-populated checkout resolves to nothing.
        if candidate.is_file() and (benchmarks / "benchmark_lib.sh").is_file():
            return str(candidate)
    return ""


def launcher_overwritten_envs(bench: Mapping[str, Any]) -> frozenset[str]:
    """Env names the launcher re-exports unconditionally.

    Raises:
        PointError: ``bench`` is not shaped as a point's ``benchmark`` block.
    """
    path = resolve_launch_server_script(BenchmarkPoint.from_block(bench))
    if not path:
        return frozenset()
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        log.warning("recipe: could not read the server script %s; assuming it constrains nothing", path)
        return frozenset()
    return frozenset(m.group(1) for m in _UNGUARDED_EXPORT_RE.finditer(text))


__all__ = [
    "launcher_overwritten_envs",
    "resolve_launch_server_script",
]
