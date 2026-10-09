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

log = logging.getLogger(__name__)

# ``export NAME=value`` with no ``${NAME:-...}`` guard: the recipe's own value
# wins over anything the caller exported under that name.
_UNGUARDED_EXPORT_RE = re.compile(r"^[^\S\n]*export\s+([A-Za-z_][A-Za-z0-9_]*)=(?!\"?\$\{?\1[:-])", re.MULTILINE)


def resolve_launch_server_script(bench: Mapping[str, Any]) -> str:
    """Path of the script that boots the server, or ``""`` when unresolvable.

    ``benchmark_script`` names a server launcher on every non-AgentX run. The
    AgentX switch pins the aiperf client there instead, so for that one recipe
    shape the launcher is the builtin the client delegates to. Resolution
    mirrors ``aiperf_client.sh``: the same ``AGENTX_SERVER_SCRIPT`` override,
    the same ``{framework}_{gpu}.sh`` fallback, the same ``<checkout>/benchmarks/``
    directory and no recursive search. Recipe-recorded values beat the ambient
    env because the recipe is the record of what actually ran.
    """
    envs = bench.get("envs") if isinstance(bench.get("envs"), dict) else {}
    script = Path(str(bench.get("benchmark_script") or "").strip()).name
    if not script:
        return ""

    native_agentx = False
    if "agentx" in bench:
        from hyperloom.inference_optimizer.agentx.native import native_agentx_enabled

        native_agentx = native_agentx_enabled(bench["agentx"])
    if is_agentx_client_script(script) or native_agentx:
        # Native launchers own the full replay lifecycle; only the framework's
        # generic proxy implements GEAK's server-only phase.
        script = (
            ""
            if native_agentx
            else str(envs.get("AGENTX_SERVER_SCRIPT") or os.environ.get("AGENTX_SERVER_SCRIPT") or "").strip()
        )
        if not script:
            framework = str(bench.get("framework") or envs.get("FRAMEWORK") or "").strip().lower()
            if not framework:
                return ""
            gpu = (
                str(
                    envs.get("GPU_TYPE")
                    or envs.get("RUNNER_TYPE")
                    or bench.get("runner_type")
                    or os.environ.get("GPU_TYPE")
                    or os.environ.get("RUNNER_TYPE")
                    or "mi300x"
                )
                .strip()
                .lower()
            )
            script = f"{framework}_{gpu}.sh"

    for root in (
        str(bench.get("inferencex_path") or "").strip(),
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
    """Env names the launcher re-exports unconditionally."""
    path = resolve_launch_server_script(bench)
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
