# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The InferenceX server script a recipe boots through, and what it does to the launch it is handed."""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from hyperloom.inference_optimizer.framework_registry import server_args_env_name

log = logging.getLogger(__name__)

# ``export NAME=value`` with no ``${NAME:-...}`` guard: the recipe's own value
# wins over anything the caller exported under that name.
_UNGUARDED_EXPORT_RE = re.compile(r"^[^\S\n]*export\s+([A-Za-z_][A-Za-z0-9_]*)=(?!\"?\$\{?\1[:-])", re.MULTILINE)

# ``"$@"`` / ``${@}``: forwarding its own positional arguments is the other way
# a script accepts extra server args, alongside the framework's args variable.
_POSITIONAL_ARGS_RE = re.compile(r"\$\{?@")

# Spelled out rather than imported from ``agentx.deploy``: this module is on the
# default benchmark path, which is pinned not to import the agentx package.
_AGENTX_CLIENT_SCRIPT = "aiperf_client.sh"


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

    if script == _AGENTX_CLIENT_SCRIPT:
        script = str(envs.get("AGENTX_SERVER_SCRIPT") or os.environ.get("AGENTX_SERVER_SCRIPT") or "").strip()
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


def recipe_launch_contract(bench: Mapping[str, Any]) -> tuple[bool, frozenset[str]]:
    """What the resolved server script accepts: ``(reads_extra_args, names_it_overwrites)``.

    ``reads_extra_args`` is False only for a script that names neither the
    framework's extra-args variable nor its positional arguments, which makes
    every ``extra_server_args`` on that recipe a no-op the measurement cannot
    distinguish from a proposal that simply did not help. Both answers default
    to "imposes nothing" when the script cannot be read, so an unresolvable
    recipe never drops a lever.
    """
    path = resolve_launch_server_script(bench)
    if not path:
        return True, frozenset()
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        log.warning("recipe: could not read the server script %s; assuming it constrains nothing", path)
        return True, frozenset()
    args_env = server_args_env_name(bench.get("framework"))
    reads_extra_args = args_env in text or bool(_POSITIONAL_ARGS_RE.search(text))
    return reads_extra_args, frozenset(m.group(1) for m in _UNGUARDED_EXPORT_RE.finditer(text))


__all__ = ["recipe_launch_contract", "resolve_launch_server_script"]
