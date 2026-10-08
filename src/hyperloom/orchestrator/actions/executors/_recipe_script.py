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


class RecipeLeverUnavailableError(ValueError):
    """Raised when the recipe cannot carry a lever the variant depends on.

    Measuring such a variant produces a precise re-run of the baseline under
    the variant's name, which no downstream reader can tell apart from a
    change that simply had no effect.
    """


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


_SOURCE_RE = re.compile(r'^\s*(?:source|\.)\s+"?([^";\s]+)"?\s*$', re.MULTILINE)


def _read_sourced_texts(script_path: Path) -> list[str]:
    """Return the text of scripts that ``script_path`` sources (one level deep).

    Only resolves sibling paths (same directory) because that is the only safe
    assumption for scripts in an InferenceX checkout. Unknown or non-sibling
    paths are silently skipped.
    """
    try:
        text = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    sourced: list[str] = []
    parent = script_path.parent
    for m in _SOURCE_RE.finditer(text):
        raw = m.group(1)
        # Resolve simple variable-prefix patterns like ${SCRIPT_DIR}/name.sh
        # by stripping any leading variable reference and keeping the filename.
        name = Path(re.sub(r"^\$\{?[A-Za-z_][A-Za-z0-9_]*\}?/", "", raw)).name
        if not name:
            continue
        candidate = parent / name
        try:
            sourced.append(candidate.read_text(encoding="utf-8", errors="replace"))
        except OSError:
            pass
    return sourced


def recipe_launch_contract(bench: Mapping[str, Any]) -> tuple[bool, frozenset[str]]:
    """What the resolved server script accepts: ``(reads_extra_args, names_it_overwrites)``.

    ``reads_extra_args`` is False only when neither the entrypoint script nor
    any script it directly sources names the framework's extra-args variable or
    forwards positional arguments. Thin shims that ``source`` a shared body
    (e.g. ``xdit_mi300x.sh`` sourcing ``xdit_bench_common.sh``) are therefore
    handled correctly. Both answers default to "imposes nothing" when the script
    cannot be read, so an unresolvable recipe never drops a lever.

    ``names_it_overwrites`` is derived from the entrypoint only: sourced scripts
    run in the same shell scope but their unguarded exports are an implementation
    detail of the shared body, not a contract the thin shim imposes on its caller.
    """
    path = resolve_launch_server_script(bench)
    if not path:
        return True, frozenset()
    script_path = Path(path)
    try:
        text = script_path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        log.warning("recipe: could not read the server script %s; assuming it constrains nothing", path)
        return True, frozenset()
    args_env = server_args_env_name(bench.get("framework"))
    reads_extra_args = args_env in text or bool(_POSITIONAL_ARGS_RE.search(text))
    if not reads_extra_args:
        for sourced_text in _read_sourced_texts(script_path):
            if args_env in sourced_text or _POSITIONAL_ARGS_RE.search(sourced_text):
                reads_extra_args = True
                break
    return reads_extra_args, frozenset(m.group(1) for m in _UNGUARDED_EXPORT_RE.finditer(text))


__all__ = ["RecipeLeverUnavailableError", "recipe_launch_contract", "resolve_launch_server_script"]
