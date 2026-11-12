# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What the environment must give a multi-branch session before anything of it is created.

Each check raises ValueError when the environment contradicts the command line (a usage error) and
:class:`EnvironmentIncomplete` when it lacks something a session needs, naming the variables to set.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from hyperloom.common.benchmark_config import BENCHMARK_CONFIG_ENV, benchmark_config_path
from hyperloom_common.env import EnvValueError
from hyperloom.common.outside_facts.image_check import mode_from_environment
from hyperloom.common.perf_metric import agentx_enabled
from hyperloom.measure.short_prompts.staging import CACHE_ENV as RUNNER_CACHE_ENV
from hyperloom.orchestrator.knowledge.remote_recipe.client import RemoteRecipeClient, RemoteRecipeConfigurationError
from hyperloom_multibranch.contracts.llm_prices import price_of
from hyperloom_multibranch.contracts.toolbox import WorkloadKind
from hyperloom_multibranch.blackboard.rsi.archive import ARCHIVE_ENV
from hyperloom_multibranch.scheduling.llm import default_model
from hyperloom_multibranch.scheduling.local_peers import AGENTX_ENV, specialist_model
from hyperloom_multibranch.scheduling.settings import RunSettings, Workload


class EnvironmentIncomplete(RuntimeError):
    """The environment lacks what a session needs; the message names the variables to set."""


def require_workload_switch(workload: Workload, env: Mapping[str, str]) -> None:
    """Raise ValueError when ``env`` sets ``HYPERLOOM_AGENTX`` to anything but what ``workload``'s kind says, or names
    a benchmark YAML in ``HYPERLOOM_BENCHMARK_CONFIG``; the session sets both for an AgentX workload itself, so an
    operator who set them means a workload ``--workload`` and its flags do not name."""

    if benchmark_config_path(env) is not None:
        raise ValueError(f"{BENCHMARK_CONFIG_ENV} names a workload YAML; --workload and its flags define the workload")
    if AGENTX_ENV not in env:
        return
    try:
        agentx = agentx_enabled(env)
    except EnvValueError as exc:
        raise ValueError(str(exc)) from exc
    if agentx != (workload.kind is WorkloadKind.AGENTX):
        raise ValueError(f"{AGENTX_ENV}={env[AGENTX_ENV]!r} contradicts --workload {workload.kind.value}")


def require_priced_models(max_llm_usd: float, env: Mapping[str, str]) -> None:
    """Raise ValueError when ``max_llm_usd`` caps the session's LLM dollars and the price sheet lists no rate for the
    model the judge, the patch resolver and the fix critic ask or the one the specialists of ``env`` run on: their
    calls would record tokens only, which the cap cannot count."""

    if not max_llm_usd:
        return
    models = {"the judge's": default_model(), "the specialists'": specialist_model(env)}
    unpriced = [f"{whose} model {model}" for whose, model in models.items() if price_of(model) is None]
    if unpriced:
        raise ValueError(
            f"--max-llm-usd cannot count the dollars of {' or '.join(unpriced)}, which the price sheet does not list;"
            " run on a listed model or without a dollar cap"
        )


def recipe_kb_from_env() -> RemoteRecipeClient | None:
    """The Recipe KB client ``KB_STORE_URL`` and ``KB_STORE_TOKEN`` configure, ``None`` when neither is set; raises
    :class:`EnvironmentIncomplete` when only one is."""

    if not any(os.environ.get(name, "").strip() for name in ("KB_STORE_URL", "KB_STORE_TOKEN")):
        return None
    try:
        return RemoteRecipeClient.from_env()
    except RemoteRecipeConfigurationError as exc:
        raise EnvironmentIncomplete(f"the Recipe KB is half configured: {exc}") from exc


def require_rsi_archive(env: Mapping[str, str]) -> None:
    """Raise :class:`EnvironmentIncomplete` naming ``HYPERLOOM_RSI_ARCHIVE`` when ``env`` does not set it, and unless
    the archive it names is a writable directory. There is no default: the archive is the deployment's shared folder."""

    if not env.get(ARCHIVE_ENV, "").strip():
        raise EnvironmentIncomplete(
            f"{ARCHIVE_ENV} is not set: name the writable shared directory the session archives its search tree in"
        )
    archive = Path(env[ARCHIVE_ENV].strip())
    if not (archive.is_dir() and os.access(archive, os.W_OK)):
        raise EnvironmentIncomplete(
            f"the RSI archive {archive} is not a writable directory: mount it, or set {ARCHIVE_ENV} to one"
        )


def require_session_environment(settings: RunSettings, env: Mapping[str, str]) -> None:
    """Raise ValueError when ``env`` contradicts the workload of ``settings``, cannot price its dollar cap or sets an
    image check mode there is none of."""

    require_workload_switch(settings.workload, env)
    require_priced_models(settings.max_llm_usd, env)
    mode_from_environment(env)


def hosted_runner_cache(env: Mapping[str, str]) -> Path:
    """This host's runner cache directory, which ``HYPERLOOM_RUNNER_CACHE`` must name as an absolute path; raises
    :class:`EnvironmentIncomplete` naming the variable otherwise. Hosted sessions take no default: a cache built where
    the measurement sandboxes do not mount it leaves them with no runner, and a sealed sandbox cannot fetch one."""

    value = env.get(RUNNER_CACHE_ENV, "").strip()
    if not value or not Path(value).is_absolute():
        raise EnvironmentIncomplete(
            f"{RUNNER_CACHE_ENV} is not set to an absolute path: name the directory of the runner cache on the shared"
            " storage the measurement sandboxes mount"
        )
    return Path(value)


__all__ = [
    "EnvironmentIncomplete",
    "hosted_runner_cache",
    "recipe_kb_from_env",
    "require_priced_models",
    "require_rsi_archive",
    "require_session_environment",
    "require_workload_switch",
]
