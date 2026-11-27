# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The framework checkout, scriptable entrypoint and operator pins a rendered point is wired to."""

from __future__ import annotations

import logging
import os
from typing import Any

from hyperloom.inference_optimizer.framework_paths import GENERIC_FRAMEWORK_ROOT_ENV, operator_benchmark_script

log = logging.getLogger(__name__)


def _scriptable_runner_type(bench: dict[str, Any], gpu_type: str | None) -> str:
    """Resolve the runner suffix a scriptable entrypoint is named after.

    Framework-independent: the suffix names the machine, not the workload. Empty when nothing names the machine.
    """
    return (
        str(gpu_type or "").strip().lower()
        or str(bench.get("runner_type") or "").strip().lower()
        or os.environ.get("GPU_TYPE", "").strip().lower()
        or os.environ.get("RUNNER_TYPE", "").strip().lower()
    )


def _sync_repo_aliases(
    bench: dict[str, Any],
    envs: dict[str, Any],
    *,
    framework: str,
    prefer_dir: bool = False,
) -> None:
    """Keep ``<FRAMEWORK>_REPO_PATH`` and ``<FRAMEWORK>_DIR`` on one checkout.

    Both spellings reach the benchmark and a script may read either, so letting
    them drift points one half of a run at a different tree than the other.

    Args:
        bench: The ``benchmark`` section; the sync applies only to its framework.
        envs: The ``benchmark.envs`` mapping, updated in place.
        framework: Framework whose prefix the two aliases carry.
        prefer_dir: Resolve ``_DIR`` first, for the caller that saw the operator
            supply only that spelling.
    """
    if str(bench.get("framework") or "").strip().lower() != framework.strip().lower():
        return
    prefix = framework.strip().upper()
    order = (f"{prefix}_DIR", f"{prefix}_REPO_PATH") if prefer_dir else (f"{prefix}_REPO_PATH", f"{prefix}_DIR")
    repo_path = str(envs.get(order[0]) or envs.get(order[1]) or "").strip()
    if repo_path:
        envs[f"{prefix}_REPO_PATH"] = repo_path
        envs[f"{prefix}_DIR"] = repo_path


def _publish_scriptable_repo_root(framework: str, repo_path: str) -> None:
    """Publish a scriptable framework's checkout into the orchestrator's own env.

    A scriptable framework runs out of a repo checkout rather than a pip-installed
    package, so ``resolve_kernel_search_roots`` can only find it through
    ``<FRAMEWORK>_REPO_PATH`` / ``<FRAMEWORK>_DIR`` in ``os.environ``. Writing the
    resolved path into the materialized YAML reaches the *benchmark* subprocess
    but not the orchestrator — so without this the framework's own code is absent
    from the roots the specialist is pointed at and from the trees a patch can be
    grounded against, on a session that otherwise looks correctly configured.

    An operator-provided value always wins; this only fills the gap when the
    checkout was resolved (or is about to be cloned) by Hyperloom itself. The path
    is published even when it does not exist yet, because the benchmark wrapper
    clones on first use and the roots are recomputed per call.

    Args:
        framework: Framework name, used for the env prefix.
        repo_path: The resolved checkout path.
    """
    name = str(framework or "").strip().upper()
    path = str(repo_path or "").strip()
    if not name or not path:
        return
    for var in (f"{name}_REPO_PATH", f"{name}_DIR"):
        if not os.environ.get(var, "").strip():
            os.environ[var] = path
            log.info(
                "%s: published %s=%s so the framework source root reaches PolicyGate",
                framework,
                var,
                path,
            )


def _resolve_framework_repo_path(
    envs: dict[str, Any],
    *,
    framework: str,
) -> str:
    """Resolve the framework checkout a session may patch, prefixed forms first.

    Resolution order is ``<FRAMEWORK>_REPO_PATH`` > ``<FRAMEWORK>_DIR`` >
    :data:`GENERIC_FRAMEWORK_ROOT_ENV`, with ``os.environ``
    ahead of the materialized ``envs`` at each rung. The prefixed names keep
    precedence because they are the more specific statement, so adding the generic
    form changes no existing behaviour; it exists because a session is
    single-framework by construction and an operator should not have to know the
    framework's name to point at its source.

    Args:
        envs: The materialized ``benchmark.envs`` mapping.
        framework: Framework name, used for the env prefix.

    Returns:
        The resolved path, or ``""`` when nothing was supplied.
    """
    prefix = str(framework or "").strip().upper()
    names = (f"{prefix}_REPO_PATH", f"{prefix}_DIR", GENERIC_FRAMEWORK_ROOT_ENV)
    for name in names:
        value = os.environ.get(name, "").strip() or str(envs.get(name) or "").strip()
        if value:
            return value
    return ""


#: Source-root envs Hyperloom writes into a vLLM launch; vLLM reports them as unknown.
VLLM_SOURCE_ROOT_ENVS: tuple[str, ...] = ("FRAMEWORK_REPO_PATH", "VLLM_REPO_PATH", "VLLM_DIR")


def _vllm_image_source(bench: dict[str, Any], envs: dict[str, Any]) -> str:
    """The prepared image checkout ``install.sh`` activates (``HYPERLOOM_VLLM_IMAGE_SOURCE=1``), which the server of a
    vLLM ``bench`` imports ahead of the installed wheel; ``""`` when there is none."""
    if str(bench.get("framework") or "").strip().lower() != "vllm":
        return ""
    if os.environ.get("HYPERLOOM_VLLM_IMAGE_SOURCE", "").strip() != "1":
        return ""
    return _resolve_framework_repo_path(envs, framework="vllm")


def _apply_vllm_source_runtime(bench: dict[str, Any], envs: dict[str, Any]) -> None:
    """Route a prepared image checkout into the vLLM server launch."""
    repo_path = _vllm_image_source(bench, envs)
    if not repo_path:
        return
    for name in VLLM_SOURCE_ROOT_ENVS:
        envs[name] = repo_path
        os.environ[name] = repo_path
    existing = str(envs.get("PYTHONPATH") or os.environ.get("PYTHONPATH") or "")
    entries = [repo_path, *(part for part in existing.split(os.pathsep) if part)]
    envs["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(entries))


def _operator_extra_env() -> dict[str, str]:
    """Return the ``--extra-env`` pins the CLI serialized, or ``{}``."""
    from hyperloom.common.env_safety import operator_extra_env

    return {key: value for key, value in operator_extra_env().items() if key.strip().upper() != "RUN_EVAL"}


def apply_scriptable_runtime_defaults(
    bench: dict[str, Any],
    envs: dict[str, Any],
    *,
    gpu_type: str | None,
    explicit_benchmark_script: bool,
) -> None:
    """Re-pin an operator's scriptable entrypoint and its runtime paths.

    Only the ``custom`` framework has such an entrypoint: its repo and script
    arrive at launch, so this only wires what was given and leaves anything
    missing to fail where it is diagnosable.

    ``--extra-env`` is the only channel an operator has for the knobs their own
    script reads, so those pins are written into ``benchmark.envs`` rather than
    left in the orchestrator's environment: the materialized config is what the
    measurement contract is read from, and a pin that never lands there is
    neither delivered to the benchmark nor protected from being overwritten by
    a variant.
    """
    if str(bench.get("framework") or "").strip().lower() != "custom":
        return

    runner_type = _scriptable_runner_type(bench, gpu_type)
    bench["runner_type"] = runner_type
    if not explicit_benchmark_script:
        script = operator_benchmark_script("custom", runner_type)
        if script:
            bench["benchmark_script"] = script

    # A variant's own overrides are applied after this hook, so they still win
    # for keys the operator did not pin; setdefault only fills the gaps.
    for key, value in _operator_extra_env().items():
        envs.setdefault(key, value)

    repo_path = _resolve_framework_repo_path(envs, framework="custom")
    if not repo_path:
        return
    envs["CUSTOM_REPO_PATH"] = repo_path
    envs["CUSTOM_DIR"] = repo_path
    # Also under the framework-agnostic name, and in the config rather than only
    # in os.environ. An entrypoint Hyperloom did not write can only be expected
    # to know the generic spelling, and the benchmark inherits its environment
    # from whichever process launched the runner, which may have started before
    # this publish, so a stale <FRAMEWORK>_DIR there outranks anything published
    # later. Config envs are
    # applied on top of that inherited environment, so this is what actually
    # reaches the script.
    envs["FRAMEWORK_REPO_PATH"] = repo_path
    _publish_scriptable_repo_root("custom", repo_path)
