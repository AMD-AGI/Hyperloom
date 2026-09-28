# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Evidence an enablement KEEP records about the tree and runtime it accepts.

Correctness of the accepted round, the accepted levers nothing reads, the build outputs the framework root does not
carry, and the runtime environment the KEEP probes. None of it depends on the executor that asks for it.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ._accuracy_gate import DEFAULT_ENABLEMENT_ACCURACY_FLOOR, accuracy_meets_floor, classify_accuracy_failure
from ._integrate_attempt import IntegrateAttempt

log = logging.getLogger(__name__)


def is_eval_origin(params: dict[str, Any]) -> bool:
    """Whether the enablement candidate came from the eval gate, not the boot gate."""
    return str(params.get("enablement_origin") or "") == "eval"


def enablement_correctness(
    params: dict[str, Any],
    gate_evidence: dict[str, Any],
) -> tuple[bool | None, dict[str, Any]]:
    """Judge the candidate's accuracy against the floor its origin demands.

    Returns:
        ``(correctness_ok, eval_provenance)``. ``correctness_ok`` is ``None``
        only for a boot-origin round with no score at all, which claimed
        nothing about accuracy; every other absence fails closed.
    """
    enablement_accuracy = gate_evidence.get("enablement_accuracy")
    param_floor = params.get("enablement_accuracy_floor")
    floor = float(param_floor) if isinstance(param_floor, (int, float)) else DEFAULT_ENABLEMENT_ACCURACY_FLOOR
    eval_origin = is_eval_origin(params)
    accuracy_kind = classify_accuracy_failure(enablement_accuracy, floor)
    correctness_ok: bool | None
    if enablement_accuracy is None:
        # Truly absent: eval-origin fails closed; boot-origin stays provisional.
        correctness_ok = False if eval_origin else None
    else:
        # Present but below floor / non-positive / non-finite is a refusal.
        correctness_ok = accuracy_meets_floor(enablement_accuracy, floor)
    # eval-origin only: a score with no task/metric did not come from a real
    # eval, so it cannot clear the gate. This reads the candidate's OWN run
    # (both keys are stamped beside the accuracy it is judging), unlike the
    # contract fingerprint it replaces: RUN_EVAL is itself a hashed contract
    # field, so an eval-less re-baseline could poison the stored digest and
    # veto every later candidate without ever consulting its accuracy.
    if (
        eval_origin
        and correctness_ok
        and not (gate_evidence.get("enablement_accuracy_task") and gate_evidence.get("enablement_accuracy_metric"))
    ):
        correctness_ok = False
        log.warning(
            "integrate_patch: eval-origin accuracy %s carries no task/metric; reverting",
            enablement_accuracy,
        )
    return correctness_ok, {
        "enablement_origin": str(params.get("enablement_origin") or ""),
        "enablement_observed_accuracy": enablement_accuracy,
        "enablement_accuracy_floor": floor,
        "accuracy_task": gate_evidence.get("enablement_accuracy_task") or "",
        "accuracy_metric": gate_evidence.get("enablement_accuracy_metric") or "",
        "enablement_eval_failure_kind": accuracy_kind or "",
    }


def levers_without_readers(
    enablement: Any,
    framework_root: Path | None,
    *,
    framework: str,
    effective_config: Mapping[str, Any] | None = None,
) -> list[str] | None:
    """Return accepted env levers in the framework's namespace that nothing reads.

    A lever is accepted because a round that set it advanced, not because
    anything was shown to read it. A knob a specialist introduced in a patch
    that was later superseded leaves its name behind in ``accepted_config``,
    and the recipe then exports an env no code consults -- a replay sets it
    and reproduces nothing, silently.

    Only the framework's own namespace is judged. ``AMD_SERIALIZE_KERNEL``
    is read by the HIP runtime and ``NCCL_*`` by the collective library;
    their absence from the framework tree says nothing about them.

    Every regular file the framework ships is searched, matched as bytes. A
    lever is as likely to be read by a kernel through ``getenv`` or by a
    launch script through shell expansion as by Python, and a reader can sit
    in a file with no extension at all -- a ``Dockerfile``, a ``Makefile``.
    A suffix list is not evidence of absence: skipping a file is what turns
    a working lever into a refusal. A match inside a compiled artifact
    counts too, which can only make this miss a dangling lever, never invent
    one.

    Returns:
        The lever names with no reader, ``[]`` when a scan found none, and
        ``None`` when the tree could not be read -- which is not evidence
        that every lever has one.
    """
    if framework_root is None or not framework.strip():
        return []
    # This KEEP's own effective config first. The standing ``accepted_config``
    # is not replaced with it until the lane re-arms on the result, so a
    # lever this round introduced -- the one the recipe will export -- is
    # not in shared state yet, and scanning only that would check every
    # round's levers except the decisive one.
    accepted = getattr(enablement, "accepted_config", None) or {}
    merged: dict[str, Any] = {}
    for source in (accepted, effective_config):
        if not isinstance(source, Mapping):
            continue
        block = source.get("extra_envs")
        if isinstance(block, Mapping):
            merged.update({str(k): v for k, v in block.items()})
    envs = merged
    prefix = f"{framework.strip().upper()}_"
    names = sorted({str(k).strip() for k in (envs or {}) if str(k).strip().startswith(prefix)})
    if not names:
        return []
    if not framework_root.is_dir():
        # An empty walk over a tree that is not there would report every
        # lever as unread, which is a refusal built out of nothing.
        return None
    needles = {name: name.encode("ascii", "ignore") for name in names}
    unread = set(names)
    try:
        for source in framework_root.rglob("*"):
            if not unread:
                break
            if not source.is_file():
                continue
            blob = source.read_bytes()
            unread -= {name for name in unread if needles[name] in blob}
    except OSError:
        return None
    return sorted(unread)


def build_output_trees(attempt_root: Path) -> list[Path]:
    """Return the trees a build names as its own output.

    The build records them in its ``result.json`` as the prefixes a runtime
    would import from; that is the build's own statement of where its output
    lives, so it is read rather than guessed at. A result that cannot be
    read falls back to the candidate worktrees the layout puts them in --
    still narrower than the attempt root, which also holds cloned
    dependencies and any provisioned virtual environment.
    """
    result = attempt_root / "result.json"
    try:
        payload = json.loads(result.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        payload = {}
    runtime = payload.get("runtime") if isinstance(payload, dict) else None
    prefixes = (runtime or {}).get("pythonpath_prefixes") if isinstance(runtime, dict) else None
    trees = [Path(str(p)) for p in prefixes if str(p).strip()] if isinstance(prefixes, list) else []
    if trees:
        return trees
    return sorted(d for d in attempt_root.glob("candidates/*/worktree") if d.is_dir())


def build_extensions_not_carried(
    enablement: Any, framework_root: Path | None, *, specialist_task_id: str = ""
) -> list[str] | None:
    """Return the build's compiled extensions the framework root does not have.

    A build does not install itself: its outputs reach the framework root
    only as artifacts a specialist declared, one by one. Declare two of
    three and the round still boots, benchmarks, and is kept -- the gap
    surfaces hours later as an op the loaded extension does not export, on
    whichever code path first needs it.

    Only the extensions built *for this framework package* are judged, and
    only inside the tree the build itself names as its output. An attempt
    root also holds the other repositories a build cloned and, where one was
    provisioned, a virtual environment with its own installed copy of this
    same package -- comparing against those would refuse a recipe over files
    the framework root was never meant to carry. Compared by
    content, so an extension the base image already shipped under the same
    name counts as not carried.

    Every shared object anywhere in the package is considered, not only
    ``.abi3.so`` directly beneath it: an extension built without the
    stable-ABI tag carries an interpreter-specific suffix instead, and one
    belonging to a subpackage sits below the package root. Each is compared
    at its path relative to the package, so a nested module is matched
    against the nested module rather than against a same-named file at the
    top, and the name reported is that relative path.

    Returns:
        The names left behind, ``[]`` only after at least one of the linked
        build's output trees was scanned and nothing was missing (or when no
        build is linked, there being nothing to carry), and ``None`` when a
        build is linked whose outputs could not be read -- an absent tree, a
        cleaned-up worktree or an unreadable file. None of those are
        evidence that anything was carried.
    """
    if framework_root is None:
        return []
    from ...enablement.recipe.projections import select_linked_build

    rounds = list(getattr(enablement, "kept_rounds", None) or [])
    current = str(specialist_task_id or "").strip()
    if current and not any(str((r or {}).get("task_id") or "").strip() == current for r in rounds):
        rounds.append({"task_id": current})
    state = {
        "build_manifest": list(getattr(enablement, "build_manifest", None) or []),
        "last_specialist_task_id": str(getattr(enablement, "last_specialist_task_id", "") or ""),
        "kept_rounds": rounds,
    }
    _sentinel, row = select_linked_build(state)
    # Validated as text first: ``Path("")`` is ``Path(".")``, whose
    # ``is_dir()`` is true, so an absent attempt root would otherwise scan
    # the working directory and report whatever it found there.
    attempt_root_text = str((row or {}).get("attempt_root") or "").strip()
    if not attempt_root_text:
        return []
    attempt_root = Path(attempt_root_text)
    if not attempt_root.is_dir():
        return None
    missing: list[str] = []
    try:
        package_roots = [
            d for d in (prefix / framework_root.name for prefix in build_output_trees(attempt_root)) if d.is_dir()
        ]
        if not package_roots:
            # The build named output trees that are gone, or named none and
            # the candidate worktrees have been cleaned up. Either way this
            # scanned nothing, which is not the same as finding nothing.
            return None
        built_files = sorted((package, built) for package in package_roots for built in package.rglob("*.so"))
    except OSError:
        return None
    for package, built in built_files:
        relative = built.relative_to(package)
        installed = framework_root / relative
        try:
            if not installed.is_file() or installed.read_bytes() != built.read_bytes():
                missing.append(str(relative))
        except OSError:
            return None
    return missing


def graded_framework(params: dict[str, Any], materialized_config: str) -> str:
    """Return the framework the graded launch served, config first.

    The materialized config is what the launch read, so its own
    ``benchmark.framework`` outranks the round's params and the ambient
    ``$FRAMEWORK``; those remain the fallback for a round whose config could
    not be read.
    """
    if materialized_config:
        from ._server_argv import _benchmark_envs

        try:
            declared, _envs = _benchmark_envs(materialized_config)
        except (OSError, ValueError):
            declared = None
        if declared:
            return str(declared).strip().lower()
    from hyperloom.inference_optimizer.framework_registry import DEFAULT_FRAMEWORK

    return str(params.get("framework") or os.environ.get("FRAMEWORK") or DEFAULT_FRAMEWORK).strip().lower()


def graded_launch_env(override: Mapping[str, Any] | None, materialized_config: str) -> dict[str, str]:
    """Return the environment the graded server was launched into.

    The override is applied first and the config's ``benchmark.envs`` over
    it, matching the order the launch itself composes them in.
    """
    from ...enablement.recipe.keep_probe import keep_probe_env

    env = keep_probe_env(override)
    if not materialized_config:
        return env
    from ._server_argv import config_launch_env

    try:
        return config_launch_env(materialized_config, env)
    except (OSError, ValueError):
        return env


def probe_keep_environment(
    attempt: IntegrateAttempt,
    params: dict[str, Any],
    *,
    specialist_task_id: str,
    provision_result: Any,
    materialized_config: str = "",
) -> tuple[dict[str, Any], dict[str, str]]:
    """Observe the closure and assertion set under the accepted runtime.

    The runtime is the round's own provisioning result when it provisioned
    one, else the override the round was dispatched with -- a KEEP reached
    through a build's launch-only probe has no provisioning stage at all, so
    keying on it would leave every accepted build permanently unobserved.
    An enablement that patches the framework tree in place has neither, and
    is graded under the *serving framework's* interpreter with no override
    applied; the probe runs there too, because a closure observed only when
    some runtime was provisioned is absent for exactly the topology whose
    replay most needs it. The composed launch environment is what both the
    interpreter resolution and the probe itself run under. That fallback is
    ``_resolve_probe_python``, the
    same resolver the accuracy probes use, and not the benchmark backend's
    own interpreter -- on a split-venv host Magpie runs from one venv while
    the server it launches runs from another, and recording Magpie's
    distributions against that KEEP would be a confidently wrong closure,
    which is worse than an absent one. The bypass backend is the one case
    where the backend's interpreter is what launched the server, so it keeps
    resolving through the backend.

    Both the framework and the environment that fallback resolves against
    come from the accepted round's own materialized config -- the same
    artifact the launch read -- overlaid with the runtime override, because
    a config's ``benchmark.envs`` can itself set ``PATH`` and decide which
    executable the graded server was. Resolving against the ambient process
    environment instead names whichever interpreter this coordinator happens
    to see.
    The packages named in the assertion set are sourced the same way, from
    the build attempt this round's probe was opened for when no provisioning
    stage ran; their versions are the probe's observation either way.
    """
    from ...enablement.recipe.keep_probe import (
        keep_assertion_packages,
        probe_environment_closure,
        resolve_keep_interpreter,
    )
    from ._benchmark_interpreter import _resolve_probe_python
    from .benchmark_backend import resolve_backend_name, resolve_benchmark_interpreter

    override: dict[str, Any] = {}
    if provision_result is not None and getattr(provision_result, "ok", False):
        override = provision_result.runtime.to_runtime_override()
    if not override:
        raw = params.get("runtime_override")
        override = dict(raw) if isinstance(raw, dict) else {}
    graded_env = graded_launch_env(override, materialized_config)
    backend = resolve_backend_name()
    if backend == "bypass":
        fallback = resolve_benchmark_interpreter()
    else:
        fallback = _resolve_probe_python(
            graded_framework(params, materialized_config),
            env=graded_env,
        )
    interpreter = resolve_keep_interpreter(
        override,
        backend_name=backend,
        backend_interpreter=fallback,
    )
    enablement = getattr(attempt.shared_state, "enablement", None)
    provision_versions = (
        None if provision_result is None else getattr(provision_result, "installed_versions", None) or {}
    )
    packages = keep_assertion_packages(
        provision_versions=provision_versions,
        build_manifest=getattr(enablement, "build_manifest", None) or [],
        specialist_task_id=specialist_task_id,
    )
    return probe_environment_closure(interpreter, env=graded_env, packages=packages)
