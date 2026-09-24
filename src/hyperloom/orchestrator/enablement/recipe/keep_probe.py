# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""KEEP-time observation of the graded environment.

The only durable version assertions today are build-time: captured before any
later setup command or patch runs, then used to validate the final image. This
observes the assertion set and the distribution closure *at* the KEEP, after
every mutation that reaches the launched image, through the interpreter the
accepted bench launched and under the override that bench applied.

A bare-environment probe would report a different environment than the one
graded -- a source build reaches its packages only through the override's
prefixes -- so both probes run with the override applied exactly as the launch
applies it. Where the launching interpreter is not the orchestrator's to
resolve, nothing is reported: naming a plausible interpreter would reproduce the
defect this closes in a new place.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess  # nosec B404 - local interpreter probe, argv-only, no shell.
from typing import Any, Mapping, Sequence

from .steps import select_linked_build

log = logging.getLogger(__name__)

_PROBE_TIMEOUT_SEC = 120

#: One invocation returns both the closure and the assertion set; the adapters'
#: own helper makes the same shape of call one package at a time.
_PROBE_SCRIPT = (
    "import json,sys\n"
    "import importlib.metadata as m\n"
    "d={}\n"
    "for dist in m.distributions():\n"
    "    name=(dist.metadata['Name'] if dist.metadata else '') or ''\n"
    "    if name:\n"
    "        d[name]=dist.version or ''\n"
    "print(json.dumps({'interpreter_tag': sys.version.split()[0], 'distributions': d}))\n"
)


def resolve_keep_interpreter(
    override: Mapping[str, Any] | None,
    *,
    backend_name: str,
    backend_interpreter: str = "",
) -> str:
    """Return the interpreter the graded server ran, or ``""``.

    Within the override the priority is ``runtime_python_exe`` then
    ``framework_python`` -- the order ``apply_runtime_override`` itself encodes
    and the launcher reads back. An override naming neither falls back to the
    interpreter the caller resolved, on every backend and not only on bypass: an
    enablement that patches the framework in place never provisions a runtime,
    so keying the fallback on the backend name would leave that topology's
    closure permanently unobserved.

    Args:
        override: The runtime override launch bound, if any.
        backend_name: The active benchmark backend, retained so a backend that
            cannot name an interpreter resolves to ``""`` rather than guessing.
        backend_interpreter: The interpreter that launched the graded server
            when the override names none. The caller resolves it -- the serving
            framework's interpreter, which on a split-venv host is not the
            benchmark backend's own.
    """
    resolved = str((override or {}).get("runtime_python_exe") or "").strip()
    resolved = resolved or str((override or {}).get("framework_python") or "").strip()
    if resolved:
        return resolved
    return str(backend_interpreter or "").strip() if str(backend_name or "").strip() else ""


def keep_assertion_packages(
    *,
    provision_versions: Mapping[str, str] | None,
    build_manifest: Sequence[Any],
    specialist_task_id: str,
) -> tuple[str, ...]:
    """Return the package names whose versions are asserted at the KEEP.

    Both sources contribute names only. Every version in the assertion set is
    the one the KEEP probe observes; substituting a provisioning-time or
    build-time version is the defect that observation exists to close.

    Args:
        provision_versions: The round's provisioning map, or ``None`` when no
            provisioning stage ran at all -- a KEEP reached through a build's
            launch-only probe. A stage that ran and installed nothing names
            nothing, and does not borrow the build's names.
        build_manifest: The durable build manifest the linked attempt is joined
            from. Read only when no provisioning stage ran, because keying the
            names on a provisioning result would otherwise leave every accepted
            build -- the principal path carrying version assertions --
            permanently unobserved.
        specialist_task_id: The round declaring this KEEP, whose probe the
            linked build's routing sentinel names.

    Returns:
        The names to assert, empty when no source names any, which the
        sufficiency rules read as an assertion set never observed at the KEEP.
    """
    if provision_versions is not None:
        return tuple(str(name) for name in provision_versions)
    _sentinel, row = select_linked_build(
        {"build_manifest": list(build_manifest or []), "last_specialist_task_id": specialist_task_id}
    )
    return tuple(str(name) for name in (row or {}).get("installed_versions") or {})


def probe_environment_closure(
    interpreter: str,
    *,
    env: Mapping[str, str],
    packages: tuple[str, ...] = (),
) -> tuple[dict[str, Any], dict[str, str]]:
    """Observe the distribution closure and the assertion set at the KEEP.

    Args:
        interpreter: The interpreter the graded server ran.
        env: The environment the graded server was launched into, composed by
            the caller. It is what the probe runs under, because a ``PYTHONPATH``
            the launch saw and the probe does not yields a closure that is
            missing distributions the KEEP actually depended on.
        packages: Names to lift into the assertion map.

    Returns:
        ``(environment_closure, installed_versions_at_keep)``; both empty when
        the probe cannot run or returns nothing, which the sufficiency rules
        read as absent rather than as an empty environment.
    """
    if not interpreter:
        return {}, {}
    env = dict(env)
    try:
        completed = subprocess.run(  # nosec B603 - argv-only, no shell.
            [interpreter, "-c", _PROBE_SCRIPT],
            capture_output=True,
            text=True,
            env=env,
            timeout=_PROBE_TIMEOUT_SEC,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        log.debug("enablement: KEEP environment probe failed to spawn", exc_info=True)
        return {}, {}
    if completed.returncode != 0:
        return {}, {}
    try:
        payload = json.loads(completed.stdout or "{}")
    except ValueError:
        return {}, {}
    distributions = {str(k): str(v) for k, v in (payload.get("distributions") or {}).items()}
    if not distributions:
        return {}, {}
    closure = {"interpreter_tag": str(payload.get("interpreter_tag") or ""), "distributions": distributions}
    lowered = {name.lower().replace("-", "_"): version for name, version in distributions.items()}
    assertions = {
        pkg: lowered[pkg.lower().replace("-", "_")] for pkg in packages if pkg.lower().replace("-", "_") in lowered
    }
    return closure, assertions


def keep_probe_env(override: Mapping[str, Any] | None) -> dict[str, str]:
    """Return the inherited environment with the graded override applied.

    Exposed because resolving *which* interpreter the graded launch used has to
    happen under the same environment the launch ran in -- an override that
    rewrites ``PATH`` selects a different executable than the ambient one does.
    """
    from ...actions.executors._grid_runner import apply_runtime_override

    env = dict(os.environ)
    if override:
        apply_runtime_override(env, dict(override))
    return env
