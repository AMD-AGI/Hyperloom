# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Name the faults in the host that no patch to the model could repair.

The detection is structural -- an :func:`importlib.util.find_spec` and an import
in the interpreter that will serve, an :func:`os.path.exists` on the checkpoint,
a connect to the port, the free device memory against the weight bytes on disk
-- and the verdict comes off exception classes, never off
error wording, which is carried into the record only for a human. Three
outcomes, never two: :data:`OK` and :data:`FAULT` are verdicts, and
:data:`UNAVAILABLE` is the answer whenever a check could not be made.
"""

from __future__ import annotations

import logging
import os
import shutil
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.bringup import BootObservation, Excerpt, LadderStage, redact
from hyperloom.orchestrator.bringup.argv_preflight import (
    INTERPRETER_UNPROVEN,
    OK,
    PROBE_FAILED,
    SERVING_PYTHON_ENV,
    UNAVAILABLE,
    ProbeFn,
    _clip,
    resolve_serving_interpreter,
    run_probe_json,
)

log = logging.getLogger(__name__)

#: Names this module's observations in downstream artifacts.
PRODUCER = "preflight.environment"

#: The marker an environment-fault observation carries, and the name of the
#: terminal it produces.
ENV_FAULT = "environment_fault"

#: The stream name recorded on the excerpt, so a reader can tell a preflight
#: record from a server log.
STREAM = "env_preflight"

#: The one outcome this module names for itself; :data:`OK` and
#: :data:`UNAVAILABLE` are the argv preflight's, shared so the two read alike.
FAULT = "fault"

#: The serving interpreter cannot reach the framework package at all.
FRAMEWORK_NOT_INSTALLED = "framework_not_installed"

#: The framework is installed and imports a module this interpreter does not
#: have. Read off ``ModuleNotFoundError.name``.
EXTENSION_MISSING = "extension_missing"

#: The framework is installed and a compiled extension it loads raised an
#: ``ImportError`` that is not a ``ModuleNotFoundError`` -- a build that exists
#: and does not load here.
EXTENSION_UNBUILT = "extension_unbuilt_for_platform"

#: The checkpoint the config names is a filesystem path that resolves to nothing.
CHECKPOINT_UNRESOLVED = "checkpoint_unresolved"

#: Something already holds the port the server would bind.
PORT_ALREADY_BOUND = "port_already_bound"

#: Where each fault sits on the boot ladder, so an environment record and a real
#: boot observation of the same wall are comparable.
_FAULT_STAGE: Mapping[str, LadderStage] = {
    FRAMEWORK_NOT_INSTALLED: LadderStage.IMPORT,
    EXTENSION_MISSING: LadderStage.IMPORT,
    EXTENSION_UNBUILT: LadderStage.IMPORT,
    CHECKPOINT_UNRESOLVED: LadderStage.CONFIG_VALIDATE,
    PORT_ALREADY_BOUND: LadderStage.PROCESS_START,
}

#: Bounded by how long the framework's top-level package takes to import.
_IMPORT_TIMEOUT_SEC = 60.0

#: Seconds a connect to the serving port may take before it counts as free.
_PORT_CONNECT_TIMEOUT_SEC = 1.0

#: Weight file suffixes, most preferred first. A checkpoint that ships two
#: formats ships the same tensors twice, so only the first one present counts.

# The verdict is the exception's class and its ``name`` attribute, never its
# message: two interpreters phrase the same missing extension differently.
_IMPORT_PROGRAM = (
    "import importlib, importlib.util as u, json, sys\n"
    "name = sys.argv[1]\n"
    "out = {'found': False, 'exc': '', 'missing': '', 'detail': ''}\n"
    "try:\n"
    "    out['found'] = u.find_spec(name) is not None\n"
    "except BaseException as exc:\n"
    "    out['exc'] = type(exc).__name__\n"
    "    out['detail'] = str(exc)[:4000]\n"
    "if out['found'] and not out['exc']:\n"
    "    try:\n"
    "        importlib.import_module(name)\n"
    "    except BaseException as exc:\n"
    "        out['exc'] = type(exc).__name__\n"
    "        out['missing'] = str(getattr(exc, 'name', '') or '')\n"
    "        out['detail'] = str(exc)[:4000]\n"
    "print(json.dumps(out))\n"
)


@dataclass(frozen=True)
class EnvVerdict:
    """What the host said about the round that is about to launch.

    Attributes:
        status: :data:`OK`, :data:`FAULT` or :data:`UNAVAILABLE`.
        fault: Which named fault was found, or the reason a check could not be
            made when the status is :data:`UNAVAILABLE`.
        detail: Probe or filesystem evidence, clipped.
        subject: What the fault is about -- the module, the path, the port.
    """

    status: str
    fault: str = ""
    detail: str = ""
    subject: str = ""

    @property
    def terminal(self) -> bool:
        """bool: True when this verdict must stop the round rather than launch."""
        return self.status == FAULT


def _checkpoint_verdict(model: str) -> EnvVerdict:
    """Judge the checkpoint the config names.

    Only an explicit filesystem path is judged; a bare repository id resolves
    from a hub and is not this host's to hold.

    Returns:
        EnvVerdict: A fault when the path does not resolve, else :data:`OK`.
    """
    raw = model.strip()
    if not raw:
        return EnvVerdict(status=OK)
    expanded = os.path.expanduser(raw)
    if not (os.path.isabs(expanded) or raw.startswith(("." + os.sep, ".." + os.sep, "~"))):
        return EnvVerdict(status=OK)
    if os.path.exists(expanded):
        return EnvVerdict(status=OK)
    return EnvVerdict(
        status=FAULT,
        fault=CHECKPOINT_UNRESOLVED,
        detail=f"the checkpoint path {expanded} does not exist on this host",
        subject=expanded,
    )


def _port_verdict(port: int) -> EnvVerdict:
    """Judge whether the serving port is free.

    Returns:
        EnvVerdict: A fault when the port answers a connect, :data:`OK` when it
        refuses one, and :data:`UNAVAILABLE` for any other socket outcome.
    """
    if port <= 0:
        return EnvVerdict(status=OK)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(_PORT_CONNECT_TIMEOUT_SEC)
            answered = sock.connect_ex(("127.0.0.1", int(port))) == 0
    except OSError as exc:
        return EnvVerdict(status=UNAVAILABLE, fault=PROBE_FAILED, detail=f"{type(exc).__name__}: {exc}")
    if not answered:
        return EnvVerdict(status=OK)
    return EnvVerdict(
        status=FAULT,
        fault=PORT_ALREADY_BOUND,
        detail=f"127.0.0.1:{port} already accepts connections, so this round's server cannot bind it",
        subject=str(port),
    )


def _proven_serving_interpreter(framework: str, launch_env: Mapping[str, str]) -> str:
    """Return the interpreter the server will run in, only when that is provable.

    Proof is either the pin the launch env carries, or an interpreter sitting
    beside the framework's own console script. Anything else is the resolution
    falling back to whatever ``python3`` is on ``PATH``.

    Returns:
        str: The proven interpreter, or ``""`` when nothing proves one.
    """
    resolved = resolve_serving_interpreter(framework, launch_env)
    if not resolved:
        return ""
    if str(launch_env.get(SERVING_PYTHON_ENV, "")).strip():
        # A pin is its own proof: it is the interpreter the launch will use.
        return resolved
    console = shutil.which(framework, path=str(launch_env.get("PATH", "")).strip() or None)
    if console and os.path.dirname(resolved) == os.path.dirname(console):
        return resolved
    return ""


def _import_verdict(
    framework: str,
    launch_env: Mapping[str, str],
    *,
    probe: ProbeFn | None,
) -> EnvVerdict:
    """Judge the serving interpreter's ability to import the framework.

    The probe runs under ``launch_env`` so it resolves the packages the server
    will, and only against an interpreter :func:`_proven_serving_interpreter`
    vouches for.

    Returns:
        EnvVerdict: The verdict; :data:`UNAVAILABLE` when the probe could not
        answer.
    """
    interpreter = _proven_serving_interpreter(framework, launch_env)
    if not interpreter:
        return EnvVerdict(
            status=UNAVAILABLE,
            fault=INTERPRETER_UNPROVEN,
            detail=f"nothing proves which interpreter would serve {framework or 'this framework'}",
        )
    result = run_probe_json(
        interpreter,
        _IMPORT_PROGRAM,
        [framework],
        env=launch_env,
        timeout_sec=_IMPORT_TIMEOUT_SEC,
        probe=probe,
    )
    if not result.ok or result.payload is None:
        return EnvVerdict(status=UNAVAILABLE, fault=PROBE_FAILED, detail=_clip(result.detail))
    payload = result.payload
    if "found" not in payload:
        # Reading a missing key as "the framework is absent" would turn any
        # unrelated JSON on that interpreter's stdout into a terminal.
        return EnvVerdict(status=UNAVAILABLE, fault=PROBE_FAILED, detail=_clip(str(payload)))
    detail = _clip(str(payload.get("detail", "")))
    exc = str(payload.get("exc", ""))
    if not payload["found"]:
        return EnvVerdict(
            status=FAULT,
            fault=FRAMEWORK_NOT_INSTALLED,
            detail=detail or f"{interpreter} cannot resolve {framework}",
            subject=framework,
        )
    if not exc:
        return EnvVerdict(status=OK)
    if exc == "ModuleNotFoundError":
        return EnvVerdict(
            status=FAULT,
            fault=EXTENSION_MISSING,
            detail=detail,
            subject=str(payload.get("missing", "")) or framework,
        )
    if exc == "ImportError":
        return EnvVerdict(status=FAULT, fault=EXTENSION_UNBUILT, detail=detail, subject=framework)
    # The framework raised for a reason that is not about what this host holds;
    # the launch will produce it and the classifier will place it.
    return EnvVerdict(status=OK)


def check_environment(
    *,
    framework: str,
    model: str,
    port: int,
    launch_env: Mapping[str, str],
    probe: ProbeFn | None = None,
) -> EnvVerdict:
    """Decide whether the host can host this round at all.

    The checks run cheapest-first and stop at the first fault.

    Args:
        framework: Framework the config serves.
        model: Model path or repository id the round would serve.
        port: The port the server would bind.
        launch_env: The environment the benchmark subprocess is launched with.
        probe: Subprocess shim, injected by tests.

    Returns:
        EnvVerdict: The first fault found, an unavailable verdict when a check
        could not be made, else :data:`OK`.
    """
    checks: Sequence[Callable[[], EnvVerdict]] = (
        lambda: _checkpoint_verdict(model),
        lambda: _port_verdict(port),
        lambda: _import_verdict(framework, launch_env, probe=probe),
    )
    unavailable: EnvVerdict | None = None
    for check in checks:
        verdict = check()
        if verdict.status == FAULT:
            return verdict
        if verdict.status == UNAVAILABLE and unavailable is None:
            unavailable = verdict
    return unavailable if unavailable is not None else EnvVerdict(status=OK)


def env_fault_observation(verdict: EnvVerdict, *, session_dir: Path | None = None) -> BootObservation:
    """Return the boot observation for a host that cannot run this round.

    Recorded like any other bring-up observation: at the ladder stage the launch
    would have died at, under this module's producer, carrying ``env_fault``.

    Args:
        verdict: The faulting verdict.
        session_dir: Session root, redacted out of the recorded text.

    Returns:
        BootObservation: The observation to persist.
    """
    roots = (str(session_dir),) if session_dir else ()
    stage = _FAULT_STAGE.get(verdict.fault, LadderStage.PROCESS_START)
    body = "\n".join(part for part in (f"environment fault ({verdict.fault})", verdict.detail) if part)
    excerpt = Excerpt(
        text=redact(body, roots=roots),
        stream=STREAM,
        byte_start=0,
        byte_end=len(body.encode("utf-8", "replace")),
    )
    return BootObservation(
        producer=PRODUCER,
        stage_reached=stage,
        stage_failed=stage,
        matched_marker=verdict.fault,
        excerpt=excerpt,
        evidence_ref=STREAM,
        env_fault=verdict.fault,
    )


def is_env_fault(observation: BootObservation | None) -> bool:
    """True when ``observation`` names one of the faults that end a run.

    Membership is this module's own vocabulary, not any observation carrying an
    ``env_fault``: the ladder classifier marks a host-resource shortfall that
    way too, and a config lever can still address that one.
    """
    return observation is not None and observation.env_fault in _FAULT_STAGE


__all__ = [
    "CHECKPOINT_UNRESOLVED",
    "ENV_FAULT",
    "EXTENSION_MISSING",
    "EXTENSION_UNBUILT",
    "FAULT",
    "FRAMEWORK_NOT_INSTALLED",
    "INTERPRETER_UNPROVEN",
    "OK",
    "PORT_ALREADY_BOUND",
    "PRODUCER",
    "UNAVAILABLE",
    "EnvVerdict",
    "check_environment",
    "env_fault_observation",
    "is_env_fault",
]
