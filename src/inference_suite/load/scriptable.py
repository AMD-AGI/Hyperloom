# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scriptable load: a server-less benchmark script (xDiT, or an operator's own) run once under the round's gates.

The script is the scriptable base's (:func:`scriptable_base`): a script this package ships, by name, or the absolute
path of the operator's, with the sha256 its content had when the session froze it. A round whose script no longer has
that content is refused. The script gets the workload's params as its environment, beside the run-scoped names it must
not be handed by a param: where to write its result, the model, the GPUs and, in a profile round, where to write its
torch trace. Its result document is kept verbatim; judging it is the caller's (D2).
"""

from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from inference_suite.contracts.base_launch import (
    BaseLaunch,
    BaseLaunchError,
    ImageFacts,
    LaunchModel,
    LaunchPoint,
    LaunchQuery,
    LaunchSource,
    SourceKind,
    Topology,
    VersionBasis,
    Workload,
)
from inference_suite.contracts.config import ScriptableRunEnv, ScriptableWorkload, ServerConfig, SuiteConfig
from inference_suite.contracts.layout import RunLayout
from inference_suite.contracts.report import FailedStep, ScriptableResult
from inference_suite.load import exit_stop, inherited_env, read_result
from inference_suite.runtime.phases import Phase, PhaseClock
from inference_suite.runtime.process import spawn
from inference_suite.runtime.watch import RoundGate, Stop, wait_child

SCRIPTS_DIR = Path(__file__).with_name("scripts")
_SCRIPT_SUFFIX = ".sh"
_WHAT = "the scriptable script"


class ShippedScript(str, Enum):
    """The scripts this package ships, each named after the framework it runs."""

    XDIT = "xdit"

    @property
    def path(self) -> Path:
        return SCRIPTS_DIR / f"{self.value}{_SCRIPT_SUFFIX}"


@dataclass(frozen=True)
class ScriptableInvocation:
    """The script as it ran: its argv, and the env the round gave it beyond what it inherits."""

    argv: tuple[str, ...]
    env: Mapping[str, str]


@dataclass(frozen=True)
class ScriptableOutcome:
    """What the scriptable load did; ``invocation`` is ``None`` when the config named nothing runnable."""

    invocation: ScriptableInvocation | None
    returncode: int | None
    result: ScriptableResult | None
    stop: Stop | None


def script_path(script: str) -> Path | None:
    """The script ``script`` names: a shipped one by name, or the operator's by absolute path, if it exists."""
    shipped = {item.value: item for item in ShippedScript}.get(script)
    path = shipped.path if shipped is not None else Path(script)
    return path if path.is_absolute() and path.is_file() else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scriptable_base(query: LaunchQuery, script: str | None, model: LaunchModel, topology: Topology) -> BaseLaunch:
    """The base of a scriptable session: ``script``, a shipped name or an absolute path, frozen with its content.

    ``script`` ``None`` runs the script the query's framework ships.

    Raises:
        BaseLaunchError: When ``query`` is not scriptable, ``script`` is ``None`` for a framework that ships none, or
            ``script`` names no shipped script and no existing file.
    """
    if query.workload is not Workload.SCRIPTABLE:
        raise BaseLaunchError(f"a scriptable base answers a scriptable query, got {query.workload.value}")
    if script is None:
        if query.framework not in {item.value for item in ShippedScript}:
            raise BaseLaunchError(f"{query.framework} ships no script; the operator names its own")
        script = query.framework
    path = script_path(script)
    if path is None:
        raise BaseLaunchError(f"{script!r} is no shipped script and no absolute script path")
    shipped = path != Path(script)
    return BaseLaunch(
        query=query,
        source=LaunchSource(
            kind=SourceKind.DEFAULT_TABLE if shipped else SourceKind.USER_FILE,
            path=str(path),
            files={script: _sha256(path)},
        ),
        point=LaunchPoint(conc=query.concurrencies[0], exact=True),
        image=ImageFacts(run=None, recipe=None, framework_version=None, version_basis=VersionBasis.NOT_APPLICABLE),
        framework=query.framework,
        entrypoint=("bash", script),
        model=model,
        topology=topology,
        args=(),
        env={},
        client=None,
        resolved_unix=int(time.time()),
    )


def _env_value(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def script_env(workload: ScriptableWorkload, server: ServerConfig, layout: RunLayout, profile: bool) -> dict[str, str]:
    """The script's env beyond what it inherits: the params, then the run-scoped names."""
    params = {name: _env_value(value) for name, value in workload.params.items()}
    result = layout.inferencex_result
    scoped = {
        ScriptableRunEnv.RESULT_DIR: str(result.parent),
        ScriptableRunEnv.RESULT_FILENAME: result.stem,
        ScriptableRunEnv.MODEL: server.model_path,
        ScriptableRunEnv.TP: str(len(server.gpus)),
        ScriptableRunEnv.ROCR_VISIBLE_DEVICES: ",".join(str(gpu) for gpu in server.gpus),
    }
    if workload.quality_reference is not None:
        scoped[ScriptableRunEnv.QUALITY_REFERENCE] = workload.quality_reference
    if profile:
        trace_dir = str(layout.profile_torch_trace)
        scoped[ScriptableRunEnv.PROFILE] = "1"
        scoped[ScriptableRunEnv.VLLM_TORCH_PROFILER_DIR] = trace_dir
        scoped[ScriptableRunEnv.SGLANG_TORCH_PROFILER_DIR] = trace_dir
    return {**params, **{name.value: value for name, value in scoped.items()}}


def scriptable_invocation(config: SuiteConfig, base: BaseLaunch) -> ScriptableInvocation | Stop:
    """The script of ``base`` and the env ``config`` runs it with, or why it runs nothing."""
    workload = config.scriptable
    if workload is None:
        raise ValueError(f"scriptable_invocation needs a scriptable workload, got {config.workload.value}")
    path = script_path(base.script)
    if path is None:
        return Stop(FailedStep.CONFIG_INVALID, f"{base.script!r} is no shipped script and no absolute script path")
    if _sha256(path) != base.source.files[base.script]:
        return Stop(FailedStep.SOURCE_FILES_CHANGED, f"the script {path} changed since the session froze its base")
    env = script_env(workload, config.server, RunLayout(Path(config.paths.run_dir)), config.profile.enabled)
    return ScriptableInvocation((*base.entrypoint[:-1], str(path)), env)


def run_scriptable(config: SuiteConfig, base: BaseLaunch, gate: RoundGate, clock: PhaseClock) -> ScriptableOutcome:
    """Run ``base``'s script once under ``gate``, its whole run the measured window, and keep the result it writes."""
    invocation = scriptable_invocation(config, base)
    if isinstance(invocation, Stop):
        return ScriptableOutcome(None, None, None, invocation)
    layout = RunLayout(Path(config.paths.run_dir))
    clock.mark(Phase.MEASURED)
    child = spawn(
        invocation.argv, {**inherited_env(config.server), **invocation.env}, layout.client_stdout, layout.client_stderr
    )
    ended = wait_child(child, gate)
    clock.mark(Phase.FINISHED)
    returncode = None if isinstance(ended, Stop) else ended.returncode
    stop = exit_stop(ended, _WHAT, layout.client_stderr)
    if stop is not None:
        return ScriptableOutcome(invocation, returncode, None, stop)
    document = read_result(layout.inferencex_result, _WHAT)
    if isinstance(document, Stop):
        return ScriptableOutcome(invocation, returncode, None, document)
    result_file = str(layout.inferencex_result.relative_to(layout.root))
    return ScriptableOutcome(invocation, returncode, ScriptableResult(result_file, document), None)


__all__ = [
    "SCRIPTS_DIR",
    "ScriptableInvocation",
    "ScriptableOutcome",
    "ShippedScript",
    "run_scriptable",
    "script_env",
    "scriptable_base",
    "script_path",
    "scriptable_invocation",
]
