# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Run a point as an inference_suite round, locally or in a Ray serving lease's actor.

Every point of a session that froze a base launch runs on inference_suite (:func:`suite_route`). A client-only round --
multi-node, or one whose session or point names a ``BENCHMARK_BASE_URL`` -- holds its server as ``external``: it
measures the server at that URL, which something else starts and stops, and samples no local GPU. A round with a Ray
serving lease runs in the lease's actor on the GPUs Ray placed it on. :func:`run_suite_round` writes each round into a
fresh run_dir below its round directory, so an attempt retried in the same directory never meets the run_dir before it.

:func:`run_accuracy_suite_round` serves the launch an AgentX point ran for the D9 accuracy round: on a server of its
own, without the base's golden layer, eval only, into ``<slot>/accuracy``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType

import yaml

from hyperloom_common.env import env_flag
from hyperloom.common.env_safety import scrub_benchmark_process_env
from hyperloom.inference_optimizer.session.paths import resolve_dep_dir
from hyperloom.orchestrator.state.base_launch import frozen_base
from inference_suite import EXIT_REFUSED, RoundResult, manifest_sha256, run_round
from inference_suite.contracts import (
    BaseLaunch,
    ConfigError,
    FailedStep,
    Lifecycle,
    LifecycleMode,
    Report,
    RunLayout,
    RunPaths,
    SuiteConfig,
    Telemetry,
)

from ...cancel_channel import cancel_scope_listener
from ...stop_attribution import ORCHESTRATOR_CANCELLED_CLASS, SESSION_TIME_EXHAUSTED_CLASS
from .._accuracy_gate import ACCURACY_DIR
from .._agentx_checkpoint import AgentxCheckpoint, SuiteRoundCheckpoint
from .._benchmark_interpreter import _resolve_probe_python
from .._failed_step import failed_step_error_class
from .._gpu_pin import session_cards_mask
from .._gpu_power import round_gpu_telemetry
from .._multi_node_env import client_only_server_url
from .._ray_backend import SuiteLeasedRound, strip_visible_devices_from_config
from .._ray_serving import ServingLease
from .._subprocess_kill import (
    AGENTX_EARLY_STOP_ERROR_CLASS,
    AGENTX_EARLY_STOP_RETURNCODE,
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    AgentxCheckpointMissed,
    resolve_benchmark_timeouts,
    session_deadline_to_remaining_sec,
)
from .config import (
    _GPU_MASK_ENV,
    _RUN_PHASE_ENV,
    _SERVER_ONLY_PHASE,
    _UTF8,
    WARMUP_SEED,
    RoundSettings,
    _bench,
    _envs,
    accuracy_config,
    agentx_client_config,
    suite_config,
)

#: Escape hatch for a round that must not pay even the cost of one HTTP GET every couple of seconds.
_KV_METRICS_ENV = "INFERENCE_OPTIMIZER_KV_METRICS"


TOOL_CACHE_DIR = "inference-suite-tools"
TOOL_CACHE_ENV = "INFERENCE_SUITE_TOOL_CACHE"
RUN_DIR_PREFIX = "suite_run_"
#: The config a leased round runs from, beside its run_dir: the run_dir must be empty when the round starts.
LEASED_CONFIG_SUFFIX = ".config.yaml"
#: A leased round Ray never ran: its actor could not be placed, answered no more, or outlived its wait.
RAY_ROUND_UNPLACED_CLASS = "ray_round_unplaced"
#: How far past the round's own deadlines, session deadline included, the process running a leased round is reaped.
_LEASED_BACKSTOP_SLACK_SEC = 300.0

_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# The round masks the server's GPUs itself; a host mask inherited beneath it would renumber them.
_GPU_MASK_ENVS = frozenset({"HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES"})


class SuiteRoundRefused(ValueError):
    """The host refused a round before it ran, recorded under ``error_class``: its point does not translate, or
    nothing is staged to run it with."""

    def __init__(self, error_class: str, detail: str) -> None:
        super().__init__(f"{error_class}: {detail}")
        self.error_class = error_class
        self.detail = detail


def _refused_at(step: FailedStep, detail: str) -> SuiteRoundRefused:
    return SuiteRoundRefused(failed_step_error_class(step), detail)


#: The error classes of a leased round whose actor exited without a report, by its exit code.
_UNRUN_LEASED_CLASSES: Mapping[int, str] = MappingProxyType(
    {
        EXIT_REFUSED: failed_step_error_class(FailedStep.CONFIG_INVALID),
        ORCHESTRATOR_CANCELLED_RETURNCODE: ORCHESTRATOR_CANCELLED_CLASS,
        SESSION_TIME_EXHAUSTED_RETURNCODE: SESSION_TIME_EXHAUSTED_CLASS,
        AGENTX_EARLY_STOP_RETURNCODE: AGENTX_EARLY_STOP_ERROR_CLASS,
    }
)


@dataclass(frozen=True)
class SuiteRoute:
    """What every inference_suite round of one attempt runs on; ``manifest_sha256`` is ``None`` when nothing is staged.

    ``server_url`` is the server the attempt's rounds measure without starting it, ``None`` when they serve locally.
    """

    base: BaseLaunch
    base_path: Path
    tool_cache: Path
    manifest_sha256: str | None
    framework_python: str
    server_url: str | None = None


@dataclass(frozen=True)
class SuiteRound:
    """One round of an attempt: the point it runs, the directory it writes below, and how it holds the server.

    ``warmup`` marks a discarded round, which draws its prompts with the warmup seed so it cannot prime the measured
    round's prefix cache. ``boot_only`` marks a round that only brings the server up, running the point as a
    ``HYPERLOOM_RUN_PHASE=server`` point: no load and no eval. ``env_drop`` names host variables the round's server and client do not inherit.
    ``serving_lease`` is the Ray lease whose actor runs the round on the GPUs it holds, ``None`` to run it here.
    ``checkpoint`` stops an AgentX round that trails its reference; its ruling lands in ``round_dir``.
    """

    config_path: Path
    round_dir: Path
    lifecycle: Lifecycle
    warmup: bool = False
    deadline_unix: int | None = None
    env_drop: frozenset[str] = frozenset()
    serving_lease: ServingLease | None = None
    boot_only: bool = False
    checkpoint: AgentxCheckpoint | None = None


@dataclass(frozen=True)
class PointServer:
    """How one round of a point holds its server; ``stop_after`` is the pid_dir it takes the server down in."""

    lifecycle: Lifecycle
    stop_after: Path | None = None


def _read_point(config_path: Path) -> Mapping[str, object]:
    return yaml.safe_load(config_path.read_text(encoding=_UTF8)) or {}


def _round_point(config_path: Path, lease: ServingLease | None, cards: int) -> Mapping[str, object]:
    """The point a round runs: a leased one without GPU masks, a local one on the session's cards when it names none.

    Ray places a leased round on GPUs of its choosing, so no host mask may size it. A local server drops the masks it
    inherits and runs on its GPU ids as absolute cards, so a point without a mask (an AgentX point never carries one)
    runs on the front of the session's mask rather than on ``0..cards-1``.
    """
    if lease is not None:
        return _read_point(strip_visible_devices_from_config(config_path))
    document = _read_point(config_path)
    bench = _bench(document)
    envs = _envs(bench)
    if envs.get(_GPU_MASK_ENV, "").strip():
        return document
    return {**document, "benchmark": {**bench, "envs": {**envs, _GPU_MASK_ENV: ",".join(session_cards_mask(cards))}}}


def _server_phase_point(document: Mapping[str, object]) -> Mapping[str, object]:
    bench = _bench(document)
    return {**document, "benchmark": {**bench, "envs": {**_envs(bench), _RUN_PHASE_ENV: _SERVER_ONLY_PHASE}}}


def staged_tool_cache() -> Path:
    """Where the tools inference_suite runs are staged."""
    return resolve_dep_dir(TOOL_CACHE_DIR, TOOL_CACHE_ENV)


def suite_route(session_dir: Path | None) -> SuiteRoute:
    """What every round of the session at ``session_dir`` runs on inference_suite with.

    Raises:
        SuiteRoundRefused: ``config_invalid`` when there is no session, or it froze no base launch to run on.
    """
    route = None if session_dir is None else session_route(Path(session_dir))
    if route is None:
        raise _refused_at(FailedStep.CONFIG_INVALID, f"the session {session_dir} froze no base launch to run on")
    return route


def session_route(session_dir: Path) -> SuiteRoute | None:
    """What a round of the session at ``session_dir`` runs on inference_suite with, ``None`` when it froze no base."""
    frozen = frozen_base(session_dir)
    if frozen is None:
        return None
    tool_cache = staged_tool_cache()
    return SuiteRoute(
        base=frozen.launch,
        base_path=frozen.path,
        tool_cache=tool_cache,
        manifest_sha256=manifest_sha256(tool_cache),
        framework_python=_resolve_probe_python(frozen.launch.framework),
        server_url=client_only_server_url(),
    )


def point_server(config_path: Path, *, finds_server: bool) -> PointServer:
    """How the round of the point at ``config_path`` holds its server, read from its ``server_lifecycle`` block.

    ``finds_server`` is whether an earlier round left the server up. Without the block the round owns its server. With
    it, a round that finds the server reuses it and, when the block asks for cleanup, takes it down after; one that
    does not find it boots it for the rounds after it, or owns it when the block asks for cleanup.
    """
    block = _bench(_read_point(config_path)).get("server_lifecycle") or {}
    if not block.get("enabled"):
        return PointServer(Lifecycle(LifecycleMode.OWN))
    pid_dir, cleanup = str(block["pid_dir"]), bool(block.get("cleanup", True))
    if finds_server:
        return PointServer(Lifecycle(LifecycleMode.REUSE, pid_dir), Path(pid_dir) if cleanup else None)
    return PointServer(Lifecycle(LifecycleMode.OWN) if cleanup else Lifecycle(LifecycleMode.BOOT, pid_dir))


def round_lifecycle(round_dir: Path, mode: LifecycleMode) -> Lifecycle:
    """How the round in ``round_dir`` holds its server; a cold-start pair shares one through the slot above it."""
    return Lifecycle(mode, str(round_dir.parent) if mode in (LifecycleMode.BOOT, LifecycleMode.REUSE) else None)


def deadline_unix(session_deadline_sec: float | None) -> int | None:
    """The session deadline, held on the monotonic clock, as the wall-clock instant a round config names."""
    remaining = session_deadline_to_remaining_sec(session_deadline_sec)
    return None if remaining is None else int(time.time() + remaining)


def inherited_env_names(environ: Mapping[str, str]) -> tuple[str, ...]:
    """The host environment a round's server and client see: the scrubbed process environment, less the GPU masks."""
    scrubbed = scrub_benchmark_process_env(dict(environ))
    return tuple(sorted(name for name in scrubbed if _ENV_NAME.fullmatch(name) and name not in _GPU_MASK_ENVS))


def _next_run_dir(round_dir: Path) -> Path:
    index = 0
    while (round_dir / f"{RUN_DIR_PREFIX}{index}").exists():
        index += 1
    return round_dir / f"{RUN_DIR_PREFIX}{index}"


def _settings(
    route: SuiteRoute, run_dir: Path, lifecycle: Lifecycle, deadline: int | None, env_drop: frozenset[str] = frozenset()
) -> RoundSettings:
    if route.manifest_sha256 is None:
        raise _refused_at(FailedStep.STAGING_MISSING, f"{route.tool_cache} holds no staging manifest")
    return RoundSettings(
        base_path=str(route.base_path),
        paths=RunPaths(str(run_dir), str(route.tool_cache), route.manifest_sha256),
        framework_python=route.framework_python,
        lifecycle=lifecycle,
        telemetry=Telemetry(round_gpu_telemetry(), kv=env_flag(_KV_METRICS_ENV, default=True)),
        silence_sec=resolve_benchmark_timeouts()[0],
        env_inherit=inherited_env_names({k: v for k, v in os.environ.items() if k not in env_drop}),
        deadline_unix=deadline,
        server_url=route.server_url,
    )


def run_suite_round(route: SuiteRoute, request: SuiteRound, *, on_poll: Callable[[], None]) -> RoundResult:
    """Run ``request`` on inference_suite; ``on_poll`` is called each time the round asks whether to stop.

    The action's cancel scope, when one is published, cancels the round. A round with a serving lease runs in the
    lease's actor, which hears the cancel scope itself and is out of reach of ``on_poll``.

    Raises:
        SuiteRoundRefused: When the point does not translate, the tool cache holds no staging manifest, or the lease's
            actor ran no round.
    """
    run_dir = _next_run_dir(request.round_dir)
    settings = _settings(route, run_dir, request.lifecycle, request.deadline_unix, request.env_drop)
    try:
        point = _round_point(request.config_path, request.serving_lease, route.base.cards)
        if request.boot_only:
            point = _server_phase_point(point)
        config = suite_config(point, route.base, settings)
    except ConfigError as exc:
        raise _refused_at(FailedStep.CONFIG_INVALID, str(exc)) from exc
    if request.warmup and config.fixed is not None:
        config = replace(config, fixed=replace(config.fixed, seed=WARMUP_SEED))
    if request.warmup and config.agentx is not None:
        config = replace(config, agentx=replace(config.agentx, seed=WARMUP_SEED, warmup_seed=WARMUP_SEED))
    checkpoint = None
    if request.checkpoint is not None:
        checkpoint = SuiteRoundCheckpoint(request.checkpoint, slot=request.round_dir, run_dir=Path(run_dir))
        checkpoint.clear()
    return _run_config(config, request.serving_lease, on_poll, checkpoint)


def external_agentx_config(
    session_dir: Path, document: Mapping[str, object], server_url: str, run_dir: Path, timeout_sec: float
) -> SuiteConfig:
    """The replay of the AgentX point ``document`` against the server at ``server_url``, which serves a launch of its
    own (:func:`~.config.agentx_client_config`), on the frozen base and staged tools of the session at
    ``session_dir`` as its rounds run, written into ``run_dir`` and bounded by ``timeout_sec``.

    Raises:
        SuiteRoundRefused: When the session froze no base, nothing is staged, or the point has no such replay.
    """
    route = suite_route(session_dir)
    settings = replace(_settings(route, run_dir, Lifecycle(LifecycleMode.EXTERNAL), None), server_url=server_url)
    try:
        return agentx_client_config(document, route.base, settings, timeout_sec)
    except ConfigError as exc:
        raise _refused_at(FailedStep.CONFIG_INVALID, str(exc)) from exc


@dataclass(frozen=True)
class AccuracyRound:
    """The D9 accuracy round of one slot: the AgentX point the slot ran, and the slot it writes ``accuracy/`` into.

    ``timeout_sec`` bounds the round's boot and its eval each; ``serving_lease`` is the Ray lease whose actor runs it.
    """

    config_path: Path
    slot: Path
    timeout_sec: float
    deadline_unix: int | None = None
    serving_lease: ServingLease | None = None


def run_accuracy_suite_round(route: SuiteRoute, request: AccuracyRound, *, on_poll: Callable[[], None]) -> RoundResult:
    """Run the accuracy round ``request`` names into ``<slot>/accuracy``, as :func:`run_suite_round` runs a round.

    Readers find the round at that fixed path, so a slot run again replaces the round an earlier attempt left there.

    Raises:
        SuiteRoundRefused: When the point has no accuracy round (:func:`accuracy_config`), the tool cache holds no
            staging manifest, or the lease's actor ran no round.
    """
    run_dir = request.slot / ACCURACY_DIR
    if run_dir.exists():
        shutil.rmtree(run_dir)
    settings = _settings(route, run_dir, Lifecycle(LifecycleMode.OWN), request.deadline_unix)
    try:
        config = accuracy_config(
            _round_point(request.config_path, request.serving_lease, route.base.cards),
            route.base,
            settings,
            request.timeout_sec,
        )
    except ConfigError as exc:
        raise _refused_at(FailedStep.CONFIG_INVALID, str(exc)) from exc
    return _run_config(config, request.serving_lease, on_poll)


def _run_config(
    config: SuiteConfig,
    lease: ServingLease | None,
    on_poll: Callable[[], None],
    checkpoint: SuiteRoundCheckpoint | None = None,
) -> RoundResult:
    if lease is not None:
        return _run_leased(config, lease, checkpoint)
    judge = None if checkpoint is None else checkpoint.judge()
    stopped_by_checkpoint = ""
    with cancel_scope_listener() as scope:

        def cancel() -> str:
            nonlocal stopped_by_checkpoint
            on_poll()
            if judge is not None and not stopped_by_checkpoint:
                try:
                    judge.tick(time.monotonic())
                except AgentxCheckpointMissed as exc:
                    stopped_by_checkpoint = str(exc)
            return stopped_by_checkpoint or ("" if scope is None else scope.reason)

        return run_round(config, cancel=cancel)


def _leased_round(
    config: SuiteConfig, config_path: Path, checkpoint: SuiteRoundCheckpoint | None = None
) -> SuiteLeasedRound:
    deadlines = config.deadlines
    remaining = (
        None if deadlines.deadline_unix is None else deadlines.deadline_unix - time.time() + _LEASED_BACKSTOP_SLACK_SEC
    )
    return SuiteLeasedRound(
        config_path=str(config_path),
        run_dir=config.paths.run_dir,
        env=dict(os.environ),
        timeout=deadlines.boot_sec + deadlines.round_sec + deadlines.eval_sec + _LEASED_BACKSTOP_SLACK_SEC,
        session_remaining_sec=remaining,
        checkpoint=checkpoint,
    )


def _run_leased(config: SuiteConfig, lease: ServingLease, checkpoint: SuiteRoundCheckpoint | None) -> RoundResult:
    """Run ``config`` in ``lease``'s actor, which holds the round's GPUs in Ray, and read back the report it wrote.

    Raises:
        SuiteRoundRefused: When the actor ran no round: the CLI refused its config, a cancel came before it started,
            or Ray could not place it; or when a stop killed the CLI before it wrote its report.
    """
    run_dir = Path(config.paths.run_dir)
    config_path = run_dir.with_name(f"{run_dir.name}{LEASED_CONFIG_SUFFIX}")
    config_path.write_text(yaml.safe_dump(config.to_document(), sort_keys=False), encoding=_UTF8)
    try:
        returncode, _stdout, stderr = lease.run_session_kill(_leased_round(config, config_path, checkpoint))
    except subprocess.TimeoutExpired as exc:
        raise SuiteRoundRefused(RAY_ROUND_UNPLACED_CLASS, str(exc)) from exc
    layout = RunLayout(run_dir)
    if not layout.report.is_file():
        error_class = _UNRUN_LEASED_CLASSES.get(returncode, RAY_ROUND_UNPLACED_CLASS)
        raise SuiteRoundRefused(error_class, stderr.strip() or f"the leased round exited {returncode} with no report")
    return RoundResult(Report.from_document(json.loads(layout.report.read_text(encoding=_UTF8))), layout)


__all__ = [
    "LEASED_CONFIG_SUFFIX",
    "RAY_ROUND_UNPLACED_CLASS",
    "RUN_DIR_PREFIX",
    "TOOL_CACHE_DIR",
    "TOOL_CACHE_ENV",
    "AccuracyRound",
    "PointServer",
    "SuiteRound",
    "SuiteRoundRefused",
    "SuiteRoute",
    "deadline_unix",
    "external_agentx_config",
    "inherited_env_names",
    "point_server",
    "round_lifecycle",
    "run_accuracy_suite_round",
    "run_suite_round",
    "staged_tool_cache",
    "session_route",
    "suite_route",
]
