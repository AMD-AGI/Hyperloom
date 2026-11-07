# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn a materialized point YAML into the inference_suite round it runs as, and run that round.

The point YAML stays Hyperloom's internal format; this module is the one place it becomes a suite config, and
inference_suite never reads it. The translation reproduces what Magpie's generic launcher would serve for the same
point: its knobs (``TP``, ``CONC`` ...) become config fields instead of server environment, ``EXTRA_<FW>_ARGS`` and
every other variable become the user layer on the frozen base, and the launcher's own server rules (vLLM's and
ATOM's ``--max-model-len`` default, sglang's radix-cache conflict) are applied to that layer. What the YAML does
not hold -- where the round writes, its interpreter, lifecycle and telemetry -- comes in :class:`RoundSettings`.

A point runs on inference_suite (:func:`suite_route`) once its session froze a base launch and the translation
carries it. A client-only round -- multi-node, or one whose session or point names a ``BENCHMARK_BASE_URL`` -- holds
its server as ``external``: it measures the server at that URL, which something else starts and stops, and samples no
local GPU. A round with a Ray serving lease runs in the lease's actor on the GPUs Ray placed it on. The points
:func:`backend_reason` names stay on the benchmark backend. :func:`run_suite_round` writes each round into a fresh
run_dir below its round directory, so an attempt retried in the same directory never meets the run_dir before it.

An AgentX point is not translated into a measured round, but :func:`run_accuracy_suite_round` serves the launch it ran
for the D9 accuracy round: on a server of its own, without the base's golden layer, eval only, into ``<slot>/accuracy``.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import time
import urllib.parse
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from hyperloom_common.env import is_truthy
from hyperloom.common.benchmark_config import AgentxLaunch, BenchmarkPoint, is_agentx
from hyperloom.common.outside_facts import (
    ENV_CONC,
    ENV_EP,
    ENV_ISL,
    ENV_MAGPIE_EVAL_LIMIT,
    ENV_MAGPIE_EVAL_TASKS,
    ENV_MAX_MODEL_LEN,
    ENV_NUM_PROMPTS,
    ENV_NUM_WARMUPS,
    ENV_OSL,
    ENV_RANDOM_RANGE_RATIO,
    ENV_RUN_EVAL,
    ENV_TP,
)
from hyperloom.common.env_safety import scrub_benchmark_process_env
from hyperloom.inference_optimizer.framework_registry import extra_args_env
from hyperloom.inference_optimizer.session.paths import resolve_dep_dir
from hyperloom.orchestrator.state.base_launch import frozen_base
from inference_suite import EXIT_REFUSED, RoundResult, manifest_sha256, run_round
from inference_suite.argv import argv_flag_occurrences, canonical_flag_name, normalize_argv
from inference_suite.contracts import (
    DEFAULT_MAX_TOKENS_CAP,
    BaseLaunch,
    BaseRef,
    ConfigError,
    Deadlines,
    EvalConfig,
    EvalProbe,
    EvalTask,
    FixedWorkload,
    GoldenUse,
    GpuTelemetry,
    FailedStep,
    LaunchDelta,
    Lifecycle,
    LifecycleMode,
    Report,
    RunLayout,
    RunPaths,
    ServerConfig,
    SuiteConfig,
    WorkloadKind,
)
from inference_suite.frameworks import Framework, FrameworkKind, framework_facts

from ..cancel_channel import cancel_scope_listener
from ._accuracy_gate import ACCURACY_DIR
from ._benchmark_interpreter import _resolve_probe_python
from ._failed_step import failed_step_error_class
from ._gpu_power import round_gpu_telemetry
from ._multi_node_env import client_only_server_url
from ..stop_attribution import ORCHESTRATOR_CANCELLED_CLASS, SESSION_TIME_EXHAUSTED_CLASS
from ._ray_backend import SuiteLeasedRound, strip_visible_devices_from_config
from ._ray_serving import ServingLease
from ._agentx_checkpoint import AgentxCheckpoint, SuiteRoundCheckpoint
from ._subprocess_kill import (
    AGENTX_EARLY_STOP_ERROR_CLASS,
    AGENTX_EARLY_STOP_RETURNCODE,
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    AgentxCheckpointMissed,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    resolve_benchmark_timeouts,
    session_deadline_to_remaining_sec,
)

#: The host the round's server binds and its client reaches; Magpie bound every interface.
SERVER_HOST = "127.0.0.1"
#: Measured and warmup prompts are drawn with different seeds so the warmup does not prime the measured prompts.
MEASURED_SEED = 0
WARMUP_SEED = 1

_PORT_ENV = "PORT"
_GPU_MASK_ENV = "ROCR_VISIBLE_DEVICES"
_RUN_PHASE_ENV = "MAGPIE_RUN_PHASE"
_SERVER_ONLY_PHASE = "server"
_CLIENT_TRUST_ENV = "MAGPIE_TRUST_REMOTE_CODE"
_TOKENIZER_MODE_ENV = "HYPERLOOM_CLIENT_TOKENIZER_MODE"
_REMOTE_SERVER_ENV = "BENCHMARK_BASE_URL"
_PROBE_ENV = "HYPERLOOM_EVAL_PROBE"
_PROBE_MIN_SAMPLES_ENV = "HYPERLOOM_EVAL_PROBE_MIN_SAMPLES"
_PROBE_LENGTH_RATIO_ENV = "HYPERLOOM_EVAL_PROBE_LENGTH_RATIO"
_EVAL_MAX_TOKENS_ENV = "HYPERLOOM_EVAL_MAX_TOKENS"
_MODEL_PATH_ENV = "MODEL_PATH"

#: The point knobs the config carries as fields; none of them reaches the server's environment.
_CONSUMED_ENV = frozenset(
    {
        ENV_TP,
        ENV_EP,
        ENV_CONC,
        ENV_ISL,
        ENV_OSL,
        ENV_RANDOM_RANGE_RATIO,
        ENV_MAX_MODEL_LEN,
        ENV_NUM_PROMPTS,
        ENV_NUM_WARMUPS,
        ENV_RUN_EVAL,
        ENV_MAGPIE_EVAL_TASKS,
        ENV_MAGPIE_EVAL_LIMIT,
        _PORT_ENV,
        _GPU_MASK_ENV,
        _RUN_PHASE_ENV,
        _CLIENT_TRUST_ENV,
        _TOKENIZER_MODE_ENV,
        _PROBE_ENV,
        _PROBE_MIN_SAMPLES_ENV,
        _PROBE_LENGTH_RATIO_ENV,
        _EVAL_MAX_TOKENS_ENV,
        _REMOTE_SERVER_ENV,
    }
)

# Magpie's launcher defaults (scripts/benchmark/*_mi3*.sh at 06b7cf6), and the eval probe's in _inferencex_patcher.
_LAUNCHER_PORT = 8888
_HTTP_PORT = 80
_UTF8 = "utf-8"
_LAUNCHER_MAX_MODEL_LEN = 4096
_CLIENT_TOKENIZER_MODE = "auto"
_PROBE_MIN_SAMPLES = 128
_PROBE_LENGTH_RATIO = 0.75
#: The few-shot count of InferenceX's ``infx/evals/gsm8k.yaml``.
_GSM8K_FEWSHOT = 5

#: The frameworks whose Magpie launcher always passes ``--max-model-len`` (sglang's leaves the context to the host).
_LAUNCHER_CONTEXT_LENGTH = frozenset({Framework.VLLM, Framework.ATOM})
#: A flag the launcher drops from its defaults when the point's args carry the flag it cannot run beside:
#: sglang's HiCache is built on the radix cache.
_LAUNCHER_CONFLICTS: Mapping[Framework, tuple[tuple[str, str], ...]] = MappingProxyType(
    {Framework.SGLANG: (("--enable-hierarchical-cache", "--disable-radix-cache"),)}
)


@dataclass(frozen=True)
class RoundSettings:
    """What one round runs with that its point YAML does not hold.

    ``base_path`` is where the frozen base the round composes on is stored; ``silence_sec`` bounds how long the
    server may log nothing once ready; ``env_inherit`` names the host environment the server and client may see;
    ``server_url`` is the server a client-only round measures, which a point's own ``BENCHMARK_BASE_URL`` overrides.
    """

    base_path: str
    paths: RunPaths
    framework_python: str
    lifecycle: Lifecycle
    telemetry: GpuTelemetry
    silence_sec: float
    env_inherit: tuple[str, ...] = ()
    deadline_unix: int | None = None
    server_url: str | None = None


@dataclass(frozen=True)
class _Point:
    """The ``benchmark`` block of a point YAML, with its environment as strings."""

    framework: Framework
    model: str
    envs: Mapping[str, str]
    timeout_sec: float
    boot_timeout_sec: float

    def text(self, name: str) -> str | None:
        value = self.envs.get(name, "").strip()
        return value or None

    def integer(self, name: str, default: int | None = None) -> int:
        value = self.text(name)
        if value is None:
            if default is None:
                raise ConfigError(f"the point sets no {name}")
            return default
        try:
            return int(value)
        except ValueError:
            raise ConfigError(f"the point's {name} is not an integer: {value!r}") from None

    def number(self, name: str, default: float) -> float:
        value = self.text(name)
        try:
            return default if value is None else float(value)
        except ValueError:
            raise ConfigError(f"the point's {name} is not a number: {value!r}") from None

    def flag(self, name: str, default: bool) -> bool:
        return is_truthy(self.text(name), default=default)


def _bench(document: Mapping[str, object]) -> Mapping[str, Any]:
    bench = document.get("benchmark")
    if not isinstance(bench, Mapping):
        raise ConfigError("a point YAML holds a benchmark mapping")
    return bench


def _framework(bench: Mapping[str, Any]) -> Framework:
    try:
        return Framework(str(bench.get("framework") or "").strip().lower())
    except ValueError:
        raise ConfigError(f"the point's framework {bench.get('framework')!r} is not one inference_suite runs") from None


def _envs(bench: Mapping[str, Any]) -> dict[str, str]:
    return {str(name): str(value) for name, value in (bench.get("envs") or {}).items() if value is not None}


def backend_reason(document: Mapping[str, object]) -> str | None:
    """Why the point ``document`` stays on the benchmark backend, or ``None`` when :func:`suite_config` carries it.

    Raises:
        ConfigError: When ``document`` holds no benchmark mapping or names a framework inference_suite does not know.
    """
    bench = _bench(document)
    framework = _framework(bench)
    if framework_facts(framework).kind is not FrameworkKind.SERVING:
        return f"the point runs {framework.value}, which serves nothing"
    if is_agentx(bench):
        return "an AgentX point is not translated"
    if ((bench.get("profiler") or {}).get("torch_profiler") or {}).get("enabled"):
        return "a profile point is not translated"
    return None


def _point_of(bench: Mapping[str, Any], timeout: float, boot: float) -> _Point:
    return _Point(_framework(bench), str(bench.get("model") or "").strip(), _envs(bench), float(timeout), float(boot))


def _point(document: Mapping[str, object]) -> _Point:
    reason = backend_reason(document)
    if reason is not None:
        raise ConfigError(f"{reason}; it stays on the benchmark backend")
    bench = _bench(document)
    timeout = bench.get("timeout_seconds")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ConfigError(f"the point's timeout_seconds must be a positive number, got {timeout!r}")
    boot = (bench.get("server_lifecycle") or {}).get("server_ready_timeout_s") or timeout
    return _point_of(bench, timeout, boot)


def _check_base(point: _Point, base: BaseLaunch) -> None:
    if point.framework.value != base.framework:
        raise ConfigError(f"the point runs {point.framework.value}, its base runs {base.framework}")
    for name, value in ((ENV_TP, base.topology.tp), (ENV_EP, base.topology.ep)):
        if point.text(name) is not None and point.integer(name) != value:
            raise ConfigError(f"the point runs {name} {point.integer(name)}, its base runs {value}")


def _user_args(point: _Point) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """The point's server args as the user layer adds them, and the base flags the launcher would have dropped."""
    # The launcher expands the variable unquoted, so the shell splits it on whitespace and nothing else.
    extra = tuple((point.text(extra_args_env(point.framework.value)) or "").split())
    flags = {name for name, _ in argv_flag_occurrences(extra)}
    added: tuple[str, ...] = ()
    context_flag = framework_facts(point.framework).server_flags.context_length
    if point.framework in _LAUNCHER_CONTEXT_LENGTH and canonical_flag_name(context_flag) not in flags:
        added = (context_flag, str(point.integer(ENV_MAX_MODEL_LEN, _LAUNCHER_MAX_MODEL_LEN)))
    conflicts = _LAUNCHER_CONFLICTS.get(point.framework, ())
    removed = tuple(drop for need, drop in conflicts if canonical_flag_name(need) in flags)
    return (*added, *extra), removed


def _gpus(point: _Point, tp: int) -> tuple[int, ...]:
    mask = point.text(_GPU_MASK_ENV)
    if mask is None:
        return tuple(range(tp))
    try:
        return tuple(int(gpu) for gpu in mask.split(","))
    except ValueError:
        raise ConfigError(f"the point's {_GPU_MASK_ENV} is not a list of GPU indices: {mask!r}") from None


@dataclass(frozen=True)
class _Address:
    host: str
    port: int


def _external_address(url: str) -> _Address:
    """The host and port of the server at ``url``, which a round reaches over plain HTTP at its root."""
    parts = urllib.parse.urlsplit(url)
    try:
        port = parts.port or _HTTP_PORT
    except ValueError:
        raise ConfigError(f"the external server URL {url!r} names no valid port") from None
    if parts.scheme != "http" or not parts.hostname or parts.path.strip("/") or parts.query:
        raise ConfigError(f"the external server URL {url!r} is not http://<host>[:<port>]")
    return _Address(parts.hostname, port)


def _user_layer(point: _Point) -> LaunchDelta:
    args_add, args_remove = _user_args(point)
    env = {name: value for name, value in point.envs.items() if name not in _CONSUMED_ENV}
    env.pop(extra_args_env(point.framework.value), None)
    return LaunchDelta(args_add=args_add, args_remove=args_remove, env_set=env)


def _server(
    point: _Point, base: BaseLaunch, settings: RoundSettings, external: _Address | None, user: LaunchDelta
) -> ServerConfig:
    if not point.model:
        raise ConfigError("the point names no model")
    return ServerConfig(
        base=BaseRef(settings.base_path, base.digest()),
        model_path=point.model,
        served_model_name=point.model,
        host=SERVER_HOST if external is None else external.host,
        port=point.integer(_PORT_ENV, _LAUNCHER_PORT) if external is None else external.port,
        gpus=_gpus(point, base.topology.tp),
        framework_python=settings.framework_python,
        user=user,
        env_inherit=settings.env_inherit,
    )


def _fixed(point: _Point) -> FixedWorkload:
    return FixedWorkload(
        concurrency=point.integer(ENV_CONC),
        isl=point.integer(ENV_ISL),
        osl=point.integer(ENV_OSL),
        random_range_ratio=point.number(ENV_RANDOM_RANGE_RATIO, 1.0),
        num_prompts=point.integer(ENV_NUM_PROMPTS),
        num_warmups=point.integer(ENV_NUM_WARMUPS),
        seed=MEASURED_SEED,
        warmup_seed=WARMUP_SEED,
        tokenizer_mode=point.text(_TOKENIZER_MODE_ENV) or _CLIENT_TOKENIZER_MODE,
        trust_remote_code=point.flag(_CLIENT_TRUST_ENV, default=False),
    )


def _eval(point: _Point, enabled: bool) -> EvalConfig:
    task = point.text(ENV_MAGPIE_EVAL_TASKS) or EvalTask.GSM8K.value
    if task != EvalTask.GSM8K.value:
        raise ConfigError(f"the point evaluates {task!r}; a round evaluates {EvalTask.GSM8K.value} only")
    limit = point.text(ENV_MAGPIE_EVAL_LIMIT)
    return EvalConfig(
        enabled=enabled,
        num_fewshot=_GSM8K_FEWSHOT,
        concurrency=point.integer(ENV_CONC),
        limit=None if limit is None else point.integer(ENV_MAGPIE_EVAL_LIMIT),
        max_tokens_cap=point.integer(_EVAL_MAX_TOKENS_ENV, DEFAULT_MAX_TOKENS_CAP),
        probe=EvalProbe(
            enabled=point.flag(_PROBE_ENV, default=True),
            min_samples=point.integer(_PROBE_MIN_SAMPLES_ENV, _PROBE_MIN_SAMPLES),
            length_ratio=point.number(_PROBE_LENGTH_RATIO_ENV, _PROBE_LENGTH_RATIO),
        ),
    )


def suite_config(document: Mapping[str, object], base: BaseLaunch, settings: RoundSettings) -> SuiteConfig:
    """The config one round of the point ``document`` runs from, composed on ``base``.

    A point whose ``MAGPIE_RUN_PHASE`` is ``server`` only boots the server, so its round runs no load and no eval. A
    client-only round (``settings.server_url`` or the point's ``BENCHMARK_BASE_URL``) holds its server as ``external``.

    Raises:
        ConfigError: When the point is one this translation does not carry (AgentX, profile or scriptable), names
            another framework or topology than ``base``, lacks a knob its load needs, or names an external server
            that is not plain HTTP.
    """
    point = _point(document)
    _check_base(point, base)
    boot_only = (point.text(_RUN_PHASE_ENV) or "").lower() == _SERVER_ONLY_PHASE
    url = point.text(_REMOTE_SERVER_ENV) or settings.server_url
    external = None if url is None else _external_address(url)
    return SuiteConfig(
        server=_server(point, base, settings, external, _user_layer(point)),
        workload=WorkloadKind.NONE if boot_only else WorkloadKind.FIXED,
        fixed=None if boot_only else _fixed(point),
        eval=_eval(point, enabled=not boot_only and point.flag(ENV_RUN_EVAL, default=False)),
        # An external server's GPUs are not this host's to sample.
        telemetry=settings.telemetry if external is None else replace(settings.telemetry, enabled=False),
        lifecycle=settings.lifecycle if external is None else Lifecycle(LifecycleMode.EXTERNAL),
        deadlines=_deadlines(point, settings),
        paths=settings.paths,
    )


def _deadlines(point: _Point, settings: RoundSettings) -> Deadlines:
    return Deadlines(
        boot_sec=point.boot_timeout_sec,
        round_sec=point.timeout_sec,
        eval_sec=point.timeout_sec,
        silence_sec=settings.silence_sec,
        deadline_unix=settings.deadline_unix,
    )


def _flag_groups(argv: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
    """Each flag of a normalized ``argv`` by canonical name, with the values after it; a repeated flag keeps its last."""
    groups: dict[str, tuple[str, ...]] = {}
    name = ""
    for token in argv:
        if argv_flag_occurrences([token]):
            name = canonical_flag_name(token)
            groups[name] = (token,)
        elif name:
            groups[name] = (*groups[name], token)
    return groups


def _layer_over_base(launch: AgentxLaunch, base: BaseLaunch) -> LaunchDelta:
    """The layer the AgentX ``launch`` runs above ``base`` and its golden layer.

    A launch composed on the session base replaces the recipe's argv with the base's, golden's and its own layers'
    flags, and sets their environment; what differs from base plus golden is its own layer. A flag the base sets and
    the launch dropped is removed.
    """
    if not launch.replace_args:
        raise ConfigError("the AgentX point does not run the session base; its launch extends the recipe's argv")
    golden = base.golden_al
    golden_args, golden_env = (golden.args, golden.env) if golden is not None else ((), {})
    below = _flag_groups(normalize_argv((*base.args, *golden_args)))
    ran = _flag_groups(normalize_argv(launch.append_args))
    below_env = {**base.env, **golden_env}
    unset = tuple(sorted(set(launch.unset_env) & set(base.env)))
    return LaunchDelta(
        args_add=tuple(token for name, group in ran.items() if below.get(name) != group for token in group),
        args_remove=tuple(f"--{name}" for name in _flag_groups(base.args) if name not in ran),
        env_set={
            name: str(value)
            for name, value in launch.env.items()
            if name not in unset and below_env.get(name) != str(value)
        },
        env_unset=unset,
    )


def accuracy_config(
    document: Mapping[str, object], base: BaseLaunch, settings: RoundSettings, timeout_sec: float
) -> SuiteConfig:
    """The D9 accuracy round of the AgentX point ``document``: its launch on ``base`` less the golden layer, eval only.

    The round owns its server, so a client-only session, whose server is someone else's, has no accuracy round.
    ``timeout_sec`` bounds the server's boot and the eval each.

    Raises:
        ConfigError: When the point is not AgentX, does not run the session base, names another framework or topology
            than ``base``, or the session's rounds measure an external server.
    """
    bench = _bench(document)
    if not is_agentx(bench):
        raise ConfigError("an accuracy round serves an AgentX point")
    if settings.server_url is not None:
        raise ConfigError("an accuracy round owns its server, and a client-only session's server is external")
    point = _point_of(bench, timeout_sec, timeout_sec)
    point = replace(point, model=point.text(_MODEL_PATH_ENV) or point.model)
    _check_base(point, base)
    layer = _layer_over_base(BenchmarkPoint.from_block(bench).agentx_launch, base)
    server = _server(point, base, settings, None, layer)
    return SuiteConfig(
        server=replace(server, golden=GoldenUse.APPLY if base.golden_al is None else GoldenUse.EXCLUDE),
        workload=WorkloadKind.NONE,
        eval=_eval(point, enabled=True),
        telemetry=settings.telemetry,
        lifecycle=Lifecycle(LifecycleMode.OWN),
        deadlines=_deadlines(point, settings),
        paths=settings.paths,
    )


#: The tool cache ``install.sh`` stages into, below the dependency cache unless the operator names one.
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
    ``MAGPIE_RUN_PHASE=server`` point: no load and no eval. ``env_drop`` names host variables the round's server and client do not inherit.
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


def _round_point(config_path: Path, lease: ServingLease | None) -> Mapping[str, object]:
    # Ray places a leased round on GPUs of its choosing, so the host's GPU masks the point carries must not size it.
    return _read_point(config_path if lease is None else strip_visible_devices_from_config(config_path))


def _server_phase_point(document: Mapping[str, object]) -> Mapping[str, object]:
    bench = _bench(document)
    return {**document, "benchmark": {**bench, "envs": {**_envs(bench), _RUN_PHASE_ENV: _SERVER_ONLY_PHASE}}}


def _declined(config_path: Path) -> bool:
    try:
        return backend_reason(_read_point(config_path)) is not None
    except ConfigError:
        # Not declined: the round's translation refuses the point with this same error, as config_invalid.
        return False


def staged_tool_cache() -> Path:
    """Where the tools inference_suite runs are staged."""
    return resolve_dep_dir(TOOL_CACHE_DIR, TOOL_CACHE_ENV)


def suite_route(session_dir: Path | None, config_path: Path) -> SuiteRoute | None:
    """The route the point at ``config_path`` runs on inference_suite, or ``None`` while it stays on the backend."""
    if session_dir is None:
        return None
    route = session_route(Path(session_dir))
    return None if route is None or _declined(config_path) else route


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
    """The host environment a round's server and client see: what a Magpie round saw, less the GPU masks."""
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
        telemetry=round_gpu_telemetry(),
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
        point = _round_point(request.config_path, request.serving_lease)
        if request.boot_only:
            point = _server_phase_point(point)
        config = suite_config(point, route.base, settings)
    except ConfigError as exc:
        raise _refused_at(FailedStep.CONFIG_INVALID, str(exc)) from exc
    if request.warmup and config.fixed is not None:
        config = replace(config, fixed=replace(config.fixed, seed=WARMUP_SEED))
    checkpoint = None
    if request.checkpoint is not None:
        checkpoint = SuiteRoundCheckpoint(request.checkpoint, slot=request.round_dir, run_dir=Path(run_dir))
        checkpoint.clear()
    return _run_config(config, request.serving_lease, on_poll, checkpoint)


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
            _round_point(request.config_path, request.serving_lease), route.base, settings, request.timeout_sec
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
    "MEASURED_SEED",
    "RAY_ROUND_UNPLACED_CLASS",
    "RUN_DIR_PREFIX",
    "SERVER_HOST",
    "TOOL_CACHE_DIR",
    "TOOL_CACHE_ENV",
    "WARMUP_SEED",
    "accuracy_config",
    "AccuracyRound",
    "PointServer",
    "RoundSettings",
    "SuiteRound",
    "SuiteRoundRefused",
    "SuiteRoute",
    "backend_reason",
    "deadline_unix",
    "inherited_env_names",
    "point_server",
    "round_lifecycle",
    "run_accuracy_suite_round",
    "run_suite_round",
    "session_route",
    "staged_tool_cache",
    "suite_config",
    "suite_route",
]
