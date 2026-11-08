# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Turn a materialized point YAML into the inference_suite round it runs as.

The point YAML stays Hyperloom's internal format; this module is the one place it becomes a suite config, and
inference_suite never reads it. The point's knobs (``TP``, ``CONC`` ...) become config fields instead of server
environment, ``EXTRA_<FW>_ARGS`` and every other variable become the user layer on the frozen base, and the server
rules a launch needs (vLLM's and ATOM's ``--max-model-len`` default, sglang's radix-cache conflict) are applied to
that layer. What the YAML does
not hold -- where the round writes, its interpreter, lifecycle and telemetry -- comes in :class:`RoundSettings`.

An AgentX point runs the agentx-harness replay on the launch its ``agentx.launch_overrides`` composed on the session
base, above the base's golden layer; a profile point turns the torch profiler on for the window it asks for. An MLPerf
point (``benchmark_script`` names the MLPerf agentic client) runs the MLPerf harness against the round's own server,
which serves under the model name the harness requests.

A scriptable point (xDiT, or a custom session's own benchmark and eval scripts) runs its session base's script once
with the point's environment (:func:`scriptable_workload`); the quality gate the script writes is its eval, so it
runs no GSM8K.

:func:`accuracy_config` builds the D9 accuracy round of an AgentX point. :mod:`.rounds` runs the configs built here.
"""

from __future__ import annotations

import json
import urllib.parse
from collections.abc import Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

from hyperloom_common.env import is_truthy
from hyperloom.common.agentx_workload import (
    CANON_WARMUP_PER_LANE,
    is_agentx_client_key,
    is_agentx_client_script,
)
from hyperloom.common.benchmark_config import AgentxLaunch, BenchmarkPoint, is_agentx
from hyperloom.common.gain_math import AGENTX_ERROR_RATE_THRESHOLD_PCT
from hyperloom.common.outside_facts import (
    ENV_CONC,
    ENV_EP,
    ENV_ISL,
    ENV_EVAL_LIMIT,
    ENV_EVAL_ONLY,
    ENV_EVAL_TASKS,
    ENV_MAX_MODEL_LEN,
    ENV_NUM_PROMPTS,
    ENV_NUM_WARMUPS,
    ENV_OSL,
    ENV_RANDOM_RANGE_RATIO,
    ENV_RUN_EVAL,
    ENV_TP,
)
from hyperloom.inference_optimizer.framework_registry import extra_args_env
from inference_suite.argv import argv_flag_occurrences, canonical_flag_name, normalize_argv, remove_server_args
from inference_suite.contracts import (
    AGENTX_CLIENT_ENV_PREFIX,
    AGENTX_DEFAULT_DURATION_SEC,
    AGENTX_FAST_MODE_ENV,
    DEFAULT_MAX_TOKENS_CAP,
    AgentxWorkload,
    BaseLaunch,
    BaseRef,
    ConfigError,
    Deadlines,
    EvalConfig,
    EvalProbe,
    EvalTask,
    FixedWorkload,
    GoldenUse,
    LaunchDelta,
    Lifecycle,
    LifecycleMode,
    MlperfWorkload,
    ProfileConfig,
    RunPaths,
    ScriptableRunEnv,
    ScriptableWorkload,
    ServerConfig,
    SuiteConfig,
    Telemetry,
    WorkloadKind,
)
from inference_suite.frameworks import Framework, FrameworkKind, framework_facts

from .._gpu_pin import ROCR_AUTOFILL_ENV
from .._workload_envs import QUALITY_REF_ENVS
from .._profile_setup import (
    _DELAY_ITERATIONS,
    _MAX_ITERATIONS,
    SGLANG_NUM_STEPS,
    SGLANG_START_STEP,
    _profiler_flag_value,
    agentx_profile,
    profiled_server,
    refuse_agentx_keys,
    torch_profiler,
)

#: The host the round's server binds and its client reaches.
SERVER_HOST = "127.0.0.1"
#: Measured and warmup prompts are drawn with different seeds so the warmup does not prime the measured prompts.
MEASURED_SEED = 0
WARMUP_SEED = 1

_PORT_ENV = "PORT"
_GPU_MASK_ENV = "ROCR_VISIBLE_DEVICES"
_RUN_PHASE_ENV = "HYPERLOOM_RUN_PHASE"
_SERVER_ONLY_PHASE = "server"
_CLIENT_TRUST_ENV = "BENCH_TRUST_REMOTE_CODE"
_TOKENIZER_MODE_ENV = "HYPERLOOM_CLIENT_TOKENIZER_MODE"
_REMOTE_SERVER_ENV = "BENCHMARK_BASE_URL"
_PROBE_ENV = "HYPERLOOM_EVAL_PROBE"
_PROBE_MIN_SAMPLES_ENV = "HYPERLOOM_EVAL_PROBE_MIN_SAMPLES"
_PROBE_LENGTH_RATIO_ENV = "HYPERLOOM_EVAL_PROBE_LENGTH_RATIO"
_EVAL_MAX_TOKENS_ENV = "HYPERLOOM_EVAL_MAX_TOKENS"
_MODEL_PATH_ENV = "MODEL_PATH"
_PROFILE_ENV = "PROFILE"
_PROFILE_BODY_ENV = "PROFILE_EXTRA_BODY"

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
        ENV_EVAL_ONLY,
        ENV_EVAL_TASKS,
        ENV_EVAL_LIMIT,
        _PORT_ENV,
        _GPU_MASK_ENV,
        ROCR_AUTOFILL_ENV,
        _RUN_PHASE_ENV,
        _CLIENT_TRUST_ENV,
        _TOKENIZER_MODE_ENV,
        _PROBE_ENV,
        _PROBE_MIN_SAMPLES_ENV,
        _PROBE_LENGTH_RATIO_ENV,
        _EVAL_MAX_TOKENS_ENV,
        _REMOTE_SERVER_ENV,
        _PROFILE_ENV,
        _PROFILE_BODY_ENV,
    }
)

# The generic launcher's defaults (``inference_suite/recipe/defaults/table.yaml``) and the eval probe's.
_LAUNCHER_PORT = 8888
_HTTP_PORT = 80
_UTF8 = "utf-8"
_LAUNCHER_MAX_MODEL_LEN = 4096
_CLIENT_TOKENIZER_MODE = "auto"
_PROBE_MIN_SAMPLES = 128
_PROBE_LENGTH_RATIO = 0.75
#: The few-shot count of InferenceX's ``infx/evals/gsm8k.yaml``.
_GSM8K_FEWSHOT = 5

#: The frameworks whose generic launcher always passes ``--max-model-len`` (sglang's leaves the context to the host).
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
    telemetry: Telemetry
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


def _point_of(bench: Mapping[str, Any], timeout: float, boot: float) -> _Point:
    return _Point(_framework(bench), str(bench.get("model") or "").strip(), _envs(bench), float(timeout), float(boot))


def _point(document: Mapping[str, object]) -> _Point:
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


def _gpus(point: _Point, cards: int) -> tuple[int, ...]:
    """The first ``cards`` GPUs of the point's mask: the mask may be the whole session's, not the round's."""
    mask = point.text(_GPU_MASK_ENV)
    try:
        gpus = tuple(range(cards)) if mask is None else tuple(int(gpu) for gpu in mask.split(","))
    except ValueError:
        raise ConfigError(f"the round's {_GPU_MASK_ENV} is not a list of GPU indices: {mask!r}") from None
    if len(gpus) < cards:
        raise ConfigError(f"the round's {_GPU_MASK_ENV}={mask!r} holds {len(gpus)} GPUs; the launch needs {cards}")
    return gpus[:cards]


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
        gpus=_gpus(point, base.cards),
        framework_python=settings.framework_python,
        user=user,
        env_inherit=settings.env_inherit,
    )


def fixed_workload(document: Mapping[str, object]) -> FixedWorkload:
    """The fixed load a round of the point ``document`` sends its server."""
    return _fixed(_point_of(_bench(document), 0.0, 0.0))


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
    task = point.text(ENV_EVAL_TASKS) or EvalTask.GSM8K.value
    if task != EvalTask.GSM8K.value:
        raise ConfigError(f"the point evaluates {task!r}; a round evaluates {EvalTask.GSM8K.value} only")
    limit = point.text(ENV_EVAL_LIMIT)
    return EvalConfig(
        enabled=enabled,
        num_fewshot=_GSM8K_FEWSHOT,
        concurrency=point.integer(ENV_CONC),
        limit=None if limit is None else point.integer(ENV_EVAL_LIMIT),
        max_tokens_cap=point.integer(_EVAL_MAX_TOKENS_ENV, DEFAULT_MAX_TOKENS_CAP),
        probe=EvalProbe(
            enabled=point.flag(_PROBE_ENV, default=True),
            min_samples=point.integer(_PROBE_MIN_SAMPLES_ENV, _PROBE_MIN_SAMPLES),
            length_ratio=point.number(_PROBE_LENGTH_RATIO_ENV, _PROBE_LENGTH_RATIO),
        ),
    )


def _start_body(point: _Point) -> dict[str, Any]:
    text = point.text(_PROFILE_BODY_ENV) or "{}"
    try:
        body = json.loads(text)
    except ValueError:
        raise ConfigError(f"the point's {_PROFILE_BODY_ENV} is not JSON: {text!r}") from None
    if not isinstance(body, dict):
        raise ConfigError(f"the point's {_PROFILE_BODY_ENV} is not a JSON object: {text!r}")
    return body


def _steps(value: object, what: str) -> int:
    try:
        return int(str(value))
    except ValueError:
        raise ConfigError(f"the profile point's {what} is not a step count: {value!r}") from None


def _fixed_profile(bench: Mapping[str, Any], point: _Point) -> ProfileConfig:
    """The capture window of a fixed point: the steps SGLang's start body or vLLM's profiler flags bound it to.

    ATOM bounds no steps, so its capture is the profiled pass, which the window's step fields then count in prompts.
    """
    if not torch_profiler(bench).get("enabled"):
        if point.flag(_PROFILE_ENV, default=False):
            raise ConfigError(f"the point sets {_PROFILE_ENV} without profiler.torch_profiler.enabled")
        return ProfileConfig(enabled=False)
    body = _start_body(point)
    window = (body.pop(SGLANG_START_STEP, None), body.pop(SGLANG_NUM_STEPS, None))
    num_prompts = point.integer(ENV_NUM_PROMPTS)
    if point.framework is Framework.VLLM:
        args = point.text(extra_args_env(point.framework.value)) or ""
        window = (_profiler_flag_value(args, _DELAY_ITERATIONS), _profiler_flag_value(args, _MAX_ITERATIONS))
    elif point.framework is Framework.ATOM:
        window = (0, num_prompts)
    if None in window:
        raise ConfigError("the profile point bounds no capture window")
    return ProfileConfig(
        enabled=True,
        start_body=body or None,
        num_prompts=num_prompts,
        delay_iters=_steps(window[0], "capture delay"),
        max_iters=_steps(window[1], "captured steps"),
    )


#: The ``agentx`` keys a point may set; a recipe selection (``recipe``, ``selector`` ...) gives way to the base.
_AGENTX_BLOCK_KEYS = frozenset({"enabled", "mode", "concurrency", "failed_request_threshold", "launch_overrides"})
#: Every ``launch_overrides`` key a point may set; ``remove_args`` edits the recipe argv a base-seeded launch replaces.
_AGENTX_LAUNCH_KEYS = frozenset(
    {"version", "append_args", "remove_args", "replace_args", "env", "unset_env", "executable"}
    | {"source_files", "absent_source_files"}
)
_AGENTX_CANONICAL = "canonical"
_AGENTX_THRESHOLD = AGENTX_ERROR_RATE_THRESHOLD_PCT / 100
_TRACE_ENV = "WEKA_LOADER_OVERRIDE"
_DURATION_ENV = "DURATION"
_EP_SIZE_ENV = "EP_SIZE"
_WARMUP_PER_LANE_ENV = "AIPERF_WARMUP_REQUESTS_PER_LANE"
#: The AgentX point's environment the config carries as fields, beside a fixed point's and the client's ``AIPERF_*``.
_AGENTX_CONSUMED_ENV = _CONSUMED_ENV | {_MODEL_PATH_ENV, _EP_SIZE_ENV, _TRACE_ENV, _DURATION_ENV}
#: The client knobs of the retired AgentX replay and the MLPerf client, which the agentx-harness replay does not read.
_FOREIGN_CLIENT_PREFIXES = ("AGENTX_", "AGENTIC_")


def _agentx_block(bench: Mapping[str, Any]) -> Mapping[str, Any]:
    block = bench.get("agentx")
    return block if isinstance(block, Mapping) else {}


def _check_agentx_block(block: Mapping[str, Any], point: _Point) -> None:
    refuse_agentx_keys(block, _AGENTX_BLOCK_KEYS, "agentx")
    refuse_agentx_keys(block.get("launch_overrides"), _AGENTX_LAUNCH_KEYS, "agentx.launch_overrides")
    if str(block.get("mode", _AGENTX_CANONICAL)).lower() != _AGENTX_CANONICAL:
        raise ConfigError("the AgentX point asks for fast mode; a round replays canonically")
    threshold = block.get("failed_request_threshold", _AGENTX_THRESHOLD)
    if threshold != _AGENTX_THRESHOLD:
        raise ConfigError(
            f"the AgentX point's failed_request_threshold {threshold!r} is not Hyperloom's {_AGENTX_THRESHOLD}"
        )
    concurrency = block.get("concurrency", point.integer(ENV_CONC))
    if concurrency != point.integer(ENV_CONC):
        raise ConfigError(f"the AgentX point's concurrency {concurrency!r} is not its {ENV_CONC}")


def _check_agentx_envs(point: _Point, base: BaseLaunch) -> None:
    refused = sorted(name for name in point.envs if name.startswith(_FOREIGN_CLIENT_PREFIXES))
    if refused:
        raise ConfigError(f"the AgentX point sets {', '.join(refused)}; the agentx-harness replay has its own knobs")
    if point.flag(ENV_RUN_EVAL, default=False):
        raise ConfigError(f"the AgentX point sets {ENV_RUN_EVAL}; its accuracy is a round of its own")
    if point.text(_EP_SIZE_ENV) is not None and point.integer(_EP_SIZE_ENV) != base.topology.ep:
        raise ConfigError(
            f"the point runs {_EP_SIZE_ENV} {point.integer(_EP_SIZE_ENV)}, its base runs {base.topology.ep}"
        )


def _aiperf_env(env: Mapping[str, str]) -> dict[str, str]:
    return {name: str(value) for name, value in env.items() if name.startswith(AGENTX_CLIENT_ENV_PREFIX)}


def _lane_warmup(client_env: Mapping[str, str]) -> int:
    value = client_env.get(_WARMUP_PER_LANE_ENV, "").strip()
    try:
        return int(value) if value else CANON_WARMUP_PER_LANE
    except ValueError:
        raise ConfigError(f"the AgentX client's {_WARMUP_PER_LANE_ENV} is not an integer: {value!r}") from None


def _agentx_trace(point: _Point, base: BaseLaunch) -> str:
    trace = point.text(_TRACE_ENV) or base.client.env.get(_TRACE_ENV)
    if trace is None:
        raise ConfigError(f"neither the AgentX point nor its base's client env names the trace corpus ({_TRACE_ENV})")
    return trace


def _agentx_workload(point: _Point, base: BaseLaunch) -> AgentxWorkload:
    """The replay: the base's client environment under the point's, every knob the harness takes explicit."""
    client_env = {AGENTX_FAST_MODE_ENV: "0", **_aiperf_env(base.client.env), **_aiperf_env(point.envs)}
    return AgentxWorkload(
        concurrency=point.integer(ENV_CONC),
        dataset=_agentx_trace(point, base),
        duration_sec=point.integer(_DURATION_ENV, AGENTX_DEFAULT_DURATION_SEC),
        warmup_requests_per_lane=_lane_warmup(client_env),
        seed=MEASURED_SEED,
        warmup_seed=MEASURED_SEED,
        failed_request_threshold=_AGENTX_THRESHOLD,
        client_env=client_env,
    )


def _agentx_server_env(point: _Point) -> dict[str, str]:
    """The point's environment, layered under the launch's: less what the config carries as fields."""
    return {
        name: value
        for name, value in point.envs.items()
        if name not in _AGENTX_CONSUMED_ENV and not name.startswith(AGENTX_CLIENT_ENV_PREFIX)
    }


def _agentx_server(point: _Point, bench: Mapping[str, Any], base: BaseLaunch, settings: RoundSettings) -> ServerConfig:
    """The AgentX point's server: its launch over the base, on the interpreter and source files the launch names."""
    launch = BenchmarkPoint.from_block(bench).agentx_launch
    layer = _layer_over_base(launch, base, _agentx_server_env(point))
    server = _server(point, base, settings, None, layer)
    overrides = _agentx_block(bench).get("launch_overrides") or {}
    return replace(
        server,
        framework_python=launch.executable or server.framework_python,
        source_files={str(path): str(digest) for path, digest in launch.source_files.items()},
        absent_source_files=tuple(str(path) for path in overrides.get("absent_source_files") or ()),
    )


def _agentx_config(
    bench: Mapping[str, Any], point: _Point, base: BaseLaunch, settings: RoundSettings, boot_only: bool
) -> SuiteConfig:
    _check_agentx_block(_agentx_block(bench), point)
    _check_agentx_envs(point, base)
    point = replace(point, model=point.text(_MODEL_PATH_ENV) or point.model)
    measures = not boot_only
    server = _agentx_server(point, bench, base, settings)
    return SuiteConfig(
        server=profiled_server(server, bench, base) if measures else server,
        workload=WorkloadKind.AGENTX if measures else WorkloadKind.NONE,
        agentx=_agentx_workload(point, base) if measures else None,
        profile=agentx_profile(bench, point.framework) if measures else ProfileConfig(enabled=False),
        eval=_eval(point, enabled=False),
        telemetry=settings.telemetry,
        lifecycle=settings.lifecycle,
        deadlines=_deadlines(point, settings),
        paths=settings.paths,
    )


#: The point's environment the MLPerf client reads, each a :class:`MlperfWorkload` field the AgentX switch settles.
_MLPERF_ENV = MappingProxyType(
    {
        "flow": "MLPERF_AGENTIC_FLOW",
        "model": "MLPERF_AGENTIC_MODEL",
        "hardware": "MLPERF_AGENTIC_HARDWARE",
        "dataset": "AGENTIC_DATASET_PATH",
        "tokenizer": "MLPERF_TOKENIZER_DIR",
        "endpoints_dir": "MLPERF_ENDPOINTS_DIR",
    }
)
_MLPERF_TRAJECTORIES_ENV = "AGENTIC_NUM_TRAJECTORIES"
_MLPERF_CONCURRENCY_ENV = "AGENTIC_CONCURRENCY"
#: What the AgentX switch writes for the MLPerf client alone, beside its ``MLPERF_*`` and ``AGENTIC_*`` knobs.
_MLPERF_CLIENT_ENV = frozenset({"MODEL", "FRAMEWORK"})
_SERVED_MODEL_FLAG = "--served-model-name"
#: An agentic turn idles for minutes between tool results, so a server defaulting to a short keep-alive drops the
#: connection mid-trajectory.
_KEEP_ALIVE_SEC = "900"
_KEEP_ALIVE_ENV: Mapping[Framework, str] = MappingProxyType(
    {Framework.VLLM: "VLLM_HTTP_TIMEOUT_KEEP_ALIVE", Framework.SGLANG: "SGLANG_TIMEOUT_KEEP_ALIVE"}
)


def _is_mlperf(bench: Mapping[str, Any]) -> bool:
    return is_agentx_client_script(str(bench.get("benchmark_script") or ""))


def _mlperf_workload(point: _Point) -> MlperfWorkload:
    knobs = {field_name: point.text(name) for field_name, name in _MLPERF_ENV.items()}
    unset = sorted(_MLPERF_ENV[field_name] for field_name, value in knobs.items() if value is None)
    if unset:
        raise ConfigError(f"the MLPerf point sets no {', '.join(unset)}")
    concurrency = point.integer(ENV_CONC)
    if point.integer(_MLPERF_CONCURRENCY_ENV, concurrency) != concurrency:
        raise ConfigError(f"the MLPerf point's {_MLPERF_CONCURRENCY_ENV} is not its {ENV_CONC} {concurrency}")
    return MlperfWorkload(concurrency=concurrency, trajectories=point.integer(_MLPERF_TRAJECTORIES_ENV), **knobs)


def _mlperf_server_point(point: _Point) -> _Point:
    """The point as its server reads it: no client knob, no served name of its own, an agentic turn's keep-alive.

    The round's config names the served model, and the AgentX switch's ``--served-model-name`` stays in the point for
    GEAK's handoff, which launches the builtin server script.
    """
    envs = {
        name: value
        for name, value in point.envs.items()
        if name not in _MLPERF_CLIENT_ENV and not is_agentx_client_key(name, point.envs)
    }
    extra = extra_args_env(point.framework.value)
    envs[extra] = remove_server_args(envs.get(extra, ""), [_SERVED_MODEL_FLAG])
    keep_alive = _KEEP_ALIVE_ENV.get(point.framework)
    if keep_alive is not None:
        envs.setdefault(keep_alive, _KEEP_ALIVE_SEC)
    return replace(point, envs=envs)


def _mlperf_config(
    bench: Mapping[str, Any], point: _Point, base: BaseLaunch, settings: RoundSettings, boot_only: bool
) -> SuiteConfig:
    """The MLPerf point's round: the harness against a server of the round's own, unprofiled and without eval.

    A boot-only round serves the harness's model name too, so the measured round that reuses its server composes the
    same launch.
    """
    if torch_profiler(bench).get("enabled"):
        raise ConfigError("the MLPerf point asks for a torch profile; an MLPerf round is not profiled")
    if point.flag(ENV_RUN_EVAL, default=False):
        raise ConfigError(f"the MLPerf point sets {ENV_RUN_EVAL}; the harness scores its accuracy inline")
    workload = None if boot_only else _mlperf_workload(point)
    served_model = point.text(_MLPERF_ENV["model"])
    if served_model is None:
        raise ConfigError(f"the MLPerf point sets no {_MLPERF_ENV['model']}")
    server_point = _mlperf_server_point(point)
    server = _server(server_point, base, settings, None, _user_layer(server_point))
    return SuiteConfig(
        server=replace(server, served_model_name=served_model),
        workload=WorkloadKind.NONE if workload is None else WorkloadKind.MLPERF,
        mlperf=workload,
        eval=_eval(point, enabled=False),
        telemetry=settings.telemetry,
        lifecycle=settings.lifecycle,
        deadlines=_deadlines(point, settings),
        paths=settings.paths,
    )


#: The ``benchmark`` keys a script reads from its environment, by the variable each fills.
_SCRIPT_BLOCK_ENV = MappingProxyType({"RUNNER_TYPE": "runner_type", "PRECISION": "precision"})
_SCRIPT_ROUND_SET = frozenset(item.value for item in ScriptableRunEnv)


def scriptable_workload(bench: Mapping[str, Any], envs: Mapping[str, str]) -> ScriptableWorkload:
    """The script's run of the point whose ``benchmark`` block is ``bench`` and whose environment is ``envs``.

    The script sees every variable of the point, and ``RUNNER_TYPE`` and ``PRECISION`` from its block where the
    environment names neither, less the names the round sets itself (the result's place, the model, the GPUs, the
    profiler's switch). The quality reference is the point's ``HYPERLOOM_QUALITY_REF`` (or ``XDIT_QUALITY_REF``),
    which stays in the environment for scripts that read either name.

    Raises:
        ConfigError: When a variable the point sets reads as an acceptance threshold, which the caller judges (D2).
    """
    block = {name: str(bench[key]) for name, key in _SCRIPT_BLOCK_ENV.items() if bench.get(key)}
    params = {name: value for name, value in {**block, **envs}.items() if name not in _SCRIPT_ROUND_SET}
    reference = next((envs[name] for name in QUALITY_REF_ENVS if envs.get(name, "").strip()), None)
    return ScriptableWorkload(params=params, quality_reference=reference)


def _scriptable_config(
    bench: Mapping[str, Any], point: _Point, base: BaseLaunch, settings: RoundSettings, boot_only: bool
) -> SuiteConfig:
    """The scriptable point's round: its base's script, once, on a server-less round of its own."""
    if boot_only or settings.lifecycle.mode is not LifecycleMode.OWN:
        raise ConfigError("a scriptable point runs no server, so its round neither boots one nor reuses one")
    return SuiteConfig(
        server=_server(point, base, settings, None, LaunchDelta()),
        workload=WorkloadKind.SCRIPTABLE,
        scriptable=scriptable_workload(bench, point.envs),
        profile=ProfileConfig(enabled=bool(torch_profiler(bench).get("enabled"))),
        eval=_eval(point, enabled=False),
        telemetry=settings.telemetry,
        lifecycle=settings.lifecycle,
        deadlines=_deadlines(point, settings),
        paths=settings.paths,
    )


def suite_config(document: Mapping[str, object], base: BaseLaunch, settings: RoundSettings) -> SuiteConfig:
    """The config one round of the point ``document`` runs from, composed on ``base``.

    A point whose ``HYPERLOOM_RUN_PHASE`` is ``server`` only boots the server, so its round runs no load and no eval; one
    that sets ``EVAL_ONLY`` runs the eval and no load, as InferenceX's ``run_benchmark_serving`` does. A client-only round (``settings.server_url`` or the point's ``BENCHMARK_BASE_URL``) holds its server as ``external``.

    Raises:
        ConfigError: When the point is one this translation does not carry (an AgentX, MLPerf or profile setting no
            round has), names another framework or topology than ``base``, lacks a knob its load needs, or names an
            external server that is not plain HTTP or serves an AgentX, MLPerf or scriptable point.
    """
    point = _point(document)
    _check_base(point, base)
    bench = _bench(document)
    boot_only = (point.text(_RUN_PHASE_ENV) or "").lower() == _SERVER_ONLY_PHASE
    url = point.text(_REMOTE_SERVER_ENV) or settings.server_url
    scriptable = framework_facts(point.framework).kind is FrameworkKind.SCRIPTABLE
    if url is not None and (is_agentx(bench) or _is_mlperf(bench) or scriptable):
        raise ConfigError(
            f"an AgentX, MLPerf or scriptable round serves its own launch, not the external server at {url}"
        )
    if scriptable:
        return _scriptable_config(bench, point, base, settings, boot_only)
    if is_agentx(bench):
        return _agentx_config(bench, point, base, settings, boot_only)
    if _is_mlperf(bench):
        return _mlperf_config(bench, point, base, settings, boot_only)
    external = None if url is None else _external_address(url)
    unloaded = boot_only or point.flag(ENV_EVAL_ONLY, default=False)
    return SuiteConfig(
        server=_server(point, base, settings, external, _user_layer(point)),
        workload=WorkloadKind.NONE if unloaded else WorkloadKind.FIXED,
        fixed=None if unloaded else _fixed(point),
        profile=ProfileConfig(enabled=False) if unloaded else _fixed_profile(bench, point),
        eval=_eval(point, enabled=not boot_only and point.flag(ENV_RUN_EVAL, default=False)),
        # An external server's GPUs are not this host's to sample.
        telemetry=settings.telemetry
        if external is None
        else replace(settings.telemetry, gpu=replace(settings.telemetry.gpu, enabled=False)),
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


def _layer_over_base(launch: AgentxLaunch, base: BaseLaunch, point_env: Mapping[str, str]) -> LaunchDelta:
    """The layer the AgentX ``launch`` runs above ``base`` and its golden layer.

    A launch composed on the session base replaces the recipe's argv with the base's, golden's and its own layers'
    flags, and sets their environment over ``point_env``, the point's own; what differs from base plus golden is its
    own layer. A flag the base sets and the launch dropped is removed, and so is a variable the launch unsets.
    """
    if not launch.replace_args:
        raise ConfigError("the AgentX point does not run the session base; its launch extends the recipe's argv")
    golden = base.golden_al
    golden_args, golden_env = (golden.args, golden.env) if golden is not None else ((), {})
    below = _flag_groups(normalize_argv((*base.args, *golden_args)))
    ran = _flag_groups(normalize_argv(launch.append_args))
    below_env = {**base.env, **golden_env}
    env = {**point_env, **{name: str(value) for name, value in launch.env.items()}}
    return LaunchDelta(
        args_add=tuple(token for name, group in ran.items() if below.get(name) != group for token in group),
        args_remove=tuple(f"--{name}" for name in _flag_groups(base.args) if name not in ran),
        env_set={
            name: value for name, value in env.items() if name not in launch.unset_env and below_env.get(name) != value
        },
        env_unset=tuple(sorted(set(launch.unset_env) & set(base.env))),
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
    server = _agentx_server(point, bench, base, settings)
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


__all__ = [
    "MEASURED_SEED",
    "SERVER_HOST",
    "WARMUP_SEED",
    "accuracy_config",
    "RoundSettings",
    "fixed_workload",
    "suite_config",
    "scriptable_workload",
]
