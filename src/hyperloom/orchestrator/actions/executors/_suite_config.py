# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Translate a materialized point YAML into the ``SuiteConfig`` one inference_suite round runs from.

The point YAML stays Hyperloom's internal format; this module is the one place it becomes a suite config, and
inference_suite never reads it. The translation reproduces what Magpie's generic launcher would serve for the same
point: its knobs (``TP``, ``CONC`` ...) become config fields instead of server environment, ``EXTRA_<FW>_ARGS`` and
every other variable become the user layer on the frozen base, and the launcher's own server rules (vLLM's and
ATOM's ``--max-model-len`` default, sglang's radix-cache conflict) are applied to that layer. What the YAML does
not hold -- where the round writes, its interpreter, lifecycle and telemetry -- comes in :class:`RoundSettings`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from hyperloom_common.env import is_truthy
from hyperloom.common.benchmark_config import is_magpie_agentx
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
from hyperloom.inference_optimizer.framework_registry import extra_args_env
from inference_suite.argv import argv_flag_occurrences, canonical_flag_name
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
    GpuTelemetry,
    LaunchDelta,
    Lifecycle,
    RunPaths,
    ServerConfig,
    SuiteConfig,
    WorkloadKind,
)
from inference_suite.frameworks import Framework, FrameworkKind, framework_facts

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
    }
)

# Magpie's launcher defaults (scripts/benchmark/*_mi3*.sh at 06b7cf6), and the eval probe's in _inferencex_patcher.
_LAUNCHER_PORT = 8888
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
    server may log nothing once ready; ``env_inherit`` names the host environment the server and client may see.
    """

    base_path: str
    paths: RunPaths
    framework_python: str
    lifecycle: Lifecycle
    telemetry: GpuTelemetry
    silence_sec: float
    env_inherit: tuple[str, ...] = ()
    deadline_unix: int | None = None


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


def _point(document: Mapping[str, object]) -> _Point:
    bench = document.get("benchmark")
    if not isinstance(bench, Mapping):
        raise ConfigError("a point YAML holds a benchmark mapping")
    try:
        framework = Framework(str(bench.get("framework") or "").strip().lower())
    except ValueError:
        raise ConfigError(f"the point's framework {bench.get('framework')!r} is not one inference_suite runs") from None
    if framework_facts(framework).kind is not FrameworkKind.SERVING:
        raise ConfigError(f"the point runs {framework.value}, which serves nothing; it stays on the benchmark backend")
    if is_magpie_agentx(bench):
        raise ConfigError("an AgentX point is not translated; it stays on the benchmark backend")
    if ((bench.get("profiler") or {}).get("torch_profiler") or {}).get("enabled"):
        raise ConfigError("a profile point is not translated; it stays on the benchmark backend")
    envs = {str(name): str(value) for name, value in (bench.get("envs") or {}).items() if value is not None}
    if envs.get(_REMOTE_SERVER_ENV, "").strip():
        raise ConfigError(f"a point with {_REMOTE_SERVER_ENV} serves off this node; it stays on the benchmark backend")
    timeout = bench.get("timeout_seconds")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ConfigError(f"the point's timeout_seconds must be a positive number, got {timeout!r}")
    boot = (bench.get("server_lifecycle") or {}).get("server_ready_timeout_s") or timeout
    return _Point(framework, str(bench.get("model") or "").strip(), envs, float(timeout), float(boot))


def _check_topology(point: _Point, base: BaseLaunch) -> None:
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


def _server(point: _Point, base: BaseLaunch, settings: RoundSettings) -> ServerConfig:
    args_add, args_remove = _user_args(point)
    env = {name: value for name, value in point.envs.items() if name not in _CONSUMED_ENV}
    env.pop(extra_args_env(point.framework.value), None)
    if not point.model:
        raise ConfigError("the point names no model")
    return ServerConfig(
        base=BaseRef(settings.base_path, base.digest()),
        model_path=point.model,
        served_model_name=point.model,
        host=SERVER_HOST,
        port=point.integer(_PORT_ENV, _LAUNCHER_PORT),
        gpus=_gpus(point, base.topology.tp),
        framework_python=settings.framework_python,
        user=LaunchDelta(args_add=args_add, args_remove=args_remove, env_set=env),
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

    A point whose ``MAGPIE_RUN_PHASE`` is ``server`` only boots the server, so its round runs no load and no eval.

    Raises:
        ConfigError: When the point is one this translation does not carry (AgentX, profile, scriptable or
            off-node), names another framework or topology than ``base``, or lacks a knob its load needs.
    """
    point = _point(document)
    if point.framework.value != base.framework:
        raise ConfigError(f"the point runs {point.framework.value}, its base runs {base.framework}")
    _check_topology(point, base)
    boot_only = (point.text(_RUN_PHASE_ENV) or "").lower() == _SERVER_ONLY_PHASE
    return SuiteConfig(
        server=_server(point, base, settings),
        workload=WorkloadKind.NONE if boot_only else WorkloadKind.FIXED,
        fixed=None if boot_only else _fixed(point),
        eval=_eval(point, enabled=not boot_only and point.flag(ENV_RUN_EVAL, default=False)),
        telemetry=settings.telemetry,
        lifecycle=settings.lifecycle,
        deadlines=Deadlines(
            boot_sec=point.boot_timeout_sec,
            round_sec=point.timeout_sec,
            eval_sec=point.timeout_sec,
            silence_sec=settings.silence_sec,
            deadline_unix=settings.deadline_unix,
        ),
        paths=settings.paths,
    )


__all__ = ["MEASURED_SEED", "SERVER_HOST", "WARMUP_SEED", "RoundSettings", "suite_config"]
