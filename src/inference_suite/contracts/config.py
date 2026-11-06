# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``config.yaml`` (``inference-suite-config/1``): everything one round runs from, stored verbatim in its run_dir.

The config names its base by path and digest; :meth:`SuiteConfig.check_base` holds a loaded base to that digest and
to the config's golden use. Every value a round depends on is explicit: seeds, eval probe thresholds, deadlines.
Thresholds that judge a result stay with the caller (D2), so none is defaulted here.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from hyperloom_common.contract_fields import ContractFields, field_names
from inference_suite.argv import normalize_argv, parse_removal_specs
from inference_suite.contracts.base_launch import _ENV_NAME, _OPTION, BaseLaunch, Workload

CONFIG_SCHEMA = "inference-suite-config/1"
DEFAULT_MAX_TOKENS_CAP = 4096
DEFAULT_PRE_READY_SILENCE_SEC = 900
#: The only client environment an AgentX round passes through, and the fast mode it must pin off.
AGENTX_CLIENT_ENV_PREFIX = "AIPERF_"
AGENTX_FAST_MODE_ENV = "AIPERF_EXPERIMENTAL_FAST"

_BASE_DIGEST = re.compile(r"sha256:[0-9a-f]{64}")
#: A scriptable parameter that reads as an acceptance threshold (``XDIT_QUALITY_LPIPS_MAX``); D2 keeps those with
#: the caller.
_THRESHOLD_NAME = re.compile(r"threshold|_(?:max|min)$", re.IGNORECASE)
_MAX_PORT = 65535


class ConfigError(ValueError):
    """A suite config or its document breaks the contract."""


_FIELDS = ContractFields(ConfigError)


class GoldenUse(str, Enum):
    APPLY = "apply"
    #: The D9 accuracy round: the base's golden layer is left out so eval scores the target model's own text.
    EXCLUDE = "exclude"


class WorkloadKind(str, Enum):
    FIXED = "fixed"
    AGENTX = "agentx"
    SCRIPTABLE = "scriptable"
    #: A boot-only round, or an accuracy round that runs eval and no load.
    NONE = "none"


class LifecycleMode(str, Enum):
    OWN = "own"
    BOOT = "boot"
    REUSE = "reuse"
    #: A client-only round against a server something else runs and tears down (multi-node).
    EXTERNAL = "external"


class EvalTask(str, Enum):
    GSM8K = "gsm8k"


#: The modes that hand a server across rounds, so they need the directory its pidfile lives in.
_PID_DIR_MODES = frozenset({LifecycleMode.BOOT, LifecycleMode.REUSE})


def _env_names(names: object, what: str) -> tuple[str, ...]:
    names = _FIELDS.string_tuple(names, what)
    for name in names:
        _FIELDS.matching(_ENV_NAME, name, f"an environment variable name in {what}")
    _FIELDS.unique(names, what)
    return names


def _env_map(value: object, what: str) -> dict[str, str]:
    env = dict(_FIELDS.string_map(value, what))
    _env_names(tuple(env), what)
    return env


def _sorted(value: Mapping[str, object]) -> dict[str, object]:
    return dict(sorted(value.items()))


def _scalar(value: object, what: str) -> None:
    if not isinstance(value, (str, bool, int)):
        _FIELDS.number(value, what)


def _fraction(value: object, what: str) -> float:
    number = _FIELDS.number(value, what)
    if not 0 <= number <= 1:
        raise ConfigError(f"{what} must lie in [0, 1], got {value!r}")
    return number


def _positive(value: object, what: str) -> None:
    if _FIELDS.number(value, what) <= 0:
        raise ConfigError(f"{what} must be positive, got {value!r}")


@dataclass(frozen=True)
class BaseRef:
    """The frozen ``base_launch.json`` a round composes on, and the digest it must still have."""

    path: str
    digest: str

    def __post_init__(self) -> None:
        _FIELDS.text(self.path, "the base path")
        _FIELDS.matching(_BASE_DIGEST, self.digest, "the base digest")

    def to_document(self) -> dict[str, object]:
        return {"path": self.path, "digest": self.digest}

    @classmethod
    def from_document(cls, document: object) -> BaseRef:
        fields = _FIELDS.known(document, field_names(cls), "the base reference")
        return cls(fields.get("path"), fields.get("digest"))


@dataclass(frozen=True)
class LaunchDelta:
    """One layer's change to the base: flags added and removed, environment set and unset.

    ``args_remove`` holds removal specs as :func:`~inference_suite.argv.parse_removal_specs` reads them: a bare flag
    drops every copy, ``--flag value`` only the copy with that value.
    """

    args_add: tuple[str, ...] = ()
    args_remove: tuple[str, ...] = ()
    env_set: Mapping[str, str] = field(default_factory=dict)
    env_unset: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "args_add", normalize_argv(_FIELDS.string_tuple(self.args_add, "the added args")))
        for spec in _FIELDS.string_tuple(self.args_remove, "the removed args"):
            if not spec.startswith("--") or not parse_removal_specs([spec]):
                raise ConfigError(f"a removed arg must name a long flag, got {spec!r}")
        object.__setattr__(self, "env_set", _env_map(self.env_set, "the set env"))
        both = sorted(set(self.env_set) & set(_env_names(self.env_unset, "the unset env")))
        if both:
            raise ConfigError(f"a layer both sets and unsets {', '.join(both)}")

    def to_document(self) -> dict[str, object]:
        return {
            "args": {"add": list(self.args_add), "remove": list(self.args_remove)},
            "env": {"set": _sorted(self.env_set), "unset": list(self.env_unset)},
        }

    @classmethod
    def from_document(cls, document: object) -> LaunchDelta:
        fields = _FIELDS.known(document, ("args", "env"), "a launch delta")
        args = _FIELDS.known(fields.get("args", {}), ("add", "remove"), "a launch delta's args")
        env = _FIELDS.known(fields.get("env", {}), ("set", "unset"), "a launch delta's env")
        return cls(
            args_add=_FIELDS.strings(args.get("add", ()), "the added args"),
            args_remove=_FIELDS.strings(args.get("remove", ()), "the removed args"),
            env_set=env.get("set", {}),
            env_unset=_FIELDS.strings(env.get("unset", ()), "the unset env"),
        )


@dataclass(frozen=True)
class RunnerLayer:
    """The runner's own layer (D12), merged last: ``args_floor`` raises a flag's value to a floor, never lowers it."""

    env_prepend: Mapping[str, str] = field(default_factory=dict)
    env_set: Mapping[str, str] = field(default_factory=dict)
    args_floor: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "env_prepend", _env_map(self.env_prepend, "the runner's prepended env"))
        object.__setattr__(self, "env_set", _env_map(self.env_set, "the runner's set env"))
        both = sorted(set(self.env_prepend) & set(self.env_set))
        if both:
            raise ConfigError(f"the runner both prepends to and sets {', '.join(both)}")
        floors = dict(_FIELDS.mapping(self.args_floor, "the runner's arg floors"))
        for flag, floor in floors.items():
            _FIELDS.matching(_OPTION, flag, "a runner floor's flag")
            _FIELDS.number(floor, f"the floor of {flag}")
        object.__setattr__(self, "args_floor", floors)

    def to_document(self) -> dict[str, object]:
        return {
            "env_prepend": _sorted(self.env_prepend),
            "env_set": _sorted(self.env_set),
            "args_floor": _sorted(self.args_floor),
        }

    @classmethod
    def from_document(cls, document: object) -> RunnerLayer:
        fields = _FIELDS.known(document, field_names(cls), "the runner layer")
        return cls(fields.get("env_prepend", {}), fields.get("env_set", {}), fields.get("args_floor", {}))


_SERVER_KEYS = (
    "base",
    "user",
    "candidate",
    "runner",
    "golden",
    "model_path",
    "served_model_name",
    "host",
    "port",
    "gpus",
    "framework_python",
    "source_files",
    "absent_source_files",
    "env",
)


@dataclass(frozen=True)
class ServerConfig:
    """The launch layers on the base, and the runtime facts (paths, address, GPUs) that stay out of its identity.

    ``user`` and ``candidate`` stay separate layers so a refused flag can be told apart: a candidate's is the
    candidate's failure, a user's or base's stops the line. ``source_files`` are re-hashed before launch;
    ``env_inherit`` names the host environment the server may see.
    """

    base: BaseRef
    model_path: str
    served_model_name: str
    host: str
    port: int
    gpus: tuple[int, ...]
    framework_python: str
    golden: GoldenUse = GoldenUse.APPLY
    user: LaunchDelta = field(default_factory=LaunchDelta)
    candidate: LaunchDelta = field(default_factory=LaunchDelta)
    runner: RunnerLayer = field(default_factory=RunnerLayer)
    source_files: Mapping[str, str] = field(default_factory=dict)
    absent_source_files: tuple[str, ...] = ()
    env_inherit: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("model_path", "served_model_name", "host", "framework_python"):
            _FIELDS.text(getattr(self, name), f"the server's {name}")
        port = _FIELDS.integer(self.port, "the server port", minimum=1)
        if port > _MAX_PORT:
            raise ConfigError(f"the server port must be at most {_MAX_PORT}, got {port}")
        if not isinstance(self.gpus, tuple) or not self.gpus:
            raise ConfigError(f"the server's gpus must be a non-empty tuple, got {self.gpus!r}")
        for gpu in self.gpus:
            _FIELDS.integer(gpu, "a server gpu", minimum=0)
        _FIELDS.unique(self.gpus, "server gpus")
        object.__setattr__(self, "golden", _FIELDS.enum(GoldenUse, self.golden, "the server's golden use"))
        for path, sha256 in _FIELDS.string_map(self.source_files, "the source files").items():
            _FIELDS.sha256(sha256, f"the sha256 of source file {path}")
        absent = _FIELDS.string_tuple(self.absent_source_files, "the absent source files")
        both = sorted(set(absent) & set(self.source_files))
        if both:
            raise ConfigError(f"source files both present and absent: {', '.join(both)}")
        _env_names(self.env_inherit, "the inherited env")

    def to_document(self) -> dict[str, object]:
        return {
            "base": self.base.to_document(),
            "user": self.user.to_document(),
            "candidate": self.candidate.to_document(),
            "runner": self.runner.to_document(),
            "golden": self.golden.value,
            "model_path": self.model_path,
            "served_model_name": self.served_model_name,
            "host": self.host,
            "port": self.port,
            "gpus": list(self.gpus),
            "framework_python": self.framework_python,
            "source_files": _sorted(self.source_files),
            "absent_source_files": list(self.absent_source_files),
            "env": {"inherit": list(self.env_inherit)},
        }

    @classmethod
    def from_document(cls, document: object) -> ServerConfig:
        fields = _FIELDS.known(document, _SERVER_KEYS, "the server config")
        env = _FIELDS.known(fields.get("env", {}), ("inherit",), "the server env")
        return cls(
            base=BaseRef.from_document(fields.get("base")),
            user=LaunchDelta.from_document(fields.get("user", {})),
            candidate=LaunchDelta.from_document(fields.get("candidate", {})),
            runner=RunnerLayer.from_document(fields.get("runner", {})),
            golden=fields.get("golden", GoldenUse.APPLY.value),
            model_path=fields.get("model_path"),
            served_model_name=fields.get("served_model_name"),
            host=fields.get("host"),
            port=fields.get("port"),
            gpus=_FIELDS.items(fields.get("gpus"), "the server's gpus"),
            framework_python=fields.get("framework_python"),
            source_files=fields.get("source_files", {}),
            absent_source_files=_FIELDS.strings(fields.get("absent_source_files", ()), "the absent source files"),
            env_inherit=_FIELDS.strings(env.get("inherit", ()), "the inherited env"),
        )


@dataclass(frozen=True)
class FixedWorkload:
    """A fixed-shape load; every client knob is explicit so two arms run the same requests."""

    concurrency: int
    isl: int
    osl: int
    random_range_ratio: float
    num_prompts: int
    num_warmups: int
    seed: int
    warmup_seed: int
    tokenizer_mode: str
    trust_remote_code: bool

    def __post_init__(self) -> None:
        for name in ("concurrency", "isl", "osl", "num_prompts"):
            _FIELDS.integer(getattr(self, name), f"the fixed {name}", minimum=1)
        for name in ("num_warmups", "seed", "warmup_seed"):
            _FIELDS.integer(getattr(self, name), f"the fixed {name}", minimum=0)
        _fraction(self.random_range_ratio, "the fixed random_range_ratio")
        _FIELDS.text(self.tokenizer_mode, "the fixed tokenizer_mode")
        _FIELDS.boolean(self.trust_remote_code, "the fixed trust_remote_code")

    def to_document(self) -> dict[str, object]:
        return {key: getattr(self, key) for key in _FIXED_KEYS}

    @classmethod
    def from_document(cls, document: object) -> FixedWorkload:
        fields = _FIELDS.known(document, _FIXED_KEYS, "the fixed workload")
        return cls(**{key: fields.get(key) for key in _FIXED_KEYS})


_FIXED_KEYS = field_names(FixedWorkload)


@dataclass(frozen=True)
class AgentxWorkload:
    """An AgentX trace replay. ``client_env`` is the client's ``AIPERF_*`` environment, fast mode pinned off."""

    concurrency: int
    dataset: str
    num_prompts: int
    warmup_requests_per_lane: int
    seed: int
    warmup_seed: int
    failed_request_threshold: float
    client_env: Mapping[str, str]

    def __post_init__(self) -> None:
        _FIELDS.integer(self.concurrency, "the AgentX concurrency", minimum=1)
        _FIELDS.text(self.dataset, "the AgentX dataset")
        _FIELDS.integer(self.num_prompts, "the AgentX num_prompts", minimum=1)
        for name in ("warmup_requests_per_lane", "seed", "warmup_seed"):
            _FIELDS.integer(getattr(self, name), f"the AgentX {name}", minimum=0)
        _fraction(self.failed_request_threshold, "the AgentX failed_request_threshold")
        env = _env_map(self.client_env, "the AgentX client env")
        foreign = sorted(name for name in env if not name.startswith(AGENTX_CLIENT_ENV_PREFIX))
        if foreign:
            raise ConfigError(f"the AgentX client env takes {AGENTX_CLIENT_ENV_PREFIX}* only, got {', '.join(foreign)}")
        if env.get(AGENTX_FAST_MODE_ENV) != "0":
            raise ConfigError(
                f"the AgentX client env must set {AGENTX_FAST_MODE_ENV}=0, got {env.get(AGENTX_FAST_MODE_ENV)!r}"
            )
        object.__setattr__(self, "client_env", env)

    def to_document(self) -> dict[str, object]:
        document = {key: getattr(self, key) for key in _AGENTX_KEYS}
        return {**document, "client_env": _sorted(self.client_env)}

    @classmethod
    def from_document(cls, document: object) -> AgentxWorkload:
        fields = _FIELDS.known(document, _AGENTX_KEYS, "the AgentX workload")
        return cls(**{key: fields.get(key) for key in _AGENTX_KEYS})


_AGENTX_KEYS = field_names(AgentxWorkload)


@dataclass(frozen=True)
class ScriptableWorkload:
    """A server-less script (xDiT); its ``params`` carry no acceptance threshold (D2)."""

    script: str
    params: Mapping[str, object] = field(default_factory=dict)
    quality_reference: str | None = None

    def __post_init__(self) -> None:
        _FIELDS.text(self.script, "the scriptable script")
        params = dict(_FIELDS.mapping(self.params, "the scriptable params"))
        for name, value in params.items():
            _FIELDS.text(name, "a scriptable param name")
            if _THRESHOLD_NAME.search(name):
                raise ConfigError(f"the scriptable param {name} is a threshold; the caller judges the result (D2)")
            _scalar(value, f"the scriptable param {name}")
        object.__setattr__(self, "params", params)
        _FIELDS.optional_text(self.quality_reference, "the scriptable quality reference")

    def to_document(self) -> dict[str, object]:
        return {"script": self.script, "params": _sorted(self.params), "quality_reference": self.quality_reference}

    @classmethod
    def from_document(cls, document: object) -> ScriptableWorkload:
        fields = _FIELDS.known(document, field_names(cls), "the scriptable workload")
        return cls(fields.get("script"), fields.get("params", {}), fields.get("quality_reference"))


#: The window fields each workload's capture needs; the caller computes them (``_profile_setup``).
_PROFILE_WINDOW = {
    WorkloadKind.FIXED: ("num_prompts", "delay_iters", "max_iters"),
    WorkloadKind.AGENTX: ("agentx_open_after_sec", "agentx_duration_sec"),
}


@dataclass(frozen=True)
class ProfileConfig:
    """A torch-profiler capture window; a disabled profile carries no window."""

    enabled: bool
    start_body: Mapping[str, object] | None = None
    num_prompts: int | None = None
    delay_iters: int | None = None
    max_iters: int | None = None
    agentx_open_after_sec: float | None = None
    agentx_duration_sec: float | None = None

    def __post_init__(self) -> None:
        _FIELDS.boolean(self.enabled, "the profile's enabled")
        if self.start_body is not None:
            object.__setattr__(self, "start_body", dict(_FIELDS.mapping(self.start_body, "the profile start body")))
        _FIELDS.optional_integer(self.num_prompts, "the profile num_prompts", minimum=1)
        _FIELDS.optional_integer(self.delay_iters, "the profile delay_iters", minimum=0)
        _FIELDS.optional_integer(self.max_iters, "the profile max_iters", minimum=1)
        if self.agentx_open_after_sec is not None:
            _FIELDS.number(self.agentx_open_after_sec, "the profile agentx_open_after_sec")
        if self.agentx_duration_sec is not None:
            _positive(self.agentx_duration_sec, "the profile agentx_duration_sec")
        if not self.enabled and any(getattr(self, key) is not None for key in _PROFILE_KEYS[1:]):
            raise ConfigError("a disabled profile carries no capture window")

    def check_window(self, kind: WorkloadKind) -> None:
        missing = [key for key in _PROFILE_WINDOW.get(kind, ()) if getattr(self, key) is None]
        if kind not in _PROFILE_WINDOW or missing:
            raise ConfigError(f"a {kind.value} profile needs {', '.join(missing) or 'a fixed or AgentX workload'}")

    def to_document(self) -> dict[str, object]:
        return {key: getattr(self, key) for key in _PROFILE_KEYS}

    @classmethod
    def from_document(cls, document: object) -> ProfileConfig:
        fields = _FIELDS.known(document, _PROFILE_KEYS, "the profile config")
        return cls(**{key: fields.get(key) for key in _PROFILE_KEYS})


_PROFILE_KEYS = field_names(ProfileConfig)


@dataclass(frozen=True)
class EvalProbe:
    """The truncation probe's thresholds, which the caller passes in (D2): nothing here defaults them."""

    enabled: bool
    min_samples: int
    length_ratio: float

    def __post_init__(self) -> None:
        _FIELDS.boolean(self.enabled, "the probe's enabled")
        _FIELDS.integer(self.min_samples, "the probe's min_samples", minimum=1)
        _positive(self.length_ratio, "the probe's length_ratio")

    def to_document(self) -> dict[str, object]:
        return {"enabled": self.enabled, "min_samples": self.min_samples, "length_ratio": self.length_ratio}

    @classmethod
    def from_document(cls, document: object) -> EvalProbe:
        keys = field_names(cls)
        fields = _FIELDS.known(document, keys, "the eval probe")
        missing = [key for key in keys if key not in fields]
        if missing:
            raise ConfigError(f"the eval probe must set {', '.join(missing)}")
        return cls(**{key: fields.get(key) for key in keys})


@dataclass(frozen=True)
class EvalConfig:
    """The accuracy eval run after the load, on the same server. ``limit`` None scores the whole task."""

    enabled: bool
    num_fewshot: int
    concurrency: int
    probe: EvalProbe
    limit: int | None = None
    task: EvalTask = EvalTask.GSM8K
    max_tokens_cap: int = DEFAULT_MAX_TOKENS_CAP

    def __post_init__(self) -> None:
        _FIELDS.boolean(self.enabled, "the eval's enabled")
        _FIELDS.integer(self.num_fewshot, "the eval's num_fewshot", minimum=0)
        _FIELDS.integer(self.concurrency, "the eval's concurrency", minimum=1)
        _FIELDS.optional_integer(self.limit, "the eval's limit", minimum=1)
        object.__setattr__(self, "task", _FIELDS.enum(EvalTask, self.task, "the eval's task"))
        _FIELDS.integer(self.max_tokens_cap, "the eval's max_tokens_cap", minimum=1)
        if not isinstance(self.probe, EvalProbe):
            raise ConfigError(f"the eval's probe must be set, got {self.probe!r}")

    def to_document(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "task": self.task.value,
            "limit": self.limit,
            "num_fewshot": self.num_fewshot,
            "concurrency": self.concurrency,
            "max_tokens_cap": self.max_tokens_cap,
            "probe": self.probe.to_document(),
        }

    @classmethod
    def from_document(cls, document: object) -> EvalConfig:
        fields = _FIELDS.known(document, field_names(cls), "the eval config")
        if fields.get("probe") is None:
            raise ConfigError("the eval config must set probe; its thresholds come from the caller")
        return cls(
            enabled=fields.get("enabled"),
            task=fields.get("task", EvalTask.GSM8K.value),
            limit=fields.get("limit"),
            num_fewshot=fields.get("num_fewshot"),
            concurrency=fields.get("concurrency"),
            max_tokens_cap=fields.get("max_tokens_cap", DEFAULT_MAX_TOKENS_CAP),
            probe=EvalProbe.from_document(fields.get("probe")),
        )


@dataclass(frozen=True)
class GpuTelemetry:
    enabled: bool
    interval_sec: float

    def __post_init__(self) -> None:
        _FIELDS.boolean(self.enabled, "the GPU telemetry's enabled")
        _positive(self.interval_sec, "the GPU telemetry interval")

    def to_document(self) -> dict[str, object]:
        return {"gpu": {"enabled": self.enabled, "interval_sec": self.interval_sec}}

    @classmethod
    def from_document(cls, document: object) -> GpuTelemetry:
        telemetry = _FIELDS.known(document, ("gpu",), "the telemetry config")
        fields = _FIELDS.known(telemetry.get("gpu"), field_names(cls), "the GPU telemetry")
        return cls(fields.get("enabled"), fields.get("interval_sec"))


@dataclass(frozen=True)
class Lifecycle:
    """Who owns the server (D14). ``boot`` and ``reuse`` hand it across rounds through ``pid_dir``."""

    mode: LifecycleMode
    pid_dir: str | None = None

    def __post_init__(self) -> None:
        mode = _FIELDS.enum(LifecycleMode, self.mode, "the lifecycle mode")
        object.__setattr__(self, "mode", mode)
        _FIELDS.optional_text(self.pid_dir, "the lifecycle pid_dir")
        if (mode in _PID_DIR_MODES) != (self.pid_dir is not None):
            raise ConfigError(f"a pid_dir is set exactly for boot and reuse, got {mode.value} with {self.pid_dir!r}")

    def to_document(self) -> dict[str, object]:
        return {"mode": self.mode.value, "pid_dir": self.pid_dir}

    @classmethod
    def from_document(cls, document: object) -> Lifecycle:
        fields = _FIELDS.known(document, field_names(cls), "the lifecycle")
        return cls(fields.get("mode"), fields.get("pid_dir"))


_DEADLINE_KEYS = ("boot_sec", "round_sec", "eval_sec", "silence_sec", "pre_ready_silence_sec", "deadline_unix")


@dataclass(frozen=True)
class Deadlines:
    """The round's time budgets; aiter JIT builds do not count against ``boot_sec``."""

    boot_sec: float
    round_sec: float
    eval_sec: float
    silence_sec: float
    pre_ready_silence_sec: float = DEFAULT_PRE_READY_SILENCE_SEC
    deadline_unix: int | None = None

    def __post_init__(self) -> None:
        for name in _DEADLINE_KEYS[:-1]:
            _positive(getattr(self, name), f"the {name} deadline")
        _FIELDS.optional_integer(self.deadline_unix, "the absolute deadline", minimum=0)

    def to_document(self) -> dict[str, object]:
        return {key: getattr(self, key) for key in _DEADLINE_KEYS}

    @classmethod
    def from_document(cls, document: object) -> Deadlines:
        fields = _FIELDS.known(document, _DEADLINE_KEYS, "the deadlines")
        required = {key: fields.get(key) for key in _DEADLINE_KEYS[:4]}
        return cls(**required, **{key: fields[key] for key in _DEADLINE_KEYS[4:] if key in fields})


@dataclass(frozen=True)
class RunPaths:
    """Where the round writes (``run_dir``, absent or empty) and the staged tools it runs (D11)."""

    run_dir: str
    tool_cache: str
    tool_manifest_sha256: str

    def __post_init__(self) -> None:
        _FIELDS.text(self.run_dir, "the run_dir")
        _FIELDS.text(self.tool_cache, "the tool cache")
        _FIELDS.sha256(self.tool_manifest_sha256, "the tool manifest sha256")

    def to_document(self) -> dict[str, object]:
        return {
            "run_dir": self.run_dir,
            "tool_cache": self.tool_cache,
            "tool_manifest_sha256": self.tool_manifest_sha256,
        }

    @classmethod
    def from_document(cls, document: object) -> RunPaths:
        keys = field_names(cls)
        fields = _FIELDS.known(document, keys, "the paths")
        return cls(*(fields.get(key) for key in keys))


#: The workload kinds with a config section; each section is the field named by its kind's value.
_SECTIONED_WORKLOADS = (WorkloadKind.FIXED, WorkloadKind.AGENTX, WorkloadKind.SCRIPTABLE)
_BASE_WORKLOAD = {WorkloadKind.FIXED: Workload.FIXED, WorkloadKind.AGENTX: Workload.AGENTX}


@dataclass(frozen=True)
class SuiteConfig:
    """One round's config. The section named by ``workload`` is set and the others are not."""

    server: ServerConfig
    workload: WorkloadKind
    eval: EvalConfig
    telemetry: GpuTelemetry
    lifecycle: Lifecycle
    deadlines: Deadlines
    paths: RunPaths
    profile: ProfileConfig = field(default_factory=lambda: ProfileConfig(enabled=False))
    fixed: FixedWorkload | None = None
    agentx: AgentxWorkload | None = None
    scriptable: ScriptableWorkload | None = None

    def __post_init__(self) -> None:
        kind = _FIELDS.enum(WorkloadKind, self.workload, "the workload kind")
        object.__setattr__(self, "workload", kind)
        present = sorted(item.value for item in _SECTIONED_WORKLOADS if getattr(self, item.value) is not None)
        expected = [kind.value] if kind in _SECTIONED_WORKLOADS else []
        if present != expected:
            raise ConfigError(f"a {kind.value} workload sets {expected or 'no workload section'}, got {present}")
        if kind is WorkloadKind.NONE and not (self.lifecycle.mode is LifecycleMode.BOOT or self.eval.enabled):
            raise ConfigError("a round with no workload boots a server or runs eval")
        if self.server.golden is GoldenUse.EXCLUDE and not (kind is WorkloadKind.NONE and self.eval.enabled):
            raise ConfigError("golden exclude is the accuracy round: no workload, eval enabled")
        if self.profile.enabled:
            if self.lifecycle.mode is not LifecycleMode.OWN:
                raise ConfigError(f"a profile round owns its server, got lifecycle {self.lifecycle.mode.value}")
            self.profile.check_window(kind)

    def check_base(self, base: BaseLaunch) -> None:
        """Refuse ``base`` unless it is the base this config names and can serve the config's golden use."""

        if base.digest() != self.server.base.digest:
            raise ConfigError(
                f"base_digest_mismatch: the config names {self.server.base.digest}, the base gives {base.digest()}"
            )
        if self.server.golden is GoldenUse.EXCLUDE and base.golden_al is None:
            raise ConfigError("golden exclude needs a base with a golden layer, and this base has none")
        expected = _BASE_WORKLOAD.get(self.workload)
        if expected is not None and base.query.workload is not expected:
            raise ConfigError(
                f"a {self.workload.value} round needs a {expected.value} base, got {base.query.workload.value}"
            )

    def to_document(self) -> dict[str, object]:
        sections = {item.value: getattr(self, item.value) for item in _SECTIONED_WORKLOADS}
        return {
            "schema": CONFIG_SCHEMA,
            "server": self.server.to_document(),
            "workload": {"kind": self.workload.value},
            **{name: None if section is None else section.to_document() for name, section in sections.items()},
            "profile": self.profile.to_document(),
            "eval": self.eval.to_document(),
            "telemetry": self.telemetry.to_document(),
            "lifecycle": self.lifecycle.to_document(),
            "deadlines": self.deadlines.to_document(),
            "paths": self.paths.to_document(),
        }

    @classmethod
    def from_document(cls, document: object) -> SuiteConfig:
        fields = _FIELDS.known(document, ("schema", *field_names(cls)), "a suite config")
        if fields.get("schema") != CONFIG_SCHEMA:
            raise ConfigError(f"suite config schema {fields.get('schema')!r} is not {CONFIG_SCHEMA}")
        if fields.get("eval") is None:
            raise ConfigError("a suite config must set eval, with its probe")
        workload = _FIELDS.known(fields.get("workload"), ("kind",), "the workload")
        profile = fields.get("profile")
        return cls(
            server=ServerConfig.from_document(fields.get("server")),
            workload=workload.get("kind"),
            fixed=None if fields.get("fixed") is None else FixedWorkload.from_document(fields.get("fixed")),
            agentx=None if fields.get("agentx") is None else AgentxWorkload.from_document(fields.get("agentx")),
            scriptable=(
                None if fields.get("scriptable") is None else ScriptableWorkload.from_document(fields.get("scriptable"))
            ),
            profile=ProfileConfig(enabled=False) if profile is None else ProfileConfig.from_document(profile),
            eval=EvalConfig.from_document(fields.get("eval")),
            telemetry=GpuTelemetry.from_document(fields.get("telemetry")),
            lifecycle=Lifecycle.from_document(fields.get("lifecycle")),
            deadlines=Deadlines.from_document(fields.get("deadlines")),
            paths=RunPaths.from_document(fields.get("paths")),
        )


__all__ = [
    "AGENTX_CLIENT_ENV_PREFIX",
    "AGENTX_FAST_MODE_ENV",
    "CONFIG_SCHEMA",
    "DEFAULT_MAX_TOKENS_CAP",
    "DEFAULT_PRE_READY_SILENCE_SEC",
    "AgentxWorkload",
    "BaseRef",
    "ConfigError",
    "Deadlines",
    "EvalConfig",
    "EvalProbe",
    "EvalTask",
    "FixedWorkload",
    "GoldenUse",
    "GpuTelemetry",
    "LaunchDelta",
    "Lifecycle",
    "LifecycleMode",
    "ProfileConfig",
    "RunPaths",
    "RunnerLayer",
    "ScriptableWorkload",
    "ServerConfig",
    "SuiteConfig",
    "WorkloadKind",
]
