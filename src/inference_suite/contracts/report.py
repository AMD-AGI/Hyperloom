# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``report.json`` (``inference-suite-report/1``): what one round measured, or the one step it failed at.

Every key table here is fixed: a metric nobody measured is written ``null``, never 0, and a reader never meets a
key it was not told about. ``gpu_monitor`` blocks use the names Hyperloom's ``gpu_metrics_from_report`` reads, so
the GPU artifact is derived from this report unchanged. Results are recorded raw; what passes a gate is decided by
the caller's mapper (D2).
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum

from hyperloom_common.contract_fields import ContractFields, field_names
from inference_suite.contracts.base_launch import DIGEST_PREFIX
from inference_suite.contracts.config import ConfigError, EvalProbe, WorkloadKind

REPORT_SCHEMA = "inference-suite-report/1"
AGENTX_ERROR_RATE_DEFINITION = "inferencex-agentx/1"

_DIGEST = re.compile(re.escape(DIGEST_PREFIX) + r"[0-9a-f]{64}")
_MAX_PORT = 65535


class ReportError(ValueError):
    """A round report or its document breaks the contract."""


_FIELDS = ContractFields(ReportError)


class RunStatus(str, Enum):
    COMPLETED = "completed"
    #: The load measured, and a later step (eval, profile) failed.
    PARTIAL = "partial"
    FAILED = "failed"
    STOPPED = "stopped"

    @classmethod
    def of(cls, failed_step: FailedStep | None, load_measured: bool) -> RunStatus:
        """The status a round ends with: the step it failed at, if any, and whether its load measured."""
        if failed_step is None:
            return cls.COMPLETED
        if failed_step is FailedStep.CANCELLED:
            return cls.STOPPED
        return cls.PARTIAL if load_measured else cls.FAILED


class FailedStep(str, Enum):
    """The closed list of steps a round can fail at; each replaces the sentinel or class noted beside it."""

    CONFIG_INVALID = "config_invalid"
    BASE_DIGEST_MISMATCH = "base_digest_mismatch"
    FRAMEWORK_VERSION_TOO_OLD = "framework_version_too_old"
    FRAMEWORK_VERSION_UNKNOWN = "framework_version_unknown"
    REUSE_NOT_OWNED = "reuse_not_owned"
    EVAL_CONTEXT_TOO_SHORT = "eval_context_too_short"
    #: -912, -918, ``git_fetch_timeout``.
    STAGING_MISSING = "staging_missing"
    SOURCE_FILES_CHANGED = "source_files_changed"
    #: ``serve_flag``.
    FLAG_UNKNOWN = "flag_unknown"
    #: ``speculative_decoding_pinned``.
    GOLDEN_OWNED_TOUCHED = "golden_owned_touched"
    PORT_IN_USE = "port_in_use"
    #: Magpie's ``gpu_selection.auto`` refusal.
    GPUS_UNAVAILABLE = "gpus_unavailable"
    SERVER_EXITED_BEFORE_READY = "server_exited_before_ready"
    #: ``hung_before_ready``.
    SERVER_SILENT_BEFORE_READY = "server_silent_before_ready"
    #: ``SERVER_BOOT_TIMEOUT_RC``.
    SERVER_BOOT_TIMEOUT = "server_boot_timeout"
    #: -911.
    SERVER_STALLED = "server_stalled"
    SERVER_DIED = "server_died"
    CLIENT_FAILED = "client_failed"
    #: ``benchmark_report_missing`` / ``invalid_metric``.
    CLIENT_OUTPUT_MISSING = "client_output_missing"
    #: ``eval_returncode``, ``lm_eval_http_400``.
    EVAL_FAILED = "eval_failed"
    #: ``magpie_timeout``.
    ROUND_TIMEOUT = "round_timeout"
    #: -915.
    DEADLINE_EXHAUSTED = "deadline_exhausted"
    #: -917.
    CANCELLED = "cancelled"
    DISK_FULL = "disk_full"
    TEARDOWN_INCOMPLETE = "teardown_incomplete"


def _optional_digest(value: object, what: str) -> str | None:
    return None if value is None else _FIELDS.matching(_DIGEST, value, what)


def _optional_count(value: object, what: str) -> int | None:
    return _FIELDS.optional_integer(value, what, minimum=0)


def _text_tuple(value: object, what: str) -> tuple[str, ...]:
    values = _FIELDS.string_tuple(value, what)
    for item in values:
        _FIELDS.text(item, f"an entry of {what}")
    return values


def _ordered(values: Mapping[str, object], keys: tuple[str, ...]) -> dict[str, object]:
    return {key: values[key] for key in keys}


def _complete(document: object, keys: tuple[str, ...], what: str) -> Mapping[str, object]:
    """``document`` holding exactly ``keys``: a fixed key table is written whole, with ``null`` for what is absent."""

    fields = _FIELDS.known(document, keys, what)
    missing = [key for key in keys if key not in fields]
    if missing:
        raise ReportError(f"{what} is missing {', '.join(missing)}; an unmeasured value is written null")
    return fields


def _measured(values: Mapping[str, object], keys: tuple[str, ...], integers: frozenset[str], what: str) -> None:
    if not isinstance(values, Mapping) or set(values) != set(keys):
        got = sorted(values) if isinstance(values, Mapping) else values
        raise ReportError(f"{what} must hold exactly {', '.join(keys)}, got {got!r}")
    for key, value in values.items():
        if key in integers:
            _optional_count(value, f"{what} {key}")
        else:
            _FIELDS.optional_number(value, f"{what} {key}")


@dataclass(frozen=True)
class MetricStats:
    """One GPU metric over the measured window.

    No ``min <= avg <= max`` check: the mean of constant float samples can land one ULP
    outside the range, and telemetry must never fail a round.
    """

    min: float
    max: float
    avg: float

    def __post_init__(self) -> None:
        for name in ("min", "max", "avg"):
            _FIELDS.number(getattr(self, name), f"the metric's {name}")

    def to_document(self) -> dict[str, object]:
        return {"min": self.min, "max": self.max, "avg": self.avg}

    @classmethod
    def from_document(cls, document: object) -> MetricStats:
        fields = _complete(document, field_names(cls), "a GPU metric")
        return cls(fields["min"], fields["max"], fields["avg"])


#: The per-card metrics, named as ``gpu_metrics_from_report`` reads them.
GPU_METRIC_KEYS = ("power_watts", "temperature_c", "gpu_clock_mhz", "mem_clock_mhz", "gpu_util_pct", "vram_pct")


@dataclass(frozen=True)
class GpuMonitorBlock:
    """One card's telemetry. A metric amd-smi gave no reading for is ``None``; telemetry never fails a round."""

    gpu: int
    sample_count: int
    power_watts: MetricStats | None = None
    temperature_c: MetricStats | None = None
    gpu_clock_mhz: MetricStats | None = None
    mem_clock_mhz: MetricStats | None = None
    gpu_util_pct: MetricStats | None = None
    vram_pct: MetricStats | None = None

    def __post_init__(self) -> None:
        _FIELDS.integer(self.gpu, "the monitored GPU index", minimum=0)
        _FIELDS.integer(self.sample_count, "the GPU sample count", minimum=0)
        for key in GPU_METRIC_KEYS:
            value = getattr(self, key)
            if value is not None and not isinstance(value, MetricStats):
                raise ReportError(f"GPU {self.gpu} {key} must be a MetricStats or None, got {value!r}")
            if value is not None and self.sample_count == 0:
                raise ReportError(f"GPU {self.gpu} reports {key} from no samples")

    def to_document(self) -> dict[str, object]:
        metrics = {key: getattr(self, key) for key in GPU_METRIC_KEYS}
        return {
            "gpu": self.gpu,
            "sample_count": self.sample_count,
            **{key: None if stats is None else stats.to_document() for key, stats in metrics.items()},
        }

    @classmethod
    def from_document(cls, document: object) -> GpuMonitorBlock:
        fields = _complete(document, ("gpu", "sample_count", *GPU_METRIC_KEYS), "a gpu_monitor block")
        metrics = {
            key: None if fields[key] is None else MetricStats.from_document(fields[key]) for key in GPU_METRIC_KEYS
        }
        return cls(gpu=fields["gpu"], sample_count=fields["sample_count"], **metrics)


#: The fixed workload's metrics, named as Hyperloom's benchmark result normalises them.
FIXED_METRIC_KEYS = (
    "output_throughput",
    "request_throughput",
    "total_token_throughput",
    "input_throughput",
    "completed_requests",
    "requested_requests",
    "failed_requests",
    "duration_seconds",
    "ttft_mean_ms",
    "ttft_p50_ms",
    "ttft_p90_ms",
    "ttft_p99_ms",
    "tpot_mean_ms",
    "tpot_p50_ms",
    "tpot_p90_ms",
    "tpot_p99_ms",
    "e2el_mean_ms",
    "e2el_p99_ms",
    "e2e_norm_intvty_p50",
    "e2e_norm_intvty_p90",
    "request_error_rate",
    "total_input_tokens",
    "total_output_tokens",
)
_FIXED_COUNTS = frozenset(
    {"completed_requests", "requested_requests", "failed_requests", "total_input_tokens", "total_output_tokens"}
)


@dataclass(frozen=True)
class FixedMetrics:
    """The fixed client's result over :data:`FIXED_METRIC_KEYS`; :meth:`of` writes the unmeasured ones as ``None``."""

    values: Mapping[str, float | int | None]

    def __post_init__(self) -> None:
        _measured(self.values, FIXED_METRIC_KEYS, _FIXED_COUNTS, "the fixed metrics")

    @classmethod
    def of(cls, measured: Mapping[str, float | int | None]) -> FixedMetrics:
        unknown = sorted(set(measured) - set(FIXED_METRIC_KEYS))
        if unknown:
            raise ReportError(f"unknown fixed metric(s) {', '.join(unknown)}")
        return cls({key: measured.get(key) for key in FIXED_METRIC_KEYS})

    def to_document(self) -> dict[str, object]:
        return _ordered(self.values, FIXED_METRIC_KEYS)

    @classmethod
    def from_document(cls, document: object) -> FixedMetrics:
        fields = _complete(document, FIXED_METRIC_KEYS, "the fixed metrics")
        return cls({key: fields[key] for key in FIXED_METRIC_KEYS})


AGENTX_COUNT_KEYS = (
    "records_total",
    "successful",
    "error_dropped",
    "warmup_dropped",
    "dropped_total",
    "error_osl_exported",
)
AGENTX_THROUGHPUT_KEYS = (
    "request_throughput",
    "output_tokens_per_second",
    "input_tokens_per_second",
    "total_tokens_per_second",
    "duration_seconds",
)
AGENTX_LATENCY_KEYS = tuple(
    f"{name}_{stat}" for name in ("ttft", "tpot", "e2el") for stat in ("mean", "p50", "p90", "p99")
)
AGENTX_INTVTY_KEYS = ("p50", "p90")


@dataclass(frozen=True)
class AgentxResult:
    """The AgentX client's raw counts and timings; validity and the error rate are the mapper's (D2)."""

    counts: Mapping[str, int]
    throughput: Mapping[str, float | None]
    latency_seconds: Mapping[str, float | None]
    e2e_norm_intvty: Mapping[str, float | None]
    stopped_by_threshold: bool
    fast_mode: bool
    request_export: str | None = None
    pairing_key: str | None = None

    def __post_init__(self) -> None:
        _measured(self.counts, AGENTX_COUNT_KEYS, frozenset(AGENTX_COUNT_KEYS), "the AgentX counts")
        if any(value is None for value in self.counts.values()):
            raise ReportError(f"the AgentX counts are always counted, got {dict(self.counts)!r}")
        _measured(self.throughput, AGENTX_THROUGHPUT_KEYS, frozenset(), "the AgentX throughput")
        _measured(self.latency_seconds, AGENTX_LATENCY_KEYS, frozenset(), "the AgentX latency")
        _measured(self.e2e_norm_intvty, AGENTX_INTVTY_KEYS, frozenset(), "the AgentX e2e_norm_intvty")
        _FIELDS.boolean(self.stopped_by_threshold, "the AgentX stopped_by_threshold")
        _FIELDS.boolean(self.fast_mode, "the AgentX fast_mode")
        _FIELDS.optional_text(self.request_export, "the AgentX request export")
        _FIELDS.optional_text(self.pairing_key, "the AgentX pairing key")

    def to_document(self) -> dict[str, object]:
        return {
            "counts": _ordered(self.counts, AGENTX_COUNT_KEYS),
            "error_rate_definition": AGENTX_ERROR_RATE_DEFINITION,
            "stopped_by_threshold": self.stopped_by_threshold,
            "fast_mode": self.fast_mode,
            "throughput": _ordered(self.throughput, AGENTX_THROUGHPUT_KEYS),
            "latency_seconds": _ordered(self.latency_seconds, AGENTX_LATENCY_KEYS),
            "e2e_norm_intvty": _ordered(self.e2e_norm_intvty, AGENTX_INTVTY_KEYS),
            "request_export": self.request_export,
            "pairing_key": self.pairing_key,
        }

    @classmethod
    def from_document(cls, document: object) -> AgentxResult:
        keys = (
            "counts",
            "error_rate_definition",
            "stopped_by_threshold",
            "fast_mode",
            "throughput",
            "latency_seconds",
            "e2e_norm_intvty",
            "request_export",
            "pairing_key",
        )
        fields = _complete(document, keys, "the AgentX result")
        if fields["error_rate_definition"] != AGENTX_ERROR_RATE_DEFINITION:
            raise ReportError(
                f"the AgentX error rate definition {fields['error_rate_definition']!r} "
                f"is not {AGENTX_ERROR_RATE_DEFINITION}"
            )
        return cls(
            counts=dict(_complete(fields["counts"], AGENTX_COUNT_KEYS, "the AgentX counts")),
            throughput=dict(_complete(fields["throughput"], AGENTX_THROUGHPUT_KEYS, "the AgentX throughput")),
            latency_seconds=dict(_complete(fields["latency_seconds"], AGENTX_LATENCY_KEYS, "the AgentX latency")),
            e2e_norm_intvty=dict(_complete(fields["e2e_norm_intvty"], AGENTX_INTVTY_KEYS, "the AgentX intvty")),
            stopped_by_threshold=fields["stopped_by_threshold"],
            fast_mode=fields["fast_mode"],
            request_export=fields["request_export"],
            pairing_key=fields["pairing_key"],
        )


_SCRIPTABLE_RESULT = "the scriptable result"


@dataclass(frozen=True)
class ScriptableResult:
    """The scriptable script's own result document, kept verbatim; the file it was read from is run_dir-relative."""

    result_file: str
    result: Mapping[str, object]

    def __post_init__(self) -> None:
        _FIELDS.text(self.result_file, "the scriptable result file")
        _FIELDS.mapping(self.result, _SCRIPTABLE_RESULT)

    def to_document(self) -> dict[str, object]:
        return {"result_file": self.result_file, "result": dict(self.result)}

    @classmethod
    def from_document(cls, document: object) -> ScriptableResult:
        fields = _complete(document, field_names(cls), _SCRIPTABLE_RESULT)
        return cls(fields["result_file"], dict(_FIELDS.mapping(fields["result"], _SCRIPTABLE_RESULT)))


_MLPERF_RESULT = "the MLPerf result"


@dataclass(frozen=True)
class MlperfResult:
    """The MLPerf harness's run: the files it wrote, run_dir-relative, and their mapping into the InferenceX result
    schema, which ``result_file`` holds. ``scores_file`` is ``None`` when the harness scored nothing inline."""

    summary_file: str
    scores_file: str | None
    result_file: str
    result: Mapping[str, object]

    def __post_init__(self) -> None:
        _FIELDS.text(self.summary_file, "the MLPerf summary file")
        _FIELDS.optional_text(self.scores_file, "the MLPerf scores file")
        _FIELDS.text(self.result_file, "the MLPerf result file")
        _FIELDS.mapping(self.result, _MLPERF_RESULT)

    def to_document(self) -> dict[str, object]:
        return {
            "summary_file": self.summary_file,
            "scores_file": self.scores_file,
            "result_file": self.result_file,
            "result": dict(self.result),
        }

    @classmethod
    def from_document(cls, document: object) -> MlperfResult:
        fields = _complete(document, field_names(cls), _MLPERF_RESULT)
        result = dict(_FIELDS.mapping(fields["result"], _MLPERF_RESULT))
        return cls(fields["summary_file"], fields["scores_file"], fields["result_file"], result)


@dataclass(frozen=True)
class EvalDefinition:
    """Everything an eval score depends on; two scores compare only under the same definition."""

    task_yaml_sha256: str
    lm_eval_version: str
    model_class_sha256: str
    chat_template_sha256: str | None
    concurrency: int
    max_length: int
    max_tokens_cap: int
    limit: int | None
    num_fewshot: int
    golden_applied: bool
    probe: EvalProbe

    def __post_init__(self) -> None:
        _FIELDS.sha256(self.task_yaml_sha256, "the eval task yaml sha256")
        _FIELDS.text(self.lm_eval_version, "the lm_eval version")
        _FIELDS.sha256(self.model_class_sha256, "the eval model class sha256")
        _FIELDS.optional_sha256(self.chat_template_sha256, "the chat template sha256")
        _FIELDS.integer(self.concurrency, "the eval concurrency", minimum=1)
        _FIELDS.integer(self.max_length, "the eval max_length", minimum=1)
        _FIELDS.integer(self.max_tokens_cap, "the eval max_tokens_cap", minimum=1)
        _FIELDS.optional_integer(self.limit, "the eval limit", minimum=1)
        _FIELDS.integer(self.num_fewshot, "the eval num_fewshot", minimum=0)
        _FIELDS.boolean(self.golden_applied, "the eval golden_applied")
        if not isinstance(self.probe, EvalProbe):
            raise ReportError(f"the eval definition's probe must be an EvalProbe, got {self.probe!r}")

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {key: getattr(self, key) for key in _DEFINITION_KEYS}
        document["probe"] = self.probe.to_document()
        return document

    @classmethod
    def from_document(cls, document: object) -> EvalDefinition:
        fields = _complete(document, _DEFINITION_KEYS, "the eval definition")
        try:
            probe = EvalProbe.from_document(fields["probe"])
        except ConfigError as exc:
            raise ReportError(f"the eval definition's probe: {exc}") from exc
        return cls(**{key: fields[key] for key in _DEFINITION_KEYS if key != "probe"}, probe=probe)


_DEFINITION_KEYS = field_names(EvalDefinition)


@dataclass(frozen=True)
class ProbeOutcome:
    """What the truncation probe saw: raw counts, judged against the thresholds in the definition."""

    tripped: bool
    samples_seen: int
    length_ratio_observed: float | None

    def __post_init__(self) -> None:
        _FIELDS.boolean(self.tripped, "the probe's tripped")
        _FIELDS.integer(self.samples_seen, "the probe's samples_seen", minimum=0)
        _FIELDS.optional_number(self.length_ratio_observed, "the probe's observed length ratio")

    def to_document(self) -> dict[str, object]:
        return {
            "tripped": self.tripped,
            "samples_seen": self.samples_seen,
            "length_ratio_observed": self.length_ratio_observed,
        }

    @classmethod
    def from_document(cls, document: object) -> ProbeOutcome:
        fields = _complete(document, field_names(cls), "the probe outcome")
        return cls(fields["tripped"], fields["samples_seen"], fields["length_ratio_observed"])


@dataclass(frozen=True)
class EvalResult:
    task: str
    scores: Mapping[str, float]
    task_version: str | None
    definition: EvalDefinition
    definition_digest: str
    results_file: str | None
    probe: ProbeOutcome

    def __post_init__(self) -> None:
        _FIELDS.text(self.task, "the eval task")
        scores = _FIELDS.mapping(self.scores, "the eval scores")
        for name, score in scores.items():
            _FIELDS.text(name, "an eval score name")
            _FIELDS.number(score, f"the eval score {name}")
        _FIELDS.optional_text(self.task_version, "the eval task version")
        if not isinstance(self.definition, EvalDefinition):
            raise ReportError(f"the eval definition must be an EvalDefinition, got {self.definition!r}")
        _FIELDS.matching(_DIGEST, self.definition_digest, "the eval definition digest")
        _FIELDS.optional_text(self.results_file, "the eval results file")
        if not isinstance(self.probe, ProbeOutcome):
            raise ReportError(f"the eval probe outcome must be a ProbeOutcome, got {self.probe!r}")

    def to_document(self) -> dict[str, object]:
        return {
            "task": self.task,
            "scores": dict(self.scores),
            "task_version": self.task_version,
            "definition": self.definition.to_document(),
            "definition_digest": self.definition_digest,
            "results_file": self.results_file,
            "probe": self.probe.to_document(),
        }

    @classmethod
    def from_document(cls, document: object) -> EvalResult:
        fields = _complete(document, field_names(cls), "the eval result")
        return cls(
            task=fields["task"],
            scores=dict(_FIELDS.mapping(fields["scores"], "the eval scores")),
            task_version=fields["task_version"],
            definition=EvalDefinition.from_document(fields["definition"]),
            definition_digest=fields["definition_digest"],
            results_file=fields["results_file"],
            probe=ProbeOutcome.from_document(fields["probe"]),
        )


_PROFILE_RESULT_KEYS = (
    "trace_dirs",
    "has_step_traces",
    "window_open_unix",
    "window_close_unix",
    "running_requests_in_window",
)


@dataclass(frozen=True)
class ProfileResult:
    """The profiler's capture. No step trace is ``has_step_traces=False``, not a failed round."""

    trace_dirs: tuple[str, ...]
    has_step_traces: bool
    window_open_unix: float | None = None
    window_close_unix: float | None = None
    running_requests_in_window: int | None = None

    def __post_init__(self) -> None:
        _text_tuple(self.trace_dirs, "the profile trace dirs")
        _FIELDS.boolean(self.has_step_traces, "the profile's has_step_traces")
        opened = _FIELDS.optional_number(self.window_open_unix, "the profile window open")
        closed = _FIELDS.optional_number(self.window_close_unix, "the profile window close")
        if opened is not None and closed is not None and closed < opened:
            raise ReportError(f"the profile window closes at {closed} before it opens at {opened}")
        _optional_count(self.running_requests_in_window, "the running requests in the profile window")

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {key: getattr(self, key) for key in _PROFILE_RESULT_KEYS}
        document["trace_dirs"] = list(self.trace_dirs)
        return document

    @classmethod
    def from_document(cls, document: object) -> ProfileResult:
        fields = _complete(document, _PROFILE_RESULT_KEYS, "the profile result")
        trace_dirs = _FIELDS.strings(fields["trace_dirs"], "the profile trace dirs")
        return cls(**{**{key: fields[key] for key in _PROFILE_RESULT_KEYS}, "trace_dirs": trace_dirs})


@dataclass(frozen=True)
class ExitCodes:
    """The return codes of the round's processes; ``None`` for one that never ran or was not waited on."""

    server: int | None = None
    client: int | None = None
    eval: int | None = None

    def __post_init__(self) -> None:
        for name in ("server", "client", "eval"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ReportError(f"the {name} return code must be an integer or null, got {value!r}")

    def to_document(self) -> dict[str, object]:
        return {"server": self.server, "client": self.client, "eval": self.eval}

    @classmethod
    def from_document(cls, document: object) -> ExitCodes:
        fields = _complete(document, field_names(cls), "the return codes")
        return cls(fields["server"], fields["client"], fields["eval"])


@dataclass(frozen=True)
class LogRange:
    """The bytes of the server's live log this round covers, copied into the run_dir's ``server.log``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        _FIELDS.integer(self.start, "the log range start", minimum=0)
        _FIELDS.integer(self.end, "the log range end", minimum=self.start)

    def to_document(self) -> dict[str, object]:
        return {"start": self.start, "end": self.end}

    @classmethod
    def from_document(cls, document: object) -> LogRange:
        fields = _complete(document, field_names(cls), "the log range")
        return cls(fields["start"], fields["end"])


_SERVER_KEYS = ("host", "port", "served_model_name", "reused", "log_range")


@dataclass(frozen=True)
class ServerFacts:
    host: str
    port: int
    served_model_name: str
    reused: bool
    log_range: LogRange | None = None

    def __post_init__(self) -> None:
        _FIELDS.text(self.host, "the server host")
        if _FIELDS.integer(self.port, "the server port", minimum=1) > _MAX_PORT:
            raise ReportError(f"the server port must be at most {_MAX_PORT}, got {self.port}")
        _FIELDS.text(self.served_model_name, "the served model name")
        _FIELDS.boolean(self.reused, "the server's reused")
        if self.log_range is not None and not isinstance(self.log_range, LogRange):
            raise ReportError(f"the server log range must be a LogRange, got {self.log_range!r}")

    def to_document(self) -> dict[str, object]:
        document: dict[str, object] = {key: getattr(self, key) for key in _SERVER_KEYS}
        document["log_range"] = None if self.log_range is None else self.log_range.to_document()
        return document

    @classmethod
    def from_document(cls, document: object) -> ServerFacts:
        fields = _complete(document, _SERVER_KEYS, "the server facts")
        log_range = None if fields["log_range"] is None else LogRange.from_document(fields["log_range"])
        return cls(**{**{key: fields[key] for key in _SERVER_KEYS}, "log_range": log_range})


REPORT_KEYS = (
    "schema",
    "run_status",
    "failed_step",
    "failure_detail",
    "codes",
    "base_digest",
    "launch_digest",
    "workload_digest",
    "workload_kind",
    "concurrency",
    "started_unix",
    "ended_unix",
    "server",
    "gpu_monitor",
    "fixed",
    "agentx",
    "scriptable",
    "mlperf",
    "eval",
    "profile",
)
_WORKLOAD_SECTIONS = {
    WorkloadKind.FIXED: "fixed",
    WorkloadKind.AGENTX: "agentx",
    WorkloadKind.SCRIPTABLE: "scriptable",
    WorkloadKind.MLPERF: "mlperf",
}
_SECTION_TYPES: dict[str, type] = {
    "fixed": FixedMetrics,
    "agentx": AgentxResult,
    "scriptable": ScriptableResult,
    "mlperf": MlperfResult,
    "eval": EvalResult,
    "profile": ProfileResult,
}


@dataclass(frozen=True)
class Report:
    """One round's report. A completed round names no failed step; any other status names exactly one."""

    run_status: RunStatus
    workload_kind: WorkloadKind
    started_unix: float
    ended_unix: float
    codes: ExitCodes = field(default_factory=ExitCodes)
    failed_step: FailedStep | None = None
    failure_detail: str | None = None
    base_digest: str | None = None
    launch_digest: str | None = None
    workload_digest: str | None = None
    concurrency: int | None = None
    server: ServerFacts | None = None
    gpu_monitor: tuple[GpuMonitorBlock, ...] = ()
    fixed: FixedMetrics | None = None
    agentx: AgentxResult | None = None
    scriptable: ScriptableResult | None = None
    mlperf: MlperfResult | None = None
    eval: EvalResult | None = None
    profile: ProfileResult | None = None

    def __post_init__(self) -> None:
        status = _FIELDS.enum(RunStatus, self.run_status, "the run status")
        object.__setattr__(self, "run_status", status)
        kind = _FIELDS.enum(WorkloadKind, self.workload_kind, "the workload kind")
        object.__setattr__(self, "workload_kind", kind)
        if self.failed_step is not None:
            object.__setattr__(self, "failed_step", _FIELDS.enum(FailedStep, self.failed_step, "the failed step"))
        if (status is RunStatus.COMPLETED) != (self.failed_step is None):
            raise ReportError(f"a {status.value} round names {self.failed_step!r} as its failed step")
        _FIELDS.optional_text(self.failure_detail, "the failure detail")
        if self.failure_detail is not None and self.failed_step is None:
            raise ReportError("a failure detail needs the failed step it explains")
        if not isinstance(self.codes, ExitCodes):
            raise ReportError(f"the return codes must be ExitCodes, got {self.codes!r}")
        self._check_identity()
        self._check_sections(kind)

    def _check_identity(self) -> None:
        _optional_digest(self.base_digest, "the base digest")
        _optional_digest(self.launch_digest, "the launch digest")
        _optional_digest(self.workload_digest, "the workload digest")
        _FIELDS.optional_integer(self.concurrency, "the concurrency", minimum=1)
        started = _FIELDS.number(self.started_unix, "the round start")
        if _FIELDS.number(self.ended_unix, "the round end") < started:
            raise ReportError(f"the round ends at {self.ended_unix} before it starts at {started}")
        if self.server is not None and not isinstance(self.server, ServerFacts):
            raise ReportError(f"the server facts must be ServerFacts, got {self.server!r}")

    def _check_sections(self, kind: WorkloadKind) -> None:
        if not isinstance(self.gpu_monitor, tuple) or not all(
            isinstance(block, GpuMonitorBlock) for block in self.gpu_monitor
        ):
            raise ReportError(f"gpu_monitor must be a tuple of GpuMonitorBlock, got {self.gpu_monitor!r}")
        _FIELDS.unique([block.gpu for block in self.gpu_monitor], "monitored GPU")
        for name, kind_type in _SECTION_TYPES.items():
            section = getattr(self, name)
            if section is not None and not isinstance(section, kind_type):
                raise ReportError(f"the {name} section must be a {kind_type.__name__}, got {section!r}")
        foreign = [
            name for other, name in _WORKLOAD_SECTIONS.items() if other is not kind and getattr(self, name) is not None
        ]
        if foreign:
            raise ReportError(f"a {kind.value} round carries the {', '.join(foreign)} section")
        own = _WORKLOAD_SECTIONS.get(kind)
        if self.run_status is RunStatus.COMPLETED and own is not None and getattr(self, own) is None:
            raise ReportError(f"a completed {kind.value} round carries its {own} section")

    def to_document(self) -> dict[str, object]:
        sections = {name: getattr(self, name) for name in _SECTION_TYPES}
        return {
            "schema": REPORT_SCHEMA,
            "run_status": self.run_status.value,
            "failed_step": None if self.failed_step is None else self.failed_step.value,
            "failure_detail": self.failure_detail,
            "codes": self.codes.to_document(),
            "base_digest": self.base_digest,
            "launch_digest": self.launch_digest,
            "workload_digest": self.workload_digest,
            "workload_kind": self.workload_kind.value,
            "concurrency": self.concurrency,
            "started_unix": self.started_unix,
            "ended_unix": self.ended_unix,
            "server": None if self.server is None else self.server.to_document(),
            "gpu_monitor": [block.to_document() for block in self.gpu_monitor],
            **{name: None if section is None else section.to_document() for name, section in sections.items()},
        }

    @classmethod
    def from_document(cls, document: object) -> Report:
        fields = _complete(document, REPORT_KEYS, "a round report")
        if fields["schema"] != REPORT_SCHEMA:
            raise ReportError(f"round report schema {fields['schema']!r} is not {REPORT_SCHEMA}")
        monitor = _FIELDS.items(fields["gpu_monitor"], "gpu_monitor")
        sections = {
            name: None if fields[name] is None else kind_type.from_document(fields[name])
            for name, kind_type in _SECTION_TYPES.items()
        }
        return cls(
            run_status=fields["run_status"],
            failed_step=fields["failed_step"],
            failure_detail=fields["failure_detail"],
            codes=ExitCodes.from_document(fields["codes"]),
            base_digest=fields["base_digest"],
            launch_digest=fields["launch_digest"],
            workload_digest=fields["workload_digest"],
            workload_kind=fields["workload_kind"],
            concurrency=fields["concurrency"],
            started_unix=fields["started_unix"],
            ended_unix=fields["ended_unix"],
            server=None if fields["server"] is None else ServerFacts.from_document(fields["server"]),
            gpu_monitor=tuple(GpuMonitorBlock.from_document(block) for block in monitor),
            **sections,
        )


__all__ = [
    "AGENTX_COUNT_KEYS",
    "AGENTX_ERROR_RATE_DEFINITION",
    "AGENTX_INTVTY_KEYS",
    "AGENTX_LATENCY_KEYS",
    "AGENTX_THROUGHPUT_KEYS",
    "FIXED_METRIC_KEYS",
    "GPU_METRIC_KEYS",
    "REPORT_KEYS",
    "REPORT_SCHEMA",
    "AgentxResult",
    "EvalDefinition",
    "EvalResult",
    "ExitCodes",
    "FailedStep",
    "FixedMetrics",
    "GpuMonitorBlock",
    "LogRange",
    "MetricStats",
    "MlperfResult",
    "ProbeOutcome",
    "ProfileResult",
    "Report",
    "ReportError",
    "RunStatus",
    "ScriptableResult",
    "ServerFacts",
]
