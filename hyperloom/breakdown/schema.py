"""Schema (TypedDict shape) for ``session_breakdown.json``.

Single contract between hyperloom (producer) and any downstream consumer
(dashboards, claw-stats-service, notebooks, KB ingest).

Design notes:
* All fields are NotRequired-by-convention: collectors return None/[]/{}
  when underlying artifacts are absent. Consumers treat missing data as
  "not available" — never fabricate values.
* JSON-serializable only; no dataclasses/enums in the wire shape.
* schema_version bumped on breaking changes only. New optional fields
  are additive.
"""

from __future__ import annotations

from typing import Any, TypedDict

SCHEMA_VERSION = "hyperloom.session_breakdown.v2"


# §1 Session metadata
class SessionMeta(TypedDict, total=False):
    session_id: str
    created_at_utc: str
    ended_at_utc: str
    stop_reason: str
    max_minutes: int
    elapsed_minutes: float
    host: str
    pid: int
    session_dir: str
    tick_count: int
    agent_count: int


# §2 Workload configuration
class WorkloadObjective(TypedDict, total=False):
    kind: str
    value: Any


class Workload(TypedDict, total=False):
    framework: str
    framework_version: str
    model_name: str
    model_path: str
    gpu_type: str
    tp: int | None
    concurrency: int | None
    isl: int | None
    osl: int | None
    precision: str
    objective: WorkloadObjective


# §3 Baseline
class BaselineResult(TypedDict, total=False):
    throughput_tok_s: float
    latency_mean_ms: float | None
    accuracy: float | None
    config_path: str | None
    benchmark_report_path: str | None
    timestamp: str


# §4 Final state
class FinalResult(TypedDict, total=False):
    throughput_tok_s: float | None
    cumulative_gain_pct: float
    validated: bool
    action_path: list[str]
    extra_args: str
    extra_envs: dict[str, Any]
    timestamp: str


# §5 Agent dispatch timeline
class AgentEvent(TypedDict, total=False):
    ts: str
    agent_id: str
    role: str
    status: str
    task_summary: str
    gpu_ids: list[int]
    runtime_s: float
    attempt: int
    failure_type: str | None
    error_snippet: str | None


# §6 Capability summary
class CapabilityEntry(TypedDict, total=False):
    status: str
    attempts: int
    keeps: int
    best_gain_pct: float | None
    reason: str


class CapabilitySummary(TypedDict, total=False):
    geak: CapabilityEntry
    oob: CapabilityEntry
    tracelens: CapabilityEntry
    magpie: CapabilityEntry
    specialist: CapabilityEntry


# §7 Kernel invocations (GEAK / OOB)
class KernelMetadata(TypedDict, total=False):
    name: str
    source_file: str
    gpu_pct: float | None
    shapes: list[dict[str, Any]]


class Invocation(TypedDict, total=False):
    kernel_id: str
    attempt_id: str
    ts: str
    backend: str
    kernel_metadata: KernelMetadata
    optimized_files: list[str]
    decision: str
    micro_speedup: float | None
    compile_passed: bool | None
    correctness_passed: bool | None
    error: str | None


# §8 Kernel lifecycle
class DetectedKernel(TypedDict, total=False):
    kernel_id: str
    name: str
    gpu_pct: float | None
    time_ms: float | None
    bottleneck: str
    source_file: str | None


class OptimizedKernel(TypedDict, total=False):
    kernel_id: str
    backend: str
    total_attempts: int
    successful_attempts: int
    best_micro_speedup: float | None
    last_decision: str


class AdoptedKernel(TypedDict, total=False):
    kernel_id: str
    patch_path: str
    e2e_gain_pct: float | None
    validated: bool
    adopted_at: str


class RejectedKernel(TypedDict, total=False):
    kernel_id: str
    reason: str
    attempt_count: int
    best_gain_pct: float | None


class KernelLifecycle(TypedDict, total=False):
    detected: list[DetectedKernel]
    optimized: list[OptimizedKernel]
    adopted: list[AdoptedKernel]
    rejected: list[RejectedKernel]


# §9 Profiling summary
class ProfilingSummary(TypedDict, total=False):
    total_gpu_time_us: float
    hot_kernel_count: int
    top_kernels: list[dict[str, Any]]
    trace_path: str | None
    tracelens_analysis_path: str | None


# §10 Benchmark sweep
class SweepPoint(TypedDict, total=False):
    variant_name: str
    concurrency: int | None
    isl: int | None
    osl: int | None
    throughput_tok_s: float | None
    latency_mean_ms: float | None
    status: str


class Sweep(TypedDict, total=False):
    grid_size: int
    best_overall: dict[str, Any]
    pareto_front: list[dict[str, Any]]
    all_variants: list[SweepPoint]


# §11 Watchdog events
class WatchdogEvent(TypedDict, total=False):
    ts: str
    severity: str
    category: str
    summary: str
    action_taken: str | None


# §12 Attribution
class StackGainEntry(TypedDict, total=False):
    ts: str
    action: str
    variant_name: str | None
    delta_pct: float | None
    cum_gain_after: float


class SourceBreakdown(TypedDict, total=False):
    geak_pct_of_total: float
    oob_pct_of_total: float
    framework_pct_of_total: float
    config_pct_of_total: float


class Attribution(TypedDict, total=False):
    gain_per_entry: list[StackGainEntry]
    source_breakdown: SourceBreakdown
    method: str
    notes: list[str]


# §13 Telemetry
class Telemetry(TypedDict, total=False):
    gpu_count: int
    gpu_type: str
    avg_power_w: float | None
    max_temp_c: float | None
    trace_paths: list[str]
    server_log_paths: list[str]


# §14 Source files
class SourceFiles(TypedDict, total=False):
    session_dir: str
    state_json: str | None
    manifests: list[str]
    agent_logs: list[str]
    patches: list[str]
    benchmark_reports: list[str]


# Top-level shape
class SessionBreakdown(TypedDict, total=False):
    schema_version: str
    exported_at_utc: str
    exporter_version: str

    session: SessionMeta
    workload: Workload
    baseline: BaselineResult
    final: FinalResult
    agent_timeline: list[AgentEvent]
    capability_summary: CapabilitySummary
    geak_invocations: list[Invocation]
    oob_invocations: list[Invocation]
    kernel_lifecycle: KernelLifecycle
    profiling: ProfilingSummary
    sweep: Sweep
    watchdog_events: list[WatchdogEvent]
    attribution: Attribution
    telemetry: Telemetry
    source_files: SourceFiles
    warnings: list[str]


__all__ = [
    "SCHEMA_VERSION",
    "AdoptedKernel",
    "AgentEvent",
    "Attribution",
    "BaselineResult",
    "CapabilityEntry",
    "CapabilitySummary",
    "DetectedKernel",
    "FinalResult",
    "Invocation",
    "KernelLifecycle",
    "KernelMetadata",
    "OptimizedKernel",
    "ProfilingSummary",
    "RejectedKernel",
    "SessionBreakdown",
    "SessionMeta",
    "SourceBreakdown",
    "SourceFiles",
    "StackGainEntry",
    "Sweep",
    "SweepPoint",
    "Telemetry",
    "WatchdogEvent",
    "Workload",
    "WorkloadObjective",
]
