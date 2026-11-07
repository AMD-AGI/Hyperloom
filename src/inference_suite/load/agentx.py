# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX load: the agentx-harness trace replay (``agentx profile``) InferenceX pins, against the round's server.

The harness runs from the venv staging built (:mod:`inference_suite.staging`), not through ``infx.bench agentic``,
which bootstraps a venv online. Every knob that shapes the replay is passed: the trace corpus, the concurrency, the
request count, the seed, the warmup requests per lane and the failed-request threshold, which is the caller's single
value. The harness draws its warmup from the same seed, so a workload whose warmup seed differs is refused rather than
recorded as a seed nothing used; a lane warmup of zero is refused because the harness would then apply its own.

The phase clock follows aiperf's progress API: warmup and measured begin at each phase's ``start_ns``, finished at the
profiling phase's ``requests_end_ns`` or the client's exit. Each reading is handed to the round's listeners (the profile
window). After the replay InferenceX's ``process_agentic_result`` aggregates the export; the counts are recounted from
``profile_export.jsonl`` and a disagreement with the aggregate's accounting fails the round's output.
"""

from __future__ import annotations

import http.client
import json
import time
import urllib.error
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference_suite.contracts.config import AGENTX_FAST_MODE_ENV, AgentxWorkload, ServerConfig, SuiteConfig
from inference_suite.contracts.launch import ClientInvocation
from inference_suite.contracts.layout import RunLayout
from inference_suite.contracts.report import AgentxResult, FailedStep, ReportError
from inference_suite.load import exit_stop, inherited_env, read_result
from inference_suite.load.fixed import OutputActivity
from inference_suite.runtime.phases import Phase, PhaseClock
from inference_suite.runtime.process import spawn
from inference_suite.runtime.server import Server, server_url
from inference_suite.runtime.watch import RoundGate, Stop, wait_child
from inference_suite.staging import AGENTX_AGGREGATOR_MODULE, AGENTX_CLI_MODULE, AGENTX_PYTHON
from inference_suite.telemetry.kv_poll import json_number, read_aiperf_phases

AGENTX_SCENARIO = "agentx"
#: aiperf's progress API listens this far above the server's port: rounds sharing a host differ in server port, so
#: their APIs stay apart, and the recorded argv does not depend on which port happened to be free.
PROGRESS_API_PORT_OFFSET = 1000
PROGRESS_API_HOST = "127.0.0.1"
#: What the aggregator names its result, under the client dir: the run_dir's ``client/inferencex_result.json``.
AGGREGATE_NAME = "inferencex_result"
#: The line the harness logs when the failed-request threshold stops the replay.
THRESHOLD_STOP_MARKER = "--failed-request-threshold exceeded"
WARMUP_PHASE = "warmup"
PROFILING_PHASE = "profiling"
_WHAT = "the AgentX client"
_AGGREGATOR = "the AgentX aggregator"
_NS_PER_SEC = 1e9
_PROGRESS_ERRORS = (urllib.error.URLError, http.client.HTTPException, OSError, ValueError)

#: Each :class:`AgentxResult` throughput figure, by its path in the aggregate's ``request_metrics``.
_THROUGHPUT_PATHS: Mapping[str, tuple[str, ...]] = {
    "request_throughput": ("qps", "mean"),
    **{f"{kind}_tokens_per_second": ("throughput", kind, "tokens_per_second") for kind in ("output", "input", "total")},
    "duration_seconds": ("throughput", "duration_seconds"),
}
_LATENCY_NAMES = ("ttft", "tpot", "e2el")
_LATENCY_STATS = ("mean", "p50", "p90", "p99")
_INTVTY_STATS = ("p50", "p90")
#: The aggregate's own accounting, by the count it must equal.
_ACCOUNTING = {
    "records_total": "records_total",
    "warmup_dropped": "records_warmup_dropped",
    "error_dropped": "records_error_dropped",
    "dropped_total": "records_dropped_total",
}

#: Hears each progress reading (``None`` when the API did not answer) with the wall-clock instant it was read.
ProgressListener = Callable[[Mapping[str, object] | None, float], None]


@dataclass(frozen=True)
class AgentxTool:
    """The staged harness: the venv interpreter it runs under and the InferenceX root its aggregator runs from."""

    python: str
    inferencex_root: str

    @classmethod
    def staged(cls, inferencex_root: Path) -> AgentxTool:
        return cls(str(inferencex_root / AGENTX_PYTHON), str(inferencex_root))


def progress_api_port(server: ServerConfig) -> int:
    above = server.port + PROGRESS_API_PORT_OFFSET
    return above if above <= 65535 else server.port - PROGRESS_API_PORT_OFFSET


def _workload(config: SuiteConfig) -> AgentxWorkload:
    if config.agentx is None:
        raise ValueError(f"the AgentX load needs an AgentX workload, got {config.workload.value}")
    return config.agentx


def refusal(workload: AgentxWorkload) -> Stop | None:
    """Why the harness cannot run ``workload`` as written, if it cannot."""
    if workload.warmup_seed != workload.seed:
        return Stop(
            FailedStep.CONFIG_INVALID,
            f"the AgentX harness draws its warmup from --random-seed, so warmup_seed {workload.warmup_seed} "
            f"cannot differ from seed {workload.seed}",
        )
    if workload.warmup_requests_per_lane == 0:
        return Stop(
            FailedStep.CONFIG_INVALID,
            "the AgentX harness takes at least one warmup request per lane; without one it applies its own default",
        )
    return None


def client_argv(tool: AgentxTool, config: SuiteConfig) -> tuple[str, ...]:
    """The ``agentx profile`` argv: every replay knob explicit, the tokenizer read from the served model's path."""
    workload = _workload(config)
    server = config.server
    base_url = server_url(server)
    artifacts = RunLayout(Path(config.paths.run_dir)).aiperf_artifacts
    return (
        tool.python,
        "-I",
        "-m",
        AGENTX_CLI_MODULE,
        "profile",
        "--scenario",
        AGENTX_SCENARIO,
        "--url",
        base_url,
        "--model",
        server.served_model_name,
        "--tokenizer",
        server.model_path,
        "--tokenizer-trust-remote-code",
        "--public-dataset",
        workload.dataset,
        "--concurrency",
        str(workload.concurrency),
        "--request-count",
        str(workload.num_prompts),
        "--random-seed",
        str(workload.seed),
        "--warmup-requests-per-lane",
        str(workload.warmup_requests_per_lane),
        "--failed-request-threshold",
        str(workload.failed_request_threshold),
        "--server-metrics",
        f"{base_url}/metrics",
        "--output-artifact-dir",
        str(artifacts),
        "--api-host",
        PROGRESS_API_HOST,
        "--api-port",
        str(progress_api_port(server)),
    )


def measured_invocation(config: SuiteConfig, tool: AgentxTool) -> ClientInvocation:
    """The replay as ``launch.json`` records it, before the round's server is up."""
    workload = _workload(config)
    return ClientInvocation(
        argv=client_argv(tool, config),
        env=dict(workload.client_env),
        seed=workload.seed,
        warmup_seed=workload.warmup_seed,
    )


def _ns_to_mono(start_ns: float) -> float:
    return time.monotonic() - (time.time() - start_ns / _NS_PER_SEC)


def _phase_instant(phases: Mapping[str, object], phase: str, key: str) -> float | None:
    stats = phases.get(phase)
    value = json_number(stats.get(key)) if isinstance(stats, Mapping) else None
    return None if value is None or value <= 0 else value


class ProgressFollower:
    """Marks the round's phases from aiperf's progress API and hands each reading to the listeners."""

    _MARKS = (
        (WARMUP_PHASE, "start_ns", Phase.WARMUP),
        (PROFILING_PHASE, "start_ns", Phase.MEASURED),
        (PROFILING_PHASE, "requests_end_ns", Phase.FINISHED),
    )

    def __init__(self, url: str, clock: PhaseClock, listeners: tuple[ProgressListener, ...]) -> None:
        self._url = url
        self._clock = clock
        self._listeners = listeners

    def check(self, now: float) -> None:
        try:
            phases = read_aiperf_phases(self._url)
        except _PROGRESS_ERRORS:
            phases = None
        if phases is not None:
            for phase, key, mark in self._MARKS:
                instant = _phase_instant(phases, phase, key)
                if instant is not None:
                    self._clock.mark(mark, _ns_to_mono(instant))
        wall = time.time()
        for listener in self._listeners:
            listener(phases, wall)


def _export_lines(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as export:
        for number, line in enumerate(export, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"line {number} of {path} is not a request record")
            yield record


def export_counts(path: Path) -> dict[str, int]:
    """The request counts of the replay's per-request export, the way InferenceX's aggregator counts them.

    A warmup record that failed is both warmup- and error-dropped, so ``dropped_total`` is their union: every record
    that is not a successful profiling one.

    Raises:
        OSError, ValueError: The export cannot be read, or a line of it is not a request record.
    """
    counts = dict.fromkeys(("records_total", "successful", "error_dropped", "warmup_dropped", "error_osl_exported"), 0)
    for record in _export_lines(path):
        metadata = record.get("metadata")
        phase = metadata.get("benchmark_phase") if isinstance(metadata, Mapping) else None
        warmup = phase is not None and phase != PROFILING_PHASE
        error = bool(record.get("error"))
        metrics = record.get("metrics")
        counts["records_total"] += 1
        counts["warmup_dropped"] += warmup
        counts["error_dropped"] += error
        counts["successful"] += not (warmup or error)
        counts["error_osl_exported"] += error and isinstance(metrics, Mapping) and "error_osl" in metrics
    counts["dropped_total"] = counts["records_total"] - counts["successful"]
    return counts


def _disagreement(counts: Mapping[str, int], aggregate: Mapping[str, Any]) -> str | None:
    accounting = aggregate.get("request_accounting")
    if not isinstance(accounting, Mapping):
        return "the aggregate carries no request_accounting"
    pairs = [(key, counts[key], accounting.get(raw)) for key, raw in _ACCOUNTING.items()]
    pairs.append(("successful", counts["successful"], aggregate.get("num_requests_successful")))
    differ = [f"{key} {mine} (aggregate {theirs!r})" for key, mine, theirs in pairs if mine != theirs]
    return "; ".join(differ) or None


def _at(document: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = document
    for key in path:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


@dataclass(frozen=True)
class ReplayFacts:
    """What the round knows of the replay beyond its aggregate: the recounted export and how the replay ran."""

    counts: Mapping[str, int]
    stopped_by_threshold: bool
    fast_mode: bool
    request_export: str


def agentx_result(aggregate: Mapping[str, Any], facts: ReplayFacts) -> AgentxResult:
    """The report's AgentX section from the aggregate's ``request_metrics`` and what the round knows of the replay.

    Raises:
        ReportError: A figure the aggregate carries is not a number.
    """
    metrics = _at(aggregate, ("request_metrics",))
    metrics = metrics if isinstance(metrics, Mapping) else {}
    latency = {
        f"{name}_{stat}": _at(metrics, ("latency", name, stat)) for name in _LATENCY_NAMES for stat in _LATENCY_STATS
    }
    return AgentxResult(
        counts=dict(facts.counts),
        throughput={key: _at(metrics, path) for key, path in _THROUGHPUT_PATHS.items()},
        latency_seconds=latency,
        e2e_norm_intvty={stat: _at(metrics, ("latency", "e2e_norm_intvty", stat)) for stat in _INTVTY_STATS},
        stopped_by_threshold=facts.stopped_by_threshold,
        fast_mode=facts.fast_mode,
        request_export=facts.request_export,
    )


def _stopped_by_threshold(layout: RunLayout) -> bool:
    for log in (layout.agentx_client_log, layout.aiperf_log):
        if log.is_file() and THRESHOLD_STOP_MARKER in log.read_text(encoding="utf-8", errors="replace"):
            return True
    return False


@dataclass(frozen=True)
class AgentxOutcome:
    """What the AgentX load did: the replay as it ran, its exit code, its result, and why it failed, if it did.

    A replay the failed-request threshold stopped exits non-zero after its export is aggregated, so it leaves both a
    result (``stopped_by_threshold``) and a stop.
    """

    invocation: ClientInvocation
    returncode: int | None
    result: AgentxResult | None
    stop: Stop | None


@dataclass(frozen=True)
class AgentxRound:
    """What the AgentX load runs in: the round's config, the staged harness, its server, gates, clock and listeners."""

    config: SuiteConfig
    tool: AgentxTool
    server: Server
    gate: RoundGate
    clock: PhaseClock
    listeners: tuple[ProgressListener, ...] = ()


def _aggregator_env(agentx_round: AgentxRound, layout: RunLayout) -> dict[str, str]:
    config = agentx_round.config
    return {
        **inherited_env(config.server),
        "PYTHONPATH": agentx_round.tool.inferencex_root,
        "RESULT_DIR": str(layout.client_dir),
        "AGENTIC_OUTPUT_DIR": str(layout.client_dir),
        "RESULT_FILENAME": AGGREGATE_NAME,
        "KV_OFFLOADING": "none",
        "CONC": str(_workload(config).concurrency),
        "MODEL": config.server.served_model_name,
    }


def _replay(
    agentx_round: AgentxRound, invocation: ClientInvocation, layout: RunLayout
) -> tuple[int | None, Stop | None]:
    config = agentx_round.config
    env = {**inherited_env(config.server), **invocation.env}
    progress = f"http://{PROGRESS_API_HOST}:{progress_api_port(config.server)}"
    follower = ProgressFollower(progress, agentx_round.clock, agentx_round.listeners)
    activity = OutputActivity(agentx_round.server, layout.agentx_client_log)

    def live(now: float) -> Stop | None:
        follower.check(now)
        return activity.check(now)

    child = spawn(invocation.argv, env, layout.agentx_client_log)
    ended = wait_child(child, agentx_round.gate, live)
    agentx_round.clock.mark(Phase.FINISHED)
    returncode = None if isinstance(ended, Stop) else ended.returncode
    return returncode, exit_stop(ended, _WHAT, layout.agentx_client_log)


def _aggregate(agentx_round: AgentxRound, layout: RunLayout) -> Mapping[str, Any] | Stop:
    argv = (agentx_round.tool.python, "-s", "-P", "-m", AGENTX_AGGREGATOR_MODULE)
    child = spawn(argv, _aggregator_env(agentx_round, layout), layout.client_stdout, layout.client_stderr)
    ended = wait_child(child, agentx_round.gate)
    stop = exit_stop(ended, _AGGREGATOR, layout.client_stderr)
    if stop is not None:
        return (
            stop if stop.step is not FailedStep.CLIENT_FAILED else Stop(FailedStep.CLIENT_OUTPUT_MISSING, stop.detail)
        )
    return read_result(layout.inferencex_result, _AGGREGATOR)


def _result(agentx_round: AgentxRound, layout: RunLayout) -> AgentxResult | Stop:
    aggregate = _aggregate(agentx_round, layout)
    if isinstance(aggregate, Stop):
        return aggregate
    try:
        counts = export_counts(layout.aiperf_request_export)
    except (OSError, ValueError) as exc:
        return Stop(FailedStep.CLIENT_OUTPUT_MISSING, f"the AgentX export cannot be counted: {exc}")
    disagreement = _disagreement(counts, aggregate)
    if disagreement is not None:
        return Stop(FailedStep.CLIENT_OUTPUT_MISSING, f"the AgentX counts disagree with the aggregate: {disagreement}")
    facts = ReplayFacts(
        counts=counts,
        stopped_by_threshold=_stopped_by_threshold(layout),
        fast_mode=_workload(agentx_round.config).client_env[AGENTX_FAST_MODE_ENV] != "0",
        request_export=layout.aiperf_request_export.relative_to(layout.root).as_posix(),
    )
    try:
        return agentx_result(aggregate, facts)
    except ReportError as exc:
        return Stop(FailedStep.CLIENT_OUTPUT_MISSING, str(exc))


def run_agentx(agentx_round: AgentxRound) -> AgentxOutcome:
    """Replay the traces, then aggregate and count the export the replay left."""
    config = agentx_round.config
    layout = RunLayout(Path(config.paths.run_dir))
    invocation = measured_invocation(config, agentx_round.tool)
    stop = refusal(_workload(config))
    if stop is not None:
        return AgentxOutcome(invocation, None, None, stop)
    returncode, stop = _replay(agentx_round, invocation, layout)
    if stop is not None and stop.step is not FailedStep.CLIENT_FAILED:
        return AgentxOutcome(invocation, returncode, None, stop)
    result = _result(agentx_round, layout)
    if isinstance(result, Stop):
        return AgentxOutcome(invocation, returncode, None, stop or result)
    return AgentxOutcome(invocation, returncode, result, stop)


__all__ = [
    "AGENTX_SCENARIO",
    "AgentxOutcome",
    "AgentxRound",
    "AgentxTool",
    "ProgressFollower",
    "ProgressListener",
    "ReplayFacts",
    "agentx_result",
    "client_argv",
    "export_counts",
    "measured_invocation",
    "progress_api_port",
    "refusal",
    "run_agentx",
]
