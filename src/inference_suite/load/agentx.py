# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The AgentX load: the agentx-harness trace replay (``agentx profile``) InferenceX pins, against the round's server.

The harness runs under the client environment staging built (:mod:`inference_suite.client_env`), as a subprocess of its
own pinned interpreter, not through ``infx.bench agentic``, which bootstraps a venv online. Every knob that shapes the
replay is passed: the trace corpus, the concurrency, the profiling duration, the seed, the warmup requests per lane and
the failed-request threshold, which is the caller's single value. The replay is bounded by its duration, as
InferenceX's ``replay.py`` runs it: the agentx scenario holds the profiling phase to 95% latency coverage of that
duration, so a request count would end it short and fail. The harness draws its warmup from the same seed, so a workload
whose warmup seed differs is refused rather than recorded as a seed nothing used; a lane warmup of zero is refused
because the harness would then apply its own.

The phase clock follows aiperf's progress API: warmup and measured begin at each phase's ``start_ns``, finished at the
profiling phase's ``requests_end_ns`` or the client's exit. The round's listeners (the profile window) hear every poll
after the clock is marked, and once more after the client exits, so they act on the finished mark before the export is
aggregated. After the replay InferenceX's ``process_agentic_result`` aggregates the export; the counts are recounted from
``profile_export.jsonl`` and a disagreement with the aggregate's accounting fails the round's output. The requests aiperf
skipped as longer than the context the server serves are in neither, only in aiperf's summary, so their count is read
from there.

Building the replay's dataset (tokenizer, trace corpus, mmap files) takes the client minutes. While the round's server
boots, :class:`DatasetPrep` runs the same client against no server to build it into aiperf's dataset cache in the
run_dir, which the replay reads; the report records whether it did. The round removes the cache however it ends.
"""

from __future__ import annotations

import http.client
import json
import shutil
import threading
import time
import urllib.error
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference_suite.aiperf_patch import ERROR_OSL_METRIC
from inference_suite.contracts.config import AGENTX_FAST_MODE_ENV, AgentxWorkload, ServerConfig, SuiteConfig
from inference_suite.contracts.launch import ClientInvocation
from inference_suite.contracts.layout import RunLayout
from inference_suite.contracts.report import AGENTX_CONTEXT_OVERFLOW, AgentxResult, FailedStep, ReportError
from inference_suite.load import exit_stop, inherited_env, read_result, server_tp
from inference_suite.load.fixed import OutputActivity
from inference_suite.runtime.phases import Phase, PhaseClock
from inference_suite.runtime.process import Spawned, spawn
from inference_suite.runtime.server import Server, server_url
from inference_suite.runtime.watch import POLL_SEC, RoundGate, Stop, wait_child
from inference_suite.staging import AGENTX_AGGREGATOR_MODULE, AGENTX_CLI_MODULE, AGENTX_PYTHON
from inference_suite.telemetry.kv_poll import json_number, read_aiperf_phases

AGENTX_SCENARIO = "agentx"
#: aiperf's progress API listens this far above the server's port: rounds sharing a host differ in server port, so
#: their APIs stay apart, and the recorded argv does not depend on which port happened to be free.
PROGRESS_API_PORT_OFFSET = 1000
PROGRESS_API_HOST = "127.0.0.1"
#: What the aggregator names its result, under the client dir: the run_dir's ``client/inferencex_result.json``.
AGGREGATE_NAME = "inferencex_result"
#: aiperf's summary export, beside the per-request one.
_AIPERF_SUMMARY = "profile_export_aiperf.json"
#: The summary's count of the profiling requests aiperf skipped as a context overflow, ``{"unit": ..., "avg": n}``; it
#: writes the key only when it skipped one (agentx-harness 1.0.13, ``exporters/metrics_json_exporter.py``).
_SKIPPED_OVERFLOW = "skipped_context_overflow_count"
#: The line the harness logs when the failed-request threshold stops the replay.
THRESHOLD_STOP_MARKER = "--failed-request-threshold exceeded"
#: aiperf's content-addressed dataset cache (agentx-harness 1.0.13, ``aiperf/dataset/mmap_cache.py``). The key is the
#: dataset, tokenizer, seed and dataset settings, never the URL, the server metrics or the progress API, so a run
#: against no server fills the entry the replay reads. An entry's ``manifest.json`` is written last, before the whole
#: entry is renamed into place.
_DATASET_CACHE_ENV = "AIPERF_DATASET_MMAP_CACHE_DIR"
_CACHE_ENTRY_MANIFEST = "*/manifest.json"
#: Where aiperf writes a run's own mmap files; the system temp dir by default, where a reaped prep would leave them.
_MMAP_SCRATCH_ENV = "AIPERF_DATASET_MMAP_BASE_PATH"
#: What aiperf logs when it restores the dataset from the cache (``aiperf/dataset/dataset_manager.py``).
_CACHE_HIT_MARKER = "Memory-mapped dataset cache HIT"
#: The discard port: the prep's requests are refused, and it is reaped once its dataset is built.
_NO_SERVER_URL = "http://127.0.0.1:9"
WARMUP_PHASE = "warmup"
PROFILING_PHASE = "profiling"
#: The agentx scenario's ``min_benchmark_duration_seconds`` (agentx-harness 1.0.13); a shorter replay needs
#: ``--unsafe-override``, which InferenceX's ``replay.py`` adds the same way.
_SCENARIO_MIN_DURATION_SEC = 900
_UTF8 = "utf-8"
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

#: The report's token counts, by the aiperf per-request metric each sums.
_TOKEN_METRICS = {"input_tokens": "input_sequence_length", "output_tokens": "output_sequence_length"}

#: Hears the replay at the monotonic instant of every poll, and once after the client exits.
ReplayListener = Callable[[float], None]


@dataclass(frozen=True)
class AgentxTool:
    """The staged harness: the client interpreter it runs under and the InferenceX root its aggregator runs from."""

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
    """Why the harness cannot run ``workload`` as written, if it cannot; the round asks before its server starts."""
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


def _profile_argv(tool: AgentxTool, config: SuiteConfig, url: str, outputs: tuple[str, ...]) -> tuple[str, ...]:
    workload = _workload(config)
    server = config.server
    return (
        tool.python,
        "-I",
        "-m",
        AGENTX_CLI_MODULE,
        "profile",
        "--scenario",
        AGENTX_SCENARIO,
        "--url",
        url,
        "--model",
        server.served_model_name,
        "--tokenizer",
        server.model_path,
        "--tokenizer-trust-remote-code",
        "--public-dataset",
        workload.dataset,
        "--concurrency",
        str(workload.concurrency),
        "--benchmark-duration",
        str(workload.duration_sec),
        "--random-seed",
        str(workload.seed),
        "--warmup-requests-per-lane",
        str(workload.warmup_requests_per_lane),
        "--failed-request-threshold",
        str(workload.failed_request_threshold),
        *outputs,
        *(("--unsafe-override",) if workload.duration_sec < _SCENARIO_MIN_DURATION_SEC else ()),
    )


def client_argv(tool: AgentxTool, config: SuiteConfig) -> tuple[str, ...]:
    """The ``agentx profile`` argv: every replay knob explicit, the tokenizer read from the served model's path."""
    base_url = server_url(config.server)
    outputs = (
        "--server-metrics",
        f"{base_url}/metrics",
        "--output-artifact-dir",
        str(RunLayout(Path(config.paths.run_dir)).aiperf_artifacts),
        "--api-host",
        PROGRESS_API_HOST,
        "--api-port",
        str(progress_api_port(config.server)),
    )
    return _profile_argv(tool, config, base_url, outputs)


def measured_invocation(config: SuiteConfig, tool: AgentxTool) -> ClientInvocation:
    """The replay as ``launch.json`` records it, before the round's server is up."""
    workload = _workload(config)
    cache = RunLayout(Path(config.paths.run_dir)).aiperf_dataset_cache
    return ClientInvocation(
        argv=client_argv(tool, config),
        env={**workload.client_env, _DATASET_CACHE_ENV: str(cache)},
        seed=workload.seed,
        warmup_seed=workload.warmup_seed,
    )


class DatasetPrep:
    """The replay's dataset, built into aiperf's cache by the replay's own client against no server.

    The prep is reaped once the cache holds an entry, or when :meth:`stop` is called, whichever comes first. A prep
    that failed or was stopped before its entry was whole leaves no cache, so the replay builds the dataset itself and
    meets no populate lock the reaped prep held.
    """

    def __init__(self, layout: RunLayout, child: Spawned) -> None:
        self._layout = layout
        self._child = child
        self._stopped = threading.Event()
        self._watcher = threading.Thread(target=self._watch, name="agentx-dataset-prep", daemon=True)
        self._watcher.start()

    @classmethod
    def start(cls, config: SuiteConfig, tool: AgentxTool) -> DatasetPrep:
        layout = RunLayout(Path(config.paths.run_dir))
        env = {
            **inherited_env(config.server),
            **measured_invocation(config, tool).env,
            _MMAP_SCRATCH_ENV: str(layout.agentx_prep_scratch / "mmap"),
        }
        outputs = ("--no-server-metrics", "--output-artifact-dir", str(layout.agentx_prep_scratch / "artifacts"))
        argv = _profile_argv(tool, config, _NO_SERVER_URL, outputs)
        return cls(layout, spawn(argv, env, layout.agentx_prep_log))

    def _built(self) -> bool:
        return any(self._layout.aiperf_dataset_cache.glob(_CACHE_ENTRY_MANIFEST))

    def _watch(self) -> None:
        while self._child.proc.poll() is None and not self._built():
            if self._stopped.wait(POLL_SEC):
                break
        self._child.reap()

    def stop(self) -> None:
        """Reap the prep if it still runs, keeping the cache only if it holds a whole entry."""
        self._stopped.set()
        self._watcher.join()
        if not self._built():
            shutil.rmtree(self._layout.aiperf_dataset_cache, ignore_errors=True)
        shutil.rmtree(self._layout.agentx_prep_scratch, ignore_errors=True)


def _ns_to_mono(start_ns: float) -> float:
    return time.monotonic() - (time.time() - start_ns / _NS_PER_SEC)


def _phase_instant(phases: Mapping[str, object], phase: str, key: str) -> float | None:
    stats = phases.get(phase)
    value = json_number(stats.get(key)) if isinstance(stats, Mapping) else None
    return None if value is None or value <= 0 else value


class ProgressFollower:
    """Marks the round's phases from aiperf's progress API."""

    _MARKS = (
        (WARMUP_PHASE, "start_ns", Phase.WARMUP),
        (PROFILING_PHASE, "start_ns", Phase.MEASURED),
        (PROFILING_PHASE, "requests_end_ns", Phase.FINISHED),
    )

    def __init__(self, url: str, clock: PhaseClock) -> None:
        self._url = url
        self._clock = clock

    def check(self) -> None:
        try:
            phases = read_aiperf_phases(self._url)
        except _PROGRESS_ERRORS:
            return
        if phases is None:
            return
        for phase, key, mark in self._MARKS:
            instant = _phase_instant(phases, phase, key)
            if instant is not None:
                self._clock.mark(mark, _ns_to_mono(instant))


def _export_lines(path: Path) -> Iterator[Mapping[str, Any]]:
    with path.open(encoding=_UTF8) as export:
        for number, line in enumerate(export, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise ValueError(f"line {number} of {path} is not a request record")
            yield record


def export_counts(path: Path) -> dict[str, int | None]:
    """The request counts of the replay's per-request export, the way InferenceX's aggregator counts them, and the
    input and output tokens of its successful profiling requests.

    A warmup record that failed is both warmup- and error-dropped, so ``dropped_total`` is their union: every record
    that is not a successful profiling one. A token count is ``None`` when a successful request carries no length.

    Raises:
        OSError, ValueError: The export cannot be read, or a line of it is not a request record.
    """
    counts = dict.fromkeys(("records_total", "successful", "error_dropped", "warmup_dropped", "error_osl_exported"), 0)
    tokens: dict[str, float | None] = dict.fromkeys(_TOKEN_METRICS, 0.0)
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
        counts["error_osl_exported"] += error and isinstance(metrics, Mapping) and ERROR_OSL_METRIC in metrics
        if not (warmup or error):
            _add_tokens(tokens, metrics)
    counts["dropped_total"] = counts["records_total"] - counts["successful"]
    return {**counts, **{key: None if total is None else int(total) for key, total in tokens.items()}}


def skipped_context_overflow(path: Path) -> int:
    """The profiling requests the replay skipped because the server refused them as longer than the context it serves,
    from aiperf's summary export at ``path``.

    Raises:
        OSError, ValueError: The summary cannot be read, is not a JSON object, or gives a count that is not one.
    """
    summary = json.loads(path.read_text(encoding=_UTF8))
    if not isinstance(summary, Mapping):
        raise ValueError(f"{path} is not a JSON object")
    metric = summary.get(_SKIPPED_OVERFLOW)
    if metric is None:
        return 0
    count = metric.get("avg") if isinstance(metric, Mapping) else None
    if isinstance(count, bool) or not isinstance(count, (int, float)) or count < 0 or count != int(count):
        raise ValueError(f"{path} gives {_SKIPPED_OVERFLOW} {metric!r}, not a count")
    return int(count)


def _add_tokens(tokens: dict[str, float | None], metrics: object) -> None:
    for key, name in _TOKEN_METRICS.items():
        entry = metrics.get(name) if isinstance(metrics, Mapping) else None
        value = entry.get("value") if isinstance(entry, Mapping) else entry
        total = tokens[key]
        valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        tokens[key] = total + value if total is not None and valid else None


def _disagreement(counts: Mapping[str, int | None], aggregate: Mapping[str, Any]) -> str | None:
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

    counts: Mapping[str, int | None]
    stopped_by_threshold: bool
    fast_mode: bool
    request_export: str
    dataset_cache_hit: bool


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
        dataset_cache_hit=facts.dataset_cache_hit,
    )


def _client_logged(layout: RunLayout, marker: str) -> bool:
    for log in (layout.agentx_client_log, layout.aiperf_log):
        if log.is_file() and marker in log.read_text(encoding=_UTF8, errors="replace"):
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
    listeners: tuple[ReplayListener, ...] = ()


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
        "TP": server_tp(config.server),
    }


def _replay(
    agentx_round: AgentxRound, invocation: ClientInvocation, layout: RunLayout
) -> tuple[int | None, Stop | None]:
    config = agentx_round.config
    env = {**inherited_env(config.server), **invocation.env}
    progress = f"http://{PROGRESS_API_HOST}:{progress_api_port(config.server)}"
    follower = ProgressFollower(progress, agentx_round.clock)
    activity = OutputActivity(agentx_round.server, layout.agentx_client_log)

    def hear(now: float) -> None:
        for listener in agentx_round.listeners:
            listener(now)

    def live(now: float) -> Stop | None:
        follower.check()
        # The follower's read can mark FINISHED after ``now``; a window closed at ``now`` would end before it.
        hear(time.monotonic())
        # A listener can block for minutes (/stop_profile answers once the trace is written): date output after it.
        return activity.check(time.monotonic())

    child = spawn(invocation.argv, env, layout.agentx_client_log)
    ended = wait_child(child, agentx_round.gate, live)
    agentx_round.clock.mark(Phase.FINISHED)
    hear(time.monotonic())
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
    try:
        counts[AGENTX_CONTEXT_OVERFLOW] = skipped_context_overflow(layout.aiperf_artifacts / _AIPERF_SUMMARY)
    except (OSError, ValueError) as exc:
        return Stop(FailedStep.CLIENT_OUTPUT_MISSING, f"the AgentX summary cannot be read: {exc}")
    facts = ReplayFacts(
        counts=counts,
        stopped_by_threshold=_client_logged(layout, THRESHOLD_STOP_MARKER),
        fast_mode=_workload(agentx_round.config).client_env[AGENTX_FAST_MODE_ENV] != "0",
        request_export=layout.aiperf_request_export.relative_to(layout.root).as_posix(),
        dataset_cache_hit=_client_logged(layout, _CACHE_HIT_MARKER),
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
    "DatasetPrep",
    "ProgressFollower",
    "ReplayListener",
    "ReplayFacts",
    "agentx_result",
    "client_argv",
    "export_counts",
    "measured_invocation",
    "progress_api_port",
    "refusal",
    "run_agentx",
]
