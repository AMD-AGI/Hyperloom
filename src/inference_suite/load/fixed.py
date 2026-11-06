# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The fixed-shape load: InferenceX's ``infx.bench_serving`` client against the round's server.

Every knob the client has a default for is passed, so two arms send the same requests whatever the client's defaults
become or whichever host runs it; a switch the load leaves off (``--use-chat-template``) is absent. The warmup is its own invocation on its own seed, with its result discarded; the measured invocation runs no
warmup of its own, so the measured window is the measured invocation and nothing else.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference_suite.contracts.config import FixedWorkload, ServerConfig, SuiteConfig
from inference_suite.contracts.launch import ClientInvocation
from inference_suite.contracts.layout import RunLayout
from inference_suite.contracts.report import FailedStep, FixedMetrics, ReportError
from inference_suite.load import exit_stop, inherited_env, read_result
from inference_suite.runtime.phases import Phase, PhaseClock
from inference_suite.runtime.process import spawn
from inference_suite.runtime.server import Server
from inference_suite.runtime.watch import RoundGate, Stop, wait_child
from inference_suite.staging import CLIENT_MODULE

#: The OpenAI-compatible request format every served framework answers.
CLIENT_BACKEND = "vllm"
CLIENT_ENDPOINT = "/v1/completions"
CLIENT_RANDOM_PREFIX_LEN = 0
#: Each prompt-generation worker draws its chunk's padding seed in turn, so the count shapes the prompts; the client's
#: own default (``min(cpu_count, 8)``) would make them depend on the client host.
CLIENT_RANDOM_NUM_WORKERS = 8
#: ``median_*`` gives the p50; these name the other percentiles the result's ``p<N>_*`` keys carry.
METRIC_PERCENTILES = (90, 99)
PERCENTILE_METRICS = ("ttft", "tpot", "itl", "e2el")
_WHAT = "the fixed client"

#: Each :data:`FIXED_METRIC_KEYS` entry the client's result fills, by the result key that fills it.
_RESULT_KEYS: Mapping[str, str] = {
    "output_throughput": "output_throughput",
    "request_throughput": "request_throughput",
    "total_token_throughput": "total_token_throughput",
    "completed_requests": "completed",
    "requested_requests": "num_prompts",
    "duration_seconds": "duration",
    "total_input_tokens": "total_input_tokens",
    "total_output_tokens": "total_output_tokens",
    "e2el_mean_ms": "mean_e2el_ms",
    "e2el_p99_ms": "p99_e2el_ms",
    **{
        f"{metric}_{stat}_ms": f"{raw}_{metric}_ms"
        for metric in ("ttft", "tpot")
        for stat, raw in (("mean", "mean"), ("p50", "median"), ("p90", "p90"), ("p99", "p99"))
    },
}
#: The client's own request-count verdict, saved beside its metrics.
_OUTCOME_KEY = "benchmark_outcome"


@dataclass(frozen=True)
class ClientTool:
    """The staged client: the interpreter it runs under and the directory holding the pinned ``infx`` package."""

    python: str
    package_root: str


@dataclass(frozen=True)
class ClientPass:
    """One invocation of the client: its prompt seed and count, and where it saves its result (``None``: nowhere)."""

    seed: int
    num_prompts: int
    result: Path | None


def client_env(tool: ClientTool) -> dict[str, str]:
    """The env the client gets beyond what the round inherits; ``launch.json`` records it."""
    return {"PYTHONPATH": tool.package_root}


def client_argv(
    tool: ClientTool, base_url: str, server: ServerConfig, workload: FixedWorkload, client_pass: ClientPass
) -> tuple[str, ...]:
    """The client argv for one pass: every load knob explicit, the tokenizer read from the served model's path."""
    argv = [
        tool.python,
        "-m",
        CLIENT_MODULE,
        "--backend",
        CLIENT_BACKEND,
        "--base-url",
        base_url,
        "--endpoint",
        CLIENT_ENDPOINT,
        "--model",
        server.model_path,
        "--served-model-name",
        server.served_model_name,
        "--tokenizer-mode",
        workload.tokenizer_mode,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(workload.isl),
        "--random-output-len",
        str(workload.osl),
        "--random-range-ratio",
        str(workload.random_range_ratio),
        "--random-prefix-len",
        str(CLIENT_RANDOM_PREFIX_LEN),
        "--random-num-workers",
        str(CLIENT_RANDOM_NUM_WORKERS),
        "--num-prompts",
        str(client_pass.num_prompts),
        "--max-concurrency",
        str(workload.concurrency),
        "--request-rate",
        "inf",
        "--ignore-eos",
        "--num-warmups",
        "0",
        "--seed",
        str(client_pass.seed),
        "--percentile-metrics",
        ",".join(PERCENTILE_METRICS),
        "--metric-percentiles",
        ",".join(str(percentile) for percentile in METRIC_PERCENTILES),
    ]
    if workload.trust_remote_code:
        argv.append("--trust-remote-code")
    if client_pass.result is not None:
        argv += ["--save-result", "--result-dir", str(client_pass.result.parent)]
        argv += ["--result-filename", client_pass.result.name]
    return tuple(argv)


def fixed_metrics(result: Mapping[str, object]) -> FixedMetrics:
    """The report's fixed metrics from the client's saved result; what the result does not carry stays ``None``.

    Raises:
        ReportError: A value the result carries is not the type its metric takes.
    """
    measured: dict[str, Any] = {key: result.get(raw) for key, raw in _RESULT_KEYS.items()}
    outcome = result.get(_OUTCOME_KEY)
    if isinstance(outcome, Mapping):
        measured["failed_requests"] = outcome.get("failed")
    return FixedMetrics.of(measured)


@dataclass(frozen=True)
class FixedOutcome:
    """What the fixed load did: the client as it ran, its last exit code, its metrics, and why it failed, if it did.

    A client that exits non-zero after saving its result (its own request-failure gate) leaves both metrics and a stop.
    """

    invocation: ClientInvocation
    returncode: int | None
    metrics: FixedMetrics | None
    stop: Stop | None


class OutputActivity:
    """Counts a child's output as the server's activity, so a quiet server log while the child works is not a stall."""

    def __init__(self, server: Server, stdout: Path) -> None:
        self._server = server
        self._stdout = stdout
        self._size = -1

    def check(self, now: float) -> Stop | None:
        size = self._stdout.stat().st_size if self._stdout.exists() else 0
        if size != self._size:
            self._size = size
            self._server.note_output(now)
        return self._server.check(now)


@dataclass(frozen=True)
class FixedRound:
    """What the fixed load runs in: the round's config, the staged client, its server, gates and phase clock."""

    config: SuiteConfig
    tool: ClientTool
    server: Server
    gate: RoundGate
    clock: PhaseClock


def _run_pass(fixed_round: FixedRound, argv: tuple[str, ...], phase: Phase) -> tuple[int | None, Stop | None]:
    config = fixed_round.config
    layout = RunLayout(Path(config.paths.run_dir))
    env = {**inherited_env(config.server), **client_env(fixed_round.tool)}
    fixed_round.clock.mark(phase)
    child = spawn(argv, env, layout.client_stdout, layout.client_stderr)
    ended = wait_child(child, fixed_round.gate, OutputActivity(fixed_round.server, layout.client_stdout).check)
    returncode = None if isinstance(ended, Stop) else ended.returncode
    return returncode, exit_stop(ended, _WHAT, layout.client_stderr)


def _workload(config: SuiteConfig) -> FixedWorkload:
    if config.fixed is None:
        raise ValueError(f"the fixed load needs a fixed workload, got {config.workload.value}")
    return config.fixed


def measured_invocation(config: SuiteConfig, tool: ClientTool, base_url: str) -> ClientInvocation:
    """The measured pass as ``launch.json`` records it, before the round's server is up."""
    workload = _workload(config)
    measured = ClientPass(workload.seed, workload.num_prompts, RunLayout(Path(config.paths.run_dir)).inferencex_result)
    return ClientInvocation(
        argv=client_argv(tool, base_url, config.server, workload, measured),
        env=client_env(tool),
        seed=workload.seed,
        warmup_seed=workload.warmup_seed,
    )


def run_fixed(fixed_round: FixedRound) -> FixedOutcome:
    """Run the warmup pass (when the workload asks for one), then the measured pass, and read the measured result."""
    config = fixed_round.config
    workload = _workload(config)
    layout = RunLayout(Path(config.paths.run_dir))
    invocation = measured_invocation(config, fixed_round.tool, fixed_round.server.base_url)
    measured_argv = invocation.argv
    if workload.num_warmups:
        warmup = ClientPass(workload.warmup_seed, workload.num_warmups, None)
        warmup_argv = client_argv(fixed_round.tool, fixed_round.server.base_url, config.server, workload, warmup)
        returncode, stop = _run_pass(fixed_round, warmup_argv, Phase.WARMUP)
        if stop is not None:
            return FixedOutcome(invocation, returncode, None, stop)
    returncode, stop = _run_pass(fixed_round, measured_argv, Phase.MEASURED)
    fixed_round.clock.mark(Phase.FINISHED)
    if stop is not None and stop.step is not FailedStep.CLIENT_FAILED:
        return FixedOutcome(invocation, returncode, None, stop)
    result = read_result(layout.inferencex_result, _WHAT)
    if isinstance(result, Stop):
        return FixedOutcome(invocation, returncode, None, stop or result)
    try:
        metrics = fixed_metrics(result)
    except ReportError as exc:
        return FixedOutcome(invocation, returncode, None, Stop(FailedStep.CLIENT_OUTPUT_MISSING, str(exc)))
    return FixedOutcome(invocation, returncode, metrics, stop)


__all__ = [
    "CLIENT_RANDOM_NUM_WORKERS",
    "CLIENT_RANDOM_PREFIX_LEN",
    "ClientPass",
    "ClientTool",
    "FixedOutcome",
    "FixedRound",
    "OutputActivity",
    "client_argv",
    "client_env",
    "fixed_metrics",
    "measured_invocation",
    "run_fixed",
]
