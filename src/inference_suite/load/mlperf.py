# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The MLPerf load: MLCommons' agentic harness (``utility/run_agentic.sh``) against the round's server.

The harness runs from the root of the operator's ``mlcommons/endpoints`` checkout, as its own scripts expect, and reads
its knobs from the environment: where to write, the corpus, the tokenizer, the trajectory count, the concurrency and
the server's port. It writes ``result_summary.json`` and, when it scored the turns inline, ``scores.json``; the round
maps the two (:mod:`inference_suite.load.mlperf_report`) into the run_dir's ``client/inferencex_result.json``. The
harness marks no phases, so its whole run is the measured window.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hyperloom_common.io import atomic_write_json
from inference_suite.contracts.config import MlperfWorkload, SuiteConfig
from inference_suite.contracts.layout import RunLayout
from inference_suite.contracts.report import FailedStep, MlperfResult
from inference_suite.load import exit_stop, inherited_env
from inference_suite.load.fixed import OutputActivity
from inference_suite.load.mlperf_report import MlperfReportError, map_mlperf
from inference_suite.runtime.phases import Phase, PhaseClock
from inference_suite.runtime.process import spawn
from inference_suite.runtime.server import Server
from inference_suite.runtime.watch import RoundGate, Stop, wait_child

HARNESS_SCRIPT = Path("utility") / "run_agentic.sh"
#: ``run_agentic.sh``'s last argument: a performance run, scored inline, as opposed to its accuracy-only mode.
HARNESS_MODE = "perf"
SUMMARY_FILE = "result_summary.json"
SCORES_FILE = "scores.json"
_WHAT = "the MLPerf harness"


def _workload(config: SuiteConfig) -> MlperfWorkload:
    if config.mlperf is None:
        raise ValueError(f"the MLPerf load needs an MLPerf workload, got {config.workload.value}")
    return config.mlperf


def refusal(workload: MlperfWorkload) -> Stop | None:
    """Why the harness cannot run ``workload``, if it cannot; the round asks before its server starts."""
    needed = (
        (Path(workload.endpoints_dir) / HARNESS_SCRIPT, Path.is_file),
        (Path(workload.dataset), Path.is_file),
        (Path(workload.tokenizer), Path.is_dir),
    )
    absent = [str(path) for path, present in needed if not present(path)]
    if not absent:
        return None
    return Stop(FailedStep.CONFIG_INVALID, f"{_WHAT} cannot run without {', '.join(absent)}")


def harness_argv(workload: MlperfWorkload) -> tuple[str, ...]:
    return ("bash", str(HARNESS_SCRIPT), workload.flow, workload.model, workload.hardware, HARNESS_MODE)


def harness_env(config: SuiteConfig, layout: RunLayout) -> dict[str, str]:
    """The harness's env beyond what it inherits: where it writes, and every knob of the run."""
    workload = _workload(config)
    return {
        "RESULTS_DIR": str(layout.mlperf_artifacts),
        "AGENTIC_DATASET_PATH": workload.dataset,
        "MLPERF_TOKENIZER_DIR": workload.tokenizer,
        "AGENTIC_NUM_TRAJECTORIES": str(workload.trajectories),
        "AGENTIC_CONCURRENCY": str(workload.concurrency),
        "PORT": str(config.server.port),
    }


def _harness_file(layout: RunLayout, name: str) -> Path | None:
    """The one ``name`` the harness wrote below its artifacts, ``None`` when it wrote none."""
    found = sorted(layout.mlperf_artifacts.rglob(name))
    if len(found) > 1:
        raise MlperfReportError(f"more than one {name}: {', '.join(str(path) for path in found)}")
    return found[0] if found else None


def _document(path: Path) -> Mapping[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise MlperfReportError(f"{path} cannot be read as JSON: {exc}") from exc
    if not isinstance(document, Mapping):
        raise MlperfReportError(f"{path} holds {type(document).__name__}, not an object")
    return document


def mlperf_result(layout: RunLayout, workload: MlperfWorkload) -> MlperfResult | Stop:
    """Map what the harness wrote, write the mapping to ``client/inferencex_result.json``, and return the section."""
    try:
        summary = _harness_file(layout, SUMMARY_FILE)
        if summary is None:
            raise MlperfReportError(f"no {SUMMARY_FILE} below {layout.mlperf_artifacts}")
        scores = _harness_file(layout, SCORES_FILE)
        scored = None if scores is None else _document(scores)
        result = map_mlperf(_document(summary), corpus=Path(workload.dataset).stem, scores=scored)
    except MlperfReportError as exc:
        return Stop(FailedStep.CLIENT_OUTPUT_MISSING, f"{_WHAT}'s output: {exc}")
    atomic_write_json(layout.inferencex_result, result)
    return MlperfResult(
        summary_file=summary.relative_to(layout.root).as_posix(),
        scores_file=None if scores is None else scores.relative_to(layout.root).as_posix(),
        result_file=layout.inferencex_result.relative_to(layout.root).as_posix(),
        result=result,
    )


@dataclass(frozen=True)
class MlperfRound:
    """What the MLPerf load runs in: the round's config, its server, gates and phase clock."""

    config: SuiteConfig
    server: Server
    gate: RoundGate
    clock: PhaseClock


@dataclass(frozen=True)
class MlperfOutcome:
    """What the harness did: its exit code, the result it left, and why it failed, if it did."""

    returncode: int | None
    result: MlperfResult | None
    stop: Stop | None


def run_mlperf(mlperf_round: MlperfRound) -> MlperfOutcome:
    """Run the harness once against the round's server, then map the result it wrote."""
    config = mlperf_round.config
    workload = _workload(config)
    layout = RunLayout(Path(config.paths.run_dir))
    env = {**inherited_env(config.server), **harness_env(config, layout)}
    activity = OutputActivity(mlperf_round.server, layout.client_stdout)
    mlperf_round.clock.mark(Phase.MEASURED)
    child = spawn(
        harness_argv(workload), env, layout.client_stdout, layout.client_stderr, cwd=Path(workload.endpoints_dir)
    )
    ended = wait_child(child, mlperf_round.gate, activity.check)
    mlperf_round.clock.mark(Phase.FINISHED)
    returncode = None if isinstance(ended, Stop) else ended.returncode
    stop = exit_stop(ended, _WHAT, layout.client_stderr)
    if stop is not None:
        return MlperfOutcome(returncode, None, stop)
    result = mlperf_result(layout, workload)
    if isinstance(result, Stop):
        return MlperfOutcome(returncode, None, result)
    return MlperfOutcome(returncode, result, None)


__all__ = [
    "HARNESS_MODE",
    "HARNESS_SCRIPT",
    "SCORES_FILE",
    "SUMMARY_FILE",
    "MlperfOutcome",
    "MlperfRound",
    "harness_argv",
    "harness_env",
    "mlperf_result",
    "refusal",
    "run_mlperf",
]
