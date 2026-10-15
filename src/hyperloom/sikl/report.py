# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Score the delivered kernel with the harness SIKL's acceptance gate runs, and write the run's summary."""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from hyperloom.sikl.definition import Definition
from hyperloom.sikl.workspace import Run, candidate_name, candidate_solution

SUMMARY_FILENAME = "summary.md"
VERDICT_FILENAME = "sikl_verify.json"


@dataclass(frozen=True)
class CaseResult:
    """One scored workload as SIKL reports it."""

    case_id: str
    axes: dict[str, int]
    status: str
    candidate_ms: float | None
    baseline_ms: float | None

    @property
    def speedup(self) -> float | None:
        if not self.candidate_ms or not self.baseline_ms:
            return None
        return self.baseline_ms / self.candidate_ms

    @property
    def unscored_reason(self) -> str | None:
        """Why this case carries no speedup, or None when it does."""
        if self.status != "PASSED":
            return f"failed correctness ({self.status})"
        if not self.candidate_ms:
            return "the candidate was not timed"
        if not self.baseline_ms:
            # SIKL reports a baseline that failed to run as absent; the harness log names the failure.
            return "the baseline produced no timing"
        return None


def _publish_delivered_kernel(definition: Definition, run: Run) -> None:
    """Point the published candidate at the kernel the loop kept, since the driver's last publish may be a
    lane the loop rejected. Only the source is replaced; the spec stays as the driver published it."""
    path = candidate_solution(definition, run)
    solution = json.loads(path.read_text(encoding="utf-8"))
    solution["sources"] = [{"path": run.kernel.name, "content": run.kernel.read_text(encoding="utf-8")}]
    path.write_text(json.dumps(solution), encoding="utf-8")


def _read_verdict(
    definition: Definition, run: Run, *, sikl_root: Path, warmup: int, iterations: int, report: Path
) -> list[CaseResult]:
    command = [
        sys.executable,
        "-m",
        "sikl.benchmarks.api",
        str(run.bundle),
        "--definition",
        definition.name,
        "--solution",
        candidate_name(definition),
        "--warmup-runs",
        str(warmup),
        "--iterations",
        str(iterations),
        "--num-trials",
        "1",
        "--time-baseline",
        "--json-report",
        str(report),
    ]
    env = {"PYTHONPATH": str(sikl_root)}
    # A report left by an earlier pass would otherwise read as this pass's verdict.
    report.unlink(missing_ok=True)
    completed = subprocess.run(command, cwd=sikl_root, capture_output=True, text=True, env={**os.environ, **env})
    # A candidate that does not qualify exits nonzero; only a missing report means there is no verdict.
    if not report.is_file() or not report.stat().st_size:
        sys.stderr.write(completed.stdout[-2000:] + completed.stderr[-2000:])
        raise SystemExit(f"sikl.benchmarks.api wrote no report (exit {completed.returncode})")
    axes_by_case = {workload.uuid: workload.axes for workload in definition.workloads}
    results = []
    for record in json.loads(report.read_text(encoding="utf-8")):
        case_id = str(record["workload"]["uuid"])
        evaluation = record["evaluation"]
        performance = evaluation.get("performance") or {}
        results.append(
            CaseResult(
                case_id=case_id,
                axes=axes_by_case.get(case_id, {}),
                status=str(evaluation.get("status")),
                candidate_ms=performance.get("latency_ms"),
                baseline_ms=performance.get("reference_latency_ms"),
            )
        )
    return results


def build_summary(definition: Definition, results: list[CaseResult]) -> str:
    """A summary a reviewer can act on, with the losing and unscored cases kept in."""
    axis_names = list(definition.variable_axes)

    def label(result: CaseResult) -> str:
        return ", ".join(f"{name}={result.axes.get(name)}" for name in axis_names)

    scored = [result for result in results if result.speedup is not None]
    speedups = [result.speedup for result in scored if result.speedup is not None]
    losses = [result for result in scored if (result.speedup or 0.0) < 1.0]
    unscored = [result for result in results if result.unscored_reason is not None]
    incorrect = [result for result in results if result.status != "PASSED"]
    mean = statistics.geometric_mean(speedups) if speedups else 0.0

    lines = [
        f"# {definition.name}",
        "",
        "Scored by `sikl.benchmarks.api` against the definition's own baseline, timed in",
        "the same window as the candidate.",
        "",
        f"- **Geometric mean speedup:** {mean:.4f}x over {len(speedups)} scored cases",
        f"- **Cases faster than baseline:** {sum(1 for value in speedups if value > 1.0)}/{len(speedups)}",
        f"- **Cases not scored:** {len(results) - len(scored)}",
        f"- **Cases failing correctness:** {len(incorrect)}",
        "",
        "| " + " | ".join(f"`{name}`" for name in axis_names) + " | candidate us | baseline us | speedup | status |",
        "|---" * (len(axis_names) + 4) + "|",
    ]
    for result in results:
        axes = " | ".join(str(result.axes.get(name, "")) for name in axis_names)
        candidate = f"{result.candidate_ms * 1000:.2f}" if result.candidate_ms else "—"
        baseline = f"{result.baseline_ms * 1000:.2f}" if result.baseline_ms else "—"
        speedup = f"{result.speedup:.3f}x" if result.speedup else "—"
        lines.append(f"| {axes} | {candidate} | {baseline} | {speedup} | {result.status} |")

    lines += ["", "## Cases this kernel loses", ""]
    lines += [f"- `{label(result)}`: {result.speedup:.3f}x" for result in losses] or ["None."]
    # A case without a speedup is outside the mean above, which alone cannot say what it left out.
    lines += ["", "## Cases not scored", ""]
    lines += [f"- `{label(result)}`: {result.unscored_reason}" for result in unscored] or ["None."]
    lines += [
        "",
        "SIKL's acceptance rule asks for one workload above 1.02x, which a geometric mean",
        "below 1.0 can still satisfy. Read the table, not the rule.",
        "",
    ]
    return "\n".join(lines)


def finalize(definition: Definition, run: Run, *, sikl_root: Path, warmup: int = 20, iterations: int = 100) -> Path:
    """Score the delivered kernel and write `summary.md`; return its path."""
    _publish_delivered_kernel(definition, run)
    results = _read_verdict(
        definition, run, sikl_root=sikl_root, warmup=warmup, iterations=iterations, report=run.root / VERDICT_FILENAME
    )
    summary = run.root / SUMMARY_FILENAME
    summary.write_text(build_summary(definition, results), encoding="utf-8")
    return summary
