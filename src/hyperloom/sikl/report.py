# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Score the delivered kernel with `sikl validate`, the check SIKL's CI runs, and write the run's summary."""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from hyperloom.sikl.schema import Schema
from hyperloom.sikl.workspace import Run

SUMMARY_FILENAME = "summary.md"
OUTCOMES_FILENAME = "sikl_outcomes.json"
# SIKL serves a workload only where the kernel's measured speedup reaches this (SIKL_MIN_SPEEDUP's default).
SERVING_SPEEDUP = 1.03


@dataclass(frozen=True)
class CaseResult:
    """One scored workload as SIKL reports it."""

    case_id: str
    incorrect: tuple[str, ...]
    candidate_ms: float | None
    baseline_ms: float | None
    speedup_bound: float | None

    @property
    def speedup(self) -> float | None:
        if not self.candidate_ms or not self.baseline_ms:
            return None
        return self.baseline_ms / self.candidate_ms

    @property
    def is_served(self) -> bool:
        return not self.incorrect and (self.speedup_bound or 0.0) >= SERVING_SPEEDUP

    @property
    def unscored_reason(self) -> str | None:
        """Why this case carries no speedup, or None when it does."""
        if self.incorrect:
            return "SIKL found it wrong: " + ", ".join(self.incorrect)
        if not self.candidate_ms:
            return "SIKL did not time the candidate"
        if not self.baseline_ms:
            return "SIKL timed no baseline"
        return None


def _read_outcomes(run: Run) -> list[CaseResult]:
    """SIKL's verdict on the kernel the workspace holds, which is the one the loop kept."""
    path = run.root / OUTCOMES_FILENAME
    path.unlink(missing_ok=True)
    completed = subprocess.run(
        [sys.executable, str(run.driver), "--outcomes-json", str(path)],
        cwd=run.workspace,
        capture_output=True,
        text=True,
    )
    if not path.is_file():
        sys.stderr.write(completed.stdout[-2000:] + completed.stderr[-2000:])
        raise SystemExit(f"the driver wrote no outcomes (exit {completed.returncode})")
    return [
        CaseResult(
            case_id=case_id,
            incorrect=tuple(sorted(set(outcome["incorrect"]))),
            candidate_ms=outcome["candidate_ms"],
            baseline_ms=outcome["baseline_ms"],
            speedup_bound=outcome["speedup_bound"],
        )
        for case_id, outcome in json.loads(path.read_text(encoding="utf-8")).items()
    ]


def build_summary(schema: Schema, results: list[CaseResult]) -> str:
    """A summary a reviewer can act on, with the losing and unscored cases kept in."""
    scored = [result for result in results if result.speedup is not None]
    speedups = [result.speedup for result in scored if result.speedup is not None]
    losses = [result for result in scored if (result.speedup or 0.0) < 1.0]
    unscored = [result for result in results if result.unscored_reason is not None]
    mean = statistics.geometric_mean(speedups) if speedups else 0.0

    lines = [
        f"# {schema.name}",
        "",
        "Scored by `sikl validate` against SIKL's baseline, timed in the same window as the candidate.",
        "",
        f"- **Geometric mean speedup:** {mean:.4f}x over {len(speedups)} scored cases",
        f"- **Cases faster than baseline:** {sum(1 for value in speedups if value > 1.0)}/{len(speedups)}",
        f"- **Cases SIKL would serve** (speedup lower bound ≥ {SERVING_SPEEDUP}): "
        f"{sum(1 for result in results if result.is_served)}/{len(results)}",
        f"- **Cases not scored:** {len(results) - len(scored)}",
        "",
        "| case | candidate us | baseline us | speedup | lower bound | served |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        candidate = f"{result.candidate_ms * 1000:.2f}" if result.candidate_ms else "—"
        baseline = f"{result.baseline_ms * 1000:.2f}" if result.baseline_ms else "—"
        speedup = f"{result.speedup:.3f}x" if result.speedup else "—"
        bound = f"{result.speedup_bound:.3f}x" if result.speedup_bound else "—"
        served = "yes" if result.is_served else "no"
        lines.append(f"| `{result.case_id}` | {candidate} | {baseline} | {speedup} | {bound} | {served} |")

    lines += ["", "## Cases this kernel loses", ""]
    lines += [f"- `{result.case_id}`: {result.speedup:.3f}x" for result in losses] or ["None."]
    # A case without a speedup is outside the mean above, which alone cannot say what it left out.
    lines += ["", "## Cases not scored", ""]
    lines += [f"- `{result.case_id}`: {result.unscored_reason}" for result in unscored] or ["None."]
    lines.append("")
    return "\n".join(lines)


def finalize(schema: Schema, run: Run) -> Path:
    """Score the delivered kernel and write `summary.md`; return its path."""
    summary = run.root / SUMMARY_FILENAME
    summary.write_text(build_summary(schema, _read_outcomes(run)), encoding="utf-8")
    return summary
