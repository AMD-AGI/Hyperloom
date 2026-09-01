# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Driver-owned full-suite SNR pre-filter.

The driver owns the complete case selection. Forge invokes that suite without
shape or mode selectors and consumes its aggregate SNR result.

This is a pre-filter, not the KEEP gate. It is cheap enough to run every
iteration and it stops an obviously broken candidate before the benchmark, but
its threshold is forge's own and no scorer uses it. A candidate that clears it
is still judged by the task's declared correctness suite -- see
``loop/canonical_correctness.py``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from kernelforge.mcp_server.tools.test import test_correctness
from kernelforge.loop.scoring import DEFAULT_SNR_THRESHOLD_DB


@dataclass
class ValidationResult:
    """Result of the driver-owned validation suite."""

    stage: int
    stage_name: str
    passed: bool
    details: str
    outcome: str = ""
    snr_db: float | None = None
    output: str = ""  # full driver output tail on failure (for the experience ledger)

    def __str__(self):
        status = _status_label(self.outcome, self.passed)
        snr = f" (SNR={self.snr_db:.1f} dB)" if self.snr_db is not None else ""
        return f"  Stage {self.stage} [{self.stage_name}]: {status}{snr} — {self.details}"


@dataclass
class ValidationReport:
    """Full report from the driver-owned validation suite."""

    results: list[ValidationResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def failed_stage(self) -> int | None:
        for r in self.results:
            if not r.passed:
                return r.stage
        return None

    @property
    def failed_output(self) -> str:
        """Full driver output tail from the first failing stage (for the ledger)."""
        for r in self.results:
            if not r.passed:
                return r.output or r.details
        return ""

    @property
    def failed_outcome(self) -> str:
        """Structured failure kind from the first failing validation result."""
        for result in self.results:
            if not result.passed:
                return result.outcome
        return ""

    def summary(self) -> str:
        lines = ["Validation Pipeline:"]
        for r in self.results:
            status = _status_label(r.outcome, r.passed)
            snr = f" SNR={r.snr_db:.1f}dB" if r.snr_db is not None else ""
            lines.append(f"  {r.stage}. {r.stage_name}: {status}{snr}")
        verdict = "ALL PASSED" if self.all_passed else f"FAILED at stage {self.failed_stage}"
        lines.append(f"  Verdict: {verdict}")
        return "\n".join(lines)


async def run_validation_pipeline(
    driver_script: str,
    snr_threshold: float = DEFAULT_SNR_THRESHOLD_DB,
    timeout_per_stage: int = 1800,
) -> ValidationReport:
    """Run the driver's complete correctness suite once.

    Args:
        driver_script: Test driver that owns all correctness cases.
        snr_threshold: SNR pre-filter threshold.
        timeout_per_stage: Max seconds for the complete suite.

    Returns:
        ValidationReport with results from all completed stages.
    """
    result = await test_correctness(
        driver_script=driver_script,
        driver_args=[],
        snr_threshold=snr_threshold,
        timeout_sec=timeout_per_stage,
    )
    return ValidationReport(
        results=[
            ValidationResult(
                stage=1,
                stage_name="Full suite",
                passed=result["passed"],
                outcome=str(result.get("outcome") or ""),
                snr_db=result.get("snr_db"),
                details=result["message"],
                output=result.get("output", ""),
            )
        ]
    )


def _status_label(outcome: str, passed: bool) -> str:
    if passed:
        return "PASS"
    if outcome == "timeout":
        return "TIMEOUT"
    if outcome in {"driver_error", "invalid_result"}:
        return "ERROR"
    return "FAIL"
