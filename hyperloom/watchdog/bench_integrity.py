"""Bench integrity checker — Tier 1 watchdog.

Validates benchmark measurements for statistical sanity before the
orchestrator trusts them for keep/revert decisions.

Checks:
  1. Completion ratio:  enough requests actually completed
  2. Noise detection:   CoV across recent runs too high → suspect jitter
  3. Monotonicity:      sudden large swings without config changes → flaky
  4. Latency sanity:    TTFT / TPOT should not be zero or negative
  5. Throughput floor:   throughput below a suspicious minimum (stall?)
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any


@dataclass
class IntegrityVerdict:
    passed: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def severity(self) -> str:
        if self.errors:
            return "error"
        if self.warnings:
            return "warning"
        return "info"


class BenchIntegrityChecker:
    """Stateful checker that accumulates benchmark history for noise detection."""

    def __init__(
        self,
        min_completion_ratio: float = 0.9,
        max_cov: float = 0.15,
        max_swing_pct: float = 30.0,
        throughput_floor: float = 1.0,
        window_size: int = 5,
    ):
        self.min_completion_ratio = min_completion_ratio
        self.max_cov = max_cov
        self.max_swing_pct = max_swing_pct
        self.throughput_floor = throughput_floor
        self.window_size = window_size
        self._history: list[float] = []

    def check(self, bench_dict: dict[str, Any]) -> IntegrityVerdict:
        """Run all integrity checks against a benchmark result.

        Expected keys: output_throughput, completed, num_prompts,
                       mean_ttft_ms, mean_tpot_ms
        """
        warnings: list[str] = []
        errors: list[str] = []
        details: dict[str, Any] = {}

        throughput = bench_dict.get("output_throughput", 0)
        completed = bench_dict.get("completed", 0)
        num_prompts = bench_dict.get("num_prompts", 0)
        ttft = bench_dict.get("mean_ttft_ms", 0)
        tpot = bench_dict.get("mean_tpot_ms", 0)

        if num_prompts > 0:
            ratio = completed / num_prompts
            details["completion_ratio"] = ratio
            if ratio < self.min_completion_ratio:
                errors.append(
                    f"Low completion ratio: {ratio:.2f} "
                    f"({completed}/{num_prompts}) < {self.min_completion_ratio}"
                )

        if throughput < self.throughput_floor:
            errors.append(
                f"Throughput {throughput:.1f} tok/s below floor {self.throughput_floor}"
            )

        if throughput < 0:
            errors.append(f"Negative throughput: {throughput}")
        if ttft < 0:
            errors.append(f"Negative TTFT: {ttft}")
        if tpot < 0:
            errors.append(f"Negative TPOT: {tpot}")

        if throughput > 0:
            self._history.append(throughput)
            if len(self._history) > self.window_size:
                self._history = self._history[-self.window_size:]

        if len(self._history) >= 3:
            mean = statistics.mean(self._history)
            stdev = statistics.stdev(self._history)
            cov = stdev / mean if mean > 0 else 0
            details["cov"] = round(cov, 4)
            details["recent_mean"] = round(mean, 1)
            details["recent_stdev"] = round(stdev, 1)

            if cov > self.max_cov:
                warnings.append(
                    f"High variance: CoV={cov:.3f} > {self.max_cov} — "
                    f"environment jitter suspected"
                )

        if len(self._history) >= 2:
            prev = self._history[-2]
            if prev > 0:
                swing_pct = abs(throughput - prev) / prev * 100
                details["swing_pct"] = round(swing_pct, 1)
                if swing_pct > self.max_swing_pct:
                    warnings.append(
                        f"Large swing: {swing_pct:.1f}% change from previous run "
                        f"({prev:.1f} → {throughput:.1f}) — possible measurement noise"
                    )

        passed = len(errors) == 0
        return IntegrityVerdict(
            passed=passed,
            warnings=warnings,
            errors=errors,
            details=details,
        )

    def reset(self) -> None:
        """Reset history (e.g. after a config change)."""
        self._history.clear()
