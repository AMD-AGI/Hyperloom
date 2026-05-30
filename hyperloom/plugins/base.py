"""Abstract base classes for benchmark and accuracy plugins.

Any framework can be supported by implementing these interfaces.
Users can also provide raw scripts via CustomBenchmarkPlugin/CustomAccuracyPlugin.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class BenchResult:
    """Standardized benchmark result returned by all plugins."""

    throughput: float = 0.0  # primary metric (e.g., tokens/sec)
    throughput_unit: str = "tok/s"
    latency_mean_ms: float = 0.0
    latency_p99_ms: float = 0.0
    completed: int = 0
    total: int = 0
    raw_output: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def success(self) -> bool:
        return self.throughput > 0


@dataclass
class AccuracyResult:
    """Standardized accuracy result returned by all plugins."""

    score: float = 0.0  # primary metric (e.g., 0.0-1.0 or percentage)
    metric_name: str = "accuracy"
    passed: bool = True
    threshold: float = 0.0
    raw_output: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class BenchmarkPlugin(ABC):
    """Interface for benchmark execution.

    Implementations wrap a specific benchmarking tool (Magpie, vLLM benchmark,
    or a user-provided script) and return standardized BenchResult.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this plugin (e.g., 'magpie', 'vllm', 'custom')."""
        ...

    @abstractmethod
    def run(self, config: dict[str, Any]) -> BenchResult:
        """Execute the benchmark and return results.

        Args:
            config: Session config dict with model path, GPU info, workload params, etc.

        Returns:
            BenchResult with throughput and latency metrics.
        """
        ...

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Optional config validation. Returns list of error messages (empty = valid)."""
        return []


class AccuracyPlugin(ABC):
    """Interface for accuracy evaluation.

    Implementations wrap an accuracy evaluation tool (lm_eval, custom script)
    and return standardized AccuracyResult.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name of this plugin."""
        ...

    @abstractmethod
    def run(self, config: dict[str, Any]) -> AccuracyResult:
        """Execute the accuracy eval and return results.

        Args:
            config: Session config dict with model path, eval parameters, etc.

        Returns:
            AccuracyResult with score and pass/fail status.
        """
        ...

    def validate_config(self, config: dict[str, Any]) -> list[str]:
        """Optional config validation."""
        return []
