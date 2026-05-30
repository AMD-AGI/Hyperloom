"""Benchmark and accuracy plugins for different frameworks."""

from .base import BenchmarkPlugin, AccuracyPlugin, BenchResult, AccuracyResult
from .custom import CustomBenchmarkPlugin, CustomAccuracyPlugin

__all__ = [
    "BenchmarkPlugin",
    "AccuracyPlugin",
    "BenchResult",
    "AccuracyResult",
    "CustomBenchmarkPlugin",
    "CustomAccuracyPlugin",
]
