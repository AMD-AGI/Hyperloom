"""Concrete ActionExecutor implementations (DESIGN v0.6 §15.2).

Each executor is an `async def fn(ExecutorContext) -> dict` that the
SubAgentRunner dispatches when a queued task's ``kind`` matches its
registered name.

P1-6 ships:

* :func:`baseline_executor` — runs an end-to-end SGLang baseline benchmark
  via Magpie CLI, parses ``benchmark_report.json``, and returns a stable
  result schema (request_throughput / output_throughput / TTFT / E2EL / ...).
"""

from .baseline import (
    BASELINE_DEFAULT_CONFIG,
    BaselineExecutor,
    baseline_executor,
)

__all__ = [
    "BASELINE_DEFAULT_CONFIG",
    "BaselineExecutor",
    "baseline_executor",
]
