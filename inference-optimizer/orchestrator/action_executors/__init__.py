"""Concrete ActionExecutor implementations (DESIGN v0.6 §15.2).

Each executor is an `async def fn(ExecutorContext) -> dict` that the
SubAgentRunner dispatches when a queued task's ``kind`` matches its
registered name.

Executors shipped:

* :func:`baseline_executor` — Magpie SGLang baseline benchmark.
* :func:`profile_executor`  — Magpie SGLang baseline + torch profiler
  (writes ``torch_trace/`` for the Kernel agent to analyze).
"""

from .baseline import (
    BASELINE_DEFAULT_CONFIG,
    BaselineExecutor,
    baseline_executor,
)
from .profile import (
    PROFILE_DEFAULT_CONFIG,
    ProfileExecutor,
    profile_executor,
)

__all__ = [
    "BASELINE_DEFAULT_CONFIG",
    "BaselineExecutor",
    "PROFILE_DEFAULT_CONFIG",
    "ProfileExecutor",
    "baseline_executor",
    "profile_executor",
]
