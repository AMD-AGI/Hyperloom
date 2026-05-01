"""Concrete ActionExecutor implementations (DESIGN v0.6 §15.2).

Each executor is an `async def fn(ExecutorContext) -> dict` that the
SubAgentRunner dispatches when a queued task's ``kind`` matches its
registered name.

Executors shipped:

* :func:`baseline_executor` — Magpie SGLang baseline benchmark.
* :func:`profile_executor`  — Magpie SGLang baseline + torch profiler
  (writes ``torch_trace/`` for the Kernel agent to analyze).
"""

from .backends import (
    DEFAULT_BACKENDS_GRID,
    BackendsExecutor,
    backends_executor,
    discover_backend_flags,
)
from .baseline import (
    BASELINE_DEFAULT_CONFIG,
    BaselineExecutor,
    baseline_executor,
)
from .params import (
    DEFAULT_NCCL_GRID,
    DEFAULT_PARAMS_GRID,
    ParamsExecutor,
    params_executor,
)
from .profile import (
    PROFILE_DEFAULT_CONFIG,
    ProfileExecutor,
    profile_executor,
)
from .sweep import (
    DEFAULT_CONC_VALUES,
    DEFAULT_ISL_OSL,
    SweepExecutor,
    sweep_executor,
)

__all__ = [
    "BASELINE_DEFAULT_CONFIG",
    "BackendsExecutor",
    "BaselineExecutor",
    "DEFAULT_BACKENDS_GRID",
    "DEFAULT_CONC_VALUES",
    "DEFAULT_ISL_OSL",
    "DEFAULT_NCCL_GRID",
    "DEFAULT_PARAMS_GRID",
    "PROFILE_DEFAULT_CONFIG",
    "ParamsExecutor",
    "ProfileExecutor",
    "SweepExecutor",
    "backends_executor",
    "baseline_executor",
    "discover_backend_flags",
    "params_executor",
    "profile_executor",
    "sweep_executor",
]
