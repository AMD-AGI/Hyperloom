"""Concrete ActionRunner implementations.

Each runner is an ``async def fn(RunnerContext) -> dict`` that the
SubAgentRunner dispatches when a queued task's ``kind`` matches its
registered name.

v0.8 M3 + KB_gaps/Dead-A consolidated the legacy ``backends`` /
``params`` / ``validate_stack`` executors into the merged
:class:`ExploreExecutor`. Their modules and yamls have been
physically deleted; the legacy dataclass fields (``backends_search`` /
``params_search`` / ``last_validate_stack`` / ``*_attempts``) stay on
:class:`SharedState` for resume parity (Inv-10.1) but no executor
backs them in fresh sessions.
"""

from .baseline import (
    BASELINE_DEFAULT_CONFIG,
    BaselineExecutor,
    baseline_executor,
)
from .explore import (
    DEFAULT_KEEP_THRESHOLD_PCT,
    DEFAULT_STACK_STABLE_PCT,
    ExploreExecutor,
    explore_executor,
)
from .report import ReportExecutor, report_executor
from .session_breakdown import SessionBreakdownExecutor, session_breakdown_executor
from .sweep import (
    DEFAULT_CONC_VALUES,
    DEFAULT_ISL_OSL,
    SweepExecutor,
    sweep_executor,
)
from .target_analysis import TargetAnalysisExecutor
from .recover import RecoverExecutor, recover_executor

__all__ = [
    "BASELINE_DEFAULT_CONFIG",
    "BaselineExecutor",
    "DEFAULT_CONC_VALUES",
    "DEFAULT_ISL_OSL",
    "DEFAULT_KEEP_THRESHOLD_PCT",
    "DEFAULT_STACK_STABLE_PCT",
    "ExploreExecutor",
    "RecoverExecutor",
    "ReportExecutor",
    "SessionBreakdownExecutor",
    "SweepExecutor",
    "TargetAnalysisExecutor",
    "baseline_executor",
    "explore_executor",
    "recover_executor",
    "report_executor",
    "session_breakdown_executor",
    "sweep_executor",
]
