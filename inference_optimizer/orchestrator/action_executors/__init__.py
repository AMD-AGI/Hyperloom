"""Concrete ActionRunner implementations.

Each runner is an ``async def fn(RunnerContext) -> dict`` that the
SubAgentRunner dispatches when a queued task's ``kind`` matches its
registered name.

The merged :class:`ExploreExecutor` is the single grid-runner entry;
it maintains the unified ``explore_search`` ledger on
:class:`SharedState`.
"""

from .baseline import (
    BASELINE_DEFAULT_CONFIG,
    BaselineExecutor,
    baseline_executor,
)
from .dynamic_action import dynamic_action_executor
from .explore import (
    DEFAULT_KEEP_THRESHOLD_PCT,
    DEFAULT_STACK_STABLE_PCT,
    ExploreExecutor,
    explore_executor,
)
from .conc_sweep import ConcSweepExecutor, conc_sweep_executor
from .framework_pr import FrameworkPrExecutor, framework_pr_executor
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
    "ConcSweepExecutor",
    "DEFAULT_CONC_VALUES",
    "DEFAULT_ISL_OSL",
    "DEFAULT_KEEP_THRESHOLD_PCT",
    "DEFAULT_STACK_STABLE_PCT",
    "ExploreExecutor",
    "FrameworkPrExecutor",
    "RecoverExecutor",
    "ReportExecutor",
    "SessionBreakdownExecutor",
    "SweepExecutor",
    "TargetAnalysisExecutor",
    "baseline_executor",
    "conc_sweep_executor",
    "dynamic_action_executor",
    "explore_executor",
    "framework_pr_executor",
    "recover_executor",
    "report_executor",
    "session_breakdown_executor",
    "sweep_executor",
]
