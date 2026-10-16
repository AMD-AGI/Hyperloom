# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Concrete ActionRunner implementations."""

from .baseline import (
    BaselineExecutor,
    baseline_executor,
)
from ._grid_base import DEFAULT_KEEP_THRESHOLD_PCT
from .explore import (
    ExploreExecutor,
    explore_executor,
)
from .conc_sweep import ConcSweepExecutor, conc_sweep_executor
from .report import ReportExecutor, report_executor
from .session_breakdown import SessionBreakdownExecutor, session_breakdown_executor
from .target_analysis import TargetAnalysisExecutor
from ._workload_envs import default_baseline_config, materialize_config_with_envs
from ._grid_base import GridVariant, VariantResult
from ._grid_runner import _num_gpus_for_config as num_gpus_for_config
from ._grid_runner import run_grid, sync_benchmark_timeout
from ._ray_backend import strip_visible_devices_from_config
from ._server_lifecycle import resolve_lifecycle_params, teardown_lifecycle_server
from ._subprocess_kill import (
    AGENTX_PREFLIGHT_RETURNCODE,
    EVAL_PROBE_UNPATCHABLE_RETURNCODE,
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
)
from .benchmark_result import harvest_leaked_artifacts
from ._agentx_prefix_cache import PREFIX_CACHE_OFF_ERROR_CLASS, refuse_cacheless_agentx_baseline
from ._launch_evidence import build_launch_evidence
from ._spec_decode_context_copy import CONTEXT_COPY_REFUSAL, context_copy_refusal
from ._spec_decode_pin import (
    SPECULATIVE_DECODING_PINNED,
    SpeculativeDecodingPinnedError,
    refuse_unpinned_speculative_change,
)

__all__ = [
    "PREFIX_CACHE_OFF_ERROR_CLASS",
    "CONTEXT_COPY_REFUSAL",
    "AGENTX_PREFLIGHT_RETURNCODE",
    "BaselineExecutor",
    "ConcSweepExecutor",
    "DEFAULT_KEEP_THRESHOLD_PCT",
    "EVAL_PROBE_UNPATCHABLE_RETURNCODE",
    "ExploreExecutor",
    "GridVariant",
    "ORCHESTRATOR_CANCELLED_RETURNCODE",
    "ReportExecutor",
    "SESSION_TIME_EXHAUSTED_RETURNCODE",
    "SPECULATIVE_DECODING_PINNED",
    "SessionBreakdownExecutor",
    "SpeculativeDecodingPinnedError",
    "TargetAnalysisExecutor",
    "VariantResult",
    "baseline_executor",
    "build_launch_evidence",
    "conc_sweep_executor",
    "context_copy_refusal",
    "default_baseline_config",
    "explore_executor",
    "harvest_leaked_artifacts",
    "materialize_config_with_envs",
    "num_gpus_for_config",
    "refuse_cacheless_agentx_baseline",
    "refuse_unpinned_speculative_change",
    "report_executor",
    "resolve_lifecycle_params",
    "run_grid",
    "session_breakdown_executor",
    "strip_visible_devices_from_config",
    "sync_benchmark_timeout",
    "teardown_lifecycle_server",
]
