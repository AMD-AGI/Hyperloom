# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Concrete ActionRunner implementations."""

from .baseline import baseline_executor
from .explore import explore_executor
from .conc_sweep import conc_sweep_executor
from .report import report_executor
from .session_breakdown import session_breakdown_executor
from .target_analysis import TargetAnalysisExecutor
from ._workload_envs import (
    BenchmarkTarget,
    LaunchLayer,
    MaterializeOptions,
    default_baseline_config,
    materialize_benchmark_config,
    materialize_config_with_envs,
)
from ._grid_base import GridVariant, VariantResult
from ._profile_setup import VLLM_PATCH_PROFILER_FIELDS
from ._server_patcher import ensure_vllm_patched_for_tracelens
from ._magpie_patcher import ensure_lm_eval_install_once
from ._inferencex_patcher import (
    benchmark_serving_path_in,
    ensure_benchmark_lib_patched,
    ensure_benchmark_serving_patched,
)
from ._grid_runner import _num_gpus_for_config as num_gpus_for_config
from ._grid_runner import _run_grid_warmup_enabled as run_grid_warmup_enabled
from ._grid_runner import GridLaunch, GridRecipe, GridStop, run_grid, sync_benchmark_timeout
from ._ray_backend import strip_visible_devices_from_config
from ._server_lifecycle import REUSE_PORT_DEFAULT, resolve_lifecycle_params, teardown_lifecycle_server
from ._server_lifecycle import _assign_free_port as assign_free_port
from ._subprocess_kill import (
    AGENTX_PREFLIGHT_RETURNCODE,
    COOPERATIVE_REAP_BUDGET_SEC,
    EVAL_PROBE_UNPATCHABLE_RETURNCODE,
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    RoundWatch,
    run_with_session_kill,
)
from .benchmark_result import harvest_leaked_artifacts
from ._agentx_prefix_cache import PREFIX_CACHE_OFF_ERROR_CLASS, refuse_cacheless_agentx_baseline
from ._launch_evidence import RequestedLaunch, build_launch_evidence
from ._spec_decode_context_copy import CONTEXT_COPY_REFUSAL, context_copy_refusal
from ._spec_decode_pin import (
    SPECULATIVE_DECODING_PINNED,
    SpeculativeDecodingPinnedError,
    refuse_unpinned_speculative_change,
)

__all__ = [
    "VLLM_PATCH_PROFILER_FIELDS",
    "ensure_vllm_patched_for_tracelens",
    "ensure_lm_eval_install_once",
    "PREFIX_CACHE_OFF_ERROR_CLASS",
    "CONTEXT_COPY_REFUSAL",
    "AGENTX_PREFLIGHT_RETURNCODE",
    "BenchmarkTarget",
    "COOPERATIVE_REAP_BUDGET_SEC",
    "EVAL_PROBE_UNPATCHABLE_RETURNCODE",
    "GridLaunch",
    "GridRecipe",
    "GridStop",
    "GridVariant",
    "LaunchLayer",
    "MaterializeOptions",
    "ORCHESTRATOR_CANCELLED_RETURNCODE",
    "REUSE_PORT_DEFAULT",
    "RequestedLaunch",
    "RoundWatch",
    "SESSION_TIME_EXHAUSTED_RETURNCODE",
    "SPECULATIVE_DECODING_PINNED",
    "SpeculativeDecodingPinnedError",
    "TargetAnalysisExecutor",
    "VariantResult",
    "assign_free_port",
    "baseline_executor",
    "benchmark_serving_path_in",
    "build_launch_evidence",
    "conc_sweep_executor",
    "context_copy_refusal",
    "default_baseline_config",
    "ensure_benchmark_lib_patched",
    "ensure_benchmark_serving_patched",
    "explore_executor",
    "harvest_leaked_artifacts",
    "materialize_benchmark_config",
    "materialize_config_with_envs",
    "num_gpus_for_config",
    "refuse_cacheless_agentx_baseline",
    "refuse_unpinned_speculative_change",
    "report_executor",
    "resolve_lifecycle_params",
    "run_grid",
    "run_grid_warmup_enabled",
    "run_with_session_kill",
    "session_breakdown_executor",
    "strip_visible_devices_from_config",
    "sync_benchmark_timeout",
    "teardown_lifecycle_server",
]
