# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Concrete ActionRunner implementations."""

from ._proposal_identity import content_fingerprint, controls_of, is_executable, normalize_proposal
from .baseline import baseline_executor
from .explore import explore_executor
from .report import report_executor
from .session_breakdown import session_breakdown_executor
from .target_analysis import TargetAnalysisExecutor
from .workload._workload_envs import (
    BenchmarkTarget,
    LaunchLayer,
    MaterializeOptions,
    default_baseline_config,
    materialize_benchmark_config,
)
from .grid._grid_base import GridVariant, VariantResult, grid_run_dirs, variant_slot
from ._profile_setup import VLLM_PATCH_PROFILER_FIELDS
from ._server_patcher import ensure_vllm_patched_for_tracelens
from .grid._grid_runner import _run_grid_warmup_enabled as run_grid_warmup_enabled
from .grid._grid_runner import GridLaunch, GridRecipe, GridStop, run_grid, teardown_variant_server
from .serving._server_lifecycle import resolve_lifecycle_params
from ._subprocess_kill import (
    BOOT_BUDGET_ENV,
    COOPERATIVE_REAP_BUDGET_SEC,
    ORCHESTRATOR_CANCELLED_RETURNCODE,
    SESSION_TIME_EXHAUSTED_RETURNCODE,
    RoundWatch,
    run_with_session_kill,
)
from .benchmark_result import harvest_leaked_artifacts
from .workload._agentx_checkpoint import CHECKPOINT_ARTIFACT_NAME, AgentxCheckpoint
from .workload._agentx_prefix_cache import PREFIX_CACHE_OFF_ERROR_CLASS, refuse_cacheless_agentx_baseline
from ._failed_step import failed_step_error_class
from ._round_folder import RoundFolder, is_round_folder, read_round_launch
from ._suite.config import CHECK_SERVER_ENVS, fixed_workload
from ._suite.rounds import (
    SERVER_CPUS_ENV,
    TOOL_CACHE_ENV,
    SuiteRoundRefused,
    external_agentx_config,
    staged_tool_cache,
)
from .framework_change._spec_decode_context_copy import CONTEXT_COPY_REFUSAL, context_copy_refusal
from .framework_change._spec_decode_pin import (
    SpeculativeDecodingPinnedError,
    declared_acceptance_reason,
    refuse_unpinned_speculative_change,
    speculative_pin_reason,
)
from .patching._patch_laying import PatchedServer, is_patched_copy
from ._profile_setup import SGLANG_PATCH_SERVER_FLAG
from ._server_patcher import ensure_patched_copy, is_kernel_shape_tool, lay_tracelens_patch

__all__ = [
    "BOOT_BUDGET_ENV",
    "SGLANG_PATCH_SERVER_FLAG",
    "VLLM_PATCH_PROFILER_FIELDS",
    "PatchedServer",
    "ensure_patched_copy",
    "ensure_vllm_patched_for_tracelens",
    "PREFIX_CACHE_OFF_ERROR_CLASS",
    "is_kernel_shape_tool",
    "is_patched_copy",
    "lay_tracelens_patch",
    "CONTEXT_COPY_REFUSAL",
    "CHECK_SERVER_ENVS",
    "BenchmarkTarget",
    "COOPERATIVE_REAP_BUDGET_SEC",
    "GridLaunch",
    "GridRecipe",
    "GridStop",
    "AgentxCheckpoint",
    "CHECKPOINT_ARTIFACT_NAME",
    "GridVariant",
    "LaunchLayer",
    "MaterializeOptions",
    "ORCHESTRATOR_CANCELLED_RETURNCODE",
    "RoundFolder",
    "RoundWatch",
    "SESSION_TIME_EXHAUSTED_RETURNCODE",
    "SpeculativeDecodingPinnedError",
    "SERVER_CPUS_ENV",
    "SuiteRoundRefused",
    "TOOL_CACHE_ENV",
    "TargetAnalysisExecutor",
    "VariantResult",
    "baseline_executor",
    "content_fingerprint",
    "controls_of",
    "context_copy_refusal",
    "default_baseline_config",
    "explore_executor",
    "failed_step_error_class",
    "external_agentx_config",
    "fixed_workload",
    "grid_run_dirs",
    "harvest_leaked_artifacts",
    "is_executable",
    "is_round_folder",
    "materialize_benchmark_config",
    "normalize_proposal",
    "read_round_launch",
    "declared_acceptance_reason",
    "refuse_cacheless_agentx_baseline",
    "refuse_unpinned_speculative_change",
    "report_executor",
    "resolve_lifecycle_params",
    "run_grid",
    "run_grid_warmup_enabled",
    "run_with_session_kill",
    "session_breakdown_executor",
    "staged_tool_cache",
    "speculative_pin_reason",
    "teardown_variant_server",
    "variant_slot",
]
