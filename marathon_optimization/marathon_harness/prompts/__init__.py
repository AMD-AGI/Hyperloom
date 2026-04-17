"""Prompt templates for all Marathon harness LLM calls."""

from .system import SYSTEM_PROMPT, HARDWARE_CONTEXT, MANDATORY_CONSTRAINTS, BENCHMARK_INTEGRITY, build_system_prompt, configure
from .orchestrator import (
    prompt_warm_start,
    prompt_re_profile,
    prompt_deep_analysis,
    prompt_execute_dispatch_fix,
    prompt_execute_operator_tuning,
    prompt_execute_framework_rebuild,
    prompt_execute_kernel_opt,
    prompt_execute_comm_optimization,
    prompt_execute_compiler_tuning,
    prompt_execute_action,
    prompt_accuracy_gate,
    prompt_rescore,
    prompt_dream,
    prompt_sweep,
    prompt_report,
    prompt_recover,
    prompt_re_explore,
    prompt_apply_instruction,
    prompt_rebuild,
    prompt_shell_command,
    prompt_verify,
    prompt_benchmark,
    prompt_diagnose_failure,
    prompt_execute_config_only,
)
from .kernel_mgr import (
    prompt_oob_round,
    prompt_local_test_compile,
    prompt_local_test_correctness,
    prompt_local_test_benchmark,
    prompt_adversarial_test,
    prompt_patch_gen,
    prompt_classify_target,
)
from .watchdog import (
    prompt_triage,
    prompt_investigate,
)
