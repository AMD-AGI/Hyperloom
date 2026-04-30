"""Tests for ``orchestrator.iron_rules`` — IMPL-CHECKLIST Phase 1 §1.1‒1.10."""
from __future__ import annotations

import pytest

from inference_optimizer.orchestrator.execution_mode import ExecutionMode
from inference_optimizer.orchestrator.iron_rules import (
    IRON_RULES,
    Severity,
    Violation,
    all_rules,
    render_for_prompt,
    rules_for_mode,
    validate_action,
)


# ---------------------------------------------------------------------------
# IRON_RULES registry
# ---------------------------------------------------------------------------
def test_seven_iron_rules_exist():
    ids = [r.id for r in IRON_RULES]
    assert ids == [f"IR-{i}" for i in range(1, 8)]


def test_all_rules_returns_full_tuple():
    assert all_rules() == IRON_RULES


def test_rules_for_quick_excludes_geak_only_rules():
    quick = {r.id for r in rules_for_mode(ExecutionMode.QUICK_PARAM_SWEEP)}
    # IR-4 / IR-5 apply to every mode; IR-1/2/3/6/7 are guided+marathon only
    assert "IR-4" in quick and "IR-5" in quick
    assert "IR-1" not in quick
    assert "IR-2" not in quick
    assert "IR-6" not in quick


def test_rules_for_marathon_includes_all():
    marathon = {r.id for r in rules_for_mode(ExecutionMode.MARATHON_MULTI_AGENT)}
    assert marathon == {f"IR-{i}" for i in range(1, 8)}


def test_rules_for_string_mode_keyword():
    assert {r.id for r in rules_for_mode("guided")} == {
        f"IR-{i}" for i in range(1, 8)
    }


# ---------------------------------------------------------------------------
# IR-1: parallel kernel candidate submission
# ---------------------------------------------------------------------------
def test_ir1_violated_when_kernel_opt_lacks_parallel_flag():
    meta = {"name": "kernel-opt", "family": "deep_kernel"}
    vs = validate_action(meta, "guided")
    assert any(v.rule_id == "IR-1" for v in vs)


def test_ir1_satisfied_with_parallel_flag():
    meta = {
        "name": "kernel-opt",
        "family": "deep_kernel",
        "execution_flags": ["parallel_geak_submission"],
        "required_follow_ups": ["integrate"],
    }
    vs = validate_action(meta, "guided")
    assert not any(v.rule_id == "IR-1" for v in vs)


def test_ir1_skipped_in_quick_mode():
    meta = {"name": "kernel-opt", "family": "deep_kernel"}
    vs = validate_action(meta, "quick")
    assert not any(v.rule_id == "IR-1" for v in vs)


# ---------------------------------------------------------------------------
# IR-2: kernel source modification only by kernel-opt / integrate
# ---------------------------------------------------------------------------
def test_ir2_violated_when_random_action_patches_kernel():
    meta = {
        "name": "params",
        "family": "shallow",
        "side_effects": ["patches_kernel_source"],
    }
    vs = validate_action(meta, "guided")
    assert any(v.rule_id == "IR-2" for v in vs)


def test_ir2_satisfied_for_kernel_opt():
    meta = {
        "name": "kernel-opt",
        "family": "deep_kernel",
        "side_effects": ["patches_kernel_source", "patches_workspace"],
        "execution_flags": ["parallel_geak_submission"],
        "required_follow_ups": ["integrate"],
    }
    vs = validate_action(meta, "guided")
    assert not any(v.rule_id == "IR-2" for v in vs)


# ---------------------------------------------------------------------------
# IR-3: integrate must follow kernel-opt
# ---------------------------------------------------------------------------
def test_ir3_violated_without_integrate_in_follow_ups():
    meta = {
        "name": "kernel-opt",
        "family": "deep_kernel",
        "execution_flags": ["parallel_geak_submission"],
        "required_follow_ups": [],
    }
    vs = validate_action(meta, "guided")
    assert any(v.rule_id == "IR-3" for v in vs)


def test_ir3_satisfied_with_integrate():
    meta = {
        "name": "kernel-opt",
        "family": "deep_kernel",
        "execution_flags": ["parallel_geak_submission"],
        "required_follow_ups": ["integrate"],
    }
    vs = validate_action(meta, "guided")
    assert not any(v.rule_id == "IR-3" for v in vs)


# ---------------------------------------------------------------------------
# IR-4: kill_server + check_gpu_memory before launch
# ---------------------------------------------------------------------------
def test_ir4_violated_when_server_lifecycle_lacks_preflight():
    meta = {
        "name": "bench_runner",
        "family": "prep",
        "requires_lanes": ["server_lifecycle", "benchmark_lane"],
        "preflight": ["kill_server"],  # missing check_gpu_memory
    }
    vs = validate_action(meta, "guided")
    assert any(v.rule_id == "IR-4" for v in vs)


def test_ir4_satisfied_with_full_preflight():
    meta = {
        "name": "bench_runner",
        "family": "prep",
        "requires_lanes": ["server_lifecycle", "benchmark_lane"],
        "preflight": ["kill_server", "check_gpu_memory"],
    }
    vs = validate_action(meta, "guided")
    assert not any(v.rule_id == "IR-4" for v in vs)


def test_ir4_skipped_for_actions_without_server_lifecycle():
    meta = {"name": "profile", "family": "analysis", "requires_lanes": ["profile_lane"]}
    vs = validate_action(meta, "marathon")
    assert not any(v.rule_id == "IR-4" for v in vs)


# ---------------------------------------------------------------------------
# IR-5: forbidden ``pkill -f sglang``
# ---------------------------------------------------------------------------
def test_ir5_violated_with_pkill_f_sglang():
    meta = {"name": "x", "commands": ["pkill -f sglang"]}
    vs = validate_action(meta, "quick")
    assert any(v.rule_id == "IR-5" for v in vs)


def test_ir5_violated_with_pkill_9_f_sglang():
    meta = {
        "name": "x",
        "shell_snippets": ["echo hi", "pkill -9 -f sglang.launch_server"],
    }
    vs = validate_action(meta, "marathon")
    assert any(v.rule_id == "IR-5" for v in vs)


def test_ir5_allows_targeted_pgrep_pipe_kill():
    meta = {
        "name": "x",
        "commands": [
            "kill $(pgrep -f sglang.launch_server)",
        ],
    }
    vs = validate_action(meta, "quick")
    assert not any(v.rule_id == "IR-5" for v in vs)


# ---------------------------------------------------------------------------
# IR-6: patch_inductor argument enforcement
# ---------------------------------------------------------------------------
def test_ir6_violated_without_target_file():
    meta = {
        "name": "kernel-opt",
        "family": "deep_kernel",
        "execution_flags": ["parallel_geak_submission"],
        "required_follow_ups": ["integrate"],
        "patch_inductor_invocations": [
            {"argv": ["--best-config", "cfg.json"]}
        ],
    }
    vs = validate_action(meta, "guided")
    assert any(v.rule_id == "IR-6" for v in vs)


def test_ir6_violated_when_cache_dir_passed():
    meta = {
        "name": "kernel-opt",
        "family": "deep_kernel",
        "execution_flags": ["parallel_geak_submission"],
        "required_follow_ups": ["integrate"],
        "patch_inductor_invocations": [
            {"argv": ["--target-file", "k.py", "--cache-dir", "/tmp"]}
        ],
    }
    vs = validate_action(meta, "guided")
    assert any(v.rule_id == "IR-6" for v in vs)


def test_ir6_violated_block_size_without_best_config():
    meta = {
        "name": "kernel-opt",
        "family": "deep_kernel",
        "execution_flags": ["parallel_geak_submission"],
        "required_follow_ups": ["integrate"],
        "patch_inductor_invocations": [
            {"argv": ["--target-file", "k.py"], "tuning_keys": ["block_size"]}
        ],
    }
    vs = validate_action(meta, "guided")
    assert any(v.rule_id == "IR-6" for v in vs)


def test_ir6_satisfied_with_full_args():
    meta = {
        "name": "kernel-opt",
        "family": "deep_kernel",
        "execution_flags": ["parallel_geak_submission"],
        "required_follow_ups": ["integrate"],
        "patch_inductor_invocations": [
            {
                "argv": ["--target-file", "k.py", "--best-config", "cfg.json"],
                "tuning_keys": ["block_size", "num_warps"],
            }
        ],
    }
    vs = validate_action(meta, "guided")
    assert not any(v.rule_id == "IR-6" for v in vs)


# ---------------------------------------------------------------------------
# IR-7: never modify GEAK config
# ---------------------------------------------------------------------------
def test_ir7_violated_when_modifying_geak_config():
    meta = {"name": "x", "side_effects": ["modifies_geak_config"]}
    vs = validate_action(meta, "marathon")
    assert any(v.rule_id == "IR-7" for v in vs)


def test_ir7_allows_tracing_headers_exception():
    meta = {
        "name": "x",
        "side_effects": ["modifies_geak_config"],
        "geak_config_exceptions": ["tracing_headers"],
    }
    vs = validate_action(meta, "marathon")
    assert not any(v.rule_id == "IR-7" for v in vs)


# ---------------------------------------------------------------------------
# render_for_prompt
# ---------------------------------------------------------------------------
def test_render_for_prompt_quick_includes_only_universal_rules():
    md = render_for_prompt("quick")
    assert "## Iron Rules" in md
    assert "IR-4" in md
    assert "IR-5" in md
    assert "IR-1" not in md


def test_render_for_prompt_marathon_includes_every_rule():
    md = render_for_prompt("marathon")
    for i in range(1, 8):
        assert f"IR-{i}" in md


# ---------------------------------------------------------------------------
# Aggregate behaviour
# ---------------------------------------------------------------------------
def test_validate_collects_multiple_violations():
    """A truly broken metadata should report all the rules it trips."""
    meta = {
        "name": "kernel-opt",
        "family": "deep_kernel",
        "requires_lanes": ["server_lifecycle"],   # IR-4
        "side_effects": ["modifies_geak_config"], # IR-7
        "commands": ["pkill -f sglang"],          # IR-5
        # missing IR-1 parallel flag, IR-3 integrate follow-up
    }
    vs = validate_action(meta, "marathon")
    rule_ids = {v.rule_id for v in vs}
    assert {"IR-1", "IR-3", "IR-4", "IR-5", "IR-7"} <= rule_ids
    # Plan A — IR-3/4/5 stay BLOCK (process safety / gain validation);
    # IR-1/2/6/7 are now WARN. Each violation inherits the rule's
    # configured severity, not a hard-coded default.
    by_id = {v.rule_id: v.severity for v in vs}
    assert by_id["IR-3"] == Severity.BLOCK
    assert by_id["IR-4"] == Severity.BLOCK
    assert by_id["IR-5"] == Severity.BLOCK
    assert by_id["IR-1"] == Severity.WARN
    assert by_id["IR-7"] == Severity.WARN


# ---------------------------------------------------------------------------
# Plan A — IR severity policy
# ---------------------------------------------------------------------------
def test_ir_severity_policy_plan_a():
    """IR-3/IR-4/IR-5 BLOCK, IR-1/IR-2/IR-6/IR-7 WARN per Plan A."""
    from inference_optimizer.orchestrator.iron_rules import IRON_RULES
    by_id = {r.id: r.severity for r in IRON_RULES}
    assert by_id["IR-1"] == Severity.WARN
    assert by_id["IR-2"] == Severity.WARN
    assert by_id["IR-3"] == Severity.BLOCK
    assert by_id["IR-4"] == Severity.BLOCK
    assert by_id["IR-5"] == Severity.BLOCK
    assert by_id["IR-6"] == Severity.WARN
    assert by_id["IR-7"] == Severity.WARN


def test_render_for_prompt_uses_must_for_block_should_for_warn():
    """Prompt rendering surfaces severity as MUST/should so the LLM can
    weight hard guarantees against soft recommendations."""
    md = render_for_prompt("marathon")
    # IR-3 is BLOCK -> "MUST"
    assert "**IR-3**" in md and "MUST" in md
    # IR-1 is WARN -> "should"
    assert "**IR-1**" in md and "should" in md
