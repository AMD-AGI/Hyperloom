"""Kernel Manager prompt templates — OOB round (per-backend), local test
(compile/correct/bench), patch-gen, classify.
"""

from __future__ import annotations

import json
from typing import Any

from .system import HARDWARE_CONTEXT, MANDATORY_CONSTRAINTS


# -----------------------------------------------------------------------
# OOB round prompt (per-backend, with session history + findings)
# -----------------------------------------------------------------------

def prompt_oob_round(
    target: dict[str, Any],
    session_history: list[dict[str, Any]],
    findings: list[dict[str, Any]],
    round_num: int,
    backend: str = "",
) -> str:
    kernel_name = target.get("kernel_name", "unknown")
    source_file = target.get("source_file", "unknown")
    gpu_pct = target.get("gpu_pct", 0)
    strategy = target.get("strategy", "oob-rewrite")

    parts: list[str] = []
    parts.append(f"# Kernel Optimization: {kernel_name}")
    parts.append(f"Round {round_num}/5 | Strategy: {strategy} | GPU%: {gpu_pct:.1f}")
    parts.append("")
    parts.append(HARDWARE_CONTEXT)
    parts.append(MANDATORY_CONSTRAINTS)
    parts.append("")
    parts.append(f"Source file: {source_file}")

    dispatch = target.get("dispatch_analysis", {})
    if dispatch:
        parts.append(f"Active path: {dispatch.get('active_path', '?')}")
        parts.append(f"Optimal path: {dispatch.get('optimal_path', '?')}")
        if dispatch.get("dispatch_bug"):
            parts.append("WARNING: dispatch bug detected")

    shapes = target.get("trace_shapes", [])
    if shapes:
        parts.append(f"Trace shapes: {shapes[:10]}")

    constraints = target.get("constraints", {})
    if constraints:
        parts.append(f"Constraints: {json.dumps(constraints)}")

    rca = target.get("rca_constraints", {})
    if rca:
        parts.append("\n--- RCA CONSTRAINTS (from Watchdog investigation) ---")
        parts.append(f"Root cause: {rca.get('root_cause', 'unknown')}")
        if rca.get("constraint"):
            parts.append(f"Constraint: {rca['constraint']}")
        if rca.get("avoid"):
            parts.append(f"AVOID: {rca['avoid']}")
        if rca.get("compiler_flags"):
            parts.append(f"Compiler flags: {rca['compiler_flags']}")

    if round_num > 1 and session_history:
        parts.append("\n--- SESSION HISTORY (previous rounds) ---")
        for entry in session_history:
            r = entry.get("round", "?")
            outcome = entry.get("outcome", "?")
            be = entry.get("backend", "?")
            summary = entry.get("attempt_summary", "")[:150]
            err = entry.get("error_analysis", "")
            if isinstance(err, str):
                err = err[:200]
            parts.append(f"Round {r} ({be}): {outcome}")
            if err:
                parts.append(f"  Error: {err}")
            if entry.get("constraints_used"):
                parts.append(f"  Constraints used: {entry['constraints_used']}")
            if summary:
                parts.append(f"  Summary: {summary}...")

    if findings:
        parts.append("\n--- WATCHDOG FINDINGS ---")
        for f in findings[-3:]:
            parts.append(f"  [{f.get('classification', '?')}] {f.get('root_cause', '')[:100]}")
            guidance = f.get("actionable_guidance", {})
            if guidance.get("constraint"):
                parts.append(f"  Constraint: {guidance['constraint']}")
            if guidance.get("avoid"):
                parts.append(f"  Avoid: {guidance['avoid']}")

    # Backend-specific instructions
    if backend == "geak":
        parts.append("\n--- BACKEND: GEAK (GPU pod) ---")
        parts.append("You have GPU access. This is your advantage — USE IT.")
        parts.append("1. Read the ORIGINAL kernel source file completely: cat <source_file>")
        parts.append("2. Analyze register pressure: count VGPRs needed, target <=64 for 4-wave occupancy")
        parts.append("3. Analyze memory access patterns: coalescing, bank conflicts, global vs shared memory")
        parts.append("4. Write the optimized kernel and COMPILE it: hipcc -O3 --amdgpu-target=gfx950")
        parts.append("5. If compilation fails, fix errors and retry (up to 5 attempts)")
        parts.append("6. Run micro-benchmark with ALL shapes from the trace")
        parts.append("7. If speedup < 1.5x, try a different optimization strategy and benchmark again")
        parts.append("Strategies to try in order: reduce register pressure → improve memory coalescing")
        parts.append("→ use MFMA instructions → software pipelining → tile size tuning → kernel fusion")
    elif backend == "codex":
        parts.append("\n--- BACKEND: Codex ---")
        parts.append("No GPU — focus on generating correct, highly-optimized code.")
        parts.append("1. Read the original kernel source. Understand every line.")
        parts.append("2. Identify the compute bottleneck: is it compute-bound or memory-bound?")
        parts.append("3. For Triton kernels: reduce BLOCK dims if register-heavy, add tl.load with")
        parts.append("   eviction_policy='evict_first' for streaming data, use tl.dot for MFMA")
        parts.append("4. Minimize live variables in inner loop to reduce VGPR pressure")
        parts.append("5. Output COMPLETE rewritten file — no stubs, no TODOs, no truncation")
        parts.append("6. If you get compile errors back, fix them precisely and resubmit")
    elif backend == "claude":
        parts.append("\n--- BACKEND: Claude ---")
        parts.append("Deep multi-turn reasoning — you have time to think carefully.")
        parts.append("1. Read the entire original kernel source file")
        parts.append("2. Trace the full data flow: inputs → computation → outputs")
        parts.append("3. Compute arithmetic intensity: FLOPs / bytes. Determine roofline bound.")
        parts.append("4. For memory-bound kernels: reduce global memory accesses, fuse operations,")
        parts.append("   use vectorized loads (tl.load with BLOCK alignment)")
        parts.append("5. For compute-bound kernels: use MFMA (matrix fused multiply-add),")
        parts.append("   optimize tile sizes for the specific shapes from the trace")
        parts.append("6. Analyze register pressure: on gfx950, 256 VGPRs/CU, target <=64/thread")
        parts.append("   for 4 waves. Count all live tl.load results, intermediates, accumulators.")
        parts.append("7. Consider: can this kernel be fused with an adjacent kernel?")
        parts.append("8. Write the COMPLETE optimized kernel — every line, no truncation")
    elif backend == "llm-proxy":
        parts.append("\n--- BACKEND: LLM Proxy ---")
        parts.append("Single-shot generation. Make it count.")
        parts.append("1. Return the COMPLETE optimized kernel file as a ```python code block")
        parts.append("2. Key optimizations to apply:")
        parts.append("   - Reduce BLOCK_M/N/K if register-heavy (target 4-wave occupancy)")
        parts.append("   - Use tl.dot for matrix operations (maps to MFMA)")
        parts.append("   - Vectorize loads: ensure BLOCK dims are multiples of 16")
        parts.append("   - Minimize tl.store calls (accumulate in registers)")
        parts.append("   - Add num_stages=2 for software pipelining if memory-bound")
        parts.append("3. NO stubs, NO TODOs, NO truncation. Every line of the file.")

    parts.append("\n--- TASK ---")
    parts.append("GOAL: Rewrite this kernel to be >=1.5x faster on MI355X (gfx950, CDNA4).")
    parts.append("")
    parts.append("APPROACH:")
    parts.append("1. First, read and understand the original code completely")
    parts.append("2. Identify the #1 bottleneck (register pressure? memory bandwidth? compute?)")
    parts.append("3. Apply the most impactful optimization for that bottleneck")
    parts.append("4. Verify the function signature is EXACTLY preserved")
    parts.append("5. Output the COMPLETE file — no truncation, no '...' placeholders")
    parts.append("")
    parts.append("CONSTRAINTS:")
    parts.append(MANDATORY_CONSTRAINTS)

    return "\n".join(parts)


# -----------------------------------------------------------------------
# Local test prompts
# -----------------------------------------------------------------------

def prompt_local_test_compile(target: dict[str, Any], optimized_code: str) -> str:
    source_type = target.get("source_type", "triton")
    kernel_name = target.get("kernel_name", "unknown")
    source_file = target.get("source_file", "unknown")

    if source_type in ("cpp_cuda", "cpp_hip"):
        compile_cmd = "hipcc -O3 --amdgpu-target=gfx950"
        timeout = "120s"
    elif "sgl_kernel" in source_file or "sgl-kernel" in source_file:
        compile_cmd = "cd /sgl-workspace/sglang/sgl-kernel && python setup_rocm.py install"
        timeout = "600s"
    else:
        compile_cmd = f"python -c 'import importlib; exec(open(\"{source_file}\").read())'"
        timeout = "30s"

    return (
        f"COMPILE CHECK for {kernel_name}\n"
        f"Source type: {source_type}\n"
        f"Compile command: {compile_cmd}\n"
        f"Timeout: {timeout}\n\n"
        f"1. Write the optimized code to a temp file\n"
        f"2. Attempt compilation/import\n"
        f"3. Report success or failure\n\n"
        f"Optimized code:\n```python\n{optimized_code}\n```\n\n"
        f"Write to $OUTPUT_FILE: compiled (bool), error_type, error_message\n"
    )


def prompt_local_test_correctness(target: dict[str, Any], optimized_code: str) -> str:
    kernel_name = target.get("kernel_name", "unknown")
    shapes = target.get("trace_shapes", [])
    return (
        f"CORRECTNESS CHECK for {kernel_name}\n"
        f"ATOL=1e-2, RTOL=1e-2\n"
        f"Shapes from trace: {shapes[:5]}\n\n"
        f"1. Run original kernel with trace shapes\n"
        f"2. Run optimized kernel with same inputs\n"
        f"3. torch.allclose(original_output, optimized_output, atol=1e-2, rtol=1e-2)\n"
        f"4. Test all shapes\n\n"
        f"Optimized code:\n```python\n{optimized_code}\n```\n\n"
        f"Write to $OUTPUT_FILE: correct (bool), shapes_tested, "
        f"max_abs_diff, failed_shape\n"
    )


def prompt_local_test_benchmark(target: dict[str, Any], optimized_code: str) -> str:
    kernel_name = target.get("kernel_name", "unknown")
    shapes = target.get("trace_shapes", [])
    return (
        f"MICRO-BENCHMARK for {kernel_name}\n"
        f"Multipliers: [1, 4, 16, 64] on xnumel\n"
        f"Warmup: 20 iters, Timed: 200 iters\n"
        f"Shapes from trace: {shapes[:5]}\n\n"
        f"1. For each shape × multiplier:\n"
        f"   a. Run original kernel (warmup + timed)\n"
        f"   b. Run optimized kernel (warmup + timed)\n"
        f"   c. Record speedup = original_time / optimized_time\n"
        f"2. Pass criteria: avg_speedup > 1.05 AND no shape < 0.95x\n\n"
        f"Optimized code:\n```python\n{optimized_code}\n```\n\n"
        f"Write to $OUTPUT_FILE: avg_speedup, per_shape (list of "
        f"{{shape, speedup, original_ms, optimized_ms}})\n"
    )


def prompt_adversarial_test(target: dict[str, Any], optimized_code: str) -> str:
    kernel_name = target.get("kernel_name", "unknown")
    shapes = target.get("trace_shapes", [])
    return (
        f"ADVERSARIAL STRESS TEST for {kernel_name}\n"
        f"Edge cases: empty/zero-dim tensors, large shapes derived from trace, "
        f"NaN/Inf inputs where safe.\n"
        f"Shapes from trace: {shapes[:5]}\n\n"
        f"1. Compare optimized vs reference on each edge case\n"
        f"2. Record crashes or tolerance violations\n\n"
        f"Optimized code:\n```python\n{optimized_code}\n```\n\n"
        f"Write to $OUTPUT_FILE: failures (list of dicts with case, error); "
        f"use empty list if all passed\n"
    )


# -----------------------------------------------------------------------
# Patch generation
# -----------------------------------------------------------------------

def prompt_patch_gen(
    target: dict[str, Any],
    optimized_code: str,
    test_result: dict[str, Any],
) -> str:
    kernel_name = target.get("kernel_name", "unknown")
    source_file = target.get("source_file", "unknown")
    source_type = target.get("source_type", "python")

    return (
        f"GENERATE MERGE-READY PATCH for {kernel_name}\n\n"
        f"Source file: {source_file}\n"
        f"Source type: {source_type}\n"
        f"Test result: {json.dumps(test_result, default=str)[:500]}\n\n"
        f"1. Create backup: cp {source_file} $PATCH_DIR/original_<filename>.bak\n"
        f"2. Write optimized code to $PATCH_DIR/optimized_<filename>.py\n"
        f"3. Write micro_benchmark.json\n"
        f"4. Determine apply_method: str-replace | file-replace | diff-apply\n"
        f"5. Determine rollback_command: cp $PATCH_DIR/original_... {source_file}\n"
        f"6. Determine rebuild requirements\n"
        f"7. Write metadata.json with ALL fields\n\n"
        f"Optimized code:\n```python\n{optimized_code}\n```\n"
    )


# -----------------------------------------------------------------------
# Classification
# -----------------------------------------------------------------------

def prompt_classify_target(target: dict[str, Any]) -> str:
    return (
        f"Classify this kernel optimization target:\n"
        f"Kernel: {target.get('kernel_name', '?')}\n"
        f"Source: {target.get('source_file', '?')}\n"
        f"Type: {target.get('source_type', '?')}\n"
        f"Strategy: {target.get('strategy', '?')}\n"
        f"Dispatch bug: {target.get('dispatch_analysis', {}).get('dispatch_bug', False)}\n\n"
        f"Classification table:\n"
        f"  dispatch-fix → [local]\n"
        f"  config-only → [local]\n"
        f"  oob-rewrite → [geak, codex, claude, llm-proxy]\n"
        f"  oob-rewrite-register-constrained → [codex, claude]\n"
        f"  triton-rewrite → [geak, codex, claude, llm-proxy]\n"
        f"  hip-kernel → [geak, claude]\n"
        f"  framework-scheduling → [claude]\n"
        f"  kernel-fusion → [geak, claude]\n\n"
        f"Write to $OUTPUT_FILE: strategy, backends\n"
    )
