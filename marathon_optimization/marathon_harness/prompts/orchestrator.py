"""Orchestrator prompt templates — warm_start (3 modes), re_profile (T1-T5),
deep_analysis (phases + routing), 13 action types, accuracy_gate, rescore,
dream, sweep, report, recover, re_explore, exploratory_probe, hypothesis_ab_benchmark.
"""

from __future__ import annotations
import json
from typing import Any
from . import system as _sys


def _state_ctx(state_summary: str) -> str:
    return f"\n--- CURRENT STATE ---\n{state_summary}\n"


# -----------------------------------------------------------------------
# Step 0  WARM-START
# -----------------------------------------------------------------------

_SCOPING_CONSTRAINT = (
    "\n\nWORKSPACE SCOPING — CRITICAL:\n"
    "- ONLY read files under $BASE_DIR/ and $INFERENCEX_PATH/.\n"
    "- Do NOT explore sibling directories, other models' data, or unrelated sessions.\n"
    "- Do NOT browse /shared_nfs/ looking for other projects.\n"
    "- Stay focused on THIS model's scripts, configs, and results.\n"
)

_READ_ONLY_CONSTRAINT = (
    "\n\nREAD-ONLY MODE — CRITICAL:\n"
    "- This is an ANALYSIS-ONLY phase. Do NOT modify any source files.\n"
    "- Do NOT use the Edit tool or write to any .py, .sh, .csv, .json source files.\n"
    "- Do NOT apply patches, change configs, or modify launch scripts.\n"
    "- You may ONLY read files, run read-only bash commands, and write to $OUTPUT_FILE.\n"
    "- All code changes MUST happen in the DFS phase where they go through "
    "benchmark and accuracy gates.\n"
)


def prompt_warm_start(
    mode: str,
    state_summary: str,
    handoff_config: dict[str, Any] | None = None,
    baseline_script: str = "",
) -> str:
    ctx = _state_ctx(state_summary)
    if mode == "sprint":
        return (
            f"{_sys.SYSTEM_PROMPT}\n{ctx}"
            f"WARM-START MODE: sprint\n"
            f"Sprint handoff config:\n```json\n{handoff_config}\n```\n\n"
            f"Tasks:\n"
            f"1. Review config.json and applied patches\n"
            f"2. Load opportunities.json — classify each as dispatch-fix, operator-tuning, "
            f"deep-kernel-opt, framework-rebuild, comm-optimization\n"
            f"3. Score opportunities using base scores: deep-kernel-analysis=9, operator-tuning=7, "
            f"comm=5, deep-kernel-opt=6. Apply handoff boosts: marathon-candidate +3, "
            f"register-pressure-fixable +3, shape-tuning-untested +2, oob-untested +2\n"
            f"4. Write scored action_stack to $OUTPUT_FILE as JSON\n"
            f"{_SCOPING_CONSTRAINT}"
            f"{_READ_ONLY_CONSTRAINT}"
        )
    elif mode == "baseline":
        return (
            f"{_sys.SYSTEM_PROMPT}\n{ctx}"
            f"WARM-START MODE: baseline\n"
            f"Baseline launch script:\n```\n{baseline_script}\n```\n\n"
            f"Tasks:\n"
            f"1. Read ONLY files under $BASE_DIR/ — look for scripts/, results/, README.md\n"
            f"2. Parse model name, flags, env vars from server launch scripts in $BASE_DIR/scripts/\n"
            f"3. Detect GPU count and type from the scripts or via rocm-smi\n"
            f"4. Apply any patches from $BASE_DIR/patches/ if they exist\n"
            f"5. Write initial state to $OUTPUT_FILE as JSON\n"
            f"{_SCOPING_CONSTRAINT}"
            f"{_READ_ONLY_CONSTRAINT}"
        )
    else:
        return (
            f"{_sys.SYSTEM_PROMPT}\n{ctx}"
            f"WARM-START MODE: cold\n\n"
            f"Tasks:\n"
            f"1. Detect GPU count via `rocm-smi --showid` or `nvidia-smi -L`\n"
            f"2. Detect GPU type\n"
            f"3. Find model launch script under $BASE_DIR/scripts/\n"
            f"4. Write initial state to $OUTPUT_FILE as JSON\n"
            f"{_SCOPING_CONSTRAINT}"
            f"{_READ_ONLY_CONSTRAINT}"
        )


# -----------------------------------------------------------------------
# Step 1  RE-PROFILE
# -----------------------------------------------------------------------

def prompt_re_profile(state_summary: str, trace_path: str = "") -> str:
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"RE-PROFILE: run torch profiler and build kernel breakdown.\n\n"
        f"{'Existing trace: ' + trace_path if trace_path else 'No existing trace — run profiler.'}\n\n"
        f"IMPORTANT: First check if the inference server is running (curl localhost:8888/health).\n"
        f"If the server is NOT running, report status='server_down' and exit — "
        f"do NOT start the server yourself. The orchestrator manages server lifecycle.\n"
        f"Only read files under $BASE_DIR/ and the framework source directories.\n\n"
        f"Steps:\n"
        f"1. Ensure the inference server is running and healthy\n"
        f"2. Run profiler or load trace (filtered TP-0 only)\n"
        f"3. Parse kernel breakdown — name, GPU time %, category\n"
        f"4. Classify each kernel into tiers:\n"
        f"   T1_TRITON: Triton JIT kernels\n"
        f"   T2_AITER_CK: aiter/CK library kernels\n"
        f"   T3_FRAMEWORK: Framework Python scheduling overhead\n"
        f"   T4_COMM: NCCL/RCCL communication\n"
        f"   T5_COMPILED: compiled/fused kernels\n"
        f"5. Score candidates: gpu_pct * expected_speedup / cost_minutes "
        f"* (1-accuracy_risk) * (1-crash_risk)\n"
        f"   Expected speedups: T1=0.5, T2=0.2, T3=0.15, T4=0.10, T5=0.05\n"
        f"   Cost minutes: T1=15, T2=30, T3=45, T4=30, T5=20\n"
        f"6. Heuristic boosts:\n"
        f"   - T2+T5 >60% → add call-stack-opt and vendor-kernel-config\n"
        f"   - T4 >15% → add comm optimization\n"
        f"   - T3 idle >20% → add scheduling optimization\n"
        f"7. Write to $OUTPUT_FILE: kernel_breakdown, tier_summary, "
        f"kernel_opt_candidates (>1% GPU), trace_path\n"
        f"{_SCOPING_CONSTRAINT}"
        f"{_READ_ONLY_CONSTRAINT}"
    )


# -----------------------------------------------------------------------
# Step 2  DEEP ANALYSIS
# -----------------------------------------------------------------------

def prompt_deep_analysis(state_summary: str, kernel_candidates: list[dict]) -> str:
    candidates_str = "\n".join(
        f"  - {k.get('name', '?')}: {k.get('gpu_pct', 0):.1f}% GPU ({k.get('category', 'unknown')})"
        for k in kernel_candidates[:10]
    )
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"DEEP KERNEL ANALYSIS — FULL CALL-STACK TRIAGE + OPTIMIZATION PLAN\n"
        f"Top {len(kernel_candidates[:10])} kernels by GPU time:\n"
        f"{candidates_str}\n\n"
        f"You must deeply reason about EVERY kernel. Do not just classify — "
        f"investigate the actual code, trace the full call stack, read source files, "
        f"and determine the concrete optimization strategy.\n\n"

        f"═══ PHASE 1: FULL CALL-STACK TRIAGE (for each kernel with GPU% >= 1) ═══\n\n"
        f"For EACH kernel, trace the COMPLETE dispatch chain by reading actual source files:\n\n"
        f"  a) PYTHON ENTRY POINT: Find the Python function that calls this kernel.\n"
        f"     - Search the framework source for the kernel name (grep -rn in framework dirs)\n"
        f"     - Discover framework source: python3 -c \"import <framework>; print(<framework>.__path__[0])\"\n"
        f"     - Also check $BASE_DIR/scripts/ for the framework install location\n"
        f"  b) DISPATCH CHAIN: Follow the call from Python through any C++/pybind layer down "
        f"to the actual GPU kernel launch. Record EVERY intermediate function.\n"
        f"  c) SOURCE FILE: `cat` the actual kernel source file. Read it completely.\n"
        f"  d) VARIANT INVENTORY: Check if multiple implementations exist for this operation:\n"
        f"     - Triton JIT variant vs CK (Composable Kernel) variant vs aiter variant\n"
        f"     - Check config files and env vars that control dispatch\n"
        f"  e) ACTIVE vs OPTIMAL PATH: Determine which variant is currently dispatching "
        f"and whether there is a faster inactive variant.\n"
        f"  f) SHAPES: Extract the actual GEMM/operation shapes this kernel runs with:\n"
        f"     - From profiler trace or config: M, N, K dimensions\n"
        f"     - From model config: hidden_size, num_heads, head_dim, intermediate_size, etc.\n"
        f"     - Compute concrete shapes: e.g., batch × seq_len × hidden → M=batch*seq, N=hidden, K=head_dim\n\n"

        f"═══ PHASE 2: OPTIMIZATION REASONING (per kernel) ═══\n\n"
        f"For each kernel, reason deeply about optimization potential:\n\n"
        f"  a) BOTTLENECK ANALYSIS:\n"
        f"     - Is this kernel compute-bound or memory-bound for these shapes?\n"
        f"     - What is the arithmetic intensity? (FLOPs / bytes_accessed)\n"
        f"     - What is the theoretical roofline speedup potential?\n"
        f"  b) REGISTER PRESSURE: For Triton kernels, estimate VGPR usage:\n"
        f"     - Count live variables in the inner loop\n"
        f"     - Check BLOCK dimensions vs available VGPRs (256 per CU on gfx950)\n"
        f"     - Target: <=64 VGPRs/thread for 4-wave occupancy\n"
        f"  c) MEMORY ACCESS PATTERNS:\n"
        f"     - Are accesses coalesced? Any bank conflicts in LDS?\n"
        f"     - Can we fuse with adjacent kernels to reduce global memory traffic?\n"
        f"     - Opportunities for async copy / software pipelining?\n"
        f"  d) INSTRUCTION SELECTION:\n"
        f"     - Is this using MFMA instructions (bf16/fp16/fp8) or scalar ops?\n"
        f"     - Can we switch to a lower-precision path (fp16→fp8) safely?\n"
        f"     - Are there CDNA4-specific instructions we're not using?\n"
        f"  e) CONFIGURATION GAPS:\n"
        f"     - Are tuning configs model-specific or generic defaults?\n"
        f"     - Missing shape entries in config files?\n"
        f"     - Sub-optimal BLOCK/WARP/STAGE parameters?\n\n"

        f"═══ PHASE 3: BACKEND ASSIGNMENT + ACTION GENERATION ═══\n\n"
        f"For each kernel, decide the concrete optimization action AND which backend(s) "
        f"should attempt it. Consider ALL available backends:\n\n"
        f"  BACKENDS:\n"
        f"  - GEAK: Has GPU access. Can compile, benchmark, iterate on real hardware. "
        f"Best for: HIP kernels, Triton rewrites needing hardware testing, "
        f"register-pressure-constrained kernels.\n"
        f"  - Codex: Fast iteration, no GPU. Best for: Triton code generation, "
        f"register-constrained rewrites, rapid prototyping.\n"
        f"  - Claude: Deep multi-turn reasoning. Best for: complex multi-file "
        f"scheduling changes, architectural rewrites, framework-level optimizations.\n"
        f"  - LLM-Proxy: Quick single-shot generation. Best for: straightforward "
        f"Triton rewrites, configuration changes, simple code transforms.\n"
        f"  - Local (self-fix): Direct code edit, no OOB. Best for: dispatch bugs, "
        f"config-only changes, trivial fixes.\n\n"

        f"  ACTION TYPES + ROUTING TABLE:\n"
        f"  - dispatch_bug detected → dispatch-fix (local), score=10\n"
        f"  - Wrong config / generic defaults → operator-tuning (local), score=5-7\n"
        f"  - Config-only change (env var, launch param) → config-only (local), score=4-6\n"
        f"  - Inactive faster variant + python dispatch → dispatch-fix (local), score=8\n"
        f"  - Inactive variant + rebuild needed → framework-rebuild (OOB: claude), score=8\n"
        f"  - Triton source, rewrite possible → oob-rewrite (OOB: geak+codex+claude+llm-proxy), score=6-9\n"
        f"  - HIP/CK source → hip-kernel (OOB: geak+claude), score=6-8\n"
        f"  - Register-constrained Triton → oob-rewrite-register-constrained (OOB: codex+claude), score=7-9\n"
        f"  - Multi-file scheduling → framework-scheduling (OOB: claude), score=5-7\n"
        f"  - Kernel fusion opportunity → kernel-fusion (OOB: geak+claude), score=7-9\n\n"

        f"  For EACH kernel, produce:\n"
        f"  1. An action_stack entry with: id, action, target_kernel, source_file, "
        f"source_type, score (computed from expected_gain_pct, cost_minutes, accuracy_risk, "
        f"crash_risk), description, prior_status, gpu_time_pct, strategy, "
        f"dispatch_analysis (dict), trace_shapes (list)\n"
        f"  2. If action is OOB, ALSO a work_queue entry with: id, kernel_name, source_file, "
        f"source_type, strategy, priority, dispatch_analysis, trace_shapes, gpu_pct, constraints\n\n"

        f"═══ PHASE 4: DEPENDENCY GRAPH WALKING ═══\n\n"
        f"For EACH kernel with GPU% >= 1, walk 2 hops outward through the import graph:\n"
        f"  a) HOP 1: For each import in the kernel's Python entry point, read the imported module.\n"
        f"     - Check: is this the fastest available backend? Are there alternative implementations?\n"
        f"     - Check: are there conditional imports (if/else) where the wrong branch is active?\n"
        f"  b) HOP 2: For each module found in hop 1, check its own imports and config references.\n"
        f"     - Check: are there config files (CSV, JSON) with tuning values? Are they optimal?\n"
        f"     - Check: does the library have design-space dimensions we haven't explored?\n"
        f"       (e.g., scheduling variants, tile sizes, pipeline options, quantization modes)\n"
        f"  c) Record EVERY file you read in a visit_log (file_path: str list).\n"
        f"  d) For any alternative backend or unexplored config dimension found, include it in\n"
        f"     the output as a 'design_space_finding' with: file_path, variants_found, current_active.\n\n"

        f"═══ SCORING FORMULA ═══\n"
        f"score = (expected_gain_pct / cost_minutes) * (1 - accuracy_risk) * (1 - crash_risk) "
        f"* gap_multiplier\n"
        f"gap_multiplier = clamp(target_gap_pct / 100, 0.1, 2.0)\n"
        f"Handoff boosts: marathon-candidate +3, register-pressure-fixable +3, "
        f"shape-tuning-untested +2, oob-untested +2\n\n"

        f"═══ OUTPUT ═══\n"
        f"Write to $OUTPUT_FILE as JSON:\n"
        f"  kernel_dispatch_map: dict mapping kernel_name → full dispatch analysis\n"
        f"  action_stack: list of scored actions (SORTED by score desc)\n"
        f"  work_queue_entries: list of OOB targets for the kernel manager\n"
        f"  dispatch_bugs_found: int count\n"
        f"  untuned_shapes: list of shape strings missing config entries\n"
        f"  optimization_reasoning: dict mapping kernel_name → reasoning summary\n"
        f"{_READ_ONLY_CONSTRAINT}"
    )


# -----------------------------------------------------------------------
# Action prompts (13 action types)
# -----------------------------------------------------------------------

def prompt_execute_dispatch_fix(state_summary: str, action: dict[str, Any]) -> str:
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"DISPATCH FIX — 4-step fast path:\n"
        f"Target: {action.get('target_kernel', '?')} at {action.get('source_file', '?')}\n"
        f"Description: {action.get('description', '')}\n\n"
        f"1. Read the source file\n"
        f"2. git log -S \"<dispatch_symbol>\" -- <file>\n"
        f"3. git log --oneline -20 <file>\n"
        f"4. Restore the optimal dispatch path or make minimal fix, clear __pycache__\n\n"
        f"Do NOT restart the server or run benchmarks — the orchestrator handles that.\n"
        f"Write to $OUTPUT_FILE: status, patch_applied, files_modified, "
        f"needs_benchmark (set true if code changed), git_commits_checked\n"
    )


def prompt_execute_operator_tuning(state_summary: str, action: dict[str, Any]) -> str:
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"OPERATOR TUNING — 5 steps:\n"
        f"Target: {action.get('target_kernel', '?')}\n"
        f"Untuned shapes: {action.get('untuned_shapes', [])}\n\n"
        f"1. Inventory tune tools (aiter tune scripts, config.json files)\n"
        f"2. Extract GEMM shapes from model config\n"
        f"3. Check shape coverage — identify gaps, append to untuned_shapes\n"
        f"4. Run autotune loop for missing shapes\n"
        f"5. Validate: micro-benchmark >1.05x speedup → deploy config/CSV\n\n"
        f"Do NOT restart the server or run E2E benchmarks — the orchestrator handles that.\n"
        f"Write to $OUTPUT_FILE: status, shapes_tuned, micro_speedup, "
        f"needs_benchmark (set true), accuracy_risk\n"
    )


def prompt_execute_framework_rebuild(state_summary: str, action: dict[str, Any]) -> str:
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"FRAMEWORK REBUILD — 4 steps:\n"
        f"Target library: {action.get('target_lib', '?')}\n"
        f"Rebuild command: {action.get('rebuild_command', '?')}\n\n"
        f"1. Backup/stash current state: git stash or cp backup\n"
        f"2. Rebuild: pip install -e / cmake / make\n"
        f"3. Verify: import check + dispatch verification + smoke test\n"
        f"4. If verification fails: rollback using backup\n\n"
        f"accuracy_risk: 0.15\n"
        f"Write to $OUTPUT_FILE: status, rebuild_log, needs_benchmark, "
        f"accuracy_risk, rollback_available\n"
    )


def prompt_execute_kernel_opt(state_summary: str, action: dict[str, Any]) -> str:
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"KERNEL OPTIMIZATION (Strategies A-G):\n"
        f"Target: {action.get('target_kernel', '?')}\n"
        f"Strategy: {action.get('strategy', '?')}\n"
        f"Source: {action.get('source_file', '?')}\n\n"
        f"Apply the optimization strategy. Run micro-benchmark with:\n"
        f"- Warmup: 20 iters, Timed: 200 iters\n"
        f"- allclose atol=1e-2, rtol=1e-2\n"
        f"- Reject if any shape speedup < 0.95\n"
        f"- Target: >=1.05x average speedup\n\n"
        f"Write to $OUTPUT_FILE: status, micro_speedup, shapes_tested, "
        f"needs_benchmark, accuracy_risk\n"
    )


def prompt_execute_comm_optimization(state_summary: str, action: dict[str, Any]) -> str:
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"COMM OPTIMIZATION — 6 steps:\n"
        f"Target: {action.get('description', 'NCCL/RCCL optimization')}\n\n"
        f"1. Comm audit: identify collective operations + % GPU time\n"
        f"2. Topology: rocm-smi --showtopoweight / nvidia-smi topo -m / ibstat\n"
        f"3. NCCL algo/proto sweep: Tree, Ring, CollnetDirect, CollnetChain, NVLS\n"
        f"4. Overlap: grep for async/overlap opportunities\n"
        f"5. Multi-node rules: NCCL_MIN_NCHANNELS=4, MAX=16, BUFFSIZE=8388608\n"
        f"6. Apply best env vars and measure\n\n"
        f"accuracy_risk: 0.05\n"
        f"Write to $OUTPUT_FILE: status, env_vars_set, comm_improvement_pct, "
        f"needs_benchmark, accuracy_risk\n"
    )


def prompt_execute_compiler_tuning(state_summary: str, action: dict[str, Any]) -> str:
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"COMPILER TUNING (Triton/Inductor codegen):\n"
        f"Target: {action.get('target_kernel', '?')}\n\n"
        f"Tune Triton/Inductor compilation flags and code generation:\n"
        f"1. Analyze current codegen output\n"
        f"2. Try num_warps, num_stages, BLOCK_* tuning\n"
        f"3. Try Inductor config flags if applicable\n"
        f"4. Benchmark each variant\n\n"
        f"Write to $OUTPUT_FILE: status, best_config, micro_speedup, "
        f"needs_benchmark\n"
    )


def prompt_execute_action(state_summary: str, action: dict[str, Any]) -> str:
    """Generic action execution for types without a dedicated prompt."""
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"Execute action: {action.get('action', '?')}\n"
        f"Description: {action.get('description', '')}\n"
        f"Target: {action.get('target_kernel', action.get('target', ''))}\n\n"
        f"Apply the optimization (code patches, config changes, CSV generation, etc.).\n"
        f"Do NOT start/restart/kill the inference server.\n"
        f"Do NOT run benchmark_serving.py or any E2E throughput benchmark.\n"
        f"The orchestrator will restart the server and run benchmarks after you finish.\n\n"
        f"Write to $OUTPUT_FILE: status, needs_benchmark (set true if changes were made), "
        f"accuracy_risk, sub_actions (list of new actions to push)\n"
    )


# -----------------------------------------------------------------------
# Accuracy gate
# -----------------------------------------------------------------------

def prompt_accuracy_gate(state_summary: str, action: dict, result: dict) -> str:
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"ACCURACY GATE — verify model accuracy after risky change.\n"
        f"Action: {action.get('action', '?')}\n"
        f"accuracy_risk: {result.get('accuracy_risk', 0)}\n"
        f"Threshold: {0.01} (1% deviation allowed)\n\n"
        f"1. Run a quick eval (GSM8K subset or completions check)\n"
        f"2. Compare against baseline_accuracy\n"
        f"3. If deviation > threshold → REVERT\n\n"
        f"Write to $OUTPUT_FILE: passed (bool), accuracy, deviation, "
        f"revert_needed (bool)\n"
        f"{_READ_ONLY_CONSTRAINT}"
    )


# -----------------------------------------------------------------------
# Re-score
# -----------------------------------------------------------------------

def prompt_rescore(state_summary: str, action_stack: list[dict]) -> str:
    stack_str = "\n".join(
        f"  [{a.get('score', 0):.1f}] {a.get('id', '?')}: {a.get('action', '?')}"
        for a in sorted(action_stack, key=lambda x: -x.get("score", 0))
    )
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"RE-SCORE the action stack based on current state.\n"
        f"Current stack:\n{stack_str}\n\n"
        f"Update rules:\n"
        f"1. Dispatch bug found → boost deep-kernel-opt to 10\n"
        f"2. Rebuild success → remaining rebuilds ×1.5\n"
        f"3. Comm gain >2% → boost comm actions\n"
        f"4. Backend discard → reduce that backend's scores ×0.8\n"
        f"5. Never zero ALL kernel-opt from one backend\n"
        f"6. Add new actions if analysis reveals opportunities\n\n"
        f"Write to $OUTPUT_FILE: rescored_stack (list of {{id, score}} dicts)\n"
        f"{_READ_ONLY_CONSTRAINT}"
    )


# -----------------------------------------------------------------------
# Dream
# -----------------------------------------------------------------------

def prompt_dream(state_summary: str, completed_since_last: list[dict],
                 strategies_tested: list[str]) -> str:
    completed_str = "\n".join(
        f"  - {c.get('id', '?')}: {c.get('action', '?')} → {c.get('result', {}).get('status', '?')} "
        f"(gain={c.get('result', {}).get('gain_pct', 0):.1f}%)"
        for c in completed_since_last[-20:]
    )
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"DREAM — consolidate learnings, re-score, AND generate new ideas.\n\n"
        f"This is your chance to think strategically about the entire optimization campaign.\n"
        f"The harness must keep running productively for hours.\n\n"

        f"Completed actions since last dream:\n{completed_str}\n"
        f"Strategies tested: {strategies_tested}\n"
        f"All strategy categories: A(dispatch-fix), B(operator-tuning), C(deep-kernel-opt), "
        f"D(framework-rebuild), E(comm-optimization), F(compiler-tuning), G(kernel-fusion)\n\n"

        f"═══ PHASE 1: ORIENT ═══\n"
        f"- What has worked? (success actions with gain > 0)\n"
        f"- What has failed? (repeated failure patterns, blocked kernels)\n"
        f"- Which strategy categories remain untested?\n"
        f"- Which kernels have >1% GPU time and haven't been attempted?\n\n"

        f"═══ PHASE 2: CONSOLIDATE ═══\n"
        f"- Identify patterns: do certain backends work better for certain kernel types?\n"
        f"- Are there recurring failure modes we can work around?\n"
        f"- What is the remaining optimization headroom?\n\n"

        f"═══ PHASE 3: RE-SCORE ═══\n"
        f"Re-score existing stack actions:\n"
        f"- Untested strategy ×1.5\n"
        f"- framework-rebuild + dispatch_bugs_found > 0 → ×2.0\n"
        f"- operator-tuning + untuned_shapes remaining → ×1.5\n"
        f"- Same backend failed 3+ times → ×0.5 for that backend's actions\n"
        f"- KB-consistent-failure → ×0.3\n"
        f"- Playbook-recommends → ×1.3\n\n"

        f"═══ PHASE 4: DESIGN SPACE ENUMERATION ═══\n"
        f"For each library in the stack (aiter, CK, triton, sglang):\n"
        f"- What configurable dimensions exist that we haven't explored?\n"
        f"- What enum types, variant selections, pipeline options are available?\n"
        f"- Read library source to enumerate them if needed.\n"
        f"- Output: design_space_findings (list of unexplored dimensions)\n\n"

        f"═══ PHASE 5: TRANSFER HYPOTHESIS GENERATION ═══\n"
        f"Read the insight bus (insights.jsonl in session_dir/kernel_manager/).\n"
        f"- For each pattern-discovery insight, generate actions applying the same pattern\n"
        f"  to all similar call sites.\n"
        f"- For each successful optimization, ask: where else does the pre-optimization\n"
        f"  pattern appear in the codebase?\n"
        f"- Output: transfer_actions (list of scored actions from cross-target transfer)\n\n"

        f"═══ PHASE 6: NEGATIVE RESULT MINING ═══\n"
        f"Read the failure journal from state.\n"
        f"- For each failed experiment, extract: what property of the target made it fail?\n"
        f"- Do other targets share that property?\n"
        f"- What would need to be different for the approach to work?\n"
        f"- Generate hypotheses that could turn failures into wins.\n"
        f"- Generate defensive_rules: trigger conditions + checks that would have prevented\n"
        f"  each failure. Format: {{trigger, check, rationale}}\n"
        f"- Output: failure_patterns, defensive_rules, hypotheses\n\n"

        f"═══ PHASE 7: GENERATE NEW ACTIONS ═══\n"
        f"CRITICALLY: Generate at least 3 NEW action ideas not currently on the stack.\n"
        f"Think about:\n"
        f"- Kernels we failed on but with a different approach\n"
        f"- Config sweeps (server params, env vars, NCCL settings)\n"
        f"- Kernel fusion opportunities between adjacent kernels\n"
        f"- Communication overlap with compute\n"
        f"- Lower-GPU% kernels that aggregate to significant time\n"
        f"- Precision changes (fp16→fp8 where safe)\n\n"

        f"═══ PHASE 8: KB CONTRIBUTION ═══\n"
        f"Write >=1 knowledge base entry per dream session.\n\n"

        f"Write to $OUTPUT_FILE:\n"
        f"  rescored_stack: list of {{id, score}} for existing actions\n"
        f"  new_actions: list of NEW scored actions to push (MIN 3)\n"
        f"  transfer_actions: list of actions from cross-target pattern transfer\n"
        f"  hypotheses: list of {{hypothesis, test_method, cost_minutes, confidence}}\n"
        f"  design_space_findings: list of unexplored design dimensions\n"
        f"  failure_patterns: list of meta-patterns from failure analysis\n"
        f"  defensive_rules: list of {{trigger, check, rationale}} from negative results\n"
        f"  kb_entries: list of knowledge base entries\n"
        f"  dream_summary: string summary of strategic thinking\n"
        f"{_READ_ONLY_CONSTRAINT}"
    )


# -----------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------

def prompt_sweep(state_summary: str, server_config: dict) -> str:
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"SWEEP — extended parameter sweep on final optimized config.\n\n"
        f"Server config: {server_config}\n\n"
        f"Run sweep across:\n"
        f"- Concurrency: 4, 8, 16, 32, 64, 128\n"
        f"- ISL/OSL: 1024:1024, 8192:1024, 1024:8192\n\n"
        f"Write to $OUTPUT_FILE: results_tsv, pareto_points, best_config\n"
        f"{_READ_ONLY_CONSTRAINT}"
    )


# -----------------------------------------------------------------------
# Report
# -----------------------------------------------------------------------

def prompt_report(state_summary: str, completed_actions: list[dict]) -> str:
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"REPORT — write final optimization report.\n\n"
        f"Total completed actions: {len(completed_actions)}\n\n"
        f"Sections: Executive Summary, Kernel Bottleneck Analysis, "
        f"Backend Exploration, Server Parameter Tuning, "
        f"GEAK Kernel Optimization, Target Comparison, "
        f"Parameter Sweep, Recommendations\n\n"
        f"Write report to $RESULT_DIR/optimization_report.md\n"
        f"Write to $OUTPUT_FILE: report_path, kb_entries\n"
        f"{_READ_ONLY_CONSTRAINT}"
    )


# -----------------------------------------------------------------------
# Recover
# -----------------------------------------------------------------------

def prompt_recover(state_summary: str, crash_type: str, crash_log: str) -> str:
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"RECOVER from crash: {crash_type}\n"
        f"Crash log:\n```\n{crash_log[:2000]}\n```\n\n"
        f"Recovery chains:\n"
        f"- oom: reduce --mem-fraction-static by 0.05 → reduce cuda-graph-max-bs /2 → restart\n"
        f"- cuda_graph: reduce cuda-graph-max-bs /2 → disable → restart\n"
        f"- patch_crash: rollback last patch → clear caches → restart\n"
        f"- nccl_timeout: NCCL_TIMEOUT=1800 → restart (wait 30s)\n"
        f"- unknown: restart → checkpoint and reload\n\n"
        f"MAX_RECOVERY_ATTEMPTS=3, BACKOFF_MULTIPLIER=2\n"
        f"Post-recovery: checkpoint, crashing action score ×0.3\n\n"
        f"Write to $OUTPUT_FILE: recovered (bool), chain_used, "
        f"actions_taken, server_healthy\n"
    )


# -----------------------------------------------------------------------
# Re-explore (plateau breaker)
# -----------------------------------------------------------------------

def prompt_re_explore(state_summary: str, loop_signatures: list[str],
                      strategies_tested: list[str], tier_breakdown: dict) -> str:
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"RE-EXPLORE — You MUST generate at least 5 new optimization actions.\n\n"
        f"The action stack is empty or stalled. The harness needs to run for hours.\n"
        f"Your job is to find NEW optimization opportunities — think creatively.\n\n"
        f"Loop signatures (last 8): {loop_signatures[-8:]}\n"
        f"Strategies already tested: {strategies_tested}\n"
        f"Tier breakdown: {tier_breakdown}\n\n"

        f"═══ STEP 1: DIAGNOSE WHY WE STALLED ═══\n"
        f"- Read the completed actions from state — what worked, what failed?\n"
        f"- Which kernels/strategies have NOT been tried yet?\n"
        f"- Are there any kernels with >1% GPU time that we haven't optimized?\n"
        f"- Did previous failures have fixable root causes?\n\n"

        f"═══ STEP 2: GENERATE NOVEL ACTIONS (minimum 5) ═══\n"
        f"Think about ALL of these idea sources:\n\n"
        f"A) UNTRIED KERNELS: Re-run profiling or check kernel_candidates for kernels\n"
        f"   not yet attempted. Even 1-3% GPU kernels add up.\n"
        f"B) DIFFERENT STRATEGY for same kernel: If oob-rewrite failed, try\n"
        f"   register-constrained rewrite, kernel-fusion, or framework-scheduling.\n"
        f"C) CONFIGURATION SWEEP: Server params we haven't tuned:\n"
        f"   - --chunked-prefill-size (1024, 2048, 4096, 8192)\n"
        f"   - --max-running-requests (256, 512, 1024)\n"
        f"   - --mem-fraction-static (0.85, 0.88, 0.90, 0.92)\n"
        f"   - --cuda-graph-max-bs (1, 2, 4, 8, 16, 32)\n"
        f"   - NCCL env vars: NCCL_ALGO, NCCL_PROTO, NCCL_MIN_NCHANNELS\n"
        f"D) KERNEL FUSION: Look at consecutive kernels in the profile that\n"
        f"   could be fused to eliminate memory round-trips.\n"
        f"E) DISPATCH PATH OPTIMIZATION: Check if any kernels are going through\n"
        f"   a suboptimal dispatch path (e.g., generic fallback instead of\n"
        f"   model-specific path).\n"
        f"F) COMMUNICATION OPTIMIZATION: Overlap compute with all-reduce,\n"
        f"   tune NCCL algorithms/protocols for the specific topology.\n"
        f"G) SCHEDULING: Batch size effects, pipeline parallelism, async overlap.\n"
        f"H) PRECISION: fp16→fp8 for kernels where accuracy allows it.\n"
        f"I) CACHE TUNING: Triton cache configs, torch.compile settings.\n"
        f"J) RETRY FAILED WITH CONSTRAINTS: Take the top 3 failed kernels,\n"
        f"   add explicit constraints from their error messages, and resubmit\n"
        f"   to a different backend.\n\n"

        f"═══ STEP 3: SCORE AND OUTPUT ═══\n"
        f"For each action, include: id, action (type), target_kernel, score, "
        f"description, strategy, source_file (if known), gpu_time_pct (if known).\n\n"
        f"Score formula: (expected_gain / cost_minutes) * (1-risk) * gap_mult\n"
        f"Set scores between 3-9. Prefer variety over one strategy.\n\n"
        f"Write to $OUTPUT_FILE: novel_actions (list, MIN 5), diagnosis (string)\n"
        f"{_READ_ONLY_CONSTRAINT}"
    )


def prompt_exploratory_probe(state_summary: str, visit_map: dict,
                             hot_path_files: list[str]) -> str:
    unvisited = [f for f in hot_path_files if visit_map.get(f, 0) == 0]
    low_visit = [f for f in hot_path_files if 0 < visit_map.get(f, 0) <= 1]
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"EXPLORATORY PROBE — undirected code reading for discovery.\n\n"
        f"You are NOT optimizing a specific kernel. You are EXPLORING the codebase\n"
        f"to find optimization opportunities that profiling alone cannot reveal.\n\n"

        f"Unvisited hot-path files ({len(unvisited)}):\n"
        + "\n".join(f"  - {f}" for f in unvisited[:15]) + "\n\n"
        f"Low-visit files ({len(low_visit)}):\n"
        + "\n".join(f"  - {f} (visited {visit_map.get(f, 0)}x)" for f in low_visit[:10]) + "\n\n"

        f"═══ STEP 1: PICK AND READ ═══\n"
        f"Pick the 3-5 most promising unvisited files and read them END-TO-END.\n"
        f"For each file:\n"
        f"  a) Follow every `from X import Y` — is Y the fastest available backend?\n"
        f"  b) Check for conditional imports / if-else dispatch — which branch is active?\n"
        f"  c) Check for config file references — are the values tuned for this model?\n"
        f"  d) Check for hardcoded defaults that could be optimized.\n"
        f"  e) Check what library APIs are available but NOT being used.\n\n"

        f"═══ STEP 2: ENUMERATE DESIGN SPACES ═══\n"
        f"For each 3rd-party library used by these files:\n"
        f"  a) What operations does it support? What variants/backends exist?\n"
        f"  b) What config knobs exist (env vars, constructor args, CSV configs)?\n"
        f"  c) Are there scheduling options, tile sizes, pipeline variants we're not using?\n\n"

        f"═══ STEP 3: GENERATE OBSERVATIONS ═══\n"
        f"For each interesting finding, produce an observation:\n"
        f"  {{observation: str, hypothesis: str, confidence: high|medium|low,\n"
        f"   files_read: list[str], potential_impact: str}}\n\n"

        f"Write to $OUTPUT_FILE:\n"
        f"  observations: list of observations (MIN 5)\n"
        f"  design_spaces: list of unexplored design dimensions\n"
        f"  visit_log: list of file paths you read\n"
        f"  insights: list of insight objects to write to insight bus\n"
        f"{_READ_ONLY_CONSTRAINT}"
    )


def prompt_hypothesis_ab_benchmark(target: dict) -> str:
    hypothesis = target.get("hypothesis", "")
    variant_a = target.get("variant_a", {})
    variant_b = target.get("variant_b", {})
    param_space = target.get("parameter_space", {})
    return (
        f"HYPOTHESIS A/B BENCHMARK\n\n"
        f"Hypothesis: {hypothesis}\n"
        f"Variant A: {variant_a.get('description', 'baseline')}\n"
        f"Variant B: {variant_b.get('description', 'candidate')}\n"
        f"Parameter space: {json.dumps(param_space, indent=2)}\n\n"

        f"Write a SELF-CONTAINED Python benchmark script that:\n"
        f"1. Imports both variant A and variant B.\n"
        f"2. For each combination in the parameter space:\n"
        f"   a. Create random input tensors of the appropriate shape.\n"
        f"   b. Run variant A with warmup (25 iters) + timed (200 iters).\n"
        f"   c. Run variant B with warmup (25 iters) + timed (200 iters).\n"
        f"   d. Record: shape, variant_a_us, variant_b_us, speedup (a/b).\n"
        f"3. Analyze results:\n"
        f"   a. For which shapes is variant B faster? By how much?\n"
        f"   b. Is there a crossover point (parameter value where the winner switches)?\n"
        f"   c. What is the overall geometric mean speedup?\n"
        f"4. Output JSON to $OUTPUT_FILE:\n"
        f"   per_shape: list of {{shape, variant_a_us, variant_b_us, speedup}}\n"
        f"   summary: {{a_wins, b_wins, crossover_point, geomean_speedup,\n"
        f"             dispatch_recommendation: str}}\n"
        f"   script_path: path to the benchmark script written\n\n"
        f"The dispatch_recommendation should be a concrete rule like:\n"
        f"  'Use variant B when M >= 32, variant A when M < 32'\n"
    )


# -----------------------------------------------------------------------
# Merge-op helpers
# -----------------------------------------------------------------------

def prompt_apply_instruction(instruction: str) -> str:
    return f"Execute this patch instruction:\n{instruction}\nReport success or failure."


def prompt_rebuild(rebuild_command: str) -> str:
    return f"Run rebuild command:\n```bash\n{rebuild_command}\n```\nReport success or failure with output."


def prompt_shell_command(command: str) -> str:
    return f"Run shell command:\n```bash\n{command}\n```\nReport output."


def prompt_verify(verification_command: str) -> str:
    return f"Run verification:\n```bash\n{verification_command}\n```\nReport if verification passed."


# -----------------------------------------------------------------------
# Benchmark
# -----------------------------------------------------------------------

def prompt_benchmark(framework: str = "sglang") -> str:
    return (
        f"{_sys.SYSTEM_PROMPT}\n"
        f"E2E THROUGHPUT BENCHMARK — measure tok/s/GPU.\n\n"
        f"Steps:\n"
        f"1. cd $INFERENCEX_PATH\n"
        f"2. Run the standard benchmark:\n"
        f"   python benchmarks/benchmark_serving.py --backend {framework} "
        f"--model $(pgrep -a python | grep -oP '(?<=--model-path )\\S+' | head -1) "
        f"--num-prompts 500 --request-rate inf --output-len 1024 --input-len 1024\n"
        f"3. Parse output for output_throughput (tok/s total) and latency\n"
        f"4. Compute tput_per_gpu = output_throughput / GPU_COUNT\n\n"
        f"Write to $OUTPUT_FILE: tput_per_gpu, latency_ms, output_throughput, "
        f"num_prompts, request_rate\n"
    )


# -----------------------------------------------------------------------
# Config-only action execution
# -----------------------------------------------------------------------

def prompt_diagnose_failure(state_summary: str, action: dict[str, Any],
                            result: dict[str, Any]) -> str:
    """Inline failure diagnosis — Orchestrator calls this on its own action failures."""
    import json as _json
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"DIAGNOSE FAILURE — analyze root cause and determine retry strategy.\n\n"
        f"The following Orchestrator action failed. This is NOT a kernel rewrite failure.\n"
        f"This is a code-level / config-level action that the Orchestrator executed directly.\n\n"
        f"Failed action:\n```json\n{_json.dumps(action, indent=2, default=str)}\n```\n\n"
        f"Failure result:\n```json\n{_json.dumps(result, indent=2, default=str)}\n```\n\n"

        f"═══ STEP 1: READ THE ERROR ═══\n"
        f"Read the error output carefully. If there's a crash log, analyze it.\n"
        f"If files were modified, read them to understand the failure context.\n\n"

        f"═══ STEP 2: CLASSIFY ROOT CAUSE ═══\n"
        f"Categorize as one of:\n"
        f"- config_error: wrong config value, invalid parameter combination\n"
        f"- import_error: missing module, wrong path, version mismatch\n"
        f"- compilation_error: build failure, syntax error, API mismatch\n"
        f"- memory_fault: segfault, OOM, invalid memory access\n"
        f"- register_spill: kernel needs register-constrained rewrite (escalate to KM)\n"
        f"- hardware: ECC error, GPU hang, device unavailable (not retryable)\n"
        f"- unknown: can't determine from available info\n\n"

        f"═══ STEP 3: DETERMINE RETRY STRATEGY ═══\n"
        f"If retryable at code level (config, import, compilation):\n"
        f"  - Describe the specific fix to apply\n"
        f"  - Produce a modified action with the fix\n"
        f"If needs kernel rewrite (register_spill, memory_fault in kernel):\n"
        f"  - Mark for escalation to Kernel Manager\n"
        f"If hardware or truly unrecoverable:\n"
        f"  - Mark as not retryable\n\n"

        f"Write to $OUTPUT_FILE:\n"
        f"  root_cause: string (concise description)\n"
        f"  category: config_error|import_error|compilation_error|memory_fault|"
        f"register_spill|hardware|unknown\n"
        f"  retryable: bool\n"
        f"  escalate_to_km: bool (true only if kernel source rewrite needed)\n"
        f"  confidence: high|medium|low\n"
        f"  retry_action: dict (modified action with fix) or null\n"
        f"  rca_constraints: dict (constraint, avoid list, compiler_flags) or null\n"
        f"  fix_description: string (what to change for retry)\n"
        f"  insights: string (what we learned)\n"
        f"{_READ_ONLY_CONSTRAINT}"
    )


def prompt_execute_config_only(state_summary: str, action: dict[str, Any]) -> str:
    return (
        f"{_sys.SYSTEM_PROMPT}\n{_state_ctx(state_summary)}"
        f"CONFIG-ONLY OPTIMIZATION — no code changes, only server/env config.\n"
        f"Target: {action.get('target_kernel', action.get('description', '?'))}\n"
        f"Config change: {action.get('config_change', action.get('description', ''))}\n\n"
        f"Steps:\n"
        f"1. Identify the relevant config file, env variable, or launch script parameter\n"
        f"2. Apply the config change (modify JSON, CSV, env export in launch script)\n"
        f"3. Report what was changed\n\n"
        f"IMPORTANT: Do NOT modify source code files. Only modify config files, "
        f"environment variables, CSV tuning configs, or server launch parameters.\n"
        f"Do NOT restart the inference server — the orchestrator handles server lifecycle.\n"
        f"Do NOT run benchmark_serving.py or any E2E benchmark.\n\n"
        f"Write to $OUTPUT_FILE: status, config_changed, needs_benchmark (set true), "
        f"accuracy_risk (should be 0.01), crash_risk\n"
    )
