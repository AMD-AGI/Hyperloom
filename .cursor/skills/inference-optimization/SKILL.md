---
name: inference-optimization
description: |
  Autonomous DFS-guided inference optimization for LLM serving on AMD MI355X GPUs.
  Uses heuristic-scored depth-first search to systematically explore optimization actions
  (backend switches, server params, kernel optimization, target comparison) and maximize
  throughput per GPU (tok/s/GPU) with accuracy as a hard constraint.
globs:
  - "**/inference*optim*"
  - "**/benchmark*"
  - "**/sglang*"
  - "**/vllm*"
---

# Inference Optimization — DFS Orchestrator

## Overview

This skill optimizes LLM inference serving throughput (tok/s/GPU) on AMD MI355X GPUs.
It runs a **depth-first search** over optimization actions, guided by a heuristic scoring
function. The search is fully autonomous — no human prompting required.

**Primary objective:** maximize `tok/s/GPU`
**Hard constraint:** accuracy (numerical correctness) must not degrade
**Optional target:** if an external baseline is provided (e.g. NVIDIA B200), the target
gap acts as an urgency multiplier on all action scores.

## Execution Mode

This skill supports two execution modes:

| Aspect | Local Mode | Claw Mode |
|--------|-----------|-----------|
| Client | Cursor IDE | Claw |
| Inference serving | Local process | RayJob on SaFE (multi-node TP) |
| Command execution | Direct shell | `exec_on_gpu` via `ray_submit.py` |
| Storage (`WORKSPACE_ROOT`) | `/workspace/inference-optimization` | `/shared_nfs/inference-optimization` |
| Multi-node | Not supported | Supported (cross-node TP) |
| SaFE MCP | Not used for inference | Used for RayJob lifecycle |
| GEAK MCP | Same | Same (remote SaFE PyTorchJob) |
| TraceLens MCP | Same | Same (traces on shared NFS) |

**Auto-detection:**
- `GEAK_LOCAL=true` → local mode (default for Cursor IDE)
- Claw client context or user specifies `Mode: claw` → claw mode

See [`modes/LOCAL.md`](modes/LOCAL.md) and [`modes/CLAW.md`](modes/CLAW.md) for mode-specific details.

## Iron Rules (non-negotiable)

These rules apply to ALL modes. Violating any invalidates the optimization run.

### IR-1: Submit ALL GEAK candidates in parallel

The kernel-opt action MUST submit `GEAK_TOP_CANDIDATES` (default 5) tasks **simultaneously** via parallel `geak_create_task` + `geak_submit_task` calls. Submitting only 1 task when multiple candidates exist = violation.

### IR-2: NEVER modify kernel source before GEAK submission

Submit the kernel source **exactly as extracted**. Do NOT strip decorators, change strides, replace `@triton_heuristics` with `@triton.jit`, or make any "cleanup" edits. GEAK's agent handles kernel adaptation internally.

### IR-3: Integration (Phase 8) is MANDATORY

After GEAK returns optimized kernels, you MUST execute the integrate action (patch → re-baseline → decide). Skipping means GEAK results are never validated end-to-end. Re-baseline uses `run_baseline.sh` — there is no `run_benchmark.sh`. See `actions/integrate.md` for details.

### IR-4: Always kill_server + check_gpu_memory before server launch

Every server launch must be preceded by killing any existing server process and verifying GPU memory is free.

### IR-5: Safe process management

**NEVER use `pkill -f sglang`** — it kills Ray workers in claw mode. Only use:

```bash
kill $(pgrep -f 'python.*-m sglang.launch_server') 2>/dev/null
# or for vLLM:
kill $(pgrep -f 'python.*-m vllm.entrypoints') 2>/dev/null
```

Wait `SERVER_KILL_WAIT_S` seconds between kill and relaunch. Always `unset PROFILE SGLANG_TORCH_PROFILER_DIR` after profiling.

### IR-6: [CLAW] Use `exec_on_gpu` for ALL GPU-side commands

**Claw mode only.** After `source scripts/executor.sh`, ALL commands that run on the Ray cluster MUST go through `exec_on_gpu()` or `exec_on_gpu_bg()`. **NEVER** manually call `ray_submit.py` directly.

```bash
# CORRECT
exec_on_gpu "export MODEL='$MODEL' ... && bash $SCRIPTS_DIR/run_baseline.sh"

# WRONG — manual ray_submit.py
python3 scripts/ray_submit.py --ray-address ... --command "..."
```

### IR-7: [CLAW] Main inference workload MUST use `kind: "RayJob"`

**Claw mode only.** The persistent inference cluster **MUST** be `kind: "RayJob"`. The only places where `PyTorchJob` is acceptable:
1. GEAK kernel optimization (GEAK MCP creates PyTorchJob internally)
2. Parallel sweep workloads (short-lived, NOT the main inference cluster)

### IR-8: Use `patch_inductor.py --target-file` for Inductor patching

Always use `scripts/patch_inductor.py` with `--target-file`. The `--cache-dir` option has been removed.

## GEAK & Tooling Constants

All values below are the **single source of truth**. All actions reference these by name.

| Constant | Value | Description |
|----------|-------|-------------|
| `GEAK_STEP_LIMIT` | 100 | Max agent steps per GEAK task |
| `GEAK_WORKSPACE` | `control-plane-moe` | GEAK workspace (user can override) |
| `GEAK_MAX_RETRIES` | 3 | Max submission retries per kernel |
| `GEAK_MAX_SUBMISSIONS` | 15 | Total GEAK submissions budget per run |
| `GEAK_TOP_CANDIDATES` | 5 | Number of top kernel candidates to submit |
| `GEAK_CONSECUTIVE_DISCARDS` | 5 | Stop after this many consecutive discards |
| `GEAK_WALL_CLOCK_MIN` | 120 | Max wall-clock minutes for kernel-opt action |
| `GEAK_POLL_INTERVAL_S` | 60 | Seconds between GEAK task status polls |
| `GEAK_POLL_TIMEOUT_MIN` | 15 | Max minutes to poll a single GEAK task |
| `MIN_GPU_PCT` | 3 | Minimum GPU time % to consider a kernel as GEAK candidate |
| `SERVER_KILL_WAIT_S` | 10 | Seconds to wait between server kill and relaunch |
| `FILTERED_TRACE_NAME` | `filtered-TP-0.trace.json.gz` | Preferred trace file for TraceLens analysis |
| `GEAK_IMAGE_SGLANG` | `harbor.oci-slc.primus-safe.amd.com/proxy/lmsysorg/sglang:v0.5.9-rocm700-mi35x` | Default GEAK image for SGLang (local mode) |
| `GEAK_IMAGE_VLLM` | `harbor.oci-slc.primus-safe.amd.com/proxy/vllm/vllm-openai-rocm:v0.17.0` | Default GEAK image for vLLM |
| `GEAK_IMAGE_SGLANG_RAY` | `harbor.oci-slc.primus-safe.amd.com/custom/lmsysorg/sglang:202603270958` | Claw mode: SGLang image with Ray 2.44.1 fix |
| `RAY_CLIENT_PORT` | 10001 | Claw mode: Ray Client port on RayJob head node |

### GEAK Image Selection — ALL kernels MUST use framework image

**ALWAYS pass a framework image to GEAK, regardless of kernel type.** For kernels whose source exists in the image (e.g., `/sgl-workspace/aiter/`), the GEAK pod uses the same image to access these paths. For runtime-generated kernels (e.g., `/tmp/torchinductor_root/` from `torch.compile`), these paths do NOT exist in the image — do NOT include `kernel_url`/`kernel_repo` in the prompt; copy files to shared NFS or rely on `files[].content` only.

| Condition | Local Mode | Claw Mode |
|-----------|-----------|-----------|
| User specified a custom image | Use user-specified | Use user-specified |
| `FRAMEWORK=sglang` (default) | `GEAK_IMAGE_SGLANG` | `GEAK_IMAGE_SGLANG_RAY` |
| `FRAMEWORK=vllm` | `GEAK_IMAGE_VLLM` | `GEAK_IMAGE_VLLM` |

**Claw mode MUST use `GEAK_IMAGE_SGLANG_RAY`** (not `GEAK_IMAGE_SGLANG`) for SGLang — the Ray-patched image matches the RayJob serving environment.

### [CLAW] Image Build (Dockerfile)

The custom SGLang image (`GEAK_IMAGE_SGLANG_RAY`) is based on upstream SGLang with Ray compatibility fixes:

```dockerfile
FROM harbor.oci-slc.primus-safe.amd.com/proxy/lmsysorg/sglang:v0.5.9-rocm700-mi35x
RUN python -m pip install ray[default]==2.44.1 click==8.1.7
```

**Why:** Upstream SGLang image ships a Ray version with a `copy.deepcopy` / `Sentinel` enum crash bug that breaks RayJob's `ray start` and `ray-job-submitter`. `click==8.1.7` pins click to avoid API incompatibilities with Ray CLI.

### [CLAW] RayJob Architecture

```
Claw Client --> Skill (this file)
                 |-> SaFE MCP (workload_create kind="RayJob")
                 |     |-> Creates Ray Cluster: head (1 pod) + workers (N pods)
                 |     |-> Exposes Ray Client port (10001)
                 |     \-> env.RAY_JOB_ENTRYPOINT = "tail -f /dev/null" (keeps cluster alive)
                 |
                 |-> exec_on_gpu (scripts/executor.sh)
                 |     \-> ray_submit.py -> ray.init("ray://<head>:10001") -> run on cluster
                 |
                 |-> GEAK MCP (remote, unchanged)
                 |     \-> SaFE API -> PyTorchJob -> kernel optimization
                 |
                 |-> TraceLens MCP (remote, unchanged)
                 |     \-> Reads traces from shared NFS
                 |
                 \-> Benchmark results on shared NFS
```

## Architecture

```
SKILL.md (this file)          — DFS orchestrator: loop, heuristic, dispatch
actions/*.md                   — Self-contained action modules (11 actions)
kb/                            — RAG knowledge base (JSONL + query/ingest scripts)
scripts/                       — Baseline/profiling shell scripts (run_baseline.sh is also used for re-baseline after kernel patching)
GEAK-INFERENCE-KERNEL.md       — GEAK MCP deep reference
KNOWLEDGE-BASE.md              — Legacy KB (archived, seeded into kb/entries.jsonl)
```

## DFS Search Tree

```
                        ┌──────────┐
                        │  SETUP   │
                        └────┬─────┘
                             │
                        ┌────▼─────┐
                        │ CLASSIFY │
                        └────┬─────┘
                             │
                   ┌─────────▼─────────┐
                   │  TARGET ANALYSIS   │ ← optional, if target dir provided
                   └─────────┬─────────┘
                             │
                        ┌────▼─────┐
                        │ BASELINE │
                        └────┬─────┘
                             │
                        ┌────▼─────┐
                        │ PROFILE  │
                        └────┬─────┘
                             │
              ┌──────────────▼──────────────┐
              │      HEURISTIC SCORING      │
              │   score each candidate action│
              └──────────────┬──────────────┘
                             │
              ┌──────────────▼──────────────┐
              │   PICK HIGHEST-SCORED ACTION │
              │                              │
              │  ┌─────────┐ ┌──────────┐   │
              │  │BACKENDS │ │ PARAMS   │   │
              │  └────┬────┘ └────┬─────┘   │
              │       │           │          │
              │  ┌────▼────┐ ┌───▼──────┐   │
              │  │KERNEL   │ │ SWEEP    │   │
              │  │  OPT    │ │          │   │
              │  └────┬────┘ └──────────┘   │
              │       │                      │
              │  ┌────▼────┐                 │
              │  │INTEGRATE│                 │
              │  └─────────┘                 │
              └──────────────┬──────────────┘
                             │
                      ┌──────▼──────┐
                      │  RE-SCORE   │ ← update heuristic, loop back
                      └──────┬──────┘
                             │
                     ┌───────▼───────┐
                     │ STOPPING MET? │
                     └───────┬───────┘
                             │ yes
                        ┌────▼────┐
                        │ REPORT  │
                        └─────────┘
```

**How the DFS works:** The orchestrator maintains a **priority stack** of candidate actions.
After each action completes, the stack is re-scored and the highest-scored action is popped
next. This is DFS because each action can push new sub-actions (e.g., PROFILE pushes
GEAK candidates, BACKENDS pushes combination tests). The agent explores depth-first along
the most promising branch, but can backtrack if scores shift.

**Exploration beyond the tree:** The agent is NOT limited to the pre-defined actions. If
profiling reveals an unexpected bottleneck or a KB query suggests a novel technique, the
agent can create ad-hoc actions and score them with the same heuristic. The tree above is
the default starting structure — the agent should actively look for opportunities outside it.

**Communication:** This is a single-agent sequential loop, not async multi-agent. Each action
runs to completion before the orchestrator re-scores. The "parallel" aspect is within actions
(e.g., GEAK submits 5 kernels in parallel), not between actions.

## Autonomy Rules

**This skill runs end-to-end without human confirmation.** Do NOT ask the user before:
- Creating/stopping RayJob on SaFE (claw mode)
- Running baseline/profiling scripts via Ray (claw mode) or locally
- Submitting GEAK tasks
- Killing/restarting servers (inside RayJob or locally)
- Patching kernels (Inductor cache or source files)
- Reverting failed patches

Execute the full DFS pipeline autonomously. Present only the **final optimization report** to the user.

## Heuristic Scoring Function

Every candidate action is scored by:

```
score = (expected_tput_gain_per_gpu / cost_minutes)
        × (1 - accuracy_risk)
        × (1 - crash_risk)
        × target_gap_multiplier
```

| Component | Source | Range |
|-----------|--------|-------|
| `expected_tput_gain_per_gpu` | KB lookup + model class priors | 0–100+ tok/s/GPU |
| `cost_minutes` | Estimated wall-clock time | 2–120 min |
| `accuracy_risk` | From KB (kernel mods = 0.15, backends = 0.1, params = 0.0) | 0.0–1.0 |
| `crash_risk` | From KB (vendor kernel mods = 0.5, scheduling = 0.05) | 0.0–1.0 |
| `target_gap_multiplier` | `1 + min(target_gap_pct, 100) / 100` | 1.0–2.0 |

### Initial Score Priors (by model class)

| Action | Dense | MoE+MLA | MoE+SWA | MoE+MLA+NSA |
|--------|-------|---------|---------|-------------|
| backends | 3 | **9** | **8** | **10** |
| params | 5 | 6 | 7 | 5 |
| kernel-opt (GEAK) | **8** | 2 | 2 | 2 |
| torch.compile | **7** | 0 | 0 | 0 |
| sweep | 1 | 1 | 1 | 1 |

Scores update after each action based on measured results.

### Score Update Rules

After each action completes:

1. **Action succeeded (gain > 0%):** Boost similar actions. E.g., if `backends` gained +5%,
   boost remaining untested backends by 1.5×. Boost `combined_test` score.
2. **Action failed (gain ≤ 0%):** Reduce similar actions by 0.5×.
3. **After 2+ backend wins:** Push `combined_backends_test` with score = sum(individual scores) × 1.5
4. **After all backends tested:** Push `re-profile` (to discover new GEAK targets)
5. **After kernel opt kept:** Push `re-profile + next-kernel` with boosted score
6. **After kernel opt discarded:** Reduce remaining kernel scores by 0.7×
7. **When all action scores < 1.0:** Proceed to sweep → report

## State Schema

The orchestrator maintains this state throughout the run:

```python
state = {
    "model_name": "",
    "model_class": "",           # dense / moe_mla / moe_swa / moe_mla_nsa
    "framework": "sglang",
    "tp": 8,
    "gpu_type": "MI355X",

    "baseline_tput_per_gpu": 0.0,
    "current_tput_per_gpu": 0.0,
    "cumulative_gain_pct": 0.0,

    "target_tput_per_gpu": None,  # from target-analysis, if available
    "target_gap_pct": None,

    "torch_compile_status": None,  # success / failed / skipped
    "accuracy_reference": None,    # path to reference output

    "action_stack": [],            # priority stack of (score, action_name, params)
    "completed_actions": [],       # log of (action_name, gain_pct, status)
    "kernel_candidates": [],       # from profiling
    "winning_backends": [],        # from backend exploration
    "winning_params": [],          # from param tuning

    "total_wall_minutes": 0,
    "total_geak_submissions": 0,
    "consecutive_discards": 0,
}
```

## Orchestrator Loop

```
PROCEDURE optimize():

  1. SETUP
     → Execute actions/setup.md
     → Set MODEL, TP, CONC, FRAMEWORK, paths

  2. CLASSIFY
     → Execute actions/classify.md
     → Set model_class, torch_compile_viable, initial score priors

  3. TARGET ANALYSIS (if $TARGET_DIR provided)
     → Execute actions/target-analysis.md
     → Set target_tput_per_gpu, target_gap_pct, target_gap_multiplier

  4. KB WARM-UP
     → Query KB for this model: python3 kb/kb_query.py --model "$MODEL_NAME" --top-k 20
     → Apply KB-informed adjustments to score priors

  5. BASELINE
     → Execute actions/baseline.md
     → Set baseline_tput_per_gpu, torch_compile_status, accuracy_reference

  6. PROFILE
     → Execute actions/profile.md
     → Populate kernel_candidates with (name, gpu_pct, source)

  7. BUILD ACTION STACK
     → Score all candidate actions using the heuristic
     → Push onto action_stack sorted by score (highest first)

  8. DFS LOOP:
     WHILE action_stack is not empty AND NOT stopping_criteria_met():
       a. Pop highest-scored action
       b. Execute the action (dispatch to actions/*.md)
       c. ACCURACY GATE: if action modified computation, verify accuracy
          - If accuracy degraded: REVERT immediately, mark action as FAIL
       d. Measure result: new_tput_per_gpu
       e. Update state: current_tput_per_gpu, cumulative_gain_pct
       f. RE-SCORE all remaining actions (gains shift after each optimization)
       g. Push any new sub-actions discovered during execution
       h. Log to completed_actions

  9. SWEEP
     → Execute actions/sweep.md (full ISL/OSL/CONC parameter sweep)

 10. REPORT
     → Execute actions/report.md (generate optimization report + KB contribution)

 11. KNOWLEDGE HOOK
     → The .cursor/hooks/knowledge-sink.py hook fires automatically
     → Ingests any new knowledge discovered during the run
```

## Accuracy Gate Protocol

**Every action that modifies computation** must pass the accuracy gate before KEEP:

1. **Kernel-level check** (for GEAK/kernel mods):
   ```python
   # torch.allclose on kernel micro-benchmark output
   assert torch.allclose(original_output, optimized_output, atol=1e-3, rtol=1e-3)
   ```

2. **E2E output check** (for all actions):
   ```bash
   curl -s http://localhost:$PORT/v1/completions \
     -H "Content-Type: application/json" \
     -d '{"model":"'$MODEL'","prompt":"The capital of France is","max_tokens":20,"temperature":0}'
   # Compare with $RESULT_DIR/accuracy_reference.json
   ```

3. **If accuracy fails:** REVERT immediately. Log to KB as `accuracy_risk=1.0` for this
   specific action + model combination.

**Actions that do NOT need accuracy gate:** setup, classify, profile, sweep (read-only),
server params that only affect scheduling (decode-steps, cuda-graph-max-bs, mem-fraction).

## Stopping Criteria

| Condition | Action |
|-----------|--------|
| All action scores < 1.0 | Proceed to sweep |
| Cumulative gain > 25% | Proceed to sweep |
| 5 consecutive discards across all actions | Proceed to sweep |
| Wall clock > 180 min total | Proceed to sweep |
| Target exceeded (gap ≤ 0%) | Proceed to sweep |
| 2+ server crashes | Emergency stop, report partial results |

## KB Integration

Before each action, query the KB for relevant knowledge:

```bash
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME $ACTION_NAME" --top-k 5 --compact
```

After each action with new findings, ingest into KB:

```bash
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category $CATEGORY --model "$MODEL_NAME" \
    --action "$WHAT_WAS_DONE" --lesson "$KEY_TAKEAWAY" \
    --tags $TAGS --gain $GAIN --status $STATUS
```

The KB provides:
- **Prior knowledge:** Skip actions already known to fail for this model class
- **Score calibration:** Adjust expected gains based on past results
- **Conflict detection:** Flag contradictory information automatically

## Action Dispatch

| Action | Module | When |
|--------|--------|------|
| Setup | [`actions/setup.md`](actions/setup.md) | Always first |
| Classify | [`actions/classify.md`](actions/classify.md) | Always second |
| Target Analysis | [`actions/target-analysis.md`](actions/target-analysis.md) | If `$TARGET_DIR` provided |
| Baseline | [`actions/baseline.md`](actions/baseline.md) | After classify |
| Profile | [`actions/profile.md`](actions/profile.md) | After baseline |
| Backend Exploration | [`actions/backends.md`](actions/backends.md) | DFS loop |
| Server Params | [`actions/params.md`](actions/params.md) | DFS loop |
| Kernel Optimization | [`actions/kernel-opt.md`](actions/kernel-opt.md) | DFS loop |
| Integration | [`actions/integrate.md`](actions/integrate.md) | Per-kernel sub-action |
| Parameter Sweep | [`actions/sweep.md`](actions/sweep.md) | After DFS loop |
| Report | [`actions/report.md`](actions/report.md) | Always last |

## Reference: Critical Lessons

These are the most important validated lessons. Full details in KB and action modules.

1. **Backend switches outperform parameter sweeps.** GLM-5: backends +16.2% vs params <1%.
   Always explore backends BEFORE sweeping parameters.

2. **Combination synergies can be super-linear.** Two +3% backends → +16.2% combined.
   Always test winners together.

3. **torch.compile is prerequisite for large GEAK wins.** With compile: up to +14.72%.
   Without: ≤1.76%.

4. **Benchmark fairness is critical.** Kimi-K2.5 "+40.4%" was invalid — actually +0.81%
   after controlling for CONC mismatch. Always save baseline config and reuse.

5. **Server param tuning can dominate.** Kimi vLLM +84% from gpu-mem + max-num-seqs.
   CUDA graph coverage +35% when misconfigured.

6. **GEAK cannot beat vendor kernels.** Never submit `Cijk_*` or `aiter::*` to GEAK.

7. **MUST patch STANDALONE files, not graph modules.** Graph module patching = 0%.
   Standalone patching = +9%.

8. **Use Python AST for source patching.** Naive regex deletes module-level variables.

## Reference: Process Management

- **Never use `pkill -f "sglang.launch_server"` inside scripts** — kills the script itself.
- **Wait `SERVER_KILL_WAIT_S` seconds** (default 10) between server kill and relaunch.
- **Always `unset PROFILE SGLANG_TORCH_PROFILER_DIR`** after profiling.
- **Always use filtered traces** for TraceLens (raw: 349MB, filtered: 5MB).
- TraceLens does NOT support `rocprofv3` format — only PyTorch Kineto.

## Reference: Benchmark Metrics

| Metric | Unit | Meaning |
|--------|------|---------|
| `output_throughput` | tok/s | Output tokens per second |
| `tput_per_gpu` | tok/s/GPU | `output_throughput / TP` |
| `mean_tpot_ms` | ms | Time Per Output Token (decode latency) |
| `mean_ttft_ms` | ms | Time to First Token (prefill latency) |

## Reference: Server Parameter Tables

See [`KNOWLEDGE-BASE.md`](KNOWLEDGE-BASE.md) for validated parameter tables (SGLang, vLLM)
and model-specific configurations. Query the KB for the latest:

```bash
python3 $SKILL_ROOT/kb/kb_query.py --category server_params --compact
```

## Reference: vLLM Integration

All scripts support `FRAMEWORK=vllm`. Parameter mapping:

| SGLang | vLLM | Notes |
|--------|------|-------|
| `--model-path` | `vllm serve <model>` (positional) | — |
| `--mem-fraction-static 0.8` | `--gpu-memory-utilization 0.85` | — |
| `--disable-radix-cache` | `--no-enable-prefix-caching` | For random benchmarks |
| `--enable-torch-compile` | Default ON (level=3) | Disable: `--enforce-eager` |
| `SGLANG_TORCH_PROFILER_DIR` | `VLLM_TORCH_PROFILER_DIR` | Set before server launch |
