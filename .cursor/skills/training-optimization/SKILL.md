---
name: training-optimization
description: |
  Autonomous DFS-guided training optimization for distributed LLM training on AMD MI355X GPUs.
  Uses heuristic-scored depth-first search to systematically explore optimization actions
  (fusion flags, parallelism configs, training params, kernel optimization) and minimize
  ms/iter with convergence correctness as a hard constraint. Global batch size must remain
  constant throughout — only config/code efficiency improvements are valid.
globs:
  - "**/training*optim*"
  - "**/primus*"
  - "**/megatron*"
  - "**/torchrun*"
---

# Training Optimization — DFS Orchestrator

## Overview

This skill optimizes distributed LLM training throughput (ms/iter) on AMD MI355X GPUs.
It runs a **depth-first search** over optimization actions, guided by a heuristic scoring
function. The search is fully autonomous — no human prompting required.

**Primary objective:** minimize `ms_per_iter` (equivalently, maximize `samples/sec/GPU`)
**Hard constraint:** global batch size (GBS) must remain identical to baseline throughout
**Hard constraint:** training convergence must not degrade (loss trajectory unchanged)
**Optional target:** if a prior run or external baseline is provided, the target gap acts
as an urgency multiplier on all action scores.

## Architecture

```
SKILL.md (this file)          — DFS orchestrator: loop, heuristic, dispatch
actions/*.md                   — Self-contained action modules (11 actions)
kb/                            — RAG knowledge base (JSONL + query/ingest scripts)
scripts/                       — Training benchmark/profiling shell scripts
GEAK-KERNEL-OPTIMIZATION.md   — GEAK CLI deep reference for kernel optimization
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
                   │  TARGET ANALYSIS   │ ← optional, if target provided
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
              │  ┌──────────┐ ┌───────────┐ │
              │  │ FUSION   │ │PARALLELISM│ │
              │  │ FLAGS    │ │           │ │
              │  └────┬─────┘ └─────┬─────┘ │
              │       │             │        │
              │  ┌────▼────┐  ┌────▼─────┐  │
              │  │ PARAMS  │  │ KERNEL   │  │
              │  │         │  │  OPT     │  │
              │  └────┬────┘  └────┬─────┘  │
              │       │            │         │
              │  ┌────▼────┐ ┌────▼─────┐   │
              │  │INTEGRATE│ │  SWEEP   │   │
              │  └─────────┘ └──────────┘   │
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
GEAK candidates, FUSION-FLAGS pushes combination tests). The agent explores depth-first
along the most promising branch, but can backtrack if scores shift.

**Exploration beyond the tree:** The agent is NOT limited to the pre-defined actions. If
profiling reveals an unexpected bottleneck or a KB query suggests a novel technique, the
agent can create ad-hoc actions and score them with the same heuristic. The tree above is
the default starting structure — the agent should actively look for opportunities outside it.

## Heuristic Scoring Function

Every candidate action is scored by:

```
score = (expected_ms_reduction / cost_minutes)
        × (1 - convergence_risk)
        × (1 - crash_risk)
        × target_gap_multiplier
```

| Component | Source | Range |
|-----------|--------|-------|
| `expected_ms_reduction` | KB lookup + model class priors | 0–500+ ms |
| `cost_minutes` | Estimated wall-clock time | 3–60 min |
| `convergence_risk` | From KB (kernel mods = 0.15, fusion flags = 0.05, params = 0.0) | 0.0–1.0 |
| `crash_risk` | From KB (parallelism changes = 0.3, env vars = 0.05) | 0.0–1.0 |
| `target_gap_multiplier` | `1 + min(target_gap_pct, 100) / 100` | 1.0–2.0 |

### Initial Score Priors (by model class)

| Action | Dense | MoE+GQA | MoE+MLA | MoE+SWA |
|--------|-------|---------|---------|---------|
| fusion-flags | 5 | **9** | **8** | **8** |
| parallelism | 4 | 6 | 6 | 6 |
| params | 3 | 5 | 5 | 5 |
| kernel-opt (GEAK) | **7** | 3 | 2 | 2 |
| attention-backend | 6 | 4 | 4 | **7** |
| sweep | 1 | 1 | 1 | 1 |

Scores update after each action based on measured results.

### Score Update Rules

After each action completes:

1. **Action succeeded (gain > 0%):** Boost similar actions. E.g., if `fusion-flags` gained
   +1.5%, boost remaining untested fusion flags by 1.5×.
2. **Action failed (gain ≤ 0%):** Reduce similar actions by 0.5×.
3. **After 2+ fusion flag wins:** Push `combined_fusion_test` with score = sum(individual) × 1.5
4. **After all fusion flags tested:** Push `re-profile` (to discover new kernel targets)
5. **After kernel opt kept:** Push `re-profile + next-kernel` with boosted score
6. **After kernel opt discarded:** Reduce remaining kernel scores by 0.7×
7. **When all action scores < 1.0:** Proceed to sweep → report

## State Schema

The orchestrator maintains this state throughout the run:

```python
state = {
    "model_name": "",
    "model_class": "",           # dense / moe_gqa / moe_mla / moe_swa
    "framework": "primus",       # primus / megatron / deepspeed / fsdp
    "num_gpus": 8,
    "tp": 1, "pp": 1, "ep": 8,  # parallelism config
    "gpu_type": "MI355X",

    "baseline_ms_per_iter": 0.0,
    "current_ms_per_iter": 0.0,
    "cumulative_gain_pct": 0.0,

    "global_batch_size": 0,      # IMMUTABLE — must match baseline throughout
    "micro_batch_size": 0,
    "seq_length": 0,

    "target_ms_per_iter": None,  # from target-analysis, if available
    "target_gap_pct": None,

    "config_yaml": "",           # path to training config
    "kept_overrides": [],        # accumulated config overrides that improved perf
    "kept_patches": [],          # code patches that improved perf

    "action_stack": [],          # priority stack of (score, action_name, params)
    "completed_actions": [],     # log of (action_name, gain_pct, status)
    "kernel_candidates": [],     # from profiling
    "winning_flags": [],         # from fusion flag exploration
    "winning_params": [],        # from param tuning

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
     → Set framework, config paths, GPU count, torchrun command template

  2. CLASSIFY
     → Execute actions/classify.md
     → Set model_class, initial score priors, GBS, parallelism topology

  3. TARGET ANALYSIS (if $TARGET_DIR or target numbers provided)
     → Execute actions/target-analysis.md
     → Set target_ms_per_iter, target_gap_pct, target_gap_multiplier

  4. KB WARM-UP
     → Query KB for this model: python3 kb/kb_query.py --model "$MODEL_NAME" --top-k 20
     → Apply KB-informed adjustments to score priors

  5. BASELINE
     → Execute actions/baseline.md
     → Set baseline_ms_per_iter, GBS verification, accuracy reference

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
       c. GBS GATE: verify global_batch_size matches baseline
          - If GBS changed: REVERT immediately, mark action as INVALID
       d. Measure result: new_ms_per_iter (iter 6–10 average)
       e. Update state: current_ms_per_iter, cumulative_gain_pct
       f. RE-SCORE all remaining actions (gains shift after each optimization)
       g. Push any new sub-actions discovered during execution
       h. Log to completed_actions

  9. SWEEP
     → Execute actions/sweep.md (micro_batch_size × precision sweep)

 10. REPORT
     → Execute actions/report.md (generate optimization report + KB contribution)

 11. KNOWLEDGE HOOK
     → Ingest any new knowledge discovered during the run via kb_ingest.py
```

## GBS Gate Protocol (CRITICAL)

**Every action that modifies training config** must pass the GBS gate before KEEP:

1. **Extract GBS from training log:**
   ```bash
   grep "global_batch_size" /tmp/attempt_N.log | tail -1
   ```

2. **Compare with baseline GBS:**
   - If `actual_gbs == baseline_gbs`: PASS — proceed to keep/discard based on ms/iter
   - If `actual_gbs != baseline_gbs`: FAIL — REVERT immediately, mark as INVALID

3. **What counts as GBS change:**
   - Direct: `global_batch_size=256` (was 512)
   - Indirect: changing DP degree without adjusting micro_batch_size
   - Indirect: changing gradient_accumulation_steps without maintaining GBS = MBS × DP × GA

**Actions that do NOT need GBS gate:** setup, classify, profile (read-only), kernel-opt
(same config, different kernel code).

## Stopping Criteria

| Condition | Action |
|-----------|--------|
| All action scores < 1.0 | Proceed to sweep |
| Cumulative gain > 15% | Proceed to sweep |
| 5 consecutive discards across all actions | Proceed to sweep |
| Wall clock > 120 min total | Proceed to sweep |
| Target exceeded (gap ≤ 0%) | Proceed to sweep |
| 2+ training crashes | Emergency stop, report partial results |

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
| Target Analysis | [`actions/target-analysis.md`](actions/target-analysis.md) | If target provided |
| Baseline | [`actions/baseline.md`](actions/baseline.md) | After classify |
| Profile | [`actions/profile.md`](actions/profile.md) | After baseline |
| Fusion Flags | [`actions/fusion-flags.md`](actions/fusion-flags.md) | DFS loop |
| Parallelism | [`actions/parallelism.md`](actions/parallelism.md) | DFS loop |
| Training Params | [`actions/params.md`](actions/params.md) | DFS loop |
| Kernel Optimization | [`actions/kernel-opt.md`](actions/kernel-opt.md) | DFS loop |
| Integration | [`actions/integrate.md`](actions/integrate.md) | Per-kernel sub-action |
| Parameter Sweep | [`actions/sweep.md`](actions/sweep.md) | After DFS loop |
| Report | [`actions/report.md`](actions/report.md) | Always last |

## Reference: Critical Lessons

These are the most important validated lessons. Full details in KB and action modules.

1. **Fusion flags outperform parallelism tuning for MoE.** GPT-OSS 20B: `moe_permute_fusion`
   +1.19% vs parallelism changes <0.5%. Always explore fusion flags BEFORE parallelism.

2. **GBS is immutable.** Any attempt that changes GBS is INVALID regardless of speedup.
   Verify GBS in training log before every KEEP decision.

3. **Combination synergies exist.** `moe_permute_fusion` + `gradient_accumulation_fusion`
   combined can exceed sum of individual gains. Always test winners together.

4. **GEAK cannot beat vendor BLAS/attention.** Never submit `Cijk_*` or `aiter::fmha_v3_*`
   to GEAK — these are hand-tuned for MFMA.

5. **aiter `deterministic=True` kills backward perf.** Falls back to `mha_bwd` (2.7× slower)
   on gfx950 with seqlen > 256. PrimusTurbo defaults to `deterministic=False`.

6. **Port conflicts after killing runs.** Increment `--master_port` (29501, 29502, ...)
   after killing training processes.

7. **Mock data C++ helpers must be compiled.** Run
   `make -C /workspace/Primus/third_party/Megatron-LM/megatron/core/datasets` first.

8. **hipBLASLt GEMMs dominate (60–70% GPU time).** Gains come from reducing everything else.

## Reference: Process Management

- **Kill lingering processes:** `pkill -9 -f "primus/cli/main.py"` before retrying.
- **Wait 5+ seconds** between kill and relaunch.
- **Increment `--master_port`** if the previous port is still bound.
- **Always use filtered traces** for TraceLens (raw training traces can be >300MB).
- TraceLens does NOT support `rocprofv3` format — only PyTorch Kineto.

## Reference: Training Metrics

| Metric | Unit | Meaning |
|--------|------|---------|
| `ms_per_iter` | ms | Milliseconds per training iteration (primary metric) |
| `samples_per_sec` | samples/s | Throughput in samples per second |
| `samples_per_sec_per_gpu` | samples/s/GPU | Per-GPU throughput |
| `tflops_per_gpu` | TFLOP/s/GPU | Compute efficiency |
| `memory_gb_per_gpu` | GB | Peak GPU memory usage |

## Reference: Measurement Protocol

- Use **iterations 6–10** for timing (skip 1–5 for warmup, JIT, NCCL init)
- Extract `elapsed time per iteration (ms)` from training log
- Compute average — this is the measured ms/iter
- If log format differs, look for `throughput`, `samples/sec`, `TFLOP/s`

## Reference: Config Override Syntax

For Primus/Megatron, overrides are key=value pairs appended after `--config`:

```bash
torchrun --nproc_per_node=8 --master_port=29500 \
  -m primus.cli.main train pretrain \
  --config <CONFIG_YAML> \
  key1=value1 key2=value2 \
  profile=false use_pytorch_profiler=false \
  2>&1 | tee /tmp/attempt_N.log
```

## Reference: vLLM Integration (for inference validation)

If the training optimization is paired with inference benchmarking, see the
inference-optimization skill at `../.cursor/skills/inference-optimization/SKILL.md`.
