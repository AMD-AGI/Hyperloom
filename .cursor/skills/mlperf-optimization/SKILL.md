---
name: mlperf-optimization
description: |
  Autonomous DFS-guided MLPerf training optimization for GPT-OSS-20B on AMD MI355X GPUs.
  Uses heuristic-scored depth-first search to systematically explore optimization actions
  (fusion flags, parallelism configs, training params, kernel optimization, runtime tunables)
  and minimize time-to-train (wall seconds to reach validation loss target) with MLPerf
  compliance as a hard constraint.
globs:
  - "**/mlperf*"
  - "**/primus*"
  - "**/megatron*"
  - "**/torchrun*"
  - "**/training_optimization/mlperf*"
---

# MLPerf Training Optimization — DFS Orchestrator

## Overview

This skill optimizes MLPerf GPT-OSS-20B training time-to-target on AMD MI355X GPUs.
It runs a **depth-first search** over optimization actions, guided by a heuristic scoring
function. The search is fully autonomous — no human prompting required.

**Primary objective:** minimize `time_to_train` (wall seconds from `run_start` to `run_stop`
with status `success`, reaching validation loss target)
**Hard constraint:** validation log perplexity must reach target (3.34) — run is invalid otherwise
**Hard constraint:** MLPerf compliance logging must be preserved (:::MLLOG format)
**Secondary metric:** `ms_per_iter` (derived from consecutive MLLOG train_loss timestamps)
**Optional target:** if a prior run or external baseline is provided, the target gap acts
as an urgency multiplier on all action scores.

## MLPerf Benchmark Context

- **Benchmark:** GPT-OSS-20B (gpt-oss-20b)
- **Model:** 20B parameter GPT with Mixture of Experts (32 experts, top-k=4)
- **Dataset:** c4/en/3.0.1 (pre-tokenized, ~80GB)
- **Quality target:** validation log perplexity = 3.34
- **Eval frequency:** every 12,288 samples (eval_interval adjusts with GBS)
- **Eval size:** 1024 sequences (64 eval iters × MBS × GPUs)
- **Precision:** BF16 / FP8 hybrid (E4M3 activations/weights, E5M2 gradients)
- **Ruleset:** MLPerf Training 5.1.0

## MCP Server References

The following MCP servers are used during optimization. The agent MUST reference
these by their exact server names when calling `CallMcpTool`. Connectivity is
verified and **self-healed** in `actions/setup.md` Step 8 (see below).

| Server | Name in mcp.json | Auth | Role | Used By |
|--------|-------------------|------|------|---------|
| TraceLens | `oci-traceLens-agent` | Bearer token (same domain) | Profiling | profile.md, re-profile.md, report.md, comm-tuning.md |
| OOB Agent | `oob-optimizer-dev` | Bearer token | Kernel opt (parallel backend) | kernel-opt/oob-claude.md, kernel-opt/oob-codex.md |
| GEAK | `oci-geak-agent` | Bearer token | Kernel opt (parallel backend) | kernel-opt/geak.md |

**TraceLens tools:** `check_trace_file`, `run_full_standalone_analysis`, `run_comparative_analysis`

**TraceLens fallback:** When TraceLens MCP is unavailable, the local profiling pipeline
(`scripts/run_profile.sh` + `scripts/parse_trace.py`) operates independently. It sets
`tensorboard_dir` to a deterministic path (`$RESULT_DIR/tb_traces`), discovers trace files
via a two-level strategy (known dir first, then `discover_trace()` with relaxed fallback),
and generates `categories.json` + `geak_candidates.json` for category-based heuristic
adjustments. TraceLens-specific metrics (roofline, overlap, MFMA utilization) are skipped;
`compute_heuristic_adjustments()` applies category-only rules automatically.

**OOB Agent tools:** `agent_create_task`, `agent_submit_task`, `agent_get_task`,
`agent_get_outputs`, `agent_download_file`, `agent_cancel_task`

**GEAK tools:** `geak_set_model_config`, `geak_create_task`, `geak_submit_task`,
`geak_get_task`, `geak_get_outputs`, `geak_download_file`

### MCP Self-Healing (setup.md Step 8)

Setup runs a Python script that probes each server using both MCP transports:
**Streamable HTTP** (POST JSON-RPC directly to the URL) and **SSE** (GET to
obtain a message endpoint, then POST). Different servers use different transports
(e.g., TraceLens uses Streamable HTTP at `.../mcp`; GEAK and OOB use SSE at
`.../sse`). The probe tries both transports automatically — **server URLs are
never modified**. The only auto-heal is **auth propagation** from sibling servers
on the same domain when a server has no Authorization header. Results populate
`state["mcp_status"]`, `state["mcp_tools"]`, and per-server boolean flags that
control downstream action dispatch.

If a service is genuinely offline after the probe, the agent records it as `down`
and falls back gracefully — no user intervention needed.

## Architecture

```
SKILL.md (this file)                — DFS orchestrator: loop, heuristic, dispatch
actions/*.md                         — Self-contained action modules (17 actions)
kernel-opt/                          — Per-backend kernel optimization references (geak, oob-claude, oob-codex)
scripts/parse_trace.py               — Trace analysis (operator summary, kernel categorization, heuristics)
kb/                                  — RAG knowledge base (JSONL + query/ingest scripts)
scripts/common.sh                    — Tiered trial runner + metric extraction helpers
scripts/trial_monitor.py             — Stdin log filter + progress display + anomaly detection
scripts/apply_quiet_config.sh        — YAML quiet/restore for noise reduction
scripts/run_baseline.sh              — Standalone baseline script
scripts/run_sweep.sh                 — GBS × LR sweep script
scripts/run_trial.sh                 — CLI wrapper for run_mlperf_trial
scripts/run_profile.sh               — Profiling run script (sets tensorboard_dir, discovers traces, runs parse_trace.py)
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
                   ┌─────────▼─────────┐
                   │ PROFILE + TRACELENS│ ← TraceLens roofline + overlap analysis (local fallback if TraceLens down)
                   └─────────┬─────────┘
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
              │  │ FUSION   │ │  PARAMS   │ │
              │  │ FLAGS    │ │           │ │
              │  └────┬─────┘ └─────┬─────┘ │
              │       │             │        │
              │  ┌────▼────┐  ┌────▼─────┐  │
              │  │ KERNEL  │  │ RUNTIME  │  │
              │  │  OPT    │  │ TUNABLES │  │
              │  └────┬────┘  └────┬─────┘  │
              │       │            │         │
              │  ┌────▼────┐  ┌────▼─────┐  │
              │  │FP8 RECIPE│ │INTEGRATE │  │
              │  └────┬────┘  └────┬─────┘  │
              │       └──────┬─────┘         │
              └──────────────┬──────────────┘
                             │
                      ┌──────▼──────┐
                      │  RE-SCORE   │ ← update heuristic, loop back
                      └──────┬──────┘
                             │
                    ┌────────▼────────┐
                    │  RE-PROFILE?    │ ← if gain > 2%, refresh bottlenecks
                    └────────┬────────┘
                             │
                     ┌───────▼───────┐
                     │ STOPPING MET? │
                     └───────┬───────┘
                             │ yes
                   ┌─────────▼─────────┐
                   │  CONFIG SELECTION  │ ← GBS/LR/TP/EP/DP (Tier 1→2L→3)
                   └─────────┬─────────┘
                             │
                   ┌─────────▼─────────┐
                   │CONVERGENCE SPEED  │ ← eval interval, LR schedule, warmup
                   └─────────┬─────────┘
                             │
                   ┌─────────▼─────────┐
                   │   COMM TUNING     │ ← NCCL/DeepEP/AllReduce (always runs)
                   │  COMM OVERLAP     │ ← TraceLens-guided overlap optimization
                   └─────────┬─────────┘
                             │
                        ┌────▼────┐
                        │ SWEEP   │ ← MBS / eval interval refinement
                        └────┬────┘
                             │
                        ┌────▼────┐
                        │ REPORT  │ ← TraceLens comparative analysis
                        └─────────┘
```

**How the DFS works:** The orchestrator maintains a **priority stack** of candidate actions.
After each action completes, the stack is re-scored and the highest-scored action is popped
next. This is DFS because each action can push new sub-actions (e.g., PROFILE pushes
kernel optimization candidates, FUSION-FLAGS pushes combination tests). The agent explores depth-first
along the most promising branch, but can backtrack if scores shift.

**Exploration beyond the tree:** The agent is NOT limited to the pre-defined actions. If
profiling reveals an unexpected bottleneck or a KB query suggests a novel technique, the
agent can create ad-hoc actions and score them with the same heuristic. The tree above is
the default starting structure — the agent should actively look for opportunities outside it.

## Heuristic Scoring Function

Every candidate action is scored by:

```
score = (expected_time_reduction_pct / cost_minutes)
        × (1 - convergence_risk)
        × (1 - crash_risk)
        × target_gap_multiplier
```

| Component | Source | Range |
|-----------|--------|-------|
| `expected_time_reduction_pct` | KB lookup + model class priors | 0–30% |
| `cost_minutes` | Estimated wall-clock for one trial | 3–60 min |
| `convergence_risk` | Likelihood of failing to reach 3.34 target | 0.0–1.0 |
| `crash_risk` | From KB (parallelism changes = 0.3, env vars = 0.05) | 0.0–1.0 |
| `target_gap_multiplier` | `1 + min(target_gap_pct, 100) / 100` | 1.0–2.0 |

## Action Classification

Every action is classified as either **throughput-only** or **convergence-affecting**.
This determines which evaluation layers apply during the DFS loop.

| Category | Actions | Evaluation Path |
|----------|---------|-----------------|
| throughput-only | fusion-flags, kernel-opt, runtime-tunables, comm-tuning, params (MBS/recompute/overlap) | Layer 1 only (ms/iter via Tier 1) |
| convergence-affecting | config-selection, convergence-speed, fp8-recipe-tuning | Layer 2 (loss_efficiency via Tier 2) + Layer 3 (projected_ttt via Tier 2.5) |

**Throughput-only** actions change kernel efficiency, communication overhead, or system-level
settings without altering training dynamics. Their effect is fully captured by ms/iter.

**Convergence-affecting** actions change the loss trajectory — they may not change ms/iter at all
but can dramatically shift iterations-to-converge. Examples: LR halving has zero ms/iter impact
but may save 500 iterations; GBS changes affect both ms/iter and sample efficiency.

The `classify_action()` helper in `scripts/common.sh` implements this classification.

## Three-Layer Evaluation Metric

The DFS loop uses a three-layer metric system instead of the single-dimensional ms/iter gain:

**Layer 1 — ms/iter gain** (throughput-only actions):
```
gain_pct = (baseline_ms_per_iter - new_ms_per_iter) / baseline_ms_per_iter × 100
```
Measured via Tier 1 trial (100 iters). Positive = faster. This is the sole metric for
throughput-only actions.

**Layer 2 — loss_efficiency** (convergence-affecting actions, quick filter):
```
loss_efficiency = (first_eval_loss - last_eval_loss) / wall_time_seconds
```
Measured via Tier 2 trial (500 iters). Higher = faster convergence per unit time. Used to
quickly discard clearly inferior candidates without running expensive Tier 2.5 trials.
Threshold: discard if `candidate_efficiency < baseline_efficiency × 0.85` (15% worse).

**Layer 3 — projected_ttt** (convergence-affecting actions, final decision):
```
ttt_gain_pct = (baseline_projected_ttt - candidate_projected_ttt) / baseline_projected_ttt × 100
```
Measured via Tier 2.5 trial (1500 iters). Uses `project_ttt()` to extrapolate time-to-target
from the loss-vs-samples curve. Positive = faster convergence. This is the final keep/discard
criterion for convergence-affecting actions.

**Composite gain for score updates:**
```
For throughput-only:       gain = ms_per_iter_gain_pct
For convergence-affecting: gain = ttt_gain_pct (from Layer 3)
```

### Initial Score Priors (GPT-OSS-20B MoE on MI355X)

| Action | Score | Rationale |
|--------|-------|-----------|
| fusion-flags | **9** | Highest impact for MoE: permute fusion, GA fusion |
| config-selection | **8** | GBS/LR/TP/EP/DP — biggest convergence-speed lever |
| convergence-speed | **7** | LR decay window + min_lr + peak LR + eval interval + warmup/wd fine-tuning |
| fp8-recipe-tuning | **6** | FP8 knobs affect both ms/iter and convergence |
| runtime-tunables | 5 | System-level knobs (NUMA, hugepages, NCCL) |
| params (training) | 5 | MBS, recompute, overlap flags |
| comm-tuning | **6** | NCCL/RCCL/DeepEP/AllReduce comm tuning + TraceLens-guided overlap (always runs, post-config-selection) |
| kernel-opt (GEAK/OOB-Claude/OOB-Codex) | 5 | All non-vendor kernels explored; GEMM-dominated but optimization surface always tested |
| sweep | 1 | Final exploration of operating points |
| re-profile | — | Not scored independently; triggered by score update rules |

Scores update after each action based on measured results.

### Score Update Rules

After each action completes:

1. **Action succeeded (gain > 0%):** Boost similar actions. E.g., if `fusion-flags` gained
   +1.5%, boost remaining untested fusion flags by 1.5×.
2. **Action failed (gain ≤ 0%):**
   - Throughput-only actions: Reduce similar actions by 0.7×.
   - Convergence-affecting actions: Reduce similar actions by 0.85×.
     (Convergence effects are harder to measure in short trials; softer penalty avoids
     premature abandonment of promising hyperparameter directions.)
3. **After 2+ fusion flag wins:** Push `combined_fusion_test` with score = sum(individual) × 1.5
4. **After all fusion flags tested:** Push `re-profile` via [`actions/re-profile.md`](actions/re-profile.md) to discover new kernel targets
5. **After kernel opt kept:** Push `re-profile + next-kernel` via [`actions/re-profile.md`](actions/re-profile.md) with boosted score
6. **After kernel opt discarded:** Reduce remaining kernel scores by 0.7×, but kernel-opt
   score never falls below 0.5 (floor). Even with multiple negative adjustments, kernel-opt
   is never removed from the stack.
7. **After any action with gain > 2%:** Push `re-profile` onto stack with score = `max(remaining_scores) × 0.8`
8. **After FP8 knob succeeds:** Boost remaining untested FP8 knobs by 1.3×
9. **After convergence-speed action:** Update `state.eval_overhead_seconds` and `state.optimal_eval_interval`
10. **When all action scores < 0.3 AND dfs_iteration_count ≥ 12:** Proceed to sweep → report.
    The iteration floor ensures at least 12 DFS iterations before stopping, even if scores are low.
11. **KB negative entry adjustment cap:** When a KB entry shows a prior negative result for
    an action, reduce the action's `expected_time_reduction_pct` by at most 30%. Never zero
    out an action based solely on KB history. The agent MUST still test the action if it has
    not been tested in this run.
12. **After LR schedule change (Dimension 5):** If `lr_decay_iters` change improved projected_ttt,
    boost `min_lr` and `peak_lr` sub-dimensions by 1.5× (they build on the decay window).
    Update `state.lr_decay_ratio`. If the ratio was previously > 5.0 (decay window >> convergence
    window), this is likely a high-value action — log it prominently. Target ratio: 2–5×.


## Orchestrator Loop

```
PROCEDURE optimize():

  1. SETUP
     → Execute actions/setup.md
     → Set framework, config paths, GPU count
     → Setup symlinks, verify data paths, kill stale processes
     → Validate trial_monitor.py and quiet_yaml functions
     → MCP self-healing: HTTP pre-flight → auto-fix mcp.json → reload → verify
     → Populate state["mcp_status"] and fallback flags (tracelens/oob/geak_available)
     → === REVIEW CHECKPOINT RC-1 ===

  2. CLASSIFY
     → Execute actions/classify.md
     → Set model_class, initial score priors, parallelism topology

  3. TARGET ANALYSIS (if $TARGET_DIR or target numbers provided)
     → Execute actions/target-analysis.md
     → Set target_time_to_train, target_gap_pct, target_gap_multiplier

  4. KB WARM-UP (REFERENCE ONLY)
     → Query KB for this model: python3 kb/kb_query.py --model "GPT-OSS-20B" --top-k 20
     → KB entries are REFERENCE ONLY — they inform priors but do NOT skip actions.
       Even if KB reports a prior result for the exact same action (e.g., "already
       tested, gain=0"), the agent MUST still run its own trial to verify under
       current conditions (hardware state, driver version, container image may differ).
     → PENALTY CAP (Score Update Rule #11): KB negative entries may only reduce an
       action's expected_time_reduction_pct by at most 30%. Never zero out an action
       based solely on KB history. MUST NOT reduce any action score below 1.0 or
       remove an action from the stack entirely.

  5. BASELINE (MANDATORY FULL RUN)
     → Execute actions/baseline.md
     → MUST run Tier 3 (full convergence): run_mlperf_trial "baseline" 3
       - Do NOT use Tier 1 or Tier 2 for baseline.
       - The baseline MUST train to convergence (eval_loss ≤ 3.34) or exhaust all iters.
       - This establishes the real baseline TTT that all optimizations compare against.
     → Extract baseline_time_to_train (actual TTT), baseline_ms_per_iter, baseline_eval_loss
     → Reference: current best known TTT = 206 min (must be re-verified by this baseline)
     → === REVIEW CHECKPOINT RC-2 ===

  6. PROFILE + TRACELENS
     → Execute actions/profile.md (Tier 1 with profiling enabled)
     → Populate kernel_candidates with (name, gpu_pct, bound_type, source)
     → Run TraceLens standalone analysis via CallMcpTool on MCP server "oci-traceLens-agent"
       Tool: run_full_standalone_analysis  (MANDATORY — do NOT skip)
     → Extract: comm_compute_overlap, mfma_utilization, memory_bound_kernel_pct
     → Store TraceLens metrics in state
     → Also compute baseline_loss_efficiency and baseline_projected_ttt from the
       baseline Tier 3 log using compute_loss_efficiency() and project_ttt()
     → === REVIEW CHECKPOINT RC-3 ===

  7. BUILD ACTION STACK
     → Score all candidate actions using the heuristic
     → Include convergence-speed, fp8-recipe-tuning, and comm-tuning in candidates
     → Include comm-tuning (both Part A and Part B) unconditionally
     → Push onto action_stack sorted by score (highest first)

  8. DFS LOOP:
     SET dfs_iteration_count = 0
     WHILE action_stack is not empty AND (dfs_iteration_count < 12 OR NOT stopping_criteria_met()):
       a. Pop highest-scored action
       b. Classify action as throughput-only or convergence-affecting (see Action Classification)

       c. IF throughput-only:
          c1. Execute with Tier 1 trial (100 iters)
          c2. Parse TRIAL_RESULT: check status (nan → REVERT, no_data → skip)
          c3. Measure: new_ms_per_iter from TRIAL_RESULT
          c4. Compute gain: compute_gain_pct(baseline_ms, new_ms_per_iter)
          c5. CONVERGENCE GATE: If gain > 1%, escalate to Tier 2 for convergence check
          c6. If Tier 2 shows eval_loss diverging: REVERT

       d. IF convergence-affecting:
          d1. Execute with Tier 2 trial (500 iters)
          d2. Parse TRIAL_RESULT: check status (nan → REVERT, no_data → skip)
          d3. Measure both ms_per_iter AND eval_loss trajectory
          d4. Compute loss_efficiency: compute_loss_efficiency(candidate_log)
          d5. QUICK FILTER: If loss_efficiency < baseline_loss_efficiency × 0.85 → DISCARD
          d6. Escalate to Tier 2.5 trial (1500 iters) for projected TTT
          d7. Compute: projected_ttt via project_ttt(candidate_log, GBS)
          d8. Compute gain: compute_ttt_gain_pct(baseline_projected_ttt, candidate_projected_ttt)
          d9. If ttt_gain_pct ≤ 0%: DISCARD. If > 0%: KEEP

       e. Update state: current_ms_per_iter, cumulative_gain_pct, projected_ttt (if applicable)
       f. RE-SCORE all remaining actions using the appropriate gain metric
       g. RE-PROFILE TRIGGER: If gain > 2%, push re-profile onto action_stack
          with score = max(remaining_scores) × 0.8 (see actions/re-profile.md)
       h. Push any new sub-actions discovered during execution
       i. Log to completed_actions with (action, gain_pct, gain_type, status)
       j. INCREMENT dfs_iteration_count
       k. IF dfs_iteration_count % 3 == 0:
          → === REVIEW CHECKPOINT RC-4 ===

  9. CONFIG SELECTION (GBS / LR / TP / EP / DP)
     → Execute actions/config-selection.md:
       a. Stage 1 (Tier 1 filter): Run each candidate config for 100 iters.
          Discard crash/NaN/OOM. Record ms/iter.
       b. Stage 2 (Tier 2L comparison): Run survivors for 2500 iters each.
          Extract loss-vs-samples curves, eval_loss trajectory, projected TTT.
          Rank candidates by projected TTT (lowest = best).
       c. Stage 3 (Tier 3 top-2 verification): Run only the top-2 candidates
          to full convergence. Measure actual TTT.
       d. Apply the winning config as the new baseline for subsequent steps.
     → === REVIEW CHECKPOINT RC-4.5 ===

  9.5 CONVERGENCE SPEED (on winning config)
     → Execute actions/convergence-speed.md
     → Dimension 1: Eval interval optimization
     → Dimension 2: LR warmup iters
     → Dimension 3: Weight decay
     → Dimension 4: Gradient clipping
     → Dimension 5: LR schedule (decay window → min_lr → peak LR) — highest leverage
     → Combined validation of all winning convergence tweaks
     → Update state: optimal_eval_interval, eval_overhead_seconds, lr_decay_iters, min_lr, lr

  9.6 COMM TUNING (always runs — explores all topologies including pure DP)
     → Execute actions/comm-tuning.md
     → Part A: NCCL/RCCL parameter tuning + DeepEP (if EP > 1)
     → Part B: TraceLens-guided overlap optimization (always runs; overlap metric is advisory context)

 10. PRE-SWEEP REVIEW
     → === REVIEW CHECKPOINT RC-5 ===
     → Execute actions/sweep.md (Tier 2 trials for remaining param sweep)
     → === REVIEW CHECKPOINT RC-6 ===

 11. REPORT
     → Execute actions/report.md:
       a. Run Tier 3: run_mlperf_trial "final" 3
          - Do NOT interrupt. Wait for training to naturally finish.
          - Training ends when eval_loss reaches 3.34 (status=success)
            or all iterations are exhausted (status=aborted).
       b. Re-profile optimized config (Tier 1, 10 iters)
       c. TraceLens comparative analysis (baseline vs optimized)
       d. === REVIEW CHECKPOINT RC-7 ===
          - Verify run_stop status from MLLOG
          - Extract time-to-train (TTT) as the primary result
          - Extract final eval_loss
       e. Generate optimization report with TTT comparison vs baseline
          and TraceLens before/after kernel analysis

 12. KNOWLEDGE HOOK
     → Ingest any new knowledge discovered during the run via kb_ingest.py
```

## Convergence Gate Protocol (CRITICAL)

**Every action that modifies training config** must pass the convergence gate:

1. **Quick check (Tier 1):** Run 100-iteration trial. Verify loss is not NaN/Inf and ms/iter
   is measurable. If the `trial_monitor.py` emits `[ALERT] NaN`, REVERT immediately.
2. **Check loss trajectory:** The `[ITER N/100] loss=X.XX` progress lines must show decreasing loss.
3. **Convergence validation (Tier 2):** For winners (gain > 1%) or high-risk changes
   (FP8 flags), run a 500-iteration trial with eval enabled. Check that eval_loss
   is on track toward 3.34.
4. **Long convergence comparison (Tier 2L):** For config selection (GBS/LR/TP/EP/DP),
   run 2500-iteration trials per candidate. Compare loss-vs-samples curves and
   project TTT to rank candidates.
5. **Full verification (Tier 3):** Only for the final report and config selection top-2 —
   run the complete training to confirm the target is actually reached.

**What invalidates a run:**
- `TRIAL_RESULT` shows `status=nan` (NaN/Inf detected by trial_monitor)
- `TRIAL_RESULT` shows `status=no_data` (zero iterations completed)
- Validation loss diverging or not converging toward 3.34
- MLPerf compliance check failure (:::MLLOG lines missing or malformed)

**Actions that do NOT need convergence gate:** setup, classify, profile (read-only), 
kernel-opt (same config, different kernel code), runtime-tunables (system-level only).

## Trial Tier Protocol

All training runs during optimization use one of four trial tiers. This enables rapid
iteration without waiting for full MLPerf runs.

### Tier 1: Quick Trial (ms/iter measurement)

| Parameter | Value |
|-----------|-------|
| `PRIMUS_TRAIN_ITERS` | 100 |
| `PRIMUS_EVAL_INTERVAL` | 10000 (suppress eval) |
| `MLLOG_TRAIN_LOSS_LOG_FREQ` | **1** (every iteration — CRITICAL for short trials) |
| `stderr_sink_level` | WARNING (quiet YAML) |
| `log_interval` | 999999 (quiet YAML) |
| Timeout | 60 minutes |
| Output | ms/iter, loss trajectory, `TRIAL_RESULT` line |

**Use for:** Initial measurement of any config change, fusion flag testing, crash detection.

### Tier 2: Convergence Trial

| Parameter | Value |
|-----------|-------|
| `PRIMUS_TRAIN_ITERS` | 500 |
| `PRIMUS_EVAL_INTERVAL` | 50 (triggers multiple evals) |
| `MLLOG_TRAIN_LOSS_LOG_FREQ` | **1** |
| `stderr_sink_level` | WARNING (quiet YAML) |
| `log_interval` | 999999 (quiet YAML) |
| Timeout | 120 minutes |
| Output | ms/iter, loss curve, eval_loss, `TRIAL_RESULT` line |

**Use for:** Validating winners, hyperparameter tuning, FP8 stability checks, sweep.

### Tier 2L: Long Convergence Trial (Config Selection)

| Parameter | Value |
|-----------|-------|
| `PRIMUS_TRAIN_ITERS` | 2500 |
| `PRIMUS_EVAL_INTERVAL` | 50 (triggers dense eval curve) |
| `MLLOG_TRAIN_LOSS_LOG_FREQ` | **1** |
| `stderr_sink_level` | WARNING (quiet YAML) |
| `log_interval` | 999999 (quiet YAML) |
| Timeout | 4 hours (14400s) |
| Output | ms/iter, loss-vs-samples curve, eval_loss trajectory, projected TTT, `TRIAL_RESULT` line |

**Use for:** Comparing GBS/LR/TP/EP/DP candidates. At GBS=32, 2500 iters = 80,000 samples
(~6 eval cycles). At GBS=16, 2500 iters = 40,000 samples (~3 eval cycles). Enough to
compare loss-vs-samples curves, detect FP8/parallelism numerical drift, and project TTT.

### Tier 2.5: Convergence Projection Trial

| Parameter | Value |
|-----------|-------|
| `PRIMUS_TRAIN_ITERS` | 1500 |
| `PRIMUS_EVAL_INTERVAL` | 50 (triggers multiple evals for curve fitting) |
| `MLLOG_TRAIN_LOSS_LOG_FREQ` | **1** |
| `stderr_sink_level` | WARNING (quiet YAML) |
| `log_interval` | 999999 (quiet YAML) |
| Timeout | 3 hours (10800s) |
| Output | ms/iter, loss-vs-samples curve, eval_loss trajectory, projected TTT, `TRIAL_RESULT` line |

**Use for:** Evaluating convergence-affecting actions (GBS, LR, warmup, weight_decay,
fp8-recipe) during the DFS loop. At GBS=32, 1500 iters = 48,000 samples (~4 eval cycles
at default interval). Enough to fit a loss curve and project TTT via `project_ttt()`.
This tier is shorter than Tier 2L (2500 iters) but longer than Tier 2 (500 iters),
balancing accuracy of TTT projection against wall-time cost (~45 min per trial at
current ms/iter).

### Tier 3: Full Verification (Run to Convergence)

| Parameter | Value |
|-----------|-------|
| `PRIMUS_TRAIN_ITERS` | Original config value (do NOT override) |
| `PRIMUS_EVAL_INTERVAL` | Original config value (do NOT override) |
| `MLLOG_TRAIN_LOSS_LOG_FREQ` | 32 (original) |
| `stderr_sink_level` | DEBUG (original YAML, NOT quieted) |
| `log_interval` | 32 (original YAML, NOT quieted) |
| Timeout | **None** — run must complete naturally |
| Exit condition | `run_stop` with `status=success` (target reached) or all iterations exhausted |
| Output | **time-to-train (TTT)**, final eval_loss, full MLLOG compliance log |

**Use for:** Final report generation AND config selection top-2 verification. The training MUST run until the model either
reaches the target validation loss of 3.34 (`status=success`) or exhausts all iterations
(`status=aborted`). Do NOT interrupt or early-stop a Tier 3 run — the entire purpose is
to measure the actual time-to-target under the optimized configuration.

**After Tier 3 completes**, extract from `run_stop` MLLOG event:
- `status=success` → target reached, TTT is the primary result metric
- `status=aborted` → did not converge, report the final eval_loss and analyze why

### Tier Escalation

```
THROUGHPUT-ONLY actions (fusion-flags, kernel-opt, runtime-tunables, comm-tuning, params):
Tier 1 (100 iters)  →  gain > 1%?  →  Tier 2 (500 iters)  →  converges?  →  KEEP
                   →  gain ≤ 0%?  →  DISCARD
                   →  NaN/crash?  →  REVERT + log to KB

CONVERGENCE-AFFECTING actions (config-selection, convergence-speed, fp8-recipe-tuning):
Tier 2 (500 iters)  →  loss_efficiency < baseline × 0.85?  →  DISCARD
                   →  loss_efficiency acceptable?  →  Tier 2.5 (1500 iters)
   ↓
Tier 2.5  →  projected_ttt improved?  →  KEEP
          →  projected_ttt worse?     →  DISCARD

Config selection (GBS/LR/TP/EP/DP):
Tier 1 (100 iters)  →  crash/NaN?  →  DISCARD
   ↓ survivors
Tier 2 (500 iters)  →  loss_efficiency filter  →  DISCARD clearly worse
   ↓ survivors
Tier 2L (2500 iters)  →  rank by projected TTT  →  top 2
   ↓ top 2
Tier 3 (full run)  →  actual TTT  →  WINNER
```

### MLLOG_TRAIN_LOSS_LOG_FREQ Override (CRITICAL)

The shell config sets `MLLOG_TRAIN_LOSS_LOG_FREQ=32`, which means `train_loss` events
are only emitted at iterations divisible by 32. For short trials, this yields very few
events and makes `extract_ms_per_iter` unreliable. Tier 1, 2, and 2L MUST override this to `1`.

### Running a Trial

All actions use `run_mlperf_trial` from `scripts/common.sh`:

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "label" <tier> [train_iters] [extra_env]
```

Output is filtered through `trial_monitor.py`. Raw log is preserved at
`$RESULT_DIR/attempt_<label>_raw.log`. The last line of output is:

```
TRIAL_RESULT label=<label> ms_per_iter=<N> gbs=<N> last_loss=<N> iters=<N> status=<ok|nan|no_data>
```

### Output Filtering

Training output passes through a three-stage filter pipeline:

1. **Source reduction:** Quiet YAML (`stderr_sink_level: WARNING`, `log_interval: 999999`)
   suppresses Megatron INFO/DEBUG logs. `PYTHONWARNINGS=ignore` suppresses Python warnings.
2. **Stream filter:** `trial_monitor.py` reads stdin, passes through only :::MLLOG lines,
   RESULT lines, and errors. Replaces `train_loss` MLLOG with compact `[ITER N/M]` progress.
3. **Raw preservation:** Full unfiltered output is always saved to `*_raw.log` for debugging.

This significantly reduces log volume per trial without affecting MLPerf
compliance. All :::MLLOG lines are preserved verbatim in the raw log.

## Review Checkpoint Protocol

The orchestrator MUST pause at these checkpoints to verify state consistency. At each
checkpoint, print a `=== REVIEW CHECKPOINT RC-N ===` header and list what was verified.
If any check fails, STOP and investigate before proceeding.

| Checkpoint | After | Must Verify |
|------------|-------|-------------|
| RC-1 | setup | Env clean, symlinks valid, no stale processes, `quiet_yaml`/`restore_yaml` functional |
| RC-2 | baseline (Tier 3 full run) | `run_stop status=success` (converged), actual TTT recorded, ms/iter in expected range, GBS matches config. Compare TTT against reference 206 min. |
| RC-3 | profile | Profile trace exists, kernel candidates identified (all non-vendor kernels ranked by GPU time), compute vs comm breakdown plausible |
| RC-4 | Every 3 DFS iterations | Cumulative gain positive, no false-positive gains from noise, all reverts were clean |
| RC-4.5 | After config selection | Winning config identified, projected TTT validated by Tier 3 actual TTT, loss-vs-samples comparison documented |
| RC-5 | Before sweep | Summarize all kept optimizations, verify each individual gain, check they compose correctly |
| RC-6 | After sweep | Optimal config identified, projected TTT is <= baseline TTT |
| RC-7 | After Tier 3 completes | `run_stop` status is `success` (target 3.34 reached), TTT extracted, MLLOG compliance log complete. If `status=aborted`, analyze final eval_loss and document why target was not reached. |

## Stopping Criteria

| Condition | Action |
|-----------|--------|
| All action scores < 0.3 AND `dfs_iteration_count ≥ 12` | Proceed to sweep |
| Cumulative ms/iter gain > 15% AND `dfs_iteration_count ≥ 8` | Proceed to sweep |
| 10 consecutive discards across all actions | Proceed to sweep |
| Wall clock > 240 min total | Proceed to sweep |
| Target exceeded (gap ≤ 0%) | Proceed to sweep |
| `projected_ttt` unchanged for 3 consecutive convergence-affecting actions | Proceed to sweep |
| 2+ training crashes | Emergency stop, report partial results |

## KB Integration

Before each action, query the KB for relevant knowledge:

```bash
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B $ACTION_NAME" --top-k 5 --compact
```

**KB is advisory, not authoritative.** Prior KB entries reflect earlier runs that may
have different hardware state, driver versions, or container images. The agent MUST:
- Always run its own trial even if KB shows a prior result for the same action
- Never skip an untested action solely because KB says it was tested before
- Use KB to inform score priors (±30% adjustment) but never to zero out an action

After each action with new findings, ingest into KB:

```bash
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category $CATEGORY --model "GPT-OSS-20B" \
    --action "$WHAT_WAS_DONE" --lesson "$KEY_TAKEAWAY" \
    --tags $TAGS --gain $GAIN --status $STATUS
```

## Action Dispatch

| Action | Module | Category | Trial Tier | When |
|--------|--------|----------|-----------|------|
| Setup | [`actions/setup.md`](actions/setup.md) | — | — | Always first (RC-1) |
| Classify | [`actions/classify.md`](actions/classify.md) | — | — | Always second |
| Target Analysis | [`actions/target-analysis.md`](actions/target-analysis.md) | — | — | If target provided |
| Baseline | [`actions/baseline.md`](actions/baseline.md) | — | **Tier 3** (full run) | After classify (RC-2) |
| Profile + TraceLens | [`actions/profile.md`](actions/profile.md) | — | Tier 1 | After baseline (RC-3) |
| Fusion Flags | [`actions/fusion-flags.md`](actions/fusion-flags.md) | throughput | Tier 1 → 2 | DFS loop |
| FP8 Recipe Tuning | [`actions/fp8-recipe-tuning.md`](actions/fp8-recipe-tuning.md) | convergence | Tier 2 → 2.5 | DFS loop |
| Training Params | [`actions/params.md`](actions/params.md) | throughput | Tier 1 → 2 | DFS loop |
| Runtime Tunables | [`actions/runtime-tunables.md`](actions/runtime-tunables.md) | throughput | Tier 1 | DFS loop |
| Kernel Optimization (GEAK/OOB-Claude/OOB-Codex) | [`actions/kernel-opt.md`](actions/kernel-opt.md) | throughput | Tier 1 → 2 | DFS loop |
| Integration | [`actions/integrate.md`](actions/integrate.md) | throughput | Tier 1 → 2 | Per-kernel sub-action |
| Re-Profile | [`actions/re-profile.md`](actions/re-profile.md) | — | Tier 1 | After gain > 2% or all fusion flags |
| Config Selection | [`actions/config-selection.md`](actions/config-selection.md) | convergence | Tier 1 → 2 → 2L → 3 | After DFS loop (RC-4.5) |
| Convergence Speed | [`actions/convergence-speed.md`](actions/convergence-speed.md) | convergence | Tier 2 → 2.5 | After config selection |
| Comm Tuning | [`actions/comm-tuning.md`](actions/comm-tuning.md) | throughput | Tier 1 → 2 | After config selection (always) |
| Parameter Sweep | [`actions/sweep.md`](actions/sweep.md) | — | Tier 2 | After config selection (RC-5/6) |
| Report | [`actions/report.md`](actions/report.md) | — | Tier 3 | Always last (RC-7) |

## Reference: MLPerf Run Commands

### Local Run (inside container, no Docker wrapper)

```bash
cd /root/Hyperloom-plus-mlperf/training_optimization/mlperf
source config_MI355X_1x8x1_fp8.sh
bash setup_container_symlinks.sh
source config_MI355X_1x8x1_fp8.sh && bash run_and_time.sh
```

### Tiered Trial Run (for optimization loop)

```bash
source "$SKILL_ROOT/scripts/common.sh"

# Tier 1: Quick (100 iters, no eval, MLLOG_TRAIN_LOSS_LOG_FREQ=1)
run_mlperf_trial "test_name" 1

# Tier 2: Convergence (500 iters, eval enabled)
run_mlperf_trial "validate_name" 2

# Tier 2 with custom iters
run_mlperf_trial "validate_name" 2 300

# Tier 2L: Long convergence for config selection (2500 iters, eval enabled, 4hr timeout)
run_mlperf_trial "gbs32_ep1" 2L
run_mlperf_trial "gbs16_ep1" 2L "" "PRIMUS_GLOBAL_BATCH_SIZE=16 PRIMUS_LR=2.0e-4"

# Tier 1 with extra env vars
run_mlperf_trial "gbs64" 1 100 "PRIMUS_GLOBAL_BATCH_SIZE=64 PRIMUS_LR=5.6e-4"

# Tier 3: Full verification
run_mlperf_trial "final" 3
```

**WARNING:** Never use raw `torchrun` or `bash run_and_time.sh` directly in actions.
Always use `run_mlperf_trial` to ensure log filtering, MLLOG_TRAIN_LOSS_LOG_FREQ override,
quiet YAML, and TRIAL_RESULT output.

### Key Environment Variables

```bash
# System
DGXSYSTEM=MI355X_1x8x1
GPUS_PER_NODE=8
NNODES=1
MASTER_PORT=29501

# Paths
PRIMUS_PATH=/workspace/Primus
EXP=/root/mlperf_primus/conf/gpt_oss_20B-pretrain-fp8.yaml
DATADIR=/shared_nfs/huangwei/gpt_oss_20b/data
MODELDIR=/shared_nfs/huangwei/gpt_oss_20b/model
LOGDIR=/root/mlperf_primus/logs

# Training
PRIMUS_MICRO_BATCH_SIZE=2
PRIMUS_GLOBAL_BATCH_SIZE=32
PRIMUS_LR=4.0e-4
PRIMUS_TRAIN_ITERS=1200000
PRIMUS_FP8_RECIPE=hybrid

# MLPerf
MLLOG_TARGET_EVAL_LOSS=3.34
MLLOG_TRAIN_LOSS_LOG_FREQ=32   # Overridden to 1 for Tier 1/2 trials!
MLLOG_SUBMISSION_BENCHMARK=gpt-oss-20b

# TE/ROCm
NVTE_CK_USES_FWD_V3=1
NVTE_CK_USES_BWD_V3=1
NVTE_USE_OPTIMIZED_HIPIFIED_CAST_TRANSPOSE=1
```

## Reference: MLLOG Output Format

All metrics are emitted as structured JSON lines prefixed with `:::MLLOG`:

```json
:::MLLOG {"namespace": "", "time_ms": 1775725581676, "event_type": "POINT_IN_TIME", "key": "train_loss", "value": 11.846, "metadata": {"samples_count": 0, "lr": 3.125e-06}}
```

### Key MLLOG Events

| Event Key | Type | Meaning |
|-----------|------|---------|
| `cache_clear` | POINT_IN_TIME | Cache cleared before run |
| `init_start` / `init_stop` | INTERVAL | Initialization phase |
| `run_start` / `run_stop` | INTERVAL | Training phase; `run_stop` has `status: success/aborted` |
| `epoch_start` / `epoch_stop` | INTERVAL | Epoch boundary |
| `block_start` / `block_stop` | INTERVAL | Training block (between evals) |
| `eval_start` / `eval_stop` | INTERVAL | Evaluation phase |
| `train_loss` | POINT_IN_TIME | Per-iteration training loss; metadata has `samples_count`, `lr` |
| `eval_loss` | POINT_IN_TIME | Validation loss at eval checkpoint |
| `train_samples` | POINT_IN_TIME | Total samples processed |
| `global_batch_size` | POINT_IN_TIME | GBS logged at init |
| `gradient_accumulation_steps` | POINT_IN_TIME | GA steps logged at init |

### Extracting ms/iter from MLLOG

**IMPORTANT:** `MLLOG_TRAIN_LOSS_LOG_FREQ` must be `1` for Tier 1/2 trials.
With the default value of 32, only iterations divisible by 32 emit `train_loss` events.
`run_mlperf_trial` handles this automatically.

For manual extraction from a raw log file:

```bash
source "$SKILL_ROOT/scripts/common.sh"
extract_ms_per_iter "$RESULT_DIR/attempt_label_raw.log"
```

Or read from the `TRIAL_RESULT` line (preferred):

```
TRIAL_RESULT label=baseline ms_per_iter=1643.9 gbs=32 last_loss=9.82 iters=100 ttt=134.8 run_status=aborted status=ok
```

### Extracting Time-to-Train

```bash
source "$SKILL_ROOT/scripts/common.sh"
extract_time_to_train "$RESULT_DIR/attempt_label_raw.log"
# Output: 134.4	aborted
```

### TRIAL_RESULT Line Format

Every `run_mlperf_trial` call ends by printing a structured result line:

```
TRIAL_RESULT label=<name> ms_per_iter=<float> gbs=<int> last_loss=<float> iters=<int> ttt=<float> run_status=<str> status=<str>
```

| Field | Type | Meaning |
|-------|------|---------|
| `label` | string | Trial identifier |
| `ms_per_iter` | float | Average ms/iter (warmup-excluded) |
| `gbs` | int | Global batch size from MLLOG |
| `last_loss` | float | Final training loss value |
| `iters` | int | Number of train_loss events observed |
| `ttt` | float | Time-to-train in seconds (from `run_start` to `run_stop`, 0.0 if not available) |
| `run_status` | string | MLLOG run_stop status: `success` (target reached), `aborted` (all iters done), `unknown` |
| `status` | enum | `converged` = target reached, `ok` = normal, `nan` = NaN/Inf, `no_data` = zero iterations |

Parse in shell:

```bash
eval "$(parse_trial_result "$(grep TRIAL_RESULT $filtered_log)")"
echo "ms/iter: $TRIAL_MS_PER_ITER, status: $TRIAL_STATUS"
```

## Reference: Validation Loss Output

```
validation loss at iteration N on validation set | lm loss value: 9.753977E+00 | lm loss PPL: 1.722258E+04
```

Parse regex: `lm loss value:\s*([\d.Ee+-]+)`

## Reference: RESULT Line

```
RESULT,GPT_OSS_20B,,176,AMD,2026-04-09 09:05:12 AM
```

Format: `RESULT,<model>,<extra>,<total_seconds>,<org>,<start_time>`

## Reference: Config Override Syntax

For MLPerf, overrides are set via environment variables BEFORE sourcing the config:

```bash
export PRIMUS_GLOBAL_BATCH_SIZE=64
export PRIMUS_MICRO_BATCH_SIZE=4
export PRIMUS_LR=8.0e-4
source config_MI355X_1x8x1_fp8.sh
```

Or modify the YAML config directly for Primus-level overrides (fusion flags, MoE settings).

## Reference: Critical Lessons

1. **GBS is tunable in MLPerf** (unlike training-optimization where GBS is fixed).
   Larger GBS means fewer iterations to process the same data, but each iteration is slower.
   The optimal GBS balances iteration time vs convergence speed.

2. **Eval overhead matters.** At GBS=32, eval runs every 512 iters. Each eval takes ~30s
   (64 eval iters). Reducing eval frequency saves wall time but risks overshooting target.

3. **FP8 stability is critical.** FP8 hybrid mode can cause loss spikes or NaN with certain
   configs. Always verify convergence after FP8-related changes.

4. **DeepEP can overlap communication.** With `moe_enable_deepep=true` and
   `turbo_deepep_num_cu=64`, expert parallelism communication overlaps with compute.

5. **Sliding window attention pattern matters.** The model uses alternating sliding/full
   attention (window_size=[128,0]). Do NOT change this — it affects model quality.

6. **Port conflicts after killing runs.** Increment `--master_port` (29502, 29503, ...)
   after killing training processes.

7. **Primus patches are auto-applied.** 16 patches are applied at startup for ROCm
   compatibility (permute fusion, FP8 context, TopK router, etc.).

8. **hipBLASLt GEMMs dominate (60–70% GPU time).** Gains come from reducing everything else.

9. **Current best known TTT is 206 minutes.** This is the reference convergence time with
   the current config. The baseline run MUST re-verify this number — do not assume it.

10. **`NVTE_USE_CAST_TRANSPOSE_TRITON=1` yields ~0.2% improvement.** Confirmed prior result.
    This enables Triton-based FP8 cast+transpose kernels. Apply early in optimization.

11. **TraceLens comm-compute overlap informs tuning priority.** High overlap (>0.7) suggests
    smaller gains expected from Part B, but the agent still runs it to verify — gains from
    NCCL buffer/channel tuning can exist even with high measured overlap.

12. **Eval interval optimization saves 2–5 minutes at zero convergence risk.**
    Each eval cycle costs ~30s — reducing from 16 to 8 evals saves ~4 min.

13. **ms/iter gain alone is insufficient for convergence-affecting actions.** LR, GBS, warmup,
    and weight_decay changes can have zero ms/iter impact but dramatically change
    iterations-to-converge. Always use the three-layer evaluation (ms/iter →
    loss_efficiency → projected_ttt) for these actions. The DFS loop classifies each
    action and applies the appropriate metric path automatically.

14. **LR decay window mismatch is a common high-impact blind spot.** When `lr_decay_iters`
    is set to `train_iters` (e.g. 1.2M) but convergence occurs at ~7200 iters, cosine
    decay barely reduces LR during actual training (essentially flat). However, setting
    `lr_decay_iters` too close to convergence iters (1.0–1.2×) is equally problematic:
    the cosine schedule reaches near-`min_lr` by convergence, starving the model of
    learning capacity in the final push toward the loss target. Set `lr_decay_iters` to
    **2–5×** the projected convergence iters (LR at convergence ≈ 55–91% of peak). This
    provides meaningful annealing without late-stage stall. After fixing the decay window,
    tune `min_lr` in both directions (lower for deeper annealing, higher to preserve
    late-stage capacity with tight windows), then try increasing peak LR.

15. **LR schedule knobs must be explored sequentially, not independently.** The three
    sub-dimensions (decay window → min_lr → peak LR) depend on each other. Always tune
    the decay window first, then adjust min_lr, then try increasing peak LR. Testing
    peak LR without first fixing a mismatched decay window will produce misleading results
    (higher LR with flat decay = instability; higher LR with proper decay = faster convergence).

## Reference: Process Management

- **Kill lingering processes:** `pkill -9 -f "train.py"` and `pkill -9 -f "torchrun"` before retrying.
- **Wait 5+ seconds** between kill and relaunch.
- **Increment `MASTER_PORT`** if the previous port is still bound.
- **Symlinks must be set** before each run: `bash setup_container_symlinks.sh`

## Reference: Training Metrics

| Metric | Unit | Meaning |
|--------|------|---------|
| `time_to_train` | seconds | Wall time from run_start to run_stop (primary metric) |
| `ms_per_iter` | ms | Milliseconds per training iteration (derived from MLLOG) |
| `train_loss` | float | Cross-entropy training loss per iteration |
| `eval_loss` | float | Validation log perplexity (target: 3.34) |
| `samples_count` | int | Total training samples processed |
| `lr` | float | Current learning rate |

## Reference: File Paths

```
/root/Hyperloom-plus-mlperf/training_optimization/mlperf/
├── config_MI355X_1x8x1_fp8.sh         # Shell config (env vars)
├── conf/gpt_oss_20B-pretrain-fp8.yaml  # Training YAML config
├── src/train.py                         # Training entry point
├── setup_container_symlinks.sh          # Symlink setup
├── run_and_time.sh                      # Training launcher
├── runtime_tunables.sh                  # System tuning (CPU, hugepages)
├── dev/                                 # Development scripts
└── patches/                             # Megatron/Primus patches
```
