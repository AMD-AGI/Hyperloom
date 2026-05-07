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
with status `success`, reaching validation loss target).
**Hard constraint:** validation log perplexity must reach `MLPERF_QUALITY_TARGET` (3.34) — run is invalid otherwise.
**Hard constraint:** MLPerf compliance logging must be preserved (`:::MLLOG` format).
**Secondary metric:** `ms_per_iter` (derived from consecutive MLLOG train_loss timestamps).
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

## Iron Rules (non-negotiable)

These rules apply to every run. Violating any invalidates the optimization run.

### IR-1: ALWAYS use `run_mlperf_trial`, never raw `torchrun` or `run_and_time.sh` in actions

Every training trial launched by an optimization action MUST go through `run_mlperf_trial`
(in `scripts/common.sh`). The wrapper guarantees log filtering, `MLLOG_TRAIN_LOSS_LOG_FREQ`
override (IR-3), quiet YAML, and the structured `TRIAL_RESULT` output line. Direct
`torchrun` / `bash run_and_time.sh` calls bypass all four and produce unparseable trials.
Full command catalog in [`REFERENCE.md`](REFERENCE.md).

### IR-2: Tier 4 baseline and final runs MUST NOT be interrupted

Tier 4 runs must complete naturally — until `run_stop status=success` (target reached) or
all iterations are exhausted (`status=aborted`). Do NOT early-stop. Do NOT swap in Tier 1/2/3
in place of a Tier 4 baseline. The entire purpose of Tier 4 is to measure real time-to-target
under the chosen configuration.

### IR-3: `MLLOG_TRAIN_LOSS_LOG_FREQ` must be `1` for all short trials

The shell config defaults to `MLLOG_TRAIN_LOSS_LOG_FREQ_DEFAULT` (32), which emits
`train_loss` only at iterations divisible by 32. At Tier 1 (100 iters) this yields ~3 events
and makes `extract_ms_per_iter` unreliable. Tiers 1, 2, and 3 MUST force it to `1`.
`run_mlperf_trial` handles this automatically — do not bypass. Only Tier 4 keeps the default.

### IR-4: NEVER zero out action scores from KB history alone

KB entries are advisory. A negative KB result may reduce an action's
`expected_time_reduction_pct` by at most `KB_PENALTY_CAP` (30%). Never remove an action from
the stack solely because KB shows a prior null/negative outcome. The agent MUST still run
its own trial — hardware state, driver version, and container image may differ. Score floor
for `kernel-opt` is `SCORE_FLOOR_KERNEL_OPT` (0.5).

### IR-5: Kill stale processes and increment `MASTER_PORT` before relaunching

Before every trial relaunch: `pkill -9 -f "train.py"` and `pkill -9 -f "torchrun"`, wait
5+ seconds, then increment `MASTER_PORT` (29502, 29503, …). Re-run
`bash setup_container_symlinks.sh`. Skipping any of these causes Exit 144, port conflicts,
or stale mount errors.

### IR-6: Do NOT modify sliding window attention configuration

The model uses alternating sliding/full attention (`window_size=[128,0]`). Changing this
pattern alters model quality beyond MLPerf tolerance and invalidates the run. SWA is a
structural property of the benchmark, not a tunable knob.

### IR-7: Preserve MLPerf compliance logging

All `:::MLLOG` lines must be emitted verbatim to the raw log. Missing or malformed MLLOG
entries (init/run/epoch/block boundaries, `train_loss`, `eval_loss`, `run_stop`) fail
MLPerf compliance and invalidate the run. `trial_monitor.py` is a pass-through filter for
MLLOG — never suppress it.

### IR-8: NEVER modify GEAK or OOB MCP configuration

GEAK and OOB are external services — treat them as **read-only infrastructure**. The skill
MUST NOT:

- Modify GEAK/OOB server config, workspace settings, or API configuration
- Call `geak_set_model_config` (the LLM backend is pre-configured; changing it breaks all tasks)
- Write to or alter files under GEAK/OOB config directories
- Modify `cursor_mcp_config.json` or MCP server URLs (setup's probe only auto-heals auth propagation)

The only interaction allowed is the documented MCP tool calls listed in
`## MCP Server References`.

### IR-9: Respect MLPerf GPT-OSS-20B hyperparameter bounds (closed division)

The MLPerf Training 5.1 / v6.0 rules for `gpt_oss_20b` + AdamW fix most hyperparameters.
Only four are `unconstrained` (tunable); the rest are either hard-fixed or **derived** from
tunable ones. Violating any invalidates the submission.

| Parameter | Status | Fixed value / formula | Primus variable |
|-----------|--------|------------------------|------------------|
| `global_batch_size` | **unconstrained** | any | `PRIMUS_GLOBAL_BATCH_SIZE` |
| `gradient_accumulation_steps` | **unconstrained** | derived: `GBS / (MBS × DP)` | (computed) |
| `opt_learning_rate_warmup_steps` | **unconstrained** | any | `PRIMUS_LR_WARMUP_ITERS` |
| `opt_base_learning_rate` | **unconstrained** | any | `PRIMUS_LR` |
| `opt_end_learning_rate` | **derived** | `opt_base_learning_rate × 0.1` | `PRIMUS_MIN_LR` (must = `PRIMUS_LR × 0.1`) |
| `opt_learning_rate_decay_steps` | **derived** | `1_200_000 − opt_learning_rate_warmup_steps` | `PRIMUS_LR_DECAY_ITERS` (must = `PRIMUS_TRAIN_ITERS − PRIMUS_LR_WARMUP_ITERS`) |
| `opt_adamw_beta_1` | **fixed** | `0.9` | (YAML `adam_beta1`) |
| `opt_adamw_beta_2` | **fixed** | `0.95` | (YAML `adam_beta2`) |
| `opt_adamw_epsilon` | **fixed** | `1e-5` | (YAML `adam_eps`) |
| `opt_gradient_clip_norm` | **fixed** | `1.0` | `PRIMUS_CLIP_GRAD` |
| `opt_adamw_weight_decay` | **fixed** | `0.1` | `PRIMUS_WEIGHT_DECAY` |
| `dropout` | **fixed** | `0.0` | (YAML) |
| `sequence_length` | **fixed** | `8192` | (YAML) |
| `max_steps` | **fixed** | `1_200_000` | `PRIMUS_TRAIN_ITERS` |

Absolute prohibitions:

- NEVER set `PRIMUS_WEIGHT_DECAY` to anything other than `0.1`.
- NEVER set `PRIMUS_CLIP_GRAD` to anything other than `1.0`.
- NEVER change `PRIMUS_MIN_LR` independently — it is a function of `PRIMUS_LR`.
  Whenever `PRIMUS_LR` changes, `PRIMUS_MIN_LR` MUST be updated to `PRIMUS_LR × 0.1`.
- NEVER change `PRIMUS_LR_DECAY_ITERS` independently — it is a function of `PRIMUS_LR_WARMUP_ITERS`.
  Whenever `PRIMUS_LR_WARMUP_ITERS` changes, `PRIMUS_LR_DECAY_ITERS` MUST be updated to
  `PRIMUS_TRAIN_ITERS − PRIMUS_LR_WARMUP_ITERS`.
- NEVER change `PRIMUS_TRAIN_ITERS` (= `max_steps` = 1,200,000).
- NEVER change AdamW betas, epsilon, dropout, or `sequence_length` anywhere (YAML or env).

Any action (DFS or ad-hoc) that proposes a trial touching the fixed/derived rows above
MUST be rejected before `run_mlperf_trial` is invoked. The sole optimization knobs on the
hyperparameter axis are: `GBS`, `LR` (with `MIN_LR = LR × 0.1` auto-followed), warmup
steps (with `DECAY_ITERS = TRAIN_ITERS − WARMUP_ITERS` auto-followed), `MBS` (which only
shifts `GA` while keeping GBS constant), and eval interval (a measurement-cadence knob,
not a training hyperparameter).

## Constants

All thresholds, iteration counts, timeouts, and magic numbers used by the orchestrator are
defined here. All other sections reference these by name. User overrides are allowed only at
prompt level.

### Heuristic / scoring

| Constant | Value | Description |
|----------|-------|-------------|
| `KB_PENALTY_CAP` | 0.30 | Max fractional reduction of `expected_time_reduction_pct` from KB negative history (IR-4) |
| `LOSS_EFFICIENCY_DISCARD_RATIO` | 0.85 | Discard convergence-affecting candidate if `loss_efficiency < baseline × ratio` (Layer 2) |
| `TTT_GAIN_KEEP_THRESHOLD_PCT` | 0.0 | Keep a convergence-affecting action only if `ttt_gain_pct > threshold` (Layer 3) |
| `CONVERGENCE_GATE_TRIGGER_PCT` | 1.0 | For throughput-only actions, gain above this triggers Tier 2 convergence check |
| `REPROFILE_TRIGGER_PCT` | 2.0 | Any action with measured gain above this pushes a re-profile onto the stack |
| `SCORE_FLOOR_KERNEL_OPT` | 0.5 | `kernel-opt` score never falls below this, even under repeated discards |
| `MIN_DFS_ITERATIONS` | 12 | Stopping criteria only apply after at least this many DFS iterations |
| `REVIEW_CHECKPOINT_INTERVAL` | 3 | Print RC-4 every N DFS iterations |

### Trial tiers

| Constant | Value | Description |
|----------|-------|-------------|
| `TIER1_ITERS` | 100 | Quick trial — ms/iter measurement, no eval |
| `TIER2_ITERS` | 500 | Convergence trial — eval enabled |
| `TIER3_ITERS` | 2500 | Long convergence trial — config selection + projected TTT |
| `TIER1_TIMEOUT_MIN` | 60 | Tier 1 wall-clock cap |
| `TIER2_TIMEOUT_MIN` | 120 | Tier 2 wall-clock cap |
| `TIER3_TIMEOUT_S` | 14400 | Tier 3 wall-clock cap (4 hours); Tier 4 uses original YAML and has no timeout |

### Stopping criteria

| Constant | Value | Description |
|----------|-------|-------------|
| `STOP_ALL_SCORES_BELOW` | 0.3 | All remaining scores below this + `dfs_iteration_count ≥ MIN_DFS_ITERATIONS` → sweep |
| `STOP_GAIN_PCT` | 15.0 | Cumulative ms/iter gain above this + ≥ `STOP_GAIN_MIN_ITERS` iterations → sweep |
| `STOP_GAIN_MIN_ITERS` | 8 | Min iterations before `STOP_GAIN_PCT` activates |
| `STOP_CONSECUTIVE_DISCARDS` | 10 | Consecutive discards across all actions → sweep |
| `STOP_WALL_CLOCK_MIN` | 240 | Total wall-clock budget before forced sweep |
| `STOP_UNCHANGED_TTT_COUNT` | 3 | Consecutive convergence-affecting actions without `projected_ttt` change → sweep |

### Reference values

| Constant | Value | Description |
|----------|-------|-------------|
| `MLPERF_QUALITY_TARGET` | 3.34 | Validation log perplexity target (`run_stop` success condition) |
| `REFERENCE_TTT_MIN` | 206 | Prior best-known TTT in minutes; baseline must re-verify |
| `DEFAULT_GBS` | 32 | Default global batch size for the reference config |
| `MLLOG_TRAIN_LOSS_LOG_FREQ_DEFAULT` | 32 | Shell config default (IR-3 overrides to 1 for short trials) |

## MCP Server References

The agent MUST reference these servers by their exact names when calling `CallMcpTool`.
Connectivity is verified and auth-propagated in `actions/setup.md` Step 8.

| Server | Name in mcp.json | Auth | Role |
|--------|-------------------|------|------|
| OOB Agent | `oob-optimizer-dev` | Bearer token | Kernel opt (parallel backend) |
| GEAK | `geak` | Bearer token | Kernel opt (parallel backend) |

**TraceLens (local CLI):** `pip install -e /hyperloom/TraceLens-internal`. Commands:
`TraceLens_generate_perf_report_pytorch` + `orchestrator_prepare.py`. No MCP server needed.
If the CLI is absent, the pipeline degrades to `scripts/parse_trace.py` + category-only heuristics.

**OOB tools:** `agent_create_task`, `agent_submit_task`, `agent_get_task`, `agent_get_outputs`,
`agent_download_file`, `agent_cancel_task`.
**GEAK tools:** `geak_create_task`, `geak_submit_task`, `geak_get_task`, `geak_get_outputs`,
`geak_download_file`. **NEVER** call `geak_set_model_config` (IR-8).

MCP connectivity is probed and self-healed in [`actions/setup.md`](actions/setup.md) Step 8.
Results populate `state["mcp_status"]`, `state["mcp_tools"]`, `state["tracelens_cli_available"]`,
and per-server fallback flags (`oob_available`, `geak_available`) that downstream actions read.

## State Schema

The orchestrator maintains a state dict with these field groups:

| Group | Key fields |
|---|---|
| Model | `model_class`, `parallelism_topology`, `framework`, `gpu_count` |
| Baseline | `baseline_time_to_train`, `baseline_ms_per_iter`, `baseline_eval_loss`, `baseline_loss_efficiency`, `baseline_projected_ttt` |
| Current | `current_ms_per_iter`, `cumulative_gain_pct`, `projected_ttt` |
| Target | `target_time_to_train`, `target_gap_pct`, `target_gap_multiplier` |
| DFS | `action_stack`, `completed_actions`, `dfs_iteration_count`, `kernel_candidates` |
| MCP | `mcp_status`, `mcp_tools`, `tracelens_cli_available`, `oob_available`, `geak_available` |
| Convergence | `optimal_eval_interval`, `eval_overhead_seconds`, `lr`, `min_lr` (= `lr × 0.1`, IR-9), `warmup_iters`, `lr_decay_iters` (= `train_iters − warmup_iters`, IR-9) |
| Counters | `total_wall_minutes`, `consecutive_discards` |

Full field list with default values and inline comments: see
[`REFERENCE.md § State Schema`](REFERENCE.md#state-schema).

## Autonomy Rules

**Execute autonomously — no human confirmation needed.** Do NOT ask the user before:

- Running any trial tier (1/2/3/4) via `run_mlperf_trial`
- Modifying the training YAML config (fusion flags, MoE settings, eval_interval)
- Killing/restarting training processes and incrementing `MASTER_PORT`
- Submitting GEAK/OOB tasks or polling their status
- Patching kernels, reverting failed changes, applying/restoring quiet YAML
- Querying and ingesting KB entries

**Autonomy means don't ask permission, NOT skip steps.** Every numbered step of the
Orchestrator Loop (1–12) is **MANDATORY**, including:

- Step 3: TARGET ANALYSIS (when target numbers are provided)
- Step 4: KB WARM-UP (reference only — never skip untested actions, see IR-4)
- Step 5: BASELINE (Tier 4 full run, see IR-2)
- Step 12: KNOWLEDGE HOOK (ingest findings even on partial runs)

Skipping any mandatory step invalidates the run. Present the **final optimization report**
to the user once Step 12 completes.

## Common Pitfalls

Recurring errors observed during MLPerf training runs. **Read before executing.**

1. **Port conflicts after killing runs.** After `pkill`, the previous `MASTER_PORT` may stay
   bound. Always increment (29502, 29503, …). Failure mode: training hangs during init with
   `Address already in use`.

2. **`MLLOG_TRAIN_LOSS_LOG_FREQ=32` breaks short-trial metrics.** Default logs every 32nd
   iteration, yielding ~3 events at Tier 1. `run_mlperf_trial` overrides to `1` for
   Tier 1/2/3 automatically (IR-3). Never bypass the wrapper. Failure mode:
   `extract_ms_per_iter` returns noisy/wrong values.

3. **Symlinks must be re-created each container boot.** Run `bash setup_container_symlinks.sh`
   before every fresh launch. Failure mode: missing model/data paths, silent fallback to empty dirs.

4. **FP8 hybrid mode can NaN silently.** After any FP8-related change, Tier 2 convergence
   check is mandatory. Failure mode: `trial_monitor.py` emits `[ALERT] NaN`, or eval loss
   plateaus high.

5. **Primus auto-applies 16 patches at startup.** Do NOT manually re-apply or revert ROCm
   compatibility patches (permute fusion, FP8 context, TopK router, etc.). Failure mode:
   double-patching corrupts the Megatron graph.

6. **hipBLASLt GEMMs dominate 60–70% of GPU time.** Gains come from reducing *everything else*
   (permute, comm, eval overhead, fusion). Treating GEMMs as the optimization target wastes
   effort — they are vendor-tuned and GEAK cannot beat them.

## Architecture

```
SKILL.md (this file)           — DFS orchestrator: loop, heuristic, dispatch
REFERENCE.md                    — Operational reference: env vars, MLLOG format, run commands
actions/*.md                    — Self-contained action modules (17 actions)
kernel-opt/                     — Per-backend kernel optimization references (geak, oob-claude, oob-codex)
scripts/common.sh               — Tiered trial runner + metric extraction helpers
scripts/parse_trace.py          — Trace analysis (operator summary, kernel categorization, heuristics)
scripts/trial_monitor.py        — Stdin log filter + progress display + anomaly detection
scripts/apply_quiet_config.sh   — YAML quiet/restore for noise reduction
scripts/run_baseline.sh         — Standalone baseline script
scripts/run_sweep.sh            — GBS × LR sweep script
scripts/run_trial.sh            — CLI wrapper for run_mlperf_trial
scripts/run_profile.sh          — Profiling run script
kb/                             — RAG knowledge base (JSONL + query/ingest scripts)
```

## DFS Search Tree

**Phases:** SETUP → CLASSIFY → TARGET ANALYSIS (optional) → BASELINE (Tier 4) → PROFILE +
TRACELENS → HEURISTIC SCORING → DFS LOOP (pick highest-scored action → execute → re-score →
repeat) → CONFIG SELECTION → CONVERGENCE SPEED → COMM TUNING → SWEEP → REPORT.

**DFS loop actions:** fusion-flags, params, runtime-tunables, kernel-opt, integrate,
fp8-recipe-tuning, comm-tuning, re-profile — scored by heuristic, popped highest-first.
Each action can push sub-actions (PROFILE pushes kernel candidates; FUSION-FLAGS pushes
combination tests; any action with gain > `REPROFILE_TRIGGER_PCT` pushes a re-profile).
The agent explores depth-first along the most promising branch and backtracks if scores shift.

The agent is NOT limited to the pre-defined actions. If profiling reveals an unexpected
bottleneck or the KB suggests a novel technique, the agent can create ad-hoc actions and
score them with the same heuristic.

This is a single-agent sequential loop. Each action runs to completion before re-scoring.
Parallelism is within actions (e.g., GEAK submits multiple kernels concurrently), not
between them.

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
| `convergence_risk` | Likelihood of failing to reach `MLPERF_QUALITY_TARGET` | 0.0–1.0 |
| `crash_risk` | From KB (parallelism changes = 0.3, env vars = 0.05) | 0.0–1.0 |
| `target_gap_multiplier` | `1 + min(target_gap_pct, 100) / 100` | 1.0–2.0 |

## Action Classification

Every action is classified as either **throughput-only** or **convergence-affecting**.
This determines which evaluation layers apply during the DFS loop.

| Category | Actions | Evaluation Path |
|----------|---------|-----------------|
| throughput-only | fusion-flags, kernel-opt, runtime-tunables, comm-tuning, params (MBS/recompute/overlap) | Layer 1 only (ms/iter via Tier 1) |
| convergence-affecting | config-selection, convergence-speed, fp8-recipe-tuning | Layer 2 (loss_efficiency via Tier 2) + Layer 3 (projected_ttt via Tier 3) |

**Throughput-only** actions change kernel efficiency, communication overhead, or system-level
settings without altering training dynamics. Their effect is fully captured by ms/iter.

**Convergence-affecting** actions change the loss trajectory — they may not change ms/iter at
all but can dramatically shift iterations-to-converge. Examples: LR halving has zero ms/iter
impact but may save 500 iterations; GBS changes affect both ms/iter and sample efficiency.

The `classify_action()` helper in `scripts/common.sh` implements this classification.

## Three-Layer Evaluation Metric

The DFS loop uses a three-layer metric system instead of the single-dimensional ms/iter gain:

**Layer 1 — ms/iter gain** (throughput-only):
```
gain_pct = (baseline_ms_per_iter - new_ms_per_iter) / baseline_ms_per_iter × 100
```
Measured via Tier 1. Positive = faster. Sole metric for throughput-only actions.

**Layer 2 — loss_efficiency** (convergence-affecting, quick filter):
```
loss_efficiency = (first_eval_loss - last_eval_loss) / wall_time_seconds
```
Measured via Tier 2. Discard if
`candidate_efficiency < baseline_efficiency × LOSS_EFFICIENCY_DISCARD_RATIO`.

**Layer 3 — projected_ttt** (convergence-affecting, final decision):
```
ttt_gain_pct = (baseline_projected_ttt - candidate_projected_ttt) / baseline_projected_ttt × 100
```
Measured via Tier 3. Uses `project_ttt()` to extrapolate from the loss-vs-samples curve.
Keep only if `ttt_gain_pct > TTT_GAIN_KEEP_THRESHOLD_PCT`.

**Composite gain for score updates:**
- Throughput-only: `gain = ms_per_iter_gain_pct`
- Convergence-affecting: `gain = ttt_gain_pct` (Layer 3)

### Initial Score Priors (GPT-OSS-20B MoE on MI355X)

| Action | Score | Rationale |
|--------|-------|-----------|
| fusion-flags | **9** | Highest impact for MoE: permute fusion, GA fusion |
| config-selection | **8** | GBS/LR/TP/EP/DP — biggest convergence-speed lever |
| convergence-speed | **7** | Peak LR + warmup steps + eval interval (only IR-9-allowed knobs; `min_lr` and `lr_decay_iters` are derived, NOT independent) |
| fp8-recipe-tuning | **6** | FP8 knobs affect both ms/iter and convergence |
| comm-tuning | **6** | NCCL/RCCL/DeepEP/AllReduce + TraceLens-guided overlap (always runs, post-config-selection) |
| runtime-tunables | 5 | System-level knobs (NUMA, hugepages, NCCL) |
| params (training) | 5 | MBS, recompute, overlap flags |
| kernel-opt (GEAK/OOB-Claude/OOB-Codex) | 5 | All non-vendor kernels explored; floor `SCORE_FLOOR_KERNEL_OPT` |
| sweep | 1 | Final exploration of operating points |
| re-profile | — | Not scored independently; triggered by score update rules |

### Score Update Rules

After every action completes, the orchestrator re-scores the remaining `action_stack`
per 12 rules covering success/failure boosts and penalties, combined-test push-down,
re-profile triggers, kernel-opt floors, KB penalty caps, FP8 escalation, convergence
state updates, the stopping criterion, and sequential LR-schedule exploration. Full
ordered rule list with exact multipliers: see
[`REFERENCE.md § Score Update Rules`](REFERENCE.md#score-update-rules).

## Orchestrator Loop

The full `PROCEDURE optimize()` pseudocode (Setup → Classify → Target → KB Warm-up →
Baseline → Profile + TraceLens → Build stack → DFS → Config Selection → Convergence
Speed → Comm Tuning → Sweep → Report → Knowledge Hook) is in
[`REFERENCE.md § Orchestrator Loop (Full)`](REFERENCE.md#orchestrator-loop-full).

Critical loop invariants:

- Each phase runs to completion before the next starts (single-agent sequential).
- DFS inner loop (Step 8) re-scores all remaining actions after every action completes,
  following the Score Update Rules above.
- Review checkpoints RC-1..RC-7 gate progression — see § Review Checkpoint Protocol below.

**What invalidates a run:** `TRIAL_RESULT` with `status=nan` or `status=no_data`,
validation loss diverging or failing to reach `MLPERF_QUALITY_TARGET`, or missing/malformed
`:::MLLOG` lines (IR-7).

**Actions exempt from the convergence gate:** setup, classify, profile (read-only),
kernel-opt (same config, different kernel code), runtime-tunables (system-level only).

## Trial Tier Summary

All trials use `run_mlperf_trial "label" <tier> [iters] [extra_env]` from `scripts/common.sh`.
Full per-tier parameter tables, MLLOG override details, and the output filtering pipeline
are in [`REFERENCE.md`](REFERENCE.md) § Trial Tier Details.

| Tier | Iters | Eval Interval | Timeout | Purpose |
|------|-------|---------------|---------|---------|
| 1    | `TIER1_ITERS` (100)    | 10000 (suppressed) | `TIER1_TIMEOUT_MIN` (60 min)  | ms/iter measurement, crash detection |
| 2    | `TIER2_ITERS` (500)    | 50                 | `TIER2_TIMEOUT_MIN` (120 min) | Convergence validation, sweep, FP8 stability |
| 3    | `TIER3_ITERS` (2500)   | 50                 | `TIER3_TIMEOUT_S` (4 hr)      | Config selection + projected TTT |
| 4    | Original YAML          | Original YAML      | None (natural end, IR-2)      | Baseline + final verification; TTT result |

Tier 1/2/3 force `MLLOG_TRAIN_LOSS_LOG_FREQ=1` (IR-3). Tier 4 keeps the original value
(`MLLOG_TRAIN_LOSS_LOG_FREQ_DEFAULT`). Only Tier 4 uses the full-verbosity YAML; shorter
tiers use quiet YAML.

### Tier Escalation

```
THROUGHPUT-ONLY (fusion-flags, kernel-opt, runtime-tunables, comm-tuning, params):
  Tier 1 → gain > CONVERGENCE_GATE_TRIGGER_PCT? → Tier 2 → converges? → KEEP
         → gain ≤ 0%? → DISCARD
         → NaN/crash? → REVERT + log to KB

CONVERGENCE-AFFECTING (config-selection, convergence-speed, fp8-recipe-tuning):
  Tier 2 → loss_efficiency < baseline × LOSS_EFFICIENCY_DISCARD_RATIO? → DISCARD
         → acceptable? → Tier 3 → projected_ttt improved? → KEEP, else DISCARD

CONFIG SELECTION (GBS/LR/TP/EP/DP):
  Tier 1 (crash/NaN filter) → Tier 2 (loss_efficiency filter) →
  Tier 3 (rank by projected TTT) → Tier 4 top-2 (actual TTT) → WINNER
```

### Running a Trial

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "label" <tier> [train_iters] [extra_env]
```

Output is filtered through `trial_monitor.py`; raw log preserved at
`$RESULT_DIR/attempt_<label>_raw.log`. Last line of output:

```
TRIAL_RESULT label=<name> ms_per_iter=<N> gbs=<N> last_loss=<N> iters=<N> ttt=<N> run_status=<s> status=<s>
```

Parse in shell: `eval "$(parse_trial_result "$(grep TRIAL_RESULT "$filtered_log")")"` →
exports `$TRIAL_MS_PER_ITER`, `$TRIAL_STATUS`, etc. Full field table in
[`REFERENCE.md`](REFERENCE.md) § TRIAL_RESULT Line Format.

## Review Checkpoint Protocol

The orchestrator MUST pause at these checkpoints, print `=== REVIEW CHECKPOINT RC-N ===`,
and verify. If any check fails, STOP and investigate.

| Checkpoint | After | Must Verify |
|------------|-------|-------------|
| RC-1 | setup | Env clean, symlinks valid, no stale processes, `quiet_yaml`/`restore_yaml` functional |
| RC-2 | baseline (Tier 4) | `run_stop status=success`, actual TTT recorded vs `REFERENCE_TTT_MIN`, GBS matches config |
| RC-3 | profile | Profile trace exists, kernel candidates ranked, compute vs comm breakdown plausible |
| RC-4 | Every `REVIEW_CHECKPOINT_INTERVAL` DFS iterations | Cumulative gain positive, no false-positive gains from noise, reverts clean |
| RC-4.5 | After config selection | Winning config identified, projected TTT validated by Tier 4 actual TTT |
| RC-5 | Before sweep | Kept optimizations summarized, individual gains verified, composition correct |
| RC-6 | After sweep | Optimal config identified, projected TTT ≤ baseline TTT |
| RC-7 | After Tier 4 final | `run_stop status=success`, TTT extracted, MLLOG compliance complete. If `status=aborted`, analyze and document. |

## Stopping Criteria

| Condition | Action |
|-----------|--------|
| All scores < `STOP_ALL_SCORES_BELOW` AND `dfs_iteration_count ≥ MIN_DFS_ITERATIONS` | Proceed to sweep |
| Cumulative ms/iter gain > `STOP_GAIN_PCT` AND `dfs_iteration_count ≥ STOP_GAIN_MIN_ITERS` | Proceed to sweep |
| `STOP_CONSECUTIVE_DISCARDS` consecutive discards | Proceed to sweep |
| Wall clock > `STOP_WALL_CLOCK_MIN` | Proceed to sweep |
| Target exceeded (gap ≤ 0%) | Proceed to sweep |
| `projected_ttt` unchanged for `STOP_UNCHANGED_TTT_COUNT` consecutive convergence-affecting actions | Proceed to sweep |
| 2+ training crashes | Emergency stop, report partial results |

## KB Integration

**KB is advisory, not authoritative (IR-4).** Prior entries reflect older runs with
potentially different hardware/driver/image state. Always run the trial.

Before each action:

```bash
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B $ACTION_NAME" --top-k 5 --compact
```

After each action with new findings:

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
| Baseline | [`actions/baseline.md`](actions/baseline.md) | — | **Tier 4** | After classify (RC-2) |
| Profile + TraceLens | [`actions/profile.md`](actions/profile.md) | — | Tier 1 | After baseline (RC-3) |
| Fusion Flags | [`throughput-knobs.md#fusion-flags`](actions/throughput-knobs.md#fusion-flags) | throughput | Tier 1 → 2 | DFS loop |
| FP8 Recipe Tuning | [`actions/fp8-recipe-tuning.md`](actions/fp8-recipe-tuning.md) | convergence | Tier 2 → 3 | DFS loop |
| Training Params | [`throughput-knobs.md#training-params`](actions/throughput-knobs.md#training-params) | throughput | Tier 1 → 2 | DFS loop |
| Runtime Tunables | [`throughput-knobs.md#runtime-tunables`](actions/throughput-knobs.md#runtime-tunables) | throughput | Tier 1 | DFS loop |
| Kernel Optimization | [`actions/kernel-opt.md`](actions/kernel-opt.md) | throughput | Tier 1 → 2 | DFS loop |
| Integration | [`actions/integrate.md`](actions/integrate.md) | throughput | Tier 1 → 2 | Per-kernel sub-action |
| Re-Profile | [`profile.md § Re-Profile Trigger`](actions/profile.md#re-profile-trigger) | — | Tier 1 | After gain > `REPROFILE_TRIGGER_PCT` or all fusion flags |
| Config Selection | [`actions/config-selection.md`](actions/config-selection.md) | convergence | Tier 1 → 2 → 3 → 4 | After DFS loop (RC-4.5) |
| Convergence Speed | [`actions/convergence-speed.md`](actions/convergence-speed.md) | convergence | Tier 2 → 3 | After config selection |
| Comm Tuning | [`actions/comm-tuning.md`](actions/comm-tuning.md) | throughput | Tier 1 → 2 | After config selection (always) |
| Parameter Sweep | [`actions/sweep.md`](actions/sweep.md) | — | Tier 2 | After config selection (RC-5/6) |
| Report | [`actions/report.md`](actions/report.md) | — | Tier 4 | Always last (RC-7) |

## Lessons

All decision-layer and operational lessons live in one table:
see [`REFERENCE.md § Lessons`](REFERENCE.md#lessons). Decision-layer entries (rows 1–6)
inform DFS scoring priors and action selection; operational entries (rows 7–12) guide
integration and troubleshooting.

## Reference

Operational details (env vars, MLLOG event format, run commands, training metrics reference,
full trial tier parameter tables, process management, config override syntax): see
[`REFERENCE.md`](REFERENCE.md).
