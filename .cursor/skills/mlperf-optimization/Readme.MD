# MLPerf Training Optimization Skill

Autonomous DFS-guided training optimization for the MLPerf GPT-OSS-20B benchmark on AMD MI355X GPUs. Uses heuristic-scored depth-first search to systematically explore optimization actions (fusion flags, parallelism configs, training params, kernel optimization) and minimize time-to-train (TTT) with convergence correctness as a hard constraint.

| File | Description |
|------|-------------|
| `SKILL.md` | Main skill — DFS orchestrator with tiered trial system and review checkpoints |
| `prompt_for_MLperf.MD` | Quick-start prompt template for Cursor chat |
| `Readme.MD` | This file — setup guide and usage |
| `actions/*.md` | 14 self-contained action modules (setup, classify, target-analysis, baseline, profile, throughput-knobs, fp8-recipe-tuning, kernel-opt, integrate, config-selection, convergence-speed, comm-tuning, sweep, report) |
| `kernel-opt/` | Per-backend kernel optimization references (geak.md, oob-codex.md, oob-claude.md) |
| `scripts/common.sh` | Tiered trial runner (`run_mlperf_trial`) + metric extraction helpers |
| `scripts/trial_monitor.py` | Real-time log filter — preserves :::MLLOG, suppresses noise, tracks progress |
| `scripts/apply_quiet_config.sh` | YAML quiet/restore for reducing Megatron debug verbosity |
| `scripts/run_baseline.sh` | Standalone baseline script |
| `scripts/run_sweep.sh` | GBS x LR parameter sweep |
| `scripts/run_trial.sh` | CLI wrapper for individual trials |
| `kb/` | RAG knowledge base (JSONL + query/ingest scripts) |

---

## Prerequisites

- **GPU**: 8x AMD MI355X (1 node)
- **Container image**: `harbor.oci-slc.example-internal-host.invalid/custom/tasimage/primus:202604070309`
- **Data**: C4 pre-tokenized dataset at `/shared_nfs/huangwei/gpt_oss_20b/data`
- **Model**: GPT-OSS-20B checkpoint at `/shared_nfs/huangwei/gpt_oss_20b/model`
- **Framework**: Primus (Megatron-based), pre-installed in the container at `/workspace/Primus`

## Setup

### 1. Launch the container

```bash
docker run --rm -it \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  --shm-size=64g \
  -v /shared_nfs:/shared_nfs \
  -v /root/Hyperloom-plus-mlperf:/root/Hyperloom-plus-mlperf \
  harbor.oci-slc.example-internal-host.invalid/custom/tasimage/primus:202604070309 \
  bash
```

### 2. Configure API keys

**HuggingFace Token** — required for model/data access. Edit `training_optimization/mlperf/config_MI355X_1x8x1_fp8.sh` line 29:

```bash
export HF_TOKEN="your_api_key"      # ← Replace with your HuggingFace API token ,W&B key is not compulsory
```

**SAFE API Key** — required for MCP server access (OOB Optimizer, GEAK, TraceLens, etc.). Edit `.cursor/mcp.json` and replace all Bearer token placeholders with your actual SAFE API keys:

```json
{
  "mcpServers": {
    "oob-optimizer-dev": {
      "headers": {
        "Authorization": "Bearer ak-YOUR_OOB_KEY_HERE"
      },
      ...
    },
    "oci-geak-agent": {
      "headers": {
        "Authorization": "Bearer ak-YOUR_GEAK_KEY_HERE"
      },
      ...
    }
  }
}
```

`oob-optimizer-dev` (OOB-Claude/OOB-Codex kernel optimization) and `oci-geak-agent` (GEAK kernel optimization) both require Bearer tokens. All kernel-opt backends run in parallel. TraceLens now runs via local CLI (`pip install -e /hyperloom/TraceLens-internal`), no MCP needed. Connectivity is verified automatically in setup Step 8.

### 3. Prepare runtime directories

```bash
mkdir -p /root/mlperf_primus/logs /root/mlperf_primus/outputs /root/mlperf_primus/conf
```

### 4. Verify symlinks and data

```bash
cd /root/Hyperloom-plus-mlperf/training_optimization/mlperf
source config_MI355X_1x8x1_fp8.sh
bash setup_container_symlinks.sh
ls -la /workspace/code/src/train.py /data /model /results
```

---

## Quick Start

Reference the skill in your Cursor chat with `@.cursor/skills/mlperf-optimization/SKILL.md`, or paste the prompt template from `prompt_for_MLperf.MD`:

```
Use the mlperf-optimization skill to optimize GPT-OSS-20B MLPerf training performance.
mode: local
Configuration:
Benchmark: gpt-oss-20b (MLPerf Training 5.1.0)
Quality target: validation log perplexity = 3.34
Precision: BF16/FP8 hybrid
MLPERF_DIR: /root/Hyperloom-plus-mlperf/training_optimization/mlperf
Config script: config_MI355X_1x8x1_fp8.sh
Training YAML: conf/gpt_oss_20B-pretrain-fp8.yaml
Framework: primus (Megatron-based)
GPU type: MI355X
GPUs per node: 8
Nodes: 1
Training params:
GBS=32, MBS=2, LR=4.0e-4, seq_length=8192
TP=1, PP=1, EP=1

Requirements:
Save all results and the optimization report to /root/mlperf_results/
Execute the full skill pipeline, including parameter sweep.
Use run_mlperf_trial for ALL training runs — never call torchrun or run_and_time.sh directly.
Follow all review checkpoints.
```

---

## What It Does

The agent runs the full MLPerf optimization pipeline autonomously:

```
 ┌─────────────┐
 │ Setup (RC-1) │  Verify env, symlinks, trial infrastructure
 └──────┬──────┘
        ▼
 ┌──────────────┐
 │ Baseline     │  Tier 4 full run → establish real TTT (ref: ~206 min)
 │ (RC-2)       │
 └──────┬───────┘
        ▼
 ┌────────────────────┐
 │ Profile + TraceLens│  Tier 1 + TraceLens roofline/overlap analysis
 │ (RC-3)             │
 └──────┬─────────────┘
        ▼
 ┌──────────────────────────────────────────────────┐
 │  DFS Optimization Loop (three-layer evaluation)  │
 │                                                  │
 │  For each candidate action (sorted by heuristic):│
 │                                                  │
 │  THROUGHPUT-ONLY (fusion, kernel, runtime, comm):│
 │  1. Tier 1 (100 iters) → ms/iter gain           │
 │  2. If gain > 1%: Tier 2 convergence gate        │
 │  3. KEEP or DISCARD                              │
 │                                                  │
 │  CONVERGENCE-AFFECTING (GBS/LR/warmup/WD/FP8):  │
 │  1. Tier 2 (500 iters) → loss_efficiency filter  │
 │  2. Tier 3 (2500 iters) → projected_ttt         │
 │  3. KEEP or DISCARD based on TTT projection      │
 │                                                  │
 │  If gain > 2%: trigger RE-PROFILE                │
 │  Every 3 iterations: RC-4 checkpoint             │
 └──────────────┬───────────────────────────────────┘
                ▼
 ┌────────────────────────┐
 │ Config Selection       │  GBS/LR/TP/EP/DP selection (Tier 1→3→4)
 │ (RC-4.5)               │
 └──────┬─────────────────┘
        ▼
 ┌────────────────────────┐
 │ Convergence Speed      │  Eval interval, LR schedule, warmup optimization
 │ + Comm Tuning          │  NCCL/RCCL/DeepEP + TraceLens CLI-guided overlap (always)
 └──────┬─────────────────┘
        ▼
 ┌──────────────────┐
 │ Sweep (RC-5/RC-6)│  MBS / eval interval refinement with Tier 2 trials
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │ Final Run (RC-7) │  Tier 4: full training to convergence (target 3.34)
 │                  │  NO early stop — wait for run_stop status=success
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │ Report           │  Optimization report with TTT + TraceLens CLI before/after
 └──────────────────┘
```

## MCP Integrations

**TraceLens** (local CLI, no MCP): TraceLens profiling now runs via local CLI
(`pip install -e /hyperloom/TraceLens-internal`). Commands:
`TraceLens_generate_perf_report_pytorch` for performance reports and
`orchestrator_prepare.py` for category analysis. The report step runs TraceLens
CLI on both baseline and final traces for before/after kernel breakdown. If the
CLI is not installed, the pipeline degrades gracefully to local Python-based
kernel analysis (`parse_trace.py`).

**GEAK** (`oci-geak-agent`, Bearer token), **OOB-Claude** and **OOB-Codex**
(`oob-optimizer-dev`, Bearer token): three parallel kernel optimization backends.
All run simultaneously for each kernel candidate; the result with the highest
verified micro-benchmark speedup is selected for integration. GEAK runs on a GPU
pod (can compile and benchmark on-pod). OOB-Claude and OOB-Codex generate code
without GPU access — verification is done locally. If all backends are unreachable,
kernel-opt is skipped.

## Trial Tier System

All training runs use `run_mlperf_trial` from `scripts/common.sh`, never raw `torchrun` or `bash run_and_time.sh`.

| Tier | Iterations | Eval | Timeout | Use Case |
|------|-----------|------|---------|----------|
| **Tier 1** (Quick) | 100 | Suppressed | 60 min | Initial ms/iter measurement, crash detection |
| **Tier 2** (Convergence) | 500 | Enabled | 120 min | Loss_efficiency filter, convergence gate |
| **Tier 3** (Long) | 2500 | Enabled | 4 hr | Config selection + projected TTT |
| **Tier 4** (Full) | Original config | Original | None | Final report — run to convergence |

Key features:
- **Three-layer evaluation**: throughput-only actions use ms/iter (Tier 1); convergence-affecting actions use loss_efficiency (Tier 2) + projected_ttt (Tier 3)
- `MLLOG_TRAIN_LOSS_LOG_FREQ=1` override for Tier 1/2/3 (fixes a bug where `=32` produces zero `train_loss` events in short trials)
- Real-time progress via `trial_monitor.py`: `[ITER 5/100] loss=11.23 lr=1.6e-5 | elapsed=42s`
- Anomaly detection: NaN/Inf loss → immediate alert
- Structured output: `TRIAL_RESULT label=baseline ms_per_iter=1643.9 gbs=32 ...`

## Output

The agent produces (saved to `/root/mlperf_results/`):
- **`optimization_report.md`** — executive summary with time-to-train result, all attempts, what worked / didn't
- **`results.tsv`** — every attempt with ms/iter, gain%, keep/discard status
- **`attempt_*_raw.log`** — full raw training logs (MLLOG-compliant, suitable for submission)
- **`attempt_*.log`** — filtered logs (clean, for review)
- **Profiler traces** (`.json`) — Chrome traces for kernel analysis

## Key Files to Configure

| File | What to Change | Why |
|------|---------------|-----|
| `training_optimization/mlperf/config_MI355X_1x8x1_fp8.sh:29` | `HF_TOKEN="your_key"` | HuggingFace model/data access |
| `training_optimization/mlperf/config_MI355X_1x8x1_fp8.sh:30` | `WANDB_API_KEY="your_key"` | W&B logging (optional) |
| `.cursor/mcp.json` | Replace Bearer token values | MCP server auth (OOB and GEAK require tokens; TraceLens does not) |

## MLPerf Compliance

This skill preserves full MLPerf compliance:
- All `:::MLLOG` lines are passed through verbatim to raw logs
- `run_and_time.sh` generates the `RESULT,` line untouched
- Tier 4 (final) runs use original config values — no overrides
- The raw log from the Tier 4 run is suitable for MLPerf submission

## Tips

- **First run**: use the prompt from `prompt_for_MLperf.MD` — it has all paths pre-configured
- **Port conflicts**: `run_mlperf_trial` auto-increments `MASTER_PORT` via `next_port()`
- **Stale processes**: `kill_training` in `common.sh` cleans up `train.py` / `torchrun` remnants
- **YAML safety**: quiet config is always backed up and restored after trials
- **Knowledge base**: the agent queries `kb/entries.jsonl` before each action to avoid known pitfalls
- **Review checkpoints**: 7 mandatory checkpoints (RC-1 through RC-7) ensure consistency throughout the pipeline
