# MLPerf Training Optimization Skill

Autonomous DFS-guided training optimization for the MLPerf GPT-OSS-20B benchmark on AMD MI355X GPUs. Uses heuristic-scored depth-first search to systematically explore optimization actions (fusion flags, parallelism configs, training params, kernel optimization) and minimize time-to-train (TTT) with convergence correctness as a hard constraint.

| File | Description |
|------|-------------|
| `SKILL.md` | Main skill — DFS orchestrator with tiered trial system and review checkpoints |
| `prompt_for_MLperf.MD` | Quick-start prompt template for Cursor chat |
| `Readme.MD` | This file — setup guide and usage |
| `actions/*.md` | 14 self-contained action modules (setup, baseline, profile, fusion-flags, etc.) |
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
- **Container image**: `harbor.oci-slc.primus-safe.amd.com/custom/tasimage/primus:202604070309`
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
  harbor.oci-slc.primus-safe.amd.com/custom/tasimage/primus:202604070309 \
  bash
```

### 2. Configure API keys

**HuggingFace Token** — required for model/data access. Edit `training_optimization/mlperf/config_MI355X_1x8x1_fp8.sh` line 29:

```bash
export HF_TOKEN="your_api_key"      # ← Replace with your HuggingFace API token ,W&B key is not compulsory
```

**SAFE API Key** — required for MCP server access (GEAK, OOB, etc.). Edit `.cursor/mcp.json` and replace all `<SAFE_API_KEY>ak-xxxxx` placeholders with your actual SAFE API key:

```json
{
  "mcpServers": {
    "oci-geak-agent": {
      "headers": {
        "Authorization": "Bearer ak-YOUR_ACTUAL_KEY_HERE"
      },
      ...
    }
  }
}
```

Two MCP servers require the key: `oci-geak-agent` and `oci-oob-agent`. TraceLens now runs via local CLI (`pip install -e /hyperloom/TraceLens-internal`).

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
 │ Baseline     │  Tier 1 trial (100 iters) → establish ms/iter
 │ (RC-2)       │
 └──────┬───────┘
        ▼
 ┌──────────────┐
 │ Profile      │  Tier 1 trial with PyTorch profiler → kernel breakdown
 │ (RC-3)       │
 └──────┬───────┘
        ▼
 ┌──────────────────────────────────────────────────┐
 │  DFS Optimization Loop                           │
 │                                                  │
 │  For each candidate action (sorted by heuristic):│
 │  1. Run Tier 1 trial (100 iters) → ms/iter      │
 │  2. If gain > 1%: validate with Tier 2 (500 iters)│
 │  3. Check convergence: loss trajectory + eval    │
 │  4. KEEP or REVERT                               │
 │  5. Every 3 iterations: RC-4 checkpoint          │
 │                                                  │
 │  Actions tried:                                  │
 │  • Fusion flags (fused router, turbo attention)  │
 │  • Parallelism (EP=8, DeepEP)                    │
 │  • Training params (MBS, precision knobs)        │
 │  • Hyperparameters (GBS, LR scaling)             │
 │  • Runtime tunables (NCCL, HSA, CPU governor)    │
 │  • Kernel optimization (GEAK)                    │
 └──────────────┬───────────────────────────────────┘
                ▼
 ┌──────────────────┐
 │ Sweep (RC-5/RC-6)│  GBS × LR grid search with Tier 2 trials
 └──────┬───────────┘
        ▼
 ┌──────────────────┐
 │ Final Run (RC-7) │  Tier 3: full training to convergence (target 3.34)
 │                  │  NO early stop — wait for run_stop status=success
 └──────┬───────────┘
        ▼
 ┌──────────────┐
 │ Report       │  Optimization report with TTT, all attempts, recommendations
 └──────────────┘
```

## Trial Tier System

All training runs use `run_mlperf_trial` from `scripts/common.sh`, never raw `torchrun` or `bash run_and_time.sh`.

| Tier | Iterations | Eval | Timeout | Use Case |
|------|-----------|------|---------|----------|
| **Tier 1** (Quick) | 100 | Suppressed | 10 min | Initial measurement, crash detection |
| **Tier 2** (Convergence) | 500 | Enabled | 60 min | Validate winners, hyperparameter tuning |
| **Tier 3** (Full) | Original config | Original | None | Final report — run to convergence |

Key features:
- `MLLOG_TRAIN_LOSS_LOG_FREQ=1` override for Tier 1/2 (fixes a bug where `=32` produces zero `train_loss` events in short trials)
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
| `.cursor/mcp.json` | Replace `<SAFE_API_KEY>ak-xxxxx` | MCP server auth (GEAK, OOB Agent, SAFE) |

## MLPerf Compliance

This skill preserves full MLPerf compliance:
- All `:::MLLOG` lines are passed through verbatim to raw logs
- `run_and_time.sh` generates the `RESULT,` line untouched
- Tier 3 (final) runs use original config values — no overrides
- The raw log from the Tier 3 run is suitable for MLPerf submission

## Tips

- **First run**: use the prompt from `prompt_for_MLperf.MD` — it has all paths pre-configured
- **Port conflicts**: `run_mlperf_trial` auto-increments `MASTER_PORT` via `next_port()`
- **Stale processes**: `kill_training` in `common.sh` cleans up `train.py` / `torchrun` remnants
- **YAML safety**: quiet config is always backed up and restored after trials
- **Knowledge base**: the agent queries `kb/entries.jsonl` before each action to avoid known pitfalls
- **Review checkpoints**: 7 mandatory checkpoints (RC-1 through RC-7) ensure consistency throughout the pipeline
