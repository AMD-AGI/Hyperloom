# How-to: Run Your First Optimization

This guide walks through a complete optimization run, end to end. It assumes you
have already followed [Installation](installation.md) and have your API key.

There are two ways to run Hyperloom: the hosted **UI** (no local setup) and
**Local Mode** (driven from Cursor against a remote AMD GPU). Both are covered
below.

## Option A — Hosted UI (PrimusClaw)

1. Open [crusoe.primus-safe.amd.com/hyperloom](https://crusoe.primus-safe.amd.com/hyperloom/).
2. Select **Claw Agent** or **Get Started** to enter PrimusClaw.
3. Pick the tab that matches your task:
   - **Hyperloom** — end-to-end model performance optimization.
   - **TraceLens-only** — performance / gap analysis and bridge planning.
   - **GEAK-only** — kernel optimization.
4. Provide your workload and launch. Jobs run in isolated sandboxes; multi-node
   workloads fan out via RayJob.

## Option B — Local Mode (Cursor), step by step

### 1. Bootstrap dependency checkouts

In your prepared GPU environment (see [Installation](installation.md)):

```bash
git clone https://github.com/AMD-AGI/Hyperloom.git
cd Hyperloom
cp .env.template .env          # then edit credentials (SAFE_API_KEY, OPENAI_BASE_URL)
export USER_DATA_PATH=/path/to/hyperloom-run
bash src/hyperloom/inference_optimizer/assets/local_setup.sh
```

When `local_setup.sh` finishes it prints the workspace path to open in Cursor,
the prompt template to paste, and the env file to source before launch.

### 2. Install runtime dependencies

Before launching the optimizer, run the install step in the same shell that will
start `inference_optimizer optimize`:

```bash
source "$USER_DATA_PATH/runtime/local-setup.env.sh"
bash src/hyperloom/inference_optimizer/assets/install.sh
source "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
```

`local_setup.sh` prepares paths and checkouts. `install.sh` installs and verifies
runtime dependencies such as Magpie, TraceLens, GEAK, Ray, and CLI auth/config
files.

### 3. Launch from Cursor

Open the printed workspace in Cursor, then paste the generated prompt into
Cursor Chat, filling in your workload:

```text
@src/hyperloom/inference_optimizer/SKILL.md

Optimize inference for this workload:
- Model: /path/to/your/model
- Framework: sglang
- GPU: MI300X
- TP: 8
- CONC: 64
- ISL: 1024
- OSL: 1024
- Goal: improve throughput by at least 10%
- Budget: 2.0 hours

Before launch, run exactly:
source '/path/to/hyperloom-run/runtime/local-setup.env.sh'
bash src/hyperloom/inference_optimizer/assets/install.sh
source '/path/to/hyperloom-run/runtime/kernel-agent.env.sh'
export USER_DATA_PATH='/path/to/hyperloom-run'

Requirements:
1. Report the session ID, log path, PID, and initial health check result.
2. Monitor the process every 300s until the optimization is complete or failed.
```

See the [README](https://github.com/AMD-AGI/Hyperloom/blob/main/README.md) for
the full prompt field reference (every field maps to a CLI flag).

### 4. Monitor the run

The agent reports a session ID, log path, and PID, then polls until the run
completes. Under the hood it walks the phase chain
`PRELUDE → FRAMEWORK_AGENT → EXPLORE → KERNEL_AGENT → SWEEP → CLOSE`; see
[How the optimization loop works](HOW_THE_OPTIMIZATION_LOOP_WORKS.md) for what
happens in each phase.

### 5. Resume an interrupted session

```text
@src/hyperloom/inference_optimizer/SKILL.md

Resume the existing Hyperloom optimization session.

Requirements:
1. Launch `inference_optimizer optimize --resume`; do not start a new session.
2. Do not pass `--model`; read the model and workload from the saved manifest.
3. Before launching, verify `manifest.json` and `state.json` exist.
4. Report the log path, PID, health check, current phase, cumulative gain, and best config.
5. Monitor the process every 300s until the optimization is complete or failed.
```

## What you get

When the loop exits, Hyperloom writes the final report, reproducible session
artifacts, and `session_breakdown.json` for downstream consumers. Delivery
systems can use those artifacts to package or review the optimized stack. For
the shape of that artifact, see
[`session_breakdown.json` integration](INTEGRATION_SESSION_BREAKDOWN.md).

## Worked examples

For full, real optimization runs with configs, patches, and measured gains, see
the case studies:

- [GLM-5 — discovering optimizations hard to spot manually](CASE_STUDY_GLM5.md)
- [DeepSeek-R1 — fast scale-up on a new workload](CASE_STUDY_DEEPSEEK_R1.md)

## Troubleshooting

If a run fails on first launch (gateway 401, Ray `--num-gpus`, VRAM issues),
see [Troubleshooting](TROUBLESHOOTING.md).
