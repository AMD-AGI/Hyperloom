---
myst:
    html_meta:
        "description": "Step-by-step guide to running a Hyperloom optimization. Covers launching from Cursor, monitoring, resuming, and reading output artifacts."
        "keywords": "Hyperloom, optimization, how-to, LLM inference, AMD GPU, ROCm, Cursor, GEAK, TraceLens, session, throughput"
---
# Run a Hyperloom optimization

This topic assumes you have already completed installation. If you haven't:

- **Docker Container** — see [Docker quickstart](../install/local-mode.md) to 
  get Hyperloom running in a Docker container.
- **Bare-metal** — see [Bare-metal quickstart](../install/setup.md), then
  return here to launch your first run.

## Launch from Cursor

Open the Hyperloom workspace in Cursor, then paste the following prompt into
Cursor Chat, filling in your workload details:

```{note}
The prompt includes `install.sh`. This is intentional: Cursor runs in its own
shell process, which does not inherit the environment you sourced during
installation. The agent must re-source the env files and re-run `install.sh` in
its own context before launching the optimizer. Because `install.sh` is
idempotent, the second run is fast and safe.
```

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
- Budget: 24 hours

Before launch, run exactly:
bash src/hyperloom/inference_optimizer/assets/install.sh
source '/path/to/hyperloom-run/runtime/kernel-agent.env.sh'
export USER_DATA_PATH='/path/to/hyperloom-run'

Requirements:
1. Report the session ID, log path, PID, and initial health check result.
2. Monitor the process every 300s until the optimization is complete or failed.
```

| Field | Meaning | How to choose |
|-------|---------|---------------|
| `TP` | Tensor-parallel size — number of GPUs the model is sharded across | Must match the number of GPUs in your server node (for example, `8` for a single 8-GPU MI300X node) |
| `CONC` | Concurrent requests — baseline benchmark concurrency (`--conc`, default `8`) | Set to your target concurrency; the post-run concurrency sweep separately measures a ladder (default `1,2,4,8,16,32,64,128`) |
| `ISL` | Input sequence length — tokens in each request's prompt | Match your production workload; `1024` is a common starting point |
| `OSL` | Output sequence length — tokens generated per response | Match your production workload; `1024` is a common starting point |

See [`src/hyperloom/inference_optimizer/SKILL.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/SKILL.md)
for the full prompt field reference (every field maps to a CLI flag defined in
`cli/parser.py`).

## Monitor the run

The agent reports a session ID, log path, and PID, then polls until the run
completes. Under the hood it walks the phase chain
`PRELUDE → FRAMEWORK_AGENT → EXPLORE → KERNEL_AGENT → SWEEP → CLOSE`; see
[Hyperloom optimization loop](../conceptual/optimization-loop.md) for what
happens in each phase.

## Resume an interrupted session

Paste this prompt into Cursor Chat to resume an existing session:

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

## Output and artifacts

When the loop exits, Hyperloom writes the final report, reproducible session
artifacts, and `session_breakdown.json` to your session directory. The three
fields to read first are:

| Field | What it tells you |
|-------|-------------------|
| `final.throughput_tok_s_per_gpu` | Validated end-of-session serving throughput — the headline number for SGLang / vLLM / Atom |
| `final.cumulative_gain_pct_validated` | Validated gain over baseline |
| `final.action_path` | Ordered list of changes that make up the final optimized stack |

For scriptable diffusion workloads (`--framework xdit`), the headline metric is
`final.e2el_mean_ms` (lower is better) and reports use `img/s` as the throughput
unit.

For the full schema — useful if you are building a dashboard, reporting
pipeline, or downstream integration on top of this file — see
[`session_breakdown.json` integration in Hyperloom](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/reference/session-breakdown.md).

## Troubleshooting

If a run fails on first launch, see [Troubleshooting Hyperloom](../reference/troubleshooting.md).
