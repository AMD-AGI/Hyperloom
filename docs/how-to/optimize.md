---
myst:
    html_meta:
        "description": "Step-by-step guide to running a Hyperloom optimization. Covers launching from Cursor, monitoring, resuming, and reading output artifacts."
        "keywords": "Hyperloom, optimization, how-to, LLM inference, AMD GPU, ROCm, Local Mode, Cursor, GEAK, TraceLens, session, throughput"
---
# Run a Hyperloom optimization

This topic assumes you have already completed installation. If you haven't:

- **Hosted UI** — see [Quickstart — hosted UI](../install/quickstart.md). No
  local setup needed; launch directly from the browser.
- **Local Mode or bare-metal** — see [Install Hyperloom](../install/hyperloom-installation.md)
  first, then return here to launch your first run.

## Launch from Cursor (Local Mode)

Open the workspace printed by `local_setup.sh` in Cursor, then paste the
following prompt into Cursor Chat, filling in your workload details:

```{note}
The prompt includes `install.sh`. This is intentional: Cursor runs in its own
shell process, which does not inherit the environment you sourced during
installation. The agent must re-source the env files and re-run `install.sh` in
its own context before launching the optimizer. Because `install.sh` is
idempotent, the second run is fast and safe.
```

```text
@inference_optimizer/SKILL.md

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
source '/path/to/hyperloom-run/runtime/local-setup.env.sh'
bash inference_optimizer/scripts/install.sh
source '/path/to/hyperloom-run/runtime/kernel-agent.env.sh'
export USER_DATA_PATH='/path/to/hyperloom-run'

Requirements:
1. Report the session ID, log path, PID, and initial health check result.
2. Monitor the process every 300s until the optimization is complete or failed.
```

| Field | Meaning | How to choose |
|-------|---------|---------------|
| `TP` | Tensor-parallel size — number of GPUs the model is sharded across | Must match the number of GPUs in your server node (for example, `8` for a single 8-GPU MI300X node) |
| `CONC` | Concurrent requests — benchmark concurrency level | Start with `64`; Hyperloom sweeps other values during SWEEP phase |
| `ISL` | Input sequence length — tokens in each request's prompt | Match your production workload; `1024` is a common starting point |
| `OSL` | Output sequence length — tokens generated per response | Match your production workload; `1024` is a common starting point |

See the [README](https://github.com/AMD-AGI/Hyperloom/blob/main/README.md) for
the full prompt field reference (every field maps to a CLI flag).

## Monitor the run

The agent reports a session ID, log path, and PID, then polls until the run
completes. Under the hood it walks the phase chain
`PRELUDE → FRAMEWORK_PR → EXPLORE → KERNEL → SWEEP → CLOSE`; see
[Hyperloom optimization loop](../conceptual/optimization-loop.md) for what
happens in each phase.

## Resume an interrupted session

Paste this prompt into Cursor Chat to resume an existing session:

```text
@inference_optimizer/SKILL.md

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
| `final.throughput_tok_s_per_gpu` | Validated end-of-session throughput — the headline number |
| `final.cumulative_gain_pct_validated` | Validated gain over baseline |
| `final.action_path` | Ordered list of changes that make up the final optimized stack |

For the full schema — useful if you are building a dashboard, reporting
pipeline, or downstream integration on top of this file — see
[`session_breakdown.json` integration in Hyperloom](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/reference/session-breakdown.md).

## Troubleshooting

If a run fails on first launch, see [Troubleshooting Hyperloom](../troubleshooting.md).
