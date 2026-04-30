# Prompt Templates for `@inference-optimizer`


## 实际使用版本:
@inference-optimizer

I need to optimize Qwen3-30B-A3B throughput via this skill ,with ClaudeBackend. Do the full routine and monitoring it until the target gain is reached. 
ANTHROPIC_API_KEY=a_k-
ANTHROPIC_BASE_URL="https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy"
MODEL_PATH=/hyperloom/models/Qwen3-30B-A3B
MAX_HOURS=7
INFERENCEX_PATH=/opt/InferenceX
INFERENCE_OPTIMIZER_SESSION_ROOT=/tmp/io-marathon-test
TARGET_GAIN_PCT=100
conc=32 isl=1024 osl=1024







## Quick Smoke Test (single-proc — short runs don't need multi-CLI overhead)

```text
@inference-optimizer
MODEL_PATH=/hyperloom/models/Qwen3-30B-A3B
MAX_HOURS=0.25
INFERENCEX_PATH=/opt/InferenceX
INFERENCE_OPTIMIZER_SESSION_ROOT=/tmp/io-smoke
OPENAI_API_KEY=<set in shell>

Run a short codex-backed smoke test. Use `--transport single-proc` (the
default); 15-minute runs don't benefit from per-agent CLI isolation.
Stop if preflight fails.
```

## Marathon Test (single-proc — legacy in-process reactor model)

Use this when you want every agent as an asyncio task in one Conductor
process (default). Best for runs ≤ 6h or when you want every action
debug-traceable in one log.

```text
@inference-optimizer
MODEL_PATH=/hyperloom/models/Qwen3-30B-A3B
MAX_HOURS=7
INFERENCEX_PATH=/opt/InferenceX
INFERENCE_OPTIMIZER_SESSION_ROOT=/tmp/io-marathon-test
ANTHROPIC_API_KEY=<set in shell>
ANTHROPIC_BASE_URL=https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy

Run a Claude-backed marathon (single-proc transport). Use a shell
timeout if this is only a validation run.
```

## Multi-CLI Marathon Test (each agent as its own persistent CLI)

Every agent (executor / critic / watchdog / sage) runs as its own
`claude --print --continue` (or `codex` with explicit `conversation.jsonl`)
restart-loop process. The Conductor only owns the Router + PolicyGate +
SQLite events bus; agents communicate via JSONL inbox/outbox files
under `$SESSION_DIR/agents/<name>/`. Use this for runs >= 6h to avoid
the 8-12h context-window exhaustion seen in long single-proc sessions.

```text
@inference-optimizer
MODEL_PATH=/hyperloom/models/Qwen3-30B-A3B
MAX_HOURS=24
INFERENCEX_PATH=/opt/InferenceX
INFERENCE_OPTIMIZER_SESSION_ROOT=/tmp/io-marathon-multicli
ANTHROPIC_API_KEY=<set in shell>
ANTHROPIC_BASE_URL=https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy
TRANSPORT=multi-cli
LAUNCH_CLI_AGENTS=tmux

Run a 24h Claude-backed marathon in --transport multi-cli with
auto-launched tmux panes for each agent.
```

## Equivalent CLI invocations

Single-proc (default):

```bash
PYTHONPATH=src python -m inference_optimizer \
  --model "$MODEL_PATH" \
  --max-hours "$MAX_HOURS" \
  --inferencex-path "$INFERENCEX_PATH" \
  --backend claude \
  --reactor-tick-s 5.0 \
  --clock-tick-s 10.0
```

Multi-CLI with auto-launched subprocess agents (no tmux required —
ideal for unattended / CI / headless runs):

```bash
PYTHONPATH=src python -m inference_optimizer \
  --model "$MODEL_PATH" \
  --max-hours "$MAX_HOURS" \
  --inferencex-path "$INFERENCEX_PATH" \
  --backend claude \
  --transport multi-cli \
  --launch-cli-agents subprocess \
  --reactor-tick-s 5.0 \
  --clock-tick-s 10.0 \
  --router-tick-s 0.5
```

Multi-CLI with tmux (best for interactive monitoring — `tmux attach -t io-multicli`):

```bash
PYTHONPATH=src python -m inference_optimizer \
  --model "$MODEL_PATH" \
  --max-hours "$MAX_HOURS" \
  --inferencex-path "$INFERENCEX_PATH" \
  --backend claude \
  --transport multi-cli \
  --launch-cli-agents tmux \
  --router-tick-s 0.5
```

Hybrid migration (only the executor is a CLI, critic stays in-process):

```bash
PYTHONPATH=src python -m inference_optimizer \
  --model "$MODEL_PATH" --max-hours 3 \
  --backend claude \
  --transport hybrid --cli-agents executor \
  --launch-cli-agents subprocess
```

## `--launch-cli-agents` choices

| value | when to use |
| --- | --- |
| `off` (default) | Conductor sets up the Router + inbox/outbox files, but does NOT spawn agent CLIs. Operator runs them externally. Compatible with marathon's separate `bash run.sh` model. |
| `subprocess` | Conductor uses `subprocess.Popen` to spawn one detached child per agent. Stdout/stderr go to `$SESSION_DIR/logs/<agent>.log`. No tmux dep. Best for unattended / CI / headless. |
| `tmux` | Conductor uses tmux to spawn one window per agent. Best for interactive monitoring with `tmux attach -t io-multicli`. |

## Multi-CLI smoke test (verifies the full subprocess loop)

Three flavours via `src/inference_optimizer/scripts/multi_cli_smoke.sh`:

```bash
# 1. mock — no Claude/Codex/GPU needed; uses the in-tree mock_agent
#    Python module to validate the full Conductor+Launcher+Router loop
#    with a real subprocess. Runs in <60s.
bash src/inference_optimizer/scripts/multi_cli_smoke.sh mock

# 2. claude — real claude --print --continue under hybrid mode (only
#    executor as CLI). Validates the inbox/outbox protocol against an
#    actual Claude session. ~15min default.
ANTHROPIC_API_KEY=... MODEL_PATH=... \
    bash src/inference_optimizer/scripts/multi_cli_smoke.sh claude

# 3. marathon — full 4-agent CLI layout in tmux. ~7h default. Use this
#    on a real GPU sandbox to validate production layout before
#    committing to a 24h+ run.
ANTHROPIC_API_KEY=... OPENAI_API_KEY=... MODEL_PATH=... \
    bash src/inference_optimizer/scripts/multi_cli_smoke.sh marathon
```

The `mock` flavour prints a summary (bus events by topic / by agent,
tasks by state, outbox line count) — use it to compare against a
single-proc baseline run.

## Checks after launch (work in single-proc + multi-cli alike)

```bash
python -m inference_optimizer.scripts.monitor --watch 5 --per-agent
sqlite3 <session_dir>/storage/conductor.db \
  "SELECT from_agent, COUNT(*) FROM events GROUP BY from_agent"
```

Multi-CLI-only inspection:

```bash
# Per-agent inbox/outbox tail
for a in executor critic watchdog sage; do
  echo "==== $a ===="
  tail -n 5 "$SESSION_DIR/agents/$a/outbox.jsonl" 2>/dev/null
  echo "  inbox lines: $(wc -l < $SESSION_DIR/agents/$a/inbox.jsonl 2>/dev/null || echo 0)"
done

# tmux session (if --launch-cli-agents tmux was used)
tmux attach -t "$(cat $SESSION_DIR/.multicli/tmux_session_name)"
```
