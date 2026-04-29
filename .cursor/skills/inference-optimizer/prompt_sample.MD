# Prompt Templates for `@inference-optimizer`

## Quick Smoke Test (force single-proc — short runs don't need multi-CLI overhead)

```text
@inference-optimizer
MODEL_PATH=/hyperloom/models/Qwen3-30B-A3B
MAX_HOURS=0.25
INFERENCEX_PATH=/opt/InferenceX
INFERENCE_OPTIMIZER_SESSION_ROOT=/tmp/io-smoke
OPENAI_API_KEY=<set in shell>
TRANSPORT=single-proc

Run a short codex-backed smoke test in single-proc mode (15-minute runs don't benefit from per-agent CLI isolation). Use the package assets under `src/inference_optimizer` and stop if preflight fails.
```

## Marathon Multi-Reactor Test (default — multi-cli)

```text
@inference-optimizer
MODEL_PATH=/hyperloom/models/Qwen3-30B-A3B
MAX_HOURS=7
INFERENCEX_PATH=/opt/InferenceX
INFERENCE_OPTIMIZER_SESSION_ROOT=/tmp/io-marathon-test
ANTHROPIC_API_KEY=<set in shell>
ANTHROPIC_BASE_URL=https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy

Run a Claude-backed marathon. Default transport is multi-cli, so executor / critic / watchdog / sage each get their own restart-loop CLI session. Use a shell timeout if this is only a validation run.
```

## Multi-CLI Marathon Test (default transport — agents as independent CLIs)

Every agent (executor / critic / watchdog / sage) runs as its own
`claude --print --continue` (or `codex` with explicit conversation log)
restart-loop process. The Conductor only owns the Router + PolicyGate +
SQLite events bus; agents communicate via JSONL inbox/outbox files
under `$SESSION_DIR/agents/<name>/`. This is the **default transport**
since v0.9.

```text
@inference-optimizer
MODEL_PATH=/hyperloom/models/Qwen3-30B-A3B
MAX_HOURS=7
INFERENCEX_PATH=/opt/InferenceX
INFERENCE_OPTIMIZER_SESSION_ROOT=/tmp/io-marathon-multicli
ANTHROPIC_API_KEY=<set in shell>
ANTHROPIC_BASE_URL=https://oci-slc.primus-safe.amd.com/api/v1/llm-proxy

Run a Claude-backed marathon test. (multi-cli is the default — every
agent gets its own CLI session, avoiding the 8-12h context window
exhaustion seen on single-process 24h+ runs.)
```

Equivalent CLI invocation (the skill expands the prompt to this):

```bash
PYTHONPATH=src python -m inference_optimizer \
  --model "$MODEL_PATH" \
  --max-hours "$MAX_HOURS" \
  --inferencex-path "$INFERENCEX_PATH" \
  --backend claude \
  --reactor-tick-s 5.0 \
  --clock-tick-s 10.0 \
  --router-tick-s 0.5
```

To force the legacy in-process model (dev / CI / fast smoke):

```bash
PYTHONPATH=src python -m inference_optimizer \
  --model "$MODEL_PATH" --max-hours 0.25 \
  --backend codex --transport single-proc
```

For a partial migration (e.g. only the executor as a CLI while you
debug critic in-process):

```bash
PYTHONPATH=src python -m inference_optimizer \
  --model "$MODEL_PATH" --max-hours 3 \
  --backend claude --transport hybrid --cli-agents executor
```

Useful checks after launch (work in single-proc + multi-cli alike):

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
done

# tmux session (if started via the launcher)
tmux attach -t "$(cat $SESSION_DIR/.multicli/tmux_session_name)"
```
