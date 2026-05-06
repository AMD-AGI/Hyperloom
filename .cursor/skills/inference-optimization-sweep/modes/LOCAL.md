# Local Mode

## Auto-detection

```bash
GEAK_LOCAL=true              # default; forces local mode
MODE="local"
```

`actions/setup.md` reads `GEAK_LOCAL` and sets `MODE` accordingly. Every
GPU-bound command is executed via `eval` directly; no Ray submission.

## Pre-conditions

- The current host has the GPUs visible to InferenceX (`amd-smi list` returns
  one or more GPUs).
- The user has provided `EXTRA_SGLANG_ARGS` or `EXTRA_VLLM_ARGS` aligned with
  the desired framework.
- `pip install -e $MAGPIE_PATH` was executed during `actions/setup.md` Step 4
  (idempotent; skipped when `magpie` is already on `PATH`).

## Iron Rule recap (local-relevant)

- IR-4: kill leftover server process before launching anything (see Magpie
  runner shell, which does this when `MAGPIE_RUN_PHASE=server`).
- IR-5: only `kill $(pgrep -f 'python.*-m sglang.launch_server')` or the vLLM
  equivalent. Never `pkill -f sglang`.

## Run the sweep

```bash
export PATH="/opt/venv/bin:$PATH"
export MODEL=/shared_nfs/models/DeepSeek-R1-0528
export TP=8
export FRAMEWORK=sglang
export EXTRA_SGLANG_ARGS="--attention-backend aiter --kv-cache-dtype fp8_e4m3 \
    --disable-radix-cache --mem-fraction-static 0.85 \
    --chunked-prefill-size 32768 --max-prefill-tokens 32768 \
    --num-continuous-decode-steps 8"

bash -c '
set -e
SKILL_ROOT=".cursor/skills/inference-optimization-sweep"
. "$SKILL_ROOT/scripts/executor.sh"     # exec_on_gpu helper (no-op in local)
# Step 1: setup
. <(awk "/^\`\`\`bash$/,/^\`\`\`$/" "$SKILL_ROOT/actions/setup.md" \
        | grep -v "^\`\`\`")
# Step 2: sweep
. <(awk "/^\`\`\`bash$/,/^\`\`\`$/" "$SKILL_ROOT/actions/sweep.md" \
        | grep -v "^\`\`\`")
'
```

The agent typically runs the steps directly inline rather than via this
shell-extracts-bash-blocks pattern; the snippet above shows the equivalent
manual flow for reproducibility.

## Long-running benchmarks (IR-8 background runner)

A whole sweep on a 671B MoE model can run for an hour. To avoid blocking
the agent's foreground:

```bash
nohup bash -c '
    magpie benchmark --benchmark-config "$SWEEP_DIR/sweep_config.yaml" \
        -o "$SWEEP_DIR" > "$SWEEP_DIR/magpie.log" 2>&1
' > /dev/null 2>&1 &
disown
```

Then poll `tail -f "$SWEEP_DIR/magpie.log"` periodically. This keeps the
agent responsive while Magpie runs the full server lifecycle and all client
cases.

## Cleanup

`magpie benchmark` kills the shared server when the sweep finishes (success
or failure). If a stray server is left behind:

```bash
kill $(pgrep -f 'python.*-m sglang.launch_server') 2>/dev/null
kill $(pgrep -f 'python.*-m vllm.entrypoints')     2>/dev/null
sleep 5
amd-smi process 2>/dev/null | grep -E '(sglang|vllm)' || echo "clean"
```
