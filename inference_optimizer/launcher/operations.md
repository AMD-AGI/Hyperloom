# Launcher Operations

Use this when the main skill says to install, launch, resume, or monitor an
optimizer session. Keep `$USER_DATA_PATH` as the workspace root and learn
`SESSION_DIR` from `--launch-info-file`; never guess by timestamp.

## Setup

Credentials must already be in the shell environment: `SAFE_API_KEY` and
`OPENAI_BASE_URL`. Optional source-root overrides are local only:
`OOB_SRC`, `INFERENCEX_PATH`, `TRACELENS_ROOT`, `TRACELENS_INTERNAL_ROOT`.

```bash
export REPO_ROOT="$(pwd)"
bash "$REPO_ROOT/inference_optimizer/scripts/install.sh"
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"
```

`install.sh` is the only full install entrypoint. Source the generated
`kernel-agent.env.sh`; do not derive auth aliases, GEAK paths, or InferenceX
paths by hand. Do not manually repair `runtime/source-mirrors/`.

Optionally write `$USER_DATA_PATH/model_arch.json` if the architecture is known.
It is advisory only; skip rather than guessing.

## Launch Flags

```bash
inference_optimizer optimize \
  --model "$MODEL_PATH" \
  --framework vllm \
  --gpu-type MI300X \
  --model-class moe_mla \
  --max-hours 2 \
  --compare-against-gpu B200
```

- `--model`: required model path.
- `--framework`: `sglang` (default), `vllm`, or `atom`; atom is single-node only.
- `--gpu-type`: optional; omitted means rocm-smi auto-detect.
- `--model-class`: defaults to `moe_mla`; examples: `dense`, `moe_mla`,
  `moe_swa`, `moe_mla_nsa`.
- `--compare-against-gpu`: optional external reference GPU.
- `--quantize`: only when requested; read `quantization.md` first.

## Smoke Test And Preflight

After IR-2, smoke-test the CLI in the same shell:

```bash
export HYPERLOOM_KERNEL_AGENT_ROOT="$REPO_ROOT/kernel-agent"
export KERNEL_AGENT_ROOT="$HYPERLOOM_KERNEL_AGENT_ROOT"
export WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"
export PYTHON="${PYTHON:-$(command -v python3)}"
export PATH="$(dirname "$PYTHON"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

"$PYTHON" -m inference_optimizer.cli --help
```

Then run the outer launcher preflight (IR-1):

```bash
"$PYTHON" "$REPO_ROOT/inference_optimizer/launcher/preflight_optimizer.py" "$MODEL_PATH"
```

Do not manually pip-install SDKs, edit `~/.claude/config.json`, start Ray, or
`curl /v1/models` unless debugging a failed preflight. `_preflight()` and
`install.sh` own those repairs.

## Launch New Optimization

Set `$USER_DATA_PATH` to the workspace root, not the session dir. For sandboxes
that do not persist exports across shell calls, copy
`inference_optimizer/scripts/setup_env.sh.example` to
`$USER_DATA_PATH/optimizer_runs/setup_env.sh`, fill in the workload block, and
source it on each call.

```bash
cd "$REPO_ROOT"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"
export PATH="$(dirname "$PYTHON"):/usr/local/bin:$PATH"
export RUN_TAG="$(basename "$MODEL_PATH")-$(date +%Y%m%d_%H%M%S)"
export RUN_DIR="${USER_DATA_PATH:-/workspace/hyperloom}/optimizer_runs"
export RUN_LOG="$RUN_DIR/run_${RUN_TAG}.log"
export PID_FILE="$RUN_DIR/run_${RUN_TAG}.pid"
export LAUNCH_INFO_FILE="$RUN_DIR/launch_${RUN_TAG}.json"
mkdir -p "$RUN_DIR"

setsid nohup inference_optimizer --verbose optimize \
  --model "$MODEL_PATH" \
  --framework "${FRAMEWORK:-sglang}" \
  --target-gain "${TARGET_GAIN:-10}" \
  --max-hours "${MAX_HOURS:-5}" \
  --tick-interval-sec 30 \
  --kernel-claude \
  --launch-info-file "$LAUNCH_INFO_FILE" \
  > "$RUN_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"
```

`setsid nohup ... &` is required for runs longer than 5 minutes. After launch,
locate the optimizer with `pgrep -af 'inference_optimizer.*optimize'`; `$!` may
be only a wrapper PID.

Health-check after 30 seconds:

```bash
sleep 30
pid="$(cat "$PID_FILE")"
test -d "/proc/$pid" && echo "optimizer_alive=true pid=$pid"

SESSION_DIR="$(jq -r '.session_dir // empty' "$LAUNCH_INFO_FILE" 2>/dev/null)"
if [ -z "$SESSION_DIR" ]; then
  echo "ERROR: no .session_dir in $LAUNCH_INFO_FILE; inspect HYPERLOOM_LAUNCH and $RUN_LOG" >&2
  return 1 2>/dev/null || exit 1
fi

test -f "$SESSION_DIR/manifest.json" && echo "manifest_present=true session_dir=$SESSION_DIR"
test -f "$SESSION_DIR/state.json" && echo "state_exists=true"
```

## Resume Existing Session

For monitored runs, prefer explicit same-session resume:
`--resume --resume-from "$SESSION_DIR"`. Bare `--resume` auto-picks the latest
`$USER_DATA_PATH/<model>/<UTC_ts>/` and is only acceptable for manual recovery
when that is the intended session. Keep `$USER_DATA_PATH` at the workspace root
so `runtime/kernel-agent.env.sh` resolves. The selected session must contain
`manifest.json` and `state.json`.

Reuse the launch template with these diffs: drop `--model`, add
`--resume --resume-from "$SESSION_DIR"`, and set
`RUN_TAG="resume-$(date +%Y%m%d_%H%M%S)"`. Set `$FRAMEWORK` when resuming a
non-default session.

## Robustness Monitor

For runs longer than 5 minutes, start the monitor in its own `setsid nohup`
process. It polls every 300s. It reads `$INFERENCE_OPTIMIZER_SESSION_DIR` first,
else `.session_dir` from `$LAUNCH_INFO_FILE`. Its only allowed relaunch is the
same session via `optimize --resume --resume-from "$SESSION_DIR"` after the
optimizer process disappears without a terminal marker; it must not start a
fresh run or auto-pick the latest session.

```bash
export RUN_DIR="${USER_DATA_PATH:-/workspace/hyperloom}/optimizer_runs"
mkdir -p "$RUN_DIR"
export LAUNCH_INFO_FILE="$RUN_DIR/launch_${RUN_TAG}.json"
cp "$REPO_ROOT/inference_optimizer/launcher/robustness_monitor.sh.example" \
   "$RUN_DIR/robustness_monitor.sh"
chmod +x "$RUN_DIR/robustness_monitor.sh"
setsid nohup bash "$RUN_DIR/robustness_monitor.sh" \
  > "$RUN_DIR/robustness_monitor_$(date +%Y%m%d_%H%M%S).log" \
  2>&1 < /dev/null &
```

## Monitoring

Poll every 300s unless debugging startup failure.

```bash
export SESSION_DIR="$(jq -r '.session_dir // empty' "$LAUNCH_INFO_FILE")"
test -n "$SESSION_DIR"
"$PYTHON" "$REPO_ROOT/inference_optimizer/launcher/read_optimizer_state.py" "$SESSION_DIR"
python3 "$REPO_ROOT/inference_optimizer/scripts/event_counts.py" "$SESSION_DIR"
```

Surface lifecycle lines from `read_optimizer_state.py` in chat verbatim. For
`stop_reason` meanings, read `troubleshooting.md`.
