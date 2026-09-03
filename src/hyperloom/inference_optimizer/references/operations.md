# Launcher Operations

Use this when the main skill says to install, launch, resume, or monitor an
optimizer session. Keep `$USER_DATA_PATH` as the workspace root and learn
`SESSION_DIR` from `--launch-info-file`; never guess by timestamp.

## Setup

Credentials must already be in the shell environment: `OPENAI_API_KEY` and
`OPENAI_BASE_URL`. Optional source-root overrides are local only:
`INFERENCEX_PATH`, `TRACELENS_ROOT`, `TRACELENS_INTERNAL_ROOT`.

```bash
export REPO_ROOT="$(pwd)"
bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh"
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"
```

`install.sh` is the only full install entrypoint. Source the generated
`kernel-agent.env.sh`; do not derive auth aliases, GEAK paths, or InferenceX
paths by hand. Do not manually repair `$USER_DATA_PATH/runtime/` or
`${HYPERLOOM_CACHE_DIR:-$REPO_ROOT/.cache}/`.

Optionally write `<session_dir>/model_arch.json` if the architecture is known.
It is advisory only; skip rather than guessing. Do not write the file at the
`$USER_DATA_PATH` workspace root because concurrent sessions share that path.

## Launch Flags

```bash
python3 -m hyperloom.inference_optimizer.cli optimize \
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
- `--model-class`: optional; when unset, Coordinator boot infers it from
  model metadata or model-path family keywords. Examples: `dense`, `moe_mla`,
  `moe_swa`, `moe_mla_nsa`.
- `--compare-against-gpu`: optional external reference GPU.
- `--quantize`: only when requested; read `quantization.md` first.

## Smoke Test And Preflight

After IR-2, smoke-test the CLI in the same shell:

```bash
export HYPERLOOM_KERNEL_AGENT_ROOT="$REPO_ROOT/src/hyperloom/agents/kernel"
export KERNEL_AGENT_ROOT="$HYPERLOOM_KERNEL_AGENT_ROOT"
export WORKSPACE_PATH="${WORKSPACE_PATH:-/workspace}"
export PYTHON="${PYTHON:-$(command -v python3)}"
export PATH="$(dirname "$PYTHON"):/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:${PATH:-}"

"$PYTHON" -m hyperloom.inference_optimizer.cli --help
```

Then run the outer launcher preflight (IR-1):

```bash
"$PYTHON" "$REPO_ROOT/src/hyperloom/inference_optimizer/tools/preflight_optimizer.py" "$MODEL_PATH" \
  || { echo "preflight failed — aborting launch"; exit 1; }
```

A non-zero exit indicates GPU occupancy above the allowed threshold, a stale
serving process, or an unreadable GPU state. Do not continue to
`python -m hyperloom.inference_optimizer.cli optimize` in any of these cases.

Do not manually pip-install SDKs, start Ray, or
`curl /v1/models` unless debugging a failed preflight. `_preflight()` and
`install.sh` own those repairs.

## Launch New Optimization

Set `$USER_DATA_PATH` to the workspace root, not the session dir. For sandboxes
that do not persist exports across shell calls, copy
`src/hyperloom/inference_optimizer/assets/setup_env.sh.example` to a **session-scoped** path:
`$USER_DATA_PATH/optimizer_runs/setup_env_${CLAW_SESSION_ID:-$(date +%s)}.sh`,
fill in the workload block, and source it on each call.

**IMPORTANT**: never use a shared filename like `setup_env.sh` — concurrent
sessions on different pods share `$USER_DATA_PATH` via WekaFS; a single file
causes MODEL_PATH race conditions where sessions launch the wrong model.

```bash
cd "$REPO_ROOT"
# .env fills gaps only: re-exporting the non-empty pre-source snapshot keeps every
# value the caller exported. Wider than install.sh, which guards a fixed list.
_dotenv_prev="$(export -p | grep -v -e '=""$' -e "=''\$")"
if [ -f "$REPO_ROOT/.env" ]; then set -a; . "$REPO_ROOT/.env"; set +a; fi
eval "$_dotenv_prev"
unset _dotenv_prev
. "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"
export PATH="$(dirname "$PYTHON"):/usr/local/bin:$PATH"
export RUN_TAG="$(basename "$MODEL_PATH")-$(date +%Y%m%d_%H%M%S)"
export RUN_DIR="${USER_DATA_PATH:-/workspace/hyperloom}/optimizer_runs"
export RUN_LOG="$RUN_DIR/run_${RUN_TAG}.log"
export PID_FILE="$RUN_DIR/run_${RUN_TAG}.pid"
export LAUNCH_INFO_FILE="$RUN_DIR/launch_${RUN_TAG}.json"
mkdir -p "$RUN_DIR"

python3 -m hyperloom.inference_optimizer.cli --verbose optimize \
  --model "$MODEL_PATH" \
  --framework "${FRAMEWORK:-sglang}" \
  --target-gain "${TARGET_GAIN:-10}" \
  --max-hours "${MAX_HOURS:-5}" \
  --tick-interval-sec 30 \
  --launch-info-file "$LAUNCH_INFO_FILE" \
  > "$RUN_LOG" 2>&1 < /dev/null
```

**Detach it the way the harness understands.** Under Claw (`$CLAW_SESSION_ID`
set), hand that block to the bash tool with `run_in_background=true` — no
`setsid nohup`, no trailing `&`. Everywhere else, prefix `setsid nohup`, append
` &`, and `echo $! > "$PID_FILE"`; that form is required for runs longer than 5
minutes under Cursor. See the **Launch** section of `SKILL.md` for why the
distinction matters: a hand-detached run is invisible to Claw, and the sandbox
is reclaimed about fifteen minutes after the agent turn ends, with the run still
going.

Either way, reconcile `$PID_FILE` to the **real** optimizer PID, which the CLI
records as `.pid` in the launch-info JSON. Under `setsid` the `$!` written above
is the wrapper, which exits immediately; under Claw the tool returns a
`shell_id` and never a pid. The robustness monitor reads `$PID_FILE` and would
misfire a spurious resume on a dead wrapper pid.

Health-check after 30 seconds (the launch-info JSON carries the authoritative
`.pid` and `.session_dir`; `jq` is not guaranteed on every node, so fall back to
a tiny `python3` reader):

```bash
sleep 30
read_json() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" "$1" "$2" 2>/dev/null; }

# Real optimizer PID (NOT the setsid wrapper in $!): take it from launch-info
# and rewrite $PID_FILE so the monitor watches the right process.
REAL_PID="$(read_json "$LAUNCH_INFO_FILE" pid)"
[ -z "$REAL_PID" ] && REAL_PID="$(pgrep -f 'hyperloom.inference_optimizer.cli .*optimize' | head -1)"
[ -n "$REAL_PID" ] && echo "$REAL_PID" > "$PID_FILE"
# Not `test -d /proc/$pid`: a zombie keeps its /proc entry and sandbox PID 1
# does not reap, so that check reports a dead optimizer as alive indefinitely.
# Ask for the process state and reject Z.
ps -o stat= -p "$REAL_PID" 2>/dev/null | grep -qv '^Z' \
  && echo "optimizer_alive=true pid=$REAL_PID"

SESSION_DIR="$(read_json "$LAUNCH_INFO_FILE" session_dir)"
if [ -z "$SESSION_DIR" ]; then
  echo "ERROR: no .session_dir in $LAUNCH_INFO_FILE; inspect HYPERLOOM_LAUNCH and $RUN_LOG" >&2
  return 1 2>/dev/null || exit 1
fi

test -f "$SESSION_DIR/manifest.json" && echo "manifest_present=true session_dir=$SESSION_DIR"
test -f "$SESSION_DIR/state.json" && echo "state_exists=true"
```

## Resume Existing Session

`--resume-from "$SESSION_DIR"` is the only way to resume; the CLI never
chooses a session for you. Take `$SESSION_DIR` from the launch-info JSON or
the `HYPERLOOM_LAUNCH` line, never from the newest timestamp dir under
`$USER_DATA_PATH/<model>/`. Keep `$USER_DATA_PATH` at the workspace root
so `runtime/kernel-agent.env.sh` resolves. The named session must contain
`manifest.json` and `state.json`.

Reuse the launch template with these diffs: drop `--model`, add
`--resume-from "$SESSION_DIR"`, and set
`RUN_TAG="resume-$(date +%Y%m%d_%H%M%S)"`. Set `$FRAMEWORK` when resuming a
non-default session.

## Robustness Monitor

For runs longer than 5 minutes, start the monitor in its own detached process,
by the same rule as the optimizer above: `run_in_background=true` under Claw,
`setsid nohup ... &` elsewhere. It polls every 300s. It reads `$INFERENCE_OPTIMIZER_SESSION_DIR` first,
else `.session_dir` from `$LAUNCH_INFO_FILE`. Its only allowed relaunch is the
same session via `optimize --resume-from "$SESSION_DIR"` after the
optimizer process disappears without a terminal marker; it must not start a
fresh run.

```bash
export RUN_DIR="${USER_DATA_PATH:-/workspace/hyperloom}/optimizer_runs"
mkdir -p "$RUN_DIR"
export LAUNCH_INFO_FILE="$RUN_DIR/launch_${RUN_TAG}.json"
cp "$REPO_ROOT/src/hyperloom/inference_optimizer/tools/robustness_monitor.sh.example" \
   "$RUN_DIR/robustness_monitor.sh"
chmod +x "$RUN_DIR/robustness_monitor.sh"
bash "$RUN_DIR/robustness_monitor.sh" \
  > "$RUN_DIR/robustness_monitor_$(date +%Y%m%d_%H%M%S).log" \
  2>&1 < /dev/null
```

## Monitoring

Poll every 300s unless debugging startup failure.

```bash
export SESSION_DIR="$(jq -r '.session_dir // empty' "$LAUNCH_INFO_FILE")"
test -n "$SESSION_DIR"
"$PYTHON" "$REPO_ROOT/src/hyperloom/inference_optimizer/tools/read_optimizer_state.py" "$SESSION_DIR"
python3 "$REPO_ROOT/src/hyperloom/inference_optimizer/tools/event_counts.py" "$SESSION_DIR"
```

Surface lifecycle lines from `read_optimizer_state.py` in chat verbatim. For
`stop_reason` meanings, read `troubleshooting.md`.
