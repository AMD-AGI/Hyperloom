---
name: hyperloom-custom-advanced
description: Run an advanced configurable Hyperloom optimization session with explicit model, framework, workload, objective, and phase toggles. Use when the user wants more control than the fixed 3h, 8h, or 24h demo presets.
---

# Hyperloom Custom Advanced Run

Read `.env` first and resolve `HYPERLOOM_SKILL_PATH`. Read and follow the
optimizer skill at `@${HYPERLOOM_SKILL_PATH}` before launching. If
`HYPERLOOM_SKILL_PATH` is missing, fall back to
`@hyperloom/inference_optimizer/SKILL.md` (wheel install) or
`@src/hyperloom/inference_optimizer/SKILL.md` (source checkout). This skill
provides an advanced workload configuration flow for users who want explicit
control over model, framework, workload shape, optimization budget, and phase
toggles.

## Run Mode

Resolve the run mode before launching Hyperloom:

1. If `HYPERLOOM_RUN_MODE=baremetal` or it is unset, run this demo directly on
   the host.
2. If `HYPERLOOM_RUN_MODE=docker`, this skill owns the Docker setup. Ask the
   user which image to use unless `HYPERLOOM_IMAGE` is already set. Use a ROCm
   image that already contains the selected framework; do not install the
   framework inside Docker.

In docker mode:
- If `hyperloom-setup` already ran, do **not** re-run setup on the host.
- Read `HYPERLOOM_DOCKER_TARGET_HOST` from `.env` when present. If it names a
  host different from `$(hostname)`, first SSH to that host and continue this
  Docker setup there; do not start Docker on the login/current host.
- Always run setup **inside the container** after `docker run`.
- Pass `--install-framework none --yes` in the container because ROCm and the
  framework must come from the image. Do **not** use `--skip-base-check`.
- Do not run `python -m hyperloom.inference_optimizer.cli optimize` on the host.

Suggested Docker images:

- `vllm`: `docker.io/primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix`
- `sglang` MI300X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix`
- `sglang` MI355X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix`

For `atom` or `xdit`, require a custom `HYPERLOOM_IMAGE` that already contains
the selected framework.

In Docker mode, start a long-running container on `HYPERLOOM_DOCKER_TARGET_HOST`
(or the current host when it is unset) before running setup or optimize:

```bash
export REPO_ROOT="$(pwd -P)"
docker run -d \
  --name "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" \
  --shm-size "${HYPERLOOM_SHM_SIZE:-64g}" \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -v "$REPO_ROOT:$REPO_ROOT" \
  "$HYPERLOOM_IMAGE" \
  tail -f /dev/null
```

Mount the Hyperloom workspace at the same absolute path
(`-v "$REPO_ROOT:$REPO_ROOT"`) so paths in `.env`, logs, and session artifacts
stay valid. If `USER_DATA_PATH` or the selected model directory is outside the
workspace, add matching `-v host_path:host_path` mounts before starting the
container.

Then run the setup backend inside the container:

```bash
docker exec -w "$REPO_ROOT" "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" bash -lc \
  'REPO_ROOT="$(pwd -P)"; PYTHONPATH="$REPO_ROOT" python3 -m hyperloom.inference_optimizer.setup -- --install-framework none --yes'
```

After that, run all remaining commands for this demo inside the same container
with `docker exec -w "$REPO_ROOT" ...`; do not run
`python -m hyperloom.inference_optimizer.cli optimize` on the host in Docker
mode. When the demo is finished, ask the user whether to stop the container. If
they say yes, run:

```bash
docker stop "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}"
```

## Advanced Configuration

Before launch, read the repository-root `.env` file if it exists and load the
needed environment variables from it, such as LLM API keys/base URLs,
`FRAMEWORK`, `USER_DATA_PATH`, and `HF_TOKEN`. Do not copy secret values into
the prompt, terminal output, reports, or logs. Do not modify `USER_DATA_PATH`.

Use the agent's structured question UI when available. Do not continue until
all required values are resolved.

Collect these required values:

- Model source:
  - existing `MODEL_PATH`, when set;
  - custom local path, which must contain `config.json`;
  - Hugging Face repo id plus a local cache directory.
- Framework: `sglang`, `vllm`, `atom`, or `xdit`. Prefer the existing
  `FRAMEWORK` value when it is set; otherwise default to `sglang`.
- Workload: `TP`, `EP`, `CONC`, `ISL`, `OSL`, `PRECISION`, and optional
  `MAX_MODEL_LEN` / `PROFILE_OSL`.
- Objective and budget: `MAX_HOURS` plus exactly one of `TARGET_GAIN` or
  `TARGET_TPUT`. Use `TARGET_GAIN=30` as the default when the user does not
  choose a throughput target.

## Default Values

Use these defaults when the user does not override a field. Always show the
resolved values in the launch plan before starting the optimizer.

- `FRAMEWORK`: existing `FRAMEWORK` from `.env` or shell, otherwise `sglang`.
- `TP=1`.
- `EP=1`.
- `CONC=64`.
- `ISL=1024`.
- `OSL=1024`.
- `PRECISION=bf16`.
- `MAX_HOURS=8`, a medium-length full optimization default. The user may choose
  a shorter smoke run such as `3`, or a long-horizon run such as `24`.
- `TARGET_GAIN=30` unless the user chooses `TARGET_TPUT`.
- `MAX_MODEL_LEN`: unset, so Hyperloom derives it from ISL/OSL and model
  metadata.
- `PROFILE_OSL`: unset, so Hyperloom uses its profile-phase default.
- `MODEL_CLASS`: unset, so Hyperloom infers it from model metadata.
- `GPU_TYPE`: unset, so Hyperloom auto-detects the target GPU.
- `FRAMEWORK_VERSION`: unset, so Hyperloom auto-detects it when possible.
- `TARGET_SUMMARY`: unset.
- `COMPARE_AGAINST_GPU`: unset.
- `SKIP_VARIANTS`: empty.
- `SERVER_ARGS`: empty.
- `REFERENCE_SCRIPT`: unset.
- `CONC_SWEEP_CONCS`: unset, so Hyperloom uses its default sweep ladder.
- `CONC_SWEEP_TIMEOUT_SEC`: unset, so Hyperloom uses its default per-variant
  timeout.
- `CONC_SWEEP_TOTAL_BUDGET_SEC`: unset, so Hyperloom uses its default total
  sweep budget.
- Phase toggles default to enabled: kernel enabled, explore enabled, framework
  agent enabled, roofline enabled, and concurrency sweep enabled.
- Kernel continues after FP8 GEMM by default.
- Multi-node defaults to disabled: `NODES=1`; do not pass multi-node or PD flags
  unless the user explicitly asks for them.

Collect these optional advanced values:

- Phase toggles: `--no-kernel`, `--no-explore`, `--no-framework-agent`,
  `--no-enable-conc-sweep`, `--no-enable-roofline`.
- Kernel option: `--no-continue-kernel-after-gemm` for GEMM-only validation.
- Routing and baseline options: `--skip-variants`, `--server-args`,
  `--reference-script`, `--model-class`, `--gpu-type`, `--framework-version`,
  `--target-summary`, `--compare-against-gpu`.
- Concurrency sweep: `--conc-sweep-concs`, `--conc-sweep-timeout-sec`,
  `--conc-sweep-total-budget-sec`.
- Multi-node and PD options only when the user explicitly requests them:
  `--nodes`, `--gpus-per-node`, `--mn-backend`, `--mn-image`, `--ep`,
  `--pd-mode disaggregated`, and the matching PD role flags. Read
  `@src/hyperloom/inference_optimizer/multi_node/SKILL.md` before launching a
  multi-node or PD run.

Guardrails:

- Do not rely on `.env` alone for `TP`, `CONC`, `ISL`, `OSL`, or `PRECISION`;
  pass explicit CLI flags in the optimize command.
- Omit `--gpu-type` unless the user explicitly chooses a hint; otherwise let
  Hyperloom auto-detect from ROCm/system info.
- Warn the user when both `--no-explore` and `--no-kernel` are selected; that
  collapses the run mostly to baseline and sweep validation.
- Warn the user that `--framework atom` is single-node only.
- For `--framework xdit`, report image/diffusion metrics instead of LLM serving
  token metrics where applicable.
- There is no generic `--skip-stage` flag and no `--no-sweep` flag. Compose
  phase behavior from the explicit flags above.

## Model Resolution

Before resolving or downloading any model, always ask the user which model path
or Hugging Face repo to use. Present the currently resolved option when
`MODEL_PATH` is already set, and always offer a custom local path. Do not
continue until the user chooses one.

Use this decision flow:

- If the user chooses the existing `MODEL_PATH`, inspect that path and use it
  only when it contains `config.json`; otherwise ask again for a valid path.
- If the user provides a custom local path, export `MODEL_PATH` to that path and
  require `config.json` before launch.
- If the user provides a Hugging Face repo id, ask for a local cache path, set
  `HF_REPO_ID` to the repo id, set `MODEL_PATH` to the cache path, and download
  the repo there when `config.json` is not already present.

Do not assume the Hugging Face CLI exists; resolve or download Hugging Face
models with Python:

```bash
python -m pip install -U huggingface_hub
export REPO_ROOT="$(pwd -P)"
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(os.environ["MODEL_PATH"]).expanduser()
repo_id = os.environ.get("HF_REPO_ID", "").strip()
if (target / "config.json").is_file():
    print(f"Using existing model at {target.resolve()}")
elif repo_id:
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
    print(target.resolve())
else:
    raise SystemExit(f"MODEL_PATH does not contain config.json: {target}")
PY
```

## Pre-launch Runtime Install

Before the first `optimize` launch, run the full runtime installer in the same
environment that will launch the optimizer. Preflight loads `kernel-agent.env.sh`
before it can reach the later Ray/Magpie/InferenceX auto-install checks, so this
step must happen before launching.

For Docker mode, run this inside the container. For bare-metal mode, run it on
the host:

```bash
export REPO_ROOT="$(pwd -P)"
set -a; . "${REPO_ROOT}/.env"; set +a
export USER_DATA_PATH="${USER_DATA_PATH:?USER_DATA_PATH missing}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
ulimit -Sn 65536 || true
INSTALL_SH="${REPO_ROOT}/hyperloom/inference_optimizer/assets/install.sh"
if [ ! -f "$INSTALL_SH" ]; then
  INSTALL_SH="${REPO_ROOT}/src/hyperloom/inference_optimizer/assets/install.sh"
fi
bash "$INSTALL_SH"
. "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
```

If `hyperloom/inference_optimizer/assets/install.sh` is not present (source
checkout layout), use `src/hyperloom/inference_optimizer/assets/install.sh`.

## Launch Command Template

Build the optimize command from the resolved values. Include optional flags only
when the user selected them.

```bash
export RUN_TAG="$(basename "$MODEL_PATH")-custom-$(date +%Y%m%d_%H%M%S)"
export RUN_DIR="${USER_DATA_PATH:?USER_DATA_PATH missing}/optimizer_runs"
export RUN_LOG="$RUN_DIR/run_${RUN_TAG}.log"
export PID_FILE="$RUN_DIR/run_${RUN_TAG}.pid"
export LAUNCH_INFO_FILE="$RUN_DIR/launch_${RUN_TAG}.json"
mkdir -p "$RUN_DIR"

export FRAMEWORK="${FRAMEWORK:-sglang}"
export TP="${TP:-1}"
export EP="${EP:-1}"
export CONC="${CONC:-64}"
export ISL="${ISL:-1024}"
export OSL="${OSL:-1024}"
export PRECISION="${PRECISION:-bf16}"
export MAX_HOURS="${MAX_HOURS:-8}"
if [ -z "${TARGET_TPUT:-}" ]; then
  export TARGET_GAIN="${TARGET_GAIN:-30}"
fi

OPT_FLAGS=(
  --model "$MODEL_PATH"
  --framework "$FRAMEWORK"
  --tp "$TP"
  --ep "$EP"
  --conc "$CONC"
  --isl "$ISL"
  --osl "$OSL"
  --precision "$PRECISION"
  --max-hours "$MAX_HOURS"
  --tick-interval-sec 30
  --launch-info-file "$LAUNCH_INFO_FILE"
)

[ -n "${TARGET_GAIN:-}" ] && OPT_FLAGS+=(--target-gain "$TARGET_GAIN")
[ -n "${TARGET_TPUT:-}" ] && OPT_FLAGS+=(--target-tput "$TARGET_TPUT")
[ -n "${MAX_MODEL_LEN:-}" ] && OPT_FLAGS+=(--max-model-len "$MAX_MODEL_LEN")
[ -n "${PROFILE_OSL:-}" ] && OPT_FLAGS+=(--profile-osl "$PROFILE_OSL")
[ -n "${MODEL_CLASS:-}" ] && OPT_FLAGS+=(--model-class "$MODEL_CLASS")
[ -n "${GPU_TYPE:-}" ] && OPT_FLAGS+=(--gpu-type "$GPU_TYPE")
[ -n "${FRAMEWORK_VERSION:-}" ] && OPT_FLAGS+=(--framework-version "$FRAMEWORK_VERSION")
[ -n "${TARGET_SUMMARY:-}" ] && OPT_FLAGS+=(--target-summary "$TARGET_SUMMARY")
[ -n "${COMPARE_AGAINST_GPU:-}" ] && OPT_FLAGS+=(--compare-against-gpu "$COMPARE_AGAINST_GPU")
[ -n "${SKIP_VARIANTS:-}" ] && OPT_FLAGS+=(--skip-variants "$SKIP_VARIANTS")
[ -n "${SERVER_ARGS:-}" ] && OPT_FLAGS+=(--server-args "$SERVER_ARGS")
[ -n "${REFERENCE_SCRIPT:-}" ] && OPT_FLAGS+=(--reference-script "$REFERENCE_SCRIPT")
[ -n "${CONC_SWEEP_CONCS:-}" ] && OPT_FLAGS+=(--conc-sweep-concs "$CONC_SWEEP_CONCS")
[ -n "${CONC_SWEEP_TIMEOUT_SEC:-}" ] && OPT_FLAGS+=(--conc-sweep-timeout-sec "$CONC_SWEEP_TIMEOUT_SEC")
[ -n "${CONC_SWEEP_TOTAL_BUDGET_SEC:-}" ] && OPT_FLAGS+=(--conc-sweep-total-budget-sec "$CONC_SWEEP_TOTAL_BUDGET_SEC")
[ "${NO_KERNEL:-0}" = "1" ] && OPT_FLAGS+=(--no-kernel)
[ "${NO_EXPLORE:-0}" = "1" ] && OPT_FLAGS+=(--no-explore)
[ "${NO_FRAMEWORK_AGENT:-0}" = "1" ] && OPT_FLAGS+=(--no-framework-agent)
[ "${NO_CONC_SWEEP:-0}" = "1" ] && OPT_FLAGS+=(--no-enable-conc-sweep)
[ "${NO_ROOFLINE:-0}" = "1" ] && OPT_FLAGS+=(--no-enable-roofline)
[ "${NO_CONTINUE_KERNEL_AFTER_GEMM:-0}" = "1" ] && OPT_FLAGS+=(--no-continue-kernel-after-gemm)

setsid nohup python3 -m hyperloom.inference_optimizer.cli --verbose optimize \
  "${OPT_FLAGS[@]}" \
  > "$RUN_LOG" 2>&1 < /dev/null &
echo $! > "$PID_FILE"

sleep 30
read_json() { python3 -c "import json,sys;print(json.load(open(sys.argv[1])).get(sys.argv[2],''))" "$1" "$2" 2>/dev/null; }
REAL_PID="$(read_json "$LAUNCH_INFO_FILE" pid)"
[ -z "$REAL_PID" ] && REAL_PID="$(pgrep -f 'hyperloom.inference_optimizer.cli .*optimize' | head -1)"
[ -n "$REAL_PID" ] && echo "$REAL_PID" > "$PID_FILE"
test -d "/proc/$REAL_PID" && echo "optimizer_alive=true pid=$REAL_PID"

SESSION_DIR="$(read_json "$LAUNCH_INFO_FILE" session_dir)"
if [ -z "$SESSION_DIR" ]; then
  echo "ERROR: no .session_dir in $LAUNCH_INFO_FILE; inspect HYPERLOOM_LAUNCH and $RUN_LOG" >&2
  return 1 2>/dev/null || exit 1
fi

test -f "$SESSION_DIR/manifest.json" && echo "manifest_present=true session_dir=$SESSION_DIR"
test -f "$SESSION_DIR/state.json" && echo "state_exists=true"
```

If adding multi-node, PD, quantization, critic, robustness, or research-lane
flags, append only real flags accepted by
`python3 -m hyperloom.inference_optimizer.cli optimize --help`; do not invent
aliases.

## User-visible Progress

Keep the user informed with concise status updates throughout the run. Do not
dump full debug logs into chat; report the important values and paths so the
user can tell that work is progressing.

Before launch, report the launch plan:

- model path and whether it is an existing local model or a downloaded repo;
- run mode (`baremetal` or `docker`) and target host/container when applicable;
- framework, TP, EP, concurrency, ISL, OSL, precision, max hours, and objective;
- selected phase toggles and advanced flags;
- `USER_DATA_PATH` and where runtime artifacts will be written.

After the runtime install, report whether it succeeded and the path to
`kernel-agent.env.sh`. After starting the optimizer, report:

- optimizer PID;
- run log path;
- launch-info JSON path;
- resolved session directory;
- `state.json` path;
- initial health check result.

During monitoring, print a short summary at each 300-second check:

- process alive/stopped;
- phase and `stop_reason`;
- baseline throughput, current best throughput, and cumulative gain when present;
- latest benchmark result or candidate decision when available;
- the most relevant recent log lines, excluding secrets.

When the run finishes, report the final status, final report path, best result,
and the stop reason. Never print API keys, tokens, or custom header values.

## Launch Requirements

1. Run the pre-launch runtime install above and source
   `$USER_DATA_PATH/runtime/kernel-agent.env.sh` before launching.
2. Keep `PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"` in the launch shell so
   robustness and critic subprocesses can import `hyperloom.agents` after
   changing cwd.
3. Run in background with `setsid nohup`.
4. Pass all required workload flags in the
   `python -m hyperloom.inference_optimizer.cli optimize` command. Do not rely
   on `.env` alone for `TP`, `CONC`, `ISL`, `OSL`, or `PRECISION`.
5. Report the session ID, log path, PID, and initial health check result.
6. Monitor the process every 300 seconds until work is done.
7. To recover an unexpected crash, only run `optimize --resume` against the same
   session dir. After the first launch, never start a new `optimize`; that
   creates a new `<UTC_ts>` session and is forbidden.
8. If `stop_reason` in the current session `state.json` is final, stop and exit.
