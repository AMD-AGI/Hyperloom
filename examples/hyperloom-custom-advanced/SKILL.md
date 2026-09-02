---
name: hyperloom-custom-advanced
description: Run an advanced configurable Hyperloom optimization session with explicit model, framework, workload, objective, and phase toggles. Use when the user wants more control than the fixed 3h or 12h demo presets.
---

# Hyperloom Custom Advanced Run

Use this skill after `/hyperloom-setup` has prepared the current Hyperloom
workspace. Setup writes `.env` with the run mode, target host, framework, LLM
configuration, and `USER_DATA_PATH`; this skill reuses those values and asks
only for the advanced workload choices that differ from the fixed demo presets.

## Setup Configuration

Load `.env` from the current Hyperloom workspace before launching. Treat it as
the source of truth for setup-owned values such as `HYPERLOOM_RUN_MODE`,
`HYPERLOOM_DOCKER_TARGET_HOST`, `FRAMEWORK`, `USER_DATA_PATH`, and LLM provider
settings. Do not ask the user to re-enter setup values that are already present.

When `HYPERLOOM_RUN_MODE=baremetal` or it is unset, run this demo directly in
the current environment.

When `HYPERLOOM_RUN_MODE=docker`, this skill owns the Docker setup. Use
`HYPERLOOM_IMAGE` when it is set. Otherwise choose a recommended ROCm image for
the selected framework. The image must already contain the framework; do not
install the framework inside Docker.

In docker mode:
- If `hyperloom-setup` already ran, do **not** re-run setup on the host.
- Read `HYPERLOOM_DOCKER_TARGET_HOST` from `.env` when present. If it names a
  host different from `$(hostname)`, first SSH to that host and continue this
  Docker setup there; do not start Docker on the login/current host.
- Always run setup **inside the container** after `docker run`.
- Pass `--install-framework none --yes` in the container because ROCm and the
  framework must come from the image. Do **not** use `--skip-base-check`.
- Do not run `python -m hyperloom.inference_optimizer.cli optimize` on the host.

### Prior workload cleanup (required)

Before any replacement launch after a failed or abandoned demo run (`docker run`,
`install.sh`, or a new/fresh `optimize`), follow **IR-1 — Prior workload cleanup
gate** in `@${HYPERLOOM_SKILL_PATH}`. Run all probes on the **docker host**; never
skip the user-approval step (#1314).

Suggested Docker images:

- `vllm`: `docker.io/vllm/vllm-openai-rocm:v0.27.1`
- `sglang` MI300X: `docker.io/lmsysorg/sglang-rocm:v0.5.18-rocm724-mi30x-20260825`
- `sglang` MI355X: `docker.io/lmsysorg/sglang-rocm:v0.5.18-rocm724-mi35x-20260825`
- `vllm` Radeon 8060S/gfx1151: `hyperloom-vllm-v027-rocm10:v16-runtime-admission-v1`
- `sglang` Radeon 8060S/gfx1151: `hyperloom-sglang-v0515-rocm10:v16-async-v7`

In Docker mode, start a long-running container on `HYPERLOOM_DOCKER_TARGET_HOST`
(or the current host when it is unset) before running setup or optimize:

```bash
export REPO_ROOT="$(pwd -P)"
docker run -d \
  --name "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" \
  --shm-size "${HYPERLOOM_SHM_SIZE:-64g}" \
  --entrypoint tail \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -v "$REPO_ROOT:$REPO_ROOT" \
  "$HYPERLOOM_IMAGE" \
  -f /dev/null
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

Before launch, load the `.env` file produced by `/hyperloom-setup`, including
LLM API keys/base URLs, `FRAMEWORK`, `USER_DATA_PATH`, and `HF_TOKEN`. Do not
copy secret values into the prompt, terminal output, reports, or logs. Do not
modify `USER_DATA_PATH`.

Use the agent's structured question UI when available. Do not continue until
all required values are resolved.

Collect these required values:

- Model source:
  - existing `MODEL_PATH`, when set;
  - custom local path, which must contain `config.json`;
  - Hugging Face repo id plus a local cache directory.
- Framework: `sglang` or `vllm`. Prefer the existing
  `FRAMEWORK` value when it is set; otherwise default to `sglang`.
- Workload: `TP`, `EP`, `CONC`, `ISL`, `OSL`, `PRECISION`, and optional
  `MAX_MODEL_LEN` / `PROFILE_OSL`.
- Objective and budget: `MAX_HOURS` plus `TARGET_GAIN`.
- Optional phase budget percentages for `PRELUDE`, `FRAMEWORK_AGENT`,
  `KERNEL_AGENT`, `SWEEP`, and `CLOSE`.

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
- `TARGET_GAIN=30`.
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
- Phase budget percentages default to:
  - `PHASE_BUDGET_PRELUDE_PCT=0.03`: startup, preflight, baseline setup, and
    initial orchestration.
  - `PHASE_BUDGET_FRAMEWORK_PCT=0.40`: the optimisation phase — serving-parameter
    search and source/upstream landing, with benchmark validation.
  - `PHASE_BUDGET_KERNEL_PCT=0.50`: kernel-agent TraceLens/GEAK/native-kernel
    optimization work.
  - `PHASE_BUDGET_SWEEP_PCT=0.05`: concurrency sweep and final throughput
    validation around the best candidate.
  - `PHASE_BUDGET_CLOSE_PCT=0.02`: final report, state closeout, and summary
    generation.
- Phase toggles default to enabled: kernel, framework agent, framework local
  exploration, roofline, and concurrency sweep.

Collect these optional advanced values:

- Phase toggles: `--no-kernel`, `--no-framework-agent`,
  `--no-framework-local-explore`, `--no-enable-conc-sweep`,
  `--no-enable-roofline`.
- Phase budget percentages:
  `PHASE_BUDGET_PRELUDE_PCT`, `PHASE_BUDGET_FRAMEWORK_PCT`,
  `PHASE_BUDGET_KERNEL_PCT`,
  `PHASE_BUDGET_SWEEP_PCT`, and `PHASE_BUDGET_CLOSE_PCT`.
  Explain what each phase does before asking. Accept only values where
  `0 < pct <= 1`; leave a value unset to use the optimizer default.
- Routing and baseline options: `--skip-variants`, `--server-args`,
  `--reference-script`, `--model-class`, `--gpu-type`, `--framework-version`,
  `--target-summary`, `--compare-against-gpu`.
- Concurrency sweep: `--conc-sweep-concs`, `--conc-sweep-timeout-sec`,
  `--conc-sweep-total-budget-sec`.

Guardrails:

- Do not rely on `.env` alone for `TP`, `CONC`, `ISL`, `OSL`, or `PRECISION`;
  pass explicit CLI flags in the optimize command.
- Omit `--gpu-type` unless the user explicitly chooses a hint; otherwise let
  Hyperloom auto-detect from ROCm/system info.
- `--no-framework-agent` skips the entire OPTIMIZE phase (PRELUDE goes straight
  to KERNEL_AGENT), dropping both of its arms: upstream-PR landing and local
  source authoring. Warn before applying it; combined with `--no-kernel` it
  leaves only baseline and sweep validation.
- `--no-framework-local-explore` keeps OPTIMIZE but drops only its local
  authoring arm, so the phase exits after three empty upstream discoveries
  instead of authoring a patch from live source. No effect under diff-only mode.
- Phase budget percentages are caps, not guaranteed time usage. A phase may end
  earlier, and disabled work phases have their share redistributed by the
  optimizer.
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
# .env fills gaps only: re-exporting the non-empty pre-source snapshot keeps every
# value the caller exported. Wider than install.sh, which guards a fixed list.
_dotenv_prev="$(export -p | grep -v -e '=""$' -e "=''\$")"
set -a; . "${REPO_ROOT}/.env"; set +a
eval "$_dotenv_prev"
unset _dotenv_prev
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
export TARGET_GAIN="${TARGET_GAIN:-30}"
export PHASE_BUDGET_PRELUDE_PCT="${PHASE_BUDGET_PRELUDE_PCT:-}"
export PHASE_BUDGET_FRAMEWORK_PCT="${PHASE_BUDGET_FRAMEWORK_PCT:-}"
export PHASE_BUDGET_KERNEL_PCT="${PHASE_BUDGET_KERNEL_PCT:-}"
export PHASE_BUDGET_SWEEP_PCT="${PHASE_BUDGET_SWEEP_PCT:-}"
export PHASE_BUDGET_CLOSE_PCT="${PHASE_BUDGET_CLOSE_PCT:-}"

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
[ -n "${PHASE_BUDGET_PRELUDE_PCT:-}" ] && OPT_FLAGS+=(--max-minutes-prelude-pct "$PHASE_BUDGET_PRELUDE_PCT")
[ -n "${PHASE_BUDGET_FRAMEWORK_PCT:-}" ] && OPT_FLAGS+=(--max-minutes-framework-pct "$PHASE_BUDGET_FRAMEWORK_PCT")
[ -n "${PHASE_BUDGET_KERNEL_PCT:-}" ] && OPT_FLAGS+=(--max-minutes-kernel-pct "$PHASE_BUDGET_KERNEL_PCT")
[ -n "${PHASE_BUDGET_SWEEP_PCT:-}" ] && OPT_FLAGS+=(--max-minutes-sweep-pct "$PHASE_BUDGET_SWEEP_PCT")
[ -n "${PHASE_BUDGET_CLOSE_PCT:-}" ] && OPT_FLAGS+=(--max-minutes-close-pct "$PHASE_BUDGET_CLOSE_PCT")
[ "${NO_KERNEL:-0}" = "1" ] && OPT_FLAGS+=(--no-kernel)
[ "${NO_FRAMEWORK_AGENT:-0}" = "1" ] && OPT_FLAGS+=(--no-framework-agent)
[ "${NO_FRAMEWORK_LOCAL_EXPLORE:-0}" = "1" ] && OPT_FLAGS+=(--no-framework-local-explore)
[ "${NO_CONC_SWEEP:-0}" = "1" ] && OPT_FLAGS+=(--no-enable-conc-sweep)
[ "${NO_ROOFLINE:-0}" = "1" ] && OPT_FLAGS+=(--no-enable-roofline)

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

If adding quantization, critic, robustness, or research-lane flags, append only
real flags accepted by
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
- phase budget percentages, showing defaults for unset values and user-selected
  overrides for set values;
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
7. To recover an unexpected crash, only run
   `optimize --resume-from "$SESSION_DIR"` against the same session dir. After
   the first launch, never start a new `optimize`; that creates a new
   `<UTC_ts>` session and is forbidden.
8. If `stop_reason` in the current session `state.json` is final, stop and exit.
