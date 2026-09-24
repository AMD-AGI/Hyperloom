---
name: hyperloom-qwen3-8b-3h
description: Run a 3-hour Hyperloom Qwen3-8B FRAMEWORK_AGENT (OPTIMIZE) session without the Kernel Agent. Use when the user wants a short, framework-only Hyperloom demo on the local AMD ROCm environment.
---

# Hyperloom Qwen3-8B 3h Framework-Only (No-Kernel) Run

Load `.env` with the preamble below and resolve `HYPERLOOM_SKILL_PATH`. Read and follow the optimizer skill at `@${HYPERLOOM_SKILL_PATH}` before launching. If `HYPERLOOM_SKILL_PATH` is missing, fall back to `@hyperloom/inference_optimizer/SKILL.md` (wheel install) or `@src/hyperloom/inference_optimizer/SKILL.md` (source checkout). This skill provides the concrete workload and launch constraints for a short Qwen3-8B demo.

## Execution Shell

Run this in the current Hyperloom workspace before setup or runtime installation.
Repeat it in each new execution shell, including inside Docker; the shared loader
fills dotenv gaps without replacing existing non-empty exports.

```bash
set -e
export REPO_ROOT="$(pwd -P)"
INSTALL_SH="${REPO_ROOT}/hyperloom/inference_optimizer/assets/install.sh"
if [ ! -f "$INSTALL_SH" ]; then
  INSTALL_SH="${REPO_ROOT}/src/hyperloom/inference_optimizer/assets/install.sh"
fi
. "${INSTALL_SH%/*}/runtime_env.sh"
load_dotenv_no_clobber
```

## Run Mode

Resolve the run mode before launching Hyperloom:

1. If `HYPERLOOM_RUN_MODE=baremetal` or it is unset, run this demo directly on the host.
2. If `HYPERLOOM_RUN_MODE=docker`, this skill owns the Docker setup. Ask the user whether they want a `vllm` or `sglang` Docker image unless `HYPERLOOM_IMAGE` is already set. Use a ROCm image that already contains the selected framework; do not install the framework inside Docker.

In docker mode:
- If `hyperloom-setup` already ran, do **not** re-run setup on the host.
- Read `HYPERLOOM_DOCKER_TARGET_HOST` from `.env` when present. If it names a
  host different from `$(hostname)`, first SSH to that host and continue this
  Docker setup there; do not start Docker on the login/current host.
- Always run setup **inside the container** after `docker run`.
- Pass `--install-framework none --yes` in the container (ROCm/framework comes from
  the image). Do **not** use `--skip-base-check` — let Phase 1 preflight validate
  the container environment.
- Do not run `python -m hyperloom.inference_optimizer.cli optimize` on the host.

### Prior workload cleanup (required)

Before any replacement launch after a failed or abandoned demo run (`docker run`,
`install.sh`, or a new/fresh `optimize`), follow **IR-1 — Prior workload cleanup
gate** in `@${HYPERLOOM_SKILL_PATH}`. Run all probes on the **docker host**; never
skip the user-approval step (#1314).

Suggested Docker images:

- `vllm`: `docker.io/rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`
- `sglang` MI300X: `docker.io/lmsysorg/sglang-rocm:v0.5.20-rocm10-mi30x-20260920`
- `sglang` MI355X: `docker.io/lmsysorg/sglang-rocm:v0.5.20-rocm10-mi35x-20260920`

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

Mount the Hyperloom workspace at the same absolute path (`-v "$REPO_ROOT:$REPO_ROOT"`) so paths in `.env`, logs, and session artifacts stay valid. If `USER_DATA_PATH` or a pre-downloaded model directory is outside the workspace, add matching `-v host_path:host_path` mounts before starting the container.

Then run the setup backend inside the container:

```bash
docker exec -w "$REPO_ROOT" "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" bash -lc \
  'REPO_ROOT="$(pwd -P)"; PYTHONPATH="$REPO_ROOT" python3 -m hyperloom.inference_optimizer.setup -- --install-framework none --yes'
```

After that, run all remaining commands for this demo inside the same container with `docker exec -w "$REPO_ROOT" ...`; do not run `python -m hyperloom.inference_optimizer.cli optimize` on the host in Docker mode. When the demo is finished, ask the user whether to stop the container. If they say yes, run:

```bash
docker stop "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}"
```

## Environment

- `MODEL_PATH=<optional; if unset, download Qwen/Qwen3-8B from Hugging Face with the Python steps below, then set MODEL_PATH to that local path>`
- `FRAMEWORK=<provided by the existing environment or repository-root .env; do not invent it>`
- `GPU_TYPE=<do not set; omit --gpu-type and let Hyperloom auto-detect from ROCm/system info>`
Required optimize CLI flags:

- `--tp 1`
- `--conc 64`
- `--isl 1024`
- `--osl 1024`
- `--precision bf16`
- `--target-gain 30`
- `--max-hours 3`
- `--max-minutes-framework-pct 0.50`
- `--max-minutes-sweep-pct 0.01`
- `--no-kernel`
- `--no-enable-conc-sweep`
- `--no-enable-roofline`

Before launch, read the repository-root `.env` file if it exists and load the needed environment variables from it, such as LLM API keys/base URLs, `FRAMEWORK`, and `HF_TOKEN`. Do not copy secret values into the prompt, terminal output, reports, or logs. Do not modify `USER_DATA_PATH`.

Before resolving or downloading any model, always ask the user which model path to use. Present the currently resolved option when `MODEL_PATH` is already set, and always offer a custom local path plus the demo default. Do not continue until the user chooses one.

Use this decision flow:

- If the user chooses the existing `MODEL_PATH`, inspect that path and use it only when it contains `config.json`; otherwise ask again for a valid path or the demo default.
- If the user provides a custom local path, export `MODEL_PATH` to that path and require `config.json` before launch.
- If the user chooses the demo default, set `MODEL_PATH=${REPO_ROOT}/.cache/hyperloom-models/Qwen3-8B` and download `Qwen/Qwen3-8B` there when `config.json` is not already present.

Do not assume the Hugging Face CLI exists; resolve or download the selected model with Python:

```bash
python -m pip install -U huggingface_hub
export REPO_ROOT="$(pwd -P)"
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/.cache/hyperloom-models/Qwen3-8B}"
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(os.environ["MODEL_PATH"]).expanduser()
if (target / "config.json").is_file():
    print(f"Using existing model at {target.resolve()}")
else:
    snapshot_download(
        repo_id="Qwen/Qwen3-8B",
        local_dir=str(target),
    )
print(target.resolve())
PY
```

## Pre-launch Runtime Install

Before the first `optimize` launch, run the full runtime installer in the same
environment that will launch the optimizer. This is required even for this
`--no-kernel` demo: preflight loads `kernel-agent.env.sh` before it can reach the
later Ray/Magpie/InferenceX auto-install checks.

For Docker mode, run this inside the container. For bare-metal mode, run it on
the host:

Use the [execution-shell preamble](#execution-shell) first, then run:

```bash
load_dotenv_no_clobber
: "${USER_DATA_PATH:?USER_DATA_PATH missing}"
export USER_DATA_PATH
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
ulimit -Sn 65536 || true
bash "$INSTALL_SH"
```

The optimizer's startup preflight loads `kernel-agent.env.sh` in process;
do not source the generated file in the launch shell.

## User-visible Progress

Keep the user informed with concise status updates throughout the demo. Do not
dump full debug logs into chat; report the important values and paths so the user
can tell that work is progressing.

Before launch, report the launch plan:

- model path and whether it is an existing local model or a downloaded default;
- run mode (`baremetal` or `docker`) and target host/container when applicable;
- framework, TP, concurrency, ISL, OSL, precision, max hours, and required demo
  flags;
- `USER_DATA_PATH` and where runtime artifacts will be written.

After the runtime install, report whether it succeeded and the path to
`kernel-agent.env.sh`. After starting the optimizer, report:

- optimizer PID;
- run log path;
- launch-info JSON path;
- resolved session directory;
- `state.json` path;
- initial health check result.

On each requested status check, read persisted state and print a short summary.
Use platform-scheduled invocations if recurring checks are requested; do not
start a background watchdog, hold a blocking polling connection, or auto-resume.
Busy logs alone are not evidence of useful progress. Include:

- process alive/stopped;
- phase and `stop_reason`;
- baseline throughput, current best throughput, and cumulative gain when present;
- latest benchmark result or candidate decision when available;
- the most relevant recent log lines, excluding secrets.

When the run finishes, report the final status, final report path, best result,
and the stop reason. Never print API keys, tokens, or custom header values.

## Launch Requirements

1. Run the pre-launch runtime install above; startup preflight loads the generated
   runtime environment in process.
2. Keep `PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"` in the launch shell so
   critic subprocesses can import `hyperloom.agents` after changing cwd.
3. Run it detached the way the harness understands: if `$CLAW_SESSION_ID` is set and your bash tool takes a `run_in_background` parameter, hand the optimizer command to it with `run_in_background=true`, without shell-level detachment (`setsid`, `nohup`, or a trailing `&`); otherwise use `setsid nohup ... &`. See the Launch section of the packaged `hyperloom/inference_optimizer/SKILL.md` for why — a hand-detached run is invisible to Claw and its sandbox is reclaimed about fifteen minutes after the turn ends.
4. Pass all required optimize CLI flags in the `python -m hyperloom.inference_optimizer.cli optimize` command. Do not rely on `.env` alone for `TP`, `CONC`, `ISL`, `OSL`, or `PRECISION`; CLI defaults can otherwise override the intended workload.
5. Include `--max-minutes-framework-pct 0.50` and `--max-minutes-sweep-pct 0.01`
   in the optimize command. These are the value *before* redistribution: with
   `--no-kernel`, KERNEL_AGENT is disabled and its freed share is added on top,
   so `0.50` becomes ~0.99 of wall clock for FRAMEWORK_AGENT. Raising `0.50`
   buys almost nothing — the post-redistribution share is capped at a full wall
   clock, and the excess is discarded.
   Do **not** pass `--no-framework-agent` — that skips OPTIMIZE entirely.
6. Include `--no-kernel` in the optimize command so the Kernel Agent phase is skipped.
7. Include `--no-enable-conc-sweep` in the optimize command so the SWEEP-phase post-optimization concurrency sweep is skipped.
8. Include `--no-enable-roofline` in the optimize command so PRELUDE uses the lighter profile path instead of roofline analysis.
9. Report the session ID, log path, PID, and initial health check result.
10. Inspect persisted state on requested status checks; report when work stops.
11. After diagnosing an unexpected crash and obtaining explicit resume approval, only run `optimize --resume-from "$SESSION_DIR"` against the same session dir. After the first launch, never start a new `optimize`; that creates a new `<UTC_ts>` session and is forbidden.
12. If `stop_reason` in the current session `state.json` is final, stop and exit.
