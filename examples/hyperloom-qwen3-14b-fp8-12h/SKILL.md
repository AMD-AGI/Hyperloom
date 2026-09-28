---
name: hyperloom-qwen3-14b-fp8-12h
description: Run a 12-hour Hyperloom Qwen3-14B-FP8 optimization session on SGLang, vLLM or ATOM. Use when the user wants a medium-length Hyperloom demo on the local AMD ROCm environment.
---

# Hyperloom Qwen3-14B-FP8 12h Run

Load `.env` with the preamble below and resolve `HYPERLOOM_SKILL_PATH`. Read and follow the optimizer skill at `@${HYPERLOOM_SKILL_PATH}` before launching. If `HYPERLOOM_SKILL_PATH` is missing, fall back to `@hyperloom/inference_optimizer/SKILL.md` (wheel install) or `@src/hyperloom/inference_optimizer/SKILL.md` (source checkout). This skill provides the concrete workload and launch constraints for a 12-hour Qwen3-14B-FP8 demo.

The serving framework is `sglang`, `vllm` or `atom`. SGLang and vLLM follow the
sections below as written. For ATOM (`FRAMEWORK=atom`, or the user asked for
ATOM, including a Docker run whose container has not been detected yet), use the
[ATOM](#atom) section for the execution shell, run mode, setup, runtime install
and launch command; the workload flags in [Environment](#environment), the model
choice, [User-visible Progress](#user-visible-progress) and
[Launch Requirements](#launch-requirements) apply to ATOM unchanged.

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
- `atom`: see [ATOM Docker container](#atom-docker-container)

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

After that, run all remaining commands for this demo inside the same container with `docker exec -w "$REPO_ROOT" ...`; do not run `python -m hyperloom.inference_optimizer.cli optimize` on the host in Docker mode.

**Do not use `docker exec -d` to launch optimize.** Detached `docker exec`
discards stdout and stderr, so an optimizer that dies on startup looks like
"backgrounding does not work." Use one **attached**
`docker exec -w "$REPO_ROOT" "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" bash -lc '…'` that
runs the Launch recipe in `@${HYPERLOOM_SKILL_PATH}`. Startup preflight loads
`kernel-agent.env.sh`; do not source it. Under Claw, hand that attached exec
to the bash tool with `run_in_background=true` and no `setsid`, `nohup`, or
trailing `&`; otherwise the command inside the exec is
`setsid nohup … > "$RUN_LOG" 2>&1 < /dev/null &` plus `--launch-info-file`.
Confirm with `pgrep -af 'hyperloom.inference_optimizer.*optimize'`. If nothing
is alive or the launch-info JSON has no `.session_dir`, read the run log and
fix that error; do not retry with a different backgrounding trick.

When the demo is finished, ask the user whether to stop the container. If they say yes, run:

```bash
docker stop "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}"
```

## Environment

- `MODEL_PATH=<optional; if unset, download Qwen/Qwen3-14B-FP8 from Hugging Face with the Python steps below, then set MODEL_PATH to that local path>`
- `FRAMEWORK=<sglang, vllm or atom; provided by the existing environment or repository-root .env; do not invent it>`
- `GPU_TYPE=<do not set; omit --gpu-type and let Hyperloom auto-detect from ROCm/system info>`
Required optimize CLI flags:

- `--tp 1`
- `--conc 64`
- `--isl 1024`
- `--osl 1024`
- `--precision fp8`
- `--target-gain 50`
- `--max-hours 12`
- `--max-minutes-framework-pct 0.43`
- `--max-minutes-kernel-pct 0.42`

Before launch, read the repository-root `.env` file if it exists and load the needed environment variables from it, such as LLM API keys/base URLs, `FRAMEWORK`, and `HF_TOKEN`. Do not copy secret values into the prompt, terminal output, reports, or logs. Do not modify `USER_DATA_PATH`.

Before resolving or downloading any model, always ask the user which model path to use. Present the currently resolved option when `MODEL_PATH` is already set, and always offer a custom local path plus the demo default. Do not continue until the user chooses one.

Use this decision flow:

- If the user chooses the existing `MODEL_PATH`, inspect that path and use it only when it contains `config.json`; otherwise ask again for a valid path or the demo default.
- If the user provides a custom local path, export `MODEL_PATH` to that path and require `config.json` before launch.
- If the user chooses the demo default, set `MODEL_PATH=${REPO_ROOT}/.cache/hyperloom-models/Qwen3-14B-FP8` and download `Qwen/Qwen3-14B-FP8` there when `config.json` is not already present.

Do not assume the Hugging Face CLI exists; resolve or download the selected model with Python:

```bash
python -m pip install -U huggingface_hub
export REPO_ROOT="$(pwd -P)"
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/.cache/hyperloom-models/Qwen3-14B-FP8}"
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(os.environ["MODEL_PATH"]).expanduser()
if (target / "config.json").is_file():
    print(f"Using existing model at {target.resolve()}")
else:
    snapshot_download(
        repo_id="Qwen/Qwen3-14B-FP8",
        local_dir=str(target),
    )
print(target.resolve())
PY
```

## Pre-launch Runtime Install

Before the first `optimize` launch, run the full runtime installer in the same
environment that will launch the optimizer. Preflight loads `kernel-agent.env.sh`
before it can reach the later Ray/Magpie/InferenceX auto-install checks, so this
step must happen before launching.

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
3. Run it detached the way the harness understands: if `$CLAW_SESSION_ID` is set and your bash tool takes a `run_in_background` parameter, hand the optimizer command to it with `run_in_background=true`, without shell-level detachment (`setsid`, `nohup`, or a trailing `&`); otherwise use `setsid nohup ... &`. See the Launch section of the packaged `hyperloom/inference_optimizer/SKILL.md` for why — a hand-detached run is invisible to Claw and its sandbox is reclaimed about fifteen minutes after the turn ends. In Docker mode that launch still runs inside one attached `docker exec … bash -lc`; never `docker exec -d`.
4. Pass all required optimize CLI flags in the `python -m hyperloom.inference_optimizer.cli optimize` command. Do not rely on `.env` alone for `TP`, `CONC`, `ISL`, `OSL`, or `PRECISION`; CLI defaults can otherwise override the intended workload.
5. Include `--max-minutes-framework-pct 0.43` and `--max-minutes-kernel-pct 0.42`
   in the optimize command. Do **not** pass `--no-framework-agent` or `--no-kernel` —
   this demo runs the full OPTIMIZE phase (FRAMEWORK_AGENT + KERNEL_AGENT).
6. Report the session ID, log path, PID, and initial health check result.
7. Inspect persisted state on requested status checks; report when work stops.
8. After diagnosing an unexpected crash and obtaining explicit resume approval, only run `optimize --resume-from "$SESSION_DIR"` against the same session dir. After the first launch, never start a new `optimize`; that creates a new `<UTC_ts>` session and is forbidden.
9. If `stop_reason` in the current session `state.json` is final, stop and exit.

## ATOM

ATOM is an AMD out-of-tree serving engine that Hyperloom launches as
`python3 -m atom.entrypoints.openai_server`. The subsections below replace the
generic execution shell, run mode, setup, runtime install and launch command for
`FRAMEWORK=atom`; everything else above applies unchanged. ATOM is single-node
(IR-8 in the optimizer skill). The KERNEL_AGENT phase runs GEAK, the same backend
every other framework gets when the operator names none; preserve an explicit
`KERNEL_OPT_BACKEND_ORDER` from the caller or `.env` rather than clearing it. For
the per-kernel KernelForge backend use the
[12h forge demo](../hyperloom-qwen3-14b-fp8-12h-forge/SKILL.md).

### ATOM run mode

Follow the setup skill's **Run Mode Resolution**, shared with the vLLM/SGLang
workflow. Reuse the user's `HYPERLOOM_RUN_MODE` selection (`baremetal` or `docker`)
from setup, the caller, or `.env`. If no valid selection is available, ask the user
to choose before setup, container creation, or launch. Do not default to either mode
or infer a preference from the framework or a previous validation run.

- `docker`: run in an ATOM container on the selected development host.
- `baremetal`: run directly in that host's ATOM/ROCm Python environment. A
  development platform that is itself a container still counts as baremetal
  when no additional Docker container is started.

Export the selected `HYPERLOOM_RUN_MODE` in the execution shell and run only its
matching entry below; both entries use the same setup, runtime install and
launch steps.

### ATOM execution shell

In the chosen Hyperloom workspace, use the shared loader with caller exports
taking precedence. Repeat this preamble in each new execution shell, including
inside Docker, with the selected `HYPERLOOM_RUN_MODE` exported. Keep
`USER_DATA_PATH` unchanged; the loader handles readonly roots and Docker's host
Python isolation without a new shell.

```bash
set -e
export REPO_ROOT="$(pwd -P)"
INSTALL_SH="${REPO_ROOT}/hyperloom/inference_optimizer/assets/install.sh"
if [ ! -f "$INSTALL_SH" ]; then
  INSTALL_SH="${REPO_ROOT}/src/hyperloom/inference_optimizer/assets/install.sh"
fi
. "${INSTALL_SH%/*}/runtime_env.sh"
load_dotenv_no_clobber
export FRAMEWORK=atom
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
```

### ATOM baremetal

After choosing direct execution, select the mode in that execution shell:

```bash
export HYPERLOOM_RUN_MODE=baremetal
```

Activate the existing ATOM environment, or select its executable with `PYTHON`;
if the environment has no ATOM yet, setup can install it (see
[ATOM Python and setup](#atom-python-and-setup)). Do not run any Docker commands.

### ATOM Docker container

Only after the user selects `HYPERLOOM_RUN_MODE=docker`: use the approved
`HYPERLOOM_DOCKER_TARGET_HOST`, or the current development host if unset. Do not
create containers or run setup/optimize on a login host. Reuse completed setup
only in the actual execution environment, not a different host Python.

Suggested MI355X image: `docker.io/rocm/atom-dev:v0.1.7-rc0`; preserve an explicit
`HYPERLOOM_IMAGE`. Choose an image compatible with the target GPU.
After approval, mount the workspace at the same absolute path. Add matching
mounts for `USER_DATA_PATH` and any model directory outside the workspace:

```bash
export HYPERLOOM_RUN_MODE=docker
export REPO_ROOT="$(pwd -P)"
export HYPERLOOM_IMAGE="${HYPERLOOM_IMAGE:-docker.io/rocm/atom-dev:v0.1.7-rc0}"
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

Enter the container and feed the ATOM steps below to Bash. This non-TTY form
forwards exported selections by name without printing their values; use `-it`
instead of `-i` for an interactive terminal. Add any other required shell-only
credential/provider variables to the list. Mounted `.env` supplies missing values.
Do not forward host `PYTHON`, `PATH`, or venv settings: select the container's
existing ATOM environment after entry.

```bash
set -e
_atom_container_env=()
for _atom_name in HYPERLOOM_RUN_MODE USER_DATA_PATH MODEL_PATH KERNEL_OPT_BACKEND_ORDER CLAW_SESSION_ID \
  FORGE_AGENT_BACKEND FORGE_AGENT_CLI CLAUDE_MODEL CODEX_MODEL HYPERLOOM_SKILL_PATH \
  ANTHROPIC_API_KEY ANTHROPIC_BASE_URL ANTHROPIC_AUTH_TOKEN OPENAI_API_KEY OPENAI_BASE_URL; do
  if printenv "$_atom_name" > /dev/null; then
    _atom_container_env+=(--env "$_atom_name")
  fi
done
unset _atom_name
docker exec -i -w "$REPO_ROOT" "${_atom_container_env[@]}" \
  "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" bash
```

Inside that shell, repeat the [ATOM execution shell](#atom-execution-shell)
preamble, then run the setup, runtime install and launch steps below. Repeat the
preamble and Python selection for each new `docker exec` shell. Stop this
session's container only with approval after the run finishes.

### ATOM prior workload cleanup

Before replacement launches, follow **IR-1 — Prior workload cleanup gate** in the
packaged optimizer skill, on the direct environment or Docker target as applicable.
ATOM worker command lines can omit the server entrypoint. Identify this session's
PIDs, process groups, ports and container before proposing cleanup; never broadly
kill Python workers or restart the machine. Check GPU usage before GPU work:

```bash
rocm-smi --showmemuse
pgrep -af "openai_server|spawn_main"
```

### ATOM Python and setup

Activate the existing ATOM venv or select an explicit `PYTHON`; otherwise use
`python3` from PATH. Do not create a fresh venv or require `/opt/venv`.

```bash
export PYTHON="${PYTHON:-$(command -v python3)}"
export INFERENCE_OPTIMIZER_FORCE_PYTHON=1
```

Setup checks the selected Python, ROCm torch and ATOM import/server readiness.
Run its read-only check even when setup previously completed:

```bash
"$PYTHON" -m hyperloom.inference_optimizer.setup --check-only -- \
  --install-framework none --frameworks atom --require-frameworks \
  --user-data-path "${USER_DATA_PATH:?USER_DATA_PATH missing}"
```

Reuse successful setup in this environment. Only if setup is needed, explain its
changes and obtain approval before running:

```bash
"$PYTHON" -m hyperloom.inference_optimizer.setup -- \
  --install-framework none --frameworks atom --require-frameworks \
  --user-data-path "${USER_DATA_PATH:?USER_DATA_PATH missing}" --yes
```

If the check reports ATOM missing in a baremetal environment and the operator
approves installing it, run setup with `atom` instead of `none`. It installs
AITER and ATOM from source into the selected Python, keeping its ROCm torch.
Setup refuses when SGLang or vLLM also imports from that Python, because those
engines would then load ATOM's plugins; report the refusal and propose a
separate container for ATOM rather than removing the other engine. Do not do
this in Docker mode, where ATOM comes from the image:

```bash
"$PYTHON" -m hyperloom.inference_optimizer.setup -- \
  --install-framework atom \
  --user-data-path "${USER_DATA_PATH:?USER_DATA_PATH missing}" --yes
```

`none` skips framework installation only: actual setup writes `.env` and may
apply ROCm hotfixes; `--yes` is not user consent. Do not bypass base checks, repeat
onboarding unnecessarily, or install SGLang/vLLM to compensate for missing ATOM.

For additional serving settings, use the optimizer's `--server-args` option.
Do not launch a separate server or export `EXTRA_ATOM_ARGS`: Hyperloom materializes
that transport variable. Keep the initial configuration untuned; do not copy
another GPU's block/KV settings or the final settings of a previous optimization.
Persist `FRAMEWORK=atom` in `.env` only when requested.

Resolve the model with the decision flow in [Environment](#environment), in the
actual execution environment, running its Python steps with the selected
`"$PYTHON"`. If `huggingface_hub` is missing, obtain approval before installing
it with `"$PYTHON" -m pip install huggingface_hub`.

### ATOM runtime install

Prepare runtime in the same environment as ATOM, using the existing installer.
Reuse a prepared runtime; installation needs approval because it installs
dependencies and may start services. If shell and setup-written `.env` disagree
on `USER_DATA_PATH`, reconcile the selected root first: the installer treats
setup's `.env` as authoritative. Never silently replace the artifact root.

Use the ATOM execution-shell preamble and selected Python above, then run:

```bash
: "${USER_DATA_PATH:?USER_DATA_PATH missing}" "${PYTHON:?Select the existing ATOM Python first}"
export USER_DATA_PATH
ulimit -Sn 65536 || true
bash "$INSTALL_SH"
```

For every launch, including resume, the optimizer loads `kernel-agent.env.sh`
in process. Startup preflight checks the selected framework. Do not source the
generated runtime file in the shell. Readiness failures require diagnosis and
approval for repairs, not a provider switch.

### ATOM first launch

Use the packaged optimizer skill's **Launch a New Optimization** instructions to
prepare `RUN_LOG`, `PID_FILE`, `LAUNCH_INFO_FILE` and run-scoped metadata in this
execution environment. Keep the selected ATOM Python and backend for launch.
After setup and runtime preparation, use this complete command for both modes.
Do not replace workload flags with environment-only settings or add
`--no-framework-agent` / `--no-kernel`:

```bash
set -e
: "${PYTHON:?PYTHON missing}" "${MODEL_PATH:?MODEL_PATH missing}"
: "${RUN_LOG:?RUN_LOG missing}" "${LAUNCH_INFO_FILE:?LAUNCH_INFO_FILE missing}"
"$PYTHON" -m hyperloom.inference_optimizer.cli --verbose optimize \
  --model "$MODEL_PATH" \
  --framework atom \
  --tp 1 --conc 64 --isl 1024 --osl 1024 \
  --precision fp8 \
  --target-gain 50 --max-hours 12 \
  --max-minutes-framework-pct 0.43 --max-minutes-kernel-pct 0.42 \
  --launch-info-file "$LAUNCH_INFO_FILE" \
  > "$RUN_LOG" 2>&1 < /dev/null
```

Detach through the existing harness-aware launch path: when `CLAW_SESSION_ID` is
set and the Bash tool supports `run_in_background`, use it without shell-level
detachment; otherwise prefix the optimize command with `setsid nohup` and append
`&`. Follow the packaged skill's separate health check and launch-info/PID
reconciliation; a shell wrapper PID is not the optimizer PID. Do not create a
second launcher or watchdog for baremetal.

### ATOM resume

After the first launch, never start another fresh `optimize` to recover a failed
run. Diagnose it, obtain explicit approval, and pass `--resume-from "$SESSION_DIR"`
for that same session instead of a new model launch. Repeat the ATOM
execution-shell preamble and Python selection, retain phase fractions `.43/.42`,
and follow the ATOM prior workload cleanup for any remaining ATOM workers before
relaunching. Do not assume resume rewrites launch-info: verify the current
process and session state rather than using an old PID. A final `stop_reason`
ends the run; do not automatically restart it.

### ATOM status checks

Report the chosen mode and Python together with the launch plan. Confirm the
recorded framework and backend immediately after launch, not just the shell
exports:

```bash
grep -o '"framework": *"[^"]*"' "$SESSION_DIR/state.json"
grep -o '"kernel_optimizer": *"[^"]*"' "$SESSION_DIR/state.json"
```

Expect `atom` and `geak` (or the backend the operator explicitly kept). Report
mismatches without silently replacing the session.
