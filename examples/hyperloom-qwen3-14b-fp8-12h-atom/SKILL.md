---
name: hyperloom-qwen3-14b-fp8-12h-atom
description: Run a 12-hour Hyperloom Qwen3-14B-FP8 optimization session on ATOM in an existing development environment or Docker, with KernelForge by default.
---

# Hyperloom Qwen3-14B-FP8 12h Run (ATOM Framework)

Read `.env` and resolve `HYPERLOOM_SKILL_PATH`. Follow `@${HYPERLOOM_SKILL_PATH}`;
if unset, use `@hyperloom/inference_optimizer/SKILL.md` (wheel installation) or
`@src/hyperloom/inference_optimizer/SKILL.md` (source checkout). This ATOM variant
uses the same workload and phase budgets as the
[12h SGLang/vLLM example](../hyperloom-qwen3-14b-fp8-12h/SKILL.md).

## Run Mode

Follow the setup skill's **Run Mode Resolution**, shared with the vLLM/SGLang
workflow. Reuse the user's `HYPERLOOM_RUN_MODE` selection (`baremetal` or `docker`)
from setup, the caller, or `.env`. If no valid selection is available, ask the user
to choose before setup, container creation, or launch. Do not default to either mode
or infer a preference from the framework or a previous validation run.

- `docker`: run in an ATOM container on the selected development host.
- `baremetal`: run directly in that host's existing ATOM/ROCm Python environment.
  A development platform that is itself a container still counts as baremetal
  when no additional Docker container is started.

Export the selected `HYPERLOOM_RUN_MODE` in the execution shell and run only its
matching entry below; both entries use the same Environment and Launch steps.

### Execution shell

Keep `USER_DATA_PATH` unchanged. If the parent shell marks it `readonly`, export
it and run all subsequent environment/setup/runtime/launch blocks together in a
new non-login Bash process:

```bash
export USER_DATA_PATH
bash --noprofile --norc
```

For non-interactive execution, feed those blocks to this process as a script or
heredoc. A `( ... )` subshell retains readonly attributes and is not a substitute.
Do not unset the isolation root or copy the child's environment back to the parent.

In the chosen workspace, load `.env` with caller exports taking precedence.
Repeat this preamble in each new execution shell, before both setup and runtime
loading, with the selected `HYPERLOOM_RUN_MODE` exported. In Docker, a mounted
`.env` may contain the host's `PYTHON`, `VIRTUAL_ENV`, and
`INFERENCE_OPTIMIZER_FORCE_PYTHON`: keep only values already set in this shell,
including empty values, rather than filling those gaps from `.env`. An explicit
Python pin remains authoritative even if invalid; otherwise activate the existing
container environment or use its `python3` from PATH. Baremetal still fills gaps
from `.env` as usual.

```bash
export REPO_ROOT="$(pwd -P)"
# .env fills gaps; existing non-empty exports remain authoritative.
_dotenv_prev="$(export -p | grep -v -e '=\"\"$' -e "=''\$")"
if [ "${HYPERLOOM_RUN_MODE:-}" = docker ]; then
  for _atom_name in PYTHON VIRTUAL_ENV INFERENCE_OPTIMIZER_FORCE_PYTHON; do
    if [ "${!_atom_name+x}" = x ]; then
      _dotenv_prev+=$'\n'"$(declare -p "$_atom_name")"
    else
      _dotenv_prev+=$'\n'"unset $_atom_name"
    fi
  done
  unset _atom_name
fi
set -a; [ ! -f "${REPO_ROOT}/.env" ] || . "${REPO_ROOT}/.env"; set +a
eval "$_dotenv_prev"
unset _dotenv_prev
```

### Baremetal

After choosing direct execution, select the mode in that execution shell:

```bash
export HYPERLOOM_RUN_MODE=baremetal
```

Activate the existing ATOM environment, or select its executable with `PYTHON`.
Continue with [Environment](#environment) below; do not run any Docker commands.

### Docker container

Only after the user selects `HYPERLOOM_RUN_MODE=docker`: use the approved
`HYPERLOOM_DOCKER_TARGET_HOST`, or the current development host if unset. Do not
create containers or run setup/optimize on a login host. Reuse completed setup
only in the actual execution environment, not a different host Python.

The recorded MI355X environment used `docker.io/rocm/atom-dev:v0.1.7-rc0`; preserve
an explicit `HYPERLOOM_IMAGE`. Other GPU/build combinations need validation.
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

Enter the container and feed the shared steps below to Bash. This non-TTY form
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

Inside that shell, repeat the [execution-shell preamble](#execution-shell), then
run Environment, Runtime Install, and Launch Requirements below. A new
`docker exec` shell needs those selections and runtime loading again. Stop this
session's container only with approval after the run finishes.

### Prior workload cleanup (required)

Before replacement launches, follow **IR-1 — Prior workload cleanup gate** in the
packaged optimizer skill, on the direct environment or Docker target as applicable.
ATOM worker command lines can omit the server entrypoint. Identify this session's
PIDs, process groups, ports and container before proposing cleanup; never broadly
kill Python workers or restart the machine. Check GPU usage before GPU work:

```bash
rocm-smi --showmemuse
pgrep -af "openai_server|spawn_main"
```

## Environment

Both modes use the following steps in the shell where ATOM will run.

### Framework

ATOM must already be installed with ROCm torch. Hyperloom does not install ATOM:
use `--install-framework none --frameworks atom --require-frameworks` below.
This example is single-node, fixes `FRAMEWORK=atom`, and lets Hyperloom detect the
GPU rather than passing `--gpu-type`.

For additional serving settings, use the optimizer's `--server-args` option.
Do not launch a separate server or export `EXTRA_ATOM_ARGS`: Hyperloom materializes
that transport variable. Keep the initial configuration untuned; do not copy
another GPU's block/KV settings or the final settings of a previous optimization.

### Kernel Backend

Preserve an explicit `KERNEL_OPT_BACKEND_ORDER` from the caller or `.env`.
Otherwise leave it unset/empty so the ATOM CLI defaults to `forge`. Matching is
exact after whitespace trimming and lowercasing: `FORGE` and ` forge ` select
Forge; other non-empty values, including `forge,geak`, route to GEAK. Report
GEAK's unproven ATOM rewrite-seam support and obtain the operator's choice before
continuing; do not silently clear or replace it. There is no backend CLI flag.

Forge is included in Hyperloom; do not clone it or set `FORGE_PATH`. Preserve the
selected agent provider and `FORGE_AGENT_CLI`; verify the executable after loading
runtime. Persist `FRAMEWORK=atom` in `.env` only when requested, and never add a
backend key merely to reproduce the CLI default.

### Selected Python and Setup

Use the existing ATOM venv or an explicit `PYTHON`; do not create a fresh venv or
require `/opt/venv`. Run this check before setup and repeat it after loading runtime:

```bash
set -e
PYTHON="${PYTHON:-$(command -v python3)}"
PYTHON="$("$PYTHON" -c 'import sys; print(sys.executable)')"
export PYTHON
export INFERENCE_OPTIMIZER_FORCE_PYTHON=1
_atom_venv="$("$PYTHON" -c 'import sys; print(sys.prefix if sys.prefix != sys.base_prefix else "")')"
if [ -n "$_atom_venv" ]; then
  export VIRTUAL_ENV="$_atom_venv"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
  printf '%s\n' 'Deactivate the unrelated venv before selecting a non-venv Python.' >&2
  exit 1
fi
unset _atom_venv
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
"$PYTHON" - <<'PY'
import os
import sys
from pathlib import Path

import atom
import torch

vllm_root = os.environ.get("VLLM_VENV_ROOT", "")
if vllm_root and Path(vllm_root).resolve() != Path(sys.prefix).resolve():
    raise SystemExit(
        f"VLLM_VENV_ROOT={vllm_root} conflicts with the selected ATOM Python prefix {sys.prefix}. "
        "Preflight would prepend its bin directory; ask the operator to reconcile this setting."
    )
print(f"Python: {sys.executable}")
print(f"ATOM: {atom.__file__}")
print(f"torch: {torch.__version__}; HIP: {torch.version.hip}")
if not torch.version.hip:
    raise SystemExit("ATOM requires a ROCm torch build in the selected Python environment")
PY
# Magpie launches the server with python3 from PATH.
[ "$(python3 -c 'import sys; print(sys.prefix)')" = "$("$PYTHON" -c 'import sys; print(sys.prefix)')" ]
"$PYTHON" -m atom.entrypoints.openai_server --help > /dev/null
```

Stop on failed imports, prefix conflicts or CLI help; do not bypass checks or
silently switch environments. The recorded ATOM build needed a fix for `%` in
argparse help text. If affected, obtain approval to repair it or select a corrected
build; vendor repair is not an automatic example step.

Verify even when setup previously completed:

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

`none` skips framework installation only: actual setup writes `.env` and may
apply ROCm hotfixes; `--yes` is not user consent. Do not bypass base checks, repeat
onboarding unnecessarily, or install SGLang/vLLM to compensate for missing ATOM.

### Model

Ask the operator to choose the existing `MODEL_PATH`, a custom local directory,
or the demo default `Qwen/Qwen3-14B-FP8`. A chosen local directory must contain
`config.json`; resolve the path in the actual execution environment.

For the demo default, use `${REPO_ROOT}/.cache/hyperloom-models/Qwen3-14B-FP8` when
no local path was selected. Use the selected Python, not an assumed Hugging Face
CLI. If `huggingface_hub` is missing, obtain approval before installing it with
`"$PYTHON" -m pip install huggingface_hub`.

```bash
export MODEL_PATH="${MODEL_PATH:-${REPO_ROOT}/.cache/hyperloom-models/Qwen3-14B-FP8}"
"$PYTHON" - <<'PY'
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

Prepare runtime in the same environment as ATOM, using the existing installer.
Reuse a prepared runtime; installation needs approval because it installs
dependencies and may start services. If shell and setup-written `.env` disagree
on `USER_DATA_PATH`, reconcile the selected root first: the installer treats
setup's `.env` as authoritative. Never silently replace the artifact root.

```bash
set -e
: "${USER_DATA_PATH:?USER_DATA_PATH missing}"
export USER_DATA_PATH
export PYTHON="${PYTHON:?Select the existing ATOM Python first}"
export INFERENCE_OPTIMIZER_FORCE_PYTHON=1
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
ulimit -Sn 65536 || true
INSTALL_SH="${REPO_ROOT}/hyperloom/inference_optimizer/assets/install.sh"
if [ ! -f "$INSTALL_SH" ]; then
  INSTALL_SH="${REPO_ROOT}/src/hyperloom/inference_optimizer/assets/install.sh"
fi
bash "$INSTALL_SH"
```

### Load runtime environment

Run this before every launch, including an approved resume. In a new shell,
repeat the execution-shell preamble and Selected Python steps first. Keep all
steps in that shell; do not source generated env files in a readonly parent.

```bash
set -e
_atom_python="${PYTHON:?Select the existing ATOM Python first}"
_atom_user_data="${USER_DATA_PATH:?USER_DATA_PATH missing}"
_atom_backend="${KERNEL_OPT_BACKEND_ORDER:-}"
. "$_atom_user_data/runtime/kernel-agent.env.sh"
export PYTHON="$_atom_python" USER_DATA_PATH="$_atom_user_data"
export INFERENCE_OPTIMIZER_FORCE_PYTHON=1
export PATH="$(dirname "$PYTHON"):$PATH"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src:${PYTHONPATH:-}"
export FRAMEWORK=atom
export KERNEL_OPT_BACKEND_ORDER="$_atom_backend"
unset _atom_python _atom_user_data _atom_backend
printf 'framework: %s  kernel backend: %s\n' "$FRAMEWORK" "${KERNEL_OPT_BACKEND_ORDER:-<CLI default: forge>}"
```

Repeat the full [Selected Python](#selected-python-and-setup) check now, including
`VLLM_VENV_ROOT`, PATH and server help, then check Forge CLI readiness below.

### Forge CLI readiness

Use Forge's actual provider/executable resolvers. This checks the local CLI only,
not API credentials or gateway availability; it never starts an agent.

```bash
set -e
"$PYTHON" - <<'PY'
import subprocess

from hyperloom.common.env import env_str
from kernelforge.config import Config

if env_str("KERNEL_OPT_BACKEND_ORDER").lower() in {"", "forge"}:
    runtime = Config.from_env().agent_runtime()
    if runtime.provider == "claude":
        from kernelforge.agent_backends.claude import resolve_claude_cli

        executable = resolve_claude_cli(runtime.executable)
        try:
            result = subprocess.run(
                [executable, "--version"], capture_output=True, text=True,
                timeout=10, check=True,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise SystemExit(
                f"Forge Claude CLI is not ready: {executable}: {exc}. "
                "Request approval to repair it or set FORGE_AGENT_CLI to a working executable."
            ) from exc
        version = (result.stdout + "\n" + result.stderr).strip()
        if "claude" not in version.lower():
            raise SystemExit(f"Forge Claude CLI returned an unexpected version: {executable}")
        print(f"Forge Claude CLI ready: {executable}")
    else:
        print(f"Forge agent backend: {runtime.provider}; Claude CLI check not applicable")
PY
```

Stop on failure; obtain approval to repair a CLI or change providers. Installing
`claude_agent_sdk` or finding its bundled CLI alone is not a successful check.

## Launch Requirements

Use the packaged optimizer skill's **Launch a New Optimization** instructions to
prepare `RUN_LOG`, `PID_FILE`, `LAUNCH_INFO_FILE` and run-scoped metadata in this
execution environment. Reuse the ATOM environment loaded above; do not replay a
generic dotenv/runtime block that would replace those selections.

### First launch

After setup, runtime loading and readiness checks, use this complete command for
both modes. Do not replace workload flags with environment-only settings or add
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

### Resume

After the first launch, never start another fresh `optimize` to recover a failed
run. Diagnose it, obtain explicit approval, and pass `--resume-from "$SESSION_DIR"`
for that same session instead of a new model launch. Reload the selected ATOM
runtime and readiness checks, retain phase fractions `.43/.42`, and follow IR-1
for any remaining ATOM workers before relaunching. Do not assume resume rewrites
launch-info: verify the current process and session state rather than using an old
PID. A final `stop_reason` ends the run; do not automatically restart it.

## User-visible Progress

Follow the packaged skill's **Monitoring** and **Report Back To User** rules.
Report the chosen mode, Python, model, framework/backend, workload and artifact
root before launch, then the real PID, session directory, log/launch-info paths
and initial health check. Never print credentials or custom header values.

Confirm the recorded framework/backend immediately, not just the shell exports:

```bash
grep -o '"framework": *"[^"]*"' "$SESSION_DIR/state.json"
grep -o '"kernel_optimizer": *"[^"]*"' "$SESSION_DIR/state.json"
```

Expect `atom` and the approved backend (`forge` by default). Report mismatches
without silently replacing the session. On requested checks report process state,
phase, accepted throughput/gain and the latest business outcome, not just heartbeats.

At completion, verify and report:

- throughput and full accuracy evidence, distinguishing warmup from measurement;
- Profile's actual serving configuration/trace, not a stale path or declared flag;
- actual kernel application/loading and finite outputs, not an unchecked PASS label;
- sweep point/pair coverage and what its baseline arm compares;
- CLOSE/report status, skipped or failed substeps, and this session's process/GPU release.

An exit code of zero, task `succeeded`, or a report file alone does not establish
all of these. Recorded direct-run evidence included an approved resume and manual
repairs to generated numerical checks, with final profiling/reporting gaps; it is
not an unattended-run guarantee or a Docker-versus-baremetal performance claim.
