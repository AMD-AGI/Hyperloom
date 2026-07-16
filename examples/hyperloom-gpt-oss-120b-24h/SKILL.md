---
name: hyperloom-gpt-oss-120b-24h
description: Run a long-horizon Hyperloom gpt-oss-120b optimization session. Use when the user wants the cyclic macro-cycle behavior for a roughly 24-hour demo.
---

# Hyperloom gpt-oss-120b Long-Horizon Run

Read `.env` first and resolve `HYPERLOOM_SKILL_PATH`. Read and follow the optimizer skill at `@${HYPERLOOM_SKILL_PATH}` before launching. If `HYPERLOOM_SKILL_PATH` is missing, fall back to `@hyperloom/inference_optimizer/SKILL.md` (wheel install) or `@src/hyperloom/inference_optimizer/SKILL.md` (source checkout). This skill provides the concrete workload and launch constraints for a long-horizon gpt-oss-120b demo.

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

Suggested Docker images:

- `vllm`: `docker.io/primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix`
- `sglang` MI300X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix`
- `sglang` MI355X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix`

In Docker mode, start a long-running container on `HYPERLOOM_DOCKER_TARGET_HOST`
(or the current host when it is unset) before running setup or optimize:

```bash
docker run -d \
  --name "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" \
  --shm-size "${HYPERLOOM_SHM_SIZE:-64g}" \
  --device /dev/kfd \
  --device /dev/dri \
  --group-add video \
  -v "$PWD:$PWD" \
  "$HYPERLOOM_IMAGE" \
  tail -f /dev/null
```

Mount the Hyperloom workspace at the same path (`-v "$PWD:$PWD"`) so paths in `.env`, logs, and session artifacts stay valid. If `USER_DATA_PATH` or a pre-downloaded model directory is outside the workspace, add matching `-v host_path:host_path` mounts before starting the container.

Then run the setup backend inside the container:

```bash
docker exec -w "$PWD" "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}" bash -lc \
  'PYTHONPATH="$PWD" python3 -m hyperloom.inference_optimizer.setup -- --install-framework none --yes'
```

After that, run all remaining commands for this demo inside the same container with `docker exec -w "$PWD" ...`; do not run `python -m hyperloom.inference_optimizer.cli optimize` on the host in Docker mode. When the demo is finished, ask the user whether to stop the container. If they say yes, run:

```bash
docker stop "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}"
```

## Long-Horizon Gate

Current Hyperloom treats `--max-hours 24` as a long-horizon run. Long-horizon cyclic macro-cycles require one of:

1. `--max-hours >= 24`
2. an unbounded run (`max_minutes == 0`)

Cyclic macro-cycling is always on; the long-horizon behavior is gated purely by the budget above (`--max-hours >= 24` or an unbounded run).

## Environment

- `MODEL_PATH=<optional; if unset, download openai/gpt-oss-120b from Hugging Face with the Python steps below, then set MODEL_PATH to that local path>`
- `FRAMEWORK=<provided by the existing environment or repository-root .env; do not invent it>`
- `GPU_TYPE=<do not set; omit --gpu-type and let Hyperloom auto-detect from ROCm/system info>`
Required optimize CLI flags:

- `--tp 1`
- `--conc 64`
- `--isl 1024`
- `--osl 1024`
- `--precision bf16`
- `--target-gain 30`
- `--max-hours 24`

Before launch, read the repository-root `.env` file if it exists and load the needed environment variables from it, such as LLM API keys/base URLs, `FRAMEWORK`, and `HF_TOKEN`. Do not copy secret values into the prompt, terminal output, reports, or logs. Do not modify `USER_DATA_PATH`.

If `MODEL_PATH` is set, inspect that path first: use it when it already contains `config.json`; otherwise download `openai/gpt-oss-120b` into that exact directory. If `MODEL_PATH` is unset, ask the user whether they want to provide a target model path. If they provide one, export `MODEL_PATH` to that path; if not, use `.cache/hyperloom-models/gpt-oss-120b`. Do not assume the Hugging Face CLI exists; resolve or download the model with Python:

```bash
python -m pip install -U huggingface_hub
export MODEL_PATH="${MODEL_PATH:-$(pwd)/.cache/hyperloom-models/gpt-oss-120b}"
python - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(os.environ["MODEL_PATH"]).expanduser()
if (target / "config.json").is_file():
    print(f"Using existing model at {target.resolve()}")
else:
    snapshot_download(
        repo_id="openai/gpt-oss-120b",
        local_dir=str(target),
        local_dir_use_symlinks=False,
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

```bash
set -a; . ./.env; set +a
export USER_DATA_PATH="${USER_DATA_PATH:?USER_DATA_PATH missing}"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
ulimit -Sn 65536 || true
bash hyperloom/inference_optimizer/assets/install.sh
. "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
```

If `hyperloom/inference_optimizer/assets/install.sh` is not present (source
checkout layout), use `src/hyperloom/inference_optimizer/assets/install.sh`.

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
2. Keep `PYTHONPATH="$PWD:${PYTHONPATH:-}"` in the launch shell so robustness
   and critic subprocesses can import `hyperloom.agents` after changing cwd.
3. Run in background with `setsid nohup`.
4. Pass all required optimize CLI flags in the `python -m hyperloom.inference_optimizer.cli optimize` command. Do not rely on `.env` alone for `TP`, `CONC`, `ISL`, `OSL`, or `PRECISION`; CLI defaults can otherwise override the intended workload.
5. Report the session ID, log path, PID, and initial health check result.
6. Monitor the process every 300 seconds until work is done.
7. To recover an unexpected crash, only run `optimize --resume` against the same session dir. After the first launch, never start a new `optimize`; that creates a new `<UTC_ts>` session and is forbidden.
8. If `stop_reason` in the current session `state.json` is final, stop and exit.
