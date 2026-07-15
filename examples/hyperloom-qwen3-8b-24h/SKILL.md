---
name: hyperloom-qwen3-8b-24h
description: Run a long-horizon Hyperloom Qwen3-8B optimization session. Use when the user wants the cyclic macro-cycle behavior for a roughly 24-hour demo.
---

# Hyperloom Qwen3-8B Long-Horizon Run

Read `.env` first and resolve `HYPERLOOM_SKILL_PATH`. Read and follow the optimizer skill at `@${HYPERLOOM_SKILL_PATH}` before launching. If `HYPERLOOM_SKILL_PATH` is missing, fall back to `@../../inference_optimizer/SKILL.md`. This skill provides the concrete workload and launch constraints for a long-horizon Qwen3-8B demo.

## Run Mode

Resolve the run mode before launching Hyperloom:

1. If `HYPERLOOM_RUN_MODE=baremetal` or it is unset, run this demo directly on the host.
2. If `HYPERLOOM_RUN_MODE=docker`, this skill owns the Docker setup. Ask the user whether they want a `vllm` or `sglang` Docker image unless `HYPERLOOM_IMAGE` is already set. Use a ROCm image that already contains the selected framework; do not install the framework inside Docker.

In docker mode:
- If `hyperloom-setup` already ran, do **not** re-run setup on the host.
- Always run setup **inside the container** after `docker run`.
- Pass `--install-framework none --yes` in the container (ROCm/framework comes from
  the image). Do **not** use `--skip-base-check` — let Phase 1 preflight validate
  the container environment.
- Do not run `inference_optimizer optimize` on the host.

Suggested Docker images:

- `vllm`: `docker.io/primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix`
- `sglang` MI300X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix`
- `sglang` MI355X: `docker.io/primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix`

In Docker mode, start a long-running container from the host before running setup or optimize:

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

After that, run all remaining commands for this demo inside the same container with `docker exec -w "$PWD" ...`; do not run `inference_optimizer optimize` on the host in Docker mode. When the demo is finished, ask the user whether to stop the container. If they say yes, run:

```bash
docker stop "${HYPERLOOM_CONTAINER_NAME:-hyperloom-local}"
```

## Long-Horizon Gate

Current Hyperloom treats `--max-hours 24` as a long-horizon run. Long-horizon cyclic macro-cycles require one of:

1. `--max-hours >= 24`
2. an unbounded run (`max_minutes == 0`)

`INFERENCE_OPTIMIZER_CYCLIC_PHASES` is enabled by default, but a falsy value (`0`, `false`, `no`, or `off`) disables cyclic macro-cycles. For this skill, ensure it is unset or truthy.

## Environment

- `MODEL_PATH=<optional; if unset, download Qwen/Qwen3-8B from Hugging Face with the Python steps below, then set MODEL_PATH to that local path>`
- `FRAMEWORK=<provided by the existing environment or repository-root .env; do not invent it>`
- `GPU_TYPE=<do not set; omit --gpu-type and let Hyperloom auto-detect from ROCm/system info>`
Required optimize CLI flags:

- `--tp 1`
- `--conc 64`
- `--isl 1024`
- `--osl 1024`
- `--precision fp8`
- `INFERENCE_OPTIMIZER_CYCLIC_PHASES=1`
- `--target-gain 30`
- `--max-hours 24`

Before launch, read the repository-root `.env` file if it exists and load the needed environment variables from it, such as LLM API keys/base URLs, `FRAMEWORK`, and `HF_TOKEN`. Do not copy secret values into the prompt, terminal output, reports, or logs. Do not modify `USER_DATA_PATH`.

If `MODEL_PATH` is unset, ask the user whether they want to provide an existing local model path. If they provide one, set `MODEL_PATH` to that path and verify it contains `config.json`. If they do not provide a path, download `Qwen/Qwen3-8B` automatically. Do not assume the Hugging Face CLI exists; download the model with Python:

```bash
python -m pip install -U huggingface_hub
python - <<'PY'
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path.cwd() / ".cache" / "hyperloom-models" / "Qwen3-8B"
snapshot_download(
    repo_id="Qwen/Qwen3-8B",
    local_dir=str(target),
    local_dir_use_symlinks=False,
)
print(target.resolve())
PY
export MODEL_PATH="$(pwd)/.cache/hyperloom-models/Qwen3-8B"
```

## Launch Requirements

1. Install packages and save artifacts to a writable folder.
2. Run in background with `setsid nohup`.
3. Pass all required optimize CLI flags in the `inference_optimizer optimize` command. Do not rely on `.env` alone for `TP`, `CONC`, `ISL`, `OSL`, or `PRECISION`; CLI defaults can otherwise override the intended workload.
4. Report the session ID, log path, PID, and initial health check result.
5. Monitor the process every 300 seconds until work is done.
6. To recover an unexpected crash, only run `optimize --resume` against the same session dir. After the first launch, never start a new `optimize`; that creates a new `<UTC_ts>` session and is forbidden.
7. If `stop_reason` in the current session `state.json` is final, stop and exit.
