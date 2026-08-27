# Pre-release E2E — setup (docker + SGLang)

You are running the Hyperloom pre-release E2E test non-interactively. Complete the
setup step for a **docker + SGLang** leg, then stop. Do not run the demo yet.

> **IMPORTANT:** you are **already** running inside the backend container. The nested
> container was started for you by the test harness (`docker-run-hyperloom.sh`) and is
> bound to exactly one GPU. You must **not** start, run, or exec any further container,
> and you must **not** run `docker` at all. Treat this environment as the place where
> setup and the demo run directly.

## Environment (already prepared)

A `.env` file exists in the current workspace (`REPO_ROOT`) with these values already
set: `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`, `USER_DATA_PATH`,
`HYPERLOOM_RUN_MODE=docker`, `FRAMEWORK=sglang`, `MODEL_PATH`, `TARGET_GAIN`,
`DEMO_HOURS`. The wheel is already installed via `pip install --target .` so a
`hyperloom/` package directory is present.

## Fixed decisions

Run the `hyperloom-setup` skill with these fixed decisions — do **not** ask
interactive questions; use the values already in `.env` and the environment:

- **Run mode:** docker, but the container already exists and IS the current shell. Do
  **not** create a container and do **not** set `HYPERLOOM_DOCKER_TARGET_HOST`. Run
  setup directly in this shell.
- **Framework:** SGLang. Ensure the SGLang framework layer is available in this
  container (install with the setup backend if needed).
- **LLM provider / model / `USER_DATA_PATH`:** use the values already in `.env`; do
  not change them.

## Hard constraints (automated release gate)

- Do **not** modify any GPU-related environment variable (`ROCR_VISIBLE_DEVICES` is
  already `0` and pins this container to its single card; leave it). Do not override
  device visibility.
- Do **not** run `docker`, do **not** start/exec containers, and do **not** choose
  GPUs via `rocm-smi`.
- Do **not** print, echo, or copy secret values (API keys) into output or logs.
- Do **not** modify `USER_DATA_PATH`.

## Termination

When setup completes successfully, stop. Report only `setup complete: docker/sglang`.
If setup hard-fails, report the failure and stop.
