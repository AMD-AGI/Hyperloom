# Pre-release E2E — setup (baremetal + vLLM)

You are running the Hyperloom pre-release E2E test non-interactively. Complete the
setup step for a **baremetal + vLLM** leg, then stop. Do not run the demo yet.

## Environment (already prepared)

A `.env` file exists in the current workspace (`REPO_ROOT`) with these values already
set: `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`, `USER_DATA_PATH`,
`HYPERLOOM_RUN_MODE=baremetal`, `FRAMEWORK=vllm`, `MODEL_PATH`, `TARGET_GAIN`,
`DEMO_HOURS`. The wheel is already installed via `pip install --target .` so a
`hyperloom/` package directory is present.

## Fixed decisions

Run the `hyperloom-setup` skill with these fixed decisions — do **not** ask
interactive questions; use the values already in `.env` and the environment:

- **Run mode:** baremetal (`HYPERLOOM_RUN_MODE` is already `baremetal`; keep it).
- **Framework:** vLLM. Install the framework layer with the setup backend
  (`--install-framework vllm`, isolated framework env).
- **LLM provider / model / `USER_DATA_PATH`:** use the values already in `.env`; do
  not change them.

## Hard constraints (automated release gate)

- Do **not** modify any GPU-related environment variable (`ROCR_VISIBLE_DEVICES`,
  `HIP_VISIBLE_DEVICES`, `GPUS_PER_NODE`, etc.). The pod already exposes exactly one
  GPU; do not override device visibility.
- Do **not** run `docker` and do **not** choose GPUs via `rocm-smi`. This is a
  baremetal leg; setup runs on the host.
- Do **not** print, echo, or copy secret values (API keys) into output or logs.
- Do **not** modify `USER_DATA_PATH`.

## Termination

When setup completes successfully, stop. Report only `setup complete: baremetal/vllm`.
If setup hard-fails, report the failure and stop.
