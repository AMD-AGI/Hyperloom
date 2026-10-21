# Pre-release E2E — setup (framework-image + ATOM)

You are running the Hyperloom pre-release E2E test non-interactively.
Complete the setup step for this leg, then stop. Do not run the demo yet.

## Environment (already prepared)

A `.env` file exists in the current workspace (`REPO_ROOT`) with these values
already set: `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`, `USER_DATA_PATH`,
`HYPERLOOM_RUN_MODE=baremetal`, `FRAMEWORK=atom`, `MODEL_PATH`. The Hyperloom
package is already installed via `pip install --target .` so a `hyperloom/`
package directory is present.

This container's image already has ATOM installed (it is the image
Hyperloom's own docker-mode ATOM example would otherwise have started itself);
there is no nested container here to start.

## Fixed decisions

Run the `hyperloom-setup` skill with these fixed decisions -- do **not** ask
interactive questions:

- **Run mode:** baremetal (`HYPERLOOM_RUN_MODE` is already `baremetal`; keep
  it). Do **not** run `docker` -- this leg's container IS the ATOM
  environment; there is nothing to nest.
- **Python:** use the image's existing ATOM Python and pin it for setup:

  ```bash
  export PYTHON="$(command -v python3)"
  export INFERENCE_OPTIMIZER_FORCE_PYTHON=1
  ```

- **Framework:** ATOM, already provided by this image. Run the read-only check
  first, then the setup, exactly as (through
  `-m hyperloom setup`: `-m hyperloom.inference_optimizer.setup` only imports
  the module and exits 0 without running the installer):

  ```bash
  "$PYTHON" -m hyperloom setup --check-only -- \
    --install-framework none --frameworks atom --require-frameworks \
    --user-data-path "${USER_DATA_PATH:?USER_DATA_PATH missing}"
  "$PYTHON" -m hyperloom setup -- \
    --install-framework none --frameworks atom --require-frameworks \
    --user-data-path "${USER_DATA_PATH:?USER_DATA_PATH missing}" --yes
  ```

  Do **not** pass `--install-framework atom`: reinstalling on top of the
  image's own build is what this leg exists to avoid. If the check fails, stop
  and report it; do not try to install ATOM, SGLang or vLLM to compensate.
- **LLM provider / model / `USER_DATA_PATH`:** use the values already in
  `.env`; do not change them.
- Do **not** change `FRAMEWORK`. Do **not** unset or rewrite
  `KERNEL_OPT_BACKEND_ORDER` if `.env` already carries `forge`.

## Hard constraints

- Do **not** modify any GPU-related environment variable
  (`ROCR_VISIBLE_DEVICES`, `HIP_VISIBLE_DEVICES`, `GPUS_PER_NODE`, etc.). This
  container already exposes exactly one GPU by index; do not override device
  visibility and do not choose GPUs via `rocm-smi`.
- Do **not** print, echo, or copy secret values (API keys) into output or
  logs.
- Do **not** modify `USER_DATA_PATH`.

## Termination

When setup completes successfully, stop. Report only
`setup complete: baremetal/atom`. If setup hard-fails, report the failure and
stop.
