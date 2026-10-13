# Pre-release E2E — setup (baremetal + ATOM)

You are running the Hyperloom pre-release E2E test non-interactively. Complete the
setup step for a **baremetal + ATOM** leg, then stop. Do not run the demo yet.

## Environment (already prepared)

A `.env` file exists in the current workspace (`REPO_ROOT`) with these values
already set: `ANTHROPIC_API_KEY`, `CLAUDE_MODEL`, `USER_DATA_PATH`,
`HYPERLOOM_RUN_MODE=baremetal`, `FRAMEWORK=atom`, `MODEL_PATH`. The Hyperloom
package is already installed via `pip install --target .` so a `hyperloom/`
package directory is present.

This container is a plain ROCm PyTorch image: it provides ROCm and ROCm torch
but no ATOM. This leg exists to exercise Hyperloom's own ATOM install
(`--install-framework atom`), which builds AITER and then ATOM from source
against the image's ROCm torch.

## Fixed decisions

Run the `hyperloom-setup` skill with these fixed decisions -- do **not** ask
interactive questions; use the values already in `.env` and the environment:

- **Run mode:** baremetal (`HYPERLOOM_RUN_MODE` is already `baremetal`; keep it).
- **Python:** use the image's `python3` and pin it for setup:

  ```bash
  export PYTHON="$(command -v python3)"
  export INFERENCE_OPTIMIZER_FORCE_PYTHON=1
  ```

- **Framework:** ATOM, installed by setup. Run the setup backend exactly as (through
  `-m hyperloom setup`: `-m hyperloom.inference_optimizer.setup` only imports
  the module and exits 0 without running the installer):

  ```bash
  "$PYTHON" -m hyperloom setup -- \
    --install-framework atom \
    --user-data-path "${USER_DATA_PATH:?USER_DATA_PATH missing}" --yes
  ```

  Do **not** pass `--install-framework none`, and do **not** install ATOM,
  AITER, SGLang or vLLM by hand with `pip`: the point of this leg is that
  Hyperloom's installer does it. The AITER and ATOM source builds take a long
  time; keep monitoring the command until it exits.
- After setup, confirm the install actually took effect in the same Python:

  ```bash
  "$PYTHON" -c "import atom, aiter, torch; print(atom.__file__, torch.version.hip)"
  grep '^FRAMEWORK=' "$REPO_ROOT/.env"
  ```

  The import must succeed, `torch.version.hip` must be non-empty, and `.env`
  must say `FRAMEWORK=atom`. Otherwise report the failure.
- **LLM provider / model / `USER_DATA_PATH`:** use the values already in `.env`;
  do not change them.
- Do **not** change `FRAMEWORK`. Do **not** unset or rewrite
  `KERNEL_OPT_BACKEND_ORDER` if `.env` already carries `forge`.

## Hard constraints (automated release gate)

- Do **not** modify any GPU-related environment variable (`ROCR_VISIBLE_DEVICES`,
  `HIP_VISIBLE_DEVICES`, `GPUS_PER_NODE`, etc.). The pod already exposes exactly one
  GPU; do not override device visibility.
- Do **not** run `docker` and do **not** choose GPUs via `rocm-smi`. This is a
  baremetal leg; setup runs on the host.
- Do **not** print, echo, or copy secret values (API keys) into output or logs.
- Do **not** modify `USER_DATA_PATH`.

## Termination

When setup completes successfully, stop. Report only `setup complete: baremetal/atom`.
If setup hard-fails, report the failure and stop.
