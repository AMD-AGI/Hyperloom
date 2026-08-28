# Pre-release E2E — 12h demo leg

You are running the Hyperloom pre-release E2E test non-interactively. Run the 12-hour
demo to completion, then stop. Setup already ran successfully in this workspace.

Invoke the `hyperloom-qwen3-14b-fp8-12h` demo skill with **one** override and otherwise
its exact default flags.

## Flags

- **OVERRIDE:** use `--target-gain 100` (NOT the skill's default of 50). This is the
  pre-release release gate — the run must reach a validated cumulative gain of 100%.
- Keep every other required flag exactly as the skill defines them:

  ```
  --tp 1 --conc 64 --isl 1024 --osl 1024 --precision fp8 --max-hours 12
  --max-minutes-framework-pct 0.01 --max-minutes-explore-pct 0.42
  --max-minutes-kernel-pct 0.42
  ```

## Model path

The skill will ask which model to use. Do **not** ask interactively — use
`MODEL_PATH` from the repository-root `.env` (it is already set to the demo model,
Qwen3-14B-FP8). Verify that path contains `config.json`; if it does, use it and
continue without asking. Load LLM API keys/base URLs and `FRAMEWORK` from `.env`.

## Hard constraints (automated release gate)

- Do **not** modify any GPU-related environment variable or device visibility.
- Do **not** choose GPUs via `rocm-smi`.
- If `HYPERLOOM_RUN_MODE=docker`, run `optimize` **inside the container you started in
  setup** via `docker exec -w "$REPO_ROOT" "$HYPERLOOM_CONTAINER_NAME" …` (per the demo
  skill's docker mode). Do **not** start a new container and do **not** change its
  device/isolation flags. Otherwise (baremetal) run directly and do not run `docker`.
- Do **not** modify `USER_DATA_PATH`.
- Do **not** print or copy secret values into output, reports, or logs.

## Termination

Let the run proceed to its terminal report (session `reports/final.json` +
`reports/final.md`). When it terminates, stop. The harness judges PASS/FAIL from the
session report — do not fabricate a result.
