# Pre-release E2E — 3h demo leg

You are running the Hyperloom pre-release E2E test non-interactively. Run the 3-hour
demo to completion, then stop. Setup already ran successfully in this workspace.

Invoke the `hyperloom-qwen3-8b-3h` demo skill with **one** override and otherwise its
exact default flags.

## Flags

- **OVERRIDE:** use `--target-gain 100` (NOT the skill's default of 30). This is the
  pre-release release gate — the run must reach a validated cumulative gain of 100%.
- Keep every other required flag exactly as the skill defines them:

  ```
  --tp 1 --conc 64 --isl 1024 --osl 1024 --precision bf16 --max-hours 3
  --max-minutes-explore-pct 0.39 --max-minutes-sweep-pct 0.01
  --explore-force-exit-budget-pct 0.01 --no-framework-agent --no-kernel
  --no-enable-conc-sweep --no-enable-roofline
  ```

## Model path

The skill will ask which model to use. Do **not** ask interactively — use
`MODEL_PATH` from the repository-root `.env` (it is already set to the demo model).
Verify that path contains `config.json`; if it does, use it and continue without
asking. Load LLM API keys/base URLs and `FRAMEWORK` from `.env`.

## Hard constraints (automated release gate)

- Do **not** modify any GPU-related environment variable or device visibility.
- Do **not** choose GPUs via `rocm-smi`.
- If `HYPERLOOM_RUN_MODE=docker`, run `optimize` **inside the container you started in
  setup** via `docker exec -w "$REPO_ROOT" "$HYPERLOOM_CONTAINER_NAME" …` (per the demo
  skill's docker mode). Do **not** start a new container and do **not** change its
  device/isolation flags. Otherwise (baremetal) run directly and do not run `docker`.
- Do **not** modify `USER_DATA_PATH`.
- Do **not** print or copy secret values into output, reports, or logs.

## Termination — do not end this turn until the run is launched

This is a **single non-interactive turn**, and anything still running as a child of it is
killed the moment the turn ends. So:

1. Finish the install and the launch **inside this turn**. Do **not** end the turn with a
   progress note such as "install started", "waiting on the pull", or "waiting on the
   monitor" — that kills the work you just started and the leg ends up with nothing
   running at all.
2. Start `optimize` **detached** with `setsid nohup` (as the demo skill does) so it
   survives the end of this turn.
3. Before you finish, confirm the run is really live and report the paths: the nested
   session run dir exists, `state.json` is present in it, and the optimizer PID is alive.

Only then stop. The harness then waits for the terminal report (`reports/final.json` +
`reports/final.md`) and judges PASS/FAIL from it — do not fabricate a result.
