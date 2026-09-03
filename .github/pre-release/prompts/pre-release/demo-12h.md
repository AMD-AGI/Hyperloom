# Pre-release E2E — 12h demo leg

You are running the Hyperloom pre-release E2E test non-interactively. Run the 12-hour
demo to completion, then stop. Setup already ran successfully in this workspace.

Invoke the `hyperloom-qwen3-14b-fp8-12h` demo skill with **one** override and otherwise
its exact default flags.

## Flags

- **OVERRIDE:** use `--target-gain 100` (NOT the skill's default of 50). It is set out of
  reach on purpose: the run must not converge early on the skill's own target, so the full
  phase sequence gets exercised. This shapes optimize prompts only; the poll gate judges
  PASS/FAIL from `stop_reason`, not gain, so 100 is not a performance goal to chase.
- Keep every other required flag exactly as the skill defines them:

  ```
  --tp 1 --conc 64 --isl 1024 --osl 1024 --precision fp8 --max-hours 12
  --max-minutes-framework-pct 0.43 --max-minutes-kernel-pct 0.42
  ```

  Do **not** pass `--no-framework-agent` or `--no-kernel` — the 12h demo runs the full
  OPTIMIZE phase (FRAMEWORK_AGENT + KERNEL_AGENT).

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

## Termination — do not end this turn until the run is launched

This is a **single non-interactive turn**, and anything still running as a child of it is
killed the moment the turn ends. So:

1. Finish the install and the launch **inside this turn**. Do **not** end the turn with a
   progress note such as "install started", "waiting on the pull", or "waiting on the
   monitor" — that kills the work you just started and the leg ends up with nothing
   running at all.
2. Start `optimize` **detached** the way the demo skill does — `run_in_background=true` under Claw, `setsid nohup` elsewhere — so it
   survives the end of this turn.
3. Before you finish, confirm the run is really live and report the paths: the nested
   session run dir exists, `state.json` is present in it, and the optimizer PID is alive.

Only then stop. The harness polls `state.json` for a clean terminal `stop_reason` to
judge PASS/FAIL — do not fabricate a result.
