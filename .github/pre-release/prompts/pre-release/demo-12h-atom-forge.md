# Pre-release E2E — 12h demo (ATOM serving framework, forge kernel backend)

You are running the Hyperloom pre-release E2E test non-interactively. Run the
12-hour demo to completion, then stop. Setup already ran successfully in this
workspace.

Invoke the `hyperloom-qwen3-14b-fp8-12h-forge` demo skill with **one** override
and otherwise its exact default flags, and follow its **ATOM** section (which
runs the ATOM section of `hyperloom-qwen3-14b-fp8-12h` with
`KERNEL_OPT_BACKEND_ORDER=forge` exported in the ATOM execution shell). This is
the ATOM counterpart of the 12h forge legs: the same workload and budget, on
ATOM, with KERNEL_AGENT on KernelForge. Do **not** run the plain
`hyperloom-qwen3-14b-fp8-12h` skill on its own: without the forge export it runs
GEAK.

## Framework and kernel backend — the point of this leg

`.env` must already carry `FRAMEWORK=atom`. Do **not** change it. Do **not**
switch to SGLang or vLLM.

An unset `KERNEL_OPT_BACKEND_ORDER` means GEAK on ATOM, as on every framework;
`forge` is an exact-match opt-in. The leg runner (`run-leg.sh`) writes
`KERNEL_OPT_BACKEND_ORDER=forge` into `.env` for the `-forge-` legs, and the
demo skill's execution-shell preamble exports it as well. Confirm
`grep '^KERNEL_OPT_BACKEND_ORDER=' "$REPO_ROOT/.env"` prints `forge` before
launching; anything else means the value was lost, so fix it rather than start
a 12-hour GEAK run labelled as this leg.

After launch, confirm the backend actually took effect:

```
grep -o '"kernel_optimizer": *"[^"]*"' "$SESSION_DIR/state.json"
```

It must be `forge`. If it is `geak`, stop and report that the kernel backend
did not resolve to forge.

Do **not** set any other `FORGE_*` variable.

## Flags

- **OVERRIDE:** use `--target-gain 100` (NOT the skill's default of 50). It is
  set out of reach on purpose: the run must not converge early on the skill's
  own target, so the full phase sequence gets exercised. This shapes optimize
  prompts only; the poll gate judges PASS/FAIL from `stop_reason`, not gain,
  so 100 is not a performance goal to chase.
- Keep every other required flag exactly as the skill defines them:

  ```
  --tp 1 --conc 64 --isl 1024 --osl 1024 --precision fp8 --max-hours 12
  --phase-budget-framework-pct 0.43 --phase-budget-kernel-pct 0.42
  ```

  Do **not** pass `--no-framework-agent` or `--no-kernel`.

## Model path

The skill will ask which model to use. Do **not** ask interactively — use
`MODEL_PATH` from the repository-root `.env` (it is already set to the demo
model, Qwen3-14B-FP8). Verify that path contains `config.json`; if it does,
use it and continue without asking. Load LLM API keys/base URLs and
`FRAMEWORK` from `.env`.

## Hard constraints (automated release gate)

- Do **not** modify any GPU-related environment variable or device visibility.
- Do **not** choose GPUs via `rocm-smi`.
- If `HYPERLOOM_RUN_MODE=docker`, run `optimize` **inside the container you
  started in setup** via `docker exec -w "$REPO_ROOT" "$HYPERLOOM_CONTAINER_NAME"
  …` (per the demo skill's docker mode). Do **not** start a new container and
  do **not** change its device/isolation flags. Otherwise (baremetal) run
  directly and do not run `docker`.
- Do **not** modify `USER_DATA_PATH`.
- Do **not** start a background watchdog or automatic resume loop. Launch the
  optimizer itself detached, preserve its logs and launch-info JSON, and let
  the release harness read persisted state and terminal artifacts without
  relaunching the session.
- Do **not** print or copy secret values into output, reports, or logs.

## Termination — do not end this turn until the run is launched

This is a **single non-interactive turn**, and anything still running as a
child of it is killed the moment the turn ends. So:

1. Finish the install and the launch **inside this turn**. Do **not** end the
   turn with a progress note such as "install started" or "waiting on the
   pull" — that kills the work you just started and the leg ends up with
   nothing running at all.
2. Start `optimize` **detached** the way the demo skill does —
   `run_in_background=true` when `$CLAW_SESSION_ID` is set and the bash tool
   offers it, `setsid nohup` otherwise — so it survives the end of this turn.
3. Before you finish, confirm the run is really live and report the paths: the
   nested session run dir exists, `state.json` is present in it, and the
   optimizer PID is alive. Report the `kernel_optimizer` value from
   `state.json` as part of this.

Only then stop. The harness polls `state.json` for a clean terminal
`stop_reason` to judge PASS/FAIL — do not fabricate a result.
