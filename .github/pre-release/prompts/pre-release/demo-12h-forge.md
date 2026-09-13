# Pre-release E2E — 12h demo leg (forge kernel backend)

You are running the Hyperloom pre-release E2E test non-interactively. Run the 12-hour
demo to completion, then stop. Setup already ran successfully in this workspace.

Invoke the `hyperloom-qwen3-14b-fp8-12h-forge` demo skill with **one** override and
otherwise its exact default flags. This is the forge counterpart of the `12h` leg: the
same workload and budget, with the KERNEL_AGENT phase on the per-kernel KernelForge
backend instead of GEAK.

## Kernel backend — the point of this leg

`KERNEL_OPT_BACKEND_ORDER=forge` must be set in the environment that launches
`optimize`. The opt-in is an **exact** match on `forge`; every other value, including
unset, silently leaves GEAK owning the kernel phase. There is no CLI flag for it.

Two ways this silently degrades into a duplicate of the plain 12h leg — avoid both:

- The demo skill's pre-launch install block sources `.env`, which sets
  `KERNEL_OPT_BACKEND_ORDER` to whatever that file carries, and then replays the
  caller's pre-existing exports on top via `eval "$_dotenv_prev"`. Export `forge`
  **after** that whole block, not before.
- In docker mode the variable must be set **inside the same `docker exec`** that runs
  `optimize`. Exporting it on the host does not reach the optimizer.

Before launching, confirm `.env` still carries the backend the setup turn selected —
`grep '^KERNEL_OPT_BACKEND_ORDER=' "$REPO_ROOT/.env"` must print `forge`. Anything else
means the value was lost between setup and here; fix that before starting a 12-hour run.

After launch, confirm the backend actually took effect by reading the resolved value the
optimizer recorded in the session `state.json`:

```
grep -o '"kernel_optimizer": *"[^"]*"' "$SESSION_DIR/state.json"
```

It must be `forge`. If it is `geak`, stop and report that the environment variable did
not reach the optimizer — do not let a 12-hour run continue as an unlabelled GEAK run,
because it would report PASS while testing nothing this leg exists to test.

Do **not** set any other `FORGE_*` variable. They are internal tuning knobs with working
defaults and are not part of this leg.

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
  OPTIMIZE phase (FRAMEWORK_AGENT + KERNEL_AGENT), and `--no-kernel` would skip the very
  phase this leg exists to exercise.

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
- **Do** start `robustness_monitor.sh` as the optimizer skill's monitoring section
  describes. It resumes only a run that died with no `stop_reason`, no `phase=CLOSE` and
  no `final.md`, so it cannot rewrite the terminal this gate reads. Skipping it makes
  this leg less crash-tolerant than its siblings and the results stop being comparable.
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
   Report the `kernel_optimizer` value from `state.json` as part of this.

Only then stop. The harness polls `state.json` for a clean terminal `stop_reason` to
judge PASS/FAIL — do not fabricate a result.
