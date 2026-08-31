---
myst:
  html_meta:
    "description": "How to debug KernelForge task preparation: the audit trail forge-loop writes for every prep attempt, what each artifact proves, and the budget knobs that decide whether preparation gets a fair chance."
    "keywords": "KernelForge, task preparation, forge-loop, prepare-task, driver contract, preflight, audit trail, FORGE_PREPARE_MIN_RETRY, debugging"
---

# Debug task preparation

Before a campaign starts, `forge-loop` checks the driver against the driver
contract it enforces at run time. If it does not conform, a
bounded repair loop hands the driver to a prep agent, re-checks it
deterministically after every attempt, and rolls the workspace back if no
attempt produces a conforming driver.

When that fails, the run aborts with `task_preparation_failed` before a single
optimization iteration happens — so it is worth being able to tell *why* it
failed.

## Read the audit trail first

Every prep writes one directory per attempt under
`<experiments-dir>/task_preparation/`:

| Artifact | What it tells you |
|---|---|
| `initial_preflight.json` | Why the caller-supplied driver was rejected, including the tail of what it actually printed |
| `attempt_NN/prompt.md` | Exactly what the agent was asked, including the previous attempt's failure |
| `attempt_NN/driver_before.py` | The driver as the attempt found it |
| `attempt_NN/driver_after.py` (or `driver_at_timeout.py`) | The driver as the attempt left it |
| `attempt_NN/agent_event.json` | `status`, `elapsed_s`, `budget_s`, and `driver_edited` |
| `attempt_NN/agent_progress.txt` | One line per assistant turn and tool call, kept even when the attempt was cancelled |
| `attempt_NN/preflight.json` | The deterministic verdict, `duration_sec`, per-stage `seconds`, and per-stage output tails |

Three questions answer most failures:

1. **Did the agent write anything?** `agent_event.json` → `driver_edited`. An
   attempt that ends `false` produced nothing salvageable; the preflight
   reasons in that case describe the *original* driver, not a failed repair.
   The failure message says so explicitly.
2. **Why did the driver fail?** `preflight.json` → `diagnostics`, which carries
   the driver's own stdout/stderr tail. A bare `DRIVER CRASHED (exit 1)` with
   no traceback means the driver died before printing anything.
3. **Where did the time go?** `preflight.json` → `duration_sec` and the
   per-stage `seconds`, against `agent_event.json` → `elapsed_s` / `budget_s`.

File timestamps in this directory are capture times, so they can be read as a
timeline.

## Give preparation enough budget

Preparation shares the per-kernel deadline with everything else, and the agent
needs real time: authoring a conforming driver takes minutes, not seconds.

| Variable | Default | Meaning |
|---|---|---|
| `FORGE_PREPARE_MAX_WALL` | `3000` | Wall-clock ceiling across all attempts |
| `FORGE_PREPARE_ATTEMPT_CAP` | `900` | Ceiling for one attempt |
| `FORGE_PREPARE_MAX_ATTEMPTS` | `3` | Attempt count ceiling |
| `FORGE_PREPARE_MIN_RETRY` | `350` | Budget a *retry* must have before it is started at all |

The effective budget is `min(FORGE_PREPARE_MAX_WALL, what the per-kernel
deadline leaves)`, and `forge-loop` logs it:

```
[prepare] budget: wall=3000s attempt_cap=900s max_attempts=3
```

If that `wall` is small, preparation is being starved by the per-kernel
deadline rather than by these knobs. A retry that would start with less than
`FORGE_PREPARE_MIN_RETRY` is skipped instead of consuming the tail of the
budget for nothing, and the failure message names the deadline as the lever.
The first attempt always runs, however little time is left.

## Other knobs

| Variable | Default | Meaning |
|---|---|---|
| `FORGE_PREFLIGHT_CORRECTNESS_TIMEOUT` | `1800` | Correctness stage timeout; raise for cold-JIT backends |
| `FORGE_PREFLIGHT_BENCH_TIMEOUT` | `1800` | Benchmark stage timeout |
| `FORGE_PREFLIGHT_GRAPH_TIMEOUT` | `900` | Graph-replay probe timeout |
| `FORGE_PREFLIGHT_PROFILE_TIMEOUT` | `900` | Profiling-contract probe timeout |
| `FORGE_PREFLIGHT_DIAG_CHARS` | `1500` | How much of a failed stage's output is kept per stage |
| `FORGE_EXTERNAL_IGNORE_DIRS` | — | Extra directory names to exclude when the driver lives outside the workspace |

## Drivers outside the workspace

A driver does not have to live in the kernel workspace. When it does not,
preparation stages its directory transactionally and publishes the result back,
so a failed attempt cannot leak edits outside the workspace.

Machine-generated caches next to the driver (`__pycache__`, `flydsl_cache`,
`jit_cache`, `build`) are excluded from that transaction. They must be: the
driver writes to its JIT cache every time it compiles a kernel, and treating
that as transaction state makes an otherwise-successful preparation fail with
`external artifact directory changed outside the staging transaction`. Add any
further cache directory names your toolchain uses to
`FORGE_EXTERNAL_IGNORE_DIRS`.
