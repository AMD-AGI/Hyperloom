# Kernel Optimization (harness, apply safety, E2E retry)

## Pre-GEAK Unittest Harness (unittest skill)

Before `backend=geak` attempts, the main agent generates a GEAK-compatible test
harness by following `kernel-agent/skills/unittest/SKILL.md`. The skill searches
for existing tests, collects shapes/dtypes from TraceLens and profiling data,
and generates a 4-mode harness (`--correctness` / `--profile` / `--benchmark` /
`--full-benchmark`) that matches GEAK's evaluation contract.

The resulting `test_command` is passed via `--test-command` to
`kernel_optimization.py`, which forwards it to GEAK. If the skill fails to
produce a valid harness (after up to 3 retries), `--test-command` is omitted and
GEAK falls back to its own test discovery cascade.

Validation uses `kernel-agent/skills/unittest/validate_harness.py` for both
static checks (argparse + 4 flags + GEAK output markers) and runtime
verification (run correctness + benchmark with reduced iterations).

The Coordinator does NOT need to drive this step — the main agent executes the
unittest skill before calling `kernel_optimization.py`. Observability shows up as
`test_command` in `optimization_attempts.jsonl[].backend_paths`.

The GEAK outer-timeout is managed by `_ensure_yaml_env_timeout()` in
`kernel_optimization.py`, which sets a fallback of 3600s so GEAK's
`LocalEnvironment.timeout` never silently inherits the 30s default.

## Kernel Apply Safety

Kernel optimization may modify `/sgl-workspace/aiter`, `/sgl-workspace/sglang`,
or compiled artifacts. Before applying a patch:

- Back up source files.
- Back up compiled `.so` / `.co` artifacts when available.
- On REVERT, restore compiled artifacts first, then source files, then restart
  the server. Avoid a rebuild on revert when the original compiled artifact was
  backed up.
- Only KEEP when correctness and E2E are acceptable.

If the user has not explicitly approved environment mutation, stop before real
apply/rebuild and ask. Dry-run and analysis are safe.

## Kernel E2E Retry Discipline

Microbench speedups are not enough. After `run_optimization` returns a candidate
kernel patch, `integrate` must validate the patch with E2E Magpie throughput and
record every attempt in `state.json`.

For the same `kernel_id + patch_path + EXTRA_SGLANG_ARGS`:

- `KEEP`: accept only when E2E gain clears the configured threshold.
- `REVERT`: reject that patch immediately and do not run it again.
- `NEEDS_REVIEW`: allow at most 3 E2E attempts. If none clears the KEEP
  threshold, reject that patch and move on to params search or a different
  reusable native kernel.

Do not repeatedly integrate the same patch because its microbench was strong. If
E2E results are unstable around zero gain, the correct action is to mark the
patch rejected, preserve the artifacts for human review, and spend the remaining
budget on untested params/backend candidates or the next kernel.
