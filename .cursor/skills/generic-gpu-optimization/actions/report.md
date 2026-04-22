# Action: Final Optimization Report

## Inputs
- `$RESULT_DIR/results.tsv`
- `$RESULT_DIR/kept_env.sh`
- `$RESULT_DIR/kept_cmake.txt`
- `$RESULT_DIR/patches/`
- `$RESULT_DIR/profile.log`
- `$RESULT_DIR/state.env`

## Output: `$RESULT_DIR/optimization_report.md`

### Template

```markdown
# Hyperloom Generic-GPU Optimization Report

**Repo:** `$REPO_ROOT`
**Project class:** `$PROJECT_CLASS`
**GPU:** `$GPU` (`$GPU_ARCH`), ROCm `$ROCM_VERSION`
**Baseline SHA:** `$BASELINE_SHA`
**Run started:** `$START_TIME`
**Run duration:** `$DURATION_MIN` minutes

## Headline

| Metric | Baseline | Optimized | Delta |
|---|---|---|---|
| `$BENCH_METRIC` | `$BASELINE_METRIC` | `$BEST_METRIC` | **`$BEST_DELTA_PCT`%** |

## What worked

| Attempt | Action | Description | Delta |
|---|---|---|---|
… extracted from results.tsv, status=KEEP …

## What didn't

| Attempt | Action | Description | Why discarded |
|---|---|---|---|
… status in {DISCARD, INVALID, NOISE} …

## Reproduction

### 1. Apply environment
```bash
source $RESULT_DIR/kept_env.sh
```

### 2. Apply build flags
```bash
$(cat $RESULT_DIR/kept_cmake.txt)
```

### 3. Apply code patches
```bash
git apply $RESULT_DIR/patches/*.patch
```

### 4. Build & run
```bash
$BUILD_COMMAND
$BENCH_COMMAND
```

## Profile diff

Top-5 kernels (by GPU %):
- Baseline: …
- Optimized: …

## Knowledge contribution

Entries appended to `$SKILL_ROOT/kb/entries.jsonl`:
- New optimizations validated for `$PROJECT_CLASS` on `$GPU_ARCH`
- New pitfalls (e.g. flag X breaks correctness when Y)

## Notes for the next run
… any remaining hot kernels, env vars not yet tried, ideas the agent ran out of
time for …
```

## Procedure
1. Read all inputs.
2. Render the markdown template.
3. Append KB entries via `python3 $SKILL_ROOT/kb/kb_ingest.py < new_entries.jsonl`
   (re-uses the `training-optimization` KB schema — see
   `.cursor/skills/training-optimization/kb/kb_schema.py`).
4. Print the report path to the user.

## Failure Handling
- Missing inputs (e.g. `kept_env.sh` doesn't exist): include "no winning env vars" in the report.
- Empty `results.tsv` beyond baseline: report "no successful optimizations" but
  still write the report so the user has the profile.
