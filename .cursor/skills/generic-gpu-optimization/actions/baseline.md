# Action: Run Benchmark and Record Metric

## Inputs
- `$BENCH_COMMAND`, `$BENCH_METRIC_REGEX`, `$BENCH_METRIC` (from detected.env)
- `$BUILD_DIR` (from build.md)
- `$ATTEMPT_ID` (e.g. "0" for baseline, "3" for attempt 3)
- `$KEPT_ENV_FILE` (sourceable, contains kept env vars from prior attempts)

## Procedure

### Step 1: Apply kept environment variables
```bash
[ -f "${KEPT_ENV_FILE:-$RESULT_DIR/kept_env.sh}" ] && \
    source "${KEPT_ENV_FILE:-$RESULT_DIR/kept_env.sh}"
```

### Step 2: Run benchmark with timing
```bash
ATTEMPT_LOG="$RESULT_DIR/bench-${ATTEMPT_ID:-0}.log"
ATTEMPT_JSON="$RESULT_DIR/bench-${ATTEMPT_ID:-0}.json"

cd "$REPO_ROOT"
START=$(date +%s)
"$SKILL_ROOT/scripts/run_bench.sh" 2>&1 | tee "$ATTEMPT_LOG"
RUN_EXIT=${PIPESTATUS[0]}
DURATION=$(( $(date +%s) - START ))
echo "Run took ${DURATION}s"

if [ $RUN_EXIT -ne 0 ]; then
    echo "BENCH_FAILED" > "$RESULT_DIR/bench_status.txt"
    exit $RUN_EXIT
fi
```

### Step 3: Extract the metric
```bash
METRIC=$("$SKILL_ROOT/scripts/common.sh" extract_metric "$ATTEMPT_LOG" "$BENCH_METRIC_REGEX")

if [ -z "$METRIC" ]; then
    echo "ERROR: could not extract metric from $ATTEMPT_LOG using regex: $BENCH_METRIC_REGEX"
    echo "First 50 lines of log:"
    head -50 "$ATTEMPT_LOG"
    exit 1
fi
echo "$BENCH_METRIC: $METRIC"
```

### Step 4: Append to results.tsv
```bash
RESULTS="$RESULT_DIR/results.tsv"
[ -f "$RESULTS" ] || echo -e "attempt\tmetric\tdelta_pct\tstatus\tdescription" > "$RESULTS"

if [ "$ATTEMPT_ID" = "0" ]; then
    DELTA="0.00"
    STATUS="baseline"
    BASELINE_METRIC="$METRIC"
    echo "BASELINE_METRIC=$METRIC" >> "$RESULT_DIR/state.env"
else
    BASELINE_METRIC=$(grep -oP '(?<=^BASELINE_METRIC=)[\d.]+' "$RESULT_DIR/state.env")
    if [ "${METRIC_LOWER_IS_BETTER:-true}" = "true" ]; then
        DELTA=$(python3 -c "print(f'{($BASELINE_METRIC - $METRIC) / $BASELINE_METRIC * 100:+.2f}')")
    else
        DELTA=$(python3 -c "print(f'{($METRIC - $BASELINE_METRIC) / $BASELINE_METRIC * 100:+.2f}')")
    fi
    STATUS="pending"  # decided after correctness gate + keep/revert logic
fi

echo -e "$ATTEMPT_ID\t$METRIC\t$DELTA\t$STATUS\t${ATTEMPT_DESCRIPTION:-}" >> "$RESULTS"
```

### Step 5: Multi-run noise check (baseline only)
On `ATTEMPT_ID=0`, run the benchmark 3 times and confirm the metric varies by
< 2%. If noise is high, set `BENCH_REPETITIONS=5` for all subsequent attempts
(via `run_bench.sh`).

```bash
if [ "$ATTEMPT_ID" = "0" ]; then
    METRICS=()
    for i in 1 2 3; do
        "$SKILL_ROOT/scripts/run_bench.sh" > "$RESULT_DIR/noise-$i.log" 2>&1
        m=$("$SKILL_ROOT/scripts/common.sh" extract_metric "$RESULT_DIR/noise-$i.log" "$BENCH_METRIC_REGEX")
        METRICS+=("$m")
    done
    NOISE=$(python3 -c "
import statistics
xs = [${METRICS[@]/#/}]
xs = [float(x) for x in '${METRICS[*]}'.split()]
print(f'{statistics.stdev(xs)/statistics.mean(xs)*100:.2f}')
")
    echo "Baseline noise: ${NOISE}%"
    echo "BENCH_NOISE_PCT=$NOISE" >> "$RESULT_DIR/state.env"
fi
```

## Outputs
- `$RESULT_DIR/results.tsv` updated
- `$RESULT_DIR/state.env` updated with `BASELINE_METRIC` (only on attempt 0)
- `$RESULT_DIR/bench-N.{log,json}` per attempt

## Keep/Revert Decision (applied AFTER correctness.md passes)
Pseudocode:
```
if correctness == FAIL:
    revert; status=INVALID; consecutive_discards++
elif delta_pct < 0:               # regressed
    revert; status=DISCARD; consecutive_discards++
elif abs(delta_pct) < bench_noise_pct:
    revert; status=NOISE; consecutive_discards++  # not enough signal
else:
    keep; status=KEEP; consecutive_discards=0
```

## Failure Handling
- Bench command exits non-zero: revert the change, do NOT record metric.
- Metric extraction fails: print log head, ask user to set `BENCH_METRIC_REGEX`.
