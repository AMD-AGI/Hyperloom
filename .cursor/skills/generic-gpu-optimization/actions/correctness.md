# Action: Correctness Gate

## Inputs
- `$TEST_COMMAND`, `$CORRECTNESS_MODE` (from detected.env)
- `$BUILD_DIR`
- `$ATTEMPT_ID` — current attempt being gated

## Procedure

### Mode: tests
Run the detected test command and require zero exit + zero failures.

```bash
cd "${BUILD_DIR:-$REPO_ROOT}"

START=$(date +%s)
"$SKILL_ROOT/scripts/run_correctness.sh" 2>&1 | tee "$RESULT_DIR/tests-${ATTEMPT_ID}.log"
EXIT=${PIPESTATUS[0]}
DURATION=$(( $(date +%s) - START ))
echo "Tests took ${DURATION}s"

if [ $EXIT -eq 0 ]; then
    echo "CORRECTNESS_PASS" > "$RESULT_DIR/correctness-${ATTEMPT_ID}.txt"
else
    echo "CORRECTNESS_FAIL" > "$RESULT_DIR/correctness-${ATTEMPT_ID}.txt"
    echo "Test failure summary:"
    grep -E "FAIL|FAILED|Error" "$RESULT_DIR/tests-${ATTEMPT_ID}.log" | head -20
fi
```

### Mode: golden-output
Compare benchmark stdout against `$RESULT_DIR/golden.json` saved at attempt 0.

```python
import json, sys, math

def almost_equal(a, b, rtol=1e-3, atol=1e-5):
    if isinstance(a, (list, tuple)):
        return all(almost_equal(x, y, rtol, atol) for x, y in zip(a, b))
    if isinstance(a, dict):
        return all(almost_equal(a[k], b[k], rtol, atol) for k in a if k in b)
    if isinstance(a, (int, float)):
        return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)
    return a == b

golden = json.load(open(sys.argv[1]))
current = json.load(open(sys.argv[2]))

# Compare only numerical output, not timing
fields = ["output", "result", "checksum", "value", "values"]
for f in fields:
    if f in golden and f in current:
        if not almost_equal(golden[f], current[f]):
            print(f"GOLDEN MISMATCH in field '{f}'")
            sys.exit(1)
print("GOLDEN OK")
```

### Mode: none
Issue a one-time warning, then accept the change if the bench just exited cleanly.
Annotate `optimization_report.md` that no correctness gate was active.

```bash
if [ "$CORRECTNESS_MODE" = "none" ]; then
    [ -f "$RESULT_DIR/.no_correctness_warned" ] || {
        echo "WARNING: no test suite or golden output detected. Optimizations will"
        echo "be accepted based only on whether the benchmark exits cleanly."
        echo "Add a TEST_COMMAND to override."
        touch "$RESULT_DIR/.no_correctness_warned"
    }
    echo "CORRECTNESS_SKIP" > "$RESULT_DIR/correctness-${ATTEMPT_ID}.txt"
fi
```

## Outputs
- `$RESULT_DIR/correctness-${ATTEMPT_ID}.txt` — one of `CORRECTNESS_{PASS,FAIL,SKIP}`
- `$RESULT_DIR/tests-${ATTEMPT_ID}.log` — full test output

## Failure Handling
- Test runner crashes (exit > 1, no test output): treat as build/env regression
  → revert and mark INVALID.
- Test timeout (no output for 5 min): kill, mark INVALID, increase timeout for
  next attempt.
- Flaky test (passes on retry): record as `CORRECTNESS_PASS_RETRY`, accept the
  change, but add the test name to `$RESULT_DIR/flaky_tests.txt` for the report.
