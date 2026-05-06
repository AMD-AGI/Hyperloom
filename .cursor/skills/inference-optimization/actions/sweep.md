# Action: Parameter Sweep

Full ISL/OSL/CONC sweep with the optimized version to map the Pareto frontier.

## Inputs
- Final optimized server config (backends + params + kernel patches)
- `$WINNING_BACKEND_ARGS`, `$ALL_WINNING_PARAMS` — combined server args

## Procedure

Magpie launches the optimized server once and reuses it across all sweep cases via
`sweep_matrix`. For N cases, expect 1 × server_startup_time + N × benchmark_duration.

**Claw mode:** Wrap the single `magpie benchmark` call with `exec_on_gpu`.

```bash
# EXTRA_ARGS_KEY: EXTRA_SGLANG_ARGS for sglang, EXTRA_VLLM_ARGS for vllm
EXTRA_ARGS_KEY="EXTRA_$(echo $FRAMEWORK | tr '[:lower:]' '[:upper:]')_ARGS"
SWEEP_DIR="$RESULT_DIR/sweep_$(date +%Y-%m-%d-%H-%M)"
mkdir -p "$SWEEP_DIR"

cat > "$SWEEP_DIR/sweep_config.yaml" <<EOF
benchmark:
  framework: $FRAMEWORK
  model: $MODEL
  precision: fp8
  run_mode: local
  runner_type: $RUNNER_TYPE
  inferencex_path: $INFERENCEX_PATH
  benchmark_script: ${FRAMEWORK}_${RUNNER_TYPE}.sh
  envs:
    TP: $TP
    RANDOM_RANGE_RATIO: 0.5
    $EXTRA_ARGS_KEY: "$WINNING_BACKEND_ARGS $ALL_WINNING_PARAMS"
  timeout_seconds: 7200
  profiler:
    torch_profiler:
      enabled: false
  sweep_matrix:
    cases:
      - { CONC: 4,  ISL: 1024, OSL: 1024 }
      - { CONC: 16, ISL: 1024, OSL: 1024 }
      - { CONC: 64, ISL: 1024, OSL: 1024 }
      - { CONC: 16, ISL: 8192, OSL: 1024 }
      - { CONC: 16, ISL: 1024, OSL: 8192 }
    on_failure: continue
    inter_client_sleep_s: 5
EOF
```

**Run via Background Runner Recipe** (see [`../SKILL.md`](../SKILL.md) "Background Runner Recipe (canonical)"):

- launch: `bash(command="export PATH=/opt/venv/bin:$PATH && magpie benchmark --benchmark-config $SWEEP_DIR/sweep_config.yaml -o $SWEEP_DIR 2>&1", run_in_background=true)` — sweep can run 30+ min for N cases × benchmark_duration
- poll: every 120s call `bash_output(shell_id)`. Sweep emits one `Benchmark Result` per case; treat the run as DONE only when the **last** case completes (regex `All N cases done|Sweep complete` or final `benchmark_report.json` count matches case count). ERROR_REGEX same as default.
- collect: `bash(command="ls -td $SWEEP_DIR/benchmark_* | head -N")` then read each `benchmark_report.json`

### Constraints

- `sweep_matrix.cases` may only override client-side env vars: `CONC`, `ISL`,
  `OSL`, `NUM_PROMPTS`, `RANDOM_RANGE_RATIO`.
- Server-side params (`TP`, backend flags, memory fraction, `EXTRA_*_ARGS`) must
  remain fixed for the whole sweep.
- `profiler.torch_profiler.enabled` must be `false`.
- `run_mode: local` is required. In Claw mode, execute the command inside the
  Ray worker via `exec_on_gpu`.

## Accuracy Validation
N/A — sweep uses the same optimized binary, no new changes to validate.

## Outputs
- `results.tsv` with all configs
- `sweep_report.json` with all per-case results and the best case
- Per-config: (CONC, ISL, OSL, output_tput, tput_per_gpu, TPOT, TTFT)
- Pareto frontier identification
- Per-config `case_*/inferencex_result.json` with raw InferenceX results
- Top-level `benchmark_report.json` populated with the best case for legacy consumers

## Heuristic Update
N/A — sweep is a measurement action, not an optimization action.

## Failure Handling
- Individual config times out or fails: skip, log, and continue by default
- Server fails to start or crashes mid-sweep: fail the sweep and inspect `server.log`
