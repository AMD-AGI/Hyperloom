# Action: Parameter Sweep

Full ISL/OSL/CONC sweep with the optimized version to map the Pareto frontier.

## Inputs
- Final optimized server config (backends + params + kernel patches)
- `$WINNING_BACKEND_ARGS`, `$ALL_WINNING_PARAMS` — combined server args

## Procedure

**Claw mode:** Multiple sweep execution options (serial via `exec_on_gpu`, SaFE parallel, Ray submit). See [`../modes/CLAW.md`](../modes/CLAW.md) "Sweep" section for all options.

### Option A: Legacy sweep script (recommended for efficiency)

`run_sweep.sh` launches **one server** and runs all benchmark configs against it. This avoids
redundant server restarts, saving 2-10 min per config for large models.

```bash
export MODEL="$MODEL_PATH" TP=$TP INFERENCEX_PATH="$INFERENCEX_PATH"
export SGLANG_EXTRA_ARGS="$WINNING_BACKEND_ARGS $ALL_WINNING_PARAMS"
export CONC_VALUES="4 16 64"
export ISL_OSL_CONFIGS="1024:1024 8192:1024 1024:8192"
export RESULT_DIR="/shared_nfs/inference-optimization/results/sweep_$(date +%Y-%m-%d-%H-%M)"

bash "$SCRIPTS_DIR/run_sweep.sh"
```

### Option B: Magpie per-config sweep

Uses Magpie for each (CONC, ISL, OSL) combination. Each call restarts the server, so this is
slower than Option A but provides richer per-config reports (TraceLens, gap analysis if enabled).

```bash
SWEEP_DIR="$RESULT_DIR/sweep_$(date +%Y-%m-%d-%H-%M)"
for CONC_VAL in 4 16 64; do
  for ISL_OSL in "1024:1024" "8192:1024" "1024:8192"; do
    ISL_VAL=${ISL_OSL%%:*}
    OSL_VAL=${ISL_OSL##*:}
    magpie benchmark $FRAMEWORK \
      -m "$MODEL" --tp $TP --concurrency $CONC_VAL \
      --input-len $ISL_VAL --output-len $OSL_VAL \
      --run-mode local --inferencex-path "$INFERENCEX_PATH" \
      --extra-envs "EXTRA_SGLANG_ARGS=$WINNING_BACKEND_ARGS $ALL_WINNING_PARAMS" \
      -o "$SWEEP_DIR/conc${CONC_VAL}_isl${ISL_VAL}_osl${OSL_VAL}"
  done
done
```

**Time trade-off:** Option B adds ~(N-1) × server_startup_time overhead compared to Option A.
For 9 configs with a 5-min server startup, that's ~40 min extra. Use Option A for pure
throughput sweeps; use Option B when you need per-config trace analysis.

### Aggregating Magpie sweep results

```bash
echo -e "CONC\tISL\tOSL\toutput_tput\ttput_per_gpu\tTPOT_mean\tTTFT_mean" > "$SWEEP_DIR/results.tsv"
for dir in "$SWEEP_DIR"/conc*/benchmark_*; do
  python3 -c "
import json, os
d = json.load(open('$dir/benchmark_report.json'))
name = os.path.basename(os.path.dirname('$dir'))
# parse conc, isl, osl from directory name
t = d['throughput']
l = d['latency']
print(f'{t[\"output_throughput\"]:.2f}\t{t[\"output_throughput\"]/$TP:.2f}\t{l[\"tpot\"][\"mean_ms\"]:.2f}\t{l[\"ttft\"][\"mean_ms\"]:.2f}')
" >> "$SWEEP_DIR/results.tsv"
done
```

## Accuracy Validation
N/A — sweep uses the same optimized binary, no new changes to validate.

## Outputs
- `results.tsv` with all configs
- Per-config: (CONC, ISL, OSL, output_tput, tput_per_gpu, TPOT, TTFT)
- Pareto frontier identification
- (Option B only) Per-config `benchmark_report.json` with full Magpie results

## Heuristic Update
N/A — sweep is a measurement action, not an optimization action.

## Failure Handling
- Individual config times out: skip and log, continue sweep
- Server crashes mid-sweep: restart with same config, resume from failed point
