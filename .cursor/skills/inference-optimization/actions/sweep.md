# Action: Parameter Sweep

Full ISL/OSL/CONC sweep with the optimized version to map the Pareto frontier.

## Inputs
- Final optimized server config (backends + params + kernel patches)
- `$WINNING_BACKEND_ARGS`, `$ALL_WINNING_PARAMS` — combined server args

## Procedure

**Claw mode:** Wrap each `magpie benchmark` call with `exec_on_gpu`. See [`../modes/CLAW.md`](../modes/CLAW.md) "Sweep" section for parallel execution options.

Each config restarts the server. For N configs, expect N × server_startup_time overhead.

```bash
# EXTRA_ARGS_KEY: EXTRA_SGLANG_ARGS for sglang, EXTRA_VLLM_ARGS for vllm
EXTRA_ARGS_KEY="EXTRA_$(echo $FRAMEWORK | tr '[:lower:]' '[:upper:]')_ARGS"
SWEEP_DIR="$RESULT_DIR/sweep_$(date +%Y-%m-%d-%H-%M)"

for CONC_VAL in 4 16 64; do
  for ISL_OSL in "1024:1024" "8192:1024" "1024:8192"; do
    ISL_VAL=${ISL_OSL%%:*}
    OSL_VAL=${ISL_OSL##*:}
    CONFIG="$SWEEP_DIR/conc${CONC_VAL}_isl${ISL_VAL}_osl${OSL_VAL}_config.yaml"
    mkdir -p "$(dirname "$CONFIG")"
    cat > "$CONFIG" <<EOF
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
    CONC: $CONC_VAL
    ISL: $ISL_VAL
    OSL: $OSL_VAL
    RANDOM_RANGE_RATIO: 0.5
    $EXTRA_ARGS_KEY: "$WINNING_BACKEND_ARGS $ALL_WINNING_PARAMS"
  timeout_seconds: 3600
  profiler:
    torch_profiler:
      enabled: false
EOF
    magpie benchmark --benchmark-config "$CONFIG" \
      -o "$SWEEP_DIR/conc${CONC_VAL}_isl${ISL_VAL}_osl${OSL_VAL}"
  done
done
```

### Aggregating sweep results

```bash
echo -e "CONC\tISL\tOSL\toutput_tput\ttput_per_gpu\tTPOT_mean\tTTFT_mean" > "$SWEEP_DIR/results.tsv"
for dir in "$SWEEP_DIR"/conc*/benchmark_*; do
  python3 -c "
import json, os, re
d = json.load(open('$dir/benchmark_report.json'))
parent = os.path.basename(os.path.dirname('$dir'))
m = re.match(r'conc(\d+)_isl(\d+)_osl(\d+)', parent)
conc, isl, osl = (m.group(1), m.group(2), m.group(3)) if m else ('?','?','?')
t = d['throughput']
l = d['latency']
print(f'{conc}\t{isl}\t{osl}\t{t[\"output_throughput\"]:.2f}\t{t[\"output_throughput\"]/$TP:.2f}\t{l[\"tpot\"][\"mean_ms\"]:.2f}\t{l[\"ttft\"][\"mean_ms\"]:.2f}')
" >> "$SWEEP_DIR/results.tsv"
done
```

## Accuracy Validation
N/A — sweep uses the same optimized binary, no new changes to validate.

## Outputs
- `results.tsv` with all configs
- Per-config: (CONC, ISL, OSL, output_tput, tput_per_gpu, TPOT, TTFT)
- Pareto frontier identification
- Per-config `benchmark_report.json` with full Magpie results

## Heuristic Update
N/A — sweep is a measurement action, not an optimization action.

## Failure Handling
- Individual config times out: skip and log, continue sweep
- Server crashes mid-sweep: restart with same config, resume from failed point
