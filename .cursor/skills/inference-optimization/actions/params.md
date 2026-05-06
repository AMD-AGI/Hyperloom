# Action: Server Parameter Tuning

## Inputs
- Winning backend config from `backends.md` (or baseline if backends skipped)
- `baseline_tput_per_gpu` (from current best, not original baseline)

## KB Query
```
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME server parameter tuning" --top-k 5 --compact
python3 $SKILL_ROOT/kb/kb_query.py --category server_params --compact
```

## Procedure

**Claw mode:** All server kill/restart + benchmark commands must use `exec_on_gpu`. See [`../modes/CLAW.md`](../modes/CLAW.md) "Server Params" section for wrapper syntax.

### SGLang parameter grid

Test each parameter independently on top of the winning backend config:

```bash
BASE_ARGS="$WINNING_BACKEND_ARGS"

PARAM_GRID=(
    "--cuda-graph-max-bs $CONC"
    "--num-continuous-decode-steps 8"
    "--num-continuous-decode-steps 16"
    "--num-continuous-decode-steps 32"
    "--mem-fraction-static 0.90"
    "--schedule-conservativeness 0.5"
    "--chunked-prefill-size 65536"
)

NCCL_GRID=(
    "export NCCL_MIN_NCHANNELS=32"
    "export NCCL_ALGO=Ring"
    "export NCCL_ALGO=Tree"
)
```

### vLLM parameter grid

```bash
VLLM_PARAM_GRID=(
    "--gpu-memory-utilization 0.90"
    "--gpu-memory-utilization 0.92"
    "--max-num-seqs 256"
    "--max-num-seqs 512"
    "--max-num-batched-tokens 16384"
    "--max-num-batched-tokens 32768"
    "--compilation-config level=0"
)
```

### Test each parameter

For each parameter, use Magpie to run a short benchmark:

```bash
# Example: test --num-continuous-decode-steps 16
EXTRA_ARGS_KEY="EXTRA_$(echo $FRAMEWORK | tr '[:lower:]' '[:upper:]')_ARGS"
cat > "$RESULT_DIR/param_decode_steps_16_config.yaml" <<EOF
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
    CONC: $CONC
    ISL: $ISL
    OSL: $OSL
    RANDOM_RANGE_RATIO: 0.5
    NUM_PROMPTS: $((CONC * 3))
    $EXTRA_ARGS_KEY: "$WINNING_BACKEND_ARGS --num-continuous-decode-steps 16"
  profiler:
    torch_profiler:
      enabled: false
EOF
```

**Run via Background Runner Recipe** (see [`../SKILL.md`](../SKILL.md) "Background Runner Recipe (canonical)"):

- launch: `bash(command="export PATH=/opt/venv/bin:$PATH && magpie benchmark --benchmark-config $RESULT_DIR/param_decode_steps_16_config.yaml -o $RESULT_DIR/param_decode_steps_16 2>&1", run_in_background=true)`
- poll every 60s with `bash_output(shell_id)` until DONE / ERROR regex
- extract result:
  ```bash
  WORKSPACE=$(ls -td "$RESULT_DIR"/param_decode_steps_16/benchmark_* | head -1)
  new_tput=$(python3 -c "import json; d=json.load(open('$WORKSPACE/benchmark_report.json')); print(d['throughput']['output_throughput'])")
  ```

Compare output_throughput and TPOT against backend-optimized baseline:

| Result | Action |
|--------|--------|
| throughput > baseline + 1% | KEEP |
| throughput within +/-1% | NEUTRAL, skip |
| throughput < baseline - 1% | DISCARD |

### Combine all winning parameters

Test the full combination of all winning backends + all winning params together:

```bash
cat > "$RESULT_DIR/param_combined_config.yaml" <<EOF
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
    CONC: $CONC
    ISL: $ISL
    OSL: $OSL
    RANDOM_RANGE_RATIO: 0.5
    NUM_PROMPTS: $((CONC * 3))
    $EXTRA_ARGS_KEY: "$WINNING_BACKEND_ARGS $ALL_WINNING_PARAMS"
  profiler:
    torch_profiler:
      enabled: false
EOF
```

**Run via Background Runner Recipe** (see [`../SKILL.md`](../SKILL.md) "Background Runner Recipe (canonical)"):

- launch: `bash(command="export PATH=/opt/venv/bin:$PATH && magpie benchmark --benchmark-config $RESULT_DIR/param_combined_config.yaml -o $RESULT_DIR/param_combined 2>&1", run_in_background=true)`
- poll every 60s with `bash_output(shell_id)` until DONE / ERROR regex
- collect: `bash(command="cat $RESULT_DIR/param_combined/benchmark_*/benchmark_report.json")`

If combined result is worse than individual winners, test subsets to find conflicting pairs.

## Accuracy Validation
Server parameter changes (decode-steps, cuda-graph-max-bs, mem-fraction) have accuracy_risk = 0.0 — they affect scheduling, not computation. No accuracy gate needed for pure scheduling params.

For precision-affecting params (kv-cache-dtype fp8): accuracy_risk = 0.3. **Run the GSM8K
accuracy gate** by starting a dedicated eval server (Magpie kills its server after benchmark):
```bash
kill_server 2>/dev/null; check_gpu_memory || exit 1

python3 -m sglang.launch_server \
    --model-path "$MODEL" --host=0.0.0.0 --port $PORT \
    --tensor-parallel-size $TP --trust-remote-code \
    $WINNING_BACKEND_ARGS $ALL_WINNING_PARAMS > "$RESULT_DIR/server_eval.log" 2>&1 &
EVAL_PID=$!
# Wait for health...

EVAL_TASK=gsm8k NUM_FEWSHOT=5 PORT=$PORT MODEL=$MODEL \
  RESULTS_DIR="$RESULT_DIR/eval_gsm8k_param_${PARAM_NAME}" \
  bash "$SKILL_ROOT/scripts/eval_accuracy.sh"

kill $EVAL_PID 2>/dev/null
```
Compare `exact_match` against `state.baseline_accuracy`. If accuracy drops by more than
`accuracy_threshold` (default 0.01): REVERT the param change, mark FAIL.

## Outputs
- `winning_params`: list of parameter flags that improved throughput
- `combined_tput_per_gpu`: throughput with all backends + params
- `combined_gain_pct`: % improvement

## Heuristic Update
- Large param gains (>5%): likely CUDA graph or scheduling, boost similar params
- All params <1%: model is already well-tuned, proceed to kernel optimization or sweep

## Failure Handling
- Server OOM with higher mem-fraction: try lower value
- Server crashes with param combo: test individually to find culprit
