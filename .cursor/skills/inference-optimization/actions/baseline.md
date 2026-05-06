# Action: Baseline Benchmark

## Inputs
- Environment set up (from `setup.md`)
- Model classification (from `classify.md`) — determines whether to try torch.compile

## KB Query
```
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME torch.compile baseline" --top-k 3 --compact
```

## Procedure

**Claw mode:** Wrap all `magpie benchmark` commands below with `exec_on_gpu`. See [`../modes/CLAW.md`](../modes/CLAW.md) "Baseline" section for the exact wrapper syntax.

**Framework constraint:** `framework:` in the YAML below MUST equal user-specified
`$FRAMEWORK`. Never substitute a different framework even when InferenceX has no
ready-made script for `${MODEL}_${FRAMEWORK}` — see SKILL.md Common Pitfalls #3 and
setup.md Step 1b.

**Try torch.compile first, then fall back if incompatible.**

### Step 1: Try with torch.compile

**For SGLang:**
```bash
cat > "$RESULT_DIR/baseline_compile_config.yaml" <<EOF
benchmark:
  framework: sglang
  model: $MODEL
  precision: fp8
  run_mode: local
  runner_type: $RUNNER_TYPE
  inferencex_path: $INFERENCEX_PATH
  benchmark_script: sglang_${RUNNER_TYPE}.sh
  envs:
    TP: $TP
    CONC: $CONC
    ISL: $ISL
    OSL: $OSL
    RANDOM_RANGE_RATIO: 0.5
    EXTRA_SGLANG_ARGS: "--enable-torch-compile --mem-fraction-static 0.6 --chunked-prefill-size 32768 --max-prefill-tokens 32768"
  timeout_seconds: 3600
  profiler:
    torch_profiler:
      enabled: false
EOF
```

**Run via Background Runner Recipe** (see [`../SKILL.md`](../SKILL.md) "Background Runner Recipe (canonical)"):

- launch: `bash(command="export PATH=/opt/venv/bin:$PATH && magpie benchmark --benchmark-config $RESULT_DIR/baseline_compile_config.yaml -o $RESULT_DIR/baseline_compile 2>&1", run_in_background=true)` — keep the shell_id
- poll: every 60s call `bash_output(shell_id)` until DONE_REGEX (`Benchmark Result|benchmark_report\.json|✅`) or ERROR_REGEX (`Traceback|exit [1-9]|signal=SIG|OOM`) matches
- collect: `bash(command="cat $RESULT_DIR/baseline_compile/benchmark_*/benchmark_report.json")`

**For vLLM (torch.compile enabled by default at level=3):**
```bash
cat > "$RESULT_DIR/baseline_compile_config.yaml" <<EOF
benchmark:
  framework: vllm
  model: $MODEL
  precision: fp8
  run_mode: local
  runner_type: $RUNNER_TYPE
  inferencex_path: $INFERENCEX_PATH
  benchmark_script: vllm_${RUNNER_TYPE}.sh
  envs:
    TP: $TP
    CONC: $CONC
    ISL: $ISL
    OSL: $OSL
    RANDOM_RANGE_RATIO: 0.5
    EXTRA_VLLM_ARGS: "--max-model-len 4096"
  timeout_seconds: 3600
  profiler:
    torch_profiler:
      enabled: false
EOF
```

**Run via Background Runner Recipe** (see [`../SKILL.md`](../SKILL.md) "Background Runner Recipe (canonical)"):

- launch: `bash(command="export PATH=/opt/venv/bin:$PATH && magpie benchmark --benchmark-config $RESULT_DIR/baseline_compile_config.yaml -o $RESULT_DIR/baseline_compile 2>&1", run_in_background=true)` — keep the shell_id
- poll: every 60s call `bash_output(shell_id)` until DONE_REGEX (`Benchmark Result|benchmark_report\.json|✅`) or ERROR_REGEX (`Traceback|exit [1-9]|signal=SIG|OOM`) matches
- collect: `bash(command="cat $RESULT_DIR/baseline_compile/benchmark_*/benchmark_report.json")`

**NOTE:** `--mem-fraction-static` is model-dependent. torch.compile needs extra memory — use 0.6 (vs 0.8 without compile).

### Step 2: Check for torch.compile failure

| Error pattern | Cause | Action |
|---------------|-------|--------|
| `get_heuristic_kernel_mla: cannot get heuristic kernel! q_type:fp8` | MLA + FP8 incompatible | Fall back |
| `CUDA error: out of memory` during Triton compilation | Model too large for 0.6 mem fraction | Try 0.5, then fall back |
| `Triton compilation failed` / `inductor error` | Unsupported op | Fall back |

Check `$RESULT_DIR/baseline_compile/benchmark_*/benchmark_stderr.log` for error patterns.

**If torch.compile fails (SGLang):**
```bash
cat > "$RESULT_DIR/baseline_eager_config.yaml" <<EOF
benchmark:
  framework: sglang
  model: $MODEL
  precision: fp8
  run_mode: local
  runner_type: $RUNNER_TYPE
  inferencex_path: $INFERENCEX_PATH
  benchmark_script: sglang_${RUNNER_TYPE}.sh
  envs:
    TP: $TP
    CONC: $CONC
    ISL: $ISL
    OSL: $OSL
    RANDOM_RANGE_RATIO: 0.5
    EXTRA_SGLANG_ARGS: "--chunked-prefill-size 196608 --max-prefill-tokens 196608 --mem-fraction-static 0.8"
  timeout_seconds: 3600
  profiler:
    torch_profiler:
      enabled: false
EOF
```

**Run via Background Runner Recipe** (see [`../SKILL.md`](../SKILL.md) "Background Runner Recipe (canonical)"):

- launch: `bash(command="export PATH=/opt/venv/bin:$PATH && magpie benchmark --benchmark-config $RESULT_DIR/baseline_eager_config.yaml -o $RESULT_DIR/baseline_eager 2>&1", run_in_background=true)` — keep the shell_id
- poll: every 60s call `bash_output(shell_id)` until DONE_REGEX (`Benchmark Result|benchmark_report\.json|✅`) or ERROR_REGEX (`Traceback|exit [1-9]|signal=SIG|OOM`) matches
- collect: `bash(command="cat $RESULT_DIR/baseline_eager/benchmark_*/benchmark_report.json")`

**If torch.compile fails (vLLM):**
```bash
cat > "$RESULT_DIR/baseline_eager_config.yaml" <<EOF
benchmark:
  framework: vllm
  model: $MODEL
  precision: fp8
  run_mode: local
  runner_type: $RUNNER_TYPE
  inferencex_path: $INFERENCEX_PATH
  benchmark_script: vllm_${RUNNER_TYPE}.sh
  envs:
    TP: $TP
    CONC: $CONC
    ISL: $ISL
    OSL: $OSL
    RANDOM_RANGE_RATIO: 0.5
    EXTRA_VLLM_ARGS: "--max-model-len 4096 --enforce-eager"
  timeout_seconds: 3600
  profiler:
    torch_profiler:
      enabled: false
EOF
```

**Run via Background Runner Recipe** (see [`../SKILL.md`](../SKILL.md) "Background Runner Recipe (canonical)"):

- launch: `bash(command="export PATH=/opt/venv/bin:$PATH && magpie benchmark --benchmark-config $RESULT_DIR/baseline_eager_config.yaml -o $RESULT_DIR/baseline_eager 2>&1", run_in_background=true)` — keep the shell_id
- poll: every 60s call `bash_output(shell_id)` until DONE_REGEX (`Benchmark Result|benchmark_report\.json|✅`) or ERROR_REGEX (`Traceback|exit [1-9]|signal=SIG|OOM`) matches
- collect: `bash(command="cat $RESULT_DIR/baseline_eager/benchmark_*/benchmark_report.json")`

### Step 3: Record baseline throughput

Magpie writes structured results to `benchmark_report.json` in the workspace:

```bash
WORKSPACE=$(ls -td "$RESULT_DIR"/baseline_*/benchmark_* | head -1)
baseline_tput=$(python3 -c "
import json
d = json.load(open('$WORKSPACE/benchmark_report.json'))
print(d['throughput']['output_throughput'])
")
baseline_tput_per_gpu=$(python3 -c "print($baseline_tput / $TP)")
```

### Step 4: Run baseline accuracy evaluation (GSM8K)

**This is mandatory.** The baseline GSM8K score is the reference for all subsequent
accuracy gates. Any action with `accuracy_risk > 0` will be compared against this score.

Since Magpie kills the server after benchmark completes, start a dedicated server for eval:

```bash
# Launch server for accuracy eval (same config as baseline)
# Use the same EXTRA_*_ARGS that produced the baseline
kill_server 2>/dev/null; check_gpu_memory || exit 1

# Start server (SGLang example)
python3 -m sglang.launch_server \
    --model-path "$MODEL" --host=0.0.0.0 --port $PORT \
    --tensor-parallel-size $TP --trust-remote-code \
    --mem-fraction-static 0.8 --disable-radix-cache \
    $BASELINE_SGLANG_ARGS > "$RESULT_DIR/server_eval.log" 2>&1 &
EVAL_SERVER_PID=$!
# Wait for health...

EVAL_TASK=gsm8k NUM_FEWSHOT=5 PORT=$PORT MODEL=$MODEL \
  RESULTS_DIR="$RESULT_DIR/eval_gsm8k_baseline" \
  bash "$SKILL_ROOT/scripts/eval_accuracy.sh"

kill $EVAL_SERVER_PID 2>/dev/null

# Extract baseline accuracy
baseline_accuracy=$(python3 -c "
import json, glob
f = sorted(glob.glob('$RESULT_DIR/eval_gsm8k_baseline/eval_summary_gsm8k.json'))[-1]
d = json.load(open(f))
scores = list(d['scores'].values())[0]
print(scores.get('exact_match,strict-match', scores.get('exact_match,none', 0)))
")
echo "Baseline GSM8K accuracy: $baseline_accuracy"
```

Set `state.baseline_accuracy = baseline_accuracy`. This becomes the hard floor — any
action that drops accuracy by more than `accuracy_threshold` (default 1 percentage point)
is automatically reverted.

### Step 5: Capture greedy reference output (fast sanity check)

Run a quick inference before killing the eval server (or start a temporary one):
```bash
curl -s http://localhost:$PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'$MODEL'","prompt":"The capital of France is","max_tokens":20,"temperature":0}' \
  > $RESULT_DIR/accuracy_reference.json
```

This is a lightweight reference for quick sanity checks during the DFS loop. It does NOT
replace the GSM8K gate — see the Accuracy Gate Protocol in SKILL.md.

## Outputs
- `baseline_tput_per_gpu`: tok/s/GPU
- `baseline_accuracy`: GSM8K exact_match score (0.0–1.0)
- `torch_compile_status`: success / failed (with reason)
- `$WORKSPACE/benchmark_report.json`: Magpie structured results (throughput + latency)
- `$WORKSPACE/benchmark_stdout.log`: benchmark output
- `$RESULT_DIR/eval_gsm8k_baseline/`: full GSM8K eval results + summary
- `$RESULT_DIR/accuracy_reference.json`: greedy reference output (fast sanity check)

## Magpie Result Schema

```json
{
  "throughput": {
    "output_throughput": 1234.56,
    "request_throughput": 12.34,
    "total_token_throughput": 2345.67,
    "completed_requests": 96,
    "duration_seconds": 45.2
  },
  "latency": {
    "ttft": {"mean_ms": 45.2, "median_ms": 42.0, "p99_ms": 120.5, "std_ms": 15.3},
    "tpot": {"mean_ms": 8.1, "median_ms": 7.5, "p99_ms": 15.2, "std_ms": 3.1},
    "itl":  {"mean_ms": 8.3, "median_ms": 7.8, "p99_ms": 16.0, "std_ms": 3.4},
    "e2el": {"mean_ms": 850.0, "median_ms": 820.0, "p99_ms": 1200.0, "std_ms": 150.0}
  }
}
```

Key mapping to skill state:
- `state.baseline_tput_per_gpu = throughput.output_throughput / TP`
- `state.current_tput_per_gpu = throughput.output_throughput / TP`

## Heuristic Update
- If torch.compile succeeded: boost GEAK kernel optimization scores (Inductor targets available)
- If torch.compile failed: reduce GEAK scores, boost backend exploration and server param scores

## Failure Handling
- If server fails to start: check `benchmark_stderr.log`, reduce mem-fraction, try different attention backend
- If benchmark times out: reduce num_prompts by adding `NUM_PROMPTS: 48` to the YAML `envs`, check server health
- Retry up to 3 times with progressively more conservative settings
