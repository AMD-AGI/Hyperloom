# Action: Baseline Benchmark

## Inputs
- Environment set up (from `setup.md`)
- Model classification (from `classify.md`) — determines whether to try torch.compile

## KB Query
```
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME torch.compile baseline" --top-k 3 --compact
```

## Procedure

**Remote mode:** Wrap all commands below with `exec_on_gpu`. See [`../modes/REMOTE.md`](../modes/REMOTE.md) "Baseline" section for the exact wrapper syntax.

**Try torch.compile first, then fall back if incompatible.**

### Step 1: Try with torch.compile

**For SGLang:**
```bash
export FRAMEWORK=sglang
export SGLANG_EXTRA_ARGS="--enable-torch-compile --mem-fraction-static 0.6 --chunked-prefill-size 32768 --max-prefill-tokens 32768"
bash "$SCRIPTS_DIR/run_baseline.sh"
```

**For vLLM (torch.compile enabled by default at level=3):**
```bash
export FRAMEWORK=vllm
export VLLM_EXTRA_ARGS="--max-model-len 4096"
bash "$SCRIPTS_DIR/run_baseline.sh"
```

**NOTE:** `--mem-fraction-static` is model-dependent. torch.compile needs extra memory — use 0.6 (vs 0.8 without compile). Override via `MEM_FRACTION` env var.

### Step 2: Check for torch.compile failure

| Error pattern | Cause | Action |
|---------------|-------|--------|
| `get_heuristic_kernel_mla: cannot get heuristic kernel! q_type:fp8` | MLA + FP8 incompatible | Fall back |
| `CUDA error: out of memory` during Triton compilation | Model too large for 0.6 mem fraction | Try 0.5, then fall back |
| `Triton compilation failed` / `inductor error` | Unsupported op | Fall back |

**If torch.compile fails (SGLang):**
```bash
export SGLANG_EXTRA_ARGS="--chunked-prefill-size 196608 --max-prefill-tokens 196608 --mem-fraction-static 0.8"
bash "$SCRIPTS_DIR/run_baseline.sh"
```

**If torch.compile fails (vLLM):**
```bash
export VLLM_EXTRA_ARGS="--max-model-len 4096 --enforce-eager"
bash "$SCRIPTS_DIR/run_baseline.sh"
```

### Step 3: Record baseline throughput

```bash
# Extract baseline throughput
baseline_tput=$(python3 -c "import json; d=json.load(open('$RESULT_DIR/baseline_*.json')); print(d['output_throughput'])")
baseline_tput_per_gpu=$(python3 -c "print($baseline_tput / $TP)")
```

### Step 4: Run baseline accuracy evaluation (GSM8K)

**This is mandatory.** The baseline GSM8K score is the reference for all subsequent
accuracy gates. Any action with `accuracy_risk > 0` will be compared against this score.

```bash
# Run GSM8K 5-shot eval against the baseline server
EVAL_TASK=gsm8k NUM_FEWSHOT=5 PORT=$PORT MODEL=$MODEL \
  RESULTS_DIR="$RESULT_DIR/eval_gsm8k_baseline" \
  bash "$SKILL_ROOT/scripts/eval_accuracy.sh"

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
- `$RESULT_DIR/baseline_*.json`: benchmark results
- `$RESULT_DIR/server_baseline.log`: server log
- `$RESULT_DIR/eval_gsm8k_baseline/`: full GSM8K eval results + summary
- `$RESULT_DIR/accuracy_reference.json`: greedy reference output (fast sanity check)
- Server stays running for profiling

## Heuristic Update
- If torch.compile succeeded: boost GEAK kernel optimization scores (Inductor targets available)
- If torch.compile failed: reduce GEAK scores, boost backend exploration and server param scores

## Failure Handling
- If server fails to start: check model compatibility, reduce mem-fraction, try different attention backend
- If benchmark times out: reduce num_prompts, check server health
- Retry up to 3 times with progressively more conservative settings
