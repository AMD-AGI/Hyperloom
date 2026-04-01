# Action: Baseline Benchmark

## Inputs
- Environment set up (from `setup.md`)
- Model classification (from `classify.md`) — determines whether to try torch.compile

## KB Query
```
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME torch.compile baseline" --top-k 3 --compact
```

## Procedure

> **[CLAW MODE]** All commands below must be wrapped with `exec_on_gpu`. Example:
> ```bash
> exec_on_gpu "export MODEL='$MODEL' TP=$TP CONC=$CONC FRAMEWORK=sglang \
>   SGLANG_EXTRA_ARGS='--enable-torch-compile --mem-fraction-static 0.6' \
>   RESULT_DIR='$RESULT_DIR' TRACE_DIR='$TRACE_DIR' INFERENCEX_PATH='$INFERENCEX_PATH' && \
>   bash $SCRIPTS_DIR/run_baseline.sh"
> ```
> Trace files and results are written to shared NFS — accessible from both Claw client and Ray cluster.

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

### Step 3: Record baseline and capture accuracy reference

```bash
# Extract baseline throughput
baseline_tput=$(python3 -c "import json; d=json.load(open('$RESULT_DIR/baseline_*.json')); print(d['output_throughput'])")
baseline_tput_per_gpu=$(python3 -c "print($baseline_tput / $TP)")

# Capture reference output for accuracy gate
curl -s http://localhost:8888/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'$MODEL'","prompt":"The capital of France is","max_tokens":20,"temperature":0}' \
  > $RESULT_DIR/accuracy_reference.json
```

## Accuracy Validation
Baseline establishes the accuracy reference. No validation needed at this step — all subsequent actions compare against this reference.

## Outputs
- `baseline_tput_per_gpu`: tok/s/GPU
- `torch_compile_status`: success / failed (with reason)
- `$RESULT_DIR/baseline_*.json`: benchmark results
- `$RESULT_DIR/server_baseline.log`: server log
- `$RESULT_DIR/accuracy_reference.json`: reference output for accuracy gate
- Server stays running for profiling

## Heuristic Update
- If torch.compile succeeded: boost GEAK kernel optimization scores (Inductor targets available)
- If torch.compile failed: reduce GEAK scores, boost backend exploration and server param scores

## Failure Handling
- If server fails to start: check model compatibility, reduce mem-fraction, try different attention backend
- If benchmark times out: reduce num_prompts, check server health
- Retry up to 3 times with progressively more conservative settings
