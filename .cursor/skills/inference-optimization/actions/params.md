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

**Version check required before building the grid:**
```bash
VLLM_VERSION=$(python3 -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "0.0.0")
```

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

# --enable-chunked-prefill is supported from vLLM v0.4.3+, but causes OOM/crashes
# on some large MoE models (e.g. Kimi-K2). Only add if vLLM >= 0.6.0.
# NEVER add --num-scheduler-steps — this flag was removed in vLLM v0.4.0.
if python3 -c "from packaging.version import Version; import vllm; assert Version(vllm.__version__) >= Version('0.6.0')" 2>/dev/null; then
    VLLM_PARAM_GRID+=("--enable-chunked-prefill")
fi
```

**FORBIDDEN vLLM flags** (will crash or error on current versions):
- `--num-scheduler-steps` — removed in vLLM v0.4.0, do not use
- `--chunked-prefill-size` — SGLang-only flag, invalid for vLLM

### Test each parameter

For each parameter:
1. Kill server → restart with winning backends + this param → warmup → benchmark
2. Compare output_throughput and TPOT against backend-optimized baseline
3. If improvement > 1%, mark as **KEEP**

| Result | Action |
|--------|--------|
| throughput > baseline + 1% | KEEP |
| throughput within +/-1% | NEUTRAL, skip |
| throughput < baseline - 1% | DISCARD |

### Combine all winning parameters

Test the full combination of all winning backends + all winning params together.

If combined result is worse than individual winners, test subsets to find conflicting pairs.

## Accuracy Validation
Server parameter changes (decode-steps, cuda-graph-max-bs, mem-fraction) have accuracy_risk = 0.0 — they affect scheduling, not computation. No accuracy gate needed for pure scheduling params.

For precision-affecting params (kv-cache-dtype fp8): accuracy_risk = 0.3. **Run the GSM8K
accuracy gate:**
```bash
EVAL_TASK=gsm8k NUM_FEWSHOT=5 PORT=$PORT MODEL=$MODEL \
  RESULTS_DIR="$RESULT_DIR/eval_gsm8k_param_${PARAM_NAME}" \
  bash "$SKILL_ROOT/scripts/eval_accuracy.sh"
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
