# Action: Backend & Scheduling Exploration

## Inputs
- Running server with baseline config
- Model classification from `classify.md`
- Profile data from `profile.md` (optional but recommended)

## KB Query
```
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_TYPE backend exploration scheduling" --top-k 10 --compact
python3 $SKILL_ROOT/kb/kb_query.py --category backend_exploration --model "$MODEL_NAME" --compact
```

## Procedure

**Claw mode:** `ServerArgs` inspection and all backend test commands must run via `exec_on_gpu`. See [`../modes/CLAW.md`](../modes/CLAW.md) "Backends" section for wrapper syntax.

### Step 1: Discover all backend and scheduling flags

```bash
python3 -c "
import ast
source = open('/sgl-workspace/sglang/python/sglang/srt/server_args.py').read()
tree = ast.parse(source)
backend_keywords = ['backend', 'enable_', 'disable_', 'fused', 'mixed', 'overlap', 'schedule', 'allreduce', 'fusion']
for node in ast.walk(tree):
    if isinstance(node, ast.Attribute) and any(kw in node.attr for kw in backend_keywords):
        print(f'  --{node.attr.replace(\"_\", \"-\")}')
" 2>/dev/null | sort -u | head -40
```

### Step 2: Identify model-specific backends

```bash
# Attention backends
grep -r "attention_backend\|decode_attention_backend\|prefill_attention_backend" \
    /sgl-workspace/sglang/python/sglang/srt/server_args.py | head -10

# NSA backends (for NSA models)
grep -r "nsa_prefill_backend\|nsa_decode_backend" \
    /sgl-workspace/sglang/python/sglang/srt/server_args.py | head -5

# MoE backends
grep -r "moe_runner_backend\|moe_a2a_backend" \
    /sgl-workspace/sglang/python/sglang/srt/server_args.py | head -5
```

### Step 3: Read the model forward pass (MANDATORY for TP>1)

```bash
MODEL_TYPE=$(python3 -c "import json; c=json.load(open('$MODEL/config.json')); print(c.get('architectures',[''])[0])")
grep -rl "$MODEL_TYPE" /sgl-workspace/sglang/python/sglang/srt/models/ | head -3

# Count all-reduces per layer
grep -n "all_reduce\|reduce_scatter" /sgl-workspace/sglang/python/sglang/srt/models/<model_file>.py

# Check communication path
grep -i "allreduce\|custom_ar\|quick\|AiterCustom\|NCCL" $SERVER_LOG | head -20
```

### Step 4: Build the backend test matrix (tiered)

**TIER 1: Attention/decode backend switches** (highest per-switch impact — change actual GPU kernels)

**TIER 2: Scheduling modes** (change batching/overlap behavior)

**TIER 3: Compute fusion flags** (fuse adjacent operations)

**TIER 4: MoE/GEMM backend switches**

**TIER 5: Communication optimizations**

### Step 5: Test backends individually, then combine winners

For each backend switch:
1. Kill server → restart with baseline + this one change → warmup → benchmark
2. Compare `tput_per_gpu` against baseline
3. If > +1%, mark as **WINNER**

**CRITICAL: Combine ALL winners in a single experiment.**

Individual gains do NOT predict combined gains — switches affecting different pipeline stages produce super-linear synergy (validated: GLM-5 +3.1% + +2.9% → +16.2% combined).

### Step 6: After combining, re-profile

Backend switches that replace vendor C++ kernels with Triton implementations create new GEAK optimization surface. Always re-run profiling after backend exploration settles.

### Step 7: Check for code-level bypasses and fast-path blockers

```bash
grep -rn "bypass\|skip\|fallback\|disabled" /sgl-workspace/aiter/aiter/*.py | head -20
grep -n "is_hip\|_is_hip\|is_cuda" /sgl-workspace/sglang/python/sglang/srt/server_args.py | head -10
```

## Accuracy Validation
After each backend switch, run the accuracy gate:
```bash
curl -s http://localhost:$PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'$MODEL'","prompt":"The capital of France is","max_tokens":20,"temperature":0}' \
  > $RESULT_DIR/accuracy_check.json
# Compare with accuracy_reference.json
```

Backend switches change code paths — accuracy_risk = 0.1.

**After throughput benchmark passes, run the GSM8K accuracy gate:**
```bash
EVAL_TASK=gsm8k NUM_FEWSHOT=5 PORT=$PORT MODEL=$MODEL \
  RESULTS_DIR="$RESULT_DIR/eval_gsm8k_backend_${BACKEND_NAME}" \
  bash "$SKILL_ROOT/scripts/eval_accuracy.sh"
```
Compare `exact_match` against `state.baseline_accuracy`. If accuracy drops by more than
`accuracy_threshold` (default 0.01): REVERT backend flags, mark FAIL.

## Outputs
- `winning_backends`: list of backend flags that improved throughput
- `combined_tput_per_gpu`: throughput with all winners combined
- `combined_gain_pct`: % improvement over baseline
- New profiling data (if re-profiled after combining)

## Heuristic Update
- If backends produced >5% combined gain: boost "combine + re-profile" scores
- If individual backends all <1%: reduce remaining backend scores, boost param tuning
- After re-profile: rescore all GEAK kernel candidates with new GPU time percentages

## Failure Handling
- Server crashes with new backend: revert, mark backend as crash_risk=1.0
- Combination produces negative synergy: test subsets to find conflicting pair
- No backends available for model type: skip to params action
