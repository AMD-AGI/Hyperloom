# Action: Target Analysis (NVIDIA / InferenceX Baseline Inspection)

Inspect competitor code, extract optimization techniques, set throughput targets, and adapt techniques for AMD GPUs.

## Inputs
- `$TARGET_DIR`: path to NVIDIA/InferenceX run directory (e.g., `agentic-rc/yanyuan_runs/glm5_optimization`)
- `$MODEL_NAME`: current model being optimized
- Current `baseline_tput_per_gpu` on MI355X

## KB Query
```
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME target comparison NVIDIA" --top-k 5 --compact
python3 $SKILL_ROOT/kb/kb_query.py --category target_comparison --compact
```

## Procedure

### Step 1: Discover target artifacts

```bash
if [ -d "$TARGET_DIR" ]; then
    echo "=== Target directory structure ==="
    find "$TARGET_DIR" -maxdepth 3 -type f \( \
        -name "*.md" -o -name "*.json" -o -name "*.yaml" -o \
        -name "*.py" -o -name "*.sh" -o -name "*.txt" -o -name "*.csv" \
    \) | head -50

    echo "=== Looking for benchmark results ==="
    find "$TARGET_DIR" -name "*result*" -o -name "*benchmark*" -o -name "*report*" | head -20

    echo "=== Looking for server configs ==="
    find "$TARGET_DIR" -name "*config*" -o -name "*launch*" -o -name "*serve*" | head -20

    echo "=== Looking for optimization notes ==="
    find "$TARGET_DIR" -name "*.md" | head -10
fi
```

### Step 2: Extract target throughput numbers

Parse benchmark results to find:
- **Target tok/s/GPU**: the number to beat or match
- **Conditions**: TP, CONC, ISL, OSL used for the target
- **Hardware**: GPU type (B200, H100, A100, etc.)
- **Framework**: TRT-LLM, vLLM, SGLang, etc.

```python
import json, glob, os

target_dir = os.environ.get("TARGET_DIR", "")
results_files = glob.glob(f"{target_dir}/**/result*.json", recursive=True)
results_files += glob.glob(f"{target_dir}/**/benchmark*.json", recursive=True)

for rf in results_files:
    try:
        data = json.load(open(rf))
        tput = data.get("output_throughput", data.get("throughput", data.get("tokens_per_second", 0)))
        tp = data.get("tensor_parallel_size", data.get("tp", "?"))
        conc = data.get("max_concurrency", data.get("concurrency", "?"))
        gpu = data.get("gpu_type", data.get("hardware", "unknown"))
        print(f"  {rf}: {tput} tok/s, TP={tp}, CONC={conc}, GPU={gpu}")
        if tput and tp and tp != "?":
            tput_per_gpu = tput / int(tp)
            print(f"    → {tput_per_gpu:.1f} tok/s/GPU")
    except Exception as e:
        print(f"  {rf}: parse error: {e}")
```

### Step 3: Extract optimization techniques used

Read competitor's optimization reports and code to identify techniques:

```bash
# Read optimization reports/notes
for f in $(find "$TARGET_DIR" -name "*.md" | head -5); do
    echo "=== $(basename $f) ==="
    head -100 "$f"
    echo ""
done

# Look for kernel optimization code
find "$TARGET_DIR" -name "*.py" | xargs grep -l "kernel\|optimize\|triton\|cuda\|flash" 2>/dev/null | head -10

# Look for server launch scripts
find "$TARGET_DIR" -name "*.sh" | xargs grep -l "launch\|serve\|server" 2>/dev/null | head -10
```

### Step 4: Map competitor techniques to MI355X equivalents

| NVIDIA Technique | MI355X Equivalent | Status | Notes |
|-----------------|-------------------|--------|-------|
| TRT-LLM engine | SGLang/vLLM | Available | Different frameworks, different optimizations |
| FlashAttention-3 | aiter MLA/FA | Available | `--attention-backend aiter` |
| FP8 KV cache | FP8 KV cache | Model-dependent | Some models crash (MLA+NSA) |
| DeepGEMM | aiter CK GEMM | Available | `--fp8-gemm-runner-backend` |
| CUDA graphs | ROCm CUDA graphs | Available | `--cuda-graph-max-bs` |
| Custom AllReduce | AiterCustomAllreduce | Available | Auto-detected |
| Piecewise CUDA graphs | **Disabled on ROCm** | Blocked | `disable_piecewise_cuda_graph=True` when `is_hip()` |
| MoE fusion | aiter fused MoE | Available | CK vendor kernels |
| FlashInfer MoE | Not available on ROCm | Blocked | CUDA-only |

### Step 5: Identify blocked techniques and workarounds

For each blocked technique, check if there's a workaround:

```python
blocked = []
workarounds = []

# Check piecewise CUDA graphs
if "piecewise" in target_techniques:
    blocked.append("Piecewise CUDA graphs disabled on ROCm")
    workarounds.append("Use --enable-mixed-chunk for decode/prefill overlap instead")

# Check FlashInfer
if "flashinfer" in target_techniques:
    blocked.append("FlashInfer is CUDA-only")
    workarounds.append("Use aiter attention backend (comparable performance on MI355X)")

# Check specific NVIDIA optimizations
if "fp8_kv" in target_techniques:
    blocked.append(f"FP8 KV cache may crash on {MODEL_NAME} (MLA/NSA models)")
    workarounds.append("Check KB for model-specific FP8 KV compatibility")
```

### Step 6: Calculate target gap and urgency

```python
target_tput_per_gpu = ...  # from Step 2
current_tput_per_gpu = ...  # our best so far

gap_pct = (target_tput_per_gpu - current_tput_per_gpu) / target_tput_per_gpu * 100

if gap_pct <= 0:
    urgency = "EXCEEDED"
    print(f"Already exceeds target by {-gap_pct:.1f}%!")
elif gap_pct <= 10:
    urgency = "CLOSE"
    print(f"Within {gap_pct:.1f}% of target — minor optimizations may close the gap")
elif gap_pct <= 30:
    urgency = "MODERATE"
    print(f"Gap of {gap_pct:.1f}% — need significant optimizations")
else:
    urgency = "LARGE"
    print(f"Gap of {gap_pct:.1f}% — may need architectural changes (DP, quantization)")
```

### Step 7: Prioritize remaining optimizations based on gap

If gap is LARGE (>30%):
- Prioritize: DP scaling, quantization changes, framework switches
- Also: blocked technique workarounds from Step 5

If gap is MODERATE (10-30%):
- Prioritize: backend exploration + combined testing, server params
- Also: adapt specific competitor techniques from Step 3

If gap is CLOSE (<10%):
- Prioritize: kernel tuning, fusion opportunities, OOB optimization on hot source-backed kernels/code paths
- Also: fine-grained parameter sweep

### Step 8: Ingest target data into KB

```bash
python3 $SKILL_ROOT/kb/kb_ingest.py \
    --category target_comparison \
    --model "$MODEL_NAME" \
    --action "Target: $TARGET_GPU $TARGET_TPUT tok/s/GPU vs MI355X $CURRENT_TPUT tok/s/GPU" \
    --lesson "Gap: ${GAP_PCT}%. Techniques: $TECHNIQUES. Blocked on MI355X: $BLOCKED" \
    --tags target,gap-analysis,$TARGET_GPU \
    --source "target-analysis from $TARGET_DIR"
```

## Accuracy Validation
N/A — analysis-only action, no changes to validate.

## Outputs
- `target_tput_per_gpu`: the throughput to beat
- `target_gap_pct`: how far we are from target
- `target_urgency`: EXCEEDED / CLOSE / MODERATE / LARGE
- `competitor_techniques`: list of techniques identified
- `blocked_techniques`: techniques not available on MI355X
- `workarounds`: possible workarounds for blocked techniques
- `prioritized_actions`: reordered action list based on gap

## Heuristic Update
The target gap acts as an urgency multiplier on ALL action scores:
```
score = base_score * (1 + target_gap_pct / 100)
```
- EXCEEDED (gap ≤ 0): multiplier = 1.0 (no urgency)
- CLOSE (gap ≤ 10%): multiplier = 1.1
- MODERATE (gap ≤ 30%): multiplier = 1.3
- LARGE (gap > 30%): multiplier = 1.0 + gap/100 (up to 2.0)

This ensures the agent explores more aggressively when far from target.

## Failure Handling
- Target directory doesn't exist: skip target analysis, use internal heuristics only
- No benchmark results found: try to extract from markdown/text reports
- Target hardware incomparable (e.g., A100 vs MI355X): note the hardware gap, still extract techniques
