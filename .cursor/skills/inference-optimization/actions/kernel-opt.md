# Action: Kernel Optimization (GEAK / LLM)

Multi-round kernel optimization loop. See also: [`../GEAK-INFERENCE-KERNEL.md`](../GEAK-INFERENCE-KERNEL.md) for GEAK MCP details.

## Inputs
- GEAK candidates from `profile.md` (kernel_name, gpu_pct, source_location)
- Current best config (backends + params)
- `baseline_tput_per_gpu` (after backends + params)

## KB Query
```
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_NAME GEAK kernel optimization" --top-k 5 --compact
python3 $SKILL_ROOT/kb/kb_query.py --category kernel_optimization --compact
```

## Procedure

**FLOW GUARD:** Do NOT skip this action if candidates exist. Running sweep with unoptimized kernels wastes compute.

### Four optimization backends (run ALL in parallel)

| | GEAK | LLM Proxy | Claude Code | Codex |
|---|---|---|---|---|
| **How** | GEAK MCP → remote GPU pod | Direct API → Claude/GPT | `claude-code-sdk` agent | `codex exec` agent |
| **Latency** | 10–30 min | 1–30s | 1–5 min | 1–5 min |
| **Best for** | Complex HIP, final polish | Fast iteration, Triton rewrites | Full autonomy with tool use | Full autonomy with tool use |

### Step 1: Locate kernel source

**Strategy A (torch.compile mode):** Extract from STANDALONE kernel files in Inductor cache.
```bash
find /tmp/torchinductor_root -name "*.py" | while read f; do
    if grep -q "@triton_heuristics" "$f" && \
       ! grep -q "async_compile\|def call(" "$f"; then
        echo "STANDALONE: $f"
    fi
done
```

**Strategy B (no torch.compile):** Find framework source kernels.
```bash
find /opt/venv -path "*/sglang/srt/layers/*.py" -exec grep -l "@triton.jit" {} \;
find /sgl-workspace/aiter -name "*.py" -exec grep -l "@triton.jit" {} \;
```

### Step 2: Submit ALL top candidates to GEAK in parallel

Each kernel is independent — GEAK tasks run on separate pods. Parallel: ~5 min vs serial: ~15 min.

Use prompt templates:
- **RMSNorm / reduction kernels**: TRUE single-pass template (eliminate second loop, 2x memory reduction → +9-14% E2E)
- **General dual-loop kernels**: Merge redundant memory loads
- **Template/GEMM kernels**: Low priority, Inductor autotuner already near-optimal

See `GEAK-INFERENCE-KERNEL.md` for full prompt templates.

### Step 3: Verify + Patch each result individually

For each GEAK output:
1. Verify function name + signature matches original exactly
2. If mismatch: re-submit with stricter prompt (max 3 attempts)
3. Patch standalone files using AST-based replacement
4. Clear binary caches (.so/.json/Triton cache)
5. Kill server, wait 10s, restart
6. Benchmark with EXACTLY same config as baseline

### Step 4: Decide per kernel

| Outcome | Action |
|---------|--------|
| `actual_e2e > 0` | **KEEP**. Update baseline_tput. |
| `actual_e2e <= 0` | **REVERT**. Restore backup. |
| Crashed | **REVERT**. Log as crash. |

### Step 5: Re-profile after kept optimizations

Kernel rankings shift after optimization. Re-profile to find new bottlenecks.

### Stopping criteria

| Condition | Action |
|-----------|--------|
| All top 5 processed AND no new >3% non-vendor candidates | Stop |
| Cumulative E2E gain > 15% | Stop — excellent |
| 5 consecutive discards | Stop — diminishing returns |
| 2+ crashes during patching | Stop — environment unstable |
| Wall clock > 120 min | Stop — time budget |
| Total GEAK submissions > 15 | Stop — cost budget |

## Accuracy Validation
After EACH kept kernel patch, run the accuracy gate:
```bash
curl -s http://localhost:$PORT/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"'$MODEL'","prompt":"The capital of France is","max_tokens":20,"temperature":0}' \
  | python3 -c "
import json, sys
new = json.load(sys.stdin)
ref = json.load(open('$RESULT_DIR/accuracy_reference.json'))
if new['choices'][0]['text'].strip() == ref['choices'][0]['text'].strip():
    print('ACCURACY: PASS')
else:
    print(f'ACCURACY: FAIL — expected [{ref[\"choices\"][0][\"text\"]}] got [{new[\"choices\"][0][\"text\"]}]')
    sys.exit(1)
"
```

Kernel modifications have accuracy_risk = 0.15 (reduction kernels) to 0.05 (pointwise). REVERT immediately on accuracy failure.

## Outputs
- Per-kernel results: (kernel_name, speedup, e2e_gain, status)
- `cumulative_gain_pct`: total improvement from all kept kernels
- Updated baseline with all kept patches applied

## Heuristic Update
- Each kept kernel: boost similar kernel type scores (other reduction kernels likely optimizable too)
- Each discarded kernel: reduce scores for that kernel type
- After 2+ discards on vendor-type kernels: reduce all GEAK scores to near-zero
- Re-profiled new candidates get fresh scores based on new gpu_pct

## Failure Handling
- GEAK workspace unavailable: retry on alternate workspace (3 attempts)
- GEAK produces wrong signature: re-submit with explicit constraint
- Register OOM during Triton compile: reduce block sizes in prompt
- Server crash after patch: revert, log crash, skip kernel
