# Action: Model Classification & Strategy Selection

## Inputs
- `$MODEL` path to model weights directory
- Access to framework test suites (`/sgl-workspace/sglang/test/`, vLLM source)

## KB Query (run before executing)
```
python3 $SKILL_ROOT/kb/kb_query.py "$MODEL_TYPE architecture compatibility" --top-k 5 --compact
```

## Procedure

### Step 0 (MANDATORY): Search for official/CI test configurations FIRST

**CRITICAL LESSON (validated 2026-03-23 on Kimi-K2.5):** Do NOT blindly guess server launch parameters. Many models have non-obvious compatibility constraints (e.g., MLA head count divisibility, split prefill/decode backends, env var overrides from Docker). Guessing wastes 30+ minutes on failed launches.

**BEFORE launching any server, search for existing test configurations:**

```bash
MODEL_TYPE=$(python3 -c "import json; c=json.load(open('$MODEL/config.json')); print(c.get('model_type',''))")
find /sgl-workspace/sglang/test -name "*.py" | xargs grep -il "$MODEL_TYPE\|$(basename $MODEL)" 2>/dev/null

for f in $(find /sgl-workspace/sglang/test -name "*.py" -exec grep -il "$MODEL_TYPE\|$(basename $MODEL)" {} \;); do
    echo "=== $f ==="
    grep -A 5 "other_args\|env\[" "$f" | head -30
done
```

**Extract from test configs:**
- `--decode-attention-backend` / `--prefill-attention-backend` (may differ!)
- `--attention-backend` (unified, simpler but may not work for all models)
- Environment variables: `SGLANG_ROCM_FUSED_DECODE_MLA`, `SGLANG_USE_AITER`, etc.
- `--kv-cache-dtype` (some models crash with fp8 KV cache)
- `--trust-remote-code` (required for custom model code)

**For vLLM:**
```bash
VLLM_PKG=$(python3 -c "import vllm, os; print(os.path.dirname(vllm.__file__))" 2>/dev/null)
find "$VLLM_PKG/../.." -path "*/tests/*" -name "*.py" | xargs grep -il "$MODEL_TYPE\|$(basename $MODEL)" 2>/dev/null | head -10
```

**If no test config found:** Check the model's README, HuggingFace page, or framework docs.

### Step 1: Classify model architecture

```bash
python3 -c "
import json, sys
config = json.load(open('$MODEL/config.json'))
text_cfg = config.get('text_config', config)
arch = text_cfg.get('architectures', config.get('architectures', ['']))[0]
has_mla = text_cfg.get('kv_lora_rank', 0) > 0 or text_cfg.get('q_lora_rank', 0) > 0
has_moe = text_cfg.get('n_routed_experts', 0) > 0 or text_cfg.get('num_local_experts', 0) > 0 or text_cfg.get('num_experts', 0) > 0
has_swa = 'sliding_attention' in str(text_cfg.get('layer_types', [])) or text_cfg.get('sliding_window_size', 0) > 0
has_mamba = text_cfg.get('num_linear_attention_heads', 0) > 0 or 'mamba' in arch.lower() or 'hybrid' in str(text_cfg.get('model_type', '')).lower()
hidden = text_cfg.get('hidden_size', 0)
n_heads = text_cfg.get('num_attention_heads', 0)
print(f'Architecture: {arch}')
print(f'MoE: {has_moe} | MLA: {has_mla} | SWA: {has_swa} | Mamba/Hybrid: {has_mamba} | Hidden: {hidden} | Heads: {n_heads}')
if has_mla and n_heads > 0:
    for tp in [1,2,4,8]:
        hpt = n_heads // tp
        ok = 'OK' if hpt % 16 == 0 else 'NEED split backend (decode=triton, prefill=aiter)'
        print(f'  TP={tp}: {hpt} heads/partition -> {ok}')
if has_mamba:
    print('WARNING: Mamba/hybrid model — torch.compile INCOMPATIBLE, needs --mamba-scheduler-strategy no_buffer on ROCm')
elif has_swa:
    print('WARNING: SWA models are torch.compile/FP8-KV/aiter-attn INCOMPATIBLE')
elif has_mla:
    print('WARNING: MLA models are likely torch.compile INCOMPATIBLE')
"
```

### Step 2: Strategy decision tree

| Model Type | torch.compile | GEAK Expected | Primary Strategy |
|-----------|--------------|--------------|-----------------|
| Dense | Try first | High (Inductor kernels) | torch.compile + GEAK (Strategy A) |
| MoE without MLA, no SWA | Try first | Medium | torch.compile + GEAK if Inductor works |
| **MoE + SWA** | **Incompatible** | **Low** | **CUDA graph coverage + backend exploration + server params** |
| **MoE + MLA** | **Skip** | **Low (~0-2%)** | **Backend exploration → server param tuning + 1 GEAK round** |
| **MoE + MLA + custom attention** | **Skip** | **Low** | **Backend exploration → kernel tuning → combined testing** |
| **MoE + Mamba/Hybrid** | **Incompatible** | **Medium** (Triton FLA kernels) | **GEAK on FLA Triton kernels + server params. ROCm: MUST pass `--mamba-scheduler-strategy no_buffer`** |
| Any model after vendor kernel >50% | — | Skip GEAK | **Backend exploration → server parameter tuning** |

### Step 3: CUDA graph coverage check (always run after baseline launch)

```bash
grep "cuda_graph_bs\|Capture cuda graph" $RESULT_DIR/server_baseline.log
```

**Rule:** Set `--cuda-graph-max-bs` to at least `CONC`.

## Outputs
- `model_class`: dense / moe / moe_mla / moe_swa / moe_mla_nsa
- `torch_compile_viable`: true / false
- `geak_expected_impact`: high / medium / low / skip
- `primary_strategy`: string describing the recommended approach
- `test_config_found`: true / false (and the config if found)

## Heuristic Update
Based on model class, set initial priors for all action scores:
- Dense: boost torch.compile and GEAK scores, moderate backend scores
- MoE+MLA: zero torch.compile, high backend scores, low GEAK scores
- MoE+SWA: zero torch.compile, high CUDA graph + backend scores

## Failure Handling
- If `config.json` is missing or unreadable: ask user for model architecture details
- If test config search finds nothing: proceed with conservative defaults
