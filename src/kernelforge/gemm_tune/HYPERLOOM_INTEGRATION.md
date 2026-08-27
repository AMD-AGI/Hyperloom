# Hyperloom Integration Guide for kernelforge gemm-tune

## Overview

`kernelforge gemm-tune` is a subcommand of the forge CLI that Hyperloom calls as a subprocess.
The integration replaces the current GEAK-based `gemm_tuning.py` wrapper with a
deterministic, LLM-free tuning path.

## CLI Contract

### Input (CLI args from Hyperloom)

```bash
kernelforge gemm-tune run \
  --model-path <MODEL_PATH> \
  --framework <sglang|vllm> \
  --precision <bf16|fp8|fp4|int8|awq> \
  --quant-type <auto|per_token|blockscale|bpreshuffle|...> \
  --gpu-type <mi300x|mi355x> \
  --tp <TP> \
  --conc <CONC> \
  --tokens <comma-separated> \
  --mp <NUM_GPUS> \
  --output-dir <SESSION_DIR>/runs/forge_gemm_tuning/<request_id> \
  --iters 80 \
  --timeout 3600 \
  --skip-gpu-check \
  [--untuned-csv <path>] \
  [--shapes-json <path>] \
  [--tunableop-input <path>] \
  [--kernel-signature-log <server.log>]
```

### Output (stdout, sentinel-wrapped)

```
FORGE_GEMM_TUNE_RESULT_BEGIN
{
  "status": "ok|skipped|failed",
  "micro_decision": "candidate|no_improvement|skipped|failed",
  "requires_e2e_validation": true|false,
  "recommended_env": {
    "AITER_CONFIG_FMOE": "/path/to/tuned.csv",
    ...
  },
  "tuners_run": [...],
  "artifacts": {...},
  ...
}
FORGE_GEMM_TUNE_RESULT_END
```

### Exit Codes

- 0: Success (may be "ok" or "no_improvement")
- 1: Tuning failed
- 2: Input validation error (missing model, bad config)

## Integration Points in kernel_request_handlers.py

### 1. Replace run_gemm_tuning_handler

The existing handler at `inference_optimizer/orchestrator/kernel_request_handlers.py:1193`
currently calls `kernel-agent/tools/gemm_tuning.py` which depends on GEAK/minisweagent.

Replace with:

```python
async def run_gemm_tuning_handler(payload: dict, *, session_dir: Path) -> HandlerResult:
    # ... existing validation (precision, framework) ...
    
    cmd = [
        "python3", "-m", "kernelforge.cli", "gemm-tune", "run",
        "--model-path", model_path,
        "--framework", framework,
        "--precision", precision,
        "--quant-type", quant_type,
        "--gpu-type", gpu_type,
        "--tp", str(tp),
        "--conc", str(conc),
        "--tokens", ",".join(str(t) for t in token_coverage),
        "--mp", str(available_gpus),
        "--output-dir", str(workspace),
        "--iters", "80",
        "--timeout", str(timeout_sec),
        "--skip-gpu-check",  # Ray handles GPU isolation
    ]
    
    # Pass optional inputs if available
    if untuned_csv:
        cmd.extend(["--untuned-csv", str(untuned_csv)])
    if shapes_json:
        cmd.extend(["--shapes-json", str(shapes_json)])
    
    rc, stdout, stderr = await _run_subprocess(cmd, timeout_sec=timeout_sec)
    result = _parse_sentinel_json(stdout)
    return result
```

### 2. Extend Framework Support

Current policy gate (`policy.py:115`) restricts GEMM tuning to FP8:

```python
FP8_ONLY_ACTIONS: frozenset[str] = frozenset({"gemm_tuning", "run_gemm_tuning"})
```

With kernelforge gemm-tune supporting bf16 MoE, this gate should be relaxed:

```python
# Remove gemm_tuning from FP8_ONLY_ACTIONS
# Add a new gate that checks framework + model type instead
GEMM_TUNING_ELIGIBLE = lambda state: (
    state.framework in ("sglang", "vllm") and
    (state.precision == "fp8" or state.is_moe)
)
```

### 3. E2E Validation Flow

After kernelforge gemm-tune returns `micro_decision: "candidate"`:

1. Hyperloom reads `recommended_env` from the result
2. Restarts the serving session with those env vars injected
3. Runs the standard E2E benchmark (same as current flow)
4. Compares throughput: if >= 3% gain, KEEP; otherwise REVERT

### 4. Phase State Update

In `phase_state.py`, the `gemm_tuning` action should remain in PHASE_KERNEL
but with extended eligibility (not just FP8).

## Data Flow

```
Hyperloom Orchestrator
    | (determines model/framework/precision from session state)
    v
kernelforge gemm-tune run --model-path ... --framework ... --output-dir ...
    | (runs tuner(s), writes artifacts to output-dir)
    v
result.json + tuned CSVs/configs
    | (Hyperloom reads recommended_env)
    v
Hyperloom: restart server with AITER_CONFIG_FMOE=<path>
    | (E2E benchmark)
    v
Hyperloom: KEEP or REVERT decision
```

## Backward Compatibility

- The existing GEAK-based path can remain as a fallback
- kernelforge gemm-tune is preferred when available (check by importing `kernelforge.gemm_tune`)
- The output JSON schema is a superset of the current gemm_tuning.py output
