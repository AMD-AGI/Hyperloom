# kernelforge gemm-tune

Deterministic GEMM tuning CLI for AMD GPUs. Supports sglang and vLLM frameworks.

No LLM dependency — all tuning is exhaustive search via aiter CK tuners or PyTorch TunableOp.

## Install

Nothing to install separately: this is a subpackage of `kernelforge`, and its
commands hang off the one forge CLI.

```bash
pip install -e .            # from the Hyperloom repo root
kernelforge gemm-tune --help
```

## Quick Start

```bash
# See what tuners would run (no GPU needed)
kernelforge gemm-tune plan --model-path /wekafs/models/Qwen3-30B-A3B --framework sglang --precision bf16

# Run tuning
kernelforge gemm-tune run \
  --model-path /wekafs/models/Qwen3-30B-A3B \
  --framework sglang \
  --precision bf16 \
  --conc 256 \
  --mp 8 \
  --output-dir /tmp/tuning_output \
  --skip-gpu-check
```

## Precision Determination

The `--precision` flag reflects the **runtime kernel precision**, not the model's storage format:

| Model Weight | Runtime Config | `--precision` | `--quant-type` |
|---|---|---|---|
| FP32/BF16 (no quant) | Default serving | `bf16` | `none` |
| FP32/BF16 + `--quantization fp8` | aiter FP8 blockscale | `fp8` | `blockscale` |
| FP32/BF16 + `--quantization fp8` | aiter FP8 per-token | `fp8` | `per_token` |
| FP8 checkpoint (native) | FP8 serving | `fp8` | `blockscale` or `per_token` |
| AWQ/GPTQ int8 | vllm int8_w8a16 | `awq` | `awq` |
| FP4/MXFP4 | aiter FP4 | `fp4` | `fp4` |

**How Hyperloom determines precision:**
- From server launch args: `--quantization fp8` → precision=fp8
- From `--fp8-gemm-backend aiter` → confirms aiter path
- From server log: `grep "QuantType"` → determines blockscale vs per_token
- Fallback: model's native dtype (bf16/fp16)

## Tuner Types (9 total)

| Tuner | Framework | Kernel Target | Time Est. | Env Var Output |
|---|---|---|---|---|
| `fmoe_ck` | sglang | MoE fused GEMM (CK 2-stage) | ~15 min | `AITER_CONFIG_FMOE` |
| `a8w8_blockscale` | sglang | Dense FP8 blockscale GEMM | ~20 min | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` |
| `a8w8_blockscale_bpreshuffle` | sglang | Dense FP8 blockscale + preshuffle GEMM | ~20 min | `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE` |
| `a8w8` | sglang | Dense FP8 per-token GEMM | ~20 min | `AITER_CONFIG_GEMM_A8W8` |
| `a8w8_bpreshuffle` | sglang | Dense FP8 preshuffle GEMM | ~20 min | `AITER_CONFIG_GEMM_A8W8_BPRESHUFFLE` |
| `sglang_dense_bf16` | sglang | Dense BF16 GEMM | ~20 min | `AITER_CONFIG_GEMM_BF16` |
| `a4w4_blockscale` | sglang | Dense FP4 GEMM (gfx950 only) | ~20 min | `AITER_CONFIG_GEMM_A4W4` |
| `vllm_moe_triton` | vllm | MoE Triton fused_moe | ~30 min | `VLLM_TUNED_CONFIG_FOLDER` |
| `vllm_dense_tunableop` | vllm | Dense hipBLASLt/rocBLAS | ~45 min | `PYTORCH_TUNABLEOP_FILENAME` |

Time estimates are for 10 token shapes on 8 GPUs (--mp 8). Single GPU takes ~8x longer.

### Multi-Tuner Execution

A single model may need **multiple tuners** simultaneously:

- **MoE model + FP8 blockscale** → `fmoe_ck` (MoE layers) + `a8w8_blockscale` (dense layers)
- **MoE model + bf16** → `fmoe_ck` only (no dense aiter tuner for bf16)
- **Dense model + FP8** → `a8w8_blockscale` only
- **vLLM MoE** → `vllm_moe_triton` + `vllm_dense_tunableop` (if shapes provided)

All tuners output to **different env vars** — they don't conflict. At serving time, set all of them.

## Time Budget Management

Use `--global-timeout` to control total wall time:

```bash
# 2-hour budget: tuners run in priority order, remaining ones skipped if time runs out
kernelforge gemm-tune run ... --global-timeout 7200

# Individual tuner cap (per-tuner, default 1h)
kernelforge gemm-tune run ... --timeout 1800 --global-timeout 7200
```

Strategy when budget < estimated total:
1. Tuners execute in priority order (MoE tuners = priority 10, Dense tuners = priority 20)
2. Before each tuner starts, remaining global budget is checked
3. If remaining time < 0, tuner is **skipped** (not killed mid-run)
4. Per-tuner timeout is capped to min(--timeout, remaining_global_budget)
5. `plan.json` shows estimated times — Hyperloom can pre-check feasibility

Example: 2h budget, 3 tuners estimated at 15+20+45 min = 80 min → all will run.
Example: 30min budget, 2 tuners at 15+20 min = 35 min → second tuner may be skipped.

## CLI Reference

### `kernelforge gemm-tune run`

#### Required Parameters

| Parameter | Description |
|---|---|
| `--model-path` | Path to model directory (must contain `config.json`) |
| `--framework` | `sglang` or `vllm` |
| `--precision` | Runtime precision: `bf16`, `fp8`, `fp4`, `int8`, `awq` |
| `--output-dir` | Directory for all outputs (logs, artifacts, result.json) |

#### Routing Control

| Parameter | Default | Description |
|---|---|---|
| `--quant-type` | `auto` | `auto`, `none`, `per_token`, `blockscale`, `bpreshuffle`, `awq`, `gptq`, `fp4`, `mxfp4` |
| `--tuner` | (empty) | Force a specific tuner name (bypass auto-routing) |
| `--kernel-signature-log` | (empty) | Server log file for detecting 1-stage ASM dispatch |

#### Model / Workload

| Parameter | Default | Description |
|---|---|---|
| `--gpu-type` | `mi300x` | `mi300x` or `mi355x` |
| `--tp` | 1 | Tensor parallel degree |
| `--conc` | 64 | Target serving concurrency (affects token coverage) |
| `--tokens` | (empty) | Explicit comma-separated token sizes (overrides auto) |

#### Tuning Control

| Parameter | Default | Description |
|---|---|---|
| `--mp` | 1 | Parallel GPUs for tuning (aiter supports embarrassing parallelism) |
| `--iters` | 80 | Benchmark iterations per config |
| `--warmup` | 20 | Warmup iterations |
| `--min-improvement-pct` | 3.0 | Min % improvement threshold to mark a shape as improved |
| `--timeout` | 3600 | Per-tuner timeout in seconds |
| `--global-timeout` | 0 | Global session timeout in seconds (0 = unlimited) |

#### External Inputs (from Hyperloom)

| Parameter | Description |
|---|---|
| `--untuned-csv` | Dense aiter tuner input CSV (M,N,K shapes) |
| `--shapes-json` | GEMM shapes JSON from TraceLens/Hyperloom |
| `--tunableop-input` | PyTorch TunableOp recorded shapes file |

#### Environment

| Parameter | Description |
|---|---|
| `--gpu-ids` | Comma-separated GPU IDs (overrides ROCR_VISIBLE_DEVICES) |
| `--skip-gpu-check` | Skip rocm-smi preflight (use when Ray manages GPUs) |
| `-v, --verbose` | Enable debug-level logging |

### `kernelforge gemm-tune plan`

Dry-run: shows model analysis and which tuners would be selected. No GPU needed.

Same parameters as `run` except: no `--output-dir`, `--mp`, `--iters`, `--timeout`, etc.

## Output

### Exit Codes

| Code | Meaning |
|---|---|
| 0 | Success (tuning produced candidate or confirmed no_improvement) |
| 1 | At least one tuner failed |
| 2 | Input validation error (missing model, bad config) |

### stdout (sentinel-wrapped JSON)

```
FORGE_GEMM_TUNE_RESULT_BEGIN
{ ... JSON ... }
FORGE_GEMM_TUNE_RESULT_END
```

All other output goes to stderr and log files. Hyperloom parses only between the sentinels.

### result.json Schema

```json
{
  "status": "ok | skipped | failed",
  "micro_decision": "candidate | no_improvement | skipped | failed",
  "requires_e2e_validation": true,
  "model_path": "/path/to/model",
  "framework": "sglang",
  "precision": "bf16",
  "quant_type": "none",
  "gpu_type": "mi300x",
  "tp": 1,
  "conc": 256,
  "tokens": [64, 128, 256],
  "recommended_env": {
    "AITER_CONFIG_FMOE": "/output/tuners/fmoe_ck/candidate_fmoe.csv"
  },
  "artifacts": {
    "fmoe_ck": "/output/tuners/fmoe_ck/candidate_fmoe.csv"
  },
  "tuners_run": [
    {
      "tuner": "fmoe_ck",
      "status": "ok",
      "elapsed_s": 70.8,
      "improved_shapes": 2,
      "total_shapes": 3,
      "best_micro_speedup": 1.2066,
      "avg_micro_speedup": 1.1079,
      "env_var": "AITER_CONFIG_FMOE",
      "env_value": "/output/.../candidate_fmoe.csv",
      "shape_results": [
        {
          "token": 64,
          "default_us": 338.9,
          "tuned_us": 320.8,
          "improve_pct": 5.35,
          "speedup": 1.0566,
          "improved": true
        }
      ]
    }
  ],
  "tuners_skipped": [
    {
      "tuner": "a8w8_blockscale",
      "skip_reason": "Requires --untuned-csv or --shapes-json..."
    }
  ],
  "total_elapsed_s": 74.7,
  "started_at": "2026-06-18T07:29:06Z",
  "finished_at": "2026-06-18T07:30:17Z"
}
```

### Output Directory Structure

```
output-dir/
├── result.json              # Structured report (same as stdout JSON)
├── plan.json                # Routing plan with time estimates
├── run.log                  # Full execution log
├── gpu_check.json           # GPU preflight status (if not skipped)
└── tuners/
    ├── fmoe_ck/
    │   ├── untuned_fmoe.csv     # Generated input shapes
    │   ├── tuned_fmoe.csv       # Raw tuner output
    │   ├── candidate_fmoe.csv   # Final candidate (only improved shapes)
    │   ├── profile_fmoe.csv     # Full profiling data
    │   └── tune.log             # Subprocess stdout/stderr
    ├── a8w8_blockscale/
    │   ├── tuned_a8w8_blockscale.csv
    │   └── tune.log
    └── vllm_moe_triton/
        ├── tuned_configs/       # VLLM_TUNED_CONFIG_FOLDER content
        │   └── E=128,N=768,...,dtype=bfloat16.json
        ├── sweep_results.json
        └── tune.log
```

## Hyperloom Integration

- Hyperloom calls `kernelforge gemm-tune run` as a subprocess
- Reads `recommended_env` from result.json
- Restarts serving with those env vars → runs E2E benchmark → decides KEEP/REVERT
- CLI does NOT make the final KEEP/REVERT decision (only `micro_decision`)

### `--kernel-signature-log` decides whether tuning does anything

The CLI does not *require* it, which is not the same as it being optional in
practice. Everything the tuners need beyond the model config comes from this
log:

- **Dense shapes.** Shapes derived from `config.json` served **0.4%** of the
  lookups the runtime actually made, across 42 measured arms. The log's own
  miss list is the shape source; without it the dense tuners either skip, or
  tune a table nothing reads.
- **The MoE dispatch key.** `fmoe_ck` refuses to tune a key inferred from the
  config, because the quantisation pair, the per-partition `inter_dim` and the
  EP path's extra masked expert slot are all chosen by the serving framework.
  The log prints the tuple aiter dispatched, which supplies all three. With no
  log, `fmoe_ck` skips every MoE model — measured as 27 skips out of 27 on a
  box with 33 models.

The log must come from a server that actually served traffic; a boot-only log
records no lookups. Hyperloom populates it from the current-best or baseline
benchmark workspace.

`--demand <demand.json>`, an already-parsed demand file from
`kernelforge gemm-tune evidence`, is equivalent and takes priority. Given only
`--kernel-signature-log`, `run` derives one from the log itself.
