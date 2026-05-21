# GEAK + Unittest Agent Latest Full Run Notes

## Run Context

- Hyperloom session: `/workspace/hyperloom_qwen3_8b_full_20260518_154544`
- Model: `/wekafs/models/Qwen-Qwen3-8B`
- Framework: `sglang`
- GPU: `mi300x`
- Workload: `TP=1, CONC=32, ISL=1024, OSL=256, MAX_MODEL_LEN=5376, bf16`
- Baseline: `2405.58 tok/s/gpu`
- Current best before kernel work: `--num-continuous-decode-steps 8`, `2456.64 tok/s/gpu`, `+2.12%`
- Trace used for GEAK candidate selection:
  `/workspace/hyperloom_qwen3_8b_full_20260518_154544/runs/profile/7412cef642fa47a5b03b20fad69a805f/benchmark_sglang_20260518_155205/torch_trace/merged-1779119721.4590955.trace.json.gz`
- Candidate file:
  `/workspace/hyperloom_qwen3_8b_full_20260518_154544/kernel-agent/runs/qwen3_full_manual/kernel_candidates.json`
- GEAK batch result root:
  `/workspace/hyperloom_qwen3_8b_full_20260518_154544/kernel-agent/runs/qwen3_full_manual_geak_batch`

## Unittest Generation

### What the unittest agent now does

For Python/Triton source files, `kernel-agent/tools/unittest_agent.py` materializes an
AgentKernelArena-style task:

```text
$USER_DATA_PATH/kernel-agent/unittests/<session_id>/<attempt_id>/
├── config.yaml
├── scripts/task_runner.py
├── source/<kernel>.py
├── source/_baseline_snapshot/<kernel>.py
└── unittest_meta.json
```

The runner exports captured runtime env vars before import, materializes
TraceLens-captured shapes/dtypes, loads both the live source and frozen snapshot
under distinct module names, and self-verifies:

- `compile`
- `correctness`
- `performance`

When `self_verify.compile == ok` and `self_verify.correctness == ok`, Hyperloom
passes the generated correctness command to GEAK via `--test-command`.

For HIP/C++ sources, the current implementation now generates a wrapper harness
instead of skipping: `task_runner.py correctness` temporarily copies
`source/<kernel>` over the live framework source, invalidates likely aiter JIT
modules, runs the TraceLens-discovered benchmark command, and restores the live
tree afterwards. Generation-time correctness is deferred because these op tests
can take minutes; GEAK executes the generated command during its own
baseline/patch loop.

### What happened in this full run

All three reusable kernels from the SGLang trace were native HIP/C++ sources:

| kernel_id | source_type | source |
|---|---|---|
| `k006` | `hip_cpp` | `/sgl-workspace/aiter/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp` |
| `k008` | `hip_cpp` | `/sgl-workspace/aiter/csrc/kernels/activation_kernels.cu` |
| `k009` | `hip_cpp` | `/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu` |

At the time of this captured run, before the HIP wrapper extension, these three
attempts returned `skipped` and GEAK fell back to `benchmark_files[0]`. With the
current code, the same candidates now produce HIP harnesses and the command
would be `python3 <unittest_dir>/scripts/task_runner.py correctness`.

Observed per-attempt unittest status:

| kernel_id | attempt | unittest_status | test command source |
|---|---|---|---|
| `k006` | `geak-ac365d48` | `skipped` | `benchmark_files[0]` |
| `k008` | `geak-21b82d84` | `skipped` | `benchmark_files[0]` |
| `k009` | `geak-79e6d4e2` | `skipped` | `benchmark_files[0]` |

Takeaway: the captured run documents the pre-extension fallback. The repo now
supports HIP/C++ wrapper harnesses, so future runs should pass the generated
HIP correctness command to GEAK.

## GEAK Integration

GEAK was invoked through `kernel-agent/tools/backends/geak_submit.py` from
`kernel_optimization.py`. The actual command is built as:

```bash
geak -t <prompt_file> \
  --yolo \
  --output <output_dir> \
  --gpu-ids 0 \
  --config /workspace/hyperloom/runtime/geak-config/local.yaml \
  --kernel-path <source_file> \
  --repo <kernel_repo> \
  --test-command "<test_command>"
```

GEAK ran under the fixed config:

```yaml
model:
  model_class: litellm
  model_name: openai/claude-opus-4-7
  base_url: https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1
tools:
  rag: true
```

The earlier `bash{}` bug is fixed in this run:

| attempt | empty bash errors | bash tool calls | submit calls |
|---|---:|---:|---:|
| `geak-ac365d48` | `0` | `55` | `1` |
| `geak-21b82d84` | `0` | `76` | `1` |
| `geak-79e6d4e2` | `0` | `23` | `1` |

## Full GEAK Commands

### k006: CK-Tile batch prefill FMHA

Candidate:

- `kernel_id`: `k006`
- `gpu_pct`: `4.618`
- `source_file`:
  `/sgl-workspace/aiter/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp`
- `kernel_repo`: `/sgl-workspace/aiter/3rdparty/composable_kernel`
- `benchmark_files[0]`: `/sgl-workspace/aiter/op_tests/test_pa.py`

Full command:

```bash
geak \
  -t /workspace/hyperloom_qwen3_8b_full_20260518_154544/kernel-agent/runs/qwen3_full_manual_geak_batch/prompts/geak-ac365d48.md \
  --yolo \
  --output /workspace/hyperloom_qwen3_8b_full_20260518_154544/kernel-agent/geak/qwen3_full_manual_geak_batch/geak-ac365d48 \
  --gpu-ids 0 \
  --config /workspace/hyperloom/runtime/geak-config/local.yaml \
  --kernel-path /sgl-workspace/aiter/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp \
  --repo /sgl-workspace/aiter/3rdparty/composable_kernel \
  --test-command "python /sgl-workspace/aiter/op_tests/test_pa.py"
```

Result:

- Final report:
  `/workspace/hyperloom_qwen3_8b_full_20260518_154544/kernel-agent/geak/qwen3_full_manual_geak_batch/geak-ac365d48/final_report.json`
- Best speedup: `1.00x`
- Summary: GEAK determined the target CK-Tile prefill kernel was not exercised
  by the provided benchmark and submitted a byte-identical safe artifact.

### k008: SGL HIP silu-and-mul activation

Candidate:

- `kernel_id`: `k008`
- `gpu_pct`: `3.508`
- `source_file`: `/sgl-workspace/aiter/csrc/kernels/activation_kernels.cu`
- `kernel_repo`: `/sgl-workspace/aiter`
- `benchmark_files[0]`: `/sgl-workspace/aiter/op_tests/test_activation.py`

Full command:

```bash
geak \
  -t /workspace/hyperloom_qwen3_8b_full_20260518_154544/kernel-agent/runs/qwen3_full_manual_geak_batch/prompts/geak-21b82d84.md \
  --yolo \
  --output /workspace/hyperloom_qwen3_8b_full_20260518_154544/kernel-agent/geak/qwen3_full_manual_geak_batch/geak-21b82d84 \
  --gpu-ids 0 \
  --config /workspace/hyperloom/runtime/geak-config/local.yaml \
  --kernel-path /sgl-workspace/aiter/csrc/kernels/activation_kernels.cu \
  --repo /sgl-workspace/aiter \
  --test-command "python /sgl-workspace/aiter/op_tests/test_activation.py"
```

Result:

- Final report:
  `/workspace/hyperloom_qwen3_8b_full_20260518_154544/kernel-agent/geak/qwen3_full_manual_geak_batch/geak-21b82d84/final_report.json`
- Best speedup: `1.00x`
- Patch artifacts:
  - `patch_0.patch`: empty baseline
  - `patch_1.patch`: empty / invalid after GEAK selection
- Summary: GEAK found the dominant shape was already close to bandwidth
  limit. It reported no keep-worthy gain.

### k009: aiter rmsnorm quant kernel

Candidate:

- `kernel_id`: `k009`
- `gpu_pct`: `2.806`
- `source_file`: `/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu`
- `kernel_repo`: `/sgl-workspace/aiter`
- `benchmark_files[0]`: `/sgl-workspace/aiter/op_tests/test_rmsnorm2dFusedAddQuant.py`

Full command:

```bash
geak \
  -t /workspace/hyperloom_qwen3_8b_full_20260518_154544/kernel-agent/runs/qwen3_full_manual_geak_batch/prompts/geak-79e6d4e2.md \
  --yolo \
  --output /workspace/hyperloom_qwen3_8b_full_20260518_154544/kernel-agent/geak/qwen3_full_manual_geak_batch/geak-79e6d4e2 \
  --gpu-ids 0 \
  --config /workspace/hyperloom/runtime/geak-config/local.yaml \
  --kernel-path /sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu \
  --repo /sgl-workspace/aiter \
  --test-command "python /sgl-workspace/aiter/op_tests/test_rmsnorm2dFusedAddQuant.py"
```

Result:

- Final report:
  `/workspace/hyperloom_qwen3_8b_full_20260518_154544/kernel-agent/geak/qwen3_full_manual_geak_batch/geak-79e6d4e2/final_report.json`
- Reported micro speedup: `2.23x`
- Reported correctness: PASS
- Selected artifact path in report:
  `/sgl-workspace/optimized_versions/rmsnorm_quant_kernels.cu`
- Reconstructed optimized source from trajectory:
  `/workspace/hyperloom_qwen3_8b_full_20260518_154544/kernel-agent/runs/qwen3_full_manual_geak_batch/reconstructed_rmsnorm_quant_kernels.cu`

## k009 E2E Follow-up

The k009 source was reconstructed from GEAK trajectory, applied to:

`/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu`

The old JIT artifact was removed to force rebuild:

`/sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so`

Quick import / call passed and rebuilt the module.

E2E result on top of params winner (`--num-continuous-decode-steps 8`):

| Run | Patch state | Output throughput |
|---|---|---:|
| Current best from params | original | `2456.64 tok/s` |
| k009 patched #1 | applied | `2459.75 tok/s` |
| k009 control | reverted | `2424.64 tok/s` |
| k009 patched #2 | applied | `2468.47 tok/s` |

Interpretation:

- The patch has a positive signal versus same-time control.
- Relative to the official current best, the gain is only `+0.13%` to
  `+0.48%`, below the standard `+1%` KEEP threshold.

Decision:

- Do not auto-keep k009.
- Live `/sgl-workspace/aiter` was restored after validation.
- `git -C /sgl-workspace/aiter diff -- csrc/kernels/rmsnorm_quant_kernels.cu`
  is empty after restore.

## Notes / Gaps

1. `cmd` is not currently persisted in `optimization_attempts.jsonl` / result
   JSON for GEAK attempts. The commands above are reconstructed from
   `geak_submit.py` command construction, prompt paths, output dirs, candidate
   metadata, and `geak_agent.log`.
2. Some GEAK HIP tasks still write optimized files under
   `/sgl-workspace/optimized_versions` rather than standard `patch_*.patch`.
   Hyperloom salvages these through `geak_per_task_best_patch`, but proper patch
   capture would be cleaner.
