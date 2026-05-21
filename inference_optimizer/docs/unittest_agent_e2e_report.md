# Unittest Agent and GEAK E2E Validation Report

This report records the first validation pass for the Hyperloom
`unittest_agent` integration with GEAK.  It focuses on the mechanics of the
generated harness, the handoff to GEAK, and the observed E2E outcome.

## Scope

- Repository: `/wekafs/zihao/2026/0518/Hyperloom`
- Model used for validation: `/wekafs/models/Qwen-Qwen3-8B`
- Early validation session:
  `/workspace/hyperloom/kernel-agent/{runs,unittests,geak}/integration_demo_2`
- Later full session:
  `/workspace/hyperloom_qwen3_8b_full_20260518_154544`

## Integration Flow

```text
inference_optimizer.cli optimize
        |
        v
kernel_request_handlers.run_optimization_handler
        |
        v
kernel-agent/tools/kernel_optimization.py
        |
        v
invoke_backend("geak", ...)
        |
        +--> _maybe_generate_unittest(candidate)
        |       |
        |       +--> build AgentKernelArena-style task
        |       +--> self-verify when possible
        |       +--> return manifest(status, test_command, paths)
        |
        +--> append unittest context to GEAK prompt when manifest.status == "ok"
        |
        +--> geak.submit(..., test_command=<generated correctness command or legacy benchmark>)
```

If unittest generation fails, raises, returns `failed`, returns `skipped`, or
returns a non-usable `degraded` manifest, Hyperloom logs the manifest status and
falls back to the legacy GEAK path:

```text
candidate.benchmark_files / benchmark_file / test_harness_path
```

The fallback is intentional.  The unittest pre-step must never block GEAK.

## Generated Harness Shape

For Python/Triton kernels, `unittest_agent.py` creates:

```text
<out_dir>/
├── config.yaml
├── scripts/task_runner.py
├── source/<kernel>.py
├── source/_baseline_snapshot/<kernel>.py
└── unittest_meta.json
```

Important properties:

- `source/<kernel>.py` is the editable mirror GEAK can test against.
- `_baseline_snapshot/<kernel>.py` is the golden reference captured before GEAK
  changes anything.
- The generated runner exports runtime env vars (`SGLANG_*`, `VLLM_*`,
  `AITER_*`, `TRITON_*`, `HIP_*`, `ROCR_*`, `CUDA_*`) before importing code.
- Correctness imports the editable source and the snapshot under different
  module names and compares tensor outputs.
- Performance uses CUDA events and emits an AgentKernelArena-compatible JSON
  schema.

For HIP/C++ kernels, the current implementation creates the same top-level task
layout but the generated runner wraps the discovered benchmark command.  At GEAK
test time it:

1. Copies `source/<kernel>` over the live framework source path.
2. Invalidates likely aiter JIT modules.
3. Runs the captured benchmark command.
4. Restores the live source and JIT artifacts.

Generation-time HIP correctness is deferred because the captured benchmarks can
take minutes.  GEAK runs the generated correctness command in its own
baseline/patch loop.

## Control Knobs

`kernel_optimization.py` now supports:

```bash
--unittest-agent auto   # default
--unittest-agent off
--unittest-agent force
```

Equivalent environment variable:

```bash
HYPERLOOM_UNITTEST_AGENT=auto|off|force
```

Backward-compatible disable knob:

```bash
HYPERLOOM_DISABLE_UNITTEST_AGENT=1
```

`off` restores the legacy GEAK behavior and bypasses harness generation.

## Early Python/Triton Validation

A Triton BMM fixture from AgentKernelArena was used to validate the Python path:

- Harness generation succeeded.
- Generated `task_runner.py` parsed correctly.
- `compile` passed.
- `correctness` passed for the unmodified source.
- A deliberate output mutation was detected by correctness.
- `performance` emitted the expected AgentKernelArena schema.

A live aiter Triton RMSNorm source was also tested:

- Source: `/sgl-workspace/aiter/aiter/ops/triton/normalization/rmsnorm.py`
- Captured shapes: `[2, 4096]` and `[4096]`
- Dtypes: `bfloat16`
- Host entry picker selected `rms_norm`, not helper functions such as
  `num_programs`.
- One scalar argument (`epsilon`) was auto-filled as `1e-6`.
- Self-verify passed: `compile=ok`, `correctness=ok`.

## GEAK Runtime Validation

The GEAK bash-tool issue was reproduced and fixed:

- Broken schema: `function.input_schema` caused Claude to emit `bash{}`.
- Fixed schema: `function.parameters` caused Claude to emit
  `bash{"command": "..."}`.

Validated in a real GEAK run:

```text
empty_bash_errors: 0
bash_tool_calls: 47
save_and_test_calls: 2
submit_calls: 1
```

The GEAK config was also fixed to use the OpenAI-compatible gateway shape:

```yaml
model:
  model_class: litellm
  model_name: openai/claude-opus-4-7
  base_url: https://core42.example-internal-host.invalid/api/v1/llm-proxy/v1
```

## Full Qwen3-8B SGLang Run

Full session:

```text
/workspace/hyperloom_qwen3_8b_full_20260518_154544
```

Baseline:

```text
output throughput: 2405.58 tok/s/gpu
mean TTFT:         517.62 ms
mean E2EL:         3395.51 ms
```

Params winner:

```text
extra_sglang_args: --num-continuous-decode-steps 8
output throughput: 2456.64 tok/s/gpu
gain over baseline: +2.12%
```

TraceLens produced three reusable native kernels:

| kernel_id | GPU % | Source |
|---|---:|---|
| `k006` | `4.618` | `/sgl-workspace/aiter/3rdparty/composable_kernel/include/ck_tile/ops/fmha/pipeline/block_fmha_batch_prefill_pipeline_qr_ks_vs_async.hpp` |
| `k008` | `3.508` | `/sgl-workspace/aiter/csrc/kernels/activation_kernels.cu` |
| `k009` | `2.806` | `/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu` |

All three are HIP/C++ candidates.  They now receive HIP wrapper harnesses in new
runs; the captured full run happened before the HIP wrapper extension and used
legacy `benchmark_files`.

## Latest GEAK Commands

### k006

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

Result: `1.00x`.  GEAK determined the target prefill kernel was not exercised by
the provided benchmark and submitted a byte-identical safe artifact.

### k008

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

Result: `1.00x`.  GEAK found the dominant shape to be bandwidth-bound and did
not produce a keep-worthy improvement.

### k009

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

- Microbench speedup: `2.23x`
- Correctness: PASS
- Reconstructed source:
  `/workspace/hyperloom_qwen3_8b_full_20260518_154544/kernel-agent/runs/qwen3_full_manual_geak_batch/reconstructed_rmsnorm_quant_kernels.cu`

## k009 E2E Follow-up

The k009 source was applied to:

```text
/sgl-workspace/aiter/csrc/kernels/rmsnorm_quant_kernels.cu
```

The old JIT module was removed to force rebuild:

```text
/sgl-workspace/aiter/aiter/jit/module_rmsnorm_quant.so
```

Quick import/call passed and rebuilt the module.

E2E on top of the params winner:

| Run | Patch state | Output throughput |
|---|---|---:|
| Current best from params | original | `2456.64 tok/s` |
| k009 patched #1 | applied | `2459.75 tok/s` |
| k009 control | reverted | `2424.64 tok/s` |
| k009 patched #2 | applied | `2468.47 tok/s` |

Decision:

- The patch has positive signal relative to same-time control.
- Relative to the official current best, it is only `+0.13%` to `+0.48%`.
- It does not clear the standard `+1%` KEEP threshold.
- Do not auto-keep k009.
- The live `/sgl-workspace/aiter` source and JIT module were restored.

## Remaining Gaps

- Persist the exact GEAK command in `optimization_attempts.jsonl`; the commands
  above are reconstructed from `geak_submit.py`, prompt paths, output dirs,
  candidate metadata, and `geak_agent.log`.
- Improve GEAK patch capture for HIP/C++ tasks that write files under
  `/sgl-workspace/optimized_versions` instead of standard `patch_*.patch`.
