---
title: CK classic device-op API — DeviceGemm* interface & lifecycle
kind: api_reference
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8, mxfp4]
regimes: [both]
status: sota
updated: 2026-07-09
sources:
  - https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1tensor__operation_1_1device_1_1_device_gemm.html
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
---

# CK classic device-op API

The public **interface** of the classic CK library (`ck::tensor_operation::device::DeviceGemm*` family):
the call surface and lifecycle. For the template *parameters* and how to choose them see
[../skills/optimize/ck_levers/ck_gemm_stack.md](../skills/optimize/ck_levers/ck_gemm_stack.md) and
[../skills/optimize/ck_levers/ck_frontend_classic.md](../skills/optimize/ck_levers/ck_frontend_classic.md); for tuning
priority see [../skills/optimize/ck_levers/ck_tuning_knobs.md](../skills/optimize/ck_levers/ck_tuning_knobs.md).

## The device-op families
| Template | Use |
|---|---|
| `DeviceGemm*` / `DeviceGemmXdlUniversal` | plain GEMM (RCR = `Y=X·Wᵀ` is the most-tuned layout) |
| `DeviceBatchedGemmXdl` | batched GEMM (batch stride) |
| `DeviceGroupedGemm*` | variable-M MoE (arrays of ptrs/strides) — the CK path behind fused-MoE |
| `DeviceGemmMultipleD*` | bias/residual/activation fused epilogue |
| `*_fp8`, `*_b_scale`, `*_ab_scale`, `*_mx_gemm` | low-precision (fp8 / weight-scale / mxfp8/mxfp4 block-scaled) |

## The five-call lifecycle (uniform across all device-op families)
```cpp
using DeviceOp = ck::tensor_operation::device::DeviceGemmXdlUniversal</* Row/Col layouts, dtypes,
    elementwise ops, GemmSpec, tile params, pipeline scheduler+version */>;
auto op   = DeviceOp{};
auto arg  = op.MakeArgument(a_ptr, b_ptr, c_ptr, M, N, K,
                            StrideA, StrideB, StrideC, KBatch, AElOp{}, BElOp{}, CElOp{});
if (!op.IsSupportedArgument(arg)) throw ...;        // (!) capability gate — NEVER skip
auto inv  = op.MakeInvoker();
float ms  = inv.Run(arg, StreamConfig{stream, /*time_kernel=*/true});
```
**`IsSupportedArgument` is a correctness gate**: it checks M/N/K divisibility vs the tile, `K` vs
`KPerBlock×KBatch`, pointer alignment vs `AK1/BK1`, and layout/spec. **Forcing an instance past a `false`
returns garbage, not an error.** For non-divisible shapes use `GemmSpecialization::MNKPadding`.

## Instance factory + sweep (this IS ckProfiler / the framework fallback)
```cpp
std::vector<DeviceOpPtr> ops;
ck::tensor_operation::device::instance::DeviceOperationInstanceFactory<
    ck::tensor_operation::device::DeviceGemm<Row,Col,Row,BF16,BF16,BF16,PT,PT,PT>>::GetInstances(ops);
for (auto& op : ops) {
    auto a = op->MakeArgumentPointer(...);
    if (!op->IsSupportedArgument(a.get())) continue;         // skip incompatible
    float ms = op->MakeInvokerPointer()->Run(a.get(), StreamConfig{nullptr, true});
    // keep the fastest supported instance; pin its index for the fixed LLM shape
}
```
Run offline; pin the winning instance per shape. The pinned instance is **build-specific** (tile/pipeline
IDs drift across CK/ROCm versions) — re-sweep on any bump; never ship a hand-copied instance table.

## Layout shorthand
`R`=row, `C`=col. **RCR** (A row, B col, C row) is the standard linear-layer layout and the most-tuned in
CK's instance DB. Pipeline selection = `BlockGemmPipelineScheduler::{Intrawave,Interwave}` ×
`BlockGemmPipelineVersion::{v1..v5}` — see ck_classic.md.

## Sources
- `DeviceGemm` base (MakeArgument / IsSupportedArgument / MakeInvoker / Run): https://rocm.docs.amd.com/projects/composable_kernel/en/docs-6.4.2/doxygen/html/structck_1_1tensor__operation_1_1device_1_1_device_gemm.html
- Instance selection / ckProfiler: https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- Repo: standalone `ROCm/composable_kernel` DEPRECATED → `ROCm/rocm-libraries:projects/composablekernel`.
