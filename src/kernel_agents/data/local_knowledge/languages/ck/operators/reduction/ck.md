---
title: reduction on Composable Kernel — SOTA card
kind: sota_card
operator: reduction
backend: ck
gens: [gfx908, gfx90a, gfx942, gfx950]
dtypes: [fp32, bf16, fp16, int32]
regimes: [both]
status: sota
updated: 2026-06-08
sources:
  - https://github.com/ROCmSoftwarePlatform/composable_kernel/pull/82
  - https://rocm.docs.amd.com/projects/composable_kernel/en/latest/
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
---

# reduction × composable_kernel

## TL;DR
CK ships **ready-made, tuned reduction device ops** (`DeviceReduce*`) covering the threadwise / blockwise /
multi-block strategies, including the two split-reduction combine paths — **`MultiBlockAtomicAdd`**
(single pass, atomics) and **`MultiBlockTwoCall`** (partial-reduce + a blockwise second call,
deterministic). Reach for CK when you want a hardened, instance-tuned reduce (pooling, norm-statistic,
large-axis sum) without hand-writing the wave/LDS/atomic plumbing, or when CK is already in your stack.

## SOTA implementation(s)
| impl | source | gens/dtypes | combine | when best |
|---|---|---|---|---|
| `DeviceReduceBlockWise` | CK `tensor_operation/.../device` | gfx9, fp32 acc | LDS within block | row reduce, axis fits a block |
| `DeviceReduceMultiBlockAtomicAdd` | CK + PR #82 | gfx9, dtypes w/ atomic support | per-block `AtomicAdd` to global out, **single pass** | huge axis / low output count, sum/mean, parity-tolerant |
| `DeviceReduceMultiBlock` (TwoCall: `multiblock_partial_reduce` → blockwise 2nd call) | CK + PR #82 | gfx9, all | partials array + 2nd kernel, **deterministic** | max/min, parity-critical, no fp atomic |
| Max/Avg pooling via Reduce | CK examples (PR #82) | gfx9 | (as above) | pooling ops |

The split logic mirrors GEMM split-K: tile the reduced (K) axis across blocks, threadwise-reduce into
registers, blockwise-combine via **LDS**, then either `AtomicAdd` to the single output (single pass) or
emit partials for a deterministic second call.

## Config space / knobs
- **strategy**: ThreadWise (tiny axis) / BlockWise (axis fits a block) / MultiBlock (split a huge axis).
- **combine**: `AtomicAdd` (single pass; gated by `InMemoryDataOperationSupportedOnDataType<AtomicAdd,T>`
  — not every dtype has HW atomic) vs **TwoCall** (deterministic, +1 launch).
- **op**: `Add`(sum/mean), `Max`, `Min`, `AMax`, custom — `AtomicMax` trait gates the atomic-max path.
- **tile / vector widths**: per-instance (block tile over the reduced axis, load vector width — drives
  alignment, the coverage gate).
- **InElementwiseOp / AccElementwiseOp**: fuse a pre-op (square for L2, abs) and a post-op (`/n` for mean,
  `sqrt` for L2) into the reduce — CK's fusion seam.

## Numerics / parity
fp32 accumulate; `AtomicAdd` path = nondeterministic order (bf16/fp32 LSB run-to-run); **TwoCall** is
deterministic — use it on parity-critical paths. `AtomicAdd`/`AtomicMax` availability is **dtype-gated**
(the trait), so e.g. a bf16 atomic reduce may not be supported → falls to TwoCall. See [numerics.md](numerics.md).

## Integration (rebind seam)
- C++: instantiate the `DeviceReduce*` op, set the `Add`/`Max` reduction + pre/post elementwise ops,
  `IsSupportedArgument` → `Run`. Tune instances with `ckProfiler` (when present in the image).
- From PyTorch: not a direct rebind — CK reductions are used inside CK/MIOpen norm/pooling paths, or via a
  custom op wrapping the device op. On the LLM serving path the reduce is usually inside a fused norm
  (aiter/CK), not a standalone CK reduce.

## Pitfalls & anti-patterns
- **Coverage gap**: no compiled instance matches your (axis len, strides, dtype, vector width) →
  `IsSupportedArgument` fails (same failure mode as CK GEMM — see
  `../../../backends/composable_kernel_lib/instances.md`).
- Choosing `AtomicAdd` for a parity-critical reduce → nondeterministic; or for a dtype without atomic
  support → unsupported.
- `ckProfiler` absent in some images → no instance sweep there.
- Repo moved: pin `ROCm/rocm-libraries:projects/composablekernel` (standalone repo deprecated).

## How to verify
Build/run the matching CK reduce example or `ckProfiler` at your axis/dtype; GB/s vs ~4.3 TB/s; for
MultiBlock, rocprof CU utilization; compare AtomicAdd vs TwoCall for the determinism you need; fp32 atol
vs torch.

## Alternatives / cross-links
triton · hip (hand-rolled equivalent) · [tuning.md](tuning.md) ·
CK language: `../../../languages/composable_kernel/ck_tile.md`,
`../../../languages/composable_kernel/ck_classic.md` ·
CK lib: `../../../backends/composable_kernel_lib/instances.md`.

## Sources
- CK reduction infra: ThreadWise/BlockWise/MultiBlock, AtomicAdd vs MultiBlockTwoCall (partial + blockwise second call), pooling examples: https://github.com/ROCmSoftwarePlatform/composable_kernel/pull/82
- CK user guide (device ops, structure): https://rocm.docs.amd.com/projects/composable_kernel/en/latest/
- Optimizing with CK (ROCm): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
