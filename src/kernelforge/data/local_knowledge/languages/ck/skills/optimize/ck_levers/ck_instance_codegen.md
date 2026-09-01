---
title: CK — where kernels come from, and why your build takes an hour
kind: language
lever: ck_instance_codegen
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
  - https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md
---

# Where CK kernels come from

## Route here when
- The build has been running for an hour and you want to know what it is compiling.
- You are about to pin a swept winner and want to know how long that pin stays valid.
- `generate.py` produced hundreds of `.cpp` files and you want fewer.
- A dtype you expected is missing from the instance list.
- You want to know what `ckProfiler` is iterating over, before you trust its ranking.

**Skip this if** you are choosing between the two front-ends — that decision is in
`ck_frontend_classic.md` and `ck_frontend_tile.md`. This card is about how either one turns into
object code.

## The short version
CK does not compile a template per call. It pre-materializes kernels ahead of time, by two entirely
different mechanisms depending on which front-end you are on. Both mechanisms are combinatorial, which
is why an unscoped build is slow, and both produce identifiers that are **valid only for the build that
produced them**, which is why a pinned winner is not portable.

| | Classic | ck_tile |
|---|---|---|
| Mechanism | C++ instance factory, registered at link time | Python script emits `.cpp` before compiling |
| What you sweep | registered instances for your layout/dtype | generated trait combinations |
| Sweep tool | `ckProfiler` | the example binary, `-v 1` |
| How to shrink it | build one instance group | prune the trait list in the generator |

## Classic: the instance factory
Two layers, and it helps to keep them separate in your head.

**Registration.** Under `library/src/tensor_operation_instance/gpu/gemm*/`, headers spell out concrete
`DeviceGemmXdlUniversal<...>` specializations — one per tile-and-pipeline combination — and hand them
to `add_device_gemm_xdl_universal_*_instances(...)`. This happens at build time; the list is fixed once
the library is compiled.

**Retrieval.** `DeviceOperationInstanceFactory<...>::GetInstances(ops)` hands back that whole
registered list for a given layout and dtype. Nothing is selected for you. The caller iterates, discards
anything whose `IsSupportedArgument()` returns false, times the rest, and remembers the index of the
winner.

That iterate-and-time loop is not a metaphor for `ckProfiler` — it is literally what `ckProfiler` does.
Which means the ranking it prints is only as good as the instance list that was compiled in, and an
instance that was never registered simply cannot appear. See `ck_frontend_classic.md` for the loop.

## ck_tile: generated, not registered
ck_tile takes the opposite approach. There is no shipped database; a Python generator writes the
kernels you asked for and nothing else.

Take FMHA (`example/ck_tile/01_fmha/`). The generator expands the kernel template across a product of
traits and writes each expansion to its **own** `.cpp` file. The stated reason is parallel compilation —
many small translation units build faster than one enormous one. The cost is that the file count is the
size of the trait product, and nothing caps that product for you.

The traits being crossed: head-dim × dtype × causal/mask spec × bias/alibi × rotary × paged-KV. Six
dimensions multiply quickly.

The grid-wise kernel (`fmha_fwd_kernel.hpp`) is parameterized on two policies:

- `FmhaPipeline` decides how the block tile is walked — `qr_ks_vs`, `qr_ks_vs_async`, and the paged-KV
  variants. Upstream calls it "a performance critical component," and that is not boilerplate: this is
  the choice that moves FMHA numbers. Details in `ck_fmha_stack.md`.
- `EpiloguePipeline` handles the final phase — transform the accumulator and write it out.

The text that gets expanded lives in a string blob named `FMHA_FWD_KERNEL_BODY` inside the generator.
If you need to understand exactly what is emitted, read that blob rather than the generated output.

Alongside the generator the example directory carries the drivers and headers you would expect
(`example_fmha_fwd.cpp`, `example_fmha_bwd.cpp`, `fmha_fwd.hpp`, `fmha_bwd.hpp`, `mask.hpp`,
`bias.hpp`, `rotary.hpp`, `quant.hpp`) plus `codegen/`, `misc/`, `script/`.

## Making the build finish
Ordered by how much time each one actually saves:

| Lever | Why it matters |
|---|---|
| `GPU_TARGETS=gfx950` at cmake | Left alone, CK compiles every architecture it knows about. This one flag usually dominates everything else on this list. |
| Name your instance group | `make device_gemm_xdl_universal_f16_instance`, `ninja tile_example_fmha_fwd` — build the one group you are testing, not the library. |
| CK-Tile dispatcher | The newer unified codegen front-end (C++ and Python, see CHANGELOG) filters by architecture and emits only the instances your shapes need. |
| Cut the generator's trait list | Restrict head-dims, dtypes and masks to what you actually serve. Six multiplied dimensions is where the `.cpp` explosion comes from. |

**One flag causes a confusing symptom.** On gfx950, fp4 and mxfp4 instances sit behind `DTYPES` build
flags. Leave them off and the instances are never generated — so the failure surfaces later, at
selection time, as "that instance does not exist." It reads like a coverage gap in CK. It is a cmake
argument you did not pass.

## A pinned winner is build-scoped
Treat a swept instance index the way you would treat a memory address: meaningful inside one process,
meaningless outside it.

Tile IDs, pipeline IDs, and the contents of the tuned database all move between CK and ROCm releases.
An index that named the fastest kernel last month may name a different kernel today, or nothing at all.

Three habits that keep this from biting:
- Re-sweep after every CK or ROCm bump. Not "if something looks slow" — every bump.
- Record the pin together with what produced it: `instance <idx> @ CK <commit>, ROCm <ver>, <shape>, <date>`.
  An index with no provenance is unusable six weeks later.
- Never hand a frozen table to another team as if it were portable.

## Verify
| Check | How | Pass condition |
|---|---|---|
| Only the traits you wanted were emitted | `ls` the generated `.cpp` files | the count matches your intended trait product |
| The instance ranking is real | `ckProfiler` (classic) or the example with `-v 1` (ck_tile), at your shapes | the winner also passes `IsSupportedArgument()` |
| A recorded pin still holds | re-sweep after the bump, compare to the recorded number | same instance, same ballpark timing |
| The dtype you need exists | check the emitted instance list, not the docs | it is present before you try to select it |

## Failure modes
| Symptom | Cause | Fix |
|---|---|---|
| Build runs for an hour or more | compiling every architecture and dtype | scope `GPU_TARGETS`, then build one instance group |
| Hundreds of FMHA `.cpp` files | the generator's trait product was never bounded | prune head-dims / dtypes / masks to serving shapes |
| A pinned instance got slow after an upgrade | IDs drifted; the index now names something else | re-sweep — the pin was only ever valid for that build |
| "That instance doesn't exist" on gfx950 | fp4 / mxfp4 gated behind `DTYPES` at cmake | enable them at configure time and rebuild |
| Cannot sweep where you deploy | `ckProfiler` is not in the runtime image | sweep on a dev node, or use a library that tunes at runtime (aiter) |

The full trap list, indexed by symptom, is in `ck_traps.md`.

## Sources
- ck_tile 01_fmha example layout, the generator, `FMHA_FWD_KERNEL_BODY`, and the
  `FmhaPipeline` / `EpiloguePipeline` policies:
  https://github.com/ROCm/composable_kernel/tree/develop/example/ck_tile/01_fmha
- Instance selection and the profiler:
  https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/optimizing-with-composable-kernel.html
- CK-Tile dispatcher, persistent async input scheduler, fp4 `DTYPES` gating:
  https://github.com/ROCm/composable_kernel/blob/develop/CHANGELOG.md
