---
myst:
  html_meta:
    "description": "KernelForge architecture: an autonomous iteration loop that drives one backend-specialized agent per kernel and decides every change on its own measurements."
    "keywords": "KernelForge, architecture, forge-loop, kernel backends, Composable Kernel, Triton, HIP, hipBLASLt, FlyDSL, AITER, fusion, knowledge base"
---

# Architecture

KernelForge is organized around one **iteration loop** per kernel. A campaign
(`kernelforge forge-loop`) owns a git workspace, the kernel it optimizes and
the driver that measures it, and drives a single writable agent — an
**implementer** carrying one kernel backend's expertise — through repeated
plan → edit → validate → benchmark cycles. The agent works through the Bash tool
inside that workspace; the loop, not the agent, owns the measurements that
decide whether a change survives.

## Loop components

| Component | Role |
|:----------|:-----|
| **Campaign** | The immutable inputs — kernel, driver, kernel backend, gates, branch — snapshotted so an interrupted run resumes on identical terms |
| **Baseline** | Benchmarks the pristine kernel before any edit, so iteration 1 is never kept unconditionally |
| **Analysis** | Read-only hardware profiling of the current best, producing a brief — where the time goes, the bottlenecks ranked, the directions expected to pay — and the evidence bundle behind it. It measures the canonical kernel only and leaves trying a change to the Implementer. In a campaign of several lanes it also divides the work among them: one focus per lane, by optimization direction or by scored case |
| **Lanes** | Every round runs its Implementer sessions as lanes, each in its own copy of the workspace: as many as the Analysis gave focuses, up to `--lanes`, and one when it gave none. A lane's session chooses its own direction, inside its focus when it has one, edits the kernel sources, and compiles and measures them through Bash. The lanes share one GPU through a queue, so two never measure at once |
| **Merge** | When more than one lane changed the kernel, an Implementer session on the canonical tree combines them: it starts from the lane that scored best, keeps another lane's change only where its own measurement shows a gain, and is judged against the current best. A merge that takes nothing leaves the lane it started from as the candidate. A round in which one lane changed the kernel needs no merge |
| **Validation** | The driver-owned complete correctness suite, run by the loop as an SNR pre-filter |
| **Benchmark** | The canonical benchmark, scored per case against the current best |
| **Acceptance** | For the assembly backend only: the task's own verdict — its `compile_command`, then its `correctness_command` — re-run on a candidate about to become the incumbent, carrying the `numerical_validation` evidence that backend requires. Every other backend is judged by the driver above and measured through it, so it is not asked a second time |
| **Keep or revert** | A measured improvement that the acceptance step passes is committed and becomes the new best; every other candidate is discarded back to the last validated commit |
| **Supervisor** | When the search stalls, reviews the trajectory and writes a ruling that redirects the next plan instead of ending the run |
| **Knowledge base** | Hardware, methodology and per-language knowledge injected into the implementer's prompt; lessons from the run are written back |

## Kernel backends

Ten kernel backends carry backend expertise. A campaign selects one with
`--kernel-backend <backend>`, and that backend contributes the domain prompt for
the kernel the loop is editing.

| KernelBackend | Backend expertise |
|:-------|:------------------|
| `ck` | Composable Kernel C++ templates: tile shapes, pipelines, instance factories |
| `flydsl` | MLIR-based DSL for MFMA-heavy compute: layouts, warp shapes |
| `triton` | Triton JIT kernels: block sizes, warps, stages, autotuning |
| `gluon` | Gluon, Triton's low-level dialect: explicit layouts, hand-authored software pipeline, register budget, MFMA intrinsics |
| `aiter` | Pre-built AITER operators: dispatch, JIT integration, baselines |
| `hip` | Raw HIP C++ and HipKittens: MFMA intrinsics, AGPR management, register pinning |
| `hipblaslt` | Dense GEMM via hipBLASLt: TensileLite solutions, FP8, fused epilogues |
| `fusion` | Decode-path kernel fusion for sglang and vLLM: CUDA-graph-safe Triton kernels |
| `ave` | [AveLang](https://github.com/causalflow-ai/avelang) GPU kernels: register tiling, LDS layouts, memory access and synchronization |

## The measurement surface

Everything the loop decides on is produced by the task's own driver, invoked as
`python driver.py <args>` and read over stdout: the correctness suite and the
benchmark. That makes the driver — together with its timing harness and
correctness reference — the measurement surface, and the loop protects it: an
implementer edit or shell write that touches it is refused while the session
can still be saved.

The kernel sources are the opposite: the anchor named by `--kernel` and any
tracked implementation file outside the protected surface may be edited. What
the agent cannot do is change how it is graded.

See the {doc}`Optimization loop </kernelforge/conceptual/optimization-loop>` for the gates
each change clears, and the
{doc}`Autonomous overnight loop </kernelforge/how-to/autonomous-loop>` for how a long
unattended campaign is structured.
