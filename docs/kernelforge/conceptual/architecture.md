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
| **Analysis** | Read-only hardware profiling of the current best, producing the evidence bundle the planning stage reads |
| **Planning** | Read-only compute, memory and algorithm specialists analyze their assigned evidence; their useful work is fused into one executable plan per iteration |
| **Implementer** | The only writable agent: reads the plan, edits the kernel sources, compiles and exercises them through Bash. `--lanes` runs several concurrently, each in its own workspace copy |
| **Validation** | The driver-owned complete correctness suite, run by the loop as an SNR pre-filter |
| **Benchmark** | The canonical benchmark, scored per case against the current best |
| **Acceptance** | The arena's own verdict — the task's `compile_command`, then its `correctness_command` — run on any candidate about to become the incumbent — a kept iteration or a knowledge-base warm start alike — under the task's tolerances rather than forge's |
| **Keep or revert** | A measured improvement that the acceptance step passes is committed and becomes the new best; every other candidate is discarded back to the last validated commit |
| **Supervisor** | When the search stalls, reviews the trajectory and writes a ruling that redirects the next plan instead of ending the run |
| **Knowledge base** | Hardware, methodology and per-language knowledge injected into the implementer's prompt; lessons from the run are written back |

## Kernel backends

Nine kernel backends carry backend expertise. A campaign selects one with
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

## The measurement surface

Everything the loop decides on is produced by the task's own driver, invoked as
`python driver.py <args>` and read over stdout: the correctness suite, the
benchmark, and the workload that hardware profiling replays. That makes the
driver — together with its timing harness and correctness reference — the
measurement surface, and the loop protects it: an implementer edit or shell
write that touches it is refused while the session can still be saved.

The kernel sources are the opposite: the anchor named by `--kernel` and any
tracked implementation file outside the protected surface may be edited. What
the agent cannot do is change how it is graded.

See the {doc}`Optimization loop </kernelforge/conceptual/optimization-loop>` for the gates
each change clears, and the
{doc}`Autonomous overnight loop </kernelforge/how-to/autonomous-loop>` for how a long
unattended campaign is structured.
