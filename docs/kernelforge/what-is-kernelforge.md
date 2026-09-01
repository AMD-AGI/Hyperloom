---
myst:
  html_meta:
    "description": "What is KernelForge: an autonomous, measurement-driven system that develops and optimizes GPU kernels on AMD Instinct hardware using domain-specialized AI agents."
    "keywords": "KernelForge, GPU kernel, AMD Instinct, MI355X, gfx950, agentic optimization, ROCm, PMC, Composable Kernel, Triton, HIP"
---

# What is KernelForge?

KernelForge is an autonomous, measurement-driven system for developing and
optimizing high-performance GPU kernels on AMD Instinct hardware. It replaces
weeks of manual expert iteration with domain-specialized AI agents that build,
benchmark, and profile every change against real hardware performance counters,
and learn from each campaign.

## The problem

High-performance GPU kernel development on AMD hardware is one of the most
time-intensive bottlenecks in AI infrastructure. A single kernel optimization
requires deep ISA knowledge, mastery of multiple frameworks, hundreds of
build-test-bench iterations, and hardware-specific pitfalls that are only
discovered through costly trial and error. This work is done by a small number
of domain experts, creating a critical bottleneck.

## The approach

KernelForge optimizes one kernel at a time with an autonomous iteration loop.
The loop drives an agent carrying one **kernel backend's** expertise —
Composable Kernel, Triton, HIP, hipBLASLt, FlyDSL, AITER, or hand-written gfx950
assembly — through an enforced development cycle: build, an SNR correctness
gate, benchmark, hardware-counter (PMC) analysis, then a keep or revert
decision, so no change is accepted without measured evidence.

Key properties:

- **Measurement-driven.** Every decision is grounded in hardware counters
  (for example, the wait/MFMA ratio classifies a kernel as compute-, memory-,
  or latency-bound) rather than guesswork.
- **Enforced discipline.** Correctness and performance gates are hard
  constraints; a change that regresses or fails validation is reverted.
- **Self-improving.** Each campaign distills lessons and pitfalls into a
  knowledge base that makes the next campaign smarter.

## Where to go next

- Install and run your first task: {doc}`Quickstart </kernelforge/install/quickstart>`.
- Understand the system: {doc}`Architecture </kernelforge/conceptual/architecture>` and the
  {doc}`Optimization loop </kernelforge/conceptual/optimization-loop>`.
- Drive a run: {doc}`Run a campaign </kernelforge/how-to/run-a-campaign>`.
