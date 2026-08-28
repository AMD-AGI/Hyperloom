# KernelForge: AI-Powered GPU Kernel Development

**Automated kernel optimization for AMD Instinct GPUs using multi-agent AI**

---

## Executive Summary

KernelForge is an agentic AI system that automates the full GPU kernel development cycle -- build, test, benchmark, profile, and optimize -- on AMD Instinct MI300X/MI355X accelerators. It replaces weeks of manual expert iteration with autonomous, measurement-driven optimization loops powered by 8 backend-specialized AI agents built on the Claude Agent SDK.

The system is born from production kernel work (Sparse Linear Attention, MoE mxfp4, Flash Attention) where every rule, pitfall, and validation gate comes from real engineering incidents. It captures and compounds institutional knowledge so that each optimization campaign makes the system smarter for the next.

**Key results delivered to date:**
- **1.35x end-to-end speedup** on Sparse Linear Attention (CK vs published Triton baseline)
- **1.33x decode throughput** on Kimi-K2 MoE inference (FlyDSL vs CK baseline)
- **1.15x speedup** on Sage Attention sparse forward (Triton optimization, fully autonomous campaign)
- Kernel development cycles reduced from **days of expert time to hours of autonomous operation**

---

## The Problem

High-performance GPU kernel development on AMD hardware is one of the most time-intensive bottlenecks in AI infrastructure. A single kernel optimization campaign typically requires:

- Deep knowledge of the gfx950 ISA, register file layout, and occupancy model
- Mastery of multiple kernel frameworks (CK, Triton, FlyDSL, AITER)
- Hundreds of build-test-bench-profile iterations to find optimal configurations
- Hardware-specific pitfalls that are only discovered through costly trial and error (stale build artifacts, register pressure cliffs, LDS aliasing bugs)

This work is done by a small number of domain experts, creating a critical bottleneck in AMD's AI software stack.

---

## The Solution

KernelForge deploys 8 kernel backend agents, each with domain-specific knowledge and GPU tooling access, driven by a measurement-gated iteration loop.

### Architecture

```
                   Campaign — kernelforge forge-loop
              One kernel, one driver, one change per iteration
                              |
        Kernel backends (8) — domain knowledge for the kernel at hand
        +----------+----------+----------+----------+----------+----------+----------+
        |          |          |          |          |          |          |          |
         CK         FlyDSL     Triton     AITER      HIP        hipBLASLt  IntelliKit
         (C++ CK    (MLIR      (JIT       (Pre-      (Raw HIP   (Dense     (ISA-level
         templates) DSL)       Python)    built)     + HK)      GEMM)      ASM tune)
        |          |          |          |          |          |          |          |
        +----------+----------+----------+----------+----------+----------+----------+
                              |
                 GPU toolchain (Bash on the host)
               build | test | bench | pmc | registers
                              |
                       Learning System
            PostMortem | Tuning DB | Skills | Sources
```

**Three session roles inside a campaign:**
- **Analysis specialists** -- Read-only sessions that turn profiler evidence into scoped findings (compute, memory, algorithm), fused into ONE executable plan per iteration
- **Implementer** -- The only writable session: applies that plan to the kernel sources and exercises its own correctness and performance gate
- **Supervisor** -- Invoked when the search stalls; inspects the run and writes a fresh direction for the next plan

### How It Works

1. **Campaign inputs** specify the kernel anchor, the driver that owns correctness and timing, the kernel backend whose knowledge is injected, and a wall-clock budget
2. **Planning** turns the current hardware evidence into one executable change per iteration
3. Each **iteration** runs a disciplined loop:
   - Build (with stale-artifact detection)
   - Test correctness (SNR >= 30 dB gate -- blocks further work if numerics are wrong)
   - Benchmark (30-iteration median, in-context measurement)
   - Profile hardware counters (rocprofv3 PMC: MFMA utilization, memory stalls, occupancy)
   - Analyze and decide the single next change to make
4. **Learning system** extracts lessons from every experiment, growing a tuning database and skill library
5. Campaign ends at its time budget, an explicit terminal gate, or an operator stop file in the workspace; stalls trigger replanning

---

## Production Results

### Sparse Linear Attention -- CK Backend (MI355X)

Full forward + backward attention kernel, optimized from scratch to beat the published Triton autotuned baseline.

| Metric | Triton (Published) | CK (KernelForge) | Speedup |
|--------|-------------------|---------------------|---------|
| Config B total | 60.85 ms | 44.97 ms | **1.35x** |
| Config C total | 58.12 ms | 42.68 ms | **1.33x** |
| Config C bwd only | 46.71 ms | 33.84 ms | **1.38x** |

The backward kernel alone went through 6 optimization phases -- split pipeline, constexpr masks, bf16 delta preprocessing, occupancy-2 for both dkdv and dq sub-kernels -- each validated by hardware counter analysis (VGPR 296 to 238, zero spills, occupancy 1 to 2).

### Kimi-K2 MoE mxfp4 Inference (MI355X)

FlyDSL replacement for CK mxfp4 MoE kernels across all decode and prefill shapes.

| Token Count | CK Baseline | FlyDSL (KernelForge) | Speedup |
|-------------|-------------|------------------------|---------|
| 64 (decode) | 287 us | 268 us | 1.08x |
| 2048 | 823 us | 620 us | **1.33x** |
| 8192 | 2159 us | 1745 us | **1.24x** |

Absolute savings scale linearly: 414 us/layer saved at 8K tokens, directly reducing time-to-first-token for long prompts.

### Sage Attention Sparse Forward (MI355X)

Fully autonomous campaign -- agents ran overnight on a SLURM cluster with no human intervention.

- The Triton campaign achieved **1.15x speedup** (1169 to 1363 TFLOPS)
- Hardware-counter evidence identified HBM bandwidth as the ceiling, with CK int8 as the path to 1.2x+
- The follow-on CK campaign started from a complete int8 integration package (7/7 pieces)

---

## Key Differentiators

### Measurement-Driven, Not Trial-and-Error

Every optimization decision is grounded in hardware counter data. The system predicts the PMC impact of a change *before* building, then validates the prediction against reality. This prevents the "try random things and hope" pattern that wastes GPU hours.

**Example:** The dq-only kernel occupancy-2 optimization was guided by register analysis (296 VGPR to 238) and verified by MFMA/wait-cycle ratio (< 5 = compute-bound). The agent identified that dropping persistent q_reg/do_reg registers and reloading from LDS would free exactly enough VGPRs -- no overshoot, no undershoot.

### Self-Improving Knowledge Base

The system gets stronger with every campaign:

- **Tuning Database** stores every (config, performance) pair and derives transfer rules across operations
- **PostMortem Analysis** automatically extracts pitfalls (>15% regression) and optimizations (>15% improvement)
- **Skill Hierarchy** compounds atomic optimizations into reusable strategies (e.g., "if memory-bound AND VGPR > 256, apply occupancy optimization first")
- **Source Ingestion** incorporates ISA manuals, papers, and reference kernel code

### Production-Hardened Pitfall Library

Every validation gate exists because of a real incident:

| Pitfall | Consequence Without Detection | How KernelForge Prevents It |
|---------|-------------------------------|-------------------------------|
| Stale .so artifacts | Benchmarks measure old code; hours wasted | Build tool auto-cleans and verifies deployment |
| BLOCK_M=64 with sparse attention | Silent data corruption across workgroups | Hard constraint in knowledge base |
| VGPR > 256 occupancy cliff | Performance drops 2x at a boundary that looks like 4 more registers | Register analysis tool predicts before building |
| AGPR inline asm register drop | 21 dB SNR (silently wrong) instead of 142 dB | Correctness gate blocks benchmarking at < 30 dB |
| Ninja dependency tracking gaps | Header edits don't compile in; stale builds appear "unchanged" | Mandatory stale-object cleanup before every build |

### Multi-Backend Flexibility

Unlike single-framework tools, KernelForge works across CK (C++ templates), FlyDSL (MLIR DSL), Triton (JIT Python), and AITER (pre-built operators). Each campaign runs with the kernel backend knowledge for the backend it is optimizing, and the driver contract is backend-agnostic -- when one backend plateaus, the same correctness oracle and benchmark carry straight over to a campaign on another, including a FlyDSL rewrite of the plateaued kernel.

---

## Deployment Modes

| Mode | Use Case | How It Works |
|------|----------|--------------|
| **Claude Code in-session** | Interactive campaigns from a Claude Code session | The campaign runs as a background command whose build, bench, PMC read, and KEEP/REVERT decision stream into the session; bills against Claude Code Max |
| **Autonomous Loop** | Overnight optimization campaigns | Evidence-planned working-tree candidates with driver-owned validation and measured KEEP/REVERT |

The autonomous loop mode (`kernelforge forge-loop`) is particularly valuable
for overnight campaigns. It keeps candidates in the working tree, runs the
driver-owned complete correctness suite, and commits only candidates that pass
three independent benchmarks. It is TIME-driven via `--max-hours`; stalled
searches invoke Supervisor guidance while measurement remains authoritative.

---

## Infrastructure Integration

- **Operator control** -- A campaign streams its iteration decisions to stdout and stops cleanly at the next iteration boundary when a `.stop` file appears in its workspace, finalizing its best commit, result JSON, and lessons
- **Token Optimization** -- RTK integration reduces CLI output tokens by 60-90%, cutting API costs for long build/test/profile loops

---

## Scale

| Metric | Value |
|--------|-------|
| Python source files | ~90 |
| Lines of code | ~12,700 |
| Test coverage | 65 unit tests |
| Knowledge base files | 30+ curated + auto-generated, plus vendored playbooks (CK, asm, HipKittens, CDNA4 ISA) |
| Supported backends | 7 (CK, FlyDSL, Triton, AITER, HIP, hipBLASLt, IntelliKit) |
| Runnable examples shipped | 9 (softmax tutorials, production hot kernels, a collective, a distributed op, 2 FlyDSL rewrites) |
| GPU toolchain stages | 5 (build, test, bench, pmc, registers) |

---

## What's Next

| Initiative | Description | Expected Impact |
|------------|-------------|-----------------|
| **Paper and doc ingestion** | Automatic learning from ISA manuals, architecture docs, and research papers | Broader optimization repertoire |

---

## Summary

KernelForge transforms GPU kernel development from a scarce-expert, weeks-long process into an automated, measurement-driven system that delivers production-quality speedups overnight. It compounds institutional knowledge across campaigns, making each optimization faster than the last. The system has already delivered 1.24-1.38x speedups on production kernels for AMD Instinct MI355X -- and it gets smarter with every run.

---

*Built on the [Claude Agent SDK](https://docs.anthropic.com/en/docs/agent-sdk) | MIT License | [github.com/AMD-AGI/KernelForge](https://github.com/AMD-AGI/KernelForge)*
