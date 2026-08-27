<h1 align="center">IntelliKit ASM Skills</h1>
<p align="center"><em>Built by AMD Research and Advanced Development (RAD)</em></p>

Skill documents for agent-driven AMDGCN assembly kernel generation on gfx950 (MI355X / CDNA4). 67 instruction docs, 6 guides, and a [methodology](METHODOLOGY.md) — all validated on real MI355X silicon.

Agent-generated kernels built with these docs: Llama 3.1 8B all-assembly inference **1.23x faster than vLLM**, DeepSeek-V4-Flash 236B single-GPU, attention FWD+BWD **23% faster** and grouped GEMM **1.74x speedup** (both contributed to MLPerf), 13 GEMVs at 81-100% HBM bandwidth.

---

## Quick Start

Add this to your agent's system prompt or CLAUDE.md:

```
Clone https://github.com/ROCm/intellikit-asm-skills and read
METHODOLOGY.md and the guides/ directory. Use the instruction docs in
instructions/ as reference when writing gfx950 assembly.
```

Assumes bare-metal or SSH access to a machine with AMD Instinct GPUs and ROCm installed.

That's it. Give it your problem.

### The Methodology

The workflow that consistently produces fast, correct kernels:

1. **Start from reference assembly** — disassemble an optimized `.co`, don't write from scratch
2. **Round-trip validate** — reassemble, verify bit-identical output before changing anything
3. **Make targeted optimizations** — one change at a time, measure after each
4. **Profile, don't guess** — roofline analysis, hardware counters, identify the actual bottleneck

Full details in **[METHODOLOGY.md](METHODOLOGY.md)**.

### I want to...

| I want to... | Start here |
|--------------|-----------|
| Write assembly by hand | [NOP hazard summary](instructions/nop_hazard_summary.md), [kernel descriptor](instructions/kernel_descriptor.md) |
| Optimize an existing kernel | [METHODOLOGY.md](METHODOLOGY.md), [register allocation](guides/register-allocation.md) |
| Debug a broken kernel | [Debugging playbook](guides/debugging-playbook.md) |
| Understand a kernel family | [Kernel architecture](guides/kernel-architecture.md) — GEMM, attention, GEMV, uber-kernel |
| Use an agent to write kernels | Point it at this repo — the YAML frontmatter is designed for agent consumption |
| Reduce VGPR pressure | [vgpr_liveness.py](tools/scripts/vgpr_liveness.py), [register allocation](guides/register-allocation.md) |
| Improve compiler codegen | [NOP hazards](instructions/nop_hazard_summary.md), [LDS patterns](guides/lds-patterns.md), [memory coherence](guides/memory-coherence-formats.md) |

---

## Reference

### Guides

| Guide | Description |
|-------|-------------|
| [kernel-optimization-workflow.md](guides/kernel-optimization-workflow.md) | The methodology: 3 approaches, round-trip workflow, profiling, optimization loop |
| [debugging-playbook.md](guides/debugging-playbook.md) | Symptom-to-cause lookup, validation harness, top 10 bugs |
| [kernel-architecture.md](guides/kernel-architecture.md) | 7 kernel families: GEMM, attention, GEMV, grouped GEMM, uber-kernel |
| [register-allocation.md](guides/register-allocation.md) | VGPR/AGPR budgeting, occupancy breakpoints, accum_offset, register maps |
| [lds-patterns.md](guides/lds-patterns.md) | Bank conflicts, double/triple buffering, direct-to-LDS, swizzle formulas |
| [memory-coherence-formats.md](guides/memory-coherence-formats.md) | Memory ordering, waitcnt FIFO rules, FP8/BF16 handling, toolchain |

### Instruction Docs

67 per-instruction docs in [`instructions/`](instructions/). Each includes YAML frontmatter for agent indexing, syntax, measured cycle counts, counter tracking, hazards, and code patterns.

**Key references:**

| Doc | Description |
|-----|-------------|
| [nop_hazard_summary.md](instructions/nop_hazard_summary.md) | Complete NOP table — the single most important reference for avoiding silent corruption |
| [kernel_descriptor.md](instructions/kernel_descriptor.md) | accum_offset, .args metadata, AGPR aliasing — every launch-failure bug in one place |
| [s_waitcnt.md](instructions/s_waitcnt.md) | vmcnt/lgkmcnt FIFO ordering — get this wrong and loads return stale data |
| [buffer_load_lds.md](instructions/buffer_load_lds.md) | Direct HBM-to-LDS loads — 17% speedup over the buffer_load + ds_write path |
| [v_mfma_f32_16x16x32_bf16.md](instructions/v_mfma_f32_16x16x32_bf16.md) | BF16 MFMA — the workhorse instruction for GEMM and attention |

### Tools

| Tool | Description |
|------|-------------|
| [vgpr_liveness.py](tools/scripts/vgpr_liveness.py) | VGPR liveness analyzer — parse assembly, find dead register windows, get remapping suggestions to improve occupancy. Supports `--json` for agent consumption. |
| [atlassian/](tools/mcp-servers/atlassian/) | MCP server for Jira + Confluence. Search issues, read specs and architecture docs during kernel development. |

---

## Contributing

PRs welcome from humans and agents alike. See [CONTRIBUTING.md](CONTRIBUTING.md) for templates.

- **New instruction docs** — undocumented hazard or instruction? Add a doc with YAML frontmatter.
- **Corrections** — wrong NOP count, stale cycle measurement, inaccurate hazard rule? Fix it.
- **Agent-generated findings** — your agent found something useful on MI355X? Open a PR.
- **Other architectures** — measurements on MI300X, MI350X, etc. Same format, different architecture tag.

---

## Origin

This knowledge base grew out of a RAD research project exploring agent-driven kernel development on MI355X. Over 25+ sessions spanning GEMM, attention, GEMV, grouped GEMM, and full inference pipelines, we used Claude Code to write, profile, and optimize hand-tuned AMDGCN assembly. Along the way, every hardware hazard, NOP rule, register trick, and LDS pattern was documented empirically — many of these findings are not in public ISA documentation.

The setup: Claude Code + these skill documents + MCP servers over Atlassian and Perforce. No custom infrastructure.

---

## Contact

Muhammad Awad — [muhaawad@amd.com](mailto:muhaawad@amd.com)
