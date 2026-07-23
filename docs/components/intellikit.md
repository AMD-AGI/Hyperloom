---
myst:
    html_meta:
        "description": "Learn about IntelliKit, Hyperloom's GPU profiling and validation toolkit. Covers Kerncap, Metrix, Linex, Nexus, Accordo, and MCP server packages for AMD GPU workloads."
        "keywords": "IntelliKit, Hyperloom, GPU profiling, AMD GPU, ROCm, Kerncap, Metrix, Linex, Nexus, Accordo, MCP, kernel optimization, HIP, hardware counters, benchmarking"
---
# IntelliKit

IntelliKit is a set of agent-first Python tools for AMD-focused performance
and validation. Most of the stack targets GPUs through ROCm, turning hardware
counters, traces, dispatch data, and Heterogeneous System Architecture (HSA) packets into clear Python APIs and
Model Context Protocol (MCP) servers. It's structured as a monorepo of
independently installable packages (Kerncap, Metrix, Linex, Nexus, Accordo, plus
the `rocm_mcp` and `uprof_mcp` server bundles).

Within Hyperloom, IntelliKit is not called directly. It sits underneath
[Magpie](magpie.md), which relies on IntelliKit for some of its low-level GPU
profiling primitives. Hyperloom therefore depends on IntelliKit transitively,
through Magpie.

- **Source**: <https://github.com/AMDResearch/intellikit>
- **License**: MIT

## Role in Hyperloom

Hyperloom doesn't reference IntelliKit directly. Kernel profiling and
validation in Hyperloom is powered by [Magpie](magpie.md), which in turn relies
on IntelliKit for some of its low-level GPU profiling primitives. IntelliKit is
an indirect, transitive dependency reached through Magpie.

## IntelliKit documentation

For detailed documentation on IntelliKit, see [IntelliKit on ROCm Docs](https://rocm.docs.amd.com/projects/intellikit/en/latest/).
