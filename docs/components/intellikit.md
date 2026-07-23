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
[GEAK](geak.md), which depends on IntelliKit's Metrix for low-level GPU
profiling. Hyperloom therefore depends on IntelliKit transitively, through
GEAK.

- **Source**: <https://github.com/AMDResearch/intellikit>
- **License**: MIT

## Role in Hyperloom

Hyperloom doesn't reference IntelliKit directly. It is a runtime dependency of
[GEAK](geak.md), which pins IntelliKit's Metrix (`metrix`) for low-level GPU
profiling. IntelliKit is an indirect, transitive dependency reached through
GEAK.

## IntelliKit documentation

For detailed documentation on IntelliKit, see [IntelliKit on ROCm Docs](https://rocm.docs.amd.com/projects/intellikit/en/latest/).
