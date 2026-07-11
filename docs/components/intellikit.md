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

## Packages and tools

IntelliKit groups its tools around a kernel optimization workflow —
**isolate → profile → inspect → validate** — with MCP servers and agent skills
to wire the tools into large language model (LLM) clients:

- **Kerncap** — isolate a kernel: capture GPU dispatches and build standalone,
  replayable reproducers (HIP, Triton).
- **Metrix** — profile: human-readable metrics from hardware counters
  (bandwidth, cache, etc.).
- **Linex** — profile: source-line timing and stalls (compile with `-g` for
  `file:line` mapping).
- **Nexus** — inspect: from HSA packets, see what actually ran, including source
  and assembly.
- **Accordo** — validate: prove an optimized kernel still matches a reference.
- **rocm_mcp** — MCP servers for HIP compile, HIP docs, `rocminfo`, and
  `amd-smi`.
- **uprof_mcp** — host-side CPU hotspot analysis through an MCP bridge to AMD uProf.

The stack requires Python 3.10+ and ROCm 6.0+ for the GPU packages (ROCm 7.0+
for Linex). Accordo and Nexus additionally compile C++ during install (through
[KernelDB](https://github.com/AMDResearch/KernelDB)) and need `cmake`,
`libdwarf-dev`, and `libzstd-dev`.

## Installation

There is no metapackage at the repository root; each package is installed
independently. The convenience installer pulls every package from Git:

```bash
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash
```

To install a subset, or an individual package directly from Git:

```bash
# Subset via the installer
curl -sSL https://raw.githubusercontent.com/AMDResearch/intellikit/main/install/tools/install.sh | bash -s -- --tools metrix,linex

# A single package from Git
pip install "git+https://github.com/AMDResearch/intellikit.git#subdirectory=metrix"
```

For development, clone the repo and use editable installs per package:

```bash
git clone https://github.com/AMDResearch/intellikit.git && cd intellikit
pip install -e ./metrix
pip install -e ./linex
```

```{note}
Accordo and Nexus build native code during `pip install`. Install `cmake`,
`libdwarf-dev`, and `libzstd-dev` (`libdwarf-devel` / `libzstd-devel` on
Fedora/RHEL) first, or use the [IntelliKit Docker image](https://github.com/AMDResearch/intellikit/blob/main/docker/Dockerfile),
which ships these dependencies.
```

## Usage

IntelliKit exposes both a Python API and console entry points, with MCP server support for integrating directly into LLM agent workflows.

### Python API

Metrix exposes a simple profiling API:

```python
from metrix import Metrix

profiler = Metrix()
results = profiler.profile("./your_app", metrics=["memory.hbm_bandwidth_utilization"])

for kernel in results.kernels:
    print(f"{kernel.name}: {kernel.duration_us.avg:.2f} us")
```

### Console entry points

Each IntelliKit package installs these console scripts (CLIs and/or `*-mcp` MCP servers):

| Package | CLI | MCP server entry point |
|---------|-----|------------------------|
| Kerncap | `kerncap` | `kerncap-mcp` |
| Metrix | `metrix` | `metrix-mcp` |
| Linex | — | `linex-mcp` |
| Nexus | — | `nexus-mcp` |
| Accordo | `accordo` | `accordo-mcp` |
| rocm_mcp | — | `hip-compiler-mcp`, `hip-docs-mcp`, `amd-smi-mcp`, `rocminfo-mcp` |
| uprof_mcp | — | `uprof-profiler-mcp` |

### MCP clients

With `uv` and a clone of the repo, point an MCP client at a package directory:

```json
{
  "mcpServers": {
    "metrix-mcp": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/intellikit/metrix", "metrix-mcp"]
    }
  }
}
```

IntelliKit also ships installable agent skills (`install/skills/install.sh`) for
Cursor, Claude, and Codex.

## Role in Hyperloom

Hyperloom doesn't reference IntelliKit directly. Kernel profiling and
validation in Hyperloom is powered by [Magpie](magpie.md), which in turn relies
on IntelliKit for some of its low-level GPU profiling primitives. IntelliKit is
an indirect, transitive dependency reached through Magpie.

## API reference

IntelliKit ships its own documentation in-repo and as a live site; see the
[IntelliKit documentation](https://amdresearch.github.io/intellikit) and the
per-package READMEs and examples under each package directory
([Kerncap](https://github.com/AMDResearch/intellikit/tree/main/kerncap),
[Metrix](https://github.com/AMDResearch/intellikit/tree/main/metrix),
[Linex](https://github.com/AMDResearch/intellikit/tree/main/linex),
[Nexus](https://github.com/AMDResearch/intellikit/tree/main/nexus),
[Accordo](https://github.com/AMDResearch/intellikit/tree/main/accordo)).
