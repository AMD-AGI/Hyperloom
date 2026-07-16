---
myst:
    html_meta:
        "description": "Learn about AgentKernelArena, a standardized AMD evaluation arena for measuring AI coding agent performance on GPU kernel optimization tasks on ROCm."
        "keywords": "AgentKernelArena, GPU kernel, optimization, AI agents, ROCm, HIP, Triton, AMD GPU, evaluation, benchmark, leaderboard, Hyperloom"
---
# AgentKernelArena

AgentKernelArena is a standardized evaluation arena, built by AMD, that measures
how well AI coding agents perform on real GPU kernel optimization tasks. It runs
LLM-powered agents side-by-side on the same tasks in isolated workspaces and
scores them with objective, reproducible metrics for compilation, correctness,
and GPU performance.

- **Source**: <https://github.com/AMD-AGI/AgentKernelArena>
- **License**: MIT

## Role in Hyperloom

AgentKernelArena is not part of Hyperloom's default install or optimization
loop. It is a companion evaluation harness for comparing Hyperloom's
kernel-optimization agents against other AI coding agents on standardized GPU
kernel tasks. Its task categories (`hip2hip`, `triton2triton`, `flydsl2flydsl`,
and others) overlap with the kernel types that Hyperloom's kernel agent handles
through GEAK, but Hyperloom does not clone or invoke
AgentKernelArena in the normal runtime path.

## AgentKernelArena documentation

For detailed documentation on AgentKernelArena, see [AgentKernelArena on ROCm Docs](https://rocm.docs.amd.com/projects/agent-kernel-arena/en/latest/).
