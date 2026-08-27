# vLLM Knowledge Base

Curated by vllm-fellow across sessions. The fellow is expected to grow this
directory over time — model forward-path maps, capture recipes that worked,
V0-vs-V1 engine quirks, and per-node achievable-BW measurements.

## Conventions

- One topic per `.md` file. Keep each under ~2K tokens.
- Filename pattern: `<scope>_<topic>.md` (e.g. `dsv32_v1_forward_map.md`,
  `mi355x_achievable_bw.md`, `<node>_capture_recipe.md`,
  `v0_v1_engine_diffs.md`).
- Reusable recipes go to `../skills/vllm_<name>.json` (cross-fellow
  indexed). The canonical capture recipe is already at
  `../skills/vllm_rocm_profiling_method.json` — read it before the first
  capture in any session.
- Update or replace stale entries; do not let this directory grow
  append-only.

## What to save here

- **Model-forward maps**: per engine version (V0 vs V1 differ in cudagraph
  capture, worker layout, profiler endpoints)
- **Capture recipes**: docker image tag, env vars, port, bench client
- **Confirmed dead ends with PMC evidence**
- **Achievable-BW rooflines** measured on specific nodes
- **Hot-kernel registry**: symbol → vllm source path → typical us per
  shape → owner backend

## What NOT to save here

- Per-run trace dumps (workspace, not KB)
- Kernel source (it's in vllm-amd / aiter repos)
- Generic backend pitfalls (those belong in the kernel fellow's KB)
