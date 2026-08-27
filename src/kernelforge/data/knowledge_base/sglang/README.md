# SGLang Knowledge Base

Curated by sglang-fellow across sessions. The fellow is expected to grow this
directory over time — model forward-path maps, capture recipes that worked,
confirmed dead ends, and per-node achievable-BW measurements.

## Conventions

- One topic per `.md` file. Keep each under ~2K tokens.
- Filename pattern: `<scope>_<topic>.md` (e.g. `dsv32_forward_map.md`,
  `mi355x_achievable_bw.md`, `<node>_capture_recipe.md`).
- Reusable, parameterizable recipes go to
  `../skills/sglang_<name>.json` (cross-fellow indexed), not here.
- Update or replace stale entries; do not let this directory grow
  append-only. A 6-month-old "hot kernel" list is worse than no list.

## What to save here

- **Model-forward maps**: which kernels at which layer, for which models
- **Capture recipes**: docker image, env vars, port, bench client
- **Confirmed dead ends**: with the PMC evidence that closed them
- **Achievable-BW rooflines** measured on specific nodes
- **Hot-kernel registry**: symbol → source path → typical us → owner backend

## What NOT to save here

- Per-run trace dumps (those belong in the workspace, not the KB)
- Kernel source (it's in the sglang/aiter repo)
- Generic Triton/CK pitfalls (those go in the respective fellow's KB)
