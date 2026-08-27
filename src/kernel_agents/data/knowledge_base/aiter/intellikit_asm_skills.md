# IntelliKit ASM Skills (gfx950 / MI355X) — agent-driven AMDGCN assembly

Pointer file. A vendored copy of AMD RAD's IntelliKit ASM Skills lives in-repo.
It is per-instruction and guide reference for **hand-written AMDGCN assembly**
on gfx950 — every NOP rule, hazard, cycle count, and register trick was measured
empirically on MI355X silicon (much of it is not in the public ISA docs).

**Canonical location (use this path for `Read`/`Grep`):**
`local_knowledge/languages/asm/skills/optimize/asm_levers/intellikit/`

`Read`/`Grep` into the canonical subdir when writing or debugging `.co` assembly. Each
instruction doc has YAML frontmatter (`instruction:`, `category:`, `tags:`) so
`grep -rn 'tags:.*<term>'` finds the relevant docs fast.

## What's in the IntelliKit pack

| Path | Contents |
|---|---|
| `METHODOLOGY.md` | The core workflow: disassemble a `.co` → round-trip validate bit-identical → one targeted change at a time → profile, don't guess |
| `guides/` (6 files) | `kernel-optimization-workflow`, `debugging-playbook` (symptom→cause + top-10 bugs), `kernel-architecture` (7 kernel families: GEMM/attention/GEMV/grouped-GEMM/uber-kernel), `register-allocation` (VGPR/AGPR budgeting, occupancy breakpoints, accum_offset), `lds-patterns` (bank conflicts, multi-buffering, direct-to-LDS, swizzle), `memory-coherence-formats` (waitcnt FIFO rules, FP8/BF16 handling) |
| `instructions/` (67 docs) | Per-instruction: syntax, **measured cycle counts**, counter tracking, hazards, code patterns. Categories: MFMA (bf16/f16/fp8/f8f6f4/i8/f64 + scaled), LDS (`ds_read/write*`, `ds_read_tr`, `ds_bpermute`), memory (`buffer_*`, `global_*`, `flat_*`, `buffer_load_lds`), sync/control (`s_waitcnt`, `s_barrier`, `nop_hazard_summary`, `kernel_descriptor`, `hsa_aql_dispatch`, `inter_wg_barrier`), VALU/convert, AGPR (`v_accvgpr_*`), DPP/cross-lane |
| `tools/scripts/vgpr_liveness.py` | VGPR liveness analyzer — parses assembly, finds dead register windows, suggests remappings to lift occupancy (`--json` for agent use) |
| `README.md`, `CONTRIBUTING.md` | Full navigation tables + doc templates |

## Highest-value entry points

- `instructions/nop_hazard_summary.md` — complete NOP table; the single most
  important reference for avoiding silent corruption.
- `instructions/kernel_descriptor.md` — `accum_offset`, `.args` metadata, AGPR
  aliasing — the launch-failure bugs in one place.
- `instructions/s_waitcnt.md` — vmcnt/lgkmcnt FIFO ordering; get it wrong and
  loads return stale data.
- `instructions/buffer_load_lds.md` — direct HBM→LDS loads, ~17% over the
  `buffer_load` + `ds_write` path.
- `guides/debugging-playbook.md` — start here when a kernel is wrong/slow.

## Relation to the other aiter ASM knowledge

Complements `asm_perf_playbook_v2.md` (technique catalogue) and
`asm_kernel_knowledge.md` (kernel-family map): those are *strategy*, IntelliKit
is the *per-instruction ground truth*. For raw ISA spec text see
`shared/cdna4_isa_reference.md`.
