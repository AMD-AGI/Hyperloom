# KernelForge `local_knowledge`

Curated, source-grounded knowledge that a **kernel-optimization agent** loads so it doesn't start from zero
when optimizing AMD GPU operators. This README is for **humans** — it explains what's here, how it's
organized, and how to add/modify knowledge without breaking the conventions the agent relies on.

> **README vs INDEX.md** — this `README.md` is the human guide. Each knowledge *folder* also has an
> `INDEX.md`, which is the machine-facing **map that gets loaded whole into the agent prompt**. When you edit
> knowledge, you update the relevant `INDEX.md`; you rarely need to touch this README (only when the
> top-level layout or conventions change).

---

## What's here (top-level layout)

| Folder | What it holds | Has `INDEX.md`? |
|---|---|---|
| `framework/<lib>/` | **Operator-library control plane** — which op to call, how it dispatches, how to tune its per-shape DB. One subfolder per library. Currently: **`aiter`, `mori`**. | yes (per lib) |
| `languages/<lang>/` | **Kernel-authoring knowledge** per language/backend — how to write & optimize kernels. Currently: **`asm`, `ck`, `hip`, `triton`, `gluon`, `flydsl`**. | yes, except `flydsl` (TODO) |
| `hardware/` | **Backend-neutral hardware facts** (peaks, LDS, MFMA, cache). `cdna3_mi300/`, `cdna4_mi350/`, `shared/`. | yes |
| `common_methodology/` | **Cross-cutting methodology** independent of any library/arch: `optimization/`, `profiling/`. | yes |

Separation of concerns: **`framework/`** = "which library op + how to dispatch/tune it"; **`languages/`** =
"how to author the underlying kernel"; **`hardware/`** = "the chip's fixed facts"; **`common_methodology/`**
= "how to profile/optimize in general". A framework card that needs kernel-authoring detail *delegates* to
the matching `languages/<lang>/` folder rather than duplicating it.

---

## How it's organized (conventions)

### 1. The `INDEX.md` convention (folder map)
Every knowledge folder carries an `INDEX.md` that is the **entry map**: it states the folder's scope, the
pinned upstream source, a **problem → which-files-to-read → in-what-order** table, and a one-line role for
every file/subfolder. **Loader rule:** a folder that has an `INDEX.md` is navigated through it (loaded
whole); a folder without one falls back to a generated "filename — one-line description" listing. So the
`INDEX.md` is the single most important file to keep current.

### 2. `framework/<lib>/` layout (three reading layers)
```
framework/<lib>/
├── INDEX.md          # the map (load first)
├── overall/          # LAYER 1 — universal basics that apply to EVERY operator (read first):
│                     #   repo layout, dispatch/engagement, DB tuning, config system, build/JIT,
│                     #   operator/API catalog, tune-vs-author decision
├── operators/<op>/   # LAYER 2 — per-operator knowledge (see card set below)
└── skills/           # LAYER 3 — pick one WHEN you hit a problem:
    ├── profile/      #   measure a real workload, prove engagement, find the Amdahl-dominant op
    ├── bottleneck/   #   diagnose failures (0-engagement, build/JIT, parity traps)
    └── optimize/     #   domain-specific optimize levers (e.g. MoE / attention / FlyDSL)
```
Reading order: **`overall/` → the relevant `operators/<op>/` → a `skills/` card when a problem arises.**

### 3. `languages/<lang>/` layout
```
languages/<lang>/
├── INDEX.md
├── operators/<op>/   # per-operator authoring knowledge for this language
├── skills/           # profile / bottleneck / optimize (authoring levers, e.g. <lang>_levers/)
└── (API_docs/, etc.) # language-specific references where applicable (e.g. flydsl/API_docs/)
```

### 4. The per-operator card set
An operator folder holds up to five cards (create the ones that add value; lighter/newer ops may ship only
`overview.md`):

| File | Role |
|---|---|
| `overview.md` | what/why, math contract, shape regimes, Amdahl weight, backend landscape, how-to-bench |
| `<backend>.md` | the **SOTA card** — `aiter.md` under `framework/aiter/`, `<lang>.md` under `languages/<lang>/`: live dispatch/impl, config knobs, measured perf, integration seam, pitfalls |
| `fusion.md` | fusion neighbors (epilogues, fused entry kernels) |
| `numerics.md` | dtype/accumulate contract, parity bands, accuracy gating |
| `tuning.md` | per-backend knob space + the tune recipe |

### 5. Frontmatter & grounding discipline
- Cards use YAML frontmatter: `title`, `kind`, `gens`, `dtypes`, `regimes`, `updated`, `sources`. An
  `INDEX.md` additionally uses `kind: index`, `scope`, and `pinned_source`.
- **Pin to a real commit.** `sources:` entries are `<repo>@<commit>:<path>` (e.g.
  `ROCm/aiter@b467ce342:aiter/tuned_gemm.py`), ideally with `path:line`.
- **Ground everything in source. Do not fabricate** symbols, signatures, env vars, or performance numbers.
  If a number can't be reproduced from the repo, label it vendor-reported / unverifiable. State only what
  you verified; if unsure, say so.

---

## Adding or modifying knowledge — the workflow

1. **Pick the layer.** Framework op vs kernel-authoring vs hardware vs methodology; and within a framework
   lib, `overall/` (universal) vs `operators/<op>/` (specific) vs `skills/` (problem-triggered).
2. **Follow the card structure + frontmatter** above. Reuse the existing cards as templates.
3. **Ground it.** Read the real source, cite `path:line`, verify symbols/paths exist at the pinned commit.
4. **Update that folder's `INDEX.md`** (the must-not-skip step):
   - add the new operator to the catalog,
   - add a row to the problem→files reading-order table if it introduces a new task/symptom,
   - update the file-roles / tree section.
5. **Fix cross-links both ways** (relative markdown links between cards, and links from `INDEX.md`).
6. **Stamp provenance**: set `updated:` and the `sources:` pin on every card you touched.

## What to sync when the upstream library changes (re-pin)

When the tracked library (e.g. `ROCm/aiter`) is upgraded:
1. Re-verify every touched card against the **new commit** (symbols, paths, dispatch keys, CLI, configs).
2. Bump `pinned_source` in the folder's `INDEX.md`, and `sources:` + `updated:` in every card you change.
3. `grep` the folder for the **old commit hash** to catch stragglers (leaving a stale pin is "outdated").
4. Fix renamed symbols/paths and any moved files; adjust a card's `gens` **only if the repo proves** the
   arch is supported (don't over-claim).
5. Keep `git` history clean: use `git mv` for relocations/renames (not delete+recreate).

## Conventions cheat-sheet

| Item | Rule |
|---|---|
| Folder map | every folder has an `INDEX.md`; keep it in sync on any add/move |
| Language | all knowledge content is written in **English** |
| Pins | `sources:` = `<repo>@<commit>:<path[:line]>`; re-pin on upgrade |
| Truth | source-grounded only; no fabricated APIs/perf; unverifiable perf is labelled |
| Delegation | framework cards link to `languages/<lang>/` for kernel-authoring detail; don't duplicate |
| Hardware facts | live once in `hardware/`; cards reference, don't copy |
| Moves | use `git mv`; fix inbound/outbound links after moving |

---

## Current status (snapshot)
- `framework/aiter/` is grounded on `ROCm/aiter@b467ce342` (v0.1.16-283); it has `INDEX.md`, an `overall/`
  layer, 33 operator folders, and `skills/`.
- `framework/mori/` (added 2026-08-04) is grounded on `ROCm/mori@dc4bc75a`; it has `INDEX.md`, an
  `overall/` layer (repo scope, launch-config tuning control plane), and one operator folder
  (`ep_dispatch_combine`, the only op with real depth so far — mori's much wider surface, MORI-IO/CCL/IR/
  UMBP, is explicitly out of scope until someone reads that source). No `skills/` yet.
- `languages/{asm,ck,hip,triton}` and `hardware/`, `common_methodology/` each have an `INDEX.md`;
  **`languages/flydsl/` still needs one** (pending).
- Each folder's own `INDEX.md` records its specific pinned source and layout — start there.
