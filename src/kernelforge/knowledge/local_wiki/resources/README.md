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
| `languages/<lang>/` | **Kernel-authoring knowledge** per language/backend — how to write & optimize kernels. Currently: **`asm`, `ck`, `hip`, `triton`, `gluon`, `flydsl`, `fusion`**. | yes |
| `hardware/` | **Backend-neutral hardware facts** (peaks, LDS, MFMA, cache) — **CDNA4 / gfx950 only**. Flat: one `mi350_*.md` card per subsystem. | yes |
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

### 2. `framework/<lib>/` layout
```
framework/<lib>/
├── INDEX.md          # the map (load first)
├── overall/          # LAYER 1 — universal basics that apply to EVERY operator (read first):
│                     #   repo layout, dispatch/engagement, DB tuning, config system, build/JIT,
│                     #   operator/API catalog, tune-vs-author decision
├── skills/           # LAYER 2 — pick one WHEN you hit a problem:
│   ├── profile/      #   measure a real workload, prove engagement, find the Amdahl-dominant op
│   ├── bottleneck/   #   diagnose failures (0-engagement, build/JIT, parity traps)
│   └── optimize/     #   domain-specific optimize levers (e.g. MoE / attention / FlyDSL)
└── operators/<op>/   # OPTIONAL — only where a cross-GPU/library seam needs it (today: mori only)
```
Reading order: **`overall/` → a `skills/` card when a problem arises.**

**`framework/aiter/` has no `operators/` layer** — see §3b. `framework/mori/` keeps one because the EP
dispatch/combine seam *is* the library, not one operator among many.

### 3. `languages/<lang>/` layout
```
languages/<lang>/
├── INDEX.md
├── skills/           # profile / bottleneck / optimize (authoring levers, e.g. <lang>_levers/)
└── (API_docs/, etc.) # language-specific references where applicable (e.g. flydsl/API_docs/)
```

> **Language folders are language-level only — no `operators/`.** `triton`, `ck`, `hip`, `asm` and
> `flydsl` used to each carry a full per-operator card set, but `overview`/`tuning`/`numerics`/`fusion`
> are operator-level facts that don't change with the authoring language: the same card existed 3–5
> times over. A language folder documents *how to author in that language*, nothing more.
>
> Two exceptions keep an `operators/`: **`gluon/`** (3 authoring cards with no counterpart elsewhere)
> and **`fusion/`** (its `operators/*.md` *are* the fused-pattern definitions, not per-operator cards).

### 3b. This base does not maintain per-operator knowledge
There is **no operator encyclopaedia here**. "What is this operator, what are its shape regimes, what's
its parity band, which kernel is fastest for it today" is not answered by `local_knowledge` — read the
source, run the benchmark, or use an external reference.

Two reasons, and the second is the decisive one:
1. Most of what we had was a verbatim copy of upstream `perf_knowledge` operator cards, repeated 3–5
   times across language folders. A second copy bought nothing.
2. **Operator-level facts rot the fastest.** Which backend wins, what the config knobs are, which env
   var gates which path — these turn over every aiter/vLLM/SGLang release. A card one release behind is
   *worse* than no card: it routes the agent to an entry point that no longer exists, confidently.

So the base documents what stays true across releases: the dispatch model, the config-DB mechanics, the
build system, the engagement-proof workflow, the hardware, the methodology. For a specific operator, the
route is `framework/aiter/overall/operator_catalog.md` (entry point + signature) → the source.

The one surviving `operators/` under `framework/` is **`framework/mori/operators/`** (EP
dispatch/combine) — that seam is MoRI's whole reason to exist and is written here end to end.

**Bar for adding an operator card:** it must describe a *library-structural* fact (a dispatch seam, a DB
schema, a cross-GPU protocol) that survives the library's next release. "Currently fastest config for X"
is a benchmark result, not a document.

### 4. The per-operator card set
Applies to `framework/<lib>/operators/<op>/` where one exists (today: `framework/mori/`). An operator
folder holds up to five cards (create the ones that add value; lighter ops may ship only `overview.md`):

| File | Role |
|---|---|
| `overview.md` | what/why, math contract, shape regimes, Amdahl weight, backend landscape, how-to-bench |
| `<backend>.md` | the **SOTA card** — e.g. `hip.md` / `triton.md` / `mori.md`: live dispatch/impl, config knobs, measured perf, integration seam, pitfalls |
| `fusion.md` | fusion neighbors (epilogues, fused entry kernels) |
| `numerics.md` | dtype/accumulate contract, parity bands, accuracy gating |
| `tuning.md` | per-backend knob space + the tune recipe |

### 4b. `hardware/` layout — flat, one card per subsystem
```
hardware/
├── INDEX.md
└── mi350_<subsystem>.md    # overview · execution · matrix_core · dtypes · lds · memory · chiplet · isa · clocks
```
No subfolders and no per-generation split: **gfx950 (MI350X / MI355X) only**. Each card carries both the
mental model *and* the concrete numbers for its subsystem, so one Read answers a question end to end.
Earlier generations appear only as porting warnings ("that value is MI300X's — here it is X").

### 4c. Filename conventions (deliberate — do not "normalize" them)
Cards in the cross-cutting folders carry a prefix that marks the folder and keeps filenames distinct
from any upstream knowledge base, so a card can never be confused with — or silently overwritten by —
an external copy:

| Prefix | Folder | Meaning |
|---|---|---|
| `lever_` | `common_methodology/optimization/` | a technique you apply |
| `measure_` | `common_methodology/profiling/` | a way to observe |
| `mi350_` | `hardware/` | a gfx950 subsystem |
| `ck_` | `languages/ck/skills/optimize/ck_levers/` | a CK authoring/tuning lever |
| `triton_` | `languages/triton/skills/optimize/triton_levers/` | a Triton authoring/tuning lever |
| `hip_` | `languages/hip/skills/optimize/hip_levers/` | a HIP/C++ authoring lever |
| `flydsl_` | `languages/flydsl/skills/optimize/flydsl_levers/` | a FlyDSL authoring/tuning lever |
| `aiter_` | `framework/aiter/skills/optimize/aiter_levers/` | an aiter-dispatch lever for one domain |

`INDEX.md` keeps its name everywhere — it is the loader contract (`build_forge_knowledge` loads a
folder's `INDEX.md` whole).

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

1. **Pick the layer.** Framework vs kernel-authoring vs hardware vs methodology; and within a framework
   lib, `overall/` (universal) vs `skills/` (problem-triggered). **Before writing anything operator-specific,
   re-read §3b** — it is almost always the wrong layer, and the fact belongs in `overall/operator_catalog.md`
   or in the source. If the knowledge is language-specific, it belongs in that language's
   `skills/optimize/<lang>_levers/`.
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
  layer, and `skills/`. **No `operators/` layer** — the per-operator cards were removed (§3b); operator
  entry points live in `overall/operator_catalog.md`.
- `framework/mori/` (added 2026-08-04) is grounded on `ROCm/mori@dc4bc75a`; it has `INDEX.md`, an
  `overall/` layer (repo scope, launch-config tuning control plane), and one operator folder
  (`ep_dispatch_combine`, the only op with real depth so far — mori's much wider surface, MORI-IO/CCL/IR/
  UMBP, is explicitly out of scope until someone reads that source). No `skills/` yet.
- All seven `languages/` folders (`asm`, `ck`, `flydsl`, `fusion`, `gluon`, `hip`, `triton`) plus
  `hardware/` and `common_methodology/` each have an `INDEX.md`.
- Each folder's own `INDEX.md` records its specific pinned source and layout — start there.
