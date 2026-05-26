# framework-agent KB - 4-file schema

> Reference for the knowledge base shape consumed by
> `framework_agent.kb`. Derived from
> `claw-dev/docs-zh/tbo-arbor-framework-kb-design.md`; this file is the
> distilled contract used at runtime.

## Layout

```
${FRAMEWORK_AGENT_KB_DIR}/
├── communication/
│   ├── README.md
│   ├── empirical_kb.md        ← domain-specific
│   ├── model_taxonomy.md      ← cross-domain shared
│   └── shared_pitfalls.md     ← cross-domain shared
├── compiler/                  ← same 4 files
├── framework/                 ← same 4 files
├── fusion/                    ← same 4 files
├── kernel/                    ← same 4 files
├── systems/                   ← same 4 files
├── pr_intelligence/           ← special: {other_domain}_knowledge.md
└── recipes/                   ← optional, free-form
```

`${FRAMEWORK_AGENT_KB_DIR}` is resolved by
`framework_agent.kb._resolve_kb_root()` in this order:

1. `FRAMEWORK_AGENT_KB_DIR` env (set by `scripts/install.sh`);
2. `${FRAMEWORK_AGENT_ROOT}/kb` env fallback;
3. `<repo>/framework-agent/kb` (development default).

## Per-file roles

| File | Scope | Mutability |
|---|---|---|
| `README.md` | Domain entry point; ~20 lines | rare manual edits |
| `empirical_kb.md` | Domain-specific findings: flag tables, quantitative perf deltas, version-specific guidance | **only append target** for `contribute_to_kb` |
| `model_taxonomy.md` | Cross-domain shared (~447 lines fixed); model archetypes | human-reviewed copy-from-Arbor |
| `shared_pitfalls.md` | Cross-domain shared (~487 lines fixed); pitfalls catalogue | human-reviewed copy-from-Arbor |

> `model_taxonomy.md` and `shared_pitfalls.md` are intentionally
> identical across the six standard domains in Arbor. `kb.py` does
> not enforce that; if you wish to mirror Arbor, copy the canonical
> versions once and treat them as read-only mirrors.

## Loading priority

`framework_agent.kb._PRIORITY_FILES = ["empirical_kb.md", "shared_pitfalls.md"]`
controls the order `select_kb()` returns files within a matched domain:

```
empirical_kb.md  ->  shared_pitfalls.md  ->  README.md / model_taxonomy.md / ...
```

LLM specialists thus see the most actionable content first.

## Domain matching keywords

Defined in `kb.DOMAIN_KEYWORDS`. Matching is case-insensitive substring;
if no keyword matches, `select_kb` falls back to a full-text scan over
each domain's priority files for the lower-cased query.

| Domain | Keywords (subset) |
|---|---|
| `kernel` | kernel, gemm, moe, attention, fmoe, ck, triton |
| `communication` | allreduce, nccl, rccl, quickreduce, collective |
| `compiler` | compiler, inductor, codegen |
| `framework` | vllm, sglang, framework, scheduler, cuda_graph |
| `fusion` | fusion, fused, overlap |
| `systems` | system, hip, rocm, driver, launch, dispatch |
| `pr_intelligence` | pr, github, upstream, patch |
| `recipes` | recipe, warm_start, best_config, prior_session |

Extend the dict in `kb.py` when adding a new domain - all readers see
the change automatically.

## Contribution rules

`contribute_to_kb(domain, finding, source, session_id)`:

* Creates `${KB}/<domain>/` if missing;
* Appends only to `empirical_kb.md` (never `model_taxonomy` /
  `shared_pitfalls`);
* Prepends a header line of the form

```
---
**[YYYY-MM-DD HH:MM:SS UTC]** source=`<src>` session=`<sid>`
```

The `fa explore --execute` hook (`_contribute_findings_to_kb` in
`explorer.py`) sets `source = "fa explore --execute"` and uses the
basename of `work_dir` as `session_id`.

## Synthesis modes

`synthesize_findings(domain, findings, *, with_llm=False, model=...)`
distils a list of `Finding` records into a single markdown blob.

| Mode | Behaviour | Dependencies |
|---|---|---|
| `with_llm=False` (default) | Pure-Python: emits H2 header + one `###` block per finding + `## Aggregate metrics` tail for repeating keys. Deterministic. | none |
| `with_llm=True` | Lazy-imports `claude_agent_sdk`; raises `RuntimeError` with install hint when SDK is absent. Drives `sdk.query()` synchronously via `asyncio.run`. | `pip install '.[claude]'` + `$OPENAI_BASE_URL` + `$SAFE_API_KEY` |

The CLI mirrors this contract:

```
fa kb synthesize --domain framework --findings findings.json
fa kb synthesize --domain framework --findings findings.json --with-llm --model claude-opus-4-7
```
