# `framework_optimization` KB partition

Shared knowledge base for the 5th Framework agent role. Read by the
patch proposer (PR-G) as priors; written by the integrate handler
(PR-I) only after a `KEEP` verdict.

## Layout

```
framework_optimization/
├── README.md           # this file
├── seeds/              # ship-with-PR-I curated entries
│   ├── fw-perf-001.md
│   ├── fw-perf-002.md
│   ├── fw-perf-003.md
│   ├── fw-boundary-001.md
│   ├── fw-boundary-002.md
│   ├── fw-boundary-003.md
│   ├── fw-pitfall-001.md
│   └── fw-pitfall-002.md
└── empirical_kb.md     # session-appended KEEP lessons (created lazily)
```

## Entry categories

| Category | Filename pattern | Source | When to use |
|---|---|---|---|
| `perf` | `fw-perf-NNN.md` | curated seeds | known-good optimisation patterns (vllm chunked prefill, sglang radix tree eviction, PagedAttention block size) |
| `boundary` | `fw-boundary-NNN.md` | curated seeds | "this lives in kernel-agent / orchestration, not framework" rules |
| `pitfall` | `fw-pitfall-NNN.md` | curated seeds | "this change tends to crash / regress" markers |
| `lesson` | `empirical_kb.md` `# fw-keep-...` headers | session writes | KEEP outcomes from prior `framework_integrate` runs |

## Entry format

Seed `*.md` files use a strict header for the parser:

```
# <entry_id>  <title>
Framework: <vllm|sglang|empty-for-cross>
Tags: <comma-separated>
... free-form body ...
```

`empirical_kb.md` is appended to by `kb_write` (PR-I) with block markers:

```
# fw-keep-<timestamp>-<hash>  KEEP: <framework> <one-line-summary>
Framework: <vllm|sglang>
Source: <session_id>
Patch: <patch_id>
Gain: <pct>%
... rationale ...
```

## Ranking

Reader (`kb_priors.read_priors`) ranks entries by `(category_rank,
framework_match_rank, entry_id)`. Pitfalls surface first to the LLM
proposer so it avoids known regressions before exploring patterns.
