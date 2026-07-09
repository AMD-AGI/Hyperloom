# `PrFilter` semantics

> Technical reference for the 9-dimension filter applied between
> Stage 2 (`enrich_candidate`) and Stage 4 (`isolate_and_run`).
> Distilled from `framework_agent.explorer._passes_filter` and
> `framework_agent.models.PrFilter`; this file is the authoritative
> contract.

## Wire shape

```jsonc
"pr_filter": {
  "require_labels":     ["perf", "rocm"],
  "exclude_labels":     ["wip", "draft"],
  "authors":            ["alice", "bob"],
  "include_paths":      ["python/sglang/srt/layers/"],
  "exclude_paths":      ["test/", "benchmark/"],
  "since":              "2025-01-01",
  "until":              "2026-06-01",
  "min_changed_files":  1,
  "max_changed_files":  20
}
```

All fields optional. An empty / missing block is equivalent to
`PrFilter.is_empty == True`, in which case the filter is a no-op.

String fields accept either a single value or a list:

```jsonc
"include_paths": "python/sglang/"        // OK, coerced to ("python/sglang/",)
"include_paths": ["python/", "csrc/"]    // OK, kept as tuple
```

## Application order

`_passes_filter(candidate, filter)` short-circuits on the **first**
failing dimension:

1. **Empty filter** -> `(True, "")`.
2. **Required labels** - all labels (case-insensitive) must be in
   `candidate.labels`. Reason on miss:
   `missing required label(s): ['<missing1>', ...]`.
3. **Excluded labels** - none of the listed labels may appear.
   Reason: `has excluded label(s): ['<found1>', ...]`.
4. **Authors** -
   * empty `candidate.author` -> reject
     `"author unknown but pr_filter.authors set"`;
   * otherwise must match a listed author case-insensitive
     (`"author <X> not in pr_filter.authors"`).
5. **Since / Until** - lexicographic compare on `candidate.updated_at`
   (ISO-8601 strings sort correctly). Reasons:
   `updated_at <X> < since <Y>` or `> until`.
6. **Path constraints precheck** - if **any** of include_paths /
   exclude_paths / min_changed_files / max_changed_files is set
   **and** `candidate.changed_files` is empty, reject with
   `"no changed_files metadata (primus enrichment likely skipped)"`.
   This guards against false-positives when Stage 2 enrichment was
   skipped (typically because primus_cortex is unconfigured).
7. **Exclude paths** - any prefix match against a changed file
   rejects the candidate.
8. **Include paths** - at least one changed file must prefix-match
   one of the include entries.
9. **Min / Max changed files** - inclusive bounds on
   `len(candidate.changed_files)`.

## Explicit-ref bypass

Candidates with `source == "explicit"` (those coming from
`ExploreRequest.candidate_refs`) **skip the filter entirely** -
operator intent wins. The `_enumerate_with_skipped` pipeline in
`explorer.py` enforces this; do not re-apply the filter to explicit
refs after the fact.

## Skipped record

For every non-explicit candidate that the filter rejects, the
explorer appends to `explore_summary.json -> skipped_candidates`:

```jsonc
{
  "ref": "PR:25112",
  "source": "primus_cortex",
  "reason": "missing required label(s): ['perf']"
}
```

The reason string is human-readable and references the failing
dimension. Tests assert on the dimension keyword (e.g. "required
label") rather than the full string so the wording can evolve.

## Interaction with Stage 2 enrichment

The filter relies on Stage 2 having populated `labels`, `author`,
`changed_files`, and `updated_at`. When primus_cortex is **not**
configured, those fields remain empty and the filter behaves as
follows:

| Dimension | Behaviour without enrichment |
|---|---|
| `require_labels` | always rejects (empty labels missing required ones) |
| `exclude_labels` | always passes (no labels to exclude) |
| `authors` | always rejects (author unknown) |
| `since` / `until` | empty `updated_at` is `""`, lex-compares as smaller than any timestamp; mainly affects `since` |
| `include_paths` / `exclude_paths` / `min_*` / `max_*` | rejected via the "no changed_files metadata" precheck |

Operators wanting hard-filtering must configure primus_cortex.
Operators wanting "best-effort over github" must leave the filter
empty or restrict it to label-only constraints they accept
rejecting when enrichment is absent.

## Style notes

* The filter is **declarative**. Code that mutates it at runtime
  based on the candidate list is an anti-pattern - it breaks the
  audit trail in `skipped_candidates`.
* For "all candidates" mode, leave `pr_filter` out of the request
  (`PrFilter.is_empty -> True`).
* For "label-gated only" mode, set `require_labels` /
  `exclude_labels` only; keep path / author / date / count fields
  unset so the path-metadata precheck does not fire.
