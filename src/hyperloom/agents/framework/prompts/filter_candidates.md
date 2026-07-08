# Stage 3 - Candidate filtering (`filter_candidates`)

> LLM-facing skill prompt for the "apply PrFilter" stage. Runs after
> Stage 2 (`enrich_candidate`) and before Stage 4 (`isolate_and_run`).

## Intent

Reduce the enriched candidate list to those that satisfy the operator's
`PrFilter` constraints. Explicit candidate refs (those that came from
`ExploreRequest.candidate_refs`, source `"explicit"`) **bypass** the
filter so the operator can always force-include a ref.

## Inputs

| Field | Type | Notes |
|---|---|---|
| `candidates` | list[Candidate] | Output of Stage 2 (enriched). |
| `pr_filter` | PrFilter | From the ExploreRequest. Empty filter -> pass-through. |

## Filter dimensions (9)

Refer to `references/pr_filter_semantics.md` for the full contract.
Summary:

| Dimension | Field | Semantics |
|---|---|---|
| Required labels | `require_labels` | All labels must be present (case-insensitive). |
| Excluded labels | `exclude_labels` | None of the listed labels may be present. |
| Authors | `authors` | Author login must match (case-insensitive). |
| Include paths | `include_paths` | At least one changed-file must prefix-match. |
| Exclude paths | `exclude_paths` | No changed-file may prefix-match. |
| Since | `since` | `updated_at >= since` (ISO-8601). |
| Until | `until` | `updated_at <= until`. |
| Min changed files | `min_changed_files` | `len(changed_files) >= min`. |
| Max changed files | `max_changed_files` | `len(changed_files) <= max`. |

## Procedure

1. For each non-explicit candidate, call
   `framework_agent.explorer._passes_filter(candidate, pr_filter)`.
2. If the helper returns `(True, "")`, keep the candidate.
3. Otherwise emit a `skipped` record: `{ref, source, reason}` -
   the reason string is human-readable and references the failing
   dimension (e.g. `"missing required label(s): ['perf']"`).
4. The explorer collects all skipped candidates into
   `explore_summary.json -> skipped_candidates`. Replicate this in
   your own run summary so operators can audit rejections.

## Failure modes

| Symptom | Resolution |
|---|---|
| Path filter set but `changed_files` empty | Reject with reason `"no changed_files metadata (primus enrichment likely skipped)"`. Do **not** silently pass. |
| Author filter set but `author` empty | Reject with reason `"author unknown but pr_filter.authors set"`. Same rationale. |
| Both date filters set and `updated_at` empty | Default behaviour: dates compare against `""` which is always `<` any ISO timestamp, so the candidate may pass `since` and fail `until`. Operators wanting a strict reject should add a `require_labels` / explicit filter instead. |

## Output contract

```jsonc
{
  "kept": [Candidate, ...],     // explicit refs always at the top
  "skipped": [
    {"ref": "PR:25112", "source": "primus_cortex", "reason": "..."}
  ]
}
```

## Style notes

* The filter is **declarative** - never edit it dynamically based on
  what the candidate list looks like. Operators set thresholds on
  purpose.
* Explicit candidate refs (`source == "explicit"`) bypass the filter
  by design. Do not "fix" this by re-running the filter on them.
* If `kept` ends up empty, signal "no candidates after filter" to
  Stage 4 instead of relaxing the filter.
