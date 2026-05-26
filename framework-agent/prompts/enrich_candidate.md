# Stage 2 - Candidate enrichment (`enrich_candidate`)

> LLM-facing skill prompt for the "enrich PR metadata" stage. Runs after
> Stage 1 (`explore_prs`) and before Stage 3 (`filter_candidates`).

## Intent

For every `PR:N` candidate returned by Stage 1, hit primus_cortex's
`pr_get` + `pr_files` endpoints to fill in:

* `head_sha`     - 40-char git SHA of the PR head commit
* `title`        - canonical PR title (may differ from search hit)
* `labels`       - tuple of label strings (lower-case match in filter)
* `author`       - author login
* `changed_files`- tuple of file paths (required by include/exclude path filters)
* `updated_at`   - ISO-8601 timestamp (compared by `since` / `until` filters)
* `html_url`     - canonical PR URL

Branch / tag / commit-SHA refs (anything that does **not** start with
`PR:`) are returned **unchanged** - enrichment only applies to PRs.

## Inputs

| Field | Type | Notes |
|---|---|---|
| `candidates` | list[Candidate] | Output of Stage 1. |
| `primus_cortex_url` | str | Required for PR-typed enrichment. Without it the stage is a no-op. |
| `primus_timeout_sec` | float | Default 10.0. |

## Tool surface

```python
from framework_agent.sources.primus_cortex import pr_get, pr_files
# higher-level wrapper (used by the explorer):
from framework_agent.explorer import _enrich_candidate_via_primus
```

## Procedure

1. Skip candidates where `primus_cortex_url` is None.
2. For each remaining PR-typed candidate, call `pr_get(slug, number)` +
   `pr_files(slug, number)`.
3. Coerce the response into the `Candidate` shape using the
   `_extract_*` helpers in `explorer.py` (head_sha / labels / author /
   ...). The helpers tolerate multiple wire shapes (`summary.*`,
   top-level keys, `head.{sha,oid}`, ...) - do not parse the raw
   payload by hand.
4. Persist the enriched `Candidate` records for the filter step.

## Failure modes

| Symptom | Resolution |
|---|---|
| primus `pr_get` returns non-dict | `PrimusCortexError`; hard-fail (the PR has no usable metadata). |
| primus `pr_files` returns non-list / 404 | The explorer best-efforts to an empty `changed_files` tuple; do the same here. The path filter step will skip path constraints for this candidate. |
| Network timeout | Surfaces as `PrimusCortexError`; do not retry inline. Operator should rerun with a larger `primus_timeout_sec`. |

## Output contract

A list of `Candidate` records where PR-typed entries have all
"enrichable" fields populated. Non-PR refs are unchanged.

## Style notes

* Enrichment is **idempotent** - running it twice on the same input
  must produce the same output.
* Never drop a candidate at this stage. Filtering happens in
  Stage 3 (`filter_candidates`) so the run log keeps the full
  candidate ledger.
