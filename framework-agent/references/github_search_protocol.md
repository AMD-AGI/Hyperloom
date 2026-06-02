# GitHub Search protocol used by framework-agent

> Technical reference for the anonymous GitHub Search backend in
> `framework_agent.sources.github`. Distilled from Arbor's
> `github_search.py` plus the live framework-agent implementation.

## Endpoint

```
GET https://api.github.com/search/issues
```

Standard params used:

| Param | Value |
|---|---|
| `q` | composed query (see below) |
| `sort` | `updated` |
| `order` | `desc` |
| `per_page` | `min(limit, 100)` |

User-Agent: `framework-agent/0.1`.
Accept: `application/vnd.github+json`.

framework-agent always issues **anonymous** requests (no Authorization
header). The 60 req/h IP rate limit therefore applies.

## Query composition

```python
"repo:<owner>/<repo> is:pr is:open (term1 OR term2 OR ...)"
```

`(term1 OR term2 OR ...)` is the OR-fold of either:

1. `extract_keywords(gap_description)` from
   `framework_agent.keywords` - whitelist hits (60+ ROCm / LLM
   terms) plus PascalCase identifiers (`RadixCache`, `KvCache`, ...)
   plus a fallback to the first five 3+ letter words.
2. If keyword extraction returns empty, the curated `PERF_TERMS`
   tuple (`"perf"`, `"performance"`, `"throughput"`, `"rocm"`,
   `"aiter"`, `"flash"`, `"decode"`).

## Repo URL parsing

`_repo_slug(repo_url)` handles the three common forms:

* `https://github.com/<owner>/<repo>` (with or without `.git` suffix)
* `git@github.com:<owner>/<repo>.git`
* Already-bare `<owner>/<repo>`

Any URL that doesn't decode to `owner/repo` raises `ValueError`. The
github backend catches that and returns `[]` (best-effort policy);
the primus_cortex backend re-raises as `PrimusCortexError`.

## Best-effort failure policy

The github backend wraps `urllib.request.urlopen` in a blanket
`try / except Exception` and returns `[]` on **any** error:

| Cause | Result |
|---|---|
| Anonymous rate-limit (HTTP 403) | `[]` |
| Validation error (HTTP 422) | `[]` |
| Non-200 status | `[]` |
| Timeout | `[]` |
| Malformed JSON | `[]` |
| Non-GitHub remote | `[]` (via `_repo_slug` ValueError) |

The rationale is that github is a **secondary** source - primus_cortex
remains the authority. A failed github call must not abort the run.
Operators who want hard-fail behaviour for github must request the
upgrade explicitly and bring an auth token.

## Result mapping

Each `items[i]` is coerced into the shared `GitHubPr` record:

```python
GitHubPr(
    number=int(item["number"]),
    title=str(item.get("title") or ""),
    html_url=str(item.get("html_url") or ""),
)
```

`item.number` must be an integer; non-integer / missing values cause
the item to be skipped silently. The result list is trimmed to
`limit` entries.

## Dispatcher integration

`framework_agent.sources.enumerate_candidates` calls the github
backend exactly once per repo per run, **after** the primus_cortex
backend. Results are unioned and de-duplicated by `ref`; primus wins
ties because it appears first in the iteration order.

Operators set `search_modes` on `ExploreRequest` to pick the
combination:

| `search_modes` | Behaviour |
|---|---|
| `["primus_cortex"]` | github backend never called (saves IP rate quota). |
| `["github"]` | github only (offline / no internal cluster). |
| `["primus_cortex", "github"]` | Both, dedup primus-first (default). |

## Future hardening

Not implemented today; tracked for a follow-up:

* `--github-token-file` CLI flag to lift the 60 req/h cap.
* Concurrency: the current loop is serial; long candidate lists may
  benefit from a small thread pool keyed on repo.
* Stable cursoring for repos with > 100 perf-ish PRs.
