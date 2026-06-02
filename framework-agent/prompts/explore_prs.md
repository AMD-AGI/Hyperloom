# Stage 1 - Candidate discovery (`explore_prs`)

> LLM-facing skill prompt for the "discover candidate PRs" stage of a
> framework-agent run. Use this prompt as the system / task message when
> driving framework-agent from a higher-level specialist (Arbor / TBO /
> Hyperloom).

## Intent

Given a free-form **gap_description** ("improve sglang fp8 MoE on
MI300X", "reduce vLLM CUDA-Graph rebuild latency", ...) and one or
more target repositories, return a ranked list of candidate PRs to
consider for isolated build + benchmark.

## Inputs

| Field | Type | Notes |
|---|---|---|
| `gap_description` | str | Required. Free-form perf problem statement. Keyword extraction relies on this. |
| `repos` | list[str] | Required. One or more GitHub-style repo URLs (https or git@). |
| `primus_cortex_url` | str | Optional. Internal REST URL; if set, the primus_cortex backend is hard-fail. |
| `include_github` | bool | Default true. Anonymous GitHub Search fallback (best-effort, rate-limit-bounded). |
| `limit_per_repo` | int | Default 5. Per-repo cap before dedup. |

## Tool surface

```python
from framework_agent.runtime.tools_api import find_relevant_prs_smart
from framework_agent.keywords import extract_keywords
```

`extract_keywords(gap_description)` is automatically invoked by the
github backend; you typically do not need to call it manually.

## Procedure

1. Validate that at least one of {`primus_cortex_url`, `include_github`}
   is truthy; otherwise return `[]` immediately with reason
   "no source configured".
2. Call `find_relevant_prs_smart(gap_description, repos=...,
   primus_cortex_url=..., include_github=..., limit_per_repo=...)`.
3. The result is a `list[Candidate]` already de-duplicated by
   `(repo_url, ref)`. Items appearing in primus_cortex retain their
   source string `"primus_cortex"`; pure-github items become
   `"github"`. Order preserves first-seen so primus wins ties.
4. Hand the list to Stage 2 (enrich) for PR-typed candidates. Branch /
   tag / commit refs may skip enrichment and proceed directly to
   Stage 3 / 4.

## Output contract

A JSON-serialisable array where each element matches the
`Candidate` dataclass:

```json
{
  "ref": "PR:22918",
  "repo": "https://github.com/sgl-project/sglang.git",
  "source": "primus_cortex",
  "title": "[RL] Support FlashInfer per-token NVFP4 MoE",
  "html_url": "...",
  "head_sha": "",
  "labels": [],
  "author": "",
  "changed_files": [],
  "updated_at": ""
}
```

Fields not yet populated at Stage 1 (head_sha / labels / author /
changed_files / updated_at) are filled in by Stage 2 (`enrich_candidate`).

## Failure modes

| Symptom | Resolution |
|---|---|
| `primus_cortex_url` set but unreachable (DNS, 5xx) | Surface as `PrimusCortexError`; do **not** swap to github silently. Operator must fix the URL or remove it. |
| GitHub anonymous rate-limit (HTTP 403) | `find_relevant_prs_smart` swallows this and returns whatever primus_cortex produced. Note in run log; do not retry inline. |
| Both sources empty | Stage 1 returns `[]`; downstream stages must short-circuit with a "no candidates" verdict. |
| Non-GitHub remote (e.g. GitLab) | `_repo_slug` raises ValueError on URL parse for primus; github backend returns `[]`. Skip the repo with a warning. |

## Style notes for LLM authors

* Do **not** invent PR numbers. If primus / github returned 0 items,
  say so and stop.
* Preserve `title` / `html_url` exactly as returned; never paraphrase.
* When ranking, prefer the order returned by the dispatcher (primus
  first, github second) unless you have an explicit reason to
  re-order. Custom ranking should be documented in the run summary.
