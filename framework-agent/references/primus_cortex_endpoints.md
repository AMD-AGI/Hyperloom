# Primus Cortex PR Monitor - endpoints used by framework-agent

> Technical reference for the four primus-cortex-pr-monitor REST
> endpoints that `framework_agent.sources.primus_cortex` talks to.
> Distilled from the upstream `primus-cortex-pr-monitor-access.md`
> service doc + the live `framework_agent.sources.primus_cortex`
> implementation; this file is the authoritative contract from
> framework-agent's point of view.

## Base URL

```
http://primus-cortex-pr-api.primus-cortex.svc.cluster.local
```

The base URL is wired into `ExploreRequest.primus_cortex.base_url`
or the `PRIMUS_CORTEX_PR_API` env var. The framework-agent client
trims a trailing slash before composing paths.

`GET /v1/healthz` returns `{"status": "ok"}` and is the recommended
liveness probe before kicking off a run.

## Endpoints

### `GET /v1/repos/{owner}/{repo}/prs`

Query params:

| Param | Type | Notes |
|---|---|---|
| `state` | `"open"` / `"closed"` / `"all"` | Default `"open"` in framework-agent. |
| `limit` | int | Capped by `ExploreRequest.max_search_candidates`. |
| `label` | str | Optional; passes through to primus's label filter. |

Response (200): either a flat list or a dict wrapping a list under one
of `items` / `prs` / `data` / `results`. framework-agent tolerates
both shapes via `_extract_pr_list`.

Each item is coerced to a `GitHubPr(number, title, html_url)` record.
`html_url` may be served from `url` instead.

Errors: HTTP 4xx/5xx, transport, or non-JSON body all surface as
`PrimusCortexError` (hard-fail).

### `GET /v1/repos/{owner}/{repo}/prs/{number}`

Returns the PR detail object. Two wire shapes are supported:

* primus_cortex canonical: `{"summary": {...}, "body": "...",
  "files": [...]}`. `summary.head_sha` / `summary.author_login` /
  `summary.labels` carry the metadata framework-agent enriches with.
* GitHub-like flat: `{"number": ..., "head": {"sha": ...}, ...}`. The
  `_extract_*` helpers in `explorer.py` fall through both shapes.

Hard-fail if the body is not a JSON object.

### `GET /v1/repos/{owner}/{repo}/prs/{number}/files`

Returns the file list. Wire shape: either a flat JSON array, or a
dict wrapping a list under `files` / `items` / `data`. Each item
should expose at least `file_path` (preferred) or `filename` /
`path`. Additional fields (`additions`, `deletions`, `status`,
`blob_sha`, ...) are passed through to `pr_files.json` verbatim
without interpretation.

Hard-fail on unrecognised wire shape.

### `GET /v1/repos/{owner}/{repo}/prs/{number}/patches`

Returns a **JSON array** of per-file patch objects (despite the
endpoint name's "patches" implying raw text):

```jsonc
[
  {
    "file": {
      "file_path": "python/x/foo.py",
      "previous_path": null,
      "status": "modified" | "added" | "deleted" | "renamed",
      ...
    },
    "patch": "@@ -1,3 +1,5 @@\n-a\n+b\n c",
    "patch_truncated": false
  }
]
```

framework-agent renders this into a unified-diff stream that
`git apply` can consume. The synthesized header for each entry is:

```
diff --git a/<old_path> b/<new_path>
--- <a/old_path | /dev/null>
+++ <b/new_path | /dev/null>
<patch body>
```

`status="added"` -> old side is `/dev/null`; `status="deleted"` /
`"removed"` -> new side is `/dev/null`. Renamed files use
`previous_path` for the old side. Items missing the `patch` field
(e.g. binary diffs) emit only the diff header and skip the hunk
body.

Wire-shape fallbacks: a dict wrapping the list under `patches` /
`items` / `data` is tolerated; a bare string body is returned
verbatim (legacy server).

Hard-fail on unrecognised wire shape.

## Error semantics

The `PrimusCortexError` class wraps all failure cases:

| Cause | Message shape |
|---|---|
| HTTP 4xx/5xx | `primus_cortex HTTP <code> at <url>: <body[:512]>` |
| URL unreachable (DNS / no route) | `primus_cortex unreachable at <url>: <reason>` |
| Transport timeout / OSError | `primus_cortex transport error at <url>: <exc>` |
| Non-JSON body | `primus_cortex returned non-JSON at <url>: ...` |
| Malformed JSON shape | `primus_cortex response at <url> ...` |

Callers should let `PrimusCortexError` propagate; the CLI's outer
`except Exception` translates it into exit code 2 with the original
message preserved on stderr. **Do not catch-and-fallback silently**;
the operator must see the original error.

## Notes

* HTTP requests are made via stdlib `urllib.request` (no `requests`
  dependency). The User-Agent is `framework-agent-primus-cortex/0.1`.
* All requests are GET; framework-agent never POSTs to primus.
* The service is an AMD-internal cluster service; offline / external
  CI must omit `primus_cortex` from `search_modes` and rely on the
  github backend.
