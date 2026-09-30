---
myst:
    html_meta:
        "description": "Reference for the Hyperloom Experience KB service: the local service every workspace runs, the optional global KB it pushes to and pulls from, configuration, and the HTTP API."
        "keywords": "Hyperloom, Experience KB, experience, knowledge base, global KB, push, pull, schema, HTTP API, configuration"
---

# Experience KB service

The Experience KB stores one immutable Experience per measured Framework
attempt and serves them back to later decisions. It ships inside Hyperloom as
the `hyperloom_kb` package.

- Every workspace runs its own **local** service. Runs read from it and write to
  it; nothing else.
- A **global** KB is the same service deployed once for a team. A workspace
  shares with it only through its local service: push sends the workspace's
  own Experiences, pull fetches the global KB's.

What a run records, and how an attempt becomes an Experience, is in
[Framework Experience publication](../experience-kb-framework.md). The record
itself is in [Experience schema](experience-kb-schema.md), and writing
Experiences from other logs in [Experience collection](experience-kb-collect.md).

## Configuration

| Variable | Set by | Meaning |
|---|---|---|
| `HYPERLOOM_KB_URL` | `hyperloom-setup` | The workspace's local service, `http://127.0.0.1:8787` by default. A loopback URL is a service the workspace starts itself; any other URL is used as is. Unset disables Experience reads and writes. |
| `HYPERLOOM_KB_TOKEN` | `hyperloom-setup` | The local service's generated access token. |
| `HYPERLOOM_GLOBAL_KB_URL` | the user | The global KB to push to and pull from. Optional. |
| `HYPERLOOM_GLOBAL_KB_TOKEN` | the user | The global KB's access token. Required with `HYPERLOOM_GLOBAL_KB_URL`. |
| `HYPERLOOM_KB_AUTO_PUSH` | the user | `1` pushes after every run's Experiences are written locally. Default off. |
| `USER_DATA_PATH` | `hyperloom-setup` | The local service keeps its data under `$USER_DATA_PATH/experience-kb`. |
| `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_MODEL` | `hyperloom-setup` | The gateway and model the service plans reads with; `LOCAL_KB_PLANNER_MODEL` overrides the model. Without a gateway the service still accepts writes and reads report `unavailable`. |

All of these live in the workspace `.env`. The service reads them when it
starts; see [the local service](#the-local-service) for how a change reaches a
running service.

## Workspace commands

Run from the workspace, in the environment the optimizer runs in (inside the
container in docker mode), with `.env` loaded:

```bash
python -m hyperloom.inference_optimizer.experience_kb_service init-env  # write the local URL and a generated token into .env
python -m hyperloom.inference_optimizer.experience_kb_service ensure    # start the local service unless it already serves
python -m hyperloom.inference_optimizer.experience_kb_service push      # send this workspace's new Experiences to the global KB
python -m hyperloom.inference_optimizer.experience_kb_service pull      # store the global KB's Experiences of this workspace's schemas
```

`init-env` only fills a key that is missing or `<PLEASE_FILL_IN>`, and prints
whether each key was written or kept, never the token. `push` and `pull` print
one line with the global URL and the `created`, `unchanged`, `skipped`, and
`rejected` counts, and exit 1 when they stopped early or rejected an Experience.

## The local service

`ensure` runs during `hyperloom-setup` (baremetal) and at every optimize
launch, so the service runs wherever the optimizer runs and outlives each run.
It looks at what answers on `HYPERLOOM_KB_URL`:

- nothing: it starts `python -m hyperloom_kb` over `$USER_DATA_PATH/experience-kb`
  and waits for it to serve;
- a service that answers with this workspace's token, the packaged
  declaration, and the settings in `.env`: it reuses it;
- such a service started with other settings, or serving another default
  declaration: it restarts it, keeping its data. Settings are compared by their
  effect, so the same key under another variable name does not restart it;
- anything else on the port, such as another user's service: it refuses, and
  the user picks another port in `HYPERLOOM_KB_URL`.

An optimize launch whose service cannot start logs a warning and continues:
reads return nothing, and writes wait in the spool until a later launch finds
the service serving. Requests to a loopback service never go through an
`HTTP_PROXY` or `HTTPS_PROXY` from the environment.

`$USER_DATA_PATH/experience-kb` holds:

| Path | Content |
|---|---|
| `canonical/` | The immutable schemas and Experiences; the source of truth. |
| `kb.sqlite3` | Write order for listing and export; rebuilt from `canonical/` when lost. |
| `sync.sqlite3` | Push and pull progress per global KB. Losing it makes the next push and pull resend everything, which the idempotent writes absorb. |
| `spool/` | Writes the service has not accepted yet. |
| `service.log` | The service's log. |

## Schemas

A service holds any number of schemas. Every write carries its declaration, and
the service registers a schema the first time it sees one, so:

- a workspace whose declaration changes keeps the Experiences of the old one;
- a global KB keeps every contributor's schema.

A read searches exactly one schema: the run's own, which is the declaration its
packaged mapping produces. Listing and export cover every schema unless a
`schema_ref` narrows them.

## Global sync

**Push** sends each Experience written to the local service that was not pushed
to this global KB yet, with its declaration. It never sends an Experience that
was pulled from a global KB, and it stops at the first failure the global KB may
recover from, keeping its place so the next push resumes there. An Experience
the global KB rejects for good is reported under `rejected` and skipped. A
workspace `push` first delivers the spool, so writes made while the local
service was down are pushed too.

**Pull** fetches, for each schema the local service holds, the global KB's
Experiences of that schema it has not fetched before. Other schemas stay on the
global KB. Pulled Experiences are readable immediately.

Both run in bounded batches; the client repeats them until nothing is left.
With `HYPERLOOM_KB_AUTO_PUSH=1`, the end of every run pushes; a failed automatic
push is logged as a warning and never fails the run.

## Deploying a global KB

The `hyperloom-global-kb` skill deploys one from a Hyperloom install and tells
each workspace which keys to add. By hand, on the host:

```bash
export HYPERLOOM_KB_TOKEN=...   # generate once and keep it
python -m hyperloom_kb --host 0.0.0.0 --port 8787 --home /srv/hyperloom-global-kb
```

| Flag | Default | Meaning |
|---|---|---|
| `--host`, `--port` | `127.0.0.1`, `8787` | Bind address. |
| `--home` | `~/.local/share/hyperloom-kb` | State directory, laid out as above. |
| `--declaration` | packaged `inference-recipe-v1` | The schema a read searches when it names none. |
| `--seed-jsonl FILE` | none | Import complete Experiences (`{"experience": ...}` or bare rows) before serving; repeatable and idempotent. |

A global KB needs no LLM gateway. It speaks plain HTTP; across an untrusted
network, put a TLS-terminating proxy in front of it and hand out its
`https://` URL.

## HTTP API

| Endpoint | Purpose | `RemoteClient` |
|---|---|---|
| `PUT /v1/experiences/{id}` | write one complete Experience | `write()`, `publish()` |
| `POST /v1/read` | rank one schema's Experiences for a decision and render them | `read()` |
| `GET /v1/list` | page through Experience summaries in write order | `list_experiences()` |
| `GET /v1/export` | page through complete Experiences in write order | `export_page()` |
| `POST /v1/push` | push one batch to this service's global KB | `push()` |
| `POST /v1/pull` | pull one batch from this service's global KB | `pull()` |
| `GET /health` | liveness, schemas, corpus size, process, settings digest | `health()` |

Every request, including `/health`, sends `Authorization: Bearer <token>`.
Unknown request fields are rejected, so a misspelled field fails loudly. A
request body may be up to 256 MiB; Experiences themselves have no size limit.

| Status | Body `error` | Meaning | Retry? |
|---|---|---|---|
| 400 | `invalid_request` | malformed body, unknown field, invalid Experience, unregistered schema, out-of-range parameter | no |
| 401 | `unauthorized` | missing or wrong token | after fixing the token |
| 404 | `not_found` | unknown path | no |
| 409 | `conflict` | the id already exists with different content | no |
| 409 | `sync_unavailable` | push or pull on a service started without a global KB | after configuring one |
| 500 | `internal_error` | storage or service failure | yes |

400, 409, and 500 bodies include a `detail` string.

### `PUT /v1/experiences/{id}`

Body: `{"experience": <complete Experience>, "declaration": <declaration>}`. The
path id must equal `experience.id` and `status` must be `complete`.
`declaration` is optional once the service holds the Experience's schema; when
present it must derive the Experience's `schema_ref`, and the service registers
it. The Experience is validated against its declaration before storage.

Response: `{"status": "created" | "unchanged", "experience_id": "exp-...", "content_hash": "..."}`.
Experiences are immutable: the same content again is `unchanged`, different
content under an existing id is 409.

### `POST /v1/read`

| Field | Required | Default | Meaning |
|---|---|---|---|
| `decision` | yes | | the decision the caller is about to make |
| `context` | no | `{}` | workload context (`identity`, `workload`, `objective`, `observations`, ...) |
| `schema_ref` | no | the service's `--declaration` | the one schema to search |
| `outcome` | no | `mixed` | `keep`, `revert`, another declared decision, or `mixed` |
| `limit` | no | `10` | maximum Experiences to return, 1–100 |
| `content_inline_limit` | no | none (all inline) | bytes above which a `change.content` is rendered as a reference instead of inline |

The service's planner turns `decision` and `context` into weighted query
signals, then ranks candidates by exact field matches plus lexical fuzzy
matches. Experiences that repeat the same change under the same identity form
one Repeat Group, which contributes at most one Experience; `outcome` filters by
decision without hiding history, because the group annotations still count
every member.

```json
{
  "read_id": "read-06b67219b0b8426593afc9914838a09c",
  "status": "completed",
  "outcome": "mixed",
  "limit": 10,
  "prompt_block": "=== Relevant Experience KB ===\n...",
  "rendered_refs": [{"id": "exp-a23def11a8b85add9f8cc4dfef6c8c7f", "purpose": "representative"}],
  "experiences": [{"experience_id": "exp-a23def11a8b85add9f8cc4dfef6c8c7f", "score": 1.59, "why_matched": ["..."]}],
  "eligible_count": 1,
  "rendered_count": 1,
  "warnings": ["capability_unavailable:semantic"]
}
```

`status` is `completed`, `unavailable` (no planner gateway), or `failed`
(planning or retrieval failed; `warnings` holds the reason); the HTTP status is
200 either way. `prompt_block` holds each rendered Experience's complete record
and Repeat Group annotations, never condensed, under an
`Experience <id>` heading. A producer that acts on a read records the returned
`rendered_refs` in the resulting Experience.

With `content_inline_limit`, a longer `change.content` appears in the record as
`<external content sha256:<hex>, <n> bytes>`, and `contents` carries its text:
`[{"ref": "sha256:<hex>", "bytes": <n>, "content": "..."}]`. Hyperloom asks for
2048 bytes, writes each one under `<session>/experience_kb/contents/<hex>.txt`
(and each patch of a source change as its own file beside it), and lists those
paths at the end of the injected block, so an agent reads a large patch only
when it needs it.

Each item in `experiences`, and in `/v1/list`, is a summary:
`experience_id`, `source_run_id`, `change_summary`, `decision`,
`baseline_value`, and `outcome_value`; reads add `score` and `why_matched`.

### `GET /v1/list` and `GET /v1/export`

Query: `after` (the previous page's `next_cursor`, default 0), `limit` (list
1–500, export 1–100), and optional `schema_ref`. Repeat while `has_more` is true.

```json
{"items": [{"sequence": 1, "experience_id": "exp-...", "...": "..."}], "next_cursor": 1, "has_more": false}
```

`/v1/list` items are summaries plus `sequence`; `/v1/export` items are
`{"sequence": 1, "experience": <complete Experience>}`.

### `POST /v1/push` and `POST /v1/pull`

Body: `{}`. The service contacts the global KB it was started with.

```json
{"status": "completed", "global_url": "https://global-kb.example", "created": 2, "unchanged": 0, "skipped": 1, "rejected": [], "has_more": false}
```

`skipped` counts pulled Experiences a push does not send back. `status` is
`incomplete`, with an `error`, when the global KB stopped answering; the batch
up to that point is kept.

### `GET /health`

```json
{"status": "ok", "schema_ref": "schema:sha256:...", "experience_count": 3, "schemas": {"schema:sha256:...": 3}, "pid": 4242, "config_digest": "..."}
```

`schema_ref` is the default read schema; `config_digest` fingerprints the
settings the service started with.

## Client failure behavior

- `read()` never raises for service failures; it returns `status="unavailable"`
  with an empty `prompt_block` and the reason in `warnings`.
- `publish()` raises `RemoteClientError` for permanent rejections (400, 409).
  Any other failure spools the write, with its declaration, and returns
  `status="spooled"`; Hyperloom's spool is `$USER_DATA_PATH/experience-kb/spool`.
- `flush_spool()` replays spooled writes and stops at the first retryable
  failure; files the service rejects for good move to `spool/rejected/`.
  Hyperloom flushes at every launch and before every workspace `push`.
- `write()` raises on any failure and never spools; push uses it.
