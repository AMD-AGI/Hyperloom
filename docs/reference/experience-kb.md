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
| `HYPERLOOM_KB_URL` | `hyperloom-setup` | The workspace's local service: `http://127.0.0.1:<port>`, with a port between 20000 and 29999 derived from the workspace path, so workspaces on one host do not share one. A loopback URL is a service the workspace starts itself; any other URL is used as is. Unset disables Experience reads and writes. |
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
python -m hyperloom.inference_optimizer.experience_kb_service pull      # bring this workspace's schema to everything the global KB holds of it
```

`init-env` only fills a key that is missing or `<PLEASE_FILL_IN>`, and prints
whether each key was written or kept, never the token. `push` and `pull` print
one line with the global URL and the `created`, `unchanged`, `skipped`, and
`rejected` counts, plus `held_back` for a push and, for a pull, the label it
saved the state under, and exit 1 when they stopped early, were refused, or
rejected an Experience.

The same entry point runs every other [`hyperloom-kb`](#the-hyperloom-kb-command)
command on the workspace's service (`health`, `labels`, `label`, `restore`,
`exclude`, `include`, `exclusions`, `list`, `export`), starting it when
nothing serves and defaulting to the schema the workspace's runs write. Its
own `push` first delivers the workspace's spool.

## The `hyperloom-kb` command

`hyperloom-kb`, or `python -m hyperloom_kb.cli`, operates whatever service
`HYPERLOOM_KB_URL` names, authenticated with `HYPERLOOM_KB_TOKEN`; a local and
a global service take the same commands. Each prints its JSON result.

| Command | Request |
|---|---|
| `health` | `GET /health` |
| `push` | `POST /v1/push`, repeated until done |
| `pull --schema REF` | `POST /v1/pull`, repeated until done |
| `labels [--schema REF]` | `GET /v1/labels` |
| `label [--schema REF] [--name NAME]` | `POST /v1/labels` |
| `restore LABEL_ID` | `POST /v1/restore` |
| `exclude EXPERIENCE_ID --reason TEXT` | `POST /v1/exclusions` |
| `include EXPERIENCE_ID` | `DELETE /v1/exclusions/{experience_id}` |
| `exclusions [--schema REF]` | `GET /v1/exclusions` |
| `list`, `export` `[--schema REF] [--include-excluded] [--after N] [--limit N]` | one page of `GET /v1/list`, `GET /v1/export` |

It exits 0 when the command worked, 1 when the service refused the request or
a push or pull stopped early, was refused, or rejected an Experience, and 2
when the URL or token is not configured.

A tool embeds the same commands in its own CLI: `add_commands(subparsers,
schema_ref=...)` registers each one whose name the tool has not registered
itself, defaulting every command that names no schema to `schema_ref`, and
`run_command(client, args)` runs one, with the same output and exit status.
Hyperloom's workspace commands are built this way, with their own `push` and
`pull`. The `hyperloom-kb` skill, installed beside `hyperloom-setup`,
describes each command for an agent.

## The local service

`ensure` runs during `hyperloom-setup` (baremetal) and at every optimize
launch, so the service runs wherever the optimizer runs and outlives each run.
It looks at what answers on `HYPERLOOM_KB_URL`:

- nothing: it starts `python -m hyperloom_kb` over `$USER_DATA_PATH/experience-kb`
  and waits for it to serve;
- a service that answers with this workspace's token and data home, the
  packaged declaration, and the settings in `.env`: it reuses it;
- such a service started with other settings, or serving another default
  declaration: it restarts it, keeping its data. Settings are compared by their
  effect, so the same key under another variable name does not restart it;
- a service holding another workspace's data, as a copied `.env` would point
  at, or anything else on the port, such as another user's service: it refuses
  without stopping it, and the user picks another port in `HYPERLOOM_KB_URL`.

A data home has one service. A service holds `service.lock` in its home while
it serves, and one started on a home another service holds exits, naming that
service's pid and port; `ensure` reports that line. Two workspaces therefore
need their own `USER_DATA_PATH`, or the second gets no service of its own.

Only an optimize launch and `ensure` restart a service. `push` and `pull`,
including the automatic push at the end of a run, use the service as it runs
and warn when its settings differ from theirs, so a push from a shell never
stops the service a running session reads from.

No Experience KB problem stops a run. An optimize launch whose service cannot
start, whose `.env` names the service without its token, or whose service
validates another schema logs a warning and continues: reads return nothing,
and writes wait in the spool until a later launch finds the service serving.
A read the service cannot answer, even one cut off mid-response, leaves the
prompt as it would be without the Experience KB.

A run waits on its local service at most 30 seconds per read, write, or health
check, and the service gives its planner 20 seconds
(`LOCAL_KB_PLANNER_TIMEOUT_SECONDS`); a read measured 4–6 seconds. After three
reads in a row that do not complete, the session stops reading, so a hung
service or gateway costs a run a few timeouts, not one per orchestration turn
and specialist dispatch. The session breakdown writes its Experiences off the
coordinator's event loop, and once one write spools, the rest of that
collection spools without waiting on the service. Requests to a loopback service never go through an
`HTTP_PROXY` or `HTTPS_PROXY` from the environment.

`$USER_DATA_PATH/experience-kb` holds:

| Path | Content |
|---|---|
| `canonical/` | The immutable schemas and Experiences; the source of truth. |
| `kb.sqlite3` | Write order for listing and export; rebuilt from `canonical/` when lost. |
| `sync.sqlite3` | Push and pull progress per global KB. Losing it makes the next push and pull resend everything, which the idempotent writes absorb. |
| `state.sqlite3` | Labels, exclusions and their history, and what a restore set outside the current state. Without it every stored Experience is in the state and none is excluded. |
| `identity.json` | The home's `kb_id`, made when it is first served. It moves with the home; a new home is a new KB. |
| `spool/` | Writes the service has not accepted yet. |
| `service.log` | The service's log. |
| `service.lock` | Held by the one service serving this home; names its pid and port. |

## Schemas

A service holds any number of schemas. Every write carries its declaration, and
the service registers a schema the first time it sees one, so:

- a workspace whose declaration changes keeps the Experiences of the old one;
- a global KB keeps every contributor's schema.

A read searches exactly one schema: the run's own, which is the declaration its
packaged mapping produces. Listing and export cover every schema unless a
`schema_ref` narrows them.

## Labels, restore, and exclusions

Each schema on a service has one current state: the stored Experiences in it,
written here or pulled, minus the ones excluded. Reads see exactly that state.
Writing an Experience adds it to the state; nothing is ever deleted.

- **Exclude** an Experience, with a reason, to hide it from reads; **include**
  lifts the exclusion. Either kind of Experience can be excluded, and every
  exclude and include is kept in the schema's exclusion history.
- **Label** the current state to keep it. A label is identified by its
  `label_id`; its name is only for people and need not be unique.
- **Restore** a label to make its state current again: the Experiences it held,
  with the exclusions it held. Experiences written after the label leave the
  state, and later writes add to the restored one. When the current state is
  not what its label saved, the restore first labels it (reason
  `before_restore`), so nothing restored over is lost: restore that label to
  get it back.

A state equals its label again once its changes are undone, such as an
exclusion lifted. With no label yet, any non-empty state counts as unlabelled.

Push sends only Experiences written here that reads see; one written here but
excluded, or outside the state, is held back and sent by the first push after
reads see it. A push cannot take back what it already sent. List and export
name only Experiences written here, and of those only what reads see unless
`include_excluded` asks for the rest.

## Global sync

**Push** sends each Experience written to the local service that was not pushed
to this global KB yet, with its declaration. It never sends an Experience that
was pulled from a global KB, holds back one reads do not see, and it stops at the first failure the global KB may
recover from, keeping its place so the next push resumes there. An Experience
the global KB rejects for good (400, 409, 413, 414, 415 or 422, including a 413
from a proxy in front of it) is reported under `rejected` and skipped. A
workspace `push` first delivers the spool, so writes made while the local
service was down are pushed too.

**Pull** names one schema and brings its state to everything the global KB
holds of it: the Experiences not fetched before, and every one of that schema
known to be on the global KB (pulled earlier or pushed from here) that a
restore set outside the state. Exclusions stand. The global KB's declaration of
the schema is registered when the local service lacks it; other schemas stay as
they are. When the current state is not what its label saved, the pull first
labels it (reason `before_pull`) and reports that label as `saved`, so
restoring it undoes the pull. A workspace `pull` names the schema its packaged
mapping writes.

**Identity.** Every service has a `kb_id`, made when its home is first served
and kept with the home. The first push or pull to a global KB records its
`kb_id`; a later sync where that URL answers with another `kb_id`, such as a
redeployed global KB, is refused, as is a pull from a global KB that holds
less of the schema than this service already pulled, such as one restored from
an older backup, and any sync with a service that reports no `kb_id`, which
predates this. A refused sync reports `status: refused` with the reason and
changes nothing.

Both run in bounded batches; the client repeats them until nothing is left, and
a pull labels its state once, before its first batch.
With `HYPERLOOM_KB_AUTO_PUSH=1`, the end of every run pushes; a failed automatic
push is logged as a warning and never fails the run. A switch value that is not
a boolean, or auto push without a global KB, is a warning at launch, and that
run does not push.

## Deploying a global KB

The `hyperloom-global-kb` skill deploys one from a Hyperloom install and tells
each workspace which keys to add. By hand, on the host:

```bash
export HYPERLOOM_KB_TOKEN=...   # generate once and keep it
python -m hyperloom_kb --name team-hub --host 0.0.0.0 --port 8787 --home /srv/hyperloom-global-kb
```

A global KB is the same service as a local one, with the same labels,
restores, and exclusions; its state decides what its export, and so every pull,
brings.

| Flag | Default | Meaning |
|---|---|---|
| `--name` | empty | A display name; the service's identity stays its `kb_id`. |
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
| `POST /v1/pull` | pull one batch of one schema from this service's global KB | `pull()` |
| `GET /v1/labels` | a schema's labels, current label, and whether its state changed since | `labels()` |
| `POST /v1/labels` | label a schema's current state | `create_label()` |
| `DELETE /v1/labels/{label_id}` | delete a label | `delete_label()` |
| `POST /v1/restore` | make a label's state current | `restore()` |
| `GET /v1/exclusions` | a schema's exclusions and their history | `exclusions()` |
| `POST /v1/exclusions` | exclude an Experience from reads | `exclude()` |
| `DELETE /v1/exclusions/{experience_id}` | lift an exclusion | `include()` |
| `GET /health` | identity, liveness, schemas, corpus size, process, settings digest | `health()` |

Every request, including `/health`, sends `Authorization: Bearer <token>`.
Unknown request fields are rejected, so a misspelled field fails loudly. A
request body may be up to 256 MiB; Experiences themselves have no size limit.

| Status | Body `error` | Meaning | Retry? |
|---|---|---|---|
| 400 | `invalid_request` | malformed body, unknown field, invalid Experience, unregistered schema, out-of-range parameter | no |
| 401 | `unauthorized` | missing or wrong token | after fixing the token |
| 404 | `not_found` | unknown path, label, or Experience | no |
| 409 | `conflict` | the id already exists with different content | no |
| 409 | `sync_unavailable` | push or pull on a service started without a global KB | after configuring one |
| 500 | `internal_error` | storage or service failure | yes |

400, 404, 409, and 500 bodies include a `detail` string.

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
| `content_inline_limit` | no | none (all inline) | bytes above which a free-text field (`reasoning`, `reflection`, `change.summary`, `change.content`, an alternative) is rendered as a reference instead of inline |
| `render_budget_chars` | no | none (every record) | characters the rendered records may fill; records are rendered whole, in rank order, while they fit |

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
200 either way. `prompt_block` holds each rendered Experience's knowledge
fields and Repeat Group annotations, never condensed, under an
`Experience <id>` heading; its metadata, such as `provenance` and
`rendered_refs`, stays in the record and out of the prompt (see
[Experience record](experience-kb-schema.md#knowledge-and-metadata)). A
producer that acts on a read records the returned `rendered_refs` in the
resulting Experience.

With `content_inline_limit`, a longer free-text field appears in the record as
`<external content sha256:<hex>, <n> bytes>`, and `contents` carries its text:
`[{"ref": "sha256:<hex>", "bytes": <n>, "content": "..."}]`. Hyperloom asks for
2048 bytes, writes each one under `<session>/experience_kb/contents/<hex>.txt`
(and each patch of a source change as its own file beside it), and lists those
paths at the end of the injected block, so an agent reads a large patch only
when it needs it.

With `render_budget_chars`, a record is never cut: the first one that does not
fit, and every one ranked after it, is left out, `rendered_refs` and `contents`
name only the records rendered, and `warnings` carries `render_budget_reached`.
Hyperloom asks for 40,000 characters, so the block an orchestration turn or a
specialist prompt carries stays bounded whatever the records hold.

Each item in `experiences`, and in `/v1/list`, is a summary:
`experience_id`, `source_run_id`, `change_summary`, `decision`,
`baseline_value`, and `outcome_value`; reads add `score` and `why_matched`.

### `GET /v1/list` and `GET /v1/export`

Query: `after` (the previous page's `next_cursor`, default 0), `limit` (list
1–500, export 1–100), optional `schema_ref`, and `include_excluded`
(`true` or `false`, default `false`). Repeat while `has_more` is true. Both name
only Experiences written to this service; `include_excluded=true` adds the ones
reads do not see, so a page may hold fewer items than `limit`.

```json
{"items": [{"sequence": 1, "experience_id": "exp-...", "...": "..."}], "next_cursor": 1, "has_more": false, "head": 1}
```

`/v1/list` items are summaries plus `sequence`; `/v1/export` items are
`{"sequence": 1, "experience": <complete Experience>}`. An export page also
carries `head`, the last write position of what it pages, and, with a
`schema_ref`, that schema's `declaration`.

### `POST /v1/push` and `POST /v1/pull`

Body: `{}` for a push, `{"schema_ref": "schema:sha256:..."}` for a pull. The
service contacts the global KB it was started with.

```json
{"status": "completed", "global_url": "https://global-kb.example", "created": 2, "unchanged": 0, "skipped": 1, "held_back": 0, "rejected": [], "has_more": false}
```

`skipped` counts pulled Experiences a push does not send back; `held_back`
counts Experiences written here that this push left for when reads see them. A
pull also answers its `schema_ref` and `saved`: the label the state before the
pull was saved under, or `null` when its label already held it. `status` is
`incomplete`, with an `error`, when the global KB stopped answering; the batch
up to that point is kept. It is `refused`, with the reason, when the global KB
is another one than this service synced with, holds less than it pulled, or
reports no identity.

### Labels and exclusions

`schema_ref` is optional on every one of these and defaults to the service's
`--declaration`; a restore, an exclude, and an include act on the schema of the
label or Experience they name.

| Request | Body | Response |
|---|---|---|
| `GET /v1/labels?schema_ref=` | | `{"schema_ref", "current_label_id", "modified", "labels": [<label>, ...]}`, newest first |
| `POST /v1/labels` | `{"schema_ref"?, "name"?}` | `<label>` |
| `DELETE /v1/labels/{label_id}` | | `{"deleted": "<label_id>"}` |
| `POST /v1/restore` | `{"label_id"}` | `{"restored": <label>, "saved": <label> or null}` |
| `GET /v1/exclusions?schema_ref=` | | `{"schema_ref", "exclusions": [{"experience_id", "reason", "excluded_at"}], "history": [{"experience_id", "action", "reason", "at"}]}` |
| `POST /v1/exclusions` | `{"experience_id", "reason"}` | `{"experience_id", "status": "excluded"}` |
| `DELETE /v1/exclusions/{experience_id}` | | `{"experience_id", "status": "included" or "not_excluded"}` |

A label is `{"label_id", "schema_ref", "name", "reason", "created_at",
"member_count", "excluded_count"}`; `reason` is `manual`, or `before_restore`
or `before_pull` for a label a restore or a pull made, whose `name` says so
with its time.

### `GET /health`

```json
{"status": "ok", "kb_id": "kb-...", "name": "team-hub", "schema_ref": "schema:sha256:...", "experience_count": 3, "schemas": {"schema:sha256:...": 3}, "pid": 4242, "config_digest": "..."}
```

`kb_id` identifies the service's home and `name` is only for people.
`schema_ref` is the default read schema;
`experience_count` and `schemas` count what reads see; `config_digest`
fingerprints the settings the service started with.

## Client failure behavior

- `read()` never raises for service failures; it returns `status="unavailable"`
  with an empty `prompt_block` and the reason in `warnings`.
- `publish()` raises `RemoteClientError` for permanent rejections (400, 409,
  413, 414, 415, 422).
  Any other failure spools the write, with its declaration, and returns
  `status="spooled"`; Hyperloom's spool is `$USER_DATA_PATH/experience-kb/spool`.
  After that, the client spools every later `publish()` without a request until
  `flush_spool()` delivers one.
- `flush_spool()` replays spooled writes and stops at the first retryable
  failure; files the service rejects for good move to `spool/rejected/`.
  Hyperloom flushes at every launch and before every workspace `push`.
- `write()` raises on any failure and never spools; push uses it.
