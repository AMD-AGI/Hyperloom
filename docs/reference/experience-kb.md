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
| `HYPERLOOM_KB_DATABASE_URL` | the operator | The PostgreSQL database a service keeps its index and state in. Unset, the service runs an embedded PostgreSQL in its home, which is what a workspace on Python 3.12 or newer does. |
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
command on the workspace's service (`health`, `rebind`, `labels`, `label`, `restore`,
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
| `rebind` | `POST /v1/rebind` |
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
- such a service started with other settings, serving another default
  declaration, or running other Experience KB code, such as one started before
  an upgrade: it restarts it, keeping its data. Settings are compared by their
  effect, so the same key under another variable name does not restart it;
- a service holding another workspace's data, as a copied `.env` would point
  at, or anything else on the port, such as another user's service: it refuses
  without stopping it, and the user picks another port in `HYPERLOOM_KB_URL`.

A data home has one service. A service runs its home's embedded PostgreSQL
and holds `service.lock` while it serves; one started on a home another service
holds exits, naming that service's pid and port, and `ensure` reports that line.
Two workspaces therefore need their own `USER_DATA_PATH`, or the second gets no
service of its own.

The embedded server is the PostgreSQL the `pgembed` wheel ships for Linux and
macOS on Python 3.12 or newer; on an older Python, Hyperloom installs without
it, and a workspace's service starts only once `HYPERLOOM_KB_DATABASE_URL`
names a PostgreSQL server. It is reachable only through a socket in the home,
or in `/tmp` when the home's path is too long for one. A root service runs it
as the system user `hyperloom-kb-db`, or as the user owning a database
directory made earlier, so a recreated container that mounts the same home
starts it again. No directory's permissions are changed for it: a home or a
PostgreSQL install that user cannot traverse to, such as one under a `700`
`/root`, is refused with the directory that blocks it, and the workspace and
`USER_DATA_PATH` belong somewhere every user may traverse.
`experience_kb_service check-home` says before a launch whether the workspace's
service could start there; setup runs it in both modes, since a `docker` run is
always root and a container image keeps its own `/root` private.

The database directory must stay private to that user, so a home whose file
system keeps no permissions (a Windows drive mounted into WSL) or refuses a
root caller's change of owner (NFS exported with `root_squash`) is refused with
that reason. Elsewhere, and on Windows, a service needs
`HYPERLOOM_KB_DATABASE_URL`. Record files need only a file and an atomic
rename; a directory is flushed after each rename where its file system can
flush one.

Only an optimize launch and `ensure` restart a service. `push` and `pull`,
including the automatic push at the end of a run, use the service as it runs
and warn when its settings differ from theirs, so a push from a shell never
stops the service a running session reads from.

No Experience KB problem stops a run. An optimize launch whose service cannot
start, whose `.env` names the service without its token, or whose packaged
mapping cannot load logs a warning and continues: reads return nothing,
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
| `postgres/` | The embedded PostgreSQL database: the KB's `kb_id`, its schemas, the index of its records in write order, labels, exclusions and their history, sync progress, and every write with the KB that sent it. |
| `<kb_id>/records/` | One immutable file per Experience, read only against the content hash the database holds for it. |
| `spool/` | Writes the service has not accepted yet. |
| `service.log` | The service's log, and the embedded database's in `postgres/server.log`. |
| `service.lock` | Held by the one service serving this home; names its pid and port. |

A home an older service kept on disk, with `canonical/`, `identity.json`, and
`sync.sqlite3`, is adopted the first time it is served: its Experiences, its
`kb_id`, and the Experiences it pulled, which push never sends back, move into
the database. The old files are left as they are.

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
reads see it. One that was excluded is sent only after an include releases it,
even when a restore to a label without that exclusion lets reads see it again.
A push cannot take back what it already sent. List and export
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
restoring it undoes the pull. When the global KB's exclusions or restores have
changed what it shows of the schema since the last pull, the pull pages the
schema from its start again, so an Experience the global KB shows again arrives
too and the ones already here count as `unchanged`. A workspace `pull` names
the schema its packaged mapping writes.

A pull's cursor is a write position on the global KB together with the
Experience it found there. When the global KB holds another Experience at that
position, or none, as after its database was restored from an older backup and
took new writes, the pull pages the schema from its start again, so nothing the
global KB holds now is skipped, and the next push sends everything written here
again, since the global KB may have lost what was pushed too. What was pulled
before and the global KB no longer holds stays here.

**Identity.** Every KB has a `kb_id`, made when its database is first served
and kept in it, so every service of one database is the same KB and a new
database is a new one. The first push or pull to a global KB records its
`kb_id`; a later sync where that URL answers with another `kb_id`, such as a
redeployed global KB, is refused, as is any sync with a service that reports
no `kb_id`, which predates this. A refused sync reports `status: refused` with
the reason and changes nothing. When the new global KB replaced the old one,
`rebind` forgets the old one, its identity, cursors, and what the service knew
of it, so the next push and pull start over with the KB that answers now. What
was pulled from the old one stays pulled and is never pushed to the new one.

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

Given `HYPERLOOM_KB_DATABASE_URL`, a service keeps its index and state in that
PostgreSQL database instead of an embedded one, and any number of services may
serve one KB from one database and one `--home` its record files live under,
such as several replicas behind one URL. Every write takes the KB's next
position under that KB's row lock, so positions commit in order and a pull
paging by position, through whichever replica, never passes one still to
commit. Labels, restores, and exclusions of a schema hold its row for their
transaction, and each push or pull of a KB holds a database-wide lock, so
replicas never interleave them. A database holds one KB for now; serving one of
several is left for when requests carry who they act for. Such a service
installs with `pip install ".[kb-service]"`, which leaves out the embedded
server, and its replicas share `--home` on one file system they all mount with
atomic rename, such as CephFS or NFS.

The database must never lose a write it committed: a workspace pulls and
pushes everything again once it notices, but what only the lost writes held is
gone. A failover may therefore promote only a replica that confirmed every
commit, as synchronous replication guarantees.
The pool checks each connection as it hands it out, so the first request after
a failover or restart gets a live connection.

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
| `POST /v1/rebind` | forget the global KB synced with, to start over with the one at its URL | `rebind()` |
| `GET /v1/labels` | a schema's labels, current label, and whether its state changed since | `labels()` |
| `POST /v1/labels` | label a schema's current state | `create_label()` |
| `DELETE /v1/labels/{label_id}` | delete a label | `delete_label()` |
| `POST /v1/restore` | make a label's state current | `restore()` |
| `GET /v1/exclusions` | a schema's exclusions and their history | `exclusions()` |
| `POST /v1/exclusions` | exclude an Experience from reads | `exclude()` |
| `DELETE /v1/exclusions/{experience_id}` | lift an exclusion | `include()` |
| `GET /health` | identity, liveness, schemas, corpus size, process, settings digest | `health()` |
| `GET /livez` | the process answers; no token | |
| `GET /readyz` | the service can take traffic; no token | |
| `GET /metrics` | Prometheus metrics; no token | |

Every other request, `/health` included, sends `Authorization: Bearer <token>`;
see [Observability](#observability) for the three that need none.
Unknown request fields are rejected, so a misspelled field fails loudly. A
request body may be up to 256 MiB; Experiences themselves have no size limit.

| Status | Body `error` | Meaning | Retry? |
|---|---|---|---|
| 400 | `invalid_request` | malformed body, unknown field, invalid Experience, unregistered schema, out-of-range parameter | no |
| 401 | `unauthorized` | missing or wrong token | after fixing the token |
| 404 | `not_found` | unknown path, label, or Experience | no |
| 409 | `conflict` | the id already exists with different content | no |
| 409 | `sync_unavailable` | push, pull, or rebind on a service started without a global KB | after configuring one |
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
| `content_inline_limit` | no | none (all inline) | bytes above which a free-text field (`reasoning`, `reflection`, `change.summary`, `change.content`, an alternative, a note) is rendered as a reference instead of inline |
| `render_budget_chars` | no | none (every record) | characters the rendered records may fill; records are rendered whole, in rank order, while they fit |

The service's planner turns `decision` and `context` into weighted query
signals, then ranks candidates by exact field matches plus lexical fuzzy
matches over identity values, the change summary and kind, `reasoning`,
[notes](experience-kb-schema.md#notes), and the outcome. Experiences that
repeat the same change under the same identity form one Repeat Group, which
contributes at most one Experience; `outcome` filters by
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
{"items": [{"sequence": 1, "experience_id": "exp-...", "...": "..."}], "next_cursor": 1, "has_more": false}
```

`/v1/list` items are summaries plus `sequence`; `/v1/export` items are
`{"sequence": 1, "experience": <complete Experience>}`. An export page also
carries `head`, the last write position of what it pages; `after_id` and
`next_cursor_id`, the Experiences written at `after` and at `next_cursor`, or
empty where none was; and, with a `schema_ref`, that schema's `declaration` and
its `state`, an opaque value that changes whenever exclusions or restores change
which stored Experiences of the schema the service shows.

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
is another one than this service synced with or reports no identity.

`POST /v1/rebind` takes `{}` and answers `{"global_url", "forgotten_kb_id"}`,
the identity forgotten, empty when the service had not synced yet.

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
{"status": "ok", "kb_id": "kb-...", "name": "team-hub", "schema_ref": "schema:sha256:...", "experience_count": 3, "schemas": {"schema:sha256:...": 3}, "pid": 4242, "config_digest": "...", "code_digest": "..."}
```

`kb_id` identifies the KB the service's database holds, and `name` is only for people.
`schema_ref` is the default read schema;
`experience_count` and `schemas` count what reads see; `config_digest`
fingerprints the settings the service started with, and `code_digest` the
Experience KB code it runs.

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
- Every request carries a fresh `X-Request-ID`, and every `RemoteClientError`
  names it, so a failure a client reports is found in the service's log by
  that id.

## Observability

Every service, local or global, reports the same way; a global KB's
dashboards and alerts are built on these, and a workspace reads them
directly.

**Probes.** Three endpoints answer without a token and carry nothing a KB
holds:

| Endpoint | Answer |
|---|---|
| `GET /livez` | `200 {"status": "alive"}` while the process answers. |
| `GET /readyz` | `200` when every check passes, `503` otherwise, with each check `ok` or `failed`: `database` (a query answers), `home_writable`, `disk_space` (at least 512 MiB free under the home), and `records` (every record the database holds had its file, of its size, when the service started). |
| `GET /metrics` | The Prometheus text format. |

**Metrics.** Counters count since the process started; gauges are sampled at
each scrape.

| Metric | Labels | Meaning |
|---|---|---|
| `hyperloom_kb_http_requests_total` | `method`, `route`, `status` | Requests answered; `route` is the path template, such as `/v1/experiences/{experience_id}`. |
| `hyperloom_kb_http_request_duration_seconds` | `method`, `route` | Histogram of answer times, 5 ms to 30 s buckets. |
| `hyperloom_kb_http_requests_in_flight` | | Requests being answered now. |
| `hyperloom_kb_http_request_bytes_total`, `hyperloom_kb_http_response_bytes_total` | `route` | Body bytes received and sent. |
| `hyperloom_kb_http_unauthorized_total` | | Requests refused for their token. |
| `hyperloom_kb_writes_total` | `schema_ref`, `result` | Writes by result: `created`, `unchanged`, `conflict`. |
| `hyperloom_kb_sync_batches_total` | `direction`, `status` | Push and pull batches by status: `completed`, `incomplete`, `refused`. |
| `hyperloom_kb_experiences` | `schema_ref`, `state` | Stored Experiences: `visible`, `excluded`, or `outside` the state after a restore. |
| `hyperloom_kb_record_bytes`, `hyperloom_kb_database_bytes`, `hyperloom_kb_disk_free_bytes` | | Record files, database, and free disk. |
| `hyperloom_kb_last_write_timestamp_seconds` | | When the KB last stored a new Experience. |
| `hyperloom_kb_records_missing` | | Records whose file was missing at start. |
| `hyperloom_kb_ready` | `check` | Each readiness check, `1` or `0`. |
| `hyperloom_kb_database_pool` | `stat` | `pool_size`, `pool_available`, `requests_waiting`. |
| `hyperloom_kb_build_info` | `kb_id`, `name`, `code_digest` | Always `1`; names the KB and the code serving it. |
| `hyperloom_kb_start_time_seconds` | | When the process started. |

Replicas of one KB each count their own requests, writes, and batches, so those
add up across replicas; `hyperloom_kb_experiences`, `hyperloom_kb_record_bytes`,
`hyperloom_kb_database_bytes`, and `hyperloom_kb_last_write_timestamp_seconds`
describe the KB, which every replica reports the same, so take their maximum.

**Logs.** A service logs one JSON object per line, to standard error; a
workspace's service writes them to `service.log`. Every line has `ts`,
`level`, `logger`, `message`, and `event`:

| `event` | When | Fields |
|---|---|---|
| `http_request` | every answered request but the probes | `method`, `route`, `status`, `duration_ms`, `bytes_in`, `bytes_out` |
| `audit` | a schema registered, a label made or deleted, a restore, an exclude, an include, a rebind | `action` and what it acted on: `schema_ref`, `label_id`, `experience_id`, `reason`, `saved_label_id`, `global_url`, `forgotten_kb_id` |
| `sync` | a push or pull batch | `direction`, `status`, `global_url`, the counts, `error` |
| `records_verified` | at start | `records`, `missing` |
| `listening`, `stopped` | start and stop | `port`, `home`, `kb_id`, `code_digest` |

A line logged while answering a request also has its `request_id`, and the
`client_kb_id` and `client_name` of the KB that sent it, when it named itself.
Tokens and request bodies are never logged.

**Which KB sent what.** A service syncing with a global KB names itself on
every request with `X-Hyperloom-KB-Client` (its `kb_id`) and
`X-Hyperloom-KB-Client-Name`, and every request carries `X-Request-ID`, which
the answer echoes. The service records every write, refused ones included,
with its result, the KB that sent it, and its request id, in the database's
`writes` table; a pull records the global KB as the sender of what it brings.

**Log rotation and stopping.** A workspace's `service.log` past 8 MiB is kept
as `service.log.1`, replacing the one kept before, when the next service
starts. `SIGTERM` stops a service taking requests, lets those in flight finish
for up to 25 seconds, and stops the embedded database the service started.
