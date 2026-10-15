---
myst:
    html_meta:
        "description": "Reference for Distills in the Hyperloom Experience KB: patterns users draw from groups of one schema's Experiences, kept as versioned entries with frozen evidence, and written into a directory a Curated Wiki searches."
        "keywords": "Hyperloom, Experience KB, Distill, Distilled Wiki, Curated Wiki, evidence, form, scope"
---

# Distills

A **Distill** is a pattern a user draws from a group of one schema's
Experiences: a scope that selects the group, a claim about it, and the
evidence, the group's members, each classified as supporting, contradicting,
or irrelevant to the claim. The KB keeps Distills the way it keeps
Experiences: in the schema's state, moved by its labels and restores, and
hidden by exclusions. Together they are the **Distilled Wiki**.

A **Curated Wiki** is guidance people write at any granularity; the KB does
not manage it. Neither wiki enters a prompt. The KB writes its Distills into
a directory, one page each, that a Curated Wiki searches and cites; the KB
itself offers only listing and reading by id.

The service, its configuration, and the commands that act on Experiences are
in [Experience KB service](experience-kb.md); the Experience record is in
[Experience schema](experience-kb-schema.md).

## Collections

The Distills of one schema form its collection; a Distill names exactly one
schema, and its evidence comes from that schema only. Entries of one
collection may use different forms. A Distill's evidence names Experiences by
id, so a schema whose declaration changes starts a new collection.

## The record

Each version is immutable and kept as one file,
`<home>/<kb_id>/distill/<entry_id>/<version>.json`, verified against its
content hash when read. Its fields, like an Experience's, are knowledge or
metadata:

| Knowledge | Meaning |
|---|---|
| `scope` | The conditions that select the group: `{"<path>": [<value>, ...]}`. |
| `claim` | The fields the form declares, filled in. |
| `evidence` | The members and their classes, as this version froze them; see [Evidence](#evidence). |
| `relations` | `{"entries": [{"kind", "target"}], "external": [{"uri", "version"}]}`: other Distills of the collection (`refines`, `related`, `contradicts`) and outside references, such as a Curated Wiki page. |
| `tags` | Labels for people and for searching the directory. |
| `notes` | Free text keyed by name. |

| Metadata | Meaning |
|---|---|
| `kind` | `distill`. |
| `id` | `dst-<32 hex>`, made at creation and kept by every version. |
| `version` | 1 for the first; each update adds the next. |
| `schema_ref`, `form_ref` | The schema it draws from and the form its claim is written in. |
| `created_at`, `author` | When the version was written, and by whom. |
| `provenance` | `{"producer": "hyperloom-kb", "producer_version": <code digest>}` for a version this KB wrote. |
| `derivation` | `{"kb_state"}`: the schema's last write position and the digest of its hidden Experiences when the version was written. |

## Forms

A form declares a claim's fields, with the
[field kinds and attributes](experience-kb-schema.md#field-attributes-and-roles)
an Experience's fields take: `{"form_version": 1, "name", "description",
"claim": [<field>, ...]}`. A claim field cannot join a group, and the only
role it takes is `summary`, on at most one field; listings and pages show that
field as the Distill's title. A form is identified by `form_ref`, the digest of
its declaration, and registered once per schema.

## Scope and conditions

A condition names a path and the values it accepts. A path is `objective` or
`<category>.<field>` for a single-scalar field the schema declares, and every
value must be one the field takes. A scope holds where every condition holds:
the value at each path is one of the condition's values. An Experience without
a value at a path is outside the scope.

`GET /v1/schemas/{schema_ref}` lists the paths a scope may name, and the
fields the schema gives each role.

## Drafts

An author writes a draft; the KB adds everything else a version records.

| Field | Required | Meaning |
|---|---|---|
| `schema_ref` | to create | The schema; an update keeps its Distill's. |
| `form_ref` | yes | A form registered for the schema. |
| `scope`, `claim` | yes | As in the record. |
| `check` | no | A condition set, in the scope's language, that classifies each member: supporting where it holds, contradicting where it does not, irrelevant where a value it names is missing. |
| `add`, `remove` | no | Visible Experiences of the schema to add to the group the scope selects, and members to leave out of it. |
| `annotations` | no | `{"<experience id>": "support" \| "contradict" \| "irrelevant"}`, overriding the check for those members. |
| `representatives` | no | Members a page names for the rest. |
| `relations`, `tags`, `notes`, `author` | no | As in the record. |

A preview answers what a create, or an update, would store, without storing
anything: `valid`, the reasons when it is not, the size of the group the scope
selects, the evidence counts, the knowledge fields, and the visible Distills
whose scope could select the same Experiences, as a hint, never a refusal.
Creating it stores version 1 and makes it visible at once; there is no review
step. An update is a whole draft: it stores the next version, its evidence
drawn again from the KB as it is now.

Every read of a version answers its `draft`: the draft that writes it again
over the KB as it is, with its check, edits, annotations, and
representatives, less any id no longer visible (`hyperloom-kb distill
draft`). An update starts from it, so a Distill keeps its author's
annotations when someone else refreshes it.

## Evidence

The members are the visible Experiences the scope selects, newest first, less
`remove`, plus `add`. Each is classified by its annotation, else by the check,
else as supporting: without a check, the author claims the whole group
supports the claim and annotates the exceptions. A version records the
members and their classes, the check, the edits (`added`, `removed`), the
members the author annotated (`annotated`), and the counts:
`{"supports", "contradicts", "irrelevant", "runs"}`, `runs` counting the
distinct runs the members come from. There is no minimum; whether the
evidence suffices is the author's call.

## Evidence drift

A version's evidence never changes. A listing, a read, the changes inbox,
and a directory sync run its scope again over the schema's visible
Experiences: a member the version holds keeps its class, a new member is
classified by the version's check, or undetermined without one, and a member
the author removed never returns. Each Distill gets one evidence status, the
first that applies:

| Status | When | Needs attention |
|---|---|---|
| `contradicted` | New members contradict the claim. | yes |
| `weakened` | Members that supported it are no longer visible. | yes |
| `undetermined` | New members no check classifies. | yes |
| `strengthened` | The evidence moved, and nothing it gained or lost tells against the claim. | no |
| `consistent` | The evidence is the version's. | no |

The KB never writes a version by itself. An author takes in the new evidence
by updating the Distill, which freezes the current evidence in the next
version and takes it out of the inbox; narrows the scope; writes a new
Distill related to it; or deletes it. A contradiction that leaves the claim
standing is taken in by an update too.

## Lifecycle

- **Delete** a Distill with a reason through `POST /v1/exclusions`, naming its
  id; `DELETE /v1/exclusions/{entry_id}` brings it back. A deleted Distill
  leaves listings and the directory, and stays readable by id, marked
  `deleted`.
- **Labels and restores** cover the collection: a label keeps the version
  each Distill shows, and a restore shows those versions again and sets
  outside the state each Distill created after the label.
- With a global Experience KB set (`HYPERLOOM_GLOBAL_KB_URL` and
  `HYPERLOOM_GLOBAL_KB_TOKEN`), `experience_kb_service push` and `pull` carry
  the workspace's Distills with its Experiences.

## The directory

`hyperloom-kb distill sync --dir DIR` writes each visible Distill as a page
`DIR/schema-<16 hex of the schema digest>/<entry_id>.md` and removes the page
of a Distill no longer visible, so each run leaves the directory a mirror of
what the KB shows; a page edited by hand is written back. It writes and
removes only files named like a page, each replaced whole, and leaves every
other file alone, so the directory can sit inside a Curated Wiki's tree.
Run through `python -m hyperloom.inference_optimizer.experience_kb_service`,
`distill sync` writes the workspace's schema into `distilled/` under the
service's data home unless `--dir` names another directory.

A page's YAML frontmatter, one JSON value per key, carries `id`, `version`,
`schema_ref`, `form`, `form_ref`, `tags`, `scope`, `evidence_status`,
`needs_attention`, `evidence` (the version's counts as `snapshot` and today's
as `current`), `representatives`, and `relations`. Its body is the claim,
titled by its summary field, then the evidence in words and a line saying how
to read the rest by id; no id appears in the body, so none becomes text a
search matches.

`representatives` names up to five visible members (`--representatives N`):
the author's, or else every contradicting member, then the newest supporting
member of each run, newest first. The complete members are read with
`hyperloom-kb distill get --follow-evidence`.

## HTTP API

| Endpoint | Purpose | `RemoteClient` |
|---|---|---|
| `GET /v1/schemas/{schema_ref}` | the declaration, the paths a scope may name, its roles, and what it holds | `schema()` |
| `GET /v1/experiences/{id}` | one held Experience, whether reads see it, and where this host holds its files | `experience()` |
| `POST /v1/distill/explore` | the group a scope selects, summarized and crossed over one or two fields | `explore()` |
| `GET /v1/distill/forms?schema_ref=` | the forms registered for a schema | `distill_forms()` |
| `PUT /v1/distill/forms` | register a form | `register_distill_form()` |
| `POST /v1/distill/preview` | what a draft would store | `preview_distill()` |
| `POST /v1/distill/entries` | create a Distill | `create_distill()` |
| `PUT /v1/distill/entries/{id}` | add a version | `update_distill()` |
| `GET /v1/distill/entries/{id}` | read one version | `distill()` |
| `GET /v1/distill/entries?schema_ref=` | list a schema's Distills | `list_distills()` |
| `GET /v1/distill/changes` | the Distills that need attention | `distill_changes()` |

| Request | Body or query | Response |
|---|---|---|
| explore | `{"schema_ref", "scope"?, "group_by"?: [<path>, at most 2], "member_limit"?: 1–500, default 20}` | `{"schema_ref", "scope", "count", "runs", "repeat_groups", "decisions", "measurement", "change_pct", "members", "members_total"}`, and with `group_by` its `cells`, largest first, each with its `values` and the same summary |
| forms | `{"schema_ref", "form"}` | `{"schema_ref", "form_ref", "name", "status": "created" \| "unchanged"}` |
| preview | `{"draft", "entry_id"?}` | `{"valid", "errors", "schema_ref", "form_ref", "population", "evidence", "knowledge", "overlaps"}` |
| create, update | `{"draft"}` | `{"status", "entry_id", "version", "evidence"}` |
| read | `version`, `representatives` (1–50, default 5), `follow_evidence`, `after`, `limit` (1–500, default 50) | `{"entry", "current_version", "versions", "deleted", "outside", "drift", "representatives", "draft"}`, and with `follow_evidence` an `evidence` page: `{"items": [{"id", "class", "held", "visible", "experience", "files"}], "next", "has_more", "total"}` |
| list | `schema_ref`, `include_deleted` | `{"schema_ref", "items": [...]}`, newest first |
| changes | `schema_ref`, or none for every schema | `{"items": [...], "checked": {"<schema_ref>": {"<status>": <count>}}}` |

`measurement` and `change_pct` are distributions, `{"count", "min", "p25",
"median", "p75", "max"}`, of the outcome measurement and of its change over
the baseline in percent; they read only the roles the schema declares.

A listing item is `{"entry_id", "version", "content_hash", "schema_ref",
"form_ref", "form_name", "summary", "tags", "scope", "supports",
"contradicts", "irrelevant", "runs", "stored_at", "state", "evidence_status",
"needs_attention", "current"}`: the counts are the version's, and `current`
counts today's evidence, `undetermined` included. A read's `drift` and a
changes item add what moved: `gained` and `lost` counts by class, and the
first 20 of `gained_contradicting`, `gained_undetermined`, and
`lost_supporting`, each `{"id", "summary"}`.

A draft the KB cannot hold is 400; an unregistered form, or a relation to a
Distill it does not hold, is 404; a version under one number with other
content than the one held is 409.
