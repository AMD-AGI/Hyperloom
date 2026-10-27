---
name: hyperloom-kb-distill
description: Draws Distills, patterns across groups of Experiences, in an Experience KB with the hyperloom-kb CLI. Explores a schema's Experiences by scope and grouping, picks or registers a form, previews and creates a Distill, keeps it current when its evidence moves, reads one with its evidence, and syncs the Distilled pages into a directory a Curated Wiki searches. Use when asked to distill, summarize a pattern from, or write wiki knowledge from Experiences, to check which Distills need attention, to look up a Distill, or to sync the Distilled directory.
---

# Distill patterns from Experiences

A Distill is a claim about a group of one schema's Experiences, with the
group as its evidence: each member supports, contradicts, or is irrelevant
to the claim. The KB keeps each version with the evidence frozen as written,
tells you when new Experiences move that evidence, and writes every visible
Distill as a page into a directory a Curated Wiki searches.

Connect as the `hyperloom-kb` skill does: `HYPERLOOM_KB_URL` and
`HYPERLOOM_KB_TOKEN`, loaded without printing them. Every command prints
JSON. A command that names no schema acts on the service's default one, the
`schema_ref` in `hyperloom-kb health`.

The user decides what a Distill claims and whether its evidence suffices.
State a pattern only from a group: one Experience is never a Distill. Never
add, remove, or annotate members to make evidence look stronger than it is.

## 1. Look at the schema and its Distills

```bash
hyperloom-kb schema show [--schema REF]
hyperloom-kb distill list [--schema REF] [--include-deleted]
```

`schema show` lists the `paths` a scope and a grouping may name (`objective`
and `<category>.<field>` for each single-scalar field) and the fields the
schema gives each role. `distill list` shows each Distill's summary, scope,
the counts it was written with, and its `evidence_status` with `current`
counts; read one with `distill get` before writing another over the same
ground.

## 2. Explore

```bash
hyperloom-kb distill explore --scope '{"identity.model": ["qwen3"]}'
hyperloom-kb distill explore --scope '{"identity.model": ["qwen3"]}' --group-by change.attention --group-by outcome.decision
hyperloom-kb experience get EXPERIENCE_ID
```

A scope maps paths to the values each accepts; every condition must hold, and
values have the field's kind (a number field takes `2`, not `"2"`). Explore
answers the group's `count`, `runs`, `repeat_groups`, `decisions`, and the
distributions of the outcome `measurement` and its `change_pct` over the
baseline, with the first member ids (`--member-limit N`). `--group-by`, at
most twice, adds a cross table, largest cell first. Narrow and regroup until
a pattern stands out, then read a few members with `experience get`.

## 3. Pick a form

```bash
hyperloom-kb distill forms [--schema REF]
hyperloom-kb distill add-form form.json [--schema REF]
```

A form declares the claim's fields, as a schema declares an Experience's;
give one text or string field the `summary` role, the page's title:

```json
{"name": "rule", "description": "What a change does under the scope.",
 "claim": [
   {"name": "statement", "description": "What holds.", "kind": "text", "required": true, "role": "summary"},
   {"name": "effect", "description": "Its direction.", "values": ["helps", "hurts", "neutral"]}]}
```

Reuse a registered form when one fits; a form is identified by its content.

## 4. Draft and preview

Write the draft to a file; `-` reads it from stdin.

```json
{"form_ref": "form:sha256:...",
 "scope": {"identity.model": ["qwen3"], "change.attention": ["triton"]},
 "claim": {"statement": "Triton attention alone does not raise qwen3 throughput.", "effect": "hurts"},
 "check": {"outcome.decision": ["revert"]},
 "tags": ["attention"], "author": "<the user's name>"}
```

```bash
hyperloom-kb distill preview draft.json   # exits 1 with the errors when the KB would refuse it
```

- `check`, in the scope's language, classifies each member: supporting where
  it holds, contradicting where it does not, irrelevant where a value it
  names is missing. Without one every member supports; annotate the others.
- `remove` leaves members out of the group; `add` brings in visible
  Experiences the scope does not select; `annotations`
  (`{"<id>": "support" | "contradict" | "irrelevant"}`) override the check.
  Record why in `notes`.
- `representatives` names the members a page shows; without it the KB picks
  every contradicting member, then the newest supporting member of each run.
- `relations` links other Distills (`refines`, `related`, `contradicts`) and
  outside references such as a Curated Wiki page.

Read `population`, the `evidence` counts and `runs`, and `overlaps`: visible
Distills whose scope could select the same Experiences. Show the user the
counts and any overlap, and let them decide whether to create, update the
overlapping one instead, or stop.

## 5. Create and sync

```bash
hyperloom-kb distill create draft.json [--schema REF]
hyperloom-kb distill sync --dir DIR [--schema REF]
```

A create is visible at once; there is no review step. `sync` makes `DIR`
mirror the visible Distills, one `<entry_id>.md` page each under a
`schema-<digest>` subdirectory: it rewrites changed pages, removes the page
of a Distill no longer visible, and leaves every other file alone. Run it
after creating, updating, deleting, restoring, or pulling, and report its
`needs_attention` list. Use the directory the user keeps for it, typically
inside their Curated Wiki's tree; never edit its pages by hand.

## 6. Keep Distills current

```bash
hyperloom-kb distill changes [--schema REF]
hyperloom-kb distill draft ENTRY_ID > draft.json   # the Distill's draft, to edit
hyperloom-kb distill update ENTRY_ID draft.json
hyperloom-kb distill delete ENTRY_ID --reason "superseded by a per-model rule"
hyperloom-kb distill restore-entry ENTRY_ID
```

`distill draft` prints the draft that writes the Distill's version again,
with its check, edits, and annotations; ids no longer visible are left out.
Start every update from it, a pulled Distill included, so the author's
annotations carry over.

`changes` lists the Distills whose evidence moved in a way that needs the
author: `contradicted` (new members contradict it), `weakened` (supporting
members are no longer visible), and `undetermined` (new members no check
classifies), with those members. `strengthened` and `consistent` need nothing.
Report what moved and let the user choose:

- the claim still holds: update with its draft unchanged, which freezes the
  current evidence in a new version;
- new members are undetermined: annotate them in the draft and update;
- it holds only for part of the group: update with a narrower scope, and
  create a related Distill for the rest;
- it no longer holds: rewrite the claim, or delete the Distill with a reason.

An update is a whole draft and adds the next version; the id never changes.
`delete` hides a Distill from listings and the directory, `restore-entry`
brings it back, and a label restore returns every Distill to the version it
showed.

## Read a Distill

```bash
hyperloom-kb distill get ENTRY_ID [--version N] [--representatives N]
hyperloom-kb distill get ENTRY_ID --follow-evidence [--after NEXT] [--limit N]
```

`get` answers the version's record, the versions it has, whether it is
deleted, its `drift` (status, current counts, and the members that moved),
and its `representatives`. `--follow-evidence` adds a page of its members
with each one's record and file paths; repeat with `--after <next>` while
`has_more` is true. A page in the directory names its id in the frontmatter:
read it this way when an agent needs more than the page.

## Share

A push to the global KB sends the latest version of each visible Distill after
the Experiences, and a pull of a schema brings the global KB's. A version both
sides wrote differently under one number is reported under `rejected`; an
update here writes the next version, which settles it.
