# Contributing skills to Arbor

Arbor's agents follow skills: Claude Code's `SKILL.md` folders, each a procedure and the know-how for one task. This
page is for someone adding one, or a family of them: how to lay it out, how to say where in a campaign it takes part,
how to check it, and how a pack loads it.

New to how skills fit into a campaign? Read [skill-integration.md](skill-integration.md) first: it explains the
chapters, the dossier's split skills and where a new skill joins them, with pictures. This page is the reference.

## Two types

- **Runtime** (the default): a campaign loads it. Its own metadata says how it takes part (`run`, `use` or `trigger`)
  and where.
- **Development:** for people working on Arbor itself, and their coding agents. A campaign never loads it
  ([below](#skills-for-developing-arbor)).

A pack sees only the first type, and only says which skills it loads.

## Flat skills and families

Every skill lives under `.claude/skills/`:

- **A flat skill** is `.claude/skills/<name>/SKILL.md`, named `<name>`. Arbor's older skills are flat: `profiler`,
  `kernels`, `hardware`, `serving-playbook`, `dossier-v1` and the rest.
- **A family** is `.claude/skills/<family>/`, marked by its `family.json`. Its skills are named `<family>:<name>`.
  - **Contributed skills come as families.** A family keeps its skills, and the files they share, in one place.
  - **Arbor ships five:** `dossier`, `benchmark`, `pack` and `architecture` for campaigns, and `arbor-dev` for
    development. A new family is added to the list in `arbor/tests/test_skill_families.py`.

```
.claude/skills/<family>/
  family.json                    # {"name": "<family>", "version", "description", "author"}
  .gitattributes                 # * text=auto eol=lf
  shared/                        # optional: what several of its skills read; no SKILL.md
  <name>/SKILL.md                # one folder per skill
  <name>/<reference>.md          # optional: detail the SKILL.md links to
```

- **A name resolves by its colon** (`skillcheck.resolve`): `dossier:discover` is
  `.claude/skills/dossier/discover/SKILL.md`; `profiler` is `.claude/skills/profiler/SKILL.md`.
- **How an agent gets a family:** Claude Code finds skills one folder deep only, so it finds flat skills itself and
  none of a family's. Arbor builds each family a campaign loads as a Claude Code plugin for the session,
  `<session>/skills/plugins/<family>/`, and gives it to the agent (`arbor/skills.py`). The skills keep their names and
  their links, and a family is built again when it changes.
- **Nest nothing** below `<name>/` but files, and keep `family.json`'s `name` the folder's.

## The SKILL.md

```yaml
---
name: <the folder's name>
description: "<what it does and when to use it: the text an agent chooses it by>"
disable-model-invocation: true           # for a run skill: Arbor runs it; agents don't pick it
metadata:
  arbor:
    type: runtime                        # runtime (the default) | development
    modes: [run]                         # run | trigger | use, any of them; use alone when absent
    chapter: map.discover                # run: its chapter (required)
    after: ["dossier:discover"]          # run: its order within the chapter
    before: []
    chapters: [iterate]                  # use: the chapters whose agent may pick it; every one when absent
    triggers: [keep]                     # trigger: its events
    blocking: false                      # trigger: whether the session waits for it
    produces: ["{dossier}/architecture-performance.md"]   # the files Arbor checks after its turn
    reuse_key: [identity, framework]     # config keys: a run whose values match is not run again
    overlaps: ["dossier:discover"]       # skills it overlaps in ways files cannot show (conflicts, below)
    gpu: false
---
```

- **`name`:** must be its folder's name.
- **`description`:** what every agent reads to choose the skill. Claude Code cuts it, with `when_to_use`, past 1,536
  characters. Quote it when it holds `: `, or YAML misreads it: two of Arbor's own skills lost their descriptions that
  way.
- **The body:** under 500 lines (Claude Code's guidance). Move detail into files beside it, and link them by relative
  path; a family's shared files are `../shared/<file>`.
- **Unknown front-matter keys** are ignored by Claude Code. `metadata.arbor` is Arbor's own, and Arbor checks it.

## Where it takes part: chapters, events and the three modes

A session runs through ordered chapters (`arbor/lifecycle.py`): `review`, `init`, `benchmark`, `accuracy`,
`map.discover`, `map.measure`, `map.directions`, `iterate`, `round_end`, `end`. A runtime skill takes part in one or
more of three ways, which its metadata declares.

| Mode | Who decides it runs | Where its metadata places it | What it must do |
| --- | --- | --- | --- |
| `run` | Arbor | in its `chapter`, after the skills it names in `after`, before those in `before`, else in load order; in an agent turn of its own | read the brief its argument names; write the files it `produces`; end its reply with a `LEDGER:` line per file (and a map skill with its one `DIRECTIONS:` line) |
| `trigger` | Arbor | on each of its `triggers` (`keep`, `round_end`, `end`, ...), in a process of its own | do its work without changing the served tree; the session goes on beside it unless it is `blocking` |
| `use` | the agent | in each of its `chapters`, or every chapter when it names none | be a reference or procedure an agent picks by its description |

- **A `run` skill's turn:**
  - **How Arbor starts it:** it writes a brief under `<session>/runs/` and dispatches `/<family>:<name> <brief path>`.
    The brief names the files to write, where things are (`{dossier}`, `{benchmark}`, `{session}`, `{workspace}`) and
    the campaign's program.
  - **After the turn:** Arbor checks the files the skill `produces`. A run whose files stand, for the same `reuse_key`
    values as its last complete run, is skipped. One that leaves a file unwritten stops the chapter, and the next turn
    takes it up again.
- **A `trigger` skill's turn:** a short brief names the event, the session and the journal
  (`<session>/lifecycle.jsonl`), where the event's entry is.
- **A `use` skill** is offered to the agent of its chapters. Nothing else changes.

## Checking it

```bash
python3 -m arbor.skillcheck                 # every flat skill and family in the tree
```

- **It refuses:**
  - front matter that is not YAML;
  - a name that is not its folder's;
  - no description;
  - unknown `metadata.arbor` keys, types, modes, chapters or events;
  - a `run` skill without a chapter, and a placement key without its mode (`chapter` without `run`, `chapters`
    without `use`, `triggers` or `blocking` without `trigger`);
  - a family's missing or misnamed `family.json`, a family with no skill, and a family that is a skill itself;
  - a development skill with a campaign's keys, outside a family, or in a family with runtime skills.
- **It warns** on a body over 500 lines, a description Claude Code would cut, and a `run` skill agents could pick
  (no `disable-model-invocation`).
- **The tests run it over the repository,** so a skill that fails it fails the tests.

## Loading it in a pack

Nothing turns a contributed skill on by itself. A pack loads it, in `pack.yaml`; each skill's metadata does the rest:

```yaml
lifecycle:
  load: ["benchmark:study", "dossier:discover", "architecture:performance", "dossier:measure",
         "dossier:directions", "dossier:update", "dossier:read", profiler]
```

- **The order of `load`** matters only between `run` skills of one chapter that name no `after` or `before` between
  them.
- **`validate_pack` refuses:**
  - a skill the tree does not hold;
  - a development skill;
  - a loaded skill that fails its own check;
  - `after` and `before` that make a cycle;
  - the keys this form replaced: `plugins:`, `triggers:`, and chapters with `run` and `use` lists.
- **A pack without `lifecycle:`** works as before chapters: its `skills:` (flat skills) are used in every chapter.
- **Under a host:** `component.lifecycle.load` replaces the load list.
- **The rest of the pack format:** [packs.md](packs.md).

## When two skills conflict

Two loaded skills **conflict** when their `produces` share a file (or a folder one declares holds a file of the
other's), or when one names the other in `overlaps`.

- **Standalone, a session pauses** before anything runs, and writes `<session>/CONFLICTS.md`. A person decides each
  conflict (`arbor/conflicts.py`):

  ```bash
  python3 -m arbor conflicts --session <dir> show
  python3 -m arbor conflicts --session <dir> pick <id> <skill>             # keep one side, drop the other
  python3 -m arbor conflicts --session <dir> merge <id> [--keep <skill>]   # an agent merges both into one
  ```

  - **To merge by hand,** edit one skill in the tree, then pick it. The record names the skills that changed.
  - **An agent's merge** keeps the kept skill's name, modes and chapter, and must produce every file both did; it is
    written into the session, and checked. One that fails is pending again.
  - **The review gate names them,** so a reviewer can decide them in the same pause.
- **Without a pause:** `run --skill-conflicts merge`, `ARBOR_SKILL_CONFLICTS=merge` or `"skill_conflicts": "merge"` in
  the config lets an agent merge every conflict.
- **Under a host,** a session refuses to start on a conflict, unless auto-merge is set.
- **A decision lasts for its session.** To make one last, drop a skill from `load`, or move the merged skill into the
  tree.

## Worked example: architecture research before profiling

A pull request contributed a `model-architecture-performance` skill. It compares a model's architecture lineage, and
serving frameworks' mechanisms for its changed components, to feed hypotheses into the profile plan.
As contributed, its place in a campaign ("after dossier parts 1-5, before 6-7") was prose in the dossier skill and a
pack's knowledge; nothing ran it there.

**It joined Arbor this way** (`.claude/skills/architecture/`; [its guide](model-architecture-performance.md)):

- **As a family:** `.claude/skills/architecture/`, with the skill at `performance/SKILL.md`
  (`architecture:performance`) and its references beside it.
- **Its declaration:**
  - `modes: [run]`;
  - `chapter: map.discover`, `after: ["dossier:discover"]`. Parts 1-5 are written by then, and `dossier:measure` comes
    later, in `map.measure`;
  - `produces: ["{dossier}/architecture-performance.md"]`;
  - `reuse_key: [identity, framework]`, so the research is not repeated every round;
  - `gpu: false`, and `disable-model-invocation: true`.
- **Its body** keeps the research procedure, and drops the placement prose: the declaration says where it runs.
- **A pack that wants it** adds `architecture:performance` to its `load` list. Every other pack is unchanged.
- **If it also wrote part 1** (`{dossier}/01-model.md`), it would conflict with `dossier:discover`, and a campaign
  loading both would pause for a person, or merge them.

## Skills for developing Arbor

Arbor's own repository has a fourth kind of skill, for people developing Arbor itself: the `develop` mode, in a
family of its own that a campaign never loads. This copy leaves that family out, since it is not edited here
(`VENDOR.md`).

## Landing it

- **Where:** a pull request against `main` in this repository, one change per commit.
- **With it:** its docs, and tests where Arbor's code changes (`python3 -m pytest -q arbor/tests evals/tests`).
- **Say which packs** would load it, and what a campaign gains from it.
