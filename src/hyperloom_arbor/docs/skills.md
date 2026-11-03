# Contributing skills to Arbor

Arbor's agents follow skills: Claude Code's `SKILL.md` folders, each a procedure and the know-how for one task. This
page is for someone adding one, or a family of them: how to lay it out, how to say where in a campaign it takes part,
how to check it, and how a pack turns it on.

New to how skills fit into a campaign? Read [skill-integration.md](skill-integration.md) first: it explains the
chapters, the dossier's split skills and where a new skill joins them, with pictures. This page is the reference.

## Flat skills and families

- **A flat skill** is `.claude/skills/<name>/SKILL.md`, named `<name>`. Arbor's older skills are flat: `profiler`,
  `kernels`, `hardware`, `serving-playbook` and the rest.
- **A family** is a Claude Code plugin under `plugins/<family>/`. Its skills are named `<family>:<name>`, as Claude Code
  names a plugin's skills.
  - **Contributed skills come as families.** A family keeps its skills, and the files they share, in one place.
  - **Arbor ships three:** `dossier`, `benchmark` and `pack`.

```
plugins/<family>/
  .claude-plugin/plugin.json     # {"name": "<family>", "version", "description", "author"}
  .gitattributes                 # * text=auto eol=lf
  shared/                        # optional: what several of its skills read
  skills/<name>/SKILL.md         # one directory per skill, one level deep
  skills/<name>/<reference>.md   # optional: detail the SKILL.md links to
```

Claude Code finds a skill one directory deep only. Nest nothing below `skills/<name>/`, and keep the family's name the
directory's.

## The SKILL.md

```yaml
---
name: <the directory's name>
description: "<what it does and when to use it: the text an agent chooses it by>"
disable-model-invocation: true           # for a run skill: Arbor runs it; agents don't pick it
metadata:
  arbor:
    modes: [run]                         # run | trigger | use, any of them; use alone when absent
    chapter: map.discover                # a run skill's chapter
    after: ["dossier:discover"]          # its order within the chapter
    before: []
    produces: ["{dossier}/architecture-performance.md"]   # the files Arbor checks after its turn
    reuse_key: [identity, framework]     # config keys: a run whose values match is not run again
    triggers: [keep]                     # a trigger skill's events
    blocking: false                      # a trigger skill that must end before the session goes on
    gpu: false
---
```

- **`name`:** must be its directory's name.
- **`description`:** what every agent reads to choose the skill. Claude Code cuts it, with `when_to_use`, past 1,536
  characters. Quote it when it holds `: `, or YAML misreads it: two of Arbor's own skills lost their descriptions that
  way.
- **The body:** under 500 lines (Claude Code's guidance). Move detail into files beside it, and link them by relative
  path; the shared files of a family are `../../shared/<file>`.
- **Unknown front-matter keys** are ignored by Claude Code. `metadata.arbor` is Arbor's own, and Arbor checks it.

## Where it takes part: chapters, events and the three modes

A session runs through ordered chapters (`arbor/lifecycle.py`): `review`, `init`, `benchmark`, `accuracy`,
`map.discover`, `map.measure`, `map.directions`, `iterate`, `round_end`, `end`. A skill takes part in one or more of
three ways.

| Mode | Who decides it runs | When | What it must do |
| --- | --- | --- | --- |
| `run` | Arbor | in its chapter, in the pack's order, in an agent turn of its own | read the brief its argument names; write the files it `produces`; end its reply with a `LEDGER:` line per file (and a map skill with its one `DIRECTIONS:` line) |
| `trigger` | Arbor | when an event it registers for fires (`keep`, `round_end`, `end`, ...), in a process of its own | do its work without changing the served tree; the session goes on beside it unless it is `blocking` |
| `use` | the agent | whenever the chapter's agent chooses | be a reference or procedure an agent picks by its description |

- **A `run` skill's turn:**
  - **How Arbor starts it:** it writes a brief under `<session>/runs/` and dispatches `/<family>:<name> <brief path>`.
    The brief names the files to write, where things are (`{dossier}`, `{benchmark}`, `{session}`, `{workspace}`) and
    the campaign's program.
  - **After the turn:** Arbor checks the files the skill `produces`. A run whose files stand, for the same `reuse_key`
    values as its last complete run, is skipped. One that leaves a file unwritten stops the chapter, and the next turn
    takes it up again.
- **A `trigger` skill's turn:** a short brief names the event, the session and the journal
  (`<session>/lifecycle.jsonl`), where the event's entry is.
- **A `use` skill** is offered to the agent of the chapters that list it. Nothing else changes.

## Checking it

```bash
python3 -m arbor.skillcheck                 # every flat skill and family in the tree
```

- **It refuses:**
  - front matter that is not YAML;
  - a name that is not its directory's;
  - no description;
  - unknown `metadata.arbor` keys, modes, chapters or events;
  - a family's missing or misnamed `plugin.json`, or a family with no skill;
  - a `develop` skill with another mode or a campaign's keys, outside a family, or in a family with campaign skills.
- **It warns** on a body over 500 lines, a description Claude Code would cut, and a `run` skill agents could pick
  (no `disable-model-invocation`).
- **The tests run it over the repository,** so a skill that fails it fails the tests.

## Turning it on in a pack

Nothing turns a contributed skill on by itself. A pack opts in, in `pack.yaml`:

```yaml
plugins: [dossier, architecture]
lifecycle:
  map.discover:   {run: ["dossier:discover", "architecture:performance"]}
  map.measure:    {run: ["dossier:measure"], use: [profiler]}
  map.directions: {run: ["dossier:directions"]}
  iterate:        {use: ["dossier:read", profiler]}
triggers:
  keep: ["<family>:<skill>"]    # a skill that declares modes: [trigger] and triggers: [keep]
```

No skill Arbor ships declares the `trigger` mode yet, so a pack's `triggers:` names a contributed one.

- **`validate_pack` refuses:**
  - a family the tree lacks, or a family for developing Arbor (below);
  - a skill no listed family or flat skill holds;
  - a skill listed in a way its metadata does not declare (a `run` list naming a `use` skill, or another chapter's);
  - a chapter's run order against `after` and `before`.
- **Under a host:** `component.lifecycle` replaces the chapters it names.
- **The rest of the pack format:** [packs.md](packs.md).

## Worked example: architecture research before profiling

A contributed pull request adds a `model-architecture-performance` skill. It compares a model's architecture lineage, and
SGLang's, vLLM's and TensorRT-LLM's mechanisms for its changed components, to feed hypotheses into the profile plan. As
written, its place in a campaign ("after dossier parts 1-5, before 6-7") is prose in the dossier skill and a pack's
knowledge; nothing runs it there.

**This shows how such a skill fits the lifecycle. It is not integrated:** how it joins Arbor is the maintainers'
call, and their guidance comes separately.

- **As a family:** `plugins/architecture/`, with the skill at `skills/performance/SKILL.md` (`architecture:performance`)
  and its references beside it.
- **Its declaration:**
  - `modes: [run]`;
  - `chapter: map.discover`, `after: ["dossier:discover"]`. Parts 1-5 are written by then, and `dossier:measure` comes
    later, in `map.measure`;
  - `produces: ["{dossier}/architecture-performance.md"]`;
  - `reuse_key: [identity, framework]`, so the research is not repeated every round;
  - `gpu: false`, and `disable-model-invocation: true`.
- **Its body** keeps the research procedure, and drops the placement prose: the declaration says where it runs.
- **A pack that wants it** lists `architecture` in `plugins:` and the skill in `map.discover`'s `run` list. Every other
  pack is unchanged.

## Skills for developing Arbor

Arbor's own repository has a fourth kind of skill, for people developing Arbor itself: the `develop` mode, in a
family of its own that a campaign never loads. This copy leaves that family out, since it is not edited here
(`VENDOR.md`).

## Landing it

- **Where:** a pull request against `main` on AMD-AGI/Arbor, one change per commit.
- **With it:** its docs, and tests where Arbor's code changes (`python3 -m pytest -q arbor/tests evals/tests`).
- **Say which packs** would turn it on, and what a campaign gains from it.
