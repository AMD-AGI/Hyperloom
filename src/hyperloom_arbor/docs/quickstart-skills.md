# Quickstart: create a skill and integrate it with Arbor

From an idea to a skill a campaign runs: where it takes part, how to lay it out and declare it, how to check it, load
it in a pack and see it run. How skills fit into a campaign is in [skill-integration.md](skill-integration.md); every
key is in [skills.md](skills.md).

| Step | Command |
| --- | --- |
| Let the dev skills guide you (optional) | `python3 -m arbor.skills plugin arbor-dev`, then the `claude --plugin-dir` line it prints |
| Check every skill and family | `python3 -m arbor.skillcheck` |
| Check a pack that loads it | `python3 -m arbor run --config <config.json> --dry-run` |
| Run the tests | `python3 -m pytest -q arbor/tests evals/tests` |
| Settle a conflict | `python3 -m arbor conflicts --session <session> show\|pick\|merge` |

## 1. The environment

- **An Arbor checkout on a branch** ([quickstart-master-worker.md](quickstart-master-worker.md), 1.2):

  ```bash
  git clone <this repository's URL> Arbor && cd Arbor && git switch -c <your-branch>
  ```

- **Python 3.11 or newer** with the driver's packages and the test runner:

  ```bash
  python3 -m pip install pyyaml claude-agent-sdk pytest
  ```

- **Claude Code,** to load the dev skill (section 10) and to try a skill by hand.
- **Linux or WSL,** and LF line endings: a family's `.gitattributes` keeps them.
- **To watch it run in a campaign:** the master's environment, sections 1.3 to 1.8 of
  [quickstart-master-worker.md](quickstart-master-worker.md).

## 2. Decide where it takes part

Answer in order; the first yes is its place.

| Question | Then it is | Example |
| --- | --- | --- |
| Does it write a part of the dossier, a file every later reader should find with the others? | a dossier part: change that part's writer, or add a skill to the `dossier` family (section 5) | `dossier:measure` writes parts 6-8 |
| Does it produce something a later chapter needs, from evidence, once per campaign or per round? | a **run** skill, in its chapter | `architecture:performance` in `map.discover` |
| Should it happen when something happens, beside the session? | a **trigger** skill, on an event | `worker-summary` on `stretch_end` |
| Is it know-how or a procedure an agent should pick while it works? | a **use** skill, offered in its chapters | `dossier:read` in `iterate` |
| Is it for people working on Arbor, never a campaign? | a **development** skill | Arbor's own repository |

- **The chapters, in order:** `review`, `init`, `benchmark`, `accuracy`, `map.discover`, `map.measure`,
  `map.directions`, `iterate`, `round_end`, `end`.
- **The events fired today:** `session_start`, `iteration_start`, `keep`, `revert`, `probe`, `invalid`,
  `agent_failed`, `map_set`, `round_end`, `stretch_end`, `end`. Six more are named but not fired yet
  (`arbor/triggers.py`).
- **Never place a skill by prose.** "Run this after parts 1-5", written in another skill's body, is a wish nothing
  enforces. The header is what Arbor places, orders, checks and records.
- **Knowledge rather than a procedure** (reference pages, a knowledge base someone contributes)? It is registered as
  a source in `arbor/knowledge/sources.json` and searched with `python3 -m arbor.knowledge find|get|ask`. That search
  is not wired into campaigns yet, so meanwhile agents get the same pages through a use skill.
  `arbor-dev:integrate-knowledge` walks through it (section 10).

## 3. Lay it out

Contributed skills come as a family:

```bash
F=.claude/skills/<family>
mkdir -p "$F/<name>"
printf '* text=auto eol=lf\n' > "$F/.gitattributes"
cat > "$F/family.json" <<'EOF'
{"name": "<family>", "version": "1.0.0", "description": "<what the family is for>", "author": {"name": "<you>"}}
EOF
```

```
.claude/skills/<family>/
  family.json          # "name" equals the folder's name
  .gitattributes
  shared/              # optional: files several of its skills read; no SKILL.md
  <name>/SKILL.md      # one folder per skill: <family>:<name>
  <name>/<detail>.md   # optional: detail the SKILL.md links by relative path
```

- **Names resolve by the colon:** `<family>:<name>` is `.claude/skills/<family>/<name>/SKILL.md`.
- **A new family is listed by name in a test:** add it to the set in `arbor/tests/test_skill_families.py`
  (`test_the_repository_s_families_live_under_claude_skills`), or the suite fails.
- **How the agent gets it:** for each family a pack loads, Arbor builds a Claude Code plugin under
  `<session>/skills/plugins/<family>/`. `family.json` becomes its manifest, and the skills keep their names and links.
- **Nothing nests below `<name>/` but files.**
- **Moving a flat skill into a family:** `git mv` it, so no copy stays in `.claude/skills/<name>/` to load as a second
  skill.

## 4. Write the header and the body

### 4.1 A run skill

```yaml
---
name: <name>
description: "<what it does and when to use it>"
disable-model-invocation: true
metadata:
  arbor:
    modes: [run]
    chapter: map.discover
    after: ["dossier:discover"]
    produces: ["{dossier}/<name>.md"]
    reuse_key: [identity, framework]
    gpu: false
---

In a campaign, Arbor runs this skill in a turn of its own. Its argument is the path of a brief: read the brief first.
...
End your reply with one line per file written:
LEDGER: {"id":"<family>-<name>","type":"FACT","claim":"<what the file holds>","evidence":"<the file>"}
```

- **`chapter`, `after`, `before`** place it and order it among the chapter's run skills.
- **`produces`:** every file it must leave. Arbor checks them after the turn. A missing one stops the chapter, and the
  next turn takes it up again.
- **`reuse_key`:** the config keys whose values decide whether its files still hold. A run with the same values and
  its files present is skipped. With no `reuse_key`, it runs every time its chapter does.
- **The placeholders in `produces`:**

  | Placeholder | Is |
  | --- | --- |
  | `{dossier}` | `docs/dossier/<the config's dossier, else the identity as a slug>/` |
  | `{benchmark}` | `benchmarks/<the config's benchmark, else the pack's name>/` |
  | `{session}` | the session directory |
  | `{workspace}` | the served tree |

### 4.2 A trigger skill

```yaml
metadata:
  arbor:
    modes: [trigger]
    triggers: [keep]
    blocking: false
```

- **Its brief** names the event, the session and the journal, `<session>/lifecycle.jsonl`, where the event's entry is.
- **It never changes the served tree.**
- **The session goes on beside it,** unless it is `blocking: true`. A blocking skill gets 30 minutes at most, and no
  more than the time the run has left.

### 4.3 A use skill

```yaml
metadata:
  arbor:
    modes: [use]
    chapters: [iterate]
```

- **Its description is all an agent chooses it by.** Say what it is for and when to use it, and quote it: an unquoted
  `: ` breaks the YAML.
- **Nothing checks that it was used.** In one run, the iterating agents never called `dossier:read`; they read the
  dossier's files directly. Write the description for the moment the agent needs it.

### 4.4 For every skill

- **`name`** equals its folder's name.
- **The description:** Claude Code cuts it past 1,536 characters.
- **The body:** under 500 lines. Move detail to linked files, and a family's shared files to `../shared/<file>`.

## 5. When it writes a dossier part

Only for the first row of section 2:

1. **`arbor/dossier/parts.py`:** add the `Part`: number, name, title, writer, `gpu`, and `required=False` when a
   dossier is whole without it.
2. **The writer's `produces`** lists the part's file (`{dossier}/NN-<name>.md`). A test holds the two equal.
3. **`.claude/skills/dossier/shared/template.md`:** the index's row for the part.
4. **`.claude/skills/dossier/read/SKILL.md`:** which skill writes which part.
5. **An update round that rewrites it:** `ROUND_NAMES` in `parts.py`, and `dossier:update`'s body.

## 6. Check it

```bash
python3 -m arbor.skillcheck                       # errors fail; warnings are worth reading
python3 -m arbor run --config <config.json> --dry-run         # a pack that loads it validates
python3 -m arbor.dossier.check docs/dossier/<name>   # a dossier part's writer only
```

- **`arbor.skillcheck` refuses:**
  - front matter that is not YAML, a name that is not its folder's, no description;
  - unknown keys, types, modes, chapters or events;
  - a run skill without a chapter, and a placement key without its mode;
  - a bad `family.json`, or a family that mixes development and runtime skills.
- **It warns** on a body over 500 lines, a description Claude Code would cut, and a run skill agents could pick (no
  `disable-model-invocation`).
- **The test suite runs it over the repository,** so a skill that fails it fails the tests.

## 7. Load it in a pack

Nothing turns a skill on by itself. A pack loads it in `pack.yaml`; its header does the rest:

```yaml
lifecycle:
  load: ["benchmark:study", "dossier:discover", "<family>:<name>", "dossier:measure",
         "dossier:directions", "dossier:update", "dossier:read", profiler, worker-summary]
```

- **Order in `load`** matters only between run skills of one chapter that name no `after` or `before` between them.
- **`validate_pack` refuses** a skill the tree does not hold, a development skill, a loaded skill that fails its own
  check, and an `after`/`before` cycle.
- **Change the serving template** (`packs/serving-m10`) only when every serving campaign should get it. Otherwise load
  it in your own pack.
- **Under a host,** `component.lifecycle.load` replaces the list.

## 8. See it run

- **Without a GPU,** a test with a stub agent: `arbor.chapters.run_turns` writes a run skill's brief and checks its
  files, as `arbor/tests/test_chapters.py` does.
- **In a campaign,** start the master on a pack that loads it ([quickstart-master-worker.md](quickstart-master-worker.md),
  section 3). The setup and map chapters run their run skills. Then look in the journal:

  ```bash
  grep '"<family>:<name>"' <session>/lifecycle.jsonl
  ```

  | Mode | What the journal holds |
  | --- | --- |
  | run | `{"kind": "run", "name": "<family>:<name>", "chapter": "...", "status": "done", "missing": []}` |
  | trigger | `{"kind": "trigger", "name": "<name>", "event": "...", "status": "ran"}` (or `skipped`, `timed_out`) |
  | use | the chapter's entry lists it: `{"kind": "chapter", "name": "iterate", "skills": [...]}`. The log says `[arbor] Chapter iterate: the agent's skills [...]` |

- **The files it produces** are where its placeholders point: `docs/dossier/<name>/`, `benchmarks/<name>/` or the
  session.

## 9. When two skills conflict

Two loaded skills conflict when their `produces` share a file, or when one names the other in `overlaps`.

- **Standalone, the session pauses** before anything runs, and writes `<session>/CONFLICTS.md`:

  ```bash
  python3 -m arbor conflicts --session <session> show
  python3 -m arbor conflicts --session <session> pick <id> <skill>             # keep one, drop the other
  python3 -m arbor conflicts --session <session> merge <id> [--keep <skill>]   # an agent merges the two
  ```

- **Without a pause:** `"skill_conflicts": "merge"` in the config (or `--skill-conflicts merge`) lets an agent merge
  them.
- **Under a host,** a conflict refuses the start unless merging is set.
- **Best:** give your skill files of its own, and name in `overlaps` what it overlaps in ways files cannot show.

## 10. Let the dev skill do it

Arbor ships two development skills, in the `arbor-dev` family, that walk a coding agent through the work:

```bash
python3 -m arbor.skills plugin arbor-dev     # prints: claude --plugin-dir <dir>
claude --plugin-dir <dir>
```

| Skill | For | Ask it |
| --- | --- | --- |
| Arbor's own repository | a skill or a family: sections 2 to 9 of this page | `/arbor-dev:integrate-skill <what your skill does, and where its files are>` |
| `arbor-dev:integrate-knowledge` | a knowledge source: where it belongs, pages a search can cite, its registry entry, corrections, golden questions, tests | `/arbor-dev:integrate-knowledge <what the source is, where it lives, its licence>` |

A campaign never loads them. The same command builds any family, to try its skills by hand.

## 11. Land it

- **A pull request against `main`** in this repository, one change per commit.
- **With it:** its docs, and a test for any change to Arbor's code
  (`python3 -m pytest -q arbor/tests evals/tests`).
- **A family's own test:** copy `arbor/tests/test_architecture_skill.py` for a run skill. Its three tests: the skill
  checks clean in its mode; a pack that loads it runs it in its place, whatever the load order; it conflicts with
  none of the dossier's skills. Add the family's name to `test_skill_families.py`.
- **Keep internal names out:** runtime skills ship with the copy of Arbor in public Hyperloom (`src/hyperloom_arbor`),
  so a skill names no internal host, path, account or URL.
- **Say** which packs would load it, and what a campaign gains, measured if it was run.
- **A worked example:** `architecture:performance`, contributed as a pull request and integrated as a run skill
  in `map.discover` ([skills.md](skills.md#worked-example-architecture-research-before-profiling)).

## 12. When something goes wrong

| What you see | Why | What to do |
| --- | --- | --- |
| `skillcheck`: the name is not its folder's | `name:` differs from the folder | rename one of them |
| `test_skill_families.py` fails on the set of families | a new family is not in its list | add the family's name there |
| the description is empty, or the YAML fails | an unquoted `: ` in the description | quote the description |
| `validate_pack`: the tree does not hold the skill | a typo, or a family without `family.json` | check the name and `family.json` |
| a family's skills do not show up in a plain `claude` session | Claude Code finds skills one folder deep only | Arbor builds families as plugins; by hand, use `arbor.skills plugin <family>` |
| the chapter stops after your run skill | a file in `produces` was not written | write every file it declares, or declare only what it writes |
| your run skill runs again every round | no `reuse_key` | add the config keys its work depends on |
| the session pauses at start with `CONFLICTS.md` | another loaded skill writes the same file | `arbor conflicts ... pick` or `merge` (section 9) |
| the agent never uses your use skill | its description does not match the moment, or `chapters` leaves out that chapter | rewrite the description; check `chapters` |
