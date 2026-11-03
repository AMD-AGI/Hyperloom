# Dossier template

A dossier is a directory in the project's repository,
`docs/dossier/<model>-<workload>-<gpu>/`: an index, `README.md`, and one file
per part, `NN-<part>.md`, plus `NN.uK-<part>.md` for each update round. Copy
the skeletons below.

## `README.md`, the index

```markdown
# Dossier: <model> on <workload>, <GPU>

| | |
| --- | --- |
| Project | <repository> at <commit> |
| Pins | framework <commit>; kernel library <commit>; image <digest>; harness <commit> |
| Platform | <cluster, node class, GPU and arch, ROCm and profiler SDK versions> |
| Workload | <benchmark, dataset snapshot, seed, concurrencies, duration> |
| Written | <date>, by <agent or person>; reviewed by <who, or not yet> |

## Parts

| # | Part | Status | Sources | Date | Stale when |
| --- | --- | --- | --- | --- | --- |
| 1 | [Model architecture](01-model.md) | written / partial / missing / not available | | | |
| 2 | [Framework and system](02-system.md) | | | | |
| 3 | [Code reading](03-code.md) | | | | |
| 4 | [Optimization target](04-target.md) | | | | |
| 5 | [Workload and CPU work](05-workload.md) | | | | |
| 6 | [Baseline and noise floor](06-baseline.md) | | | | |
| 7 | [Profile](07-profile.md) | | | | |
| 8 | [Kernel usage](08-kernels.md) | | | | |
| 9 | [Reference platform](09-reference.md) | optional | | | |
| 10 | [Earlier optimizations](10-history.md) | | | | |
| 11 | [Directions](11-directions.md) | | | | |
| 12 | [Fast bench](12-fast-bench.md) | non-blocking | | | |

"Stale when" names the pins, node class, workload or harness change that
would make the part out of date. After an update round (the skill's "Updating
the dossier"), a part's row links its newest round, `[Code reading](03.u1-code.md)`.

## Updates

| Round | Date | Why | Tree | Parts | Campaign stopped / restarted |
| --- | --- | --- | --- | --- | --- |
| u1 | <date> | <merge of which branch, or the KEEPs since the last round> | <commit> | 03, 06, 07, 10, 11 | <session, stopped at iteration N> / <new session, bar from 06.u1> |

## Open asks

- [ ] Reference-platform traces, or access to take them (part 9). Asked <date>; answer: <...>
- [ ] Where earlier optimizations are recorded (part 10). Asked <date>; answer: <...>
- [ ] The target, if not given (part 4).

## In short

<Five to ten lines, written last: where the time goes, the top directions with
their ceilings against the floor, and what the dossier could not establish.>
```

## `NN-<part>.md`, one per part

```markdown
# <N>. <Part>

**Question.** <the part's question, from the skill>

**Status.** <written / partial / missing / not available>, <date>. <What would
fill a gap, and who or what it waits on.>

<The contents the skill lists for this part. Each number with its source
(file, job, commit) and its label (node or host, GPU, seed, concurrency, n);
measured and inferred marked apart.>

**Sources.** <paths, jobs, commits, pull requests>

**Done when.** <the skill's check for this part, and whether it holds>
```

## `NN.uK-<part>.md`, one per part an update round rewrites

The same skeleton, with one line under the title:

```markdown
# <N>. <Part>, update round K

**Round K** of part N, for the tree at <commit> (<why: merge or KEEPs>).
It replaces `<previous file>` for this tree; that file stays, the record of
the tree before.
```

In `03.uK-code.md`, every in-path method of the model's code has a row:

```markdown
| Method (file:line) | Does | Costs (measured / reasoned) | Opportunity | Since part 3 |
| --- | --- | --- | --- | --- |
| `<Class>.<method>` (`<file>:<line>`) | <what it does here> | <cost, and how known> | none / candidate: <what could change, the mechanism, what it needs> | unchanged / changed by <KEEP or commit> / added |
```

`10.uK-lessons.md` covers each campaign since the last round. Its sources are
that campaign's `OUTCOMES.md` and `FINDINGS.md` from the `knowledge-collector`
skill, and its session:

```markdown
## <campaign>: <session>, iterations <first>-<last>, bar <a> -> best <b>

| Gated | Probes | KEEPs | KEEPs by level (config / tuning / framework / kernel) |
| ---: | ---: | ---: | --- |

### KEEPs
| Iteration | Changed | Gate gain | On this tree |
| --- | --- | --- | --- |
| <N> | <what> | <+x %> | in it / to re-land / superseded by <what> |

### Kills
| Lead | Iteration | Why it died | Rests on | Reopenable |
| --- | --- | --- | --- | --- |
| <id> | <N> | <reason> | <instrument, floor and tree; or "argument"> | no / yes: <which reason no longer holds, or why the evidence is weak> |

### What the probes established
- <fact that moves a ceiling or a cost, with its number and source iteration>

### The map against the work
- Worked: <last round's ranked directions, with iterations>
- Never touched: <directions>
- Off the map: <what, and whether it paid>

### Process lessons
- <harness or measurement failure, and the iterations it cost>
```

`11-directions.md` (and each `11.uK-directions.md`) ranks the two kinds
apart, and says whether the increments are running out:

```markdown
## Are the increments running out?
| Sign | Reading | Source |
| --- | --- | --- |
| Median of the last five KEEPs vs 3x the gate | <x %> vs <3g %> | <session, iterations> |
| Probes in a row that killed their own lead | <n> | <iterations> |
| Increments left with a ceiling of 3x the gate | <n> | <this table> |
| `barren_streak` vs its notice | <n> / 8 | <session.json> |
| A region well above its floor | <region, gap as % of step> | <07 / 07.uK> |

<Two signs and a region above its floor: rewrites rank first.>

## Increments
| Rank | Map ID | Direction | Component, kernel | Ceiling (target's unit) | vs floor | Depends on | Tried before |
| --- | --- | --- | --- | --- | --- | --- | --- |

## Rewrites
| Rank | Map ID | Region | Share of step | Floor | Gap | Plausible gain vs gate | Kernel plan |
| --- | --- | --- | --- | --- | --- | --- | --- |
| <N> | `m-<region>` | <ops, first launch to last> | <% per regime> | <µs, from what> | <µs, % of run> | <x % vs g %> | `11.uK-plan-<region>.md` |

## Big directions from the serving playbook
| Direction (big-directions.md) | Preconditions met here? | Regime fit (regime-tables.md) | In the image? (image-manifest.md) | Here as | Or set aside because |
| --- | --- | --- | --- | --- | --- |
| <name> | <yes / no: which one fails> | <favoured / neutral / against at this operating point, with the row> | <ships, ships off, absent> | <map ID> / — | <already on / below the floor: <x %> / image lacks it / regime against it> |

## Map
<The line an Arbor campaign reads (arbor/directions.py): every direction above, by its map ID, with its
status (open, done, killed, below_floor, or plan for a rewrite whose kernel plan is written) and, if closed,
why. Check it with `python3 -m arbor.directions check`.>

DIRECTIONS: [{"id": "m-<component>", "kind": "increment", "title": "<direction>", "rank": 1, "ceiling": "<x %>", "status": "open"}, ...]
```

