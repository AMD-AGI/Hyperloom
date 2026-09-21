---
name: review-pr
description: "Review a Hyperloom pull request. Invoke with a PR number whenever asked to review, re-review, or sanity-check a PR in this repository. Collects the PR into a scratch directory, selects from rules.md only the rules this diff can trigger, forces a semantic and core-file risk assessment before any finding is written, refutes every finding twice, and publishes an English conclusion to the PR. The output has two parts only: what the PR does, and blocking issues."
---

# Hyperloom PR review

Written for an agent to execute top to bottom. Every step writes an artifact into `$WORK`; a step
that leaves nothing behind is one a review can skip without the skip being visible. `$WORK` defaults
to `/tmp/hl-review-<PR>` and is printed by `fetch.sh`.

## Output contract

The published review has exactly two parts: what the PR does, and blocking issues. Nothing else.

Non-blocking observations are deleted, not softened. Not "note", not "FYI", not "non-blocking
but", not "for reference", not a postscript under the conclusion. If the only way to keep an
observation is to relabel it, it fails the bar and goes.

The bar is one question: **if this is not changed, is the current behaviour wrong?**

Blocking means one of two things, and the repo treats them as equal:

| Kind | What counts |
|---|---|
| Wrong behaviour | correctness, crash, data or precision error, compatibility break, security, performance regression |
| Desync | an operator-observable change the PR description never states; PR title or description does not match the diff at the current head |

If there are no blocking issues, say so explicitly. Do not manufacture small ones to fill space.

## Links out, not restatement

The authoring contract lives in the repo and is not copied here. Read and cite, do not restate:
`AGENTS.md`, `.github/copilot-instructions.md` (including its explicit do-not-flag list),
`docs/contributing/style-guide.md`, `.github/PULL_REQUEST_TEMPLATE.md`. Anything already gated by
ruff, pylint, mypy, CodeQL, gitleaks, bandit, REUSE, pre-commit or the coverage floor is out of
scope: re-flagging a gated rule is a defect in the review, not a finding.

---

## Step 0 — Scope

Before fetching anything, answer two questions from the title and body and hold them open:

1. What is this change for — what wrong behaviour or missing capability does it address?
2. By what route — which layer does it change, and is that the layer that owns the concern?

No artifact. A body that answers neither is already a signal: a description naming no mechanism
usually accompanies a diff nobody traced end to end.

## Step 1 — Fetch

```bash
bash .claude/skills/review-pr/fetch.sh <PR>
```

Keep the `$WORK` it prints. Read `diff.txt` and `body.txt` before going on.

| file | what it is for |
|---|---|
| `meta.txt` | number, title, author, state, head sha, base ref, url, mergeable |
| `title.txt` | the title, one line — rule X2 checks it against the diff |
| `body.txt` | the description, checked the same way |
| `diff.txt` | the full diff against the merge base |
| `files.txt` | changed paths, one per line — the input to rule selection |
| `numstat.txt` | added, deleted, path — size and shape of the change |
| `commits.txt` | commit subjects, oldest first; a late commit is where a description goes stale |
| `base.txt` | merge-base sha — every "is this pre-existing" question is answered against it |
| `ci.txt` | check runs at the current head: name, conclusion, url |
| `comments.txt` | existing review and issue comments — do not repeat a point already made |
| `testfiles.txt` | changed paths under a `tests/` directory |
| `docfiles.txt` | changed docs, prompts and `.md`/`.rst` paths — empty beside a `src/` change is X3's shape |
| `openprs.txt` | other open PRs touching the same files — conflicting in-flight work |

## Step 1b — Select the rules

Open the index at the top of [`rules.md`](rules.md) and take every row whose trigger matches
`files.txt` and a skim of `diff.txt`. Write the union of their rule ids into `$WORK/rules.txt`, one
per line, then read only those bodies. **Never read `rules.md` whole** — it holds 52 rules across 9
families, and a reviewer told to attend to all of them attends to none. Match rows generously: a row
you are unsure about is taken, never dropped. V1-V6 and X2 are on every list.

## Step 2 — Semantic understanding

Answer all five from the diff, not the description. One line each into `$WORK/answers.txt`,
prefixed `Q1:` … `Q5:`. An answer that anchors in nothing — no path, symbol or condition — is not
an answer; rewrite it before going on.

- **Q1 — State the root cause in one sentence.** Not the symptom, not a list of mitigations. A body
  listing three improvements instead of one causal change is the tell that nobody found the cause.
- **Q2 — Name the route.** Which module, which function, which call path now behaves differently,
  and why there rather than at the layer above or below.
- **Q3 — What does this make observable to an operator?** A behaviour, an interface, a default, a
  flag, an artifact path, a metric, a log line. If the answer is "nothing", say what makes that
  true — that claim is what the exemption in `AGENTS.md` rests on.
- **Q4 — Name every consumer of a changed value.** For each new or changed status string, enum
  member, field, keyword argument or return shape: the membership tests, the call sites, the
  readers of the persisted document, the prompt that names it. Grep, do not guess.
- **Q5 — What would it take for this change to be wrong?** Name the input, configuration, phase or
  race that would make it produce a wrong answer. "Nothing" is not an answer.

## Step 3 — Core-file risk

Write one line per backbone file the diff touches into `$WORK/core_files.txt`:

```
<path> TIER1|TIER2|TIER3 COVERED|GAP|N/A -- <reason naming what THIS PR changed>
```

Every changed non-test file under `src/` gets a line, Tier 3 included — check the list against
`files.txt`. Writing a tier lower than `references/tiers.md` gives it is not a route past the checks
the real tier requires.

`COVERED` = the blast radius is exercised by this PR's tests or is unreachable from the change.
`GAP` = it is not, and that goes on the card. `N/A` = the change cannot reach it. A reason that
names no file or symbol this PR changes is worthless — "core file, large blast radius" is equally
true of every PR ever opened against that file.

A docs-, CI- or test-only diff touches nothing under `src/`: write one
`NONE -- <reason naming what it does touch>` line.

Tiers come from [`references/tiers.md`](references/tiers.md): a table of the backbone files, and
Q1–Q4 for anything not in it, including new files. Q1b is the one that bites — `Coordinator`
resolves its 21 collaborators by string through a metaclass `__getattr__`, so a rename passes
every import check, passes lint, passes collection, and fails only hours into a session when
that phase is entered. Grep the string, not the symbol.

## Step 4 — Rule checklist

Adjudicate every rule id in `$WORK/rules.txt`. One line each into `$WORK/verdicts.txt`:

```
<RULE-ID> FIRE|CLEAR|N/A -- <reason naming file:line, symbol, or the condition>
```

`CLEAR` is a claim that you looked and it does not apply *to this diff*; the reason is what makes
it checkable. "ok", "n/a", "fine" are not reasons. Step 1b already cut the list to what this diff
can trigger, so there is no rule here you may pass over because the list looked long. A `FIRE`
must cite a file this PR changes; everything else — the header stating the contract, the prompt
naming the flag, the doc that did not move — is evidence and welcome beside it.

## Step 5 — AI-code diagnostic

A clean, well-written description is something AI produces easily; these are the structural places
a diff that reads well hides its defects. One line per check into `$WORK/ai_diagnostic.txt`:

```
<check>: CLEAN|HIT -- <what you looked at and what you found>
```

"clean" alone is not an answer; a reason that names nothing in the diff is not one either.

1. `wiring` — **both directions.** Every first-party import the diff adds resolves against the merge
   base. Every name-resolved entry (`_COLLAB_MODULES`, `KERNEL_REQUEST_HANDLERS`,
   `ACTION_CATALOGUE`, action-surface names) has a module and class that exist. Then the inverse:
   an added identifier whose head-tree occurrence count is 1 is a writer with no reader; a removed
   caller whose helper survives is a reader with no writer. Both are blocking.
2. `twins` — **Twin divergence.** Mirrored code half-adapted: sibling executors, per-framework patchers
   (vllm / sglang / atom / xdit), sync and async variants, the duplicated frozensets in
   `protocol/intent.py` and `agents/critic`. The defect is the asymmetry, not the copy.
3. `claims` — **Claim against code.** Does the code enforce what the description, the docstring, the
   comment and the prompt assert? Every number, flag name, path and default in the description must
   be greppable in the head tree. A number you cannot trace to an output is `[unverified]` and is
   never repeated as fact.
4. `silent-failure` — **Safety theater and silent failure.** For each new guard, `try`, `except` or default: is it
   reachable, will it ever fire, does it convert a failed measurement, a failed patch apply or a
   failed artifact write into a plausible-looking value? A fail-closed gate that coalesces a
   missing input to a passing default is the blocking shape.
5. `test-falsifies` — **Test calibrated to pass, not to falsify.** Does the test drive the production entry point or a
   mock of it? Does it assert behaviour or restate the implementation? Would it fail on the merge
   base? A regression test that passes on base pins nothing.
6. `constants` — **Unjustified constants and unbounded lifetimes.** A new timeout, budget, deadline or threshold
   literal: finite, positive, single-sourced, and derived from somewhere. A `create_task` whose
   handle is not stored, an `await`-less blocking call inside `async def`, a subprocess with no
   timeout and no kill path.

## Step 6 — Free-form pass, then the blind-spot line

Read the diff as someone who knows this system. Does the approach belong at this layer? Any
correctness risk the rules missed — a phase entered twice, a resume landing on a state shape the
new code cannot read, a destructive sweep scoped by pattern rather than by what this run owns, a
number compared against one another measurement system produced?

Then answer this in full, appended to `$WORK/answers.txt` as a `BLIND:` line: **"Is there any
correctness risk, resource hazard, or behavioural edge case in this diff that none of Steps 1-5
caught?"** A bare "no" is not an answer — say what you looked for and did not find. Anything found
after this point goes on the card marked `-- late finding` rather than back into a finished
artifact, so that the order the review ran in stays legible.

## Step 7 — Refutation

Try to kill each candidate finding before it reaches the card. One block per finding in
`$WORK/refutations.txt`:

```
FINDING: <RULE-ID|free:slug>
ATTEMPT: <what you opened, ran or compared to kill it -- must name a file, symbol or command>
OUTCOME: SURVIVES|DROPPED -- <what that showed>
```

The key is the same one the card will use, and every `FIRE` in `verdicts.txt` needs a block. An
`ATTEMPT` that names nothing openable is not a refutation: nobody can repeat it. With nothing to
refute, write a single `FINDING: none -- <reason>` line; an empty file reads the same as a skipped
step.

Attack in this order: is the line added by this PR or pre-existing context around an added line
(compare against `base.txt`); does the symbol resolve somewhere the diff did not show; is the
trigger reachable in a real configuration; is the point already made in `comments.txt`.

Then hand `card.md`, the diff and `base.txt` to a reader who has not seen your reasoning — a second
agent or a person — with every finding false until defended. `$WORK/independent.txt` takes the same
`FINDING` / `ATTEMPT` / `OUTCOME` blocks, one per finding that survived above. With no such reader,
write `FINDING: none -- no independent reader available` and say so on the card.

## Step 8 — Verdict

Before writing the card, walk `verdicts.txt` in both directions: nothing goes on the card that is
anchored in no changed file, and nothing marked `FIRE` quietly vanishes from it — report it, change
the verdict, or write `-- not reported: <reason>`.

### Card contract

The bracketed key joins the finding back to its `FIRE` line in `verdicts.txt` and to its surviving
blocks in `refutations.txt` and `independent.txt`. A finding with no key cannot be cross-checked.

```
## PR #NNN -- <title>

What it does: <one or two sentences: the wrong behaviour or missing capability, and the route
taken. Written for someone who has not read the diff.>

Blocking issues: <N | none>

1. [<RULE-ID|free:slug>] <one-line summary> [verified|inferred]
   Problem: <what is wrong> path/to/file.py:123
   Impact: <what it costs at runtime>
   Action: <verb> <what to change>
2. ...

deferred: <key> -- <reason>

Checked: <paths and symbols read> | Ran: <commands> | Base: <merge-base sha> | Head: <head sha>
```

At most 5 findings, ranked most-severe first by (severity, then blast radius). A sixth that
survived both refutation passes is not dropped in silence — it goes on a `deferred:` line, which is
the only thing that may stand in place of reporting it.

Use `free:<slug>` as the key for a Step 6 free-form finding; it is held to the same two refutation
passes as a rule finding.

A clean review writes `Blocking issues: none` and no numbered findings — but the
surviving-but-unreported scan still runs, so `none` cannot be used to bury a `FIRE` that survived
refutation.

### Evidence threshold

- `[verified]` — you opened the code, followed the chain, and can name the input, configuration or
  phase that triggers it.
- `[inferred]` — plausible, not confirmed. An inferred finding is not blocking: verify it or delete
  it, do not ship it as a question at the bottom of the card.

A finding with no `path:line` does not go on the card, and one that stops at "likely the root
cause" with no chain is not shippable.

### CI failure classification

Read `ci.txt` before blaming the PR:

- Which sha did each check run against? A success attached to an older head does not cover commits
  pushed afterwards. A narrow subset of workflows reporting while the diff touches runtime logic
  means the full matrix has not run.
- Does the same job fail the same way on `main` in the same window? Then it is baseline, not a
  regression introduced here.
- Infra shapes — runner acquisition, artifact upload, dependency resolution, expired logs — are
  flakes: ask for a rerun rather than quoting them as failures.
- `mergeable: CONFLICTING` in `meta.txt` is blocking on its own; ask for the rebase.

### SKIPPED protocol

When a step genuinely cannot run — no network for `openprs.txt`, no second reader for Step 7, a
diff too large to read — write `SKIPPED: <step> -- <why, and what is therefore unchecked>` into
that step's artifact and carry the same line onto the card. A `SKIPPED` artifact means that axis
was **not checked**; silence there is never read as clean. Skipping a step without recording it is
itself a review defect.

## Publishing

Post the conclusion to the PR as an English comment. No Chinese, no emoji, no signature.

```bash
# no blocking issues
gh pr review <n> --approve --body "<conclusion>"
# blocking issues
gh pr comment <n> --body "<conclusion>"
```

Approve and comment go together in one call when there is nothing blocking; do not ask first. With
blocking issues, comment only — no approve, no request-changes. Never merge a PR. The no-blocking
comment still carries the `Checked:` / `Ran:` line so the author sees the depth of the review
rather than a bare LGTM; it has two parts and stops there.

## Verifying a sub-agent's finding

A finding produced by a sub-agent is a hypothesis. Before publishing it, do all four:

1. Open the cited code and read it. Confirm the described logic is what is there.
2. Reproduce with the smallest possible run — call the changed function directly, or run the single
   test that covers it. Never the full local suite; CI runs that.
3. Diff against `base.txt` to confirm this PR introduced it.
4. Judge whether the trigger is reachable in a real configuration, not only in a constructed one.

A finding that fails any of these is deleted, not downgraded. Report one verified finding rather
than five unconfirmed ones. Nothing in the output says "a sub-agent found" — you own every line.

## Adding a new rule

When a human reviewer catches something this skill missed:

1. Add the rule body to `rules.md` under its family, with the real PR it was learned from quoted as
   evidence. A rule with no PR behind it is a hypothetical and does not go in.
2. Put its id on an index row whose trigger a reviewer can match against `files.txt` and the diff,
   or add a row. A rule on no row is never read.
3. Write the false-positive self-check into the body: the condition under which the shape is
   correct and the rule must stay silent.
4. Commit as `review-pr: add <ID> from #<NNN> -- <one line>`.

**Nothing new goes in this file.** SKILL.md is budgeted at 350 lines and holds only what every
review needs. Past that length a skill stops being read in full, and an unread instruction is worth
less than no instruction because it looks like coverage. Conditional content goes in `rules.md` and
reaches the reviewer through the index; lookup tables go in `references/` and are opened at the step
that needs them; executable content goes in a script and gets called. Raising the budget is a
decision to make this file less likely to be read — make it in a commit that says why.
