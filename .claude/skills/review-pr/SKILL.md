---
name: review-pr
description: "Review a Hyperloom pull request. Invoke with a PR number whenever asked to review, re-review or sanity-check a PR in this repository. Collects the PR into a scratch directory, reads only the rules.md rules the diff can trigger, forces a semantic and core-file risk assessment before any finding is written, refutes every finding twice, and publishes an English card of two parts only — what the PR does, and its blocking issues — plus the PR's risk, the decisions a person must make, and the code the author can delete in the same PR."
---

# Hyperloom PR review

An agent executes this top to bottom. Every step writes an artifact into `$WORK` (default `/tmp/hl-review-<PR>`;
keep the one `fetch.sh` prints): a step that leaves nothing behind can be skipped unseen.

## Output contract

The card has two parts, what the PR does and its blocking issues; beside them stand only the PR's risk, the decisions
a person must make (Step 3, Step 8) and Step 6's deletions — none of them a finding. Nothing else. Non-blocking
observations are deleted, not softened into a "note", "FYI", "non-blocking but", "for reference" or a postscript
under the conclusion: one that survives only by relabelling fails the bar. The bar is two questions: **if this is not
changed, is the current behaviour wrong?** and **was the reason for the change true, and does the change end the
failure it names?** Blocking means one of four kinds, equal in weight:

| Kind | What counts |
|---|---|
| Wrong behaviour | correctness, crash, data or precision error, compatibility break, security, performance regression |
| Ineffective | the premise does not reproduce at the merge base, or the failure the PR names still happens at the head, or a check is dropped as covered by something that decides less — rules E1-E5 |
| Desync | an operator-observable change the description never states; a title or description that does not match the diff at the current head |
| Contract violation | the diff breaks a rule `AGENTS.md` states outright; the finding names the bullet and, for an ownership bullet, the module in this repo that already owns the concern |

The second kind is the one a correct diff passes: right line by line, tested, accurately described, and still
resting on a measurement that does not reproduce, or leaving the reported failure happening. E1-E5 each name the tool
to re-run, the branch to trace or the predicate to read; an E finding that names none of them is taste and goes.

The fourth kind is narrow. "Could be simpler", "I would put it elsewhere", "extract a helper" are taste and go. It
survives only where the repo has already decided — a detour around an existing path, a concern implemented outside
its owner — in writing both sides can read; a finding that cannot name the bullet and the owner is taste wearing a
citation. With no blocking issue, say so explicitly; never manufacture small ones.

Read and cite, never restate, the authoring contract: `AGENTS.md`, `.github/copilot-instructions.md` (with its
do-not-flag list), `docs/contributing/style-guide.md`, `.github/PULL_REQUEST_TEMPLATE.md`. Whatever ruff, pylint,
mypy, CodeQL, gitleaks, bandit, REUSE, pre-commit or the coverage floor gates is out of scope: re-flagging it is a
review defect.

## Step 1 — Fetch

```bash
bash .claude/skills/review-pr/fetch.sh <PR>
```

Read `diff.txt`, `body.txt` and `comments.txt` first. When `comments.txt` holds an earlier card for this PR, or a
human's blocking objection the author has not answered in code, settle those first and still review the whole diff,
as [`references/re-review.md`](references/re-review.md) says.

| file | for |
|---|---|
| `meta.txt` | number, title, author, state, head sha, base ref, url, mergeable |
| `title.txt`, `body.txt` | the title and the description; X2 checks both against the diff |
| `diff.txt`, `files.txt` | the full diff against the merge base; its paths, one per line, the input to rule selection |
| `numstat.txt`, `netlines.txt` | added, deleted, path; and E3's production total, `src/` outside `tests/`, binary rows excluded |
| `commits.txt` | subjects, oldest first; a late commit is where a description goes stale |
| `base.txt` | merge-base sha; every "is this pre-existing" question is answered against it |
| `ci.txt` | check runs at the current head: name, conclusion, url |
| `comments.txt` | review and issue comments, each headed by author and time; an earlier card is settled (re-review.md), any other point already made is not repeated |
| `testfiles.txt`, `docfiles.txt` | changed paths under a `tests/` directory; changed docs, prompts, `.md`/`.rst` — an empty `docfiles.txt` beside a `src/` change is X3's shape |
| `openprs.txt` | other open PRs touching the same files; a `# PARTIAL:` line names what was never compared, reported as `SKIPPED: V4` |
| `skill.txt` | this skill's version (its tree hash, `+dirty` with uncommitted edits) and the environment's model; the card's `Skill:` is the version, its `Model:` the model your system prompt names, else this one |
| `risk.txt` | the PR's risk, graded from its changed paths by `risk.py` with `references/tiers.md` as its data, then one line per path; the floor of the card's `Risk:` |

## Step 1b — Select the rules

Take every row of the index atop [`rules.md`](rules.md) whose trigger matches `files.txt`, a skim of `diff.txt`, or a
`label:` line in `meta.txt`; write the union of their ids into `$WORK/rules.txt`, one per line, and read only those
bodies. **Never read `rules.md` whole**: a reviewer told to attend to every rule attends to none. Match rows
generously: a row you are unsure about is taken, never dropped. V1-V8 and X2 are on every list, V2 and V7 without a
verdict line; on a re-review, so are the rules of earlier findings that still hold.

## Step 2 — Semantic understanding

Answer all six from the diff, not the description, one line each into `$WORK/answers.txt` as `Q1:` … `Q6:`. An answer that
anchors in no path, symbol or condition is rewritten before going on.

- **Q1 — Root cause in one sentence, and what pins it.** Not the symptom or a list of mitigations. What pins it is
  evidence outside the description — failing output, a session artifact, a measurement re-run at `base.txt`; sourced
  to the body alone it is `[unverified]`, and E1 decides what that costs. A body listing three improvements says
  nobody found the cause.
- **Q2 — The route, and the layer at each end.** Place two frames on this upstream-first vocabulary — `admission`
  (could refuse the input), `producer` (computes the value), `intermediate` (passes it on), `consumer` (acts on it):
  the frame this diff changes, and the earliest frame that could have refused the bad value. Both go on the card;
  strictly downstream with no hunk upstream is E4.
- **Q3 — What an operator now observes.** A behaviour, interface, default, flag, artifact path, metric, log line.
  "Nothing" needs its reason: the `AGENTS.md` exemption rests on it.
- **Q4 — Every consumer of a changed value.** For each new or changed status, enum member, field, keyword or return
  shape: membership tests, call sites, readers of the persisted document, prompts naming it. Grep, never guess.
- **Q5 — What would make it wrong.** The input, configuration, phase or race. "Nothing" is not an answer, nor is a
  differential run that never built the shapes whose early return the diff moved (E1).
- **Q6 — The scenario, and whether it ends.** The input, configuration or sequence that produces the failure, in
  actual values and order — "the interpreter already carries ray 2.44.1 with click 8.5.0 and pip cannot write to it",
  never "when the config is wrong". Trace the branch the head runs under exactly that and say what the outcome
  becomes; it goes on the card. Where the failure recurs the diff may still be worth landing, but it closes nothing:
  that is E2.

## Step 3 — Core-file risk

One line per changed non-test `src/` file, Tier 3 included — check the list against `files.txt` — into
`$WORK/core_files.txt`:

```
<path> TIER1|TIER2|TIER3 COVERED|GAP|N/A -- <reason naming what THIS PR changed>
```

Tiers come from [`references/tiers.md`](references/tiers.md): its table, then its Q1–Q4 for any other file, new ones
included. Q1b bites hardest: a `Coordinator` attribute rename passes imports, lint and collection and fails hours into
a session — grep the class name, not just the symbol. A lower tier than it gives is no route past the real tier's
checks. `COVERED`: this PR's tests exercise the blast radius, or the change cannot reach it. `GAP`: they do not, and
it goes on the card. `N/A`: the change cannot reach it. A reason naming nothing this PR changed is worthless — "core
file, large blast radius" is true of every PR against it. A docs-, CI- or test-only diff writes one
`NONE -- <reason naming what it does touch>` line.

The card's `Risk:` is `risk.txt`'s level, or higher when the diff does what paths cannot show — deletes or weakens a
test, changes a default, a persisted shape or a gate, widens what an agent may do — written `(raised from <level>)`.
Never lower: paths are the floor, and only a person changes what they mean (`references/tiers.md`).

## Step 4 — Rule checklist

One line per id in `rules.txt` into `$WORK/verdicts.txt`:

```
<RULE-ID> FIRE|CLEAR|N/A -- <reason naming file:line, symbol, or the condition>
```

`CLEAR` claims you looked and it does not apply to this diff; the reason makes that checkable ("ok", "n/a", "fine"
are not reasons). Step 1b already cut the list, so no rule is skipped for length. A `FIRE` must cite a file this PR
changes; the contract's header, the prompt naming a flag, the doc that did not move are evidence beside it.

No line goes to a rule whose `Severity:` is advisory in every shape it names, nor to one whose body says it cannot be
a finding — V2 is the output contract and V7 is Q4, each recorded by the step that runs it. The card cannot publish
an advisory rule, so adjudicate one only when it fires, as a note outside the card. Across 23 recorded reviews, 815 verdict lines
carried 12 `FIRE`s: spend the `CLEAR` reasons where a finding could come out.

## Step 5 — AI-code diagnostic

A clean description is cheap for an AI; these are where a diff that reads well hides defects. One line per check into
`$WORK/ai_diagnostic.txt` as `<check>: CLEAN|HIT|N/A -- <what you looked at and found>`; "clean" alone, or a reason
naming nothing in the diff, is no answer.

1. `claims` — **claim against code.** Does the code enforce what the description, docstring, comment and prompt
   assert? Every number, flag, path and default the description names must be greppable in the head tree; a number
   traced to no output is `[unverified]` and never repeated as fact. A command or artifact name the diff adds to an
   instruction — a rule's evidence step, a prompt, a runbook — is run here once and its output read: one naming the
   wrong artifact or a deprecated form prints nothing and fails silently forever.
2. `silent-failure` — **safety theater.** For each new guard, `try`, `except` or default: is it reachable, does it
   ever fire, does it turn a failed measurement, patch apply or artifact write into a plausible value? A fail-closed
   gate that coalesces a missing input to a passing default is the blocking shape.
3. `test-falsifies` — **calibrated to pass, not to falsify, or not needed.** Does the test drive the production entry
   point or a mock, assert behaviour or restate the implementation, fail on the merge base? A regression test passing
   on base pins nothing. A test that repeats another, pins plumbing, or re-pins a contract at each layer below it is a
   deletion ([`references/subtract.md`](references/subtract.md)).
4. `patch-shape` — **a fix that cannot reach its own cause.** Run on a fix PR as the card contract defines one,
   against Q2's two frames, each shape naming its file:line; elsewhere write `N/A`. Any hit is an E4 candidate, and
   the first two block on their own:
   - every production hunk lands in failure classification, naming or reporting — a diff whose whole footprint is
     what the failure is *called* cannot stop it happening (#2015);
   - the error class, message or identifier the body quotes appears in neither the changed files nor the code that
     writes it — then the diff cannot be on that path at all (#1981);
   - the new guard reads a value its own function did not produce, and no hunk touches it (#1982);
   - the permissive branch that produced the symptom survives unchanged, with the new branch added above it —
     narrowing a default is not changing it (#2099);
   - state is cleared at a reader's entry instead of where it is written or named (#2188);
   - the body or a new test concedes the rejected input is a reasonable thing to emit, and the diff rejects it
     anyway — the cause is a missing capability, not a bad caller (#1983).

## Step 6 — Free-form pass, then build-it-again

Read the diff as someone who knows this system. Does the approach belong at this layer? Any correctness risk the
rules missed — a phase entered twice, a resume landing on a state shape the new code cannot read, a destructive sweep
scoped by pattern rather than by what this run owns, a number compared against another measurement system's? Most
findings come from this step; the rule list prompts it and does not replace it.

Then append to `$WORK/answers.txt`:

- `BUILD:` **"If none of this code existed, what would you build — and does every mechanism this diff adds or patches
  need to exist?"** Name it, then the diff's parts it lacks. A part surviving only because it is written is what this
  catches; on a fix, so is a guard added where deleting or changing existing code removes the cause. A refactor
  settles it under E3 on the net line count; anything else carries an `AGENTS.md` bullet and an owner module, or
  stops here.
- `TRIM:` lines, as [`references/subtract.md`](references/subtract.md) says.

## Step 7 — Refutation

Try to kill each candidate before it reaches the card, one block per finding in `$WORK/refutations.txt`:

```
FINDING: <RULE-ID|free:slug>
ATTEMPT: <what you opened, ran or compared to kill it -- must name a file, symbol or command>
OUTCOME: SURVIVES|DROPPED -- <what that showed>
```

The key is the card's; every `FIRE` needs a block; an `ATTEMPT` naming nothing openable cannot be repeated and is no
refutation. With nothing to refute write a single `FINDING: none -- <reason>` line: an empty file reads as a skipped
step.

Attack in order: is the line added by this PR, or context around one (against `base.txt`)? Does the symbol resolve
where the diff did not show? Is the trigger reachable in a real configuration? Has `comments.txt` already made the
point? An earlier review-pr card does not count: its findings are settled in `prior_verdicts.txt`, kept if they still
hold and dropped if fixed.

Then hand the surviving findings as they will appear, the diff and `base.txt` to a reader who has not seen your
reasoning — a second agent or a person — with every finding false until defended; `$WORK/independent.txt` takes the
same blocks for each survivor. With no such reader write `FINDING: none -- no independent reader available` and say
so on the card.

## Step 8 — Verdict

Before writing the card, walk `verdicts.txt` both ways: nothing reaches the card anchored in no changed file, and no
`FIRE` vanishes from it — report it, change the verdict, or write `-- not reported: <reason>`.

### Card contract

```
## PR #NNN -- <title>

What it does: <one or two sentences for someone who has not read the diff: the wrong behaviour or missing
capability, and the route taken>

Scenario: <Q6's values and order, then what the head does under them. Fix PRs only.>
Layer: <Q2's two frames: the one this diff changes -> the earliest that could have refused.>

Risk: <low|medium|high>[ (raised from <level>)] -- <the path or change that sets it>
Decisions for a human:
- <paths> -- <what changes for an operator or a gate> -- <the question a person answers>

Blocking issues: <N | none>

1. [<RULE-ID|free:slug>] <one-line summary> [verified]
   Problem: <what is wrong> path/to/file.py:123
   Impact: <what it costs at runtime>
   Action: <verb> <what to change>
2. ...

deferred: <key> -- <reason>
Trim in this PR:
- path/to/other.py:40-58 -- <what to delete> -- <evidence>

Checked: <paths and symbols read> | Ran: <commands> | Skill: <skill> | Model: <model> | Prior: <KEY status, ...> | Base: <merge-base sha> | Head: <head sha>
```

- `Scenario:` and `Layer:` are written when `title.txt` carries a `fix` prefix or the body names a concrete failure —
  an error string, a session id, an RSI problem, `Fixes #N` — and only then; that is E2's trigger. "when the config is
  invalid" is not a scenario: if neither the body nor a surviving artifact pins it, write
  `Scenario: [unverified] -- <where you looked>`, E2's finding.
- `Decisions for a human:` is what a person must accept before merge: every operator-observable change from Q3, and
  every `high` path in `risk.txt`, named whole or under a directory written with its trailing `/`. A medium card with
  nothing to decide writes `Decisions for a human: none -- <why>`; a high card names at least one; a low card has
  none, since anything a person must decide makes it medium. Decisions never count toward `Blocking issues`.
- The key joins a finding to its `FIRE` in `verdicts.txt` and its blocks in `refutations.txt` and `independent.txt`;
  a finding with no key cannot be cross-checked. A Step 6 free-form finding is `free:<slug>`, held to the same two passes.
- At most 5 findings, most severe first (severity, then blast radius). A sixth that survived both refutation passes
  goes on a `deferred:` line, the only thing that may stand in for reporting it.
- A finding names one shape: `Problem:` lists every `path:line` where it occurs, and `Action:` covers them all,
  preferring to delete or change existing code over adding any.
- A clean card is `Blocking issues: none` with no numbered findings; the unreported-`FIRE` walk still runs, so `none`
  cannot bury a `FIRE` that survived refutation.
- `Prior` is written only on a re-review ([`references/re-review.md`](references/re-review.md)).
- `[verified]`: you opened the code, followed the chain, and can name the triggering input, configuration or phase;
  for an E1 or E2 evidence gap, naming where you looked — body, artifacts, commits — and what was not there verifies
  it. `[inferred]` — plausible, not confirmed — is not blocking: verify it or delete it; `check-card.sh`
  refuses a card whose blocking findings are not all `[verified]`. No `path:line`, or a chain that stops at "likely
  the root cause", keeps a finding off the card.
- Before quoting `ci.txt` or `meta.txt`'s `mergeable`, read [`references/ci.md`](references/ci.md).

Last, run `bash .claude/skills/review-pr/check-card.sh "$WORK" <PR> <head sha> "$WORK/risk.txt"` and fix the card
until it prints nothing. CI publishes only a card that passes the same check against a risk it grades from GitHub.

### SKIPPED protocol

A step that genuinely cannot run — no network for `openprs.txt`, no second reader, a diff too large to read — writes
`SKIPPED: <step> -- <why, and what is therefore unchecked>` into its artifact and onto the card. `SKIPPED` means not
checked; silence is never read as clean, and an unrecorded skip is itself a review defect.

## Publishing

Post `card.md` as `check-card.sh` passed it, in English: no Chinese, no emoji, no signature.

```bash
# blocking issues
gh pr review <n> --request-changes --body-file "$WORK/card.md"
# no blocking issues: first dismiss your own earlier CHANGES_REQUESTED review, then
gh pr comment <n> --body "$(cat "$WORK/card.md")"$'\n\nLGTM'
```

Never `gh pr review --approve`, never merge. A local run posts under your account, so an approval would be your
code-owner approval of a model's verdict; both belong to the human reviewer the author brings in after LGTM. The bot
never approves either, whatever the risk: `Risk:` and `Decisions for a human:` show that reviewer where to look. The
clean comment is the whole card, `Checked:` line included so the author sees its depth, then `LGTM`.

## Verifying a sub-agent's finding

A sub-agent's finding is a hypothesis. Before publishing: (1) open the cited code and confirm the logic is there;
(2) reproduce with the smallest run — call the changed function directly, or the one test covering it, never the
full local suite (CI runs that); (3) diff against `base.txt` to confirm this PR introduced it; (4) confirm the
trigger is reachable in a real configuration, not only a constructed one. Failing any deletes it, not downgrades it.
One verified finding beats five unconfirmed; nothing says "a sub-agent found" — you own every line.

**Nothing new goes in this file.** It is budgeted at 365 lines and holds only what every review needs: past that a
skill stops being read in full, and an unread instruction is worth less than none because it looks like coverage.
Conditional content goes in `rules.md` behind the index, lookup tables in `references/` opened at the step that needs
them, executable content in a script that gets called. Raising the budget makes this file less likely to be read —
do it in a commit that says why.
