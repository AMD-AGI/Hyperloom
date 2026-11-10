# Subtraction

Read from Step 5's `test-falsifies` and Step 6 of [`../SKILL.md`](../SKILL.md). The card tells the author exactly
what to change and, of two changes that both remove a defect, names the one leaving less code. Answering each finding
with one more guard grows the tree without converging (`AGENTS.md` *Review feedback is a hypothesis*); the correct
implementation is usually smaller than the diff.

## Tests: does each need to exist

Name the contract each added or changed test pins: the entry point it drives and the behaviour a caller outside it
observes (`AGENTS.md` *Test the contract, not the plumbing*; `docs/contributing/style-guide.md` *What to test*).
Three shapes are deletions:

- **Repeat** — another test, named as `file::test`, already asserts the same behaviour through the same or a higher entry point
  on input of the same shape; the higher one stays.
- **Plumbing pin** — asserts call order, mock arguments or counts, a private attribute or a private helper's return
  shape, for code with no logic of its own; it fails on the next refactor and on nothing else.
- **Layered pin** — one contract pinned again at each layer below the entry point (CLI, handler, helper), each
  defending against the others; the one at the boundary the contract is exported from stays.

Not a deletion when any one holds: the helper carries real logic of its own, the test is the one failing on the
merge base for this defect (T4), or it pins a failure mode the higher test cannot reach. Deleting a test never lowers
what is pinned: name the test that keeps the contract (T2).

## The change: build it again

`BUILD:` covers the existing mechanisms the diff patches as well as those it adds: a fix extending a mechanism that
should not exist is answered by deleting the mechanism. On a fix, take the first route that removes the cause:
(1) delete the code producing the wrong behaviour; (2) change it in place, at the producer every reader shares (D11);
(3) only then add, and only what the cause needs. A guard, retry, fallback or second resolution path added while the
cause stays is the shape to catch (D13, A13). Ask what the fix would need if the condition could not arise; when a
deletion makes it unable to arise, the `Action:` says delete and names the lines.

## Dead ends the change touches

Search only what the diff touches — the functions and modules it edits, their direct callers and callees, and those
modules' test files — for:

- a branch the change makes unreachable, or a fallback whose trigger can no longer occur;
- a parameter, field, flag, return value or status no caller reads;
- a helper referenced only by its own tests;
- a test for behaviour the tree no longer has, or a repeat, plumbing pin or layered pin in a touched test file.

Evidence is tree-wide, not frame-limited (V7, V8): `git grep -n '<name>'` in the head tree, plus entries resolved by
name (dispatch tables, `-m` targets, registries, prompts, readers of persisted data). A search finding nothing is not
evidence that an entry point is unused (`AGENTS.md` *Finish the replacement*): a CLI flag, public function, persisted field,
or anything read outside this repo is dead only once that consumer is named and gone.

## Blocking or trim

- **The diff adds it** — a finding, held to both refutation passes:
  - an added repeat, plumbing pin or layered pin: `Contract violation`, *Test the contract, not the plumbing*;
  - an added guard, fallback or second path where deleting or changing existing code removes the cause:
    `Contract violation`, *Simplify by removing a mechanism* or *Trust the caller*, or D13, A13 or D11 where one fits;
  - code the diff leaves dead: C3, *Finish the replacement*.
- **It was there before, inside what the diff touches** — a `TRIM:` line in `$WORK/answers.txt`, not blocking; the
  author decides whether this PR takes it:

  ```
  TRIM: <path>:<line>[-<line>] -- <what to delete> -- <evidence: the command and what it returned>
  ```

  With nothing to trim, one `TRIM: none -- <what you searched, and where>`. A line that would add code — a test to
  write, a guard to add, a rename — is a non-blocking note and is deleted.

## On the card

`Trim in this PR:` carries up to five `TRIM:` lines, most lines removed first, as
`- <path>:<line>[-<line>] -- <what to delete> -- <evidence>`, between the findings and `Checked:`. It is omitted when
empty, never counts toward `Blocking issues`, takes no `deferred:` line, and a re-review writes it afresh rather than
settling it in `prior_verdicts.txt`.
