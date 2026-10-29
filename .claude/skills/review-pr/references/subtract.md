# Subtraction

Consulted from Step 5's `test-falsifies` check and Step 6 of [`../SKILL.md`](../SKILL.md). The card
exists to tell the author exactly what to change, and of two changes that both remove a defect it
names the one that leaves less code. A fix that answers each finding with one more guard grows the
tree without converging (`AGENTS.md` *Review feedback is a hypothesis*); the correct implementation
is usually smaller than the diff, and this file is how the review finds it.

## Tests: does each one need to exist

For every test the diff adds or changes, name the contract it pins: the entry point it drives and
the behaviour a caller outside that entry point can observe (`AGENTS.md` *Test the contract, not the
plumbing*; `docs/contributing/style-guide.md` *What to test*). Then look for three shapes:

- **Repeat** — another test, named as `file::test`, already asserts the same behaviour through the
  same or a higher entry point on input of the same shape. The higher one stays.
- **Plumbing pin** — the test asserts call order, mock arguments or call counts, a private attribute
  or a private helper's return shape, for code with no logic of its own. It fails on the next
  refactor and on nothing else.
- **Layered pin** — one contract pinned again at each layer below the entry point (the CLI, the
  handler, the helper), each a defence against the others failing. The one at the boundary the
  contract is exported from stays.

Not a deletion: the helper carries real logic of its own, the test is the one that fails on the
merge base for this defect (T4), or it pins a failure mode the higher test cannot reach. Deleting a
test never lowers what is pinned: name the test that keeps the contract (T2).

## The change: build it again

The `BUILD:` question covers the existing mechanisms the diff patches as well as the ones it adds:
a fix that extends a mechanism which should not exist is answered by deleting the mechanism.

On a fix, rank the routes and take the first one that removes the cause:

1. Delete the code that produces the wrong behaviour.
2. Change it in place, at the producer every reader shares (D11).
3. Only then add, and only what the cause needs.

A fix that adds a guard, retry, fallback or second resolution path while the cause stays is the
shape to catch (D13, A13). Ask what the fix would need if the condition could not arise; when
deleting something makes it unable to arise, the finding's `Action:` says delete it and names the
lines.

## Dead ends the change touches

Search only what the diff touches: the functions and modules it edits, their direct callers and
callees, and the test files of those modules. Look for:

- a branch the diff's own change makes unreachable, or a fallback whose trigger can no longer occur;
- a parameter, field, flag, return value or status no caller reads;
- a helper whose only remaining references are its own tests;
- a test for behaviour the tree no longer has, or a repeat, plumbing pin or layered pin in a touched
  test file.

The evidence is tree-wide, not frame-limited (V7, V8): `git grep -n '<name>'` in the head tree, plus
the entries resolved by name (dispatch tables, `-m` targets, registries, prompts, readers of
persisted data). A search that found nothing is not evidence that an entry point is unused
(`AGENTS.md` *Finish the replacement*): a CLI flag, a public function, a persisted field, or
anything a consumer outside this repo reads is a dead end only once that consumer is named and gone.

## Blocking or trim

- **The diff adds it** — a finding, held to both refutation passes like any other:
  - an added repeat, plumbing pin or layered pin: `Contract violation`, *Test the contract, not the
    plumbing*;
  - an added guard, fallback or second path where deleting or changing existing code removes the
    cause: `Contract violation`, *Simplify by removing a mechanism* or *Trust the caller*, or D13,
    A13 or D11 where one fits;
  - code the diff leaves dead: C3, *Finish the replacement*.
- **It was there before, inside what the diff touches** — a `TRIM:` line in `$WORK/answers.txt`. It
  is not blocking; the author decides whether this PR takes it.

```
TRIM: <path>:<line>[-<line>] -- <what to delete> -- <evidence: the command and what it returned>
```

With nothing to trim, write one `TRIM: none -- <what you searched, and where>`. A line that would
add code — a test to write, a guard to add, a rename — is not a trim: it is a non-blocking note,
and it is deleted.

## On the card

`Trim in this PR:` carries up to five `TRIM:` lines, most lines removed first, each written as
`- <path>:<line>[-<line>] -- <what to delete> -- <evidence>`, between the findings and the
`Checked:` line. It is omitted when there are none, never counts toward `Blocking issues`, never
takes a `deferred:` line, and a re-review writes it afresh instead of settling it in
`prior_verdicts.txt`.
