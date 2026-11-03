# CI failure classification

## What the gate decides, and the review does not

`scripts/code_metrics_design.py` reports design findings under the A-family rule ids. A finding on
an id this table gives to the gate is the `code-metrics` job's, and a card that re-derives it is
re-flagging a gated rule. An id the table gives to the review has a rule body in `rules.md`, and
that rule's `Severity:` line is the single place stating where the boundary falls and why -- this
table does not restate it, so the two cannot drift apart.

| id | decided by | function |
|---|---|---|
| A9, A10, A11 | the gate, always | `tuple_return_problems`, `raw_vocabulary_problems`, `global_write_problems` |
| A12 single-implementation | the gate, always | `single_implementation_problems` |
| A1, A8, A12 forwarder | **split** -- see each rule's `Severity:` | `private_reach_problems`, `flag_argument_problems`, `forwarder_problems`, vulture |
| D12, D14 | the gate, except the question it cannot answer -- see each rule | the complexity and parameter ceilings |

One fact applies to every row and belongs here rather than in any rule: a `baseline-raise` label
waives the gate's design findings entirely. `code_metrics_report.py:132-134` puts `design` in
`waivable_failed`, and `PRIVATE_REACH`, `FLAG_ARGUMENT` and `FORWARDER` are absent from
`code_metrics_design.py`'s `ADVISORY`, so on a labelled PR the gate decides none of them and every
split row returns to the reviewer in full.

Before giving any id to the gate, read the function's own decision, not its name: the four defects
#2227 shipped were all a rule id matching a function whose body judged something narrower. Rule E5
is that check.

## Reading `ci.txt`

Consulted from Step 8 of [`../SKILL.md`](../SKILL.md) before `ci.txt` or `meta.txt`'s `mergeable` is
quoted on the card. Read `ci.txt` before blaming the PR:

- Which sha did each check run against? A success attached to an older head does not cover commits
  pushed afterwards. A narrow subset of workflows reporting while the diff touches runtime logic
  means the full matrix has not run.
- Does the same job fail the same way on `main` in the same window? Then it is baseline, not a
  regression introduced here.
- Infra shapes — runner acquisition, artifact upload, dependency resolution, expired logs — are
  flakes: ask for a rerun rather than quoting them as failures.
- `mergeable: CONFLICTING` in `meta.txt` is blocking on its own; ask for the rebase.
- An earlier round's green does not carry. A job that reads `main` — `code-metrics` against the
  tree at the base, the import and architecture contracts — re-runs on a body edit or a base move
  and can turn red at an unchanged head, because what it compares against moved. Re-read `ci.txt`
  at the head you are publishing on, and when a check now fails, find what landed on `main`
  between the two runs before attributing it to this PR.
