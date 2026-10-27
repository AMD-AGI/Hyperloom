# CI failure classification

## What the gate decides, and the review does not

`scripts/code_metrics_design.py` runs `private_reach_problems`, `tuple_return_problems`,
`raw_vocabulary_problems`, `global_write_problems`, `forwarder_problems` and
`single_implementation_problems` on every PR, reporting them as A1, A9, A10, A11 and A12. Those
five are not review rules: a finding on one of them belongs to the `code-metrics` job, and a card
that re-derives it is re-flagging a gated rule. Two of A8's shapes are gated too, each with a
limit: `flag_argument_problems` reports a flag argument only on a function new at this head,
because it walks `change.new_functions()`, and vulture reports a parameter no body reads. Neither
sees a flag added to a function that already existed, and nothing here sees a parameter only tests
vary -- `pyproject.toml` selects no `ARG` rule. Those two are A8's, which is why it survives. The
same job measures the complexity and parameter ceilings; D12 and D14 only ask the question it
cannot, which is whether a PR carrying the `baseline-raise` label gives its reason.

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
- An earlier round's green does not carry. A job that reads `main` — `code-metrics` against
  `scripts/code_metrics_baseline.txt`, the import and architecture contracts — re-runs on a body
  edit or a base move and can turn red at an unchanged head, because what it compares against
  moved. Re-read `ci.txt` at the head you are publishing on, and when a check now fails, find what
  landed on `main` between the two runs before attributing it to this PR: in PR #2102 the
  `_ensure_ray` complexity entry #2189 added to the baseline made a stale-baseline failure out of a
  branch whose own code had improved that number.
