# CI failure classification

## What the gate decides, and the review does not

`scripts/code_metrics_design.py` runs `tuple_return_problems`, `raw_vocabulary_problems`,
`global_write_problems`, `forwarder_problems` and `single_implementation_problems` on every PR,
reporting them as A9, A10, A11 and A12. Those four are not review rules: a finding on one of them
belongs to the `code-metrics` job, and a card that re-derives it is re-flagging a gated rule.

Two cautions before reading any gate as total coverage. First, scope: `private_reach_problems`
resolves ownership with `Packages.unit()`, built from `.importlinter`, which makes
`hyperloom.orchestrator` a single unit -- a private import between two of its subpackages is
reported by nobody, which is A1's. Second, the waiver: `code_metrics_report.py:136-138` puts
`design` in `waivable_failed`, and `PRIVATE_REACH` and `FORWARDER` are not in `ADVISORY`, so a
`baseline-raise` label turns both into listed-but-passing findings. On a labelled PR the gate
decides none of them. A8 is the partial case. `flag_argument_problems`
reports a flag argument, but only on a function absent from the base tree, so the flag added to a
function that already existed stays A8's. `vulture` reports a parameter no body reads, which is
also the gate's. Nothing here measures a parameter only tests vary, on any function --
`pyproject.toml` selects no `ARG` rule, and `_selecting_flag` matches only a body opening with
`if <flag>:` and ending that branch in `return` or `raise`. That shape is A8's wherever it
appears, new function or not, which is why the rule survives with two differently-scoped halves. The
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
