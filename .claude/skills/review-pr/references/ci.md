# CI failure classification

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
