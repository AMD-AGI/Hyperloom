# Forge-loop search policy

The search policy decides which kernel version each iteration starts from and
which measured candidates later iterations build on. It is chosen with
`--search-policy` when a campaign is created and is fixed for the life of the
campaign.

| Policy | A candidate becomes the next starting version when |
| --- | --- |
| `sequential` (default) | it is correct and beats the current best under the KEEP bar |
| `seqany` | it is correct and its benchmark completed with every scored case |

Under `sequential` the starting version is always the current best, which is
how forge-loop has always searched. Under `seqany` the starting version is the
most recent correct candidate, which may be slower than the best.

Everything other than the acceptance rule is shared: the planning chain, the
Implementer, the in-session gate, the correctness checks, the KEEP bar and the
publication of the best result are the same under both policies.

## Per-iteration decision

Every measured candidate is judged on two facts:

- **valid**: canonical validation passed and the benchmark produced all three
  measurements with complete scored-case coverage. For the assembly backend
  the canonical correctness suite must pass as well.
- **improves best**: the candidate clears the KEEP bar against the current
  best (see [Scoring](forge_long_horizon_state.md#scoring)).

The policy maps these to one of three outcomes:

| valid | improves best | `sequential` | `seqany` |
| --- | --- | --- | --- |
| no | - | `REVERT_*` | `REVERT_*` |
| yes | yes | `KEEP` | `KEEP` |
| yes | no | `REVERT_PERF` | `ACCEPT` |

| Outcome | Git | Best record | Publication |
| --- | --- | --- | --- |
| `KEEP` | commit on the campaign branch | updated to this candidate | `best/` and the recovery checkpoint are published |
| `ACCEPT` | commit on the campaign branch | unchanged | none |
| `REVERT_*` | working tree discarded back to the branch's latest commit | unchanged | none |

`ACCEPT` never occurs under `sequential`. An `ACCEPT` counts as an iteration
without a new best for the stall counters, the Supervisor trigger and the
EXPLOIT/DIVERSIFY switch.

`KEEP` and `ACCEPT` both commit before the run state is saved, so both are
journaled in `pending_keep.json`. The journal's `promotes_best`
field says whether resume must also restore the best record and finish the
publication; an `ACCEPT` journal only restores the commit and the run state.

## Starting version

`run_state.json` keeps one record per iteration that produced a candidate
diff, in `candidates`:

| Field | Meaning |
| --- | --- |
| `iteration` | the iteration that produced the candidate |
| `parent_iteration` | the iteration whose version the candidate was made from; `0` is the campaign's starting version |
| `parent_commit` | the commit the candidate was made from |
| `decision` | the iteration's decision label |
| `commit_hash` | the commit, for `KEEP` and `ACCEPT`; empty otherwise |
| `mean_case_speedup` | the candidate's score, whenever it was scored |
| `case_times` | the candidate's per-case times, for `KEEP` and `ACCEPT` |

Iterations that measured nothing (`NO_CHANGES`, `API_ERROR`, `AGENT_ERROR`,
`ORCHESTRATION_ERROR`, a held device) add no record.

The starting version of the next iteration is not stored. It is derived from
the records by the policy:

- `sequential`: the best version.
- `seqany`: the most recent `KEEP` or `ACCEPT` record.

Before any candidate is committed, both policies start from the campaign's
starting version: the pristine base commit, or the warm-start commit when one
was applied. It is recorded once, when a fresh campaign starts, as
`run_state.start_commit`.

At the start of every iteration the loop requires the campaign branch's latest
commit to equal the commit of the derived starting version. A mismatch stops
the campaign. Under both policies the starting version is always the branch's
latest commit, so this is a check and never moves the branch.

## Branch and best result

A campaign uses exactly one Git branch, the campaign branch, under both
policies.

- Under `sequential` the branch's latest commit is the best version.
- Under `seqany` the branch's latest commit is the next iteration's starting
  version. The best version is `run_state.best.commit_hash`. Because the
  branch is linear and every best version was once the branch's latest
  commit, the best commit is always an ancestor of (or equal to) the branch's
  latest commit and stays reachable for as long as the branch exists.

Callers never read the branch's latest commit as the result. The result is
the published best: `forge_experiments/best/`, the `best_commit` in
`forge-result.json`, and the recovery checkpoint written on every `KEEP`.
Patches are exported as `git diff <base>..<best_commit>`. Under `seqany` the
end-of-run summary also prints the number of `ACCEPT`s, the best commit and
the branch's latest commit.

### Interrupted campaigns

The best result is published after every `KEEP` and before the next Agent
session starts; the interval between the `KEEP` commit and that publication is
covered by `pending_keep.json` and the recovery checkpoint. A campaign killed
at any point therefore leaves the last `KEEP` readable by callers, regardless
of which commit the branch is on and whether the working tree holds an
unfinished candidate. The kernel rewrite controller's reclaim and release
logic is unaffected, because the workspace never leaves the campaign branch.

## Best version and starting version consumers

| Consumer | Reads |
| --- | --- |
| KEEP bar, sigma resolution, `best_case_times` | best version |
| Publication, recovery checkpoint, `forge-result.json` | best version |
| Merge stacking and pinned near-misses (compared against the incumbent) | best version |
| In-session gate pass threshold | best version |
| Current per-case timings given to Orchestration, specialists and the Supervisor | starting version |
| Analysis evidence commit and refresh score | starting version |
| Long-horizon header and in-session gate messages | both, when they differ |

The Analysis bundle is refreshed when the starting version's score has moved
by at least 5% from the score measured at the evidence commit, in either
direction. Under `sequential` the starting version's score never decreases,
so only the upward trigger can fire.

## In-session gate

The gate's decision is identical under both policies: it blocks a stop while
the kernel is incorrect or does not clear the KEEP bar against the best
version, up to its block budget, and then hands the candidate to the outer
loop. Under `seqany` a candidate the gate never passed is still `ACCEPT`ed by
the outer loop when it is valid. When the starting version differs from the
best version, the gate's block message states both scores.

## Interaction with other options

- `--lanes`: `seqany` requires a single lane. The default is `3` under
  `sequential` and `1` under `seqany`; an explicit value above `1` with
  `seqany` is refused.
- `--merge-stacking`: stacks two `REVERT_PERF` candidates. Under `seqany`
  every valid candidate is `ACCEPT`ed, so there is never a candidate to
  stack and the option has no effect. Pinned near-misses follow the same rule.
- EXPLOIT/DIVERSIFY and the Supervisor are policy-independent; both read the
  stall counters, which only a `KEEP` resets.

## Configuration and resume

`--search-policy` is stored in `campaign_config.json` and read
back on `--resume`. A resume that omits the option uses the stored policy; a
resume that names a different policy is refused.

Resume requires the campaign branch's latest commit to equal the derived
starting version and the best commit to be an ancestor of (or equal to) it.

## Audit records

The `iteration_result` event and each archived candidate's `meta.json` carry
`parent_iteration`, `parent_commit` and `accepted`, alongside the existing
`decision`, `commit_hash` and `is_new_best`. The search trajectory and the
best trajectory can both be rebuilt from `events.jsonl`.
