# KernelForge long-horizon state

Forge-loop stores durable control state under
`<workspace>/forge_experiments/`. Files are the source of truth; prompts contain
only compact state and paths to detailed artifacts.

## Storage layout

```text
forge_experiments/
  campaign_config.json
  run_state.json
  events.jsonl
  pending_keep.json
  candidates/
  lessons/
  insights.md
  handoffs/
  analysis/
    work/<commit>/
    <commit>/
      report.md
      lanes.md
      source_map.md
      cases/<case>/profile/
  rounds/
    iter_NNN/
      lane_K/
        change.diff
        lane.json
  device_usage.jsonl
  supervisor/
    intervention_iter_NNN.md
    latest.md
  best/
```

The loader accepts exactly the field set the current code writes: a checkpoint
with a missing or unknown field, top-level or nested, is rejected rather than
filled in. A checkpoint written by a version of the loop that planned its
rounds therefore does not load: such a campaign is not resumed.

The state contains:

- Campaign, session, branch, task, and Git HEAD identity.
- Current and next iteration numbers.
- KEEP, REVERT and API error counters.
- EXPLOIT/DIVERSIFY search state.
- Pristine per-case baseline and complete KEEP/REVERT scoring state.
- Current best commit and score.
- Active Analysis evidence commit, score anchor, status, and last attempt.
- Stall and supervisor intervention state, as two separate counters: how many
  iterations the search has gone without a real KEEP, which drives the phase
  label and the EXPLOIT/DIVERSIFY switch, and the supervisor cooldown window,
  which an intervention resets so a newly injected direction gets its fair
  chance. Sharing one counter made the two mutually exclusive: the reset erased
  the stall evidence the mode switch reads later in the same iteration.
- The campaign's own wall-clock across all of its sessions.
- Pinned iterations and termination reason.

Detailed candidate code, diffs, measurements, profiles, lessons, and what each
round's lanes left remain in their dedicated artifact directories and are not
copied into `run_state.json`.

## Events

`events.jsonl` is an append-only audit stream. Each row contains `ts`, `type`,
and `iter`. Event types include:

- `baseline_measured`
- `session_started`
- `session_interrupted`
- `iteration_started`
- `search_policy_decision`
- `analysis_refresh_decision`
- `analysis_result`
- `supervisor_ruling`
- `round_dispatch`
- `iteration_result`
- `run_terminated`

`run_state.json` is the control checkpoint. `events.jsonl` is not replayed as a
general event-sourcing mechanism; replay is limited to reconciling a completed
iteration event that was durably appended before the corresponding state save.

## Resume contract

Resume is fail-closed. It requires:

- A `campaign_config.json` and `run_state.json` written by the current code.
  Files from a different field set are rejected, never migrated.
- Matching task fingerprint, driver digest, Git branch, and canonical HEAD.
- Complete pristine per-case baseline and current best score.
- No unexplained tracked working-tree changes.

`pending_keep.json` is the crash journal for the narrow interval between
canonical validation, Git commit, state publication, and archive publication.
Resume reconciles this journal before admitting another Implementer session.
A journal whose key set differs from the one the current code writes is
rejected.

Every process-local resume creates a new experiment segment while retaining the
campaign identity and cumulative state.

## Round artifacts

The loop has no planning stage. Every iteration is one round of lanes, and a
round has the same shape whatever its width:

1. The round runs one lane per focus the Analysis evidence in force divides the
   work into, up to `--lanes`, and one lane with no focus when it divides
   nothing. The division is the Analysis Agent's: `analysis/<commit>/lanes.md`
   holds one `## Lane N` section per focus, and the loop reads only the
   sections. It stays in force until the next Analysis.
2. A round's only lane is an Implementer session in the workspace itself: it
   reaches the device directly, and what it leaves on the tree is the
   candidate. Steps 3 to 5 apply to a round of several lanes. There each lane
   is an Implementer session in its own copy of the workspace. The copy leaves
   out `forge_experiments/analysis/`, which a session reads by absolute path.
   The copies live beside the workspace under `forge-lanes-*/` and are removed
   when the iteration ends. A workspace whose repository is selected through
   `GIT_DIR` or `GIT_WORK_TREE` cannot be copied, and `--lanes` above 1 is
   refused on it at startup.
3. Every lane reaches the device through its copy's `forge_device.py`, which
   queues at `<workspace>.forge-device-queue/` beside the workspace. The queue
   is a directory of numbered tickets: a ticket is created with a number above
   every existing one, its owner holds the device while it is the lowest, and a
   ticket whose owner died is removed by whoever finds it. A lane's stop check
   takes a ticket in the same queue. Each hold is one line of
   `device_usage.jsonl`.
4. What each lane changed is read as a patch against the round's base, the
   allowlisted new files it created included, and filed under
   `rounds/iter_NNN/lane_K/` as `change.diff`, with `lane.json` recording the
   lane's focus, how its session ended, whether its own stop check passed, the
   score it measured and its per-case times. A patch that touches the
   measurement surface is refused and its reason is recorded there.
5. One changed lane is the round's candidate. Several are merged by an
   Implementer session on the canonical tree, which starts from the lane whose
   stop check scored highest above the current best, or from the base when none
   did. Its stop is judged against the current best; a merge that takes nothing
   stops on the lane it started from, which is then the candidate.
6. The canonical validation and benchmark judge the candidate; nothing a lane
   measured beside other lanes decides a KEEP. The three benchmark runs of a
   stop check taken on the canonical tree with no other lane running, a
   round's only lane's or a merge session's, are reused as the canonical ones
   when they still describe the tree.

Nothing of a round outlives its iteration except these artifacts and the
iteration's lesson, so there is no queue to restore on resume: an iteration
interrupted before its verdict is run again from its base.

Analysis evidence is commit-bound but is not rebuilt after every KEEP. The
refresh threshold is currently a code-level constant of 5%. A stale bundle is
refreshed when the current canonical mean-case score reaches the score measured
at the evidence commit multiplied by `1.05`, or immediately before a Supervisor
intervention. Supervisor admission does not reprofile evidence that already
matches the current canonical.

Between refreshes, the Supervisor and the round's sessions receive the last
published bundle, its absolute artifact paths, the
commit it measured, the current canonical commit, current case timings, and the
cumulative Git diff between those commits. Historical profiling is explicitly
marked stale and is never presented as a current measurement. Cumulative diff
generation has its own 60-second timeout. If it fails, the bundle remains
available as explicitly degraded historical evidence, the failure is recorded,
and the campaign continues without forcing an Analysis refresh. A failed
refresh keeps the last published bundle available and adds every usable
artifact from the current partial checkpoint.

When a refresh is admitted, the Analysis Agent receives the cumulative diff and
previous published bundle so it can update only affected profiling and analysis
artifacts: it re-profiles only the cases whose measurements the change
invalidates, once each, keeps the parent evidence for the rest, and rewrites
`report.md` as the brief for the new canonical. The diff may span multiple
accepted KEEP commits. Each canonical
commit may start at most two Analysis sessions across resume; the
`AnalysisSessionJournal` is the sole owner of that budget. The refresh policy
prevents duplicate calls within one iteration, retries a failed
Analysis in the next iteration, and stops after the journal reports
that both session attempts are exhausted. A PARTIAL bundle is eligible for its
second attempt in the next iteration.

Every session and every Analysis Bash command is bounded by the earlier of the
configured Analysis timeout and the campaign deadline; session cleanup
terminates remaining staging process groups and workspace orphans. A profiled
session never reuses a static-only cache entry. A case is marked profiled only
when raw output, normalized metrics, and successful per-case command provenance
are all present.

DIVERSIFY reaches a session as the instruction to choose a mechanism materially
different from the ones already tried. The search turns to it after one round
without a new best and stays there until a round produces one.

## Lessons and Supervisor rulings

After a round's verdict, each of its sessions -- every lane, then the merge --
is resumed read-only, one after another, to write a free-form factual record of actions it actually attempted and results it
actually observed. The prompt forbids global optimization conclusions and
recommendations to future iterations. The loop appends its machine-authored
`OUTCOME` line after canonical validation and benchmarking.

Lesson text has no required output schema and is not parsed into a headline,
direction status, suppression list, or PR adoption classification. A
`REVERT_PERF` records only that the concrete candidate missed the KEEP
threshold; it never suppresses the broader direction.

Only the most recent lesson documents fit a prompt in full, so two things carry
the earlier ones forward. The next session is shown a list of measured
negatives with one line for every past iteration that recorded any: what its
`NEGATIVES` line says measured worse, the constants and cases it was measured
under, and whether that still stands. A line is `IN SCOPE` while those
conditions hold, `RE-OPENABLE` once a pinned constant has moved or a scored case
was not measured, and `UNSCOPED` when the conditions were never recorded. The
session is asked to check its direction against the list before choosing it.

The same lesson turn also maintains `insights.md`, the campaign's insight list:
at most 12 one-sentence facts about the kernel and about measuring it, each with
the measurement behind it -- what makes a quick bench agree with the driver, a
bound that was measured, how the toolchain compiles a construct, a correctness
condition a change has to keep. The turn is shown the list as it stands and
returns it as it should stand, on `INSIGHT:` lines that are taken out of the
lesson document and replace the stored list whole; a reply with no such line
leaves the list unchanged. Every later session reads the whole list. The record
turns of a round run one after another, so each reads the list the one before
it left. A round of several sessions sets each record under its own heading in
the iteration's one lesson document.

Every Supervisor attempt is archived verbatim in
`supervisor/intervention_iter_NNN.md`. A non-empty review also atomically
replaces `supervisor/latest.md`, which is loaded on resume and passed verbatim to
the sessions of the rounds that follow. The latest ruling may override subjective
conclusions in historical lessons but never objective validation or measurement
facts. Supervisor output is free-form and is not parsed, repaired, or translated
into a programmatic search-policy action. The active ruling expires when a KEEP
ends the stall episode or when the next Supervisor attempt begins; immutable
intervention files remain available for audit.

## Prompt view

`render_long_horizon_header()` derives a bounded prompt header from
`run_state.json` and recent events. It includes the current phase, best score,
stall state, recent factual outcomes, and retrieval paths.

The Implementer reads detailed candidate, lesson, handoff, Analysis, and
round artifacts from disk on demand. Objective measurements are
authoritative; the latest Supervisor Ruling outranks subjective conclusions in
historical lesson records.

## Scoring

The pristine baseline uses per-case medians from three independent measurements
and never changes. Every candidate is measured three times; each run is scored
independently with the equal-weight mean of its per-case speedups. Which mean is
the campaign's `--case-aggregate`, fixed when the campaign is created:

```text
arithmetic (default): mean(pristine_case_ms / candidate_run_case_ms)
geometric:            geomean(pristine_case_ms / candidate_run_case_ms)
```

Every case weighs the same under both. The arithmetic mean counts a large gain
on one case in full; the geometric mean counts its ratio, so a 4x on one case
and a 0.25x on another cancel. The in-session gate, the KEEP decision, the
Implementer's instructions and every published score
use the campaign's aggregate, and the result names it as `case_aggregate`.
Choose the one whoever consumes the result scores by: a campaign that
optimizes one mean and is graded on the other keeps candidates its grader
scores lower.

Drivers must emit complete, unique `case_ms` coverage for every scored case.
The mean of the three run scores must be at least
`current_best + t * sigma / sqrt(3)`, where `sigma` is the spread of those same
three scores and `t` is the one-sided 95% Student-t value for the degrees of
freedom the sigma estimate earned (2.920 at the usual three samples). This is a
one-sided 95% t test on the candidate's own measurements. The bar follows the
candidate's own noise because that noise varies by more than an order of
magnitude between kernels: a 0.3% gain is certain on one that repeats to 0.022%
and invisible on one that spreads over 0.281%. It is floored at 0.1% of the
current best, both so three near-identical measurements cannot drive it to zero
and because a gain under 0.1% is not worth the KEEP even when it is real.

`sigma` is the sample standard deviation of the three scores, except when one
case supplies the majority of the objective's variance (under the geometric
aggregate, of the variance of its logarithm, which splits between cases the
same way) while carrying less than
its equal share of the suite's wall time. Three aggregate scores estimate that
case's spread to within 50% of itself, and the margin then charges the draw to
every candidate: on one campaign a 10 us case holding 87% of the variance drew a
bar ranging from 0.32% to 8.42% of the incumbent, and two candidates gaining
0.92% each were decided opposite ways. Forge then buys up to two further
whole-suite benches, re-estimates every scored case's spread from the larger
sample and rescales `sigma` by the ratio the two per-case models account for.
The rule, the objective, the margin and the three scores whose mean must clear
the bar are all unchanged; only the estimate of `sigma` is sharpened, and it can
move either way. A sharper `sigma` is also charged the `t` its larger sample
earned -- 2.015 at six samples, 1.860 at nine -- since the estimate is no longer
a three-sample one. The extra measurements never become scores. The `[bench]` line names
the case, the benches bought and the before/after sigma whenever this happens. There
is no upper limit on the score a candidate may claim. The mean of the three
passing scores becomes the new monotonic best -- the same statistic the bar is
set on, so the incumbent and the threshold are measured the same way. Neither raw aggregate wall time nor
individual case regressions decide KEEP/REVERT.
