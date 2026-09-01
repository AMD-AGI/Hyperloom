# Forge-loop long-horizon state

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
  handoffs/
  analysis/
    work/<commit>/
    <commit>/
      report.md
      source_map.md
      cases/<case>/profile/
  orchestration/
    iter_NNN/
      context.json
      dispatch.json
      specialists.json
      structured_output.json
      lane_plans.json
      draft_plan.md
      critic_review.md
      optimization_plan.md
      lane_NNN.md
  supervisor/
    intervention_iter_NNN.md
    latest.md
  best/
```

`run_state.json` uses schema v19. The loader validates the current field set
strictly and migrates older checkpoints forward without discarding control
state: v13 gains an empty Analysis refresh anchor, which causes one safe refresh
instead of guessing which score historical profiling measured; v14 gains an
empty Plan Critic ruling, which is what a campaign that never recorded a verdict
actually knows; v15 gains an empty round cost history, which the round admission
guard treats exactly as it treats a campaign's first round; v16's recorded
rounds gain a zero measurement cost, which is read as no observation rather than
as a free validate-and-benchmark cycle; v17 gains a campaign wall-clock for its
cumulative planning to be a share of, seeded from what its rounds cost, since
that is the longest span such a checkpoint can honestly claim to have run and it
already covers the planning inside it; v18 gains a separate unresolved-stall
counter, seeded from its no-improvement streak, which is a lower bound on the
real stall because every past intervention had already reset that streak, and a
lower bound is the fail-safe direction here. Other malformed, incomplete,
unknown, or differently versioned checkpoints are rejected. The loader also rejects a
campaign wall-clock shorter than the planning charged to it, so the share cannot
exceed 100 by way of a hand-edited or future-written checkpoint.

The state contains:

- Campaign, session, branch, task, and Git HEAD identity.
- Current and next iteration numbers.
- KEEP, REVERT, API error, and orchestration error counters.
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
- What the campaign's own rounds have cost: planning, canonical measurement and
  total wall-clock per round for the last few, plus campaign-wide totals. This
  is what the round admission guard prices the next round from. The campaign's
  own wall-clock is carried here too, advanced on the same call that charges
  planning to it, so the cumulative planning has a span of the same campaign to
  be reported as a share of rather than the current process's elapsed time.
- Pinned iterations and termination reason.

Detailed candidate code, diffs, measurements, profiles, lessons, and
orchestration analysis remain in their dedicated artifact directories and are
not copied into `run_state.json`.

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
- `round_admission`
- `round_dispatch`
- `round_cost`
- `iteration_result`
- `run_terminated`

`run_state.json` is the control checkpoint. `events.jsonl` is not replayed as a
general event-sourcing mechanism; replay is limited to reconciling a completed
iteration event that was durably appended before the corresponding state save.

## Resume contract

Resume is fail-closed. It requires:

- A valid current or explicitly migratable `campaign_config.json` and
  `run_state.json`.
- Matching task fingerprint, driver digest, Git branch, and canonical HEAD.
- Complete pristine per-case baseline and current best score.
- No unexplained tracked working-tree changes.

`pending_keep.json` is the crash journal for the narrow interval between
canonical validation, Git commit, state publication, and archive publication.
Resume reconciles this journal before admitting another Implementer session.
The journal uses schema v2.

Every process-local resume creates a new experiment segment while retaining the
campaign identity and cumulative state.

## Planning artifacts

Each iteration runs:

1. Evidence-scoped specialist dispatch.
2. Parallel read-only specialist analysis.
3. Orchestration synthesis: one plan per lane at `--lanes N`, one fused plan at
   `--lanes 1`.
4. For sessions longer than two hours, one same-model, independent Critic
   review. `REVISE`/`REPLACE` permits one Orchestration revision without
   rerunning specialists. The revision resumes the original synthesis session
   when its backend exposes a session handle, otherwise it uses one fresh
   revision session. Both calls have a 100-turn ceiling, and incomplete output
   is rejected. The revision has a 10-minute runtime ceiling; the Critic's is
   10 minutes per plan it reads, bounded by the Orchestration session timeout,
   because a round of several plans is several times the reading. Sized for one
   plan it was not enough: a measured two-lane review ran about eleven minutes,
   failed open to `ACCEPT`, and lost a verdict that had found a lane not worth
   its session. Critic failure uses the draft; revision failure publishes a
   non-executable framework fallback. Diagnostics retain phase duration and
   whether the Critic verdict was explicit or inferred. Shorter sessions skip
   this step.
5. Publication of the final `optimization_plan.md`.
6. Implementer execution of that plan.
7. Canonical validation, benchmark, and KEEP/REVERT.

`optimization_plan.md` is the Implementer's planning source of truth for that
iteration. Handoff schema v2 records its path together with the canonical
verdict, the latest Supervisor Ruling path, and audit pointers. Handoffs do not
restore control state. When Critic review runs, `draft_plan.md` and
`critic_review.md` are immutable audit artifacts for that planning cycle.

A fan-out round (`--lanes N`) synthesizes one plan per lane instead of one for
the round. It first buys one partition call that reads every specialist analysis
and names each lane's ground in files, functions and mechanisms — the terms an
edit lands in, rather than the specialist role the evidence came from, because
the roles are several readings of one kernel and dividing by role divides no
code. Each lane is then given its own ground, every other lane's, and the whole
round's evidence. The partition may return fewer lanes than asked for when the
evidence supports fewer, and one ground is planned as an ordinary single-lane
round. A partition that cannot be bought collapses the round to a single lane
rather than dealing the analyses out across many: dividing the evidence by role
divides no code, so a wide fallback would spend N Implementer sessions on lanes
that overlap and may get one answer for them.

A ground is a planning boundary, not an enforced one. It reaches the Implementer
as an instruction and is reviewed as one, but a lane's candidate is admitted on
the rules every candidate is admitted on — it applies, and it leaves the
measurement surface alone — and nothing checks the files it touched against the
ground it was given. Two lanes that overlap therefore still cost two sessions
for one answer; what the partition buys is that they usually do not.

Lane 1's stays `optimization_plan.md`, and lanes 2..N are published
beside it as `lane_002.md`, `lane_003.md` and so on, so the round is auditable
after the fact. `lane_plans.json` records the count and the commit the plans
describe, and is written last: an iteration's plans are readable only once that
file is present. The latest iteration BEFORE the one asking that started and
never reported a result is the one round whose plans were never dispatched, and
the next process picks those plans back up rather than paying to synthesize them
again -- unless the tree has moved off the commit they were written against,
which makes them stale. The asking iteration is excluded because the loop marks
an iteration started before it plans anything, so that iteration is always
itself started and unfinished. A round refused for budget after planning leaves
exactly this state on purpose.

What the round produced is published too. Its candidates are spent one per
iteration, so a process that ends with any of them unspent -- a budget that ran
out mid-round, not only a crash -- would otherwise throw away finished
Implementer sessions whose lane workspaces are already deleted. `lane_queue.json`
holds what is still owed a measurement, is rewritten as each candidate is taken,
and is read before the first iteration of the next process, which measures those
candidates before it plans a new round.

A fan-out round that ends with no candidate to measure -- because planning was
unavailable, because only one plan came back, because the lane workspaces could
not be made, or because no lane wrote anything -- hands the iteration to the
ordinary single-session path. It hands over its plan with it, so the iteration
runs its session on the round it has already paid for; a planning outage is
reported as this iteration's `ORCHESTRATION_ERROR` rather than re-asking the
backend that just refused.

The same `max_hours > 2` long-horizon gate enables both the Plan Critic and
hardware profiling. Shorter sessions keep Analysis static-only and do not
inject self-profiling guidance into the Implementer.

The Plan Critic reviews every round a synthesis produced, at any width. A wide
round is reviewed once, with its division in view: whether any lane's ground is
worth an Implementer session, whether two lanes are one change described twice,
whether a lane would have to edit code another lane owns, and whether the round
as a whole is working at a level that has stopped paying. One verdict covers the
round, so `REVISE` and `REPLACE` reach every lane, each resuming its own
synthesis session. A lane whose revision fails keeps its draft and is named in
`plan_revision.unrevised_lanes`; a single-lane round that cannot be revised
still publishes the non-executable fallback, because nothing else is left in it.

How wide the round runs is answered per lane rather than by the verdict, and it
is the one part of the review a machine reads. The review stays prose — a person
reads it and the revision is fed it — and ends with one JSON object carrying the
width decision:

```json
{"lane_narrowing": [{"lane_id": 2, "reason": "it is lane 1's change in different words"}]}
```

A named lane is not published, so the round spends fewer Implementer sessions
than it planned. A round the review wants whole ends with the same block and an
empty list, because an empty list and a missing block are different answers and
only the first means "run every lane". The finding is older than the outlet:
across sixty-eight measured rounds, six reviews said a specific lane was not
worth its session and every one of those rounds ran it, because a round-wide
verdict cannot single a lane out. Narrowing is applied before the revision, so no
revision turn is spent on a lane that will not run, and `lanes.published` records
what the round actually handed to Implementer sessions beside what it planned.

The block is read with the same extractor the round partition uses, searched
from the end of the review and anchored on its own key, because the prose before
it is free to quote an autotune config and the first JSON object in a review is
not necessarily its ruling. A review that ends with no readable block is asked
once more — no tools, two turns, two minutes — to restate the decision it
already made in its own words. That is the round's only conditional call, and it
is spent only when the alternative is running a lane the review said was not
worth its session. A block that *was* read and named a lane the round does not
have is never repaired: correcting it would mean inventing the decision.

Three decisions can move a round's width, and they are ordered so they cannot
contradict each other. The partition decides how wide the round is planned and
its collapse fallback is the floor. The narrowing decides how many of those
lanes are published; it runs last and therefore wins on width. A pending
`REPLACE` outranks both: exactly one lane is validating the alternative that
verdict named and which lane that is was never written down, so a challenged
round refuses narrowing whole. Under all of them one lane is the floor — a
narrowing that would empty the round is refused whole rather than applied down
to a survivor the review never ranked. A lane the partition widened to joint
ground is no exception: the review is given that lane's `joint` flag and its
fallback and rules on the lane with them in view, so a drop naming it is
carried out like any other, and the round records under `dropped_joint` that
it spent wider ground than a region and measured none of it. Every drop, every
refusal, and everything the round could not read is recorded under `lane_narrowing` in
`structured_output.json` with the reason: `status` says what happened to the
ruling and `block` (`answered`, `repaired`, `absent`, `malformed`, `not_asked`)
says where it came from. So a round that published fewer lanes than it planned
can be audited afterwards, and "the review wanted every lane" never arrives
looking the same as "nobody could read what it wanted".

The record is split so it cannot disagree with itself. A note under `notes`
says what was seen while the block was being read — the review ended with no
block, an entry named no lane — and never what became of it, because it is
written before the repair pass and the round have answered; `status` and
`dropped` are what say how the round ended. The one note added afterwards is
the joint-lane cost, which is not how the ruling ended but what carrying it out
spent, and has nowhere else to be read. The logs follow the same line: the
round warns that a narrowing was not applied where that is the outcome, and a
review whose block one repair pass recovered and the round then acted on is
reported as the narrowing it was.

`structured_output.json` also records `phase_durations_sec`: what dispatch, the
specialists, the partition, the synthesis, the Critic review and the revision
each cost, and the round's own total. Ten production campaigns spent a median
21.6 minutes per round on planning, about a quarter of an eleven-hour budget,
and roughly a third of that window could only be recovered by subtracting the
phases that persisted their timings from the total — which made the second most
expensive phase the only invisible one. `total` is the orchestration call's own
wall-clock rather than the sum of the parts, so whatever the named phases do not
account for stays visible as the difference. Publishing the plans happens in the
loop outside that call and is part of that remainder.

A `REPLACE` verdict also outlives the round it judged. It says the route itself
is dominated, which the round it was passed on can no longer act on, so the
ruling is carried into the next round's partition: that round gives exactly one
lane to validating the alternative the review names, and divides the rest over
ground the challenger does not touch. The fallback carries it too — a partition
that could not be bought collapses to the single challenger lane rather than
dealing every lane back onto the route the verdict just dominated. The challenger
is measured under the same
correctness and KEEP gates as any other lane and is allowed to lose.

The verdict is control state, so it survives the process that recorded it.
`run_state.json` holds the verdict and the path to the review that made it,
because a critic rules on a round already planned and a campaign routinely
reaches its budget between that round and the one the ruling is spent on. The
review stays where it was published; a ruling whose review can no longer be read
is dropped rather than resumed, since the alternative to validate is named in
the review and not in the word `REPLACE`. A review that failed open records no
ruling at all: its artifact holds the outage that stopped it.

Planning Agent output is best-effort. The Framework binds specialist roles,
cases, and exact evidence paths; partial or failed specialists are recorded but
do not block synthesis. If every planning Agent fails, the Framework still
writes an `optimization_plan.md` that points the Implementer at the current Analysis
bundle and asks it to plan directly. `ORCHESTRATION_ERROR` is reserved for
deterministic infrastructure failures such as being unable to persist that plan.

Analysis evidence is commit-bound but is not rebuilt after every KEEP. The
refresh threshold is currently a code-level constant of 5%. A stale bundle is
refreshed when the current canonical mean-case score reaches the score measured
at the evidence commit multiplied by `1.05`, or immediately before a Supervisor
intervention. Supervisor admission does not reprofile evidence that already
matches the current canonical.

Between refreshes, Orchestration, specialists, the Supervisor, and the
Implementer receive the last published bundle, its absolute artifact paths, the
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
artifacts. The diff may span multiple accepted KEEP commits. Each canonical
commit may start at most two Analysis sessions across resume; the
`AnalysisSessionJournal` is the sole owner of that budget. The refresh policy
prevents duplicate calls within one planning iteration, retries a failed
Analysis in the next planning iteration, and stops after the journal reports
that both session attempts are exhausted. A PARTIAL bundle is eligible for its
second attempt in the next planning iteration.

Every session and every Analysis Bash command is bounded by the earlier of the
configured Analysis timeout and the campaign deadline; session cleanup
terminates remaining staging process groups and workspace orphans. A profiled
session never reuses a static-only cache entry. A case is marked profiled only
when raw output, normalized metrics, and successful per-case command provenance
are all present.

DIVERSIFY influences Framework dispatch and the requested planning objective;
missing specialist coverage is recorded in diagnostics rather than treated as a
hard gate. Three consecutive infrastructure-level `ORCHESTRATION_ERROR`
outcomes open the circuit and pause the campaign.

## Lessons and Supervisor rulings

After each Implementer session, the same session is resumed read-only to write a
free-form factual record of actions it actually attempted and results it
actually observed. The prompt forbids global optimization conclusions and
recommendations to future iterations. The loop appends its machine-authored
`OUTCOME` line after canonical validation and benchmarking.

Lesson text has no required output schema and is not parsed into a headline,
direction status, suppression list, or PR adoption classification. A
`REVERT_PERF` records only that the concrete candidate missed the KEEP
threshold; it never suppresses the broader direction.

Every Supervisor attempt is archived verbatim in
`supervisor/intervention_iter_NNN.md`. A non-empty review also atomically
replaces `supervisor/latest.md`, which is loaded on resume and passed verbatim to
Orchestration and the Implementer. The latest ruling may override subjective
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
orchestration artifacts from disk on demand. Objective measurements are
authoritative; the latest Supervisor Ruling outranks subjective conclusions in
historical lesson records.

## Scoring

The pristine baseline uses per-case medians from three independent measurements
and never changes. Every candidate is measured three times; each run is scored
independently with the equal-weight arithmetic mean:

```text
mean(pristine_case_ms / candidate_run_case_ms)
```

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
case supplies the majority of the objective's variance while carrying less than
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
