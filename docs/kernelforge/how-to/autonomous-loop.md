---
myst:
  html_meta:
    "description": "How to run the KernelForge autonomous overnight optimization loop with specialist plan synthesis, canonical validation, git keep/revert, and stalled-search supervision."
    "keywords": "KernelForge, autonomous loop, overnight optimization, kernelforge forge-loop, supervisor, validation pipeline, git commit revert"
---

# Autonomous overnight loop

The autonomous loop runs unattended for hours, proposing one change per
iteration and keeping only measured improvements.

## Run the loop

```bash
kernelforge forge-loop \
    --workspace /work/aiter-amd \
    --kernel csrc/hk_sla/vsa_sparse_attention_bwd.cpp \
    --driver op_tests/test_sla_bwd.py \
    --gpu-target gfx950 \
    --snr-threshold 30 \
    --max-hours 8
```

`forge-loop` runs ONE campaign as a standalone, hard-killable subprocess (the
entry the Hyperloom forge backend shells out to). The campaign's immutable
inputs are snapshotted into `<workspace>/forge_experiments/campaign_config.json`,
so `--resume` continues an interrupted run from the same workspace.

The campaign derives its backend from the selected kernel_backend. Its immutable
implementation signature contains the complete canonical editable-source path
set and stable symbols derived from those sources. Resume and knowledge-base
reuse therefore use the same source contract without a separate implementation
type input.

The source-owner framework is inferred from the file that defines the target
operation, including a defining file listed through `--source-files`; a direct
kernel path under `aiter`, `vllm`, or `sglang` is also recognized. If no owner
can be identified, the campaign records `unknown`. An explicit `--framework`
value is authoritative, and resume reuses the value stored in the campaign.

Each iteration:

1. Orchestration dispatches evidence-scoped work to read-only compute, memory,
   and algorithm specialists.
2. The specialists produce independent Markdown analyses. One partition call
   then reads all of them and divides the round into at most `--lanes` lanes
   (default 3), naming each lane's ground in the terms an edit lands in — files,
   functions and mechanisms — and each lane synthesizes its own plan from that
   ground and the whole round's evidence. The lanes are implemented
   concurrently, each in its own workspace copy, and their candidates are
   measured one per iteration. `--lanes 1`, or evidence that supports only one
   direction, fuses the analyses into a single plan instead: Orchestration
   compares their expected value, evidence, feasibility, cost, dependencies and
   risks, and synthesizes one. Concurrent lanes need an agent provider that
   declares `stop_hooks` and `session_env`; one that does not is refused by name
   and must be given `--lanes 1`.
3. For long-horizon sessions (`--max-hours > 2`), an independent read-only
   Critic session using the same resolved backend and model reviews the draft
   once. `ACCEPT` publishes it unchanged; `REVISE`/`REPLACE` allows
   Orchestration one revision without rerunning specialists. The revision
   resumes the synthesis session when the backend provides a session handle,
   preserving the planner's context; otherwise it uses one fresh revision
   session. Both are capped at 100 turns. The revision is given 10 minutes; the
   Critic is given 10 minutes **per plan it has to read**, capped by the
   Orchestration session timeout, because a round of several plans is several
   times the reading. A turn cap, timeout, SDK truncation, or empty answer is
   never published as a complete plan. Empty or failed Critic calls fail open to
   the draft. Review diagnostics record duration and whether the verdict was
   explicit or inferred. Shorter sessions publish the synthesized plan directly.
   A multi-lane round is reviewed once, with its division in view, and the one
   verdict reaches every lane.
4. How wide the round runs is a separate answer, given per lane, and it is the
   one part of the review a machine reads. A multi-lane review stays prose and
   ends with one JSON object —
   `{"lane_narrowing": [{"lane_id": 2, "reason": "..."}]}` — naming the lanes it
   judges not worth an Implementer session (ground the evidence does not
   support, or another lane's change in different words). Those lanes are not
   published, so the round spends fewer sessions than it planned. A review that
   wants the round whole ends with `{"lane_narrowing": []}`: an empty list and a
   missing block are different answers, and only the first means "run every
   lane". The reason travels with the lane into `structured_output.json`,
   dropping happens before the revision so no revision turn is spent on a lane
   that will not run, and three rules bound it: at least one lane always runs, a
   drop naming a lane the round does not have or carrying no readable reason
   keeps its lane and is reported, and a round carrying a challenger for a
   pending `REPLACE` refuses narrowing whole because which lane is the
   challenger was never written down. A lane the partition widened to joint
   ground is dropped like any other — the review is shown its `joint` flag and
   its fallback and rules with them in view — and the round records under
   `dropped_joint` that it spent that width without measuring it. A review that
   ends with no readable block
   is asked once more for it — one call, no tools, two turns, two minutes, and
   only when the alternative is running a lane the review said was not worth its
   session. Every refusal, and everything the round could not read, is recorded
   under `lane_narrowing`: `status` says what happened to the ruling and `block`
   says whether it was `answered`, `repaired`, `absent`, `malformed`, or
   `not_asked` (a one-lane round, which is never held to a block). A `notes`
   entry says only what was seen while reading the block and leaves the outcome
   to `status` and `dropped`, so the record cannot contradict itself, and the
   warning that a narrowing was not applied is logged where that is what
   happened rather than wherever a note exists.
5. A `REPLACE` verdict is spent on the round *after* the one it judged. It says
   the implementation route itself is dominated, which the round already
   synthesized cannot act on, so the ruling is carried forward: the next round's
   partition gives exactly one lane to validating the alternative the review
   names, and divides the rest over ground that lane does not touch. A partition
   that could not be bought carries it too. The challenger passes the unchanged
   correctness and KEEP gates, so it is allowed to lose — one lane is what the
   round spends to find out. The verdict and the path to its review are control
   state in `run_state.json`, so a campaign that reaches its budget between the
   two rounds resumes with the challenge intact.
6. The final plan is published at
   `<workspace>/forge_experiments/orchestration/iter_NNN/optimization_plan.md`.
   A round of several lanes publishes lane 1's there and the rest beside it as
   `lane_002.md`, `lane_003.md` and so on, with `lane_plans.json` written last
   to mark the round complete.
7. The writable Implementer reads that plan, edits the kernel sources, and
   exercises its in-session correctness and performance gate. Each Implementer
   session runs under a wall-clock budget sized from `--max-hours`
   (`min(210, max(90, 0.15 * campaign_minutes))` minutes, overridable with
   `--session-timeout-sec`); the session is told this deadline and asked to hand
   off its best candidate before it, and the backend cuts the session at the
   deadline if it does not. A high fixed turn ceiling remains only as a runaway
   backstop -- it does not bound a session's time.
8. The outer loop runs the driver-owned full correctness suite and canonical
   benchmark.
9. A measured improvement is committed and becomes the new best; every other
   candidate is discarded back to the last validated commit.

The plan file is the Implementer's planning source of truth for that iteration.
Dispatch inputs and specialist analyses are retained beside it.
`draft_plan.md` and `critic_review.md` are added when Critic review actually
runs.
The Framework owns role/case/evidence binding, so malformed paths or partial
specialist output do not block the Implementer. If every planning Agent fails, the
Framework writes a minimal plan that points to the current Analysis artifacts
and asks the Implementer to plan directly. Only deterministic plan-persistence or
workspace failures produce `ORCHESTRATION_ERROR`.

After a KEEP, the previous Analysis bundle remains available and is marked
stale. The loop refreshes it only after the canonical mean-case score has
improved by the code-level 5% threshold from the evidence score, or
immediately before a Supervisor intervention when the evidence does not match
the current canonical. Until then, planning agents receive the previous bundle,
current timings, and the cumulative diff between the evidence and canonical
commits. That diff may span multiple accepted KEEP commits.

Hardware profiling uses the same long-horizon gate: sessions at or below two
hours keep Analysis static-only and omit Implementer self-profiling guidance.

If the cumulative diff cannot be generated within its independent 60-second
timeout, the loop keeps the old bundle as explicitly degraded historical
evidence and continues. A missing auxiliary diff never forces profiling or
terminates the campaign.

Analysis is limited to two session attempts per commit across resume. The loop
does not issue the same Analysis request twice in one planning iteration, but a
failed request is eligible for the next iteration while the service still has
an attempt available. PARTIAL evidence may use the second attempt; an exhausted
commit continues with its published or checkpoint evidence.

The campaign is TIME-driven (`--max-hours`). When the search stalls, a
supervisor injects fresh directions rather than self-terminating on plateau.

## Round admission

A round is what the loop buys when it plans: orchestration, the lane sessions it
fans out to, and the canonical validation and benchmark that judge the first
candidate. Planning is the dominant part of that and the most variable — 12.5 to
31.6 minutes across 75 measured production rounds — so the decision is taken
twice, for two different questions.

**Before planning**, the loop refuses only a round that could not run even if
planning were as fast as any campaign has ever seen it: the cheapest planning
observed at that width, plus the least a session can be given and still return
something measurable, plus the canonical measurement. This check exists so the
campaign does not buy a plan nothing can run. A round that does not fit is
narrowed one width at a time before it is refused, because each lane the round
drops is one plan fewer for the Plan Critic to read.

**After planning returns**, when its cost is a measurement rather than an
estimate, the loop decides whether to dispatch. What is left to buy is one
Implementer session and the measurement that judges it; a round that cannot pay
for both would spend the campaign's last minutes on a candidate nobody ever
sees. This is the decisive check: replayed over ten production campaigns, the
two rounds killed by the external timeout came out of planning with 7.3 and 8.3
minutes left, against a worst survivor at 24.8.

The two checks price the same session differently, because they face opposite
asymmetries. Before planning nothing is committed and being generous only
refuses a round that would have worked, so a session is priced at the p25 of
the 219 production sessions (8 minutes). At dispatch the loop is about to start
something it cannot interrupt: too small a price starts a session the external
timeout kills, too large costs one iteration — so the same session is priced at
the median (12.3 minutes).

**The dispatch requirement also has a floor of 19.6 minutes that no observation
lowers.** A session's own wall-clock bound is sized from the campaign's total
length, not from what is left of it, so once dispatched a round runs for however
long the session takes. What the check is really guarding is therefore the
external timeout — which the loop does not set, cannot measure and cannot stop,
and which does not recede because this campaign's own validation got faster.
The floor is derived from that kill: production allowed 15 minutes of grace past
the loop's own deadline, and a session at the p90 (34.6 minutes) needs
34.6 - 15 = 19.6 minutes in hand to land inside it. The estimate is still
observation-driven above the floor, so a campaign whose measurement cycle is
genuinely expensive requires more than 19.6 minutes.

The floor sizes the session against that kill and no more — paying for the
measurement cycle on top is the estimate's job, which is why the estimate wins
whenever it is the larger of the two. It cannot go much higher either: it has
to stay under the 24.8 minutes the worst surviving round had in hand. And it
assumes the deployment leaves grace between the budget the loop counts down and
the deadline that kills it; a campaign whose `--max-hours` *is* its external
timeout has no grace, and no constant here can invent one.

Because the two checks are priced apart, a round can pass the first and be
refused by the second even when planning costs exactly what was estimated. The
pre-planning check bounds whether a round could run at all; it is not a promise
of dispatch.

Neither check charges the finalize reserve. It is a bound of its own: both
checks are handed the same unreserved remaining time the reserve is compared
with, so a round runs when what remains clears the reserve and clears the
round's own price, independently — the larger of the two binds, and no round has
to cover `reserve + its own cost`. The loop already holds the reserve back
before every iteration, and charging it a second time inside a round's price
refused rounds that went on to produce a KEEP.

Both halves are otherwise priced from what THIS campaign has observed — planning
speed and measurement cost are properties of the kernel, the evidence, the case
set and the device, not universals. A campaign with no round of its own falls
back to constants derived from production measurements.

An iteration that only drains a lane candidate an earlier round already paid for
is not a round and is never refused. When a round is refused, the campaign ends
with termination reason `round_budget_exhausted`, says so in the run summary,
and still writes its report — which is the point. A fan-out round refused after
planning keeps the plans it bought: they are published before dispatch and the
iteration records no result, so the next session runs them instead of buying
them again. The published `optimization_report.md` carries a `Round Budget`
section with the rounds planned, the planning wall-clock, the round wall-clock,
and planning's share of the campaign's wall-clock. All four are campaign totals,
not this session's: a campaign that ran over several sessions reports what the
whole campaign spent, and the share divides one campaign total by another rather
than by the current process's elapsed time.

## Auto-measured baseline

When a task does not supply `baseline_wall_ms`, the loop benches the pristine
kernel before the agent touches anything. This anchor prevents a
slower-than-baseline first iteration from being kept unconditionally, and is the
anchor the campaign's `baseline_ms` and `mean_case_speedup` results are reported
against.

## Knowledge-base read status

The final result and experiment tracker record expose the warm-start lookup under
`kb_experience.read`. `read_reason` distinguishes a reusable candidate (`hit`)
from expected skips or misses (`not_configured`, `missing_arch`,
`kernel_page_not_found`, `no_same_arch`, `solution_pages_missing`, `resume`, or
`deadline`) and failures (`read_error` or `warm_start_error`). `read_error` is
empty for non-error outcomes; failures contain a bounded, credential-redacted
exception summary. Apply decisions remain separate in `reference_reason`.

## Stalled-search supervisor

When the search plateaus, `forge-loop` escalates to a supervisor backend:

```bash
kernelforge forge-loop --workspace /work/aiter-amd --resume \
    --supervisor-backend codex
```

`--supervisor-backend` can override the Implementer backend used for this review.
The Supervisor inspects the stalled run and writes a free-form ruling for the
next fresh orchestration plan. Every interaction is archived under
`forge_experiments/supervisor/intervention_iter_NNN.md`; the latest non-empty
ruling is also stored verbatim at `forge_experiments/supervisor/latest.md` and
restored on resume. It remains active only for the current stall episode: a KEEP
or the start of another Supervisor attempt expires it, while the archived
interaction remains available for review.

It adds API calls only while the search is stuck. Bench remains the final gate;
the supervisor only raises the quality of the next edit before the loop spends a
bench cycle on it.

Per-iteration lesson documents are free-form factual records written by resuming
the same Implementer session read-only. They record attempted actions and
observed results, not recommendations to future iterations. Supervisor rulings
may reject subjective conclusions in older lessons, while objective validation
and benchmark facts remain authoritative. Neither lesson nor Supervisor output
is required to follow a machine-readable response schema.
