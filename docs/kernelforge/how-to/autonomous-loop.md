---
myst:
  html_meta:
    "description": "How to run the KernelForge autonomous overnight optimization loop with lanes of self-directed sessions, canonical validation, git keep/revert, and stalled-search supervision."
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

The loop has no planning stage. Every iteration is one round, and a round has
one shape whatever its width:

1. **Analysis.** Before the first round, and again once the score has improved
   by 5% since the evidence was collected, the Analysis Agent profiles the
   canonical kernel and writes its brief. In a campaign of several lanes
   (`--lanes`, default 3) it also writes `lanes.md`, dividing the work into at
   most that many focuses: by optimization direction, by scored case, or a mix.
   The loop reads only how many focuses there are; what a focus says is between
   the Analysis Agent and the session that works on it. The division stays in
   force until the next Analysis.
2. **Lanes.** The round runs one lane per focus, and one lane with no focus when
   the Analysis divided nothing or `--lanes` is 1. Each lane is an Implementer
   session in its own copy of the workspace. The copy leaves out the Analysis
   evidence, which a session reads by the absolute path its prompt names.
3. **Direction.** A lane's session chooses its own direction from the Analysis
   report and the lesson records, inside its focus when it has one. The focus
   says where to start, not where to stop: a session whose own measurements
   show the focus has no room left moves on, and it is shown the other lanes'
   focuses so that it does not repeat them. The search mode reaches it there:
   while the search is exploiting it is asked for the largest expected gain, a
   structural change included; once the search has stalled it is asked for a
   mechanism materially different from the ones already tried; and an active
   Supervisor ruling names the directions to take. A stall is counted in
   rounds: one that ends without a new best turns the search to diversifying,
   two in a row are a blockage, which hands the search to another Agent profile
   or, when none is available, brings the Supervisor's review (see *Model
   relay* below), and the search stays diversifying until a round produces a
   new best.
4. **Session.** Every lane runs the same gated session. It works under a
   wall-clock budget sized from `--max-hours`
   (`min(210, max(90, 0.15 * campaign_minutes))` minutes, overridable with
   `--session-timeout-sec`); the session is told this deadline, and the backend
   cuts the session at the deadline, after which the loop takes whatever
   tracked changes the lane's copy holds. A high fixed turn ceiling remains only
   as a runaway backstop -- it does not bound a session's time. The session is
   meant to use its budget rather than hand off its first improvement: it
   states the score it expects before editing, measures the kernel it starts
   from, then repeats edit, correctness and benchmark, keeping a copy of the
   best version it has measured and restoring it into the workspace after every
   change that does not help. It reads the clock with every measurement and
   paces itself by four numbers: it leaves a direction that measurement
   disproved or that 8 measured changes in a row failed to improve, and takes
   the next one; with more than 20 minutes before its wrap-up minute, the time
   left is no reason to pass over a direction; it starts to finish at the
   wrap-up minute (80% of the session budget), after 12 measured changes in a
   row without a new best, or once every direction worth measuring has been
   measured; and it keeps each measurement command in the foreground and under
   4 minutes.
5. **Stop check.** A session's stop is judged by its in-session gate, which
   repeats correctness and takes three benchmark measurements. A round's only
   lane has to beat the current best to stop, and hands off no candidate when
   nothing did. A lane that runs beside others has to be correct: what it
   gained on some cases is for the merge to use, whatever its overall score.
6. **Device.** The lanes of a round share the campaign's GPU. A lane's session
   sees no device of its own; every command that needs one -- the driver, a
   benchmark script of the session's own, a profiler -- goes through the
   lane's device runner, `python3 <lane>/forge_device.py <command>`, which
   waits its turn, makes the device visible to that command and runs it
   unchanged. Commands are served in the order they arrived, one command may
   hold the device for at most 4 minutes, and a lane's stop check waits in the
   same queue. Waiting counts toward the session's budget, so a round never
   outlasts one session budget however wide it is. Every hold is recorded in
   `forge_experiments/device_usage.jsonl` with the lane, how long it waited and
   how long it held.
7. **Merge.** What each lane changed is read as a patch against the round's
   starting point; a patch that touches the driver or the harness is refused.
   When one lane changed the kernel, its change is the round's candidate as it
   stands. When several did, an Implementer session on the canonical tree
   merges them. It starts from the lane whose own stop check scored highest
   above the current best -- or from the round's starting point when none did
   -- and is given the other lanes' patches, their workspaces and their
   per-case times. How to merge is its own decision: apply a patch, rewrite
   where two lanes changed the same code, dispatch per case between two lanes'
   implementations. Its stop has to beat the score it started from. Each
   lane's patch and stop-check result are kept under
   `forge_experiments/rounds/iter_NNN/lane_K/`.
8. The outer loop runs the driver-owned full correctness suite and canonical
   benchmark.
9. A measured improvement is committed and becomes the new best; every other
   candidate is discarded back to the last validated commit.
10. **Records.** Once the verdict is known, every session of the round -- each
    lane, then the merge -- is asked in turn for its record (see below).

Nothing a session started outlives it. On the Claude backend the session's CLI
is started with a mark in its environment that every process it starts
inherits, and when the session ends — on its own, on an error, or at its
deadline — everything carrying that mark is terminated, wherever it is working:
a benchmark detached from its shell, a profiler run from a scratch directory
outside the workspace. Sessions that share a directory do not share a mark, so
one ending leaves the others' processes alone. A leftover that survives is
reported as workspace contention and the candidate is not measured. A session
cut off at its deadline additionally has its workspace checked for anything
else holding the device, as before.

The same mark bounds what a session may stop while it runs. A host that runs
several campaigns has the same driver, profiler and scratch-script names in each
of them, so a kill that chooses its targets by name is never scoped to the
session that issues it. The Implementer's in-session hooks and the Analysis
session's deny `pkill` and `killall`, a `kill` given no process of its own, and
any command that both searches processes by name (`pgrep`, `pidof`, `ps`) and
runs `kill`, however the search reaches it: a substitution, quoted or not, a
pipe, a loop or a variable. An inline program is denied the same way when it
names `pkill` or `killall`, or searches by name and sends a signal. The check
reads one command at a time, so it stops the habit rather than the intent: a
search in one command and a `kill` of the PIDs it printed in the next is not
seen. The denial names what to run instead: `python -m kernelforge.utils.session_kill
'<pattern>'` signals the processes carrying the calling session's mark whose
command line matches, and never the shell or the agent CLI that ran it; a
`kill` of a named PID is left alone. The Implementer's prompt states the same
rule only on a provider that both marks its sessions and runs the hooks, which
today is Claude: a Codex session carries no mark and no hook, is told nothing
about either, and its kills are not checked. The Claude CLI is also handed its
system prompt in a private file rather than as an argument, so the driver and
tool names in the prompt are not on a command line for another session's
pattern to match.

A session does not write a takeaway for the next one. After the round's verdict
a separate lesson turn asks each of its sessions to record what it explored,
including the directions it kept or dropped, on the lines `HELD-FIXED`,
`NEGATIVES`, and `DISPROOF`. The turns run one after another, the lane whose
stop check scored lowest first and the merge session last, and a round of
several sessions sets each record under its own heading. The loop then appends
the scope and outcome lines. That file is `lessons/iter_NNN.md`, and it is the
only reasoning record. Each turn also returns the campaign's insight list as it
should stand, which is stored apart in `insights.md` and read in full by every
later session, alongside one line of measured negatives for each past
iteration; because the turns are serial, each one reads the list the one before
it left. When the loop
starts it hides `forge_experiments/` from git with a `forge_experiments/.gitignore`
holding `*`, so nothing written there is committed or rejected by the workspace
guard as a new file, whatever the workspace itself ignores.
`kernelforge export-runs` turns the whole lesson into the reasoning of a run's
Experiences; see
{doc}`the CLI reference </kernelforge/reference/cli>`.

After a KEEP, the previous Analysis bundle remains available and is marked
stale. The loop refreshes it only after the canonical mean-case score has
improved by the code-level 5% threshold from the evidence score, or
immediately before a Supervisor intervention when the evidence does not match
the current canonical. Until then, the round's sessions receive the previous
bundle, current timings, and the cumulative diff between the evidence and
canonical commits, and the lanes keep the focuses that bundle gave them. That
diff may span multiple accepted KEEP commits.

What an Analysis session delivers is a brief. `report.md` holds, in order, where
each scored case's time goes, the bottlenecks ranked by their effect on the
score, and for each bottleneck the direction the session expects to pay with
the gain it expects; raw evidence stays in the case directories and is cited by
path. The session measures the canonical kernel only: it does not edit a copy,
sweep a parameter or benchmark a variant, because testing an idea is the
Implementer's work. One collection per profiled case is the default. A refresh
re-profiles only the cases whose measurements the kept changes invalidate, once
each, and rewrites the brief for the new canonical. `--profile-timeout-sec` is
the session's time box: the session is told to treat it as a budget and to
write the brief as soon as the first evidence supports it, so a session cut at
the deadline still leaves one.

Hardware profiling uses the same long-horizon gate: sessions at or below two
hours keep Analysis static-only and omit Implementer self-profiling guidance.

If the cumulative diff cannot be generated within its independent 60-second
timeout, the loop keeps the old bundle as explicitly degraded historical
evidence and continues. A missing auxiliary diff never forces profiling or
terminates the campaign.

Analysis is limited to two session attempts per commit across resume. The loop
does not issue the same Analysis request twice in one iteration, but a
failed request is eligible for the next iteration while the service still has
an attempt available. PARTIAL evidence may use the second attempt; an exhausted
commit continues with its published or checkpoint evidence.

The campaign is TIME-driven (`--max-hours`). When the search stalls, the loop
hands it to another Agent profile (see *Model relay* below), or the supervisor
injects fresh directions when no other profile is available, rather than
self-terminating on plateau.

## Round admission

A round is admitted against the time the campaign has left. The loop cannot
interrupt a session once it runs, so before a round starts its lanes it
requires what remains to cover one session and the canonical validation and
benchmark that judge the round's candidate: the larger of 19.6 minutes and
12.3 + 10 minutes. The lanes of a round run side by side, so its width does
not change the price.

When a round is refused, the campaign ends with termination reason
`round_budget_exhausted`, says so in the run summary, and still writes its
report. The published `optimization_report.md` then carries a `Round Budget`
section naming the campaign's wall-clock and the refusal.

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

## Model relay

Every new campaign relays its search between Agent profiles: one profile per
auto-selectable provider (`claude` runs Claude Code, `codex` runs Codex), the
`--agent-backend` provider first as the primary. `deepseek` (the DeepSeek
Harness) is not auto-selectable: it joins the relay only as the primary, when
the campaign starts on it through `--agent-backend deepseek`,
`FORGE_AGENT_BACKEND=deepseek` or `INFERENCE_OPTIMIZER_AGENT_BACKEND=deepseek`
(the last is how a Hyperloom run selects it). The campaign records only the
members and the primary. Each session resolves every
member's model: the primary takes `--model`; every member without one takes its
provider's `CLAUDE_MODEL` / `CODEX_MODEL` / `DSH_MODEL`, else the provider's
default model.
`--model`, `--agent-reasoning-effort` and the model variables therefore apply on
`--resume` as well. An `--agent-backend` (or `FORGE_AGENT_BACKEND`) that differs
from the campaign's primary is refused on resume; leave it unset to keep the
campaign's primary.

Before the first iteration, every profile's exact provider and model is checked
once, with one minimal model call per profile. Like the rest of the campaign
setup, the check runs before the `--max-hours` budget starts. A profile that
fails the check stays unavailable for the session, and the startup log gives
the reason. When only one profile passes, the log says model relay is off for
the session and how to turn it on.

A blockage is two consecutive rounds without a new best produced by the active
profile; the startup log states the count. Infrastructure, API,
device-contention and integrity failures do not count; a KEEP ends the
blockage. Every blockage takes the first of these the session can
run, checked against the startup check:

1. Switch to an available profile on a different harness and model (Claude Code
   and Codex).
2. Otherwise, switch to an available profile on the same harness with a
   different model.
3. Otherwise, run the supervisor below on the active profile.

A switch replaces the complete search actor -- the lanes' sessions and the merge
session -- at the next clean iteration boundary and forces the next round to
diversify. With both `claude` and `codex` available, the search
therefore alternates between them at every blockage until a KEEP.

A session starts on the profile a resumed campaign was running, else the
primary, else the `--agent-fallback-provider` member, else any available member.
With `--agent-fallback-provider none`, a session that has no profile to resume
does not start on a member other than the primary.

When a profile's API retries run out, its startup check runs again at the next
iteration boundary: the search keeps the profile if the check passes and
switches to an available one if it fails.

A profile without a gateway of its own borrows the other configured one: Codex
uses the `ANTHROPIC_*` line with `/v1` appended, and Claude uses the `OPENAI_*`
line without its trailing `/v1`. This works only on a gateway that serves both
protocols, and the startup check decides. Set the provider's own variables when
your gateway routes them differently.

The result's `agent_relay` field and the `relay_*` events in
`forge_experiments/events.jsonl` record the active profile, the availability of
every profile and each switch.

## Stalled-search supervisor

When the search plateaus and no other Agent profile is available (see
*Model relay* above), `forge-loop` escalates to a supervisor backend.
A campaign without relay escalates after the same number of stalled iterations
instead, at most once in that many; the startup log states which schedule
applies.

```bash
kernelforge forge-loop --workspace /work/aiter-amd --resume \
    --supervisor-backend codex
```

`--supervisor-backend` can override the provider of the active Agent profile
for this review; under relay it runs on that provider's profile.
The Supervisor inspects the stalled run and writes a free-form ruling for the
sessions of the rounds that follow, which read it directly. Every interaction is archived under
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
