---
myst:
    html_meta:
        "description": "Step-by-step migration guide for upgrading Hyperloom. Covers required and recommended changes from 0.5.x to 0.6.0 and the generic upgrade procedure."
        "keywords": "Hyperloom, upgrade, migration, version, release notes, 0.6.0, USER_DATA_PATH, GEAK, TraceLens, Ray, AMD GPU, ROCm, session, inference optimizer"
---
# Upgrade Hyperloom version

Per-version migration steps. This page is a companion to
[Releases](https://github.com/AMD-AGI/Hyperloom/releases): a release answers *what
changed*, this page answers *what you have to do about it*.

If you are starting fresh, skip this page and follow the
[installation instructions](../install/install.md).

---

## Conventions

These labels are used throughout this page to indicate urgency:

* **Required**: Your run will fail or behave incorrectly until you do
  this.
* **Recommended**: Your run will still work, but you'll get a
  deprecation warning or sub-optimal behavior.
* **Optional**: Strictly improves UX or unlocks new features.

Hyperloom doesn't mutate your `.env` on upgrade; all migrations
below are explicit.

```{note}
Shell paths on this page follow the recommended `pip install --target .` layout.
In a source checkout, replace the `hyperloom/` prefix with `src/hyperloom/`.
The command blocks assume `REPO_ROOT` points at that workspace. No installer
exports it for you, so run `export REPO_ROOT="$(pwd -P)"` from the workspace
first — it does not survive a new shell.

The `${USER_DATA_PATH:-/workspace/hyperloom}` fallback below is the last resort
in the chain and only correct where `/workspace` is writable; on a bare-metal
host the CLI defaults to `session/` under the current directory instead. Prefer
`KERNEL_AGENT_ENV`, which the installer writes for you.
```

---

## Upgrading from 1.2.x

### Required: removed flags

These flags are rejected with `unrecognized arguments`:

* `--proposal-scoring`, `--no-proposal-scoring` and `--proposal-scorer-models`
  (the advisory proposal scorer is removed; while it was off they changed
  nothing): drop them.
* `--critic-mock`, `--critic-agent`, `--critic-protocol` and `--critic-prompt`
  (the Critic agent is removed): drop them. See
  [the Critic agent is removed](#required-the-critic-agent-is-removed).
* `hyperloom-multibranch run --make-before-break` and `--no-make-before-break`
  (a hosted test now closes its sandbox when it ends): drop them.

A campaign spec may keep `multibranch.make_before_break`, which is ignored with a
warning. Where the grant fits three or more tests, a hosted session now runs one more
test at once than the default did, and each test's sandbox queues for a node.

`INFERENCE_OPTIMIZER_PROPOSAL_SCORER_TIMEOUT_SEC` is no longer read and is
silently ignored, so drop it.

### Required: `session_breakdown.json` drops `ensemble_scores`

The schema stays `v6.1`, but neither a specialist round's action row nor a
Framework run row carries `ensemble_scores` any more. Only a run launched with
`--proposal-scoring` wrote it. The action row is closed to other properties, so
a breakdown whose action row holds the key no longer validates against the
packaged schema. Consumers that read it should stop.

### Required: the integrate re-baseline starts with the session's memory budget

`HL_INTEGRATE_VRAM_GUARD` and `HL_INTEGRATE_VRAM_UTIL_CAP` are no longer read
and are silently ignored, so drop them. On vLLM, an integrate re-baseline now
starts its server with the session's own `--gpu-memory-utilization`, where it
appended `--gpu-memory-utilization 0.9` unless the integrate request set the
flag itself. To keep a 0.9 budget, pass `--gpu-memory-utilization 0.9` in
`--server-args`.

A launch that pins `--moe-runner-backend` on a checkpoint that has no runner
for that backend now fails its round with the server's error, where the round
was relaunched without the flag. Fix the pin where it was set.

### Required: the hosted single-session flow is retired

`hyperloom hosted run` now only runs the multi-branch flow, handing every run
to `hyperloom-multibranch hosted run`. In 1.2.0 a campaign spec with no `flow`
ran the single-session flow; that default is gone.

* A spec must now set `"flow": "multibranch"` or carry a `multibranch`
  section, and a multi-branch run needs `multibranch.workload`,
  `workload.conc` and `workload.precision`. An unchanged 1.2.0 single spec
  (neither `flow` nor `multibranch`) ends the run with a final `exit.json`
  (`"final": true`, return code 2) and the error
  `campaign spec refused: a spec with neither flow nor multibranch ran the single flow, which is retired; set flow to 'multibranch' and add a multibranch section with workload, and give workload.conc and workload.precision`.
* A spec that still says `"flow": "single"` ends the same way with
  `campaign spec refused: flow 'single' is retired; leave flow out or set it to 'multibranch'`.
* `placement.measurement` now defaults to `hosted` for every spec; the single
  flow defaulted it to `local`. Set `"measurement": "local"` (with
  `placement.profile` `hosted-gpu`) to measure on the application container's
  own cards.
* A single-flow run in flight does not resume on this release: restarted, its
  spec is judged as a multi-branch one and refused as above, and the single
  flow's session files are ignored. Let such runs finish on 1.2.0 before
  upgrading the host.
* The `hosted-gpu` and `hosted-cpu` plug profiles are removed:
  `hyperloom optimize --plug-profile hosted-gpu` (or `hosted-cpu`) exits 2
  with `ERROR: plug configuration refused: unknown profile 'hosted-gpu'; available: local`.
  Drop the flag.
* `HYPERLOOM_EVENTS_DIR`, `HYPERLOOM_MCP_HOST`, `HYPERLOOM_MCP_PORT` and
  `HYPERLOOM_MCP_TOKEN` are no longer read or checked; a host may stop
  setting them (any value is accepted and removed from the environment of
  every child). The single flow's MCP host tools and progress events are gone,
  and the Coordinator no longer reads `hosted/control/stop.json`; stop a run
  with a signal.
* `options.conc_sweep` is no longer part of the spec: it switched the single
  flow's sweep, and like any unknown field it is now accepted and ignored.
  Turn a multi-branch run's sweep off with `"multibranch": {"conc_sweep": false}`.
  The single flow's two other sweep behaviours go with it: restarted, the
  hosted entry no longer runs the unfinished sweep of a closed single-flow
  session, and that sweep no longer warns when its estimate outlasts the
  hosted grant.

### Required: `session_breakdown.json` drops the held Critic verdict

The schema stays `v6.1`, but a Critic ruling no longer carries `held_to_rule`,
`effective_verdict` or `failure_reason_code`: not in `critic_review` on a phase
or framework proposal, not on its `critic_review.variants[]` rows, and
`critic.iterations[].framework_reviews[]` no longer carries `effective_verdict`
or `failure_reason_code`. Consumers that read `effective_verdict` read
`verdict` instead: the loop no longer downgrades a ruling, so `verdict` is the
one it acted on.

### Required: `session_breakdown.json` drops `critic_review.variants`

The schema stays `v6.1`, but `critic_review` on a phase or framework proposal
no longer carries `variants`. A `review_verdict` rules on a proposal with one
`verdict`; its per-variant `verdict_map` form had no producer and is now
rejected as a `review_verdict` with no `verdict`. 1.2.x wrote `variants` as an
empty list on every reviewed phase proposal. Consumers that read it should stop.

### Required: `state.json` drops the policy denial log

`state.json` no longer carries `policy_denial_history` or
`policy_denial_streak`, the Orchestration prompt no longer has a
`=== Recent policy denials ===` block, and the `why_denied` context tool is
removed. A session written by an earlier release still resumes: the two keys
are dropped on load. A denial is still recorded where it was read from before:
as a `policy_denied` observation in the denied agent's inbox, as a `denials`
row on the phase event in `session_breakdown.json`, and as a
`PolicyGate denied intent` warning in the log. Anything that read the two
`state.json` keys reads the `denials` rows instead.

### Required: the Critic agent is removed

`hyperloom optimize` no longer runs a Critic. Orchestration is the only LLM
role that takes turns, and an admitted proposal is queued without a review.
KEEP or REVERT is decided as before, by the measurement gate (apply, benchmark,
accuracy) and the Coordinator's promotion.

Do these first:

* Drop `--critic-mock`, `--critic-agent`, `--critic-protocol` and
  `--critic-prompt` from every launcher: each is rejected with
  `unrecognized arguments` (exit 2). Nothing replaces `--critic-mock`: a
  session without a reviewer is now the only kind. `--orch-prompt` stays.
* Drop `INFERENCE_OPTIMIZER_DEFAULT_CRITIC_BACKEND`, `CRITIC_AGENT_ROOT`,
  `CRITIC_AGENT_PROBE_TIMEOUT_SEC`, `CRITIC_AGENT_MAX_COMPLETION_TOKENS`,
  `CRITIC_AGENT_LLM_CONNECT_TIMEOUT_S`, `CRITIC_AGENT_LLM_RW_TIMEOUT_S` and
  `CRITIC_SESSION_MEMORY_DIR`: they are no longer read. `install.sh` no longer
  reads `CRITIC_WEB_TOOLS_ENABLED`, and the `web` packaging extra
  (`markdownify`, `cachetools`) is removed, so drop `web` from any install
  command that names it.
* Stop calling `python -m hyperloom.agents.critic.runtime.cli`: the
  `hyperloom.agents.critic` package is deleted.
* Finish in-flight sessions under the build that started them: resuming a
  session started before this change is not supported.

Preflight no longer sends the review model a probe request, so the exit 2
ending `critic model ... did not answer ... Refusing to start` cannot occur,
and it no longer runs the critic runtime `--help` check. The launch banner
reads `Backends        : orchestration=<...>, kernel=<...>`. The stop reason
`backend_unhealthy` now only comes from the Orchestration backend, and its hint
no longer names `--critic-mock`. While the Orchestration backend is backing off
after a failed call (2 s, doubling to 5 minutes), the run loop wakes at least
every 10 s, the dispatcher's poll, so queued tasks are still dispatched and
finished ones booked between retries. Before, the loop slept until the first
retry when both LLM backends were failing, and ticked without a pause while the
Critic was idle or healthy.

What a proposal and a specialist deliverable now do:

* Nothing waits for a review. Orchestration queues every action with a
  `delegate`, which passes PolicyGate and the admission chain and is queued in
  the same step, announced by a `task_queued` event. The `propose_action`
  intent, which 1.2.x held until the Critic ruled, is removed: see
  [`propose_action` is removed](#required-propose_action-is-removed-and-delegate-queues-every-task).
* When a specialist finishes with a patch or a routable artifact, or an
  ENABLEMENT round finishes with a config-only deliverable, the Coordinator
  queues the `integrate_patch` for that deliverable itself. A deliverable has
  one idempotency key, `integrate_patch:<specialist_task_id>`, whoever asks.
* An `integrate_patch` that Orchestration asks for is that same task: it
  carries the deliverable's own params
  (its patch name, companion flags and environment, ENABLEMENT round context
  and owner), with anything else Orchestration sent beside them. It is denied
  with `policy_denied` rule `integrate_patch_specialist_unfinished` while the
  specialist it names has not ended with a result (or is no task of the
  session), and with `integrate_patch_no_deliverable` when that specialist
  left no patch or artifact on disk and is not an ENABLEMENT round. A
  `delegate` naming a deliverable whose `integrate_patch` is queued or running
  is denied with rule `duplicate_idempotency_key_running`; one whose
  `integrate_patch` already benchmarked the deliverable is denied with
  `duplicate_idempotency_key`; one whose earlier `integrate_patch` ended
  without a benchmark starts a new task. The same rule now answers every
  `delegate`: one that repeats a `baseline` that already measured is denied
  with `duplicate_idempotency_key` where it started a second baseline.
* Orchestration cannot queue an `integrate_patch` with
  `patch_source=upstream_pr`: it is denied with rule
  `integrate_patch_upstream_pr`. Only the FRAMEWORK_AGENT phase lands the
  upstream PRs it discovered.
* `integrate_patch` applies only the patches named by `params.patches` or by
  `specialist_done.patches_written`, and no longer scans the specialist
  workspace's `patches/` directories.
* The Coordinator routes each upstream-PR candidate that
  `candidate_discovery_specialist` found by the route discovery named, with no
  pre-screen: the apply track (`integrate_patch` with
  `patch_source=upstream_pr`), the author track (an authoring specialist), or
  both.

Update anything that matches on the following, which no longer occur:

* The intent type `review_verdict`; the bus topics `proposal`,
  `review_verdict` and `advice`; `critic:*` bus rows. The Orchestration inbox
  carries no `review_verdict` or `advice` message, and the context tool
  `get_recent_outcomes` returns `delegated_result` rows only.
* The `policy_denied` rule `integrate_patch_requires_critic_verdict`, the
  integrate result `status=rejected_by_critic`, and the 30-minute review
  timeout with its coordinator-authored reject.
* The `needs_review` re-author loop and its observations
  `specialist_reauthor_dispatched` and `reauthor_cap_reached`, and the
  framework candidate progress statuses `critic_denied`,
  `needs_review_no_evidence`, `repeated_review_abort` and `reauthor_cap`.
* The observations `framework_agent_candidate_submitted_for_review`,
  `framework_agent_verdict_received` and `verdict_for_unknown_proposal`, and
  the decision row `approved_proposal`: read the `task_queued` event instead.

Three observations are renamed:

| 1.2.x | Now |
|---|---|
| `specialist_patch_autosubmitted_for_review` | `specialist_patch_integrate_queued` |
| `framework_config_autosubmitted_for_review` | `enablement_config_integrate_queued` |
| `specialist_patch_autosubmit_skipped_no_files` | `specialist_patch_integrate_skipped_no_files` |

`state.json` no longer carries `specialist_patch_verdicts`,
`framework_agent_review_counts` or `specialist_reauthor_attempts`.

`session_breakdown.json` keeps schema `v6.1`; the packaged schema required none
of the keys that go:

* the top-level `critic` section;
* `timeline[type=phase].ext.proposals`, with its `count`, `reviewed` and
  `materialized` counters and its `rows`, including each row's `critic_review`
  and `outcome`;
* `critic_review` on the proposal rows of a `framework_agent` event, whose
  `lifecycle` no longer has a `reviewed` step. Every grid Orchestration sends
  has the lifecycle a delegated grid had, `proposed` then `attempted`, without
  the `routed` step a reviewed proposal had in between.

A consumer reads the task a proposal became from the dispatch row in
`ext.actions.rows[]` of the phase event, and a refused intent from a row in
`ext.denials.rows[]`. `ext.denials.rows[].proposal_msg_id` is still written,
because the packaged schema requires it; no intent type names a proposal any
more, so it is `null`. The two
`session_breakdown.json` entries above name keys inside the removed sections,
so a consumer that stops reading those sections has nothing left to do for
them. `hyperloom session report` has no "Review" section.

A session no longer writes `critic-workdir/`, `critic-session-memory/`,
`agents/critic/` or `reports/trace/proposal_task_map.jsonl`.
`reports/trace/llm_calls.jsonl` has no `component=critic` rows and no
`reviewed_msg_ids` key, and the trajectory ledger has no `proposal` event type.

### Required: `SWEEP` is renamed `CYCLE_END` and runs no concurrency sweep

The phase between `KERNEL_AGENT` and `CLOSE` is now `CYCLE_END`. It integrates
pending KEEPs and validates the stack, then loops back or closes; it no longer
dispatches a `conc_sweep` task. Update anything that matches on the old names:

* The phase is `CYCLE_END` in `state.json`, `phase_history` and
  `session_breakdown.json`.
* It has no phase budget and no budget flag: the session budget alone gates it.
  The SWEEP budget flag is rejected with `unrecognized arguments`, and so is
  `--conc-sweep-total-budget-sec`. The share SWEEP held moves to the
  optimisation phases: `FRAMEWORK_AGENT` defaults to 0.40 and `KERNEL_AGENT` to
  0.50, what a session with the sweep off already ran with.
* Its exit is `cycle_end_done` (also the stop reason when no further cycle
  runs); the SWEEP stop reasons no longer occur.
* Its predicate input is `cycle_end_result` (`settled`, `reloop`) instead of
  `sweep_result`.
* A session that reached it reports `outcome.stage_reached` `cycle_end` where it
  reported `conc_sweep`.
* The workflow-evaluation contract is `hyperloom.workflow_evaluation.v4`
  (`workflow_contract.v4.json`); a consumer that matches the version string or
  the SWEEP vocabulary moves to it.

The concurrency sweep runs after CLOSE instead of inside the session.
`--enable-conc-sweep` is on by default (`--no-enable-conc-sweep` turns it off),
and `hyperloom conc-sweep SESSION_DIR` runs it by hand. It writes the same files:
`reports/conc_sweep_summary.json`, `reports/conc_sweep_raw.csv`,
`reports/conc_sweep_curve.png` and the `conc_sweep_summary` /
`conc_sweep_curve_png` keys of `final.json`, and packages the session again so
the package carries them. The `conc_sweep` timeline event of
`session_breakdown.json` and the sweep's result in `state.json` are gone. A
session recorded in `SWEEP` does not resume: finish it under the build that
started it.

The sweep also adds its report to `session_breakdown.json` as the top-level
`conc_sweep_summary`, rewritten after every point. A consumer that read the
sweep from the `conc_sweep` timeline event reads it there instead, and has to
read the breakdown again after CLOSE to see it. The key is the one breakdowns
carried the sweep under before #1455, but the section is the schema `2.0`
report: its point rows, `comparison` and `summary` keep their fields, while its
header names the ladder `concs`, drops the budget keys, `was_skipped` and the
report paths, and adds `partial`, `in_progress` and `cancelled` to `status`.
Both lineages' shapes are in
[`session_breakdown.json` integration in Hyperloom](session-breakdown.md).

### Required: `hyperloom session report` is removed

`hyperloom session report` (and `python -m
hyperloom.inference_optimizer.tools.dump_session_report`) no longer exists:
the command prints the `hyperloom` usage and exits 2, and the module fails
with `No module named`. Nothing writes `session_report.md`,
`session_report_prompt.json` or `session_report_llm_raw.txt` any more; a
session never wrote them on its own, only this command did. Drop the command
from scripts, and read `reports/final.md` for the session's own summary and
`session_breakdown.json` for the recorded facts. Neither file changes.

`HYPERLOOM_REPORT_LLM_BACKEND`, `HYPERLOOM_REPORT_MODEL` and
`HYPERLOOM_REPORT_MAX_TOKENS` are no longer read and are silently ignored, so
drop them.

### Recommended: stop matching the `role` and `custom_specialist_only` denial rules

A `policy_denied` observation, and the `denials` row it becomes on a phase
event in `session_breakdown.json`, no longer carries rule `role` or rule
`custom_specialist_only`. Neither could fire in 1.2.x once Orchestration was
the only agent taking turns: `role` refused an unknown agent or an intent type
outside an agent's set, and Orchestration's set was every intent type;
`custom_specialist_only` refused Orchestration's `specialist`, `explore` and
`integrate_patch` in FRAMEWORK_AGENT under `HYPERLOOM_CUSTOM_SPECIALIST_ONLY=1`,
where Orchestration takes no turn. Anything that counts or alerts on those two
rule ids can drop them; every other rule id is unchanged.

The `emit_intent` tool no longer refuses a call with `this role cannot emit`.
An intent type outside the vocabulary is refused as before, by the tool schema
and with `unknown intent_type`. The tool's schema, the output-format block and
Orchestration's tool list do not change.

### Recommended: stop reading the records that nothing produced or read

`hyperloom optimize` no longer writes a set of fields, rows and log lines that
had a writer and no reader, or a reader and no writer. A session behaves as it
did; what changes is what is left behind for anything that inspects a session
by hand.

In `state.json`:

* `baseline_eager_fallback` is gone. It armed a one-shot retry of the baseline
  with CUDA graphs disabled after a `cuda_graph_capture_failed` baseline, and
  nothing returns that error class, so the flag was always `false`. A baseline
  whose server needs CUDA graphs off takes the flag in `extra_server_args`, as
  before.
* A row of `kb_stage_outbox` or `kb_stage_dead_letter` no longer carries
  `include_patches`; it was `true` on every row. A row names a realized diff
  by its canonical path, as it already named a delivered patch, and the
  `host_origin.sources` of the staged provenance row carries the same path.
* `last_baseline` never has the decision `discarded`, and its
  `anchor_held.reason` is never `no_measurement`. Neither could be written: a
  baseline is promoted only with a positive throughput. (The roofline and
  profile rows that could not be `skipped` or `discarded` either are removed
  altogether; see the section on a task's result below.) A failed roofline is
  still counted in `roofline_failure_streak`.

In `storage/coordinator.db`:

* The `round_events` table and its index are gone. It recorded every open,
  renew, handoff and settle of a bring-up round and nothing read it. A refused
  attempt is still answered to its caller with the reason, and
  `bringup_rounds` still holds each round's state and outcome.
* `events` has no `in_reply_to` column, and the payload of a `response` event
  has no `in_reply_to` key. Neither was read; a reply still names what it
  answers in `kind` (`<request kind>_done`). Orchestration's inbox line for a
  `response` is shorter by that key.
* A kernel step that runs inline no longer appends a `kernel_step_running`
  observation every 150 seconds, so `hyperloom session events` and the
  `event_counts_by_topic` of the final report count fewer `observation` rows
  for a session with long kernel steps. The step's liveness stamp is unchanged.

A database written by 1.2.x keeps its `round_events` table and `in_reply_to`
column; this release leaves both alone and writes neither.

In `agents/orchestration/mcp_setup.json`, `emit_intent.registered` is gone.
It was `false` only for a backend that had started without the tool's server,
which now does not start (see below). `emit_intent.offered` still says whether
Claude Code offered the tool to the model.

In the log, three lines no longer appear: `dispatch audit: ... has no
registered executor` (the task still ends `failed` with `no_executor`),
`dispatcher: reclaimed N running task(s) with dead holders`, and `patch kb:
KEEP at stack index N references missing patch members` (a KEEP whose patch
files are gone is still dropped to `kb_stage_dead_letter` with a warning).

Five behaviours change at the edges:

* After a restart, the tasks the previous process left `running` are failed by
  the reconciler at the start of the tick, as before; the dispatcher no longer
  repeats the same query later in that tick. If the reconciler's query fails
  on a database error, the reclaim is retried on the next tick instead of
  later in the same one. The failure charged for such a task (`error_class`
  `dead_holder_reaped`) ends `task reclaimed by the reconciler`; 1.2.x ended
  it `task reclaimed by dead_holder_pump`.
* A delegate whose key names a task another author created is answered like
  any other repeated key: with `duplicate_idempotency_key` or
  `duplicate_idempotency_key_running`, or with a new task under
  `<key>-retry<n>` when the earlier one ended unmeasured. 1.2.x raised
  `idempotent task dispatch_class mismatch`, recorded as a
  `handle_intent_exception` observation. The existing task keeps the dispatch
  class it was created with.
* `get_inbox` shows the text of Orchestration's own `alert` as it was written;
  1.2.x rewrote prompt-structure markers in it. Orchestration is the only
  author of an `alert`, and its own rows are not part of the inbox it is sent
  each turn.
* A baseline result whose `output_throughput` is a numeric string anchors as
  the number it spells. 1.2.x admitted such a result to promotion and then
  recorded it `discarded`, neither anchoring it nor counting a failure. The
  optimizer's own executors report the throughput as a number.
* A `claude-agent-sdk` that lacks `tool` or `create_sdk_mcp_server` ends
  `hyperloom optimize` at start-up with `claude_agent_sdk loaded but missing
  ...`. 1.2.x started the session without the `emit_intent` tool, and it ended
  later on turns that emitted no intent. Every SDK the `llm` extra allows
  (`>=0.2.110`) has both.

### Required: `state.json` drops its second copies and its prompt-only counters

`hyperloom optimize` kept several facts twice, and a few counters only to
print one line of the Orchestration prompt. Each fact now has one copy.
Anything that reads the removed copy reads the one that stays.

In `state.json`:

| Removed key | Read instead |
|---|---|
| `current_best.optimization_stack` | `optimization_stack` at the top level. Every KEEP appended to it and rewrote `current_best` in the same step, so the two were equal. |
| `explore_search.accepted` | The rows of `optimization_stack` with `action` `explore` for the variants the session adopted; `explore_search.winners_history` for every variant a round measured KEEP. The removed list also held winners the KEEP gate went on to refuse. |
| `rejected_kernel_patches` | `kernel_integrate_attempts.<key>.rejected`, the same record under the patch's key. |
| `enablement.observed_accuracy`, `observed_task`, `observed_metric`, `eval_definition_digest`, `baseline_eval_evidence` | The `trigger` block of the enablement event in `session_breakdown.json`, which is where they were copied to. `enablement.baseline_eval_kind`, `probe_config_path` and `accuracy_floor` stay. |
| `research_scout_seen_pr_ids` | Nothing; see the discovery change below. |
| `params_no_promote_streak` | `explore_search.winners_history` and `attempts`. |
| `gain_gated_action_count` | Nothing read it. |
| `agent_last_active` | Nothing read it. |

`reports/final.json` copies `current_best` from `state.json`, so its
`current_best` no longer carries `optimization_stack` either, in the full
report and in the crash-safe one. `optimization_stack_len` stays.

`storage/coordinator.db` no longer has a `lane_capacity` table. The capacity of
`research_lane` is the session's `--research-lane-capacity`, held in
`state.json` as `research_lane_capacity`; every other lane admits one holder.
Editing the table during a session, which 1.2.x read on every lease, no longer
changes a capacity. A database written by 1.2.x keeps the table; this release
neither reads nor writes it.

The Orchestration prompt changes with them:

* The `=== Tick ===` block is `tick=N`. The `agent_last_active=` line under it
  is gone.
* `elapsed=` in the `=== Time budget ===` block is the time the session has
  been charged across every run of it, so `elapsed` plus `remaining` is the
  budget on a resumed session. 1.2.x printed the time since the current
  process started next to a `remaining` that already counted earlier runs.
* The `=== Shared session state ===` block has no `params_no_promote_streak=`
  or `gain_gated_action_count=` line, and the first line under
  `explore_search=` has no `accepted=N`. `rejected_kernel_patches=` prints the
  same five most recent rejections as before.

Six behaviours change:

* No `<workload>#explore_plateau` gap is created. 1.2.x added it to
  `state.json` `gaps`, the `=== Current gaps ===` block, `get_gaps` and the
  `remaining_gaps` of a published recipe after three explore tasks in a row
  set no new `current_best` and none of the last five recorded winners had a
  gain. Whether the configuration search has run dry is still judged from
  `attempts`, and the plateau advisory still reports it.
* A research-scout task's `params` have no `seen_pr_ids`; the specialist was
  never shown them. Its `already_proven` list names the warm recipe's proven
  items and the explore variants on `optimization_stack`; 1.2.x listed every
  winner an explore round reported, adopted or not.
* FRAMEWORK candidate discovery drops a proposed pull request as already known
  only when an earlier discovery batch holds it. 1.2.x also dropped one whose
  URL a research hint gave as its `source`, a specialist proposal cited in
  `pr_evidence` or `source_evidence`, or a specialist listed in the `research`
  block of its result (`prs_fetched`, `pr_diffs_read`, `nvidia_refs`), so a
  pull request a specialist had only read could never become a candidate. The
  `research` block is no longer read, and the research scout's prompt no
  longer says the Coordinator de-duplicates by pull-request id.
* The `trigger` block of the enablement event in `session_breakdown.json`
  reports `observed_accuracy: null` when the failed baseline eval scored
  nothing (an `eval_runtime_failure`, for example). 1.2.x reported `0.0`.
* A queued task that still holds a row on a lane it needs is dispatched; the
  lease refreshes its row. 1.2.x counted the task's own row against the lane
  and left it queued on every pass. Two tasks that need the same lane, or
  lanes that exclude each other, still never run together.
* `explore_search.rejected` is removed as well (see the section on a task's
  result below), so a variant that a later round measures KEEP has one row,
  its `tested` row, and that row says KEEP. In 1.2.x its `rejected` row
  stayed behind whenever the KEEP came from a stack-revalidation round.

The log no longer prints `depth: aggregated N new research reference(s) into
seen-set`, and `specialist findings harvested` reports `hints_added` only.

### Recommended: stop reading the discarded-hint record, and stop sending `skip_to_sweep`

`escalate_strategy_change` has one hint Orchestration may send,
`next_action_hint='skip_to_kernel'`, accepted in FRAMEWORK_AGENT. The
Orchestration prompt already offered only that one. The hint is checked once,
when the intent arrives, and a phase transition clears a pending hint without
keeping a record of it.

* `state.json` no longer has `last_discarded_escalate_hint` or
  `last_discarded_escalate_hint_ts`. `pending_escalate_hint` stays, and is
  empty after every phase transition, as before.
* The Orchestration prompt has no `=== Discarded escalation hint (advisory)
  ===` section. 1.2.x showed it for one turn after a phase transition dropped
  a pending hint, and asked for the hint to be sent again. By then the session
  had left the phase the hint was about, so a re-sent `skip_to_kernel` was
  denied (`escalate_hint_phase`); and the section also reported a
  `skip_to_sweep` the Coordinator had set itself as Orchestration's own. The
  Phase block and the `phase_transition` event report the transition, as
  before.
* `escalate_strategy_change{next_action_hint='skip_to_sweep'}` from
  Orchestration is denied with rule `escalate_hint_not_in_vocab`. 1.2.x
  accepted it in every phase, and in FRAMEWORK_AGENT it ended the phase with
  no round run, which `skip_to_kernel` does not. The Coordinator still sets
  `skip_to_sweep` itself when a custom-specialist-only cycle
  (`HYPERLOOM_CUSTOM_SPECIALIST_ONLY=1`) has run all its work; the exit
  evidence and the `hint` of `predicate_inputs` in `session_breakdown.json`
  record it as before, and the values `hint` can take do not change.
* The `hint` of an `escalate_hint_not_in_vocab` denial reads `skip_to_kernel,
  sent in FRAMEWORK_AGENT, is the one hint; the run closes itself.` Rule ids
  `escalate_hint_not_in_vocab` and `escalate_hint_phase` are unchanged.
* The log no longer prints `phase_machine: discarded stale
  pending_escalate_hint=...`.

### Required: a malformed timeout or retry variable stops the launch

`INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC`,
`INFERENCE_OPTIMIZER_CLAUDE_TURN_TIMEOUT_SEC`,
`INFERENCE_OPTIMIZER_CODEX_CALL_TIMEOUT_SEC` and
`INFERENCE_OPTIMIZER_DEEPSEEK_CALL_TIMEOUT_SEC` must be a positive finite
number of seconds. `INFERENCE_OPTIMIZER_LLM_RETRY_ATTEMPTS`, `_BASE_S`,
`_MAX_S`, `_MULT` and `_JITTER_S` must be a non-negative finite number. Any
other value ends `hyperloom optimize` with an `EnvValueError` that names the
variable, when the Orchestration backend is built at start-up. 1.2.x logged a
warning and ran on the default, so a session could run to its end on a
timeout its operator had not set. Unset and blank still mean the default, and
`INFERENCE_OPTIMIZER_LLM_RETRY_ATTEMPTS=0` still means one attempt.

Check the launchers and environment files that set any of the nine, and remove
or correct a value such as `0`, `-1`, `inf` or `30s`.

### Recommended: stop matching the denials the Coordinator no longer issues

Checks that refused an intent for its shape, or that repeated a decision
another component owns, are removed. Each input is now handled by the code
that acts on it, and none of them is recorded as a failed action.

* `send_message` is never denied with rule `payload`. A blank `topic` is
  stored under `observation`, as an unknown topic already was. A `to` that is
  an empty string or an empty list broadcasts. Any other `to` is addressed as
  written: one message per list entry, each name converted to a string. 1.2.x
  denied a blank topic, and a `to` that was neither a non-empty string nor a
  non-empty list of them.
* `prune_branch` is never denied with rule `prune_scope`. `scope='family'`,
  or no scope, retires the family as before. Every other value cancels the
  family's queued tasks and leaves the family usable, as `scope='queued'`
  does. A blank `family` is still denied (`payload`).
* `extend_lease` is never denied with rule `extend_lease_task_not_live`. An
  extension for a task that is not running is refused by the task registry
  and reported as an `extend_lease_rejected` observation, with the task's
  state in `error`; that was already the outcome when the task finished
  between the check and the extension. It has no row in the `denials` of a
  phase event in `session_breakdown.json`. The removed check read the running
  tasks at the start of the tick, so it also refused a task that had started
  during the Orchestration turn; that extension is now granted.
  `extend_lease_bounds` is unchanged.
* A `delegate` is never denied with rule `execution_order`, which also
  refused the removed `propose_action`. The rule refused `explore`,
  `integrate` and `report`
  before a baseline, and the phase table already does (`phase_incompatible`):
  a session with no baseline is in PRELUDE or ENABLEMENT, which offer none of
  the three, or in CLOSE, where the rule did not apply. A kernel `request`
  before a baseline is still denied with `execution_order`.
* The Coordinator no longer broadcasts the observations
  `stale_stack_revalidation` and `current_best_drift`. A stack revalidation
  that measured an older recipe generation is still discarded, with the same
  `not validating` warning in the log. A current one still restamps
  `cumulative_gain_validated`, which is where a lower re-measurement shows.
* The Claude backend keeps every `emit_intent` call of a turn. 1.2.x dropped
  a call whose input arrived in Claude Code's `__unparsedToolInput` wrapper
  when the turn had already carried the same intent, and logged `claude
  fallback intent deduped`. Both forms of the wrapper are still decoded.

### Required: five Orchestration context tools are removed

Orchestration's read-only context server no longer mounts five tools. Four
returned a block every turn's prompt already carries, and the fifth returned a
file the prompt now names. A model that calls one gets a tool error inside its
turn; no session state changes.

| Removed tool (`mcp__inference_optimizer_context__...`) | Where the same content is |
|---|---|
| `get_mission_status` | The `=== Mission progress ===` block of every turn. |
| `get_shared_state` | The `=== Shared session state ===` block of every turn. |
| `get_warm_start` | The `=== Warm start (Recipe KB T0) ===` block of the session context. |
| `get_intervention_mix` | The `=== Intervention mix (telemetry) ===` block of the session context. |
| `show_analysis_md` | The file on the `analysis_md=` line of the session-state block, which is `last_trace_analyze.analysis_md_path` in `state.json`. |

* If you pass your own Orchestration prompt with `--orch-prompt` and it names
  one of the five, remove the name: the tool is neither mounted nor allowed.
* `agents/orchestration/mcp_setup.json` lists eight context tools in
  `allowed_tools` and `mcp_tools`: `get_gaps`, `get_runtime_findings`,
  `get_inbox`, `get_recent_outcomes`, `get_running_tasks`, `read_reference`,
  `get_failure` and `get_variant_failures`. Those eight and their arguments
  are unchanged; the section on a task's result below lists what four of them
  answer differently.
* `show_analysis_md` replaced each embedded base64 image of `analysis.md` with
  a placeholder. The file itself holds the images as TraceLens wrote them.
* `last_trace_analyze` in `state.json` no longer has `analysis_md_text`, the
  whole text of `analysis.md`, embedded images included, which only
  `show_analysis_md` returned and every save rewrote. Read the file at
  `last_trace_analyze.analysis_md_path`. The new
  `last_trace_analyze.has_analysis_md` is `true` when that file held anything
  at the time the roofline was recorded, and `false` for a report that was
  missing or empty, which 1.2.x stored as an empty `analysis_md_text`.
* In the session-state block, the `analysis_md=` line of a session with a
  TraceLens snapshot ends `Report file: <path>` where it named the tool.
  Before the first snapshot the line said that `roofline` and `profile` could
  also be proposed directly; no phase admits that proposal
  (`phase_incompatible`), and the line now says it is denied.

The built-in Orchestration system prompt changes with them. None of this
changes what the Coordinator admits or dispatches:

* The `## 3. PIPELINE & TIME BUDGET` section is gone, and `## 3a. PHASE
  CONTRACT` is `## 3. PHASE CONTRACT`. The section summed a typical runtime per
  action group against the session budget (`--max-hours`) and advised skipping
  analysis or adding an explore round before the report, both of which the
  Coordinator schedules.
  The Phase block of every turn carries the remaining budget, and each
  catalogue entry its own typical cost.
* The catalogue entry of `roofline` shows no proposal template; it showed
  `propose_action{action_name='roofline', ...}` beside the rule that forbids
  proposing it.
* The prompt no longer says that the session-state block carries the full
  `analysis.md`, that dispatches are deferred while a roofline refresh runs,
  or that a `=== Phase-allowed actions ===` block, a working-memory block or a
  `KB hints` block exists. It names `untried_hot_kernels` and the `tested:`
  rows of `explore_search`, which are what a turn shows, where it named
  `untried_hot_reusable_kernels`, `explore_search.winners_history`,
  `explore_search.rejected` and `<action>_attempts`.

### Recommended: stop reading the pruning counts; the session database keeps every row

The Coordinator no longer deletes anything from `storage/coordinator.db`.
1.2.x deleted rows on every maintenance pass (every 30 minutes) and at the
start of each macro-cycle: the rows of `events` older than the newest 5,000,
except a `proposal` the Critic had not ruled on yet, and every `succeeded` or
`cancelled` row of `tasks` but the 2,000 most recently updated.

The 5,000 counted events of every topic, so in a session that wrote more, the
deleted rows included `delegated_result`, the record of what each task
returned. In 1.2.x `--resume-from` read those rows to replay a KEEP whose
promotion a crash interrupted, and `reports/final.json` counts them: such a
session resumed without its earlier results and reported only its newest
events.
Deleting a finished task row also released its idempotency key, so the same
action could have been queued a second time.

* The `events` and `tasks` tables now hold the whole session, and the database
  grows with it: a session of 17 hours, 3,361 events and 399 tasks left a
  database of 15.7 MB, about 20 MB a day. Nothing needs to be configured: no
  flag or environment variable set the two limits.
* A session that runs for about three days or longer is packaged without its
  database. The must-ship files of a session package share a ceiling of 64 MB,
  and `storage/coordinator.db` is left out once it is larger than that. In
  1.2.x the two limits kept the file under the ceiling (it levelled off near
  54 MB at the row sizes of one recorded session); at 20 MB a day it now
  passes 64 MB on the fourth day. The package says so: `PACKAGE_MANIFEST.json`
  has `complete: false`, `truncated: true` and `reserved_overflow: true` and
  names the file in `dropped_files`, `PACKAGE_MANIFEST.txt` carries the line
  `!! RESERVED PARTITION OVERFLOWED`, and the log has an error that starts
  `session package: MUST-SHIP artifact storage/coordinator.db`. Shortly before
  that, a must-ship file listed after the database (`state.json`,
  `manifest.json`, `session_breakdown.json`, `reports/final.json`) is the one
  left out when it no longer fits under the ceiling beside it. Nothing is
  removed from the session directory: read the database and those files there
  when a package reports the overflow.
* `events_pruned` and `tasks_pruned` are gone from the `maintenance tick N:
  {...}` line of the log and from the payload of the `cycle_soft_restart`
  observation. Stop parsing them; the other keys are unchanged.
* `event_counts_by_topic` and the highlights of `reports/final.json` and
  `reports/final.md` are taken from the newest 10,000 events. In 1.2.x they
  covered what the last pass had left: the newest 5,000 events and those
  written since.
* `hyperloom session events --all` counts every event of the session, where it
  counted what the last pass had left. Without `--all` it still reads the
  newest 500.
* `get_inbox` called without `since_seq` returns every event of the session
  that is addressed to Orchestration or broadcast, where it returned what the
  last pass had left. Pass `since_seq` to page.
* The `resource_lock` warning for a lane held under an id that is no task (a
  kernel request holds its lanes under its message id) ends `the holder has no
  task row, so nothing can attest it ended`. It ended `the holder's task row
  no longer exists, ...`, which described a row the pruning had deleted.

### Required: a task's result is kept once, on its task row

`hyperloom optimize` wrote what a finished task returned to several places:
the task's row, the `delegated_result` event that announces it, and for some
actions one or two lists in `state.json`. It now writes it to the task row,
and each of the other places either goes or keeps only what the row does not
hold. Anything that read a removed copy reads the row.

In `storage/coordinator.db`:

* The payload of a `delegated_result` row of `events` is
  `{"task_id", "kind", "state"}`. It no longer carries `result` or `error`.
  Both are in the task's row, in the newest entry of `tasks.history` that has
  `evidence.outcome`: `outcome.result`, `outcome.error` and
  `outcome.error_class`, written before the event:

  ```bash
  sqlite3 "$SESSION_DIR/storage/coordinator.db" "
    SELECT e.seq, json_extract(e.payload, '\$.kind'), json_extract(h.value, '\$.evidence.outcome.result')
    FROM events e JOIN tasks t ON t.task_id = json_extract(e.payload, '\$.task_id'), json_each(t.history) h
    WHERE e.topic = 'delegated_result' AND json_extract(h.value, '\$.evidence.outcome') IS NOT NULL
    ORDER BY e.seq"
  ```

  The event is still written once per booked result, in the same order, and a
  task whose attempt was retried instead of booked still has no event.
  `hyperloom session events` prints the same `delegated:<kind>:<state>` counts.
* The event's copy of an authored patch's result carried `reauthor_attempt`.
  Read `params.reauthor_attempt` of the task row.
* The database grows more slowly: in one recorded session of 21 hours the
  result copies were 3.5 MB of a 19.2 MB database.
* An open bring-up round no longer holds a row of `leases` under the lane
  `bringup_round`. The round, its holder and its expiry are in
  `bringup_rounds`, as before. A database written by 1.2.x for a session that
  stopped while a round was open still holds such a row, and `--resume-from`
  refuses that session, naming the holder task with `lane='bringup_round'` and
  `execution owner scope or PID is unobservable`. Once you have checked that
  the task's processes are gone, delete the row:

  ```bash
  sqlite3 "$SESSION_DIR/storage/coordinator.db" "DELETE FROM leases WHERE lane = 'bringup_round'"
  ```

  The task's other leases are then checked as on any resume. `hyperloom
  recover --session-dir <session> --confirm-stopped <task-id>
  --confirmation-reason <reason>` removes the row only when it is the only
  lease the task holds. A session that stopped in the middle of a round
  usually still records the task's execution lease too; `--confirm-stopped`
  then fails with `confirmation is limited to strictly empty owner scope on
  every target lease`, and `--confirm-scope-gone` is refused on the round row.

In `state.json`:

| Removed key | Read instead |
|---|---|
| `baseline_attempts` | `last_baseline` for the latest attempt and the decision on it; the rows of `tasks` with `kind = 'baseline'` for every attempt and what it returned; the baseline rows of `session_breakdown.json` for each attempt's decision. |
| `last_profile`, `profile_attempts` | `last_profile_trace`, `last_profile_status` and `last_profile_args` for what the latest profile left; the `profile` rows of `tasks` for every attempt. |
| `last_roofline`, `roofline_attempts` | `last_trace_analyze` and `roofline_snapshots` for what each roofline produced, `roofline_failure_streak` for failures in a row; the `roofline` rows of `tasks`. |
| `last_explore`, `explore_attempts` | `explore_search.last_round` and `explore_search.tested` for what the rounds measured; the `explore` rows of `tasks`. |
| `explore_search.rejected` | The rows of `explore_search.tested` whose `outcome` is not `KEEP`. Each rejected row was a shorter copy of its `tested` row. |
| `warm_history_injected` | Nothing read it but the copy below. |
| `failures` | The files `reports/failures/<failure_id>.json`, one per failed explore variant, which were written beside it. |

`last_baseline` stays as it was, `extras.anchor_held` included: a result is
announced before the write-back decides on it, so this is the only record of a
re-baseline that measured and did not move the anchor. `last_action_failures`,
`explore_search.last_round`, `gain_per_stack_entry`, `last_gemm_tuning` and
`gemm_tuning_attempts` stay too.

The rows of a warm recipe's `what_failed` are no longer copied into
`explore_search.rejected` after the first baseline. They are still in
`warm_start_context.do_not_repeat` and in the `avoid` line of the warm-start
block. No check read the copies: a variant is answered without a launch from
its `tested` row only.

In `reports/`:

* Each `delegated_result` entry of `highlights` in `final.json` has the
  three-key payload above, and its `summary` is `kind=<kind> state=<state>`.
  1.2.x embedded the whole result in `payload` and ended the summary with
  `tput=` and `decision=`.
* The stop report of a `baseline_failed` session (`failure_summary` in
  `final.json`) is built from `last_baseline`, which holds the same fields as
  the last row of the removed list.
* The emergency `final.md`, written when CLOSE did not finish, has a `## Last
  baseline attempt` section with the `last_baseline` bullet. It was `## Last
  action attempts` with three bullets.
* A failure packet that cannot be written to `reports/failures/` is now an
  error, and the booking of that explore round stops there. When the tick
  itself was booking the round, it records the exception in `state.json`
  `last_tick_exception` (stage `tick_body`) and counts it in `crash_count`.
  When the round was booked while an Orchestration turn was running, the
  exception is only the log line `turn_pump: the dispatcher pump beside a
  reactor turn raised; paused until the turn ends`, with its traceback, and
  nothing is written to `state.json`. 1.2.x logged it at debug level and
  kept the copy in `state.json`.

In `session_breakdown.json`, `outcome.stage_reached` of a session that ended in
PRELUDE with no typed timeline event no longer counts a `profile` task that
failed, which PRELUDE runs in place of the roofline under
`--no-enable-roofline`: a session whose only profile failed reports the stage
before it (`warm_replay` or `baseline`), where 1.2.x reported `profile` from
the failed attempt's row. A roofline that failed is still reported as
`roofline`, from `roofline_failure_streak`.

What Orchestration reads changes with them:

* A `delegated_result` line of the inbox, of `get_inbox` and of
  `get_recent_outcomes` no longer prints `kept=` or `tput=`, and `status=` and
  `gain=` are taken from `status` / `runner_status` and `gain_pct` /
  `delta_pct` only. No executor writes the other names the line looked for
  (`verdict`, `outcome`, `validated_gain_pct`, `predicted_gain_pct`,
  `tokens_per_s`, `tput`, `throughput`, `tput_tok_s`, `kept`), so no recorded
  line carried them. The rest of the line, and the failure and skip rows
  under it, are the same text, read from the task row.
* The `=== Shared session state ===` block has no `last_profile=`,
  `last_explore=` or `attempts_history=` line. `last_baseline=` and
  `last_gemm_tuning=` stay. The first line under `explore_search=` has no
  `rejected=N`.
* The `=== Acceptance threshold (advisory) ===` block lists each measured
  variant once. 1.2.x listed every variant that was not a KEEP twice, and
  listed the warm recipe's failed rows that carried a gain.
* `get_failure` returns any packet of the session. 1.2.x answered `not in
  memory` for all but the 30 newest and named the file to read.
  `get_variant_failures` lists from the same files, newest first, up to
  `top_k` (50 at most), so it also reaches past the 30 newest.
* In a `=== Tasks in flight ===` line, `lanes=[...]` no longer lists
  `bringup_round` for the task that holds an open round, and
  `lease_expires_in_sec` counts down to the expiry of the task's own lanes.
  The lanes an `extend_lease` observation reports as refreshed no longer
  include it.
* The failure-recovery reference (`read_reference('failure_recovery')`) and
  the FAILURE RECOVERY block name `last_baseline` and `last_action_failures`,
  where they named `last_<action>` and `<action>_attempts`.

Four behaviours change on `--resume-from`:

* A KEEP whose promotion a crash interrupted is adopted from the task row and
  the `pending_integrate` sentinel alone. This is where 1.2.x looked second;
  it first searched every `delegated_result` event for a `kept` result. The
  row is written before the event and holds the same result, so the same
  KEEPs are adopted, and one whose event was never written is too.
* The resume no longer searches the events for a `kept` result that is not on
  `optimization_stack`. That search re-tried on every start a KEEP the
  promotion gate had refused, and after `--reset-state` replayed the KEEPs of
  the discarded state onto the new one. The `resume_consistency` observation
  therefore never carries `replayed_orphaned_keep`, `orphaned_keep`,
  `orphaned_keep_discarded` or `orphaned_keep_replay_noop`, and no
  `orphaned_keep` observation is sent.

  With `--reset-state` this leaves the patches the discarded state had kept
  on the framework tree (in a git tree, as the commits whose subject starts
  `hyperloom KEEP`), and nothing in the new state records them:
  `optimization_stack` starts empty, no stack rebench is queued, the next
  baseline is measured on a tree that carries them, and the recipe the
  session publishes has no step for them. 1.2.x adopted them again from the
  event log and queued a rebench of the stack. Put the framework tree back
  to what it was before the session before you pass `--reset-state`; the
  `optimization_stack` of the backup `state.json.preReset.<unix time>` lists
  what was kept. Otherwise the new baseline includes those patches.
* A database error while the open integrate window is read ends the start-up
  with that error, and no stop reason is set. 1.2.x ended it with `pending
  integrate recovery is incomplete; refusing resume before measurement` and
  `stop_reason=patch_recovery_incomplete`. Either way the window is left as
  it was.
* An authored patch whose outcome was booked while its phase had moved on is
  recovered from the task rows of the booked results, not from the events'
  copies; the same rows are recovered.

### Required: `propose_action` is removed and `delegate` queues every task

Orchestration had two intents that queue an action. `propose_action` waited
for the Critic in 1.2.x; with the Critic gone it created the same task as
`delegate`, seeded and keyed a little differently, and for a `specialist` it
skipped the specialist contract and took only `research_lane`, so a bench
specialist sent that way could start beside a serving benchmark. `delegate` is
now the only one, and a delegate names its task by the work it asks for.

Do these first:

* In an Orchestration prompt you pass with `--orch-prompt`, replace every
  `propose_action{action_name=..., predicted_gain_pct=...}` with
  `delegate{action_name=..., params={...}}`. A `propose_action` is refused in
  the turn that sends it: the `emit_intent` tool answers
  `emit_intent: unknown intent_type 'propose_action'`, a JSON envelope fails
  with `intents[<i>].intent_type 'propose_action' not in allowed set`, and
  the tool's schema no longer lists the type.
* Drop the top-level `predicted_gain_pct`: no intent requires it and nothing
  reads it there. An estimate written as `params.predicted_gain_pct` is still
  recorded as the action's predicted gain in the recipe's journal.
* Stop sending `idempotency_key`, at the top level or under `params`: both
  are ignored, and the one under `params` is removed before the task is
  created. To run identical work again, send it again after its task ended.
* Stop sending a specialist wave (`params.tasks=[...]`). Send one `delegate`
  per task; a turn carries as many intents as you emit. A `scope=freeform`
  delegate that carries `tasks` and no `task_description` is denied with rule
  `specialist_freeform_empty_description`; beside a `task_description` the
  list is not read.

The key of a task (`tasks.idempotency_key` in `storage/coordinator.db`, shown
in the `=== Tasks in flight ===` block of the prompt) is now always derived by
the Coordinator:

| Delegate | Key | What it is taken over |
|---|---|---|
| `integrate_patch` naming a specialist | `integrate_patch:<specialist_task_id>` | The deliverable, as before. |
| `baseline` | `baseline:<16 hex>` | The seven launch fields: `benchmark_script`, `result_dir`, `extra_server_args`, `extra_envs`, `model_path`, `gpu_type`, `config_path`. |
| `explore` | `explore:<16 hex>` (1.2.x: `explore:<10 hex>`) | The grid with the stack, KEEP bar and round cursor the Coordinator seeds it with, as before. |
| `specialist` | `specialist:<16 hex>` | The params as delegated, with its owner and a cited predictor mandate resolved, before the dispatch is warmed with session context. |
| any other action | `<action>:<16 hex>` | Its `params`. |

For a `delegate` other than an `explore`, 1.2.x took the `idempotency_key`
Orchestration sent, and without one used
`orchestration:<action>:t<tick>:<10 hex>`, which changed every tick. What
that changes:

* A `delegate` that repeats work whose task is still queued or running is
  denied with rule `duplicate_idempotency_key_running`. In 1.2.x a `baseline`
  delegated again on a later tick was a second bring-up, and a specialist
  delegated again with the same params was a second task.
* A `delegate` that repeats a `baseline` that measured, or an
  `integrate_patch` that benchmarked its deliverable, is denied with rule
  `duplicate_idempotency_key`; the hint names the task that holds the answer.
* A `delegate` that repeats work whose task ended any other way starts a new
  task under `<key>-retry<n>`, five times at most. The seventh identical
  delegate of a session is denied with rule `duplicate_idempotency_key` and
  the reason `idempotency_key exhausted retries`. That limit now also holds
  for a `specialist` and a `target_analysis` delegated with unchanged params;
  change the params to ask for different work.
* Each of these refusals is a `policy_denied` observation in Orchestration's
  inbox and a row in `ext.denials.rows[]` of the phase event in
  `session_breakdown.json`.

Update anything that matches on the following, which no longer occur:

* The observation `proposal_materialize_skipped` and its reasons. A refused
  repeat is a `policy_denied` observation instead:

  | 1.2.x reason | Now, rule of the `policy_denied` |
  |---|---|
  | `duplicate_proposal_content` | `duplicate_idempotency_key_running` |
  | `baseline_already_succeeded`, `integrate_patch_already_measured` | `duplicate_idempotency_key` |
  | `idempotency_key_exhausted` | `duplicate_idempotency_key`, reason `... idempotency_key exhausted retries for '<key>'` |

* The observations `proposal_pruned_advisory` and `proposal_rejected`; a
  delegate in a pruned family or for a patch no phase owns is reported as
  `delegate_pruned_advisory` and `delegate_rejected`, as it already was.
* The outcome `submitted` of the `proposed` step in the `lifecycle` of a
  `framework_agent` proposal row; a grid Orchestration sends has `delegated`.
* The `policy_denied` rules `specialist_freeform_wave_too_large` and
  `specialist_freeform_wave_invalid_task`, and task keys ending `-w<index>`.
* An `intent` event whose `name` is `propose_action` in the trajectory
  ledger.

The operator-defined specialist (`--custom-specialist-prompt-file`) is no
longer sent through Orchestration's intent path; the Coordinator creates its
task itself. The task, its key, its lanes and its owner are what they were.
What you can see change:

* Its dispatch row in `ext.actions.rows[]` has `dispatch_class` `coordinator`;
  1.2.x recorded `llm`.
* No `task_queued` event and no `delegate_pruned_advisory` observation is
  written for it.
* When the admission chain refuses it, the log line
  `custom specialist: dispatch <key> refused (<rule>); not retried this cycle`
  names the rule and one `custom_specialist_refused` observation carries
  `rule` and `reason`. 1.2.x recorded that refusal as a `policy_denied`
  observation addressed to Orchestration, for an intent it had not sent, and
  as a `denials` row of the phase event with actor `orchestration`; neither
  is written now.

Two smaller changes:

* A patch-mode specialist refused because no git source tree backs it leaves
  a row in `last_action_failures` whose `task_id` is empty. No task exists for
  it; 1.2.x copied the `idempotency_key` Orchestration had sent into that
  field.
* The Orchestration prompt shows `EMIT: delegate{action_name='<action>',
  params?={...}}` for `baseline`, `target_analysis`, `explore`, `report` and
  `session_breakdown`, its free-form specialist example is one delegate with a
  `task_description`, and its hard rules say that the Coordinator keys every
  delegate. The hint of rule `specialist_freeform_empty_description` reads
  `A freeform specialist needs a natural-language task_description (the whole
  mandate).`

---

## Upgrading from 1.1.x

### Required: launch through `hyperloom <command>`

Every command now runs under one entry point, `hyperloom <command>`, or
`python -m hyperloom <command>` where no console script is installed. Update
launchers, skills and scripts:

| Before | After |
|---|---|
| `python -m hyperloom.inference_optimizer.cli --verbose optimize ...` | `python -m hyperloom optimize --verbose ...` |
| `inference_optimizer optimize ...` | `hyperloom optimize ...` |
| `... recover-session --session-dir S` | `hyperloom recover --session-dir S` |
| `python -m hyperloom.inference_optimizer.setup ...` / `hyperloom --check-only` | `hyperloom setup ...` / `hyperloom setup --check-only` |
| `quantization-agent ...` | `hyperloom quantize ...` |
| `python -m hyperloom.inference_optimizer.multi_node SUB ...` | `hyperloom multi-node SUB ...` |
| `python -m hyperloom.inference_optimizer.tools.dump_session_breakdown` | `hyperloom session breakdown` |
| `python -m hyperloom.inference_optimizer.tools.dump_session_report` | removed; see *Upgrading from 1.2.x* above |
| `python -m hyperloom.inference_optimizer.tools.backfill_langfuse` | `hyperloom session backfill` |
| `.../tools/event_counts.py` | `hyperloom session events` |
| `.../tools/read_optimizer_state.py` | `hyperloom session state` |
| `.../tools/preflight_optimizer.py MODEL_PATH` | `hyperloom check MODEL_PATH` |

`--verbose` now follows the command name. `hyperloom` without a command prints
its usage and exits 2 instead of running setup.

### Required: removed flags

These flags are rejected with `unrecognized arguments`:

* `--kernel-codex`, `--kernel-claude`, `--kernel-prompt` (no-ops since the
  kernel LLM role was retired): drop them.
* `--max-minutes-{prelude,framework,explore,kernel,sweep,close}-pct` and
  `--phase-budget-explore-pct`: use `--phase-budget-<phase>-pct`
  (`--phase-budget-framework-pct` covers what the explore spelling set).
* `recover --backfill-trace`: run `hyperloom session backfill --session-dir S`.
* On the rayjob backend, `--pd-prefill-ep`, `--pd-decode-ep` and
  `--pd-*-extra-args` now fail the server restart instead of being ignored;
  they are supported on infera only.
* `--pr-feed-window-days` and `--local-kb-root` (no reader left): drop them;
  select the knowledge root with `KNOWLEDGE_LOCAL_ROOT`.
* `kernelforge forge-loop --return-after-read-kb`, and the TraceLens kernel tool's
  `--skip-split` / `--source-root` (with `TRACELENS_SOURCE_ROOT`): drop them.
* `hyperloom multi-node kernel-bench` is removed; nothing in the optimizer calls it.
* The TraceLens kernel tool's `--use-llm-orchestrator` / `--no-llm-orchestrator`:
  use `--route {agent,non-llm}`. Its `--num-denoise-steps`, `--model-path` and
  `--precision` fed the removed diffusion latency roofline: drop them.
* `python -m hyperloom.orchestrator.trace_analysis.bypass_trace_analysis` is
  removed and fails with `No module named`; use `tracelens_analysis --route non-llm`.
* `python -m hyperloom.measure.consistency` and
  `python -m hyperloom.measure.consistency_resume`, the local-versus-hosted
  consistency check, are removed and fail with `No module named`; nothing in
  Hyperloom runs them, so drop any script that does.

### Required: the `bypass` trace-analysis route is replaced by `non-llm`

`HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass`, or `analysis_route: bypass` in a
`trace_analyze` request, now fails the request with `invalid_analysis_route`.
Set `non-llm` instead; it needs a TraceLens checkout and writes `analysis.json`
rather than `analysis.md`.

These are no longer read and are silently ignored, so drop them:

* `HYPERLOOM_BYPASS_STEADY_STATE`, `HYPERLOOM_BYPASS_CORR_WARN_PCT`,
  `HYPERLOOM_BYPASS_OTHERS_WARN_PCT`, `HYPERLOOM_TRACE_SHAPE_MANIFEST` and
  `HYPERLOOM_TRACE_SHAPE_MANIFEST_MAX_CAPTURES` (the bypass route).
* `HYPERLOOM_NUM_DENOISE_STEPS` and the `trace_analyze` keys `num_denoise_steps`,
  `model_path`, `precision` and `analysis_mode`.
* `KERNEL_AGENT_USE_LLM_ORCHESTRATOR`; the agent route always runs unless
  `--route non-llm` is given.

### Required: pass `--workspace-path` to `tracelens_analysis`

`python -m hyperloom.orchestrator.trace_analysis.tracelens_analysis` now
requires `--workspace-path` and exits 2 without it; it no longer falls back to
`$USER_DATA_PATH`, `$WORKSPACE_PATH` or the workspace root. Scripts that run
the tool directly must pass the root its output lands under
(`<workspace_path>/kernel-agent/runs/<session_id>/`); the optimizer's
`trace_analyze` step already passes the session directory.

### Required: sessions created before this release do not resume

Every session's `storage/coordinator.db` now records schema version 8, and
there are no in-place migrations. `optimize --resume-from` on a session whose
database records an older version, which is every session started by 1.1.5 or
earlier, exits 2 at resume admission, before `state.json` or the database is
touched:

```text
ERROR: --resume-from coordinator.db is at schema version 7, older than this build's 8:
sessions created before this release cannot be resumed. Finish the session under the
build that started it, or start a new session.
```

Opening such a database through the coordinator storage raises the same
message without writing to it. The legacy shapes the old migrations handled
(`events.priority`, a `leases` table without `owner_scope`, `state.json`
without `elapsed_charged_sec` or `phase_elapsed_totals`, manifests without
`workflow_contract_version`) are no longer read. Finish in-flight sessions
before upgrading.

### Required: `session_breakdown.json` drops `robustness`

The top-level `robustness` key, `close.robustness` and the
`robustness_escalated` stop reason are gone; they were always empty since the
robustness agent was retired. Consumers that read them should stop.

### Required: `session_breakdown.json` drops the latency roofline and `bypass` keys

The schema stays `v6.1` but drops the keys only the diffusion latency roofline
or the `bypass` route produced; read `ceiling_available` instead of
`close.roofline_progress.ceiling_kind`. Each key and its disposition is under
Versioning in [`session_breakdown.json` integration in Hyperloom](session-breakdown.md).
The trace certificate's `production_*` keys go with them; read `selected_role`.

### Required: the Critic's knowledge-base client is removed

`python -m hyperloom.agents.critic.runtime.cli replay-dead-letter` is removed
and exits 2 as an unknown subcommand; there was never anything to replay.

These are no longer read and are silently ignored, so drop them:

* `CRITIC_KB_CLIENT_MODE` and `KB_BASE_URL`; `optimize` no longer exits 2 on
  an unknown mode, or on `live` without a base URL.
* `KB_READ_ENABLED`, `KB_WRITE_ENABLED`, `KB_TIMEOUT_MS`, `KB_RETRY_MAX`,
  `KB_DEAD_LETTER_DIR` and `KB_SERVICE_TOKEN`.
* `CRITIC_PRIOR_CACHE_TTL_SECONDS`, `CRITIC_KB_PRIOR_LIMIT`,
  `CRITIC_KB_BREAKER_THRESHOLD` and `CRITIC_KB_BREAKER_COOLDOWN_SECONDS`.

`session_breakdown.json` drops `kb_priors` from critic iteration rows and `kb`
from `critic_review` on framework proposals. Consumers that read them should
stop.

### Required: the Critic stops writing unread session memory

`critic-session-memory/<session_id>/decisions.jsonl` and `context.json` are no
longer written. Every verdict is still in `critic-workdir/<NNNNNN>/emit.json`,
which keeps the newest 50 turns, and the applied ones are in the session
breakdown's phase-event `proposal` rows under `critic_review`.
`judge_bundle.json` renames `merged_context` to `context` and drops
`missing_context`; it, `emit.json` and the `reviewed_msg_ids.json` entries
drop the always-null `decision_id`. Consumers that read them should stop.

`python -m hyperloom.agents.critic.runtime.cli` ignores a request's
`proposals`, `options` and `decision_id`, and no longer fills an empty
`context` from the prompt's `Shared session state` section: a request
without `context` now lists `model` and `framework` in `required_context`.
Pass `context` explicitly.

`hyperloom optimize` no longer sets `WORKSPACE_PATH`, which nothing read.

### Required: the multi-branch image check is removed

A multi-branch session no longer checks the rule pack's facts against its
environment before PRELUDE.

`HYPERLOOM_IMAGE_CHECK` is no longer read and is silently ignored, so drop it:
`strict` no longer stops a session before PRELUDE, and a value that is no mode
is no longer refused as a usage error.

`python -m hyperloom.common.outside_facts.image_check` is removed. A session no
longer writes the `image_check` field of `MB-manifest.json` or
`MB-reports/image_check.json`; consumers that read them should stop. A manifest
an earlier session wrote with the field still reads.

A rule pack named by `HYPERLOOM_RULE_PACK` must drop the `image_check` block of
every entry: a pack that keeps one fails every import of
`hyperloom.common.outside_facts` with `RulePackError`. The shipped pack is
version 25 and no longer holds `aiter_merged_view`.

### Required: quantization failure exits 4

A failed `--quantize` prelude now exits 4 instead of 3; exit 3 means only
that another optimizer holds the session lock. Launchers that branch on the
exit code should treat 4 as a non-retryable failure. See also
[operations.md](operations.md) for the full exit-code table.

### Required: stop the local Ray head and end running sessions

A single-node `hyperloom optimize` no longer uses Ray: serving rounds and GPU
specialists run as local subprocesses, and GPU specialists get their GPUs from
the session's GPU pool. The installer no longer installs Ray or starts a Ray
head, and preflight no longer checks Ray. A head that an earlier installer
started keeps running until you stop it, so run this once on each host or
container that ran an earlier version:

```bash
ray stop --force
```

There is no migration for a session in flight: end running `optimize` sessions
before upgrading. These environment variables are no longer read and can be
removed from `.env` and launchers: `INFERENCE_OPTIMIZER_RAY_EXEC`,
`INFERENCE_OPTIMIZER_RAY_GPU_PENDING_LIMIT`,
`INFERENCE_OPTIMIZER_RAY_SERVING_PRIORITY`,
`INFERENCE_OPTIMIZER_RAY_ROUND_WAIT_SEC`, `INFERENCE_OPTIMIZER_RAY_NUM_GPUS`,
`INFERENCE_OPTIMIZER_RAY_SPECIALIST_SCHED_TIMEOUT_SEC`,
`HYPERLOOM_RAY_{CONNECT_PROBE,READY,STATUS,STOP,INIT}_TIMEOUT_SEC`,
`HL_RAY_HEAD_PORT`, `RAY_MIN_NOFILE`, `RAY_VERSION`, `RAY_NUM_GPUS` and
`SKIP_RAY_START`. The failure classes `ray_round_unplaced`,
`ray_cluster_infeasible` and `ray_actor_error` and the return codes -913 and
-916 no longer occur, and `session_breakdown.json` no longer reports the
conc-sweep arm strategy `serving_lease_held` or the `ensure_ray` install step.
Multi-node runs (RayJob through the Ray Dashboard) are unchanged.

### Recommended: commands that stay outside `hyperloom.cli`

`hyperloom <command>` covers setup, check, optimize, recover, quantize,
multi-node, and `session {breakdown,backfill,events,state}`. These
entry points are **intentionally separate** (same as before the unification,
only the optimizer-family commands moved under `hyperloom`):

| Entry | Typical use |
|---|---|
| `kernelforge` / `python -m kernelforge.cli` | KernelForge campaigns (orchestrator dispatches `-m kernelforge.cli`) |
| `hyperloom-kb-serve`, `hyperloom-kb-collect`, `python -m hyperloom_kb` | Experience KB service and collect CLI |
| `hyperloom-kb` / `python -m hyperloom_kb.cli` | Operate an Experience KB service: health, labels, restore, exclusions, list, export |
| `python -m hyperloom.inference_optimizer.experience_kb_service {init-env,ensure,...}` | Per-workspace Experience KB sidecar, with every `hyperloom-kb` command (setup skill, optimizer SKILL) |
| Kernel agent tool modules (`tracelens_analysis`, …) | Orchestrator subprocess scripts with their own `__main__` |
| `multi_node/scripts/*.py` on remote pods | Payload scripts SSH'd by `hyperloom multi-node`, not local `-m` entry points |

There is no `hyperloom kb` subcommand in 1.1.x; keep using
`experience_kb_service`, `hyperloom-kb`, or the `hyperloom-kb-*` console scripts above.

---

## Upgrading from 0.5.x → 0.6.0

Apply the following changes in order. Required steps must be completed before running; recommended and optional steps improve behavior or unlock new features.

### Required: rename multi-node image / GPU flags and envs

The `optimize` CLI multi-node image and per-node GPU flags were renamed
and the legacy names are no longer accepted (no alias). Any launcher that
passes the old flags will fail with `unrecognized arguments`. The image
is platform-owned and `--mn-image` is accepted but ignored; the per-node GPU
count is set through `--gpus-per-node`. The previous
`INFERENCE_OPTIMIZER_RAYJOB_IMAGE` env is no longer read.

```diff
# run launchers, k8s ConfigMaps, .env
- --rayjob-image harbor/...
+ --mn-image harbor/...
- --rayjob-gpus-per-node 8
+ --gpus-per-node 8
- INFERENCE_OPTIMIZER_RAYJOB_IMAGE=harbor/...
```

The new flags cover both multi-node backends (`rayjob` head+workers and
`infera` worker/prefill/decode pods), which is why they dropped the
`rayjob`-specific prefix.

### Recommended: review `--model-class` if you relied on live classification

The live `classify` action was removed. Current Coordinator boot still infers
and persists `model_class` from model metadata or model-path family keywords
when possible, but launchers that know the class should pass it explicitly to
avoid a generic fallback:

```diff
python3 -m hyperloom optimize \
    --model /path/to/GLM-5-FP8 \
    --framework sglang \
    --gpu-type mi355x \
+   --model-class moe_mla_nsa \
    --isl 1024 --osl 1024 \
    --max-hours 2.0
```

Recognised `--model-class` values (case-insensitive, with `-`/`+`/space
tolerated; see [Inference Optimizer Skill](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/SKILL.md)):
`dense`, `moe_mla`, `moe_swa`, `moe_mla_nsa`.

If `--model-class` is omitted and inference cannot determine the family, the
Coordinator falls back to a generic dense prior — likely sub-optimal for mixture of experts (MoE) /
multi-head latent attention (MLA) / native sparse attention (NSA) models.

### Required: pass `--compare-against-gpu` to opt into InferenceX reference fetching

The `classify` action used to derive this implicitly. Without
`--compare-against-gpu`, `target_analysis` writes a
`no_target_gpu_configured` marker and the run proceeds without an
external reference (the optimisation still works; you just don't get
the "vs B200" comparison number).

```diff
+ --compare-against-gpu B200
```

### Required: setup is no longer an in-loop action

Earlier launchers might have waited for the Coordinator to emit a
`setup` action. Move all setup work to **before** the
`python -m hyperloom optimize` call:

```diff
# launcher.sh
- python3 -m hyperloom.inference_optimizer.cli optimize ... # expects setup as first action
+ bash "$REPO_ROOT/hyperloom/inference_optimizer/assets/install.sh"
+ kernel_agent_env_file="$(. "$REPO_ROOT/hyperloom/inference_optimizer/assets/runtime_env.sh"; runtime_env_kernel_agent_env_path)" && . "$kernel_agent_env_file"
+ ray stop --force; ulimit -Sn "${RAY_MIN_NOFILE:-65536}" 2>/dev/null || true; ray start --head --num-gpus="$RAY_NUM_GPUS" --include-dashboard=false --resources='{"serving_slot":1}'
+ python3 -m hyperloom optimize ...
```

### Recommended: review the `KERNEL_OPT_BACKEND_ORDER` default

Bare-metal installs now default `KERNEL_OPT_BACKEND_ORDER` to `geak`. If you
had a custom order hard-coded, confirm it is still intentional.

### Recommended: set `INFERENCE_OPTIMIZER_RESCUE_PATHS` if you use model-specific benchmark scripts

When `--gpu-type` is set, Hyperloom pins
`benchmark_script=<framework>_<gpu_type>.sh` to stop InferenceX-native
scripts from leaking `result.json` outside the session dir. If you
intentionally use a model-specific script (for example, `dsr1_fp8_mi300x.sh`),
keep passing `benchmark_script=` explicitly — operator overrides
still win against the generic-script pin. You might additionally want
to set `$INFERENCE_OPTIMIZER_RESCUE_PATHS` so the harvest step can
recover any leaked `result.json` files written to hardcoded
`--result-dir` locations.

### Recommended: stop expecting `standalone_analysis.md` or `tracelens_report.md`

The kernel-agent no longer aliases the TraceLens v0.3 report. The
canonical analysis path is now surfaced by the `trace_analyze` request
handler as `trace_report_path` and forwarded to the lifecycle as
`analysis_report_path` (pointing at
`$SESSION_DIR/kernel-agent/runs/<session_id>/tracelens/analysis.md`).
The `--compat-report-path` argument was removed.

### Roofline is now on by default (no PMC env toggle)

There is no `HYPERLOOM_ENABLE_PMC_ROOFLINE` environment variable. Roofline
analysis (the composite `profile` + `trace_analyze` + `analysis.md` path) is
controlled by the CLI flag `--enable-roofline`, which defaults **on**. Pass
`--no-enable-roofline` for a profile-only run. See
[Environment variables](environment-variables.md).

### Schema compatibility

`session_breakdown.json` emits `hyperloom.session_breakdown.v6.1`. 6.1 changes
three things inside the 6 major, a declared exception to the major-bump rule: it
renames `throughput_tok_s_per_gpu` in `outcome.baseline`, `outcome.final` and
the `baseline` event's `measurement` to `output_throughput` (the old key is no
longer written), removes keys no producer populated, that only carried a
constant or whose producer was removed, and reports two unmeasured figures as null. Archived 6.0 documents and documents exported by 1.1.x keep
the old key, so readers must accept both keys or migrate.
Re-exporting a 6.0 session directory emits a null
`outcome.baseline.output_throughput` and the `contract_digest` recorded at
session creation, and the default output overwrites the archived
`<session_dir>/session_breakdown.json`. The removed keys (disposition: removed
in 6.1, never populated) are `policy.force_exit_budget_pct`,
`policy.variant_timeout_sec`, `policy.overtime_kill_ratio`,
`trace_validate_ref`, `critic_review.iteration`, `geak.product.cfg_hash`,
`geak.product.final_overlay_digest`, `entry.budget_remaining_sec` and the
roofline rows' `rocprof_roofline`; the roofline rows' `roofline_measured` and the
baseline action's `convergence` (disposition: removed in 6.1, constant) are
removed as well.
`attributed_gain_pct`, `unattributed_gain_pct` and run `empty` are null when
unmeasured or not provided.

V6 was the preceding cutover, for the timeline: each action records its own event as it
runs, so an event's `start_time` is when the work started rather than when its
artefacts were written, and the KERNEL and BASELINE projections are no longer
emitted. Ordering that relied on the old collapsed windows changes as a result.

V5 was the preceding cutover, for optimization results: adopted optimizations
are reported only through `optimizations`, and the `optimization_stack`,
`attribution`, GEAK invocation, Forge invocation, and GEMM-tuning projections
are no longer emitted. Consumers reading archived v2 / v3 / v4 / v5 documents
need a downstream migration, as described in
[`session_breakdown.json` integration in Hyperloom](session-breakdown.md).

---

## Upgrading from pre-0.5 (training and MLPerf-training)

Hyperloom is **inference-only** since v0.4. If you are still on a
training-mode build:

* The training-mode TraceLens CLI
  (`TraceLens_generate_perf_report_pytorch_training`) is no longer
  accepted by `install.sh`. Use the `_inference` variant.
* Training and MLPerf-training skills have been removed from this
  repo. There is no in-place migration; switch to the inference flow
  documented in the [setup and examples guide](../../examples/README.md).

---

## Required: migrate to exclusive KnowledgePlane modes

KnowledgePlane now selects exactly one Recipe backend. This is a breaking
change for deployments that previously got remote-first reads merely by
exporting GBrain credentials:

```diff
  GBRAIN_BASE_URL=https://gbrain.example
  GBRAIN_TOKEN=...
+ KNOWLEDGE_STORE_MODE=remote
```

Without the explicit mode, local/default ignores both credentials. Set
`KNOWLEDGE_STORE_MODE=local` (or leave it unset) for local-only operation.

The implicit local root changed from `$USER_DATA_PATH/kb` (or
`/workspace/hyperloom/kb`) to `$USER_DATA_PATH/knowledge` (or
`~/.cache/hyperloom/knowledge`). Data under the old root is neither read nor
migrated; `--local-kb-root` and `HYPERLOOM_LOCAL_KB_ROOT` are removed, so use
`KNOWLEDGE_LOCAL_ROOT` to select a root.

---

## Generic upgrade procedure

For any minor or patch upgrade:

1. Pull the new Hyperloom revision into `$REPO_ROOT`.
2. Re-run `bash "$REPO_ROOT/hyperloom/inference_optimizer/assets/install.sh"`. The
   installer is idempotent: it picks up new GEAK / TraceLens versions,
   refreshes generated LLM gateway aliases, and regenerates `kernel-agent.env.sh`.
3. Re-source the env file. `runtime_env.sh` prints it only when `install.sh` wrote it in this
   container and no other account could have changed it; otherwise it says why and nothing is sourced:
   ```bash
   kernel_agent_env_file="$(. "$REPO_ROOT/hyperloom/inference_optimizer/assets/runtime_env.sh"; runtime_env_kernel_agent_env_path)" && . "$kernel_agent_env_file"
   ```
4. To resume an ongoing session across the upgrade, first check the target
   release's upgrade section: a release that raises the `coordinator.db`
   schema version (1.1.x → this release does) refuses sessions started before
   it, so finish those under the build that started them. Otherwise verify
   `manifest.json`, `state.json` and `storage/coordinator.db` are intact, then run
   `python -m hyperloom optimize --resume-from "$SESSION_DIR"`.

Upgrades do not move storage or rewrite historical sessions.

---

## Related guides

Use these resources for related reference information:

* [Releases](https://github.com/AMD-AGI/Hyperloom/releases): Full per-release notes.
* [Hyperloom authentication and credentials](authentication.md): Credential and path env reference.
* [Environment variables](environment-variables.md): Every
  environment variable read by the runtime.
* [Hyperloom self-hosting and operations guide](operations.md): Self-host runbook.
