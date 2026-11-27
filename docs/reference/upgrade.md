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

* A `propose_action` from Orchestration passes PolicyGate and the admission
  chain, and its task is queued in the same step, announced by a `task_queued`
  event as a `delegate` is. A repeat is answered by what already exists: a
  task of the same content still queued or running, a `baseline` that already
  measured, or an `integrate_patch` that already benchmarked that specialist's
  deliverable. It is recorded as a `proposal_materialize_skipped` observation
  with reason `duplicate_proposal_content`, `baseline_already_succeeded`,
  `integrate_patch_already_measured` or `idempotency_key_exhausted`.
* When a specialist finishes with a patch or a routable artifact, or an
  ENABLEMENT round finishes with a config-only deliverable, the Coordinator
  queues the `integrate_patch` for that deliverable itself. A deliverable has
  one idempotency key, `integrate_patch:<specialist_task_id>`, whoever asks.
* An `integrate_patch` that Orchestration asks for, by `delegate` or by
  `propose_action`, is that same task: it carries the deliverable's own params
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
  `delegate`: one that reuses the key of a `baseline` that already measured is
  denied with `duplicate_idempotency_key` where it started a second baseline.
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
  `lifecycle` no longer has a `reviewed` step. A grid Orchestration sent as a
  `propose_action` now has the lifecycle a delegated grid has, `proposed` then
  `attempted`, without the `routed` step in between.

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
  `include_patches`; it was `true` on every row.
* The audit rows (`last_roofline`, `last_profile`, `last_baseline` and their
  `*_attempts` lists) no longer use the decisions `skipped` (roofline,
  profile) or `discarded` (roofline, baseline), and a baseline row's
  `anchor_held.reason` is never `no_measurement`. None could be written: a
  roofline settles `succeeded`, `recovered` or `failed`, a failed one is not
  promoted, and a baseline is promoted only with a positive throughput. A
  failed roofline is still counted in `roofline_failure_streak`.

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
  later in the same one.
* A delegate whose `idempotency_key` names a task another author created is
  answered like any other reused key: with `duplicate_idempotency_key` or
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
