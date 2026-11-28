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
* `optimize` no longer selects plug implementations: `--plug-profile`,
  `--manifest`, `--plug` and `--plug-config` are removed and, like any
  unknown argument, exit 2 with `unrecognized arguments`. Drop them.
  `HYPERLOOM_MANIFEST` and `HYPERLOOM_MANIFEST_OVERRIDES` are no longer read,
  and a session's `plugs.resolved.json` is no longer written or read back on
  `--resume-from`; an old session that has one resumes and ignores it.
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

### Required: files outside a round's run directory are no longer copied into it

Since 1.2.0 a round on InferenceSuite writes its server log, results and
traces inside its own run directory. Hyperloom still looked for them elsewhere
after every explore, integrate, rebench, concurrency-sweep and multi-branch
round, and that search is removed:

* `server.log`, `gpu_metrics.csv`, `profile_*.trace.json.gz`,
  `inferencex_result*.json` and `results*.json` files written during the round
  at the top of `/workspace`, `$INFERENCEX_PATH` and `$RESULT_DIR` are no
  longer copied into the round's run directory, and a round's
  `nonfatal_warnings` no longer carry `harvested_leaked_artifact:<path>`. A
  file another process wrote under one of those names could replace the
  round's own `server.log` or be read as its accuracy result.
* `INFERENCE_OPTIMIZER_LEAK_ROOTS`, which named other directories to search, is
  no longer read and is silently ignored, so drop it.
* The multi-branch measurement runner no longer copies those names from the top
  of its work directory into the files it signs. On a host that reports itself
  multi-node it also no longer writes a merged `gpu_metrics.csv` at the top of
  those files; each round's run directory still gets its own.

### Required: the accuracy eval runs after the measured bench

On a single node, `hyperloom optimize` tests a served fixed workload in two
rounds on one server: a bench that boots the server and is discarded, then the
measured bench. The accuracy eval now runs after the measured bench, in the
measured round, and never in the discarded one. This holds for a genuine
baseline and its retries, a PRELUDE warm replay, the KERNEL stack validation,
the KERNEL integrate re-baseline, the FRAMEWORK integrate bench and an EXPLORE
variant. In 1.2.x the eval ran in the discarded round, before the measured
bench; the integrate bench ran it in both rounds, and the KERNEL re-baseline
ran it in a third round after a second load. An eval before the measured bench
left the server's KV pool filled, so the measured bench read lower, and a
KERNEL candidate measured on a server no eval had touched was graded against
an anchor measured after one.

* **Throughput shifts once.** Every figure a single-node fixed-workload
  session measures is now taken on a server no eval has loaded. A session's
  anchor and its candidates move together. Do not resume a session started on
  an earlier build: its anchor was measured after an eval, and the candidates
  measured after the upgrade would be graded against it. Do not compare
  sessions, reports or Recipe KB records taken before and after the upgrade.
  Multi-node sessions do not move: their discarded pass still runs the eval
  before the measured pass. AgentX, MLPerf and scriptable workloads, profiled
  rounds and concurrency sweeps run the rounds they ran.
* The integrate bench runs the eval once, in its measured round; it ran it
  twice.
* The KERNEL integrate re-baseline evaluates every candidate in its measured
  round. It used to evaluate only a candidate whose throughput cleared the
  KEEP gate, in a third round that repeated the load. Its task directory has
  no `accuracy_round/`, and its result no longer carries `accuracy_stage` with
  `status: succeeded` or `skipped`. The task parameters
  `defer_accuracy_until_after_measure`, `post_measure_accuracy_min_tput` and
  `post_measure_accuracy_keep_policy` are no longer read. The re-baseline is
  now held to the budget check a baseline's measured round is: when the
  session cannot pay for the measured round it keeps its cold warm-up figure
  (`baseline_measure_round_dropped_low_budget`) without booting the round; it
  used to run the round into the session clock, which left the same figure.
* An EXPLORE variant's decision round evaluates and its warm-up round does
  not (`eval.enabled: false` in the `config.yaml` of the warm-up's run
  directory). The eval's results file lies under the decision round's run
  directory (`vNN_<name>/variant_00_<name>/`), no longer under
  `vNN_<name>/warmup_round/`.
* An eval that fails after the measured bench leaves the bench standing. The
  attempt's result is `succeeded` with its throughput, carries
  `accuracy_stage` (`status: failed`, the eval's `error_class` and `error`,
  and the run directory as `workspace`) and `post_measure_accuracy_failed` in
  `nonfatal_warnings`, and has no `accuracy`. In 1.2.x the eval failed the
  warm-up round and the attempt with it (`baseline_warmup_round_failed`),
  before any measured bench ran. What follows depends on the caller:
  * A genuine baseline whose eval exited non-zero (`eval_failed`) is ruled as
    an eval failure, as before: with `--enablement eval` or `all` on a single
    node it is recorded with `baseline_eval_failure_kind:
    eval_runtime_failure` and not promoted; otherwise the run stops with
    `stop_reason: baseline_accuracy_failed`. In the second case the baseline
    is now promoted with its throughput and no accuracy reference, where it
    used to be a failed baseline.
  * A genuine baseline whose eval was stopped by the session clock or the
    round's cap, or whose server died during the eval, is ruled like any
    baseline without a score: recorded as an eval failure of kind
    `accuracy_unavailable` and not promoted, or the run stops with
    `baseline_accuracy_failed`. `accuracy_stage.error_class` names the stop
    (`session_time_exhausted`, `round_timeout`, a server-death class). In
    1.2.x the same stop ended the warm-up round, and the attempt ended under
    that class.
  * A warm replay is admitted without an accuracy verdict and marked
    (`eval_ran: false`), as a replay without a score always was. It used to
    be a failed replay.
  * A KERNEL re-baseline is graded on its throughput, and a KEEP-eligible
    candidate ends `NEEDS_REVIEW` in a session that holds an accuracy
    reference, as one whose third round failed did.
  * An integrate bench or EXPLORE variant keeps its row
    (`nonzero_rc_after_complete_protocol:1` in its warnings) and its accuracy
    gate rules on the missing score (`accuracy_unavailable`). One whose eval
    the session clock stopped is skipped and stays out of the ledger, as a
    round the clock stops anywhere is.
* A baseline that kept its cold warm-up figure
  (`baseline_measure_round_dropped_low_budget`) carries no accuracy and has
  `run_eval_disabled: true`: the round the budget refused is the one that
  evaluates. It used to carry the warm-up's score. It is not ruled on
  accuracy: it neither stops the run nor is recorded as an eval failure, and
  an enablement revalidation that ends this way leaves its window open for
  another revalidation (recorded as a window the run stopped). Once promoted
  it leaves `baseline_accuracy: 0.0` in `state.json`. PRELUDE still retries a
  cold anchor or closes on it (`baseline_anchor: cold`).
* An eval the round refuses before it starts (`eval_context_too_short`, eval
  tooling that is not staged) is reported by the measured round, after the
  discarded bench ran. The warm-up used to report it before it booted.
* The measured round's wall-clock, which `state.json` keeps as
  `baseline_warm_runtime_sec` and which prices the next measured round, an
  EXPLORE decision round and one more variant, now includes the eval. A
  session's first measured round is priced at its warm-up's time after ready:
  `priced_by` in `measure_round_dropped` reads `warmup_post_ready` (was
  `warmup_post_ready_less_eval`), and `eval_runtime_sec` is no longer
  subtracted anywhere.
* In `session_breakdown.json`, a baseline action has no round row labelled
  `accuracy`. Its `measure` row carries the eval's figures and
  `run_eval_disabled: false`; its `warmup` row has `run_eval_disabled: true`
  and no accuracy.
* The error lines for an eval-rooted baseline failure read `the accuracy eval
  failed on a genuine baseline, ...` and `the accuracy eval failed; the
  attempt gets no throughput-only retry.` (they named `InferenceX run_eval`),
  and the line for an eval that never reached its server ends `it is not
  ruled on as an eval failure.`

### Required: a baseline takes its accuracy from its own attempt

A benchmark attempt that runs a discarded warm-up round and then a measured
round on the same server (a genuine baseline, a PRELUDE warm replay, the KERNEL
stack validation) runs the accuracy eval after its measured bench (see the
section above). The result the attempt returns carries that round's score. In
1.2.x the eval ran in the warm-up round, the result of the measured round
carried no score, and each reader searched the attempt's directory for the
warm-up's results file. A baseline is no longer given a score found by
searching the directory above its round: it is ruled on what its own attempt
measured.

* The result of such an attempt carries `accuracy`, `accuracy_task`,
  `accuracy_metric`, `accuracy_source`, `accuracy_definition_digest` and
  `accuracy_eval_kind` of the round that ran the eval. The `measurement` the
  session breakdown records for the action carries the first four of them;
  for a warm replay and a stack validation its `accuracy` was `null`.
  `accuracy_eval_kind` did not appear on such a result before; it is `""` for
  an `lm-eval` score. The score, and every verdict taken on it, is the one
  the search found.
* A warm replay evaluates even when its recipe sets `RUN_EVAL=false`. The
  round that runs the forced eval, now the measured round, is recorded as one
  that evaluated: its round row has `run_eval_disabled: false` and its score.
  In 1.2.x the forced eval ran in the warm-up round, whose row had `true` and
  no score.
* A warm replay is judged on the score its result carries; it no longer looks
  for an `lm-eval` results file under its task directory. `eval_ran` in
  `warm_replay_outcome` (`state.json`) and in the warm-replay `measurement` of
  `session_breakdown.json` is `true` when the replay's result carries a score
  and `false` otherwise. A results file that could not be decoded, or that held
  no recognised metric, used to give `eval_ran: true` with no score; it now
  gives `false`. For a replay without a score, `eval_error`, the reason on its
  accuracy gate row and the log line `warm-replay admitted without an accuracy
  verdict` read `the replay's result carries no accuracy score`, where they
  quoted the results-file search (such as `no results*.json in <directory>`).
  The verdict on a score is unchanged: a collapsed score rejects the replay,
  and a score taken under another eval definition than the baseline's is no
  verdict. A replay without a score is admitted and marked as before, now also
  when a results file lies elsewhere under its task directory; for a
  multi-node replay that file was its own discarded warm pass's (see below).
* A genuine baseline whose own attempt left no usable score no longer
  continues on a score found in the directory above its round. For an attempt
  of one round (a multi-node, AgentX or scriptable baseline) that directory is
  `runs/baseline`, so the score could come from any other attempt of the
  session, and nothing tied it to this attempt's launch. The search also
  replaced a score the attempt itself measured as zero or below the
  enablement floor; the baseline is now ruled on the score it measured.
  Such a baseline takes the route a first baseline without a usable score
  always took. With `--enablement eval` or `all` on a single node it is
  recorded as an eval failure and not promoted: a first baseline opens the
  enablement lane, a re-baseline leaves the anchor and its reference as they
  were. A re-baseline that used to pass this ruling on the score the search
  found now leaves what a failed baseline leaves: a `baseline` row in
  `last_action_failures` (`state.json`, shown to the model each turn), the gap
  derived from that row, and a `what_failed` entry in the published recipe.
  Otherwise, and in a multi-node session, the run stops with
  `stop_reason: baseline_accuracy_failed`; so does a genuine baseline whose
  eval aborted its round, which another attempt's score used to keep
  running. The stop line and the round's eval output name the failure to fix.
  A run that used to continue on another attempt's score now takes one of
  these two routes.
* In a multi-node session the searched directory also held the attempt's own
  discarded warm pass (`mn_warmup`), which runs the same eval as the measured
  pass, so the score the search found could be that pass's as well as another
  attempt's. A multi-node attempt is ruled on its measured pass alone, and
  three outcomes change when that pass has no usable score of its own. A
  baseline whose measured pass succeeded without one stops the run with
  `baseline_accuracy_failed`, where the found score let the run continue on
  it. A baseline whose measured pass was aborted by its eval stops the
  run on that attempt, where the found score left a failed attempt that was
  retried. A warm replay whose measured pass left no score is admitted
  without an accuracy verdict and marked (`eval_ran: false`), where it was
  judged on its warm pass's score and rejected when that score had collapsed.
  A multi-node attempt whose measured pass scores is unchanged.
* A native AgentX baseline whose accuracy round did not run, because it was
  refused before it started or stopped by the run (the session clock, a
  shutdown), has no score and is ruled like any baseline without one:
  recorded as an eval failure and not promoted, with the
  `last_action_failures` row above, or the run stops with
  `baseline_accuracy_failed`. That was already the ruling for the first
  baseline of a session; a later attempt used to pass on the score an earlier
  attempt's accuracy round had left. `agentx_accuracy_round.error_class` on
  the result names why the round did not run.
* The session's accuracy reference (`baseline_accuracy` and its digest in
  `state.json`) is written only when a baseline is promoted, together with
  its throughput. Measuring a baseline used to write the score the search
  found into the session state while the baseline was still running, so a
  baseline that was not promoted (`anchor_held`, an eval failure, a failed
  attempt) replaced the reference while the throughput anchor stayed; it now
  leaves both as they were.
* `baseline_accuracy_salvaged_from_sibling_attempt` no longer appears in a
  result's `nonfatal_warnings`. Every log line that names a salvage or a
  sibling attempt is gone: `cold-start guard — reading accuracy=... from the
  warmup round`, `salvaged a measured baseline accuracy=... from a sibling
  attempt`, `eval-rooted baseline failure, but salvaged a valid baseline
  accuracy=... from a sibling attempt (...); not stopping the run` and
  `eval-rooted baseline failure and the sibling attempt measured
  accuracy=... (...); stopping the run`. The warning for an eval failure that
  goes to enablement reads `eval-rooted failure; routing to enablement
  (RUN_EVAL stays on).`
* A promoted baseline that scored nothing leaves the session without an
  accuracy reference: `baseline_accuracy` is `0.0` and
  `baseline_accuracy_definition_digest` is empty in `state.json`. The score
  and digest of the anchor it replaced used to stay beside the new
  `baseline_tput`. A re-baseline whose eval left no score reaches this when
  the run stops with `stop_reason: baseline_accuracy_failed` (`--enablement
  off` or `launch`, or a multi-node session) and it is promoted: it measured
  higher than the anchor, it is an enablement revalidation, or it corrects a
  cold anchor. Its throughput becomes the anchor as before, and the reference
  is dropped with the anchor it belonged to. One that is not promoted
  (`anchor_held`) leaves the anchor and its reference as they were. A session
  resumed after a promoted one stopped the run grades candidates on
  throughput alone, as it does after a first baseline that stopped the same
  way; it used to grade them against the score of the previous stack. With
  `--enablement eval` or `all` on a single node such a re-baseline is recorded
  as an eval failure and not promoted, so the anchor and its reference stay
  as they were.

### Recommended: drop `INFERENCE_OPTIMIZER_RESCUE_PATHS` and `INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC`

Both are no longer read and are silently ignored. Since 1.2.0 the first has
had no effect, because a round's results are read from its run directory
alone, and an integer value of the second has had none, because every round
already held its server boot to the round's own time limit. A value of the
second that was not an integer raised `ValueError` when the configuration of a
round that keeps its server across rounds was written, before that round ran;
it is now ignored like any other. The readers that went with them could not
run either, so the warnings
`rescued_from_leaked_path:<path>`, `rescued_copy_into_workspace_failed`,
`raw_inferencex_result_used` and `benchmark_report_success_false`, and the
error classes `no_benchmark_workspace` and `benchmark_report_invalid_metric`,
no longer occur.

### Recommended: `result_dir` is no longer a task parameter

`result_dir` told the benchmark script of the harness before 1.2.0 where to
write. No round has been handed it since 1.2.0, and the `baseline`, `explore`
and `integrate` tasks now ignore it, whatever its value:

* `integrate` still read it: a task that carried `result_dir` had its bench's
  accuracy searched in that directory, where no round writes, and its accuracy
  round written below it. The accuracy of a bench is now always read from, and
  its accuracy round always written into, the bench's own directory.
* A `result_dir` with whitespace or a shell metacharacter failed a `baseline`
  attempt and an `explore` round as `bad_param`; it no longer does.
* A baseline result no longer carries `result_dir`. On a succeeded round it
  repeated `workspace`, which still names the round's run directory.
* The baseline fingerprint, which decides whether a proposed baseline repeats
  one already queued or measured, is six parameters instead of seven. Two
  proposals that differ only in `result_dir` are now the same baseline. In a
  session resumed across the upgrade, a baseline proposed again is not matched
  with one measured before the upgrade and runs once more.

Custom workload scripts are not affected: a script is still run with
`RESULT_DIR` in its environment and writes its result there.

### Recommended: the session package no longer collects `benchmark_report.json` and `summary.txt`

The package collected every `runs/**/benchmark_report.json` and
`runs/**/summary.txt`, the report and summary of a round before 1.2.0. No
round writes either. A 1.2.0 session that replayed a GEAK candidate on GEAK's
own harness has a `benchmark_report.json` per replayed point under
`runs/integrate/`; packaged again by this build, the package leaves those
files out. They stay in the session directory. A round's `report.json`,
`launch.json`, `config.yaml`, `phase_marks.json` and
`inferencex_result.json` are collected as before.

### Recommended: drop the EXPLORE pre-filter settings

The opt-in EXPLORE pre-filters are removed: the cross-session measurement
store, the InferaSim projection, the recording of measured anchors and the
anchor harvest. A session that set none of their switches measures the same
variants and reaches the same verdicts as before. Its explore result no longer
carries `projection`, which was `null` unless the projection was on.

These variables are no longer read and are silently ignored, so drop them:

* `HYPERLOOM_MEASUREMENT_STORE` and `HYPERLOOM_MEASUREMENT_MAX_AGE_DAYS`;
* `HYPERLOOM_EXPLORE_PROJECTION`, `HYPERLOOM_EXPLORE_PROJECTION_MARGIN_PCT`,
  `HYPERLOOM_EXPLORE_PROJECTION_CALIBRATED_MARGIN_PCT` and
  `HYPERLOOM_EXPLORE_PROJECTION_TOP_K`;
* `HYPERLOOM_INFERASIM_ROOT`, `HYPERLOOM_INFERASIM_MODE`,
  `HYPERLOOM_INFERASIM_ESTIMATOR`, `HYPERLOOM_INFERASIM_MODEL`,
  `HYPERLOOM_INFERASIM_WORKLOAD`, `HYPERLOOM_INFERASIM_GPU_ARCH`,
  `HYPERLOOM_INFERASIM_HBM_GB` and `HYPERLOOM_INFERASIM_SERVING_MODEL`;
* `HYPERLOOM_INFERASIM_ANCHOR_STORE`, `HYPERLOOM_INFERASIM_ANCHOR`,
  `HYPERLOOM_INFERASIM_ANCHOR_SCALING`, `HYPERLOOM_INFERASIM_HARVEST`,
  `HYPERLOOM_INFERASIM_HARVEST_PYTHON`, `HYPERLOOM_INFERASIM_HARVEST_GPUS` and
  `HYPERLOOM_INFERASIM_HARVEST_TIMEOUT_SEC`.

A session that had set one of the switches now sees this:

* A variant a filter used to drop is benchmarked. The `skipped_dup` reasons
  `measured_revert_before`, `projection_decisively_slower` and
  `projection_outranked` no longer occur, and a round takes as long as its
  whole grid. Nothing replaces the cross-session skip: a launch that REVERTed
  in an earlier session is measured again in a new one.
* A round no longer writes `_projection/` or `_anchor_spec/` under its output
  directory.
* Hyperloom no longer writes into the anchor-store directory: no
  `measured/*.json` point from a decision round, no `harvested/*.json`, no
  index update and no `.harvest-*.lock`. Files already there are left as they
  are. Tooling outside Hyperloom that read the measured anchors gets no new
  point from a session.
* The measurement-store directory is no longer read or written. Its records
  are left in place and can be deleted.
* No server is booted for an anchor harvest
  (`HYPERLOOM_INFERASIM_MODE=benchmark` booted one when the store had no anchor
  for the regime).

Two packaging changes come with it:

* The `inferasim` extra is removed: drop it from install commands.
* The wheel no longer ships the projection's workload template,
  `hyperloom/inference_optimizer/assets/inferasim/inferasim_workload.yaml`.

The multi-node `infera` backend is a different thing and is unchanged.

### Recommended: read a baseline round's config from its run directory

A benchmark attempt that runs a discarded warm-up round and a measured round
on one server (a genuine baseline, a PRELUDE warm replay, the KERNEL stack
validation and the KERNEL integrate re-baseline) used to write a copy of its
config for each round, `warmup_round/baseline_lifecycle.yaml` and
`measure_round/baseline_lifecycle.yaml`, with a `server_lifecycle` block, a
pinned `PORT` and that round's `RUN_EVAL`. Neither file is written. Both
rounds run the attempt's one config, `baseline_config.with_envs.yaml`; which
round boots the server, the port it answers on and which round evaluates are
set on the round, and the rounds they produce are the ones they produced.

* If you read `baseline_lifecycle.yaml`, read `config.yaml` in the round's
  run directory (`warmup_round/suite_run_<n>/`, `measure_round/suite_run_<n>/`)
  instead: it holds what the round ran, its port and its eval switch included.
* `baseline_config.with_envs.yaml` of such an attempt now carries the round's
  cap as `timeout_seconds` and `PYTHONUNBUFFERED: '1'` under `envs`, as the
  config of a single-round attempt (multi-node, AgentX, a profiled round)
  already did. The server was started with that variable before; the
  attempt's `launch_evidence.json` now also lists it in
  `requested_server_env`, and its `recipe_digest` is the digest of the file
  with those two values.
* A single-round attempt no longer looks for a server record in its task
  directory when it ends; its round stops the server it started, as before.
  The info line `cold-start double-run not eligible (...)` is no longer
  logged.

### Recommended: drop `INFERENCE_OPTIMIZER_RUN_GRID_WARMUP` from `hyperloom optimize` launches

The FRAMEWORK integrate bench discards a first bench exactly where a baseline
does: on a single node, for a served fixed workload that is not profiled.
There it boots one server, runs the discarded bench, then the measured bench
and its eval, and stops the server. `hyperloom optimize` no longer reads
`INFERENCE_OPTIMIZER_RUN_GRID_WARMUP`: set to `0` it used to make the bench a
single round on a server booted for it, while the baseline the bench is
graded against was measured on a warmed one.

* Remove the variable from `hyperloom optimize` launches. A launch that set it
  to `0` now runs the discarded bench: the integrate bench takes one more
  load, and its row carries `run_grid_warmup_discarded_first` and
  `warmup_round_tput:<value>` in `nonfatal_warnings`. A launch that left it
  unset runs the rounds it ran.
* `hyperloom-multibranch` still reads the variable, as the default of a test
  whose request names no warm-up count.
* A variant's two rounds run from its one config,
  `variant_NN_<name>/config.yaml`. `variant_NN_<name>/warmup_round/config.yaml`
  is no longer written, and the variant's `config.yaml` no longer carries a
  `server_lifecycle` block or the port picked for the pair (it keeps a `PORT`
  the point itself names, which the pair does not run on). Read
  `config.yaml` in a round's run directory
  (`variant_NN_<name>/warmup_round/suite_run_<n>/`,
  `variant_NN_<name>/suite_run_<n>/`) for what the round ran, its port
  included. The variant's `launch_evidence.json` follows the file: its
  `requested_server_env` no longer lists the picked `PORT`, and its
  `recipe_digest` is the digest of the file without the block.
* A variant whose discarded bench failed (`warmup_round_failed`) leaves a
  `config.yaml` that carries the round's cap as `timeout_seconds` and
  `PYTHONUNBUFFERED: '1'`, as one whose measured bench ran always did.
* The error class `warmup_yaml_build_error` no longer occurs: the variant's
  config is built once, and a build that fails is `yaml_build_error` or the
  refusal's own class, as before.
* The info line `warmup-before-measure not eligible (...)` is no longer
  logged. Server pids that outlive the stop of the pair's server are logged
  as `held test: pids ... outlived SIGKILL` (was `grid: pids ...`).
* On a multi-node host, and for a profiled or scriptable point, a launch
  that set the variable to `0` runs the rounds it ran, but an integrate bench
  or an EXPLORE variant there must now have session time for one more round
  before it starts, as on a launch that left the variable unset.

### Recommended: read an EXPLORE variant's round config from its run directory

An EXPLORE variant whose server can be held (a single node, a served workload
that is not profiled) runs the rounds it ran: for a fixed workload a
discarded warm-up bench and then the decision bench with its eval, on one
server in the variant's directory `vNN_<name>/`; for AgentX one round. Which
round boots the server, the port it answers on and which round evaluates are
now set on the round. They are no longer written into the two configs the
rounds run from.

* `vNN_<name>/warmup_round/variant_00_<name>/config.yaml` and
  `vNN_<name>/variant_00_<name>/config.yaml` no longer carry a
  `server_lifecycle` block or the port picked for the variant (each keeps a
  `PORT` the point itself names, which the rounds do not run on), and the
  warm-up's file carries the variant's own `RUN_EVAL`, where it used to say
  `false`. Read `config.yaml` in a round's run directory
  (`.../warmup_round/variant_00_<name>/suite_run_<n>/`,
  `.../variant_00_<name>/suite_run_<n>/`) for what the round ran: its
  `lifecycle`, its `server.port` and `eval.enabled`, which is `false` for
  the warm-up whatever the variant asks.
* The `launch_evidence.json` beside each of the two configs follows its file,
  and so does the `launch_evidence` of the variant's ledger row and of a
  KEEP: `requested_server_env` no longer lists the picked `PORT`,
  `recipe_digest` is the digest of the file without the block, and the
  evidence of a variant whose warm-up failed lists the variant's own
  `RUN_EVAL`.
* Every variant is tested on a port of its own. The variants of one EXPLORE
  round used to share one port, one after the other.
* An `explore` task's `baseline_double_run` parameter is no longer read.
  Hyperloom never set it on an `explore` task; a task that carried `false`
  ran the variant's decision bench on a freshly booted server, without the
  warm-up the baseline it is graded against had. A variant of a fixed
  workload whose server can be held now always runs both rounds.
* A variant's server is stopped once, when the variant ends or before its
  AgentX accuracy round. Server pids that outlive the stop are logged once,
  as `held test: pids ... outlived SIGKILL` (was `explore: pids ...`).

### Recommended: allow a specialist's rebench two benches

The optional rebench a GPU specialist can run on its leased cards
(`python -m hyperloom.orchestrator.specialists.rebench`) now measures the way
the integrate bench does. On a single node, for a served fixed workload that
is not profiled, it boots one server on `--port`, runs a bench it discards,
then the measured bench and the eval the point asks for, and stops the
server. In 1.2.x it ran one bench on a server booted for it, while the
integrate bench its figure is compared with was measured on a warmed server.
The flags and the keys of the JSON result are unchanged, and `port` still
reports the port the server answered on.

* `output_throughput` and the latency figures are the second bench's. Do not
  compare a rebench result from before the upgrade with one from after.
* A rebench takes about two benches of wall time: one boot, as before, and
  one more load. The specialist prompt says so.
* `warnings` of a rebench that measured carries
  `run_grid_warmup_discarded_first` and `warmup_round_tput:<value>`. A rebench
  whose discarded bench failed is `ok: false` with `status: failed`, carries
  `run_grid_warmup_round_failed` beside `rebench_failed:<error>` in
  `warnings`, and names the discarded bench's run directory as `workspace`;
  its `abort_reason.json` has `error_class: warmup_round_failed`.
* The discarded bench writes its run directory under
  `<output>/variant_00_specialist-rebench/warmup_round/`. The variant's
  `config.yaml` no longer carries a `server_lifecycle` block or `--port` as
  `PORT` (it keeps a `PORT` the point itself names, which the rounds do not
  run on); read `config.yaml` in a round's run directory for the port and for
  which round evaluates. `launch_evidence.json` follows the file: its
  `requested_server_env` no longer lists that `PORT`, and its `recipe_digest`
  is the digest of the file without the block.
* A rebench of a point that is measured in one round runs the rounds it ran:
  one round on a server of its own on `--port` for an AgentX or profiled
  point, and on a multi-node host the cluster's restart, its discarded pass
  and its measured pass. The `config.yaml` and `launch_evidence.json` changes
  above apply there too. On a single node where `BENCHMARK_BASE_URL` is
  exported, the rebench sends both benches to that server, as the integrate
  bench does.

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
| `python -m hyperloom.agents.critic.runtime.cli {prepare-review,commit-review}` | Critic backend subprocess (not operator-facing) |
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
