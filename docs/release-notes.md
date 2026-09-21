---
myst:
  html_meta:
    "description": "Hyperloom release notes: headline capabilities for version 1.1.1, a patch release that corrects the Recipe knowledge base warm-start path, unifies agent-backend selection, bounds supervisor watchdog restarts, and bumps the bare-metal vLLM default; plus the 1.1.0 feature release."
    "keywords": "Hyperloom, release notes, LLM inference, AMD GPU, ROCm, agentic optimization, TraceLens, GEAK, KernelForge, Primus-Claw, bare metal, kernel optimization"
---

# Hyperloom release notes

The current packaged version is 1.1.1 (`pyproject.toml`). For the
per-change history since the initial snapshot, and for a detailed breakdown of
all previous Hyperloom pre-release versions, see
[Releases](https://github.com/AMD-AGI/Hyperloom/releases); this page
summarizes the headline capabilities and carries the changes on `main` that no
release has shipped yet.

## Unreleased

Merged to `main` and not yet carried by a tagged release. Each entry moves
into the [release](https://github.com/AMD-AGI/Hyperloom/releases) that ships
it.

- **Simplify optimizer lifecycle and benchmark limits.** Remove the Robustness
  agent/runtime RCA, runtime `recover` action, Monitor/Supervisor automatic
  supervision and resume, and task/lease age expiry. Each actual benchmark spawn
  uses a 7800-second hard deadline (including boot and accuracy), plus a
  600-second output-silence limit only after a ready marker is observed in this
  round's logs; both are finite positive settings. A warm-reuse hint alone does
  not arm silence, avoiding false kills when original Magpie buffers client
  output and the reused server writes its previous round's log. Reuse rounds
  without a current ready marker remain bounded by the hard deadline, session
  budget, and cancellation. Output cannot extend the hard deadline. Session
  cancellation and admission/phase budgets remain, as do explicit `--resume-from`,
  offline `recover-session`, process cleanup, and historical SBDv6 readers.

### Added

- **An enablement session now ships an ordered replay recipe, and a verdict on
  whether it can be replayed at all.** The session's durable state said what was
  kept and nothing about how to reproduce it: an operator holding a KEEP had a
  list of patches, no order to apply them in, no record of which tree each was
  written against, and no way to tell a complete stack from one whose evidence
  was never captured.

  `recipe_steps` projects the state onto an ordered array — setup, build, patch —
  each step naming the root it applies to, the targets its own diff declares and
  the identity of what each install consumed. `replay_sufficiency` judges that
  array and reports `sufficient` or `insufficient` against a closed vocabulary of
  reasons, each naming what it blocks: replay, assertion validation, or both. A
  patch whose targets nobody recorded, a build nothing replayed, a root with no
  base commit and a credential the recipe cannot supply are all refusals, not
  assumptions. Both appear in `session_breakdown.json` under `enablement.recipe`,
  and the session package now carries the KEEP's source overlay the steps
  reference, so a `sufficient` recipe does not ship with its own evidence
  missing.

  The KEEP records what the two need: each root's identity and base commit taken
  before the round's first mutation, byte-exact snapshots of every declared
  target, the environment closure and installed versions read through the
  interpreter the accepted bench launched, and an append-only ledger of the setup
  commands as they ran. Credentials are classified and sanitised on emission.

- **`ray.init`'s connect is bounded.** It had no timeout at all, so an
  unreachable head node hung the leg until the session clock ran out instead of
  failing it.

### Fixed

- **KernelForge results could be lost or integrated in the wrong order.** KB
  warm-start commits now force only their approved pathspecs, so tracked files
  matched by a repository's ignore rules cannot reject the whole commit. Task
  preparation stages tracked edits and only newly created files, preventing
  pre-existing JIT caches and generated artifacts from leaking into exported
  patches. Forge-loop stdout and stderr are retained beside each task result,
  and Controller patches are integrated by `task.json.priority` rather than
  encoded directory-name order so cumulative E2E validation follows the
  opportunity analyst's ordering.

- **Inline MCP actions no longer block the coordinator's event loop.** The
  `run_action_now` context tool awaits the action's result without blocking the
  loop or occupying the thread pool needed by database operations. Action
  execution, timers and caller cancellation can proceed concurrently. Direct
  calls to the synchronous bridge on the coordinator loop now fail immediately
  without scheduling work. Existing inline wait limits and registered-action
  completion after a caller timeout remain unchanged.

- **A Slurm row declaring a workload shape was benchmarked at the defaults.**
  The optimizer resolves `tp` / `conc` / `ep` / `isl` / `osl` / `precision` as
  flag > persisted state > default and deliberately never reads them from the
  environment, but `_incontainer.sh.in` only exported them. A row declaring
  `tp=4` therefore materialised a `tp=1` baseline, where sglang's rank math
  divided by zero; a row declaring `isl=8192 osl=512` silently measured
  1024/1024, and `precision` came from the checkpoint sniffer rather than the
  row. The whole declared shape is now passed as flags in both the python and
  claude backends. The exports stay, because the framework recipes downstream
  do read them -- that split is now stated where the command is spelled out,
  since the old wording pointed the carrier at the ignored mechanism. `ep_size`
  also gains a producer: it is a new trailing `models.tsv` column, parsed and
  exported by `run_hyperloom.sbatch` and added to the docker backend's `-e`
  allowlist so enroot and docker agree on the shape. It trails `target_gain` so
  existing 13-column rows still parse, and an absent value keeps expert
  parallelism at 1.

- **A cancelled job's container kept its GPUs and broke the next job on that
  node.** `docker run --rm` cleans up when its client exits normally, but a
  `scancel` or NODE_FAIL kills the client and leaves the container running, so
  `--rm` never fires and the weights stay resident. The next job then failed in
  whichever way its framework noticed first: sglang sat in
  `wait_for_amd_gpu_clean` for the full fifteen minutes because that gate maxes
  VRAM% over every GPU on the box, while vLLM was refused outright with
  `Free memory on device cuda:N ... is less than desired GPU memory
  utilization`. Observed on two nodes at once, where a container from a job
  cancelled three hours earlier still held ~168 GiB on each of GPU 0-3 --
  exactly where the next tp=4 server wanted to land, because the container gets
  no ROCR mask and every framework starts from GPU 0. The launcher now reclaims
  those containers first, deciding ownership by liveness rather than by name or
  image: `docker run` carries `-e CLAW_SESSION_ID=`, so a container whose
  session id has no live client is ours and orphaned, and a co-tenant job that
  still has its client is left alone. `docker run` also gains `--init`, without
  which the container's pid 1 becomes an unreapable zombie once the client is
  killed and even the daemon refuses to remove it (`PID <n> is zombie and can
  not be killed. Use the --init option ...`); the reclaim keeps a cgroup-level
  SIGKILL fallback for containers already in that state. `--name
  hl-<key>-<jobid>` makes a running container traceable back to its job.

- **A vLLM profile round had its profiler bounds dropped before launch.** The
  argv preflight probe sees only `EXTRA_VLLM_ARGS`, while the launcher appends
  `--profiler-config.profiler torch` and a trace directory of its own
  afterwards. `ProfilerConfig` refuses the iteration bounds this layer injects
  unless both are present in the same fragment, so the probe rejected an argv
  that is valid once the launcher's flags are appended, and the round then ran
  with no bound on the capture window. Both flags are now asserted alongside
  the bounds so the probed fragment is self-consistent on its own. The trace
  directory is a placeholder: the launcher's own value has to win vLLM's
  last-wins dotted-flag merge, so the bypass backend now emits its profiler
  flags after `EXTRA_VLLM_ARGS` the way Magpie's launcher already does, rather
  than before it where the placeholder would have won and sent the trace
  somewhere trace discovery never looks. An operator-set profiler flag is left
  untouched.
- **A campaign the host killed cost the next task in the same repository.**
  Every in-place task is handed the same `forge_experiments` directory, and the
  release archives it -- but a run that was killed never reaches the release.
  The leftover then did two things: forge-loop refused the workspace outright
  ("already contains a Forge campaign; pass --resume to continue it"), failing
  the next task at dispatch, and recovery read the stale manifest as that
  task's own best result, reporting the dead campaign's commit as a missing
  base commit -- a reason with nothing to do with the task it lost. Measured: a
  session lost a `chunk_gated_delta_rule` task to a fusion campaign left behind
  at a timeout an hour earlier. A leftover is now archived on the way in as
  well as on the way out, kept rather than deleted because it is the only
  account of what that run did, and a trusted manifest has to name a commit the
  repository still has before it is read as this task's result. An archive that
  cannot be made says so, since the dispatch refusal that follows is otherwise
  undiagnosable.
- **A campaign killed mid-search lost a fusion it had already published.**
  `run_campaign` publishes each winning iteration to the shadow repo's
  `forge_experiments/best/` and points `forge_loop_<stem>.json` at it, but the
  exported patch and the aggregate manifest only land once the campaign
  returns. A wrapper timeout while it was still iterating therefore reported
  REVERT with `patch: null` even though correctness had passed. Measured: a
  session lost a 5.011x fusion of qkvgate split + QK norm + RoPE with its
  experiment still running when the 5400s timeout fired. Salvage now falls back
  to the per-campaign artifacts, and each salvaged row carries the env flag its
  fused path is gated behind -- read back from the driver the campaign wrote,
  since the patch does not carry it and without it the re-baseline server boots
  un-gated, measures the eager path and rejects the win one stage later. A
  campaign whose flag cannot be read is not salvaged at all, rather than queued
  to fail that way. Artifacts a previous run left in the same output directory
  are swept before the run starts, so they cannot be salvaged as its own.
- **A KB recipe carrying code overlays could not be replayed into a framework
  installed from a wheel.** A Recipe whose patch timeline is non-empty replays
  as required, and that path refused any tree without a git HEAD -- which a
  pip-installed framework never has. Every code-level optimization a session
  published therefore became unreplayable the moment it reached the KB: warm
  replay failed before booting a server, and the recorded gain could only be
  re-earned from scratch. What promotion needs is a way to unwind the tree if
  the replay is rejected, not a sha, so it now accepts either channel: a git
  checkout's snapshot, or the backups a nogit apply records as it writes. An
  overlay the tree already carried applies as a no-op and records neither,
  which is not the same as an apply whose artifacts were lost, so each tree now
  states whether the round wrote to it -- a no-op tree promotes and is skipped
  by the rollback, a written-to tree still has to answer with a channel, and a
  record persisted ahead of the apply reads as written-to so a resume restores
  it. Framework-agnostic; the shape of the checkout decides the channel.
- **A shipped aiter tuned CSV naming a kernel this host never compiled failed
  every boot of the session.** aiter resolves its tuned tables two ways: a
  pinned `AITER_CONFIG_*` env is taken as the exact `:`-joined list, and an
  unset one falls back to the shipped default plus every `model_configs`
  overlay whose name matches. A round that tunes one operator sets only that
  operator's variable, so the others take the unset branch and pull in overlays
  cut on another host -- tables whose `kernelName`s are absent from this
  machine's compiled `module_*.so`. Serving then aborted at load with a
  registry mismatch, and because the table is shipped rather than produced by
  the run, every retry hit the same wall: the whole session failed at boot with
  nothing to roll back. The CSV set a boot will actually load is now resolved
  by aiter's own two-branch rule and checked against the compiled modules
  before the config is materialized; an uncovered module is unlinked so the
  next boot rebuilds it. On integrate, a registry mismatch also drops the
  modules the error names, not only the ones the round's environment mapped.
  Framework-agnostic; it is the aiter install that is repaired, not the server.
- **An author whose transport died took the whole fusion lane with it.** A lane
  costs hours and an authoring call costs minutes, but a provider that never
  delivered an answer -- a stream stalled mid-response, a connection reset --
  ended the lane on the first failure. The classification that tells "the model
  never answered" apart from "the model answered nothing" already exists in
  `llm_failure`, so authoring now consults it where the exception is still in
  hand and retries only the transport, with backoff and against a deadline. A
  provider safety stop is still never retried, because retrying one is the
  anti-pattern the session-resume allowlist already refuses; neither is a
  timeout, which has just spent a full attempt's budget. The deadline defaults
  to what the configured attempts can legitimately cost, since the generic
  1800s LLM default is shorter than a single 7200s authoring attempt and would
  have made the retry unreachable.
- **A fusion campaign killed before it returned reported REVERT while proven
  work sat on disk.** `fusion_manifest.json` is the only artifact that points
  at a keeper, and it was written once every campaign had returned. `on_keep`
  exports the patch and smokes it the moment a recipe is kept, so a wrapper
  killed between the last keeper and the aggregate reported `patch: null` even
  though a sibling had already passed correctness and a serving smoke.
  Measured: a session lost a 5.011x fusion of qkvgate split + QK norm + RoPE
  this way, the iteration having published 44 minutes before the timeout fired.
  The manifest is now published as each keeper is proved, so it is never
  missing -- only as complete as the run got -- and the existing salvage path
  reads it unchanged. The end-of-run write still overwrites it with the final
  loop, compile-pass and error fields before any exit, and a sibling the smoke
  rejected has its patch unlinked so no reader can find work the run refused.
- **AgentX baselines retain request-quality grading with `RUN_EVAL=false`.**
  Missing AIPerf error-rate metrics are derived from profiling request counts;
  zero errors are inferred only with successful requests and an explicitly empty
  error summary. Warmup accounting is excluded, and invalid or unknown evidence
  remains fail-closed. Valid zero-error baselines no longer lose their quality
  signal merely because serving lm-eval is disabled.
- **Honor concurrency-sweep budgets without treating the hard cap as a start cost.**
  Admission uses the measured expected duration when available; unknown-duration
  work may start while budget remains. Boot retries, reuse and fallback share
  the earlier sweep/session deadline, and reports retain the actual stop source.
  The manual sweep driver no longer passes or advertises the retired
  `--variant-timeout-sec` option.
- **Complete cooperative build and specialist cancellation without accepting
  unconfirmed cleanup.** Cancellation reaches pending work and running workers;
  confirmed cleanup records a cancelled outcome, while unknown cleanup retains
  ownership. Completed outcomes remain in task history for diagnosis even when
  cleanup fails, without publishing them for promotion or retry. Completion
  callback failures after confirmed cleanup no longer retain execution entries,
  and repeated inline calls return stored terminal results instead of rerunning.
- **Refuse unsafe session resumes explicitly.** Legacy or foreign execution
  ownership and unproven historical cancellations now produce bounded task/lane
  diagnostics before resume writes or dispatch. Resume admission itself clears
  no ownership. After independently verifying that a task's complete process tree,
  remote workers and Ray actor have stopped, operators can use
  `recover-session --confirm-stopped TASK_ID --confirmation-reason TEXT` to record
  that confirmation, cancel an unfinished task, and release only its unattributed
  execution/GPU records under the POSIX session lock. This explicit operation preserves rounds
  and other tasks, rejects nonempty owner scopes, and neither stops workers nor
  accepts old results or starts a resume. Existing `--force` remains report-only.
  Resume admission and
  round reconciliation honor the latest recorded cleanup outcome, including
  results recorded after an earlier terminal transition. A Ray worker whose root
  exited is not treated as proof that detached descendants exited, and a missing
  stop acknowledgment no longer destroys the specialist actor's cleanup channel.
  Specialist cleanup makes one bounded follow-up confirmation on the same lease
  before retaining unconfirmed ownership; a late acknowledgment is consumed by
  the existing completion path rather than requiring a new background reaper.
- **Restore safe AITER lock cleanup at baseline startup.** Stale locks are cleaned
  only when compiler absence is established. Unreadable live-process identity
  leaves locks untouched; known zombies do not block cleanup.
- **The AITER version in `stack_fingerprint` is now the AITER that is actually
  installed.** Two independent faults made that field untrustworthy.

  The env tuple read `AITER_COMMIT` and `AITER_VERSION`, neither of which
  anything in this repo writes, so the env path never produced a value.
  `install_baremetal.sh` already resolves the exact tag it installs, exports it,
  and persists it to `.env` as `AITER_REF`, which the dotenv loader admits under
  its `AITER_` prefix — so the value was sitting one key away the whole time.
  `AITER_REF` joins the tuple, behind `AITER_COMMIT`. It also covers the default
  isolated vLLM path, where aiter lives in the framework venv and no in-process
  probe can see it under any name.

  The probe then looked up a distribution named `aiter`, but AITER renamed itself
  to `amd-aiter` at v0.1.8, so the lookup missed every host running v0.1.8 or
  newer. Worse, on PyPI `aiter` is an unrelated 2019 async-iterator library, so
  where that package happened to be installed the probe recorded its version —
  `0.13.20191203` — as the AITER version. The old name is corrected rather than
  kept as a fallback, precisely so that value can no longer be produced: nothing
  recorded is better than something that looks like an answer. Hosts older than
  v0.1.8 are covered by `AITER_REF`, which is exact.

  Only what gets *written* changes. No read path compares `rocm` or `aiter`
  against the pod today; that gap is tracked in #1507, and getting the recorded
  value right is a prerequisite for it — a comparison fed `0.13.20191203` would
  report a confident mismatch against every real AITER build.

- **AgentX grading failures no longer fall back to throughput KEEP.** When an
  AgentX session cannot grade on interactivity because either side is missing
  the axis pair, explore, ``_lift_to_current_best``, and
  ``resolve_graded_comparison`` fail closed instead of promoting on the
  diagnostic output figure. The removed ``ANCHOR_DEGRADED`` round-local output
  fallback is part of the same rule. ``degrade_reason`` still travels on the
  explore, stack-validation and integrate rows that record a refused
  comparison, so the breakdown can still name why a variant did not KEEP; an
  *adoption* row can no longer carry one, because the resolver now returns
  REVERT on a degraded pair and a REVERT never reaches the adoption writeback.

- **A partitioned card is now a different machine in the KB key, so a warm-start
  hit can no longer replay a config tuned on a differently shaped one.** The
  `canonical_id` is a seven-tuple of model, hardware, framework name, model type,
  architectures, framework version and precision. The compute-partition mode was
  not among those dimensions, and neither was expert parallelism on a single
  node, so `kb_hardware_slug` collapsed to the bare GPU type and a run in SPX and
  a run in CPX landed on one identity — `inference:qwen3-32b:mi355x:...` either
  way. The warm-start cascade only relaxes `conc`/`isl`/`osl`, so nothing
  downstream caught it either: an `exact` tier hit at confidence 1.0 could hand
  the auto-replay a config recorded with eight times the partitions, and the
  `--warm-replay-min-reproduce-pct` backstop only noticed after spending the
  verify round.

  `kb_hardware_slug` now suffixes the partition mode and `ep` at any node count,
  not just on a cluster: both are fixed at launch rather than explored, which is
  the argument `_tp{tp}` already makes for itself. A CPX pod therefore cannot
  read an SPX row because it is asking a different `canonical_id` — no flag, no
  demotion, and no second comparison that could be applied to a different row
  than the one that gets replayed, since `resolve_kb_topology` is the single call
  both the reader and the writer build the key from. Every suffix is omitted at
  its default value (`ep <= 1`, SPX, or a mode nobody published, including one
  this build does not recognise), so existing keys stay byte-identical and
  nothing in the corpus moves. `_TOPOLOGY_SUFFIX_RE` learned the single-node
  forms too, so `_hardware_fallback_values` still offers the same-ISA SKUs for
  exactly the rows these suffixes were added for.

  `workload_shape` still publishes `ep` and `partitions` as a description of the
  run, and `knowledge_to_warm_recipe` derives its projection allowlist from the
  publisher rather than restating it, which is what had been silently dropping
  keys the publisher emitted. Both are omitted at their default: `--ep` defaults
  to 1, so publishing it would have every dense run claim a formation it never
  chose, and one partition is the whole card. The count a launch published wins
  over one re-derived from the mode name, so there is only ever one derivation to
  keep in agreement.

### Removed

- **Every non-official LLM key name as a Hyperloom credential.** `OPENAI_API_KEY`
  and the three Anthropic-side names are the only credentials the optimizer
  authenticates with. A survey of the eight vendor and gateway aliases found a
  credential read path behind exactly one of them: `LLM_GATEWAY_KEY`, which
  `resolve_openai_client_config` accepted one rung above the Anthropic fallback
  and which `codex_session` would name as the Codex key variable. `LLM_API_KEY`
  had one too — the robustness RCA resolver, when the OpenAI side carried no key
  — and it went with the agent itself; nothing reads it as a credential to
  remove here. The other six — `AMD_API_KEY`, `AMD_LLM_API_KEY`,
  `GEAK_API_KEY`, `LLM_PROXY_API_KEY`, `CLAW_API_KEY` and the LLM-side
  `SAFE_API_KEY` — never authenticated anything here at all.

  Two of the gateway key's read sites were a credential name doing a second job
  it had no business doing. `_provider_only_mode` treated a bare `LLM_GATEWAY_KEY` as
  proof that the host fronted both protocols, which suppressed single-provider
  detection; single-provider intent is now read off the two sides' own URL and
  key. TraceLens took the same variable as an on-its-own gateway marker, which
  `HYPERLOOM_STRICT_GATEWAY_MARKERS` exists to express three lines below the
  check that was removed. The specialist secret allowlist also stops forwarding
  `LLM_GATEWAY_KEY` to the child, which could not have authenticated with it
  once `codex_session` no longer resolves it.

  A deployment whose `.env` carries `LLM_GATEWAY_KEY` as its only OpenAI-side
  credential has to rename it to `OPENAI_API_KEY`; the value and the endpoint
  are unchanged. An Anthropic-only host that also carried the gateway key now
  resolves its Anthropic bearer for OpenAI-protocol calls instead, which is the
  credential it was already using for everything else.

  The OpenAI key also stops being copied into `LLM_API_KEY`, `AMD_LLM_API_KEY`,
  `AMD_API_KEY` and `LLM_GATEWAY_KEY` — by CLI preflight into its own
  environment, and by the kernel agent into the Ray runtime env — and the four
  names leave the Ray allowlist, so an operator's own export no longer reaches a
  worker either. That mirroring was kept on the reading that GEAK receives its
  credential through those names. It does not: GEAK authenticates on
  `GEAK_AMDKEY` or `OPENAI_API_KEY`, and of the four only `AMD_LLM_API_KEY`
  appears in that tree at all, in a quick-start line invoking a script the
  repository no longer carries. `LLM_API_BASE` stays — it addresses an endpoint
  rather than authenticating to one, and preflight still derives it from the
  resolved OpenAI-side URL. All eight names also stay in the benchmark secret,
  variant, external and specialist-redaction lists: membership there asserts
  that a name holds a secret worth scrubbing, not that anything consumes it, and
  an operator's stale export still needs stripping from child processes.

- **`--recipe-kb-strict-fingerprint`.** It was declared in the parser and read
  nowhere, and it promised to refuse rows whose `stack_fingerprint` disagreed
  with the pod — which was never the exposure, since framework version and
  precision are already identity dimensions. Encoding the partition mode in the
  key makes the mismatch it would have caught unrepresentable, so a read-side
  comparison has nothing left to do. `rocm_version` and `aiter_commit` are still
  written into every row's `stack_fingerprint` and compared nowhere at read time;
  that is a real gap and is tracked in #1507 rather than under a flag whose name
  says fingerprint and whose behaviour would have been workload shape.

### Changed

- **breaking: the SBD V6 concurrency-sweep comparison rows are named for the
  axis they carry rather than for throughput.** `baseline_throughput` →
  `baseline_value` and `optimized_throughput` → `optimized_value` in each
  `conc_sweep` event's `comparison` rows. The old names were a lie on an AgentX
  session: the figure they held is whatever `result.metric` names, which is the
  slow-tail interactivity percentile (`e2e_norm_intvty_p90`, milliseconds)
  whenever the session grades on one, so a consumer reading `*_throughput` was
  plotting latencies as throughputs against rungs measured in tokens/s. No
  alias is kept — an alias would preserve exactly the misreading the rename
  exists to stop. External parsers of `reports/sbd_v6/` must be updated.

  The rows also gained `baseline_guard`, `optimized_guard` and `guard_holds`,
  and the roll-up gained `guard_axis` and `best_conc_guard_holds`: the throughput
  the session would have held a promotion to, reported beside the ranked axis
  rather than enforced, so a rung that bought interactivity by giving up
  throughput is visible as such instead of reading as a clean win. All are null
  off the interactivity objective. `guard_holds` is tri-state — `null` means the
  framework never answered, which is not the same fact as `false`.

- **SBD V6 publishes the axis a session graded on, at the session level and on
  each settled figure.** Three additions to the wire shape, all of them facts no
  consumer could previously recover:

  - `metadata.grading` — `{benchmark_mode, objective, tput_guard: {enabled,
    noise_pct}}`. An AgentX replay is ranked on `e2e_norm_intvty_p90` with total
    throughput held as a guard; a synthetic run is ranked on output throughput
    alone. Every throughput field elsewhere in the document is the output axis by
    construction and `benchmark_mode` never reached the breakdown at all, so
    without this block the two kinds of session are indistinguishable — and on
    the canonical corpus the two axes differ by roughly two orders of magnitude,
    which is enough for a consumer to sort one against the other and never
    notice. Recorded from `SharedState.grading` rather than re-derived at export:
    the axis is settled once at seed, where the run can still see its own
    configuration, and re-deriving it here would read the exporting subprocess's
    environment. `tput_guard.noise_pct` is `null` on a session seeded before the
    band was recorded — the band that session applied is unknown, and today's
    default is not evidence of it.
  - `outcome.baseline.perf` and `outcome.final.perf` — the four graded axes the
    measurement reported (`e2e_norm_intvty_p90`, `total_throughput`,
    `input_throughput`, `tpot_p90_ms`), each an explicit `null` where nothing
    measured it. All four keys are always present: absent would be
    indistinguishable from an axis the framework failed to report, and zero reads
    as "measured, and it was zero". A synthetic run publishes four nulls.
  - `outcome.final.graded_on` and `outcome.validation.graded_on` — the axis the
    gain beside them is on. The stack reconciliation has to be single-axis, since
    an attributed figure on one axis against an unattributed figure on another
    makes the gap meaningless, and `graded_on` is what names it.
    `outcome.validation.perf` carries the settled measurement's own axes on the
    same row as the gain they produced, because a revalidation moves the
    cumulative figure without re-promoting the recipe.

- **One definition of the agent default model ids, and the backend picks its own
  last rung.** `DEFAULT_CLAUDE_MODEL` and `DEFAULT_CODEX_MODEL` lived in
  `orchestrator/roles/agent_role.py` and again in
  `kernelforge/agent_backends/{claude,codex}.py`, and the two values were also
  spelled as bare literals in the Forge provider registry beside the module that
  defined them, in the Claude allowlist head, in the GEAK model default, in the
  TraceLens Claude path and in the Critic's Codex field. Eight copies of two
  strings, each free to drift. They now live in `hyperloom/common/llm_config.py`
  next to `AGENT_BACKEND_CLAUDE` / `AGENT_BACKEND_CODEX`, which is the pair they
  are keyed by, following the `DEFAULT_REASONING_EFFORT` precedent that both
  packages already import from `common`.

  `resolve_forge_llm_model` lost its `default` parameter. Every caller passed the
  chosen backend's own default, and `request_handlers` reimplemented the
  per-backend branch the function already performs to work out what to pass;
  `patch_conflict_merge` carried a comment at each call site explaining that a
  default is mandatory because `CLAUDE_MODEL` is unset on OAuth-token runs and
  the resolver would otherwise post an empty model id. The function knows the
  backend, so it now answers that itself and the parameter that could be
  forgotten is gone.

  The test that asserted the allowlist head equals `DEFAULT_CLAUDE_MODEL` is
  gone with it: the head is now that constant by construction, so the drift it
  watched for is unrepresentable. Model knobs that merely share a value today
  are deliberately untouched — the narrative report model, the RCA model, the KB
  synthesis model and the quantization driver model each have their own
  override and their own reason to move, and collapsing them onto one constant
  would couple decisions that should stay free to diverge.

- **Roofline CUDA graph capture failures are classified instead of retried in
  eager mode.** When profiling cannot capture a graph, the executor records a
  structured failure category and writes a diagnosis artifact rather than
  rebooting the server in eager mode and retrying. Timeline rows now publish
  ``graph_capture_disabled`` instead of ``eager_fallback_applied``. Blocking
  filesystem and liveness probes run in ``asyncio.to_thread`` so the roofline
  action no longer stalls the coordinator event loop.

- **ENABLEMENT is the sixth phase of the optimization loop.** Bring-up used to
  run inside FRAMEWORK_AGENT, which left it a lane with no lifecycle of its
  own: it could not be entered, exited or reported on, and a phase that owned
  optimisation work was also carrying the work of making the combo run at all.
  It is now a phase of its own between PRELUDE and FRAMEWORK_AGENT, with entry
  and exit predicates in `compute_next_phase`, its own `phase_history` rows and
  a section in the Markdown session report. `PHASE_NAMES` is six long.

  **No wall-clock budget is apportioned to it.** `DEFAULT_PHASE_BUDGET_PCT` has
  no ENABLEMENT key, and an absent key means no cap rather than a zero one: a
  budget apportions optimisation effort, and a combo that cannot run has
  nothing to optimise yet. `PHASE_FRAMEWORK_AGENT` drops 0.40 → 0.38 and
  `PHASE_KERNEL_AGENT` 0.50 → 0.47, so the table now sums to 0.95 and bring-up
  is bounded by the run's wall clock and by `ENABLEMENT_MAX_ATTEMPTS` instead.
  The phase's terminal exit is `enablement_attempts_exhausted`, which
  `enablement/lane.py` sets.

  **Runnability is decided from the measurement, not from a log scan.** A combo
  counts as served once it has produced positive throughput and completed
  requests, which is a signal the baseline already carries; the `booted`
  property it replaces scanned the server log for bring-up milestones and could
  not witness one past the head of a chatty log. Both enablement origins — a
  boot failure and the accuracy gate — now open the same baseline revalidation
  window, and the baseline is Coordinator-owned while the phase is ENABLEMENT.

### Fixed

- **A KernelForge Controller KEEP now reaches the stack ledger in every
  benchmark mode, not only under AgentX.** `_run_kernel_rewrite_controller`
  handed `integrate_controller_patches` the session-owned writeback only when
  `agentx_active` held; every other mode fell back to a module-local recorder
  that wrote `optimization_stack`, `current_best` and
  `cumulative_gain_validated` directly. That writer bypassed
  `_lift_to_current_best`, which is the only caller of
  `stack_event.record_adoption`, so a controller KEEP in synthetic mode landed
  in `state.json` and never produced an adoption row — `session_breakdown` sat
  behind the state it was meant to describe.

  This is visible in the document: for a non-AgentX run the kernel bucket and
  its `by_backend.forge` split go from empty to populated, and
  `validated_total_gain_pct` / `at_head` now account for controller KEEPs. The
  KEEP result carries `backend: forge` / `engine: kernel_rewrite_controller`, so
  the adoption is attributed rather than folded into `unattributed`. The local
  recorder and the `record_keep is None` branch that selected it are gone and
  `record_keep` is now required, so a caller cannot silently reacquire the
  no-ledger path.

- **A measured Controller patch was dropped because the patch kept before it
  had moved its context.** Lanes run in parallel from one pinned base commit,
  so two lanes touching the same file each ship a diff written against that
  same base; integration applies them one at a time and commits every KEEP,
  which leaves the later diff stale by the time its turn comes. Measured in the
  Kimi-K3 session of 2026-09-13: `flydsl_moe_stage2` (1.1727x micro) was lost
  to `error: patch failed: aiter/ops/flydsl/moe_kernels.py:14` once
  `flydsl_moe_stage1` had landed, although the two touched disjoint functions
  and defined disjoint module-level symbols -- they collided only because each
  inserted its own sweep helpers at the same anchor. A refused diff is now
  rebuilt in escalating steps: `git apply -3` for pure line drift, then keeping
  both sides of every conflict region whose merge base is empty, then an LLM
  for the regions where the lanes genuinely edited the same lines -- one region
  at a time, carrying the surrounding source as context rather than the file to
  rewrite. Whatever the last two steps reconstruct is discarded unless it still
  contains every line the incoming patch and every landed KEEP added, parses,
  carries no conflict marker and redefines no module-level name; a lane that
  fails any of those is dropped exactly as it was before, with the worktree put
  back to HEAD. Nothing here decides a KEEP: the E2E gate downstream still
  measures and still reverts, so a bad merge costs what a dropped patch already
  cost and can never produce an unmeasured KEEP. A lane that landed as a merge
  rather than verbatim is reported as `merge_strategy` on the integration
  result and in `summary.json`. A patch that applies cleanly takes the path it
  always took, and the resolver runs on whichever backend
  `preferred_agent_backend` picks for this deployment -- Claude through the
  single-shot Anthropic transport, Codex through `achat_completion` -- and is
  skipped entirely when neither side is credentialed.

- **Orchestration picked its CLI from the endpoint shape instead of the rule
  every other agentic role had been moved onto.** `cli/backends.py` was outside
  #1472's reach and still asked `is_openai_only()` for the coordinator, so it
  read neither credential nor installed SDK. One shape moves now that it falls
  through to `preferred_agent_backend()`: both sides configured with only the
  Codex extra installed runs orchestration on Codex rather than on a
  `ClaudeBackend` that cannot import its SDK. A single-provider launch still
  pins its own CLI ahead of the shared rule, because by that point the CLI has
  already rewritten one model id into the other's, and that is the only launch
  the ranked path is not consulted for.

  **The ranking was written twice.** `preferred_agent_backend` and the Forge
  registry's `select_default_agent_provider` each built the credential-then-SDK
  sort key themselves and were kept in step by two docstrings pointing at each
  other. Both now sort on `llm_config.agent_backend_rank`, the registry probes
  the optional SDKs through `llm_config` instead of repeating `find_spec`, and
  the CLI's pass-through wrappers around `is_anthropic_only` / `is_openai_only`
  are gone along with the model defaults that re-derived the same shape test by
  hand. That last one was a narrower copy: it compared the two base URLs and
  never read `OPENAI_API_KEY`, so an `ANTHROPIC_API_KEY` + `OPENAI_BASE_URL`
  launch defaulted `--claude-model` to a Codex model id.

  **The two sides disagreed about what authenticates the OpenAI side, and the
  disagreement is settled by retiring the extra name rather than by accepting
  it.** `openai_agent_credentialed` recognised `OPENAI_API_KEY` while
  `codex_session` would also name `LLM_GATEWAY_KEY` as the Codex key variable,
  each keeping its own copy of the list. Teaching the predicate the session's
  longer list was the wrong end to fix: `_validate_credentials` has only ever
  admitted a run on `OPENAI_API_KEY`, so a host whose sole OpenAI-side
  credential is the gateway key exits at preflight and never reaches a ranking
  to be misrouted by. Both now name `OPENAI_API_KEY` alone, which is what the
  retirement under *Removed* makes true of the remaining read paths as well. No
  running deployment moves on this paragraph; it was the two lists that were the
  exposure.

- **An accuracy eval that failed because the server was gone was read as a
  missing framework capability.** `run_eval` reports a vanished server and a
  model that scored badly the same way -- a non-zero exit -- so the eval-rooted
  branch stamped both as an eval-failure contract and handed them to the
  enablement lane. Measured: a baseline whose throughput pass had already
  completed lost its server mid-eval, the client's next request was refused,
  and the run then spent five specialist rounds hunting a capability gap the
  evidence never supported before stopping on the stall cap with a terminal
  reason that named enablement rather than the server. A refused connection is
  now separated out: the baseline still fails, nothing is salvaged and the
  accuracy gate is untouched, but it counts as an ordinary baseline failure so
  the existing total-failure backstop ends the run on the cause it actually
  had. Framework-agnostic; the eval path is shared by vLLM, SGLang and ATOM.

- **`--framework atom` defaults the kernel backend to forge.** The kernel phase
  runs on atom, but GEAK -- the backend every framework gets unless the
  environment opts into forge -- is on weaker ground there: its extraction rules
  forbid guessing a rewrite seam on a quantized, non-vLLM backend and require
  resolving one from the live server, which is unproven on atom. The default phase split gives that phase half the session, so
  defaulting to GEAK on atom meant defaulting to half a session of nothing,
  while the CLI printed that kernel-agent was "wired for atom". On atom an unset
  `KERNEL_OPT_BACKEND_ORDER` is now filled in with `forge` before the session
  records its backend, and the choice is reported at launch. A value the
  operator set is kept, so running GEAK on atom deliberately stays possible; the
  CLI warns that its seam resolution is unproven there. `--no-kernel` skips the
  defaulting entirely. Only the `framework == "atom"` branch is touched; SGLang
  and vLLM keep GEAK.

- **Recognize recorded ATOM servers during lifecycle teardown and recovery.**
  The serving-process checks now accept `atom.entrypoints`. Recovery records
  the members of a recognized ATOM process group before sending TERM, then
  checks each recorded PID, group and start time before sending KILL. This
  allows anonymous workers to be reaped after their leader exits without
  treating a reused PID or a newly discovered process as an owned worker.
  A rank that forked after the snapshot is in neither the recorded set nor any
  cmdline that still reads as an owner, so a confirmed group also receives a
  closing group KILL; measured on an 8-rank bring-up, those ranks otherwise
  survived holding their cards. That kill reaches members this pass never
  enumerated, so it is not treated as proof the group exited. Ownership must be
  confirmed first: when the recorded leader is absent from the group, no longer
  reads as an ATOM server, or no longer matches the group and start time
  recorded for it, nothing is signalled at all and the pidfile is kept for a
  later pass -- a recorded pgid the kernel has since recycled would otherwise
  take the teardown meant for ours.
  Recovery retains the pidfile while the group is still alive and reports the
  worker PIDs actually signalled. Measured on one MI355X serving
  Qwen3-14B-FP8: against a fully booted server recovery reaped the leader and
  three anonymous workers and the card went from 87% to 0% VRAM; fired mid-boot
  it left no engine process behind. Normal warmup/measure reuse and the existing
  vLLM/SGLang recovery paths are unchanged. This does not recover ownership of
  anonymous workers whose leader had already exited before recovery began;
  the generic subprocess teardown and third-party benchmark scripts are unchanged.

- **A bare `--resume-from` rebuilt the budget and the stop target from the
  flags.** `--max-hours` carried an argparse default, so a resume passing no
  flags at all was indistinguishable from one passing the default: a 24 h
  session was shortened to 2 h and closed as `time_exhausted` before its first
  action, and the objective was dropped on the way. This is the path
  `robustness_monitor.sh` takes, which auto-resumes with no flags. The flag now
  defaults to `None` and resolves to `DEFAULT_MAX_HOURS` only after the archive
  has had its chance at the persisted budget, so an absent flag restores 8 h
  while an explicit `--max-hours 2` wins over it.

- **Argv preflight read every dotted vLLM flag as unrecognised.** It probed
  through `parse_known_args`, which is not what vLLM's parser uses to expand
  `--<group>-config.<field>`, so preflight spent its one repair dropping flags
  the server would have accepted. One of them bounded the profiler, and the
  roofline that followed recorded 25.7 GB of trace over the whole workload
  instead of over a steady-state window. The probe goes through the entry point
  that performs the expansion.

- **The robustness monitor called a session over while it was still running.**
  It read the presence of `reports/final.*` as terminal, but the crash path
  writes one as a safety net and a resume clears `stop_reason` without removing
  it. `state.json` decides now, and the artifacts stand in only when there is
  no state to read.

- **The IR-1 stale-process scan failed on an idle machine, and could not see an
  ATOM server.** It excluded only `os.getpid()`, so the launcher shell — whose
  argv quotes the whole command — matched the scan's own patterns and failed the
  gate; it now excludes its ancestry. Separately, the pattern list named only
  vLLM and SGLang, so a leftover ATOM server still holding every rank's VRAM
  read as a clean machine. Matched on `atom.entrypoints` alone: the per-rank
  workers are `multiprocessing.spawn` children carrying no identifying argv, so
  only descent from the wrapper reaches them, which teardown already covers.

### Removed

- **The orchestrator drops five mechanisms nothing read: the `kernel_agent`
  inbox subscription with `IntentType.RESPONSE`, `Message.priority`,
  `IntentSpec.builder`, the `PHASE_EXIT_REASONS` vocabulary, and 11 stop
  reasons no code path can produce.** Each had writers, or a schema column, or
  a test suite holding it up — everything except a production reader.

  `IntentType.RESPONSE` was dispatched, policy-gated and validated end to end
  while the `kernel_agent` process that would consume it is never instantiated.
  The `request` / `response` topics and `kernel_agent` as a routing target are
  untouched; only the subscription and the intent type go. `Message.priority`
  had 15 writers and no reader, no index and no `ORDER BY`.
  `IntentSpec.builder` kept two unwired builders looking alive — their
  validators stay, because the role may still emit both intent types, just not
  through a builder. `PHASE_EXIT_REASONS` was a closed 37-member vocabulary
  with no production reader at all, unlike its load-bearing twin
  `STOP_REASON_VOCAB`, which gates `machine.py`, `close.py` and
  `set_stop_reason`: a phase exit reason only reaches `phase_history` for a
  human to read, so a typo there cannot misroute anything.

  The 11 stop reasons were not "not yet triggered". The crash-threshold path
  sets `emergency`, and `compute_plateau_explore` / `compute_plateau_kernel`
  return booleans, so no exit rule ever named `plateau_explore` or
  `plateau_kernel`. The comment calling these legacy sentinels kept for
  resuming old sessions did not survive checking, and goes with them:
  `SharedState.from_dict` restores `stop_reason` as a dataclass field,
  bypassing `set_stop_reason`, and `_global_terminal` returns unrecognised
  values verbatim under `vocab: "unknown"` — vocabulary membership was never
  what let an old session report its terminal. `enablement_stalled` remains a
  legal value in the enablement timeline event namespace, which is a different
  vocabulary and is untouched.

  **Resuming across the change needs nothing from the operator.**
  `CREATE TABLE IF NOT EXISTS` leaves `priority INTEGER NOT NULL` on a
  `coordinator.db` written before this release, where a column with no default
  would refuse every append the new code writes, so `ensure_schema` drops it on
  the way in. An in-flight session resumes with its event history intact.

  The rewrite kernel lane is a **refactor, not a removal**. `allocate()` built
  a `LaneAllocation` for a lane whose budget the rewrite route never read — it
  sizes itself against the wall clock — but the `0.5` was load-bearing as a
  divisor, holding the other two lanes down to 0.3 and 0.2 of the phase. It
  becomes `REWRITE_RESERVE_SHARE`, taken off the top before the two real lanes
  divide the rest, and each lane's share of the phase is unchanged.

- **`--max-minutes-enablement-pct` / `--phase-budget-enablement-pct`.** The
  flag parsed and reached `DEFAULT_PHASE_BUDGET_PCT`, but no ENABLEMENT branch
  in `compute_next_phase` ever calls `phase_cap_exceeded`, so the cap it
  advertised was never enforced against anything. Wiring it up would have
  contradicted the phase having no budget by design. It was new in this
  release and nothing depended on it.

## Hyperloom 1.1.1 release

The [1.1.1 release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.1.1)
is a patch release on top of 1.1.0. The session record does not move, so a
session recorded by 1.1.0 resumes on this build. The command line and the
environment contract do: one optimizer option, one console script and eight
environment variables that 1.1.0 accepted are gone, one variable's accepted
values narrow, and one option is added. See "Before upgrading from 1.1.0"
below.

Most of it is the Recipe knowledge base telling the truth: the prior work a
session had actually earned was not reaching the model at all. The other
corrections are in agent-backend selection, which three places answered
differently, and in the supervisor watchdog, whose restarts are now resumable
and bounded.

### Before upgrading from 1.1.0

One optimizer option 1.1.0 accepted is removed. The parser is strict, so it is
not an ignored token: a launch or resume command that still carries it exits
with `unrecognized arguments` before the session starts. Check operator scripts
before upgrading.

| Removed option | What to do |
|---|---|
| `--breakdown-include-transcripts` | Delete it. The Session Breakdown section it inlined into is gone, so there is nothing left to inline; specialist transcripts are still written to disk and carried as `transcript_path`. |

The deprecated KernelForge console-script alias is also gone, so a script
invoking it fails with `command not found` rather than a parser error. Call
`kernelforge`, which has been the name since v1.0.0b2; the retired spelling is
`kernel-agents`.

Eight environment variables 1.1.0 read are also gone, and these fail differently
from the options above: nothing refuses them. A box that still exports them
starts normally and behaves as though they were never set, so they have to be
found by reading launch scripts and `.env` rather than by watching a run fail.

| Removed variable | What to do |
|---|---|
| `FORGE_CLAUDE_MODEL`, `FORGE_CODEX_MODEL`, `FORGE_AGENT_MODEL` | Set `CLAUDE_MODEL` / `CODEX_MODEL` instead. Forge now walks Hyperloom's ladder and nothing above it, so one spelling configures both. A box left on the old names does not fail; it falls through to the provider default. |
| `HYPERLOOM_SKIP_COLLECTIVE`, `HYPERLOOM_COLLECTIVE_ONLY`, `HYPERLOOM_COLLECTIVE_KEEP_PCT`, `FORGE_COLLECTIVE_TIMEOUT`, `FORGE_COLLECTIVE_AGENT_TIMEOUT` | Delete them. They steered the collective optimization lane, which is now part of the rewrite controller; communication operators are picked up as ordinary rewrite candidates and need no separate switches. |

One more variable survives with a narrower accepted set rather than being
removed. `HYPERLOOM_REASONING_EFFORT` no longer takes `minimal` or `none`; the
ladder is `low | medium | high | xhigh | max`. The two sides that read it
disagree about a value outside that ladder, which is why it needs naming here:
Forge refuses at startup (`'minimal' is not a reasoning effort`), while
Hyperloom's own `chat.completions` drops the field and takes the gateway
default, deeper and more expensive than `minimal` was. A deployment sitting on
either value has to move to `low` by hand, and only one half of the run will
say so.

### 1.1.1 highlights

- **The warm-start block, the KB's lessons, and its pitfalls reach the model
  again.** All three read field shapes no writer produces. An exact hit carrying
  a full config printed `(no recipe text — first session for this workload/hw)`,
  and sections 5b and 5c rendered `(none)` no matter how much a prior session had
  learned. The block now renders `warm_start_context`, the view `recipe_kb_t0`
  already persists on every anchor, and a borrowed config is labelled with the
  model it came from. No recorded knowledge was lost; until now none of it was
  being shown.

- **One rule picks the agent backend across both packages: a configured
  credential first, then an installed SDK, with Claude ahead of Codex.** Four
  places answered this and three disagreed, so `forge-loop` on an OpenAI-only
  host could resolve to Claude and then fail to authenticate. The Robustness
  Agent's RCA engine follows the same precedence instead of checking the OpenAI
  side unconditionally. One consequence worth naming: a deployment whose runtime
  is logged in by other means is no longer refused up front — `forge-fuse`
  dropped its `--agent-backend auto` usage error and forge-fusion dropped the
  `llm_provider_unconfigured` result, leaving the real authentication failure to
  the preflight that can see it.

- **Supervisor watchdog restarts are resumable and bounded.** A wedged
  coordinator takes SIGHUP and keeps its interrupted phase segment rather than
  recording a session outcome; three restart attempts, counted durably before
  the signal goes out, make the wedge terminal. Separately, the robustness
  monitor now reads the real stop-reason vocabulary — it had been importing a
  module that does not exist and silently falling back to a subset missing 16
  terminal reasons, so a finished session could be relaunched.

- **`--extend-hours` grants a resumed session more budget.** Elapsed time is
  summed forward across every leg and never reset, so this is the only way to
  continue a run that has already spent its budget. The grant is applied on
  `--resume-from` and recorded in the session state with its reason; it defaults
  to `0.0`, so an invocation that does not pass it behaves as before.

- **The bare-metal `vllm` default is `0.29.0+rocm723`, up from `0.27.1`.**
  `install_baremetal.sh`, the compatibility matrix, the install guide, the
  example `SKILL.md` recipes, and `assets/slurm/models.tsv` all name it, and the
  pinned TraceLens ref ships the matching profiler-config patch.
  `VLLM_VERSION` and `VLLM_ROCM_VARIANT` override it as before.

- **Dead surfaces are removed**: the pre-rename KernelForge
  console script kept as an alias since v1.0.0b2, superseded by `kernelforge`
  (the retired spelling is `kernel-agents`); and its `learning/` tuning
  database with the tracker's superseded scoring layer, about 1.4k lines whose
  writes had already been disabled. The one observable difference: a
  `forge-loop` run no longer writes lesson markdown under the writable
  knowledge base's `learned/` directory and no longer prints
  `Lessons learned: N`. Nothing read that directory.

## Hyperloom 1.1.0 release

The [1.1.0 release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.1.0)
is the first feature release after the 1.0.0 stable release. Kernel
optimization now ships in the box: KernelForge is vendored into the Hyperloom
wheel as the built-in kernel-opt agent, so there is no separate checkout to
clone or install. The optimization loop itself is shorter — configuration
search and source landing become two arms of one phase, and the redundant
confirmation benchmark after every KEEP is gone.

This release also adds first-class support for agentic-replay workloads: an
AgentX run is graded on total token throughput under an interactivity veto,
which is the axis a submission is ranked on, and the sweep produces the chart
that goes with it. Compute-partition shape (`SPX`/`DPX`/`QPX`/`CPX`) is now
recorded and checked on every session, with a separate operator script to
measure which shape a workload wants.

1.1.0 carries a number of breaking changes to the CLI, the environment
variables, and the session record. The ones that need a plan before upgrading
are marked below; the full per-change list is in the
[1.1.0 release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.1.0).

### 1.1.0 highlights

- **The EXPLORE phase is merged into FRAMEWORK_AGENT** *(breaking change — CLI, exit reasons, in-flight sessions)*:
  the chain is now `PRELUDE → FRAMEWORK_AGENT → KERNEL_AGENT → SWEEP → CLOSE`.
  Configuration search and source/upstream landing are two arms of one phase,
  worked in parallel; the phase advances only when both are dry, and one arm
  plateauing raises `switch_bottleneck` for the next macro-cycle instead of
  ending the phase while the other lever still pays.

  **Before upgrading**: `--no-explore` is removed rather than aliased, because
  the two arms cannot be disabled separately and the flag's new meaning would be
  wider than what an operator script asked for — use `--no-framework-agent`.
  `--max-minutes-explore-pct` and `--phase-budget-explore-pct` are aliases for
  the framework budget option (merged phase default share `0.40`, against `0.50`
  for KERNEL_AGENT). The `explore_*` and `framework_agent_*` exit reasons are
  replaced by `optimize_no_more_leverage`, `optimize_phase_budget_exhausted` and
  `optimize_budget_cap`. **A session recorded at `EXPLORE` cannot be resumed by
  this build** — the Coordinator refuses at startup rather than re-running
  PRELUDE on top of an existing baseline and KEPT stack. Archived sessions still
  read; they just cannot be continued.

  Two consequences worth naming: the `framework_agent` action is retired, so
  upstream PRs land through `integrate_patch` with `patch_source='upstream_pr'`
  and their workspaces move from `runs/framework_agent/<task_id>/` to
  `runs/integrate_patch/<task_id>/`; and `pr_intel_specialist` is replaced by
  `candidate_discovery_specialist`, which owns finding, ranking and judging
  upstream candidates. Because one phase now carries both levers, gain is
  attributed by `lever_kind` (`config`, `source_patch`, `upstream_pr`,
  `enablement`, `kernel`) rather than by phase, and
  `attribution.lever_breakdown` splits validated gain by it.

- **KernelForge ships inside Hyperloom as the built-in kernel-opt agent**:
  its source is vendored into `src/kernelforge/`, and Hyperloom is the sole
  source from here on. Installing Hyperloom installs forge — there is no private
  repository to clone and no separate distribution to `pip install`. Its
  knowledge base, examples and serving patches ship in the wheel, so they
  resolve from an installed distribution rather than from a checkout. The
  orchestrator's kernel-agent dispatch path is unchanged, `KERNEL_OPT_BACKEND_ORDER`
  still selects between the forge and geak backends, and eight kernel backends
  remain (CK, FlyDSL, Triton, Gluon, AITER, HIP, hipBLASLt, and fusion). The
  `intellikit` backend is removed: nothing in Hyperloom could reach it.

- **The forge CLI stops absorbing options it does not declare** *(breaking change — remove these from your scripts and environment)*:
  `forge-loop` and `forge-rewrite-by-flydsl` were the last tolerant entry points,
  discarding an undeclared option with a warning and proceeding on the defaults.
  That exemption existed for a consumer in a separate repository; vendoring put
  producer and consumer in one wheel, so what the tolerance still absorbed was
  typos and renames — seven shipped examples kept passing a flag that had been
  renamed out from under them and ran an inferred backend while exiting 0. Both
  commands now fail with click's own error and exit 2 before any GPU work starts.

  Alongside it: `$FORGE_PATH` is removed and **not** honoured as an override
  (use `$KERNELFORGE_PROJECT_ROOT`, which defaults to
  `$USER_DATA_PATH/kernelforge`, else `~/.cache/hyperloom/kernelforge`);
  `forge-gemm-tune` is gone as a console script and as a distribution, invoked
  now as `kernelforge gemm-tune`; the kernel-backend vocabulary is normalized, so
  the CLI flag is `--kernel-backend` taking a bare name (`triton`), the
  campaign-config key is `kernel_backend`, and a config carrying the retired key
  fails at load rather than migrating silently; and `FORGE_MAX_ITERS` /
  `FORGE_COMPILED_MAX_ITERS` are gone, having fed a cap that was never applied.
  Because `FORGE_` stays on the dotenv prefix allowlist, a stale `FORGE_PATH` or
  a retired spelling of `FORGE_DISABLE_COMPILED_KERNEL_BACKENDS` is still
  forwarded into the run and then ignored; the latter is detected and warned
  about once per run, because an operator who had switched compiled kernel
  backends off would otherwise silently get them back. The retired spelling a
  migrating script will be carrying is `FORGE_DISABLE_COMPILED_FELLOWS`.

- **An AgentX run is graded on total token throughput under an interactivity
  constraint**: the corpus an agentic replay runs averages ~114k prompt tokens
  against ~810 output tokens per request, so grading on output throughput alone
  optimizes about 1% of the token budget — a measured baseline read 25978 tok/s
  total against 183 tok/s output. Total token throughput is now the objective and
  interactivity p90 (`OSL/E2EL`) is a veto rather than a weighted term. It is
  default-on under `HYPERLOOM_AGENTX=1`; `HYPERLOOM_PERF_METRIC` overrides in
  both directions and `HYPERLOOM_PERF_NOISE_PCT` (default `5.0`) sets the veto
  band. Scriptable frameworks keep output-throughput grading, candidate and
  reference are always read off the same axis, and the final report names the
  grading mode. A measurement the scenario judged invalid is no longer
  selectable anywhere in the run, so an unverified number cannot become the
  denominator of every gain that follows it.

- **AgentX installs and repairs its own benchmark client**: the pinned aiperf
  install was gated on a runtime mode flag being true in the installer's
  process, so provisioning without `HYPERLOOM_AGENTX`/`INSTALL_AIPERF` and
  turning AgentX on later left a box that could not run it — 11 of 13
  provisioning runs on one cluster logged the skip. Install time now keys on
  whether the build ships `assets/agentx/`, and run time repairs what is still
  missing, once per process. A client that is genuinely absent stops the run on
  the first occurrence with the new `agentx_client_unavailable` stop reason
  instead of opening an enablement round — routed as an ordinary launch failure
  it cost a full 24h budget, because a specialist cannot tell a supply gap from a
  framework bug. `install.sh` gains `--only-aiperf` to add AgentX support to a
  box provisioned without it.

- **SWEEP is one concurrency sweep, and it produces the chart a submission is
  read on** *(breaking change — resumed sessions carrying the old exit reasons)*:
  the workload sweep over `(CONC, ISL, OSL)` is deleted. Two of its three axes
  carried nothing under an agentic replay — request shapes come from the trace
  corpus — and the concurrency axis is what `conc_sweep` already swept.
  `conc_sweep` is now the only sweep, on by default for both workloads, and every
  rung carries `intvty_p90`, `input_throughput` and `tpot_p90_ms` alongside the
  output-axis figures. The default ladder is per workload
  (`256,128,64,32,16,8,4,2` synthetic, `1,4,8,10,14,20,28` under
  `HYPERLOOM_AGENTX`); `--conc-sweep-concs` still overrides both. The `sweep`
  action is gone from the LLM catalogue, the executor registry and the phase
  contract, and `conc_sweep_done` / `conc_sweep_failed` collapse into
  `sweep_done` / `sweep_failed` with no alias for the old spelling.

- **The post-KEEP confirmation round is removed** *(breaking change — session record)*:
  an `explore` variant and an `integrate_patch` candidate were each re-benched
  once more after they had already been graded, and the second measurement
  overwrote the first as the reported number. For `explore` that round ran third
  on an already-warmed server, so it carried more cache than the round it
  overwrote and the inflated value became the anchor the next variant was graded
  against; removing it takes the bias out of the reported gain and saves a full
  benchmark per KEEP. Both now report the round that graded them. Removed from
  the session record: the `KEEP_UNSTABLE` outcome, the `keep_unstable_in_stack`
  result key, and the `stack_rebench_tput` / `stack_rebench_workspace` /
  `stack_rebench_warnings` fields; `enable_stack_rebench` and
  `rebench_stable_threshold_pct` are no longer read from task params. Expect more
  `fallback` and `no_promote` verdicts from GEAK's same-harness revalidation,
  which now measures cold like every other explore.

- **The card's compute-partition shape is recorded, checked, and published**:
  an MI300-series card split into `SPX`/`DPX`/`QPX`/`CPX` trades per-request
  latency for aggregate throughput, and until now nothing recorded which shape a
  number came from — two runs of the same configuration in `SPX` and in `CPX`
  were indistinguishable in the history. The observed mode now goes into the
  platform fingerprint alongside NPS and the session report names it on
  partitioned runs. `--compute-partition-mode` *asserts* the mode the card is
  already in and refuses the session if it is in another one or cannot be read;
  `--streams-per-partition` (default `2`) sizes the concurrent streams per
  partition. **The optimizer never sets the mode** — that is privileged and
  evicts every GPU context, so the card must be in its mode before `optimize`
  starts. The new `python3 scripts/partition_mode_sweep.py` is where the
  privileged set lives: it sets each mode in turn on one card, runs the same
  benchmark on every partition that mode creates with all of them loaded
  together, sums the throughput, and restores the entry mode on the way out,
  including after a failure or a Ctrl-C.

  **Operator note**: launch now refuses a session whose streams provably will not
  fit one partition, sized from the checkpoint's weight bytes as a lower bound.
  The arithmetic costs milliseconds and the failure it replaces is an
  out-of-memory crash hours in. When the checkpoint cannot be sized the session
  runs and says so.

- **The `deterministic` trace-analysis route is gone** *(breaking change — remove it from your configuration)*:
  `HYPERLOOM_TRACE_ANALYSIS_ROUTE` and the `analysis_route` payload key take
  `agent` or `bypass`, and `tracelens_analysis.py` no longer accepts
  `--analysis-route`. The route maintained a second candidate-extraction pipeline
  beside the one `analysis.md` already defines; `bypass` serves the same no-LLM
  intent by reading the profiler trace directly and needs no TraceLens checkout.
  A request still naming `deterministic` is rejected with `invalid_analysis_route`
  before TraceLens or an LLM is started. Only an omitted route defaults to
  `agent`; an explicit unknown value no longer falls back to a route that may
  spend an LLM session. Relatedly, the two tool-free LLM source tiers and
  `HYPERLOOM_LLM_SOURCE_PROVIDER` / `HYPERLOOM_LLM_SOURCE_PREVIEW` are removed in
  favour of one tool-enabled review session — their failure mode was not coming
  up empty but coming up confidently wrong, which ranking paths by keyword cannot
  tell apart. `HYPERLOOM_LLM_SOURCE_MODEL` still selects the model.

- **A published Recipe carries three columns instead of five**:
  `config`/`explore`/`framework`/`kernel`/`patch_timeline` collapse to
  `config`/`patch`/`kernel`, each owned end to end by one SDK facade. The
  `explore` and `framework` source overlays merge into a single `patch` column,
  and replay order is the lexicographic order of the zero-padded stack/member
  indices in each overlay ref, so `patch_timeline` is gone. Warm replay is keyed
  to the recorded apply root: a record that cannot name the checkout its gain was
  measured on is skipped whole rather than applied to a different tree.

- **Fixes an operator will notice**: the server-boot timeout default is 7200s,
  up from 2700s, because a 1.56 TB MXFP4 MoE checkpoint reads for ~37 minutes
  before the first JIT and the baseline died to a timeout unrelated to the
  workload. `--extra-env` now reaches the benchmark for every framework, not only
  the `custom` path, and outranks the config — a change in precedence for a
  `custom` workload whose YAML sets the same key. Untrusted diffs are vetted
  before `git apply`, which previously skipped every patch supplied directly,
  including every upstream PR diff. The framework accuracy gate never passed:
  `_bench_candidate` read a field that does not exist on `VariantResult`, so a
  baseline accuracy blocked every KEEP with `accuracy_unavailable_reject`. The
  upstream-PR arm was gated shut at dispatch. Test trees no longer ship in the
  wheel (627 test entries), and rocprof-compute's Python dependencies — which
  `install.sh` claimed arrived with the KernelForge root install and never did —
  now ship as the `forge-profiling` extra.

## Hyperloom 1.0.0 release

The [1.0.0 release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0)
marks the first stable release of Hyperloom, following the tech preview
release in July 2026. This release supports full end-to-end inference workload
optimization on AMD Instinct GPUs (MI300X, MI325X, and MI355X), the vLLM and
SGLang inference frameworks, the HIP, Triton, and FlyDSL kernel languages,
and extends support to ROCm 10.0.

Over the last several releases, we have enabled numerous features, including
Knowledge Base, Search Orchestration and Self-Evolution, Dynamic Agents, Gap
Analysis, Roofline Support, Automated CI/CD, Release Engineering, and many
others. Additional details on these feature enhancements are provided in
the [previous release notes](https://github.com/AMD-AGI/Hyperloom/releases).

This release contains various fixes highlighted below.

### 1.0.0 highlights

- **Configuration fingerprints are flag-aware** *(breaking change — existing sessions)*:
  `canonical_fingerprint` sorted arg tokens as a flat list, so
  `--max-num-seqs 128 --max-model-len 4096` and
  `--max-num-seqs 4096 --max-model-len 128` hashed identically and the second
  was skipped as a duplicate by the `explore_search` dedup ledger. Args are now
  parsed into sorted `(flag, value)` pairs with last-wins semantics for repeated
  flags.

  **If you resume an existing session after upgrading**: all fingerprints stored
  in `explore_search.tested`, `accepted`, `rejected`, and `name_index` inside
  `state.json` are invalidated. Hyperloom treats every previously-tested
  configuration as unseen and re-benchmarks from scratch, repeating work already
  done before the upgrade.

- **Kernel fusion runs again**: The fusion wrapper passed `--llm-model` to
  `forge-fuse` after KernelForge renamed the option to `--model`. Because
  `forge-fuse` rejects an unknown option outright rather than ignoring it, every
  fusion run was exiting 2 before it started and surfacing only as a missing
  `fusion_manifest.json`. The `llm_model` key in the wrapper's own input JSON is
  unchanged.

- **A hard-killed kernel campaign keeps its result**: `best_result.json` was
  gated on `schema_version == 1` while KernelForge has stamped `2` into that
  file since 2026-08-13, so every published best was rejected and the kernel
  backend fell through to the caller checkpoint or the stdout sentinel — losing
  the one record that exists specifically to survive a hard kill. The
  version gate is removed rather than corrected: the commit, the timings, and
  the score are each still checked on their own.

- **A reproduced warm replay is recorded as an adopted optimization**: The
  replay was mirrored into the canonical recorder streams before the keep
  decision was reached, and because a replay's executor settles on `succeeded`
  either way, every replay was recorded as `discarded`. A session that had
  measurably gained therefore came back with an empty `optimizations.entries`
  and the whole gain surfaced as a `reconciliation_gap_pct`. The ledger and
  `cumulative_gain_validated` are now a single number, and drift or failed
  replays carry the measured gain, the threshold, and the reason on their
  attempt row. Sessions completed before this fix are not retroactively updated —
  existing breakdowns that show a `reconciliation_gap_pct` remain as-is.

- **Agent-proposed environment overrides are filtered where they enter the loop**:
  The `extra_envs` argument to `materialize_config_with_envs` was
  checked only for key shape, which let `LD_PRELOAD`, `PYTHONPATH` and `PATH`
  through into the rendered YAML and from there into the benchmark subprocess. A
  specialist's `config_changes` / `extra_envs` proposal is now filtered once at
  assembly in `integrate_patch`, so the benchmarked configuration and the
  recorded one can no longer differ; dropped keys are logged and reported as
  `dropped_env_overrides`. Multi-node SSH forwarding drops its own shorter
  denylist for the shared definitions, which additionally cover `CDPATH`,
  `GIT_SSH_COMMAND`, `NODE_OPTIONS`, `PERL5OPT`, `PYTHONSTARTUP`,
  `PYTHONINSPECT`, `PYTHONUSERBASE` and `SHELLOPTS`.

- **`FORGE_MAX_ITERS` and `FORGE_COMPILED_MAX_ITERS` are gone** *(breaking change — remove these variables from your environment)*:
  This also removes the `--max-iters` flag that was added to every `forge-loop` and
  `forge-rewrite-by-flydsl` invocation. KernelForge deleted the option because its
  campaigns are bounded by `--max-hours`, so the cap those variables fed was a
  no-op that logged a limit it never applied. `--max-hours` and the hard-kill
  timeout are the only budget controls.

- **A hung `ray stop` no longer blocks recovery**: `force_restart_local_cluster`
  inlined its own `ray stop --force` with neither a timeout nor an `OSError`
  guard. It now routes through `_stop_ray_force`, so
  `DEFAULT_RAY_STOP_TIMEOUT_SEC` (30 s, overridable through
  `HYPERLOOM_RAY_STOP_TIMEOUT_SEC`) covers all three stop sites instead of only
  one. Log output is unchanged.

- **Two no-op internals are removed** *(breaking change — affects external importers of `stop_ray_if_owned`)*:
  `stop_ray_if_owned` (whose only caller went away with `parallel_e2e_runner.py`)
  and the `reference_envs` filter inside `materialize_config_with_envs` (whose
  only writer already applied a strictly stronger filter). Neither dropped
  anything in production. If your code imports `stop_ray_if_owned` directly,
  remove that import.

### 1.0.0b2 highlights

- **Official upstream vLLM ROCm image**: Every vLLM image reference moves from
  `rocm/hyperloom:vllm-v0.27.1-rocm7.2.3` to `vllm/vllm-openai-rocm:v0.27.1`,
  since AMD deprecated `rocm/vllm` and `rocm/vllm-dev`. The tag is a 1:1
  replacement, but its entrypoint is `vllm serve`, so override it (for example
  `--entrypoint tail`) when starting a long-running Hyperloom container. SGLang
  images are unchanged.

- **Magpie benchmark upgraded to v0.2.0**: The default benchmark dependency
  moves from v0.1.0 to v0.2.0. Both the installer and the runtime preflight stay
  pinned to the immutable v0.2.0 release commit, so installs remain
  reproducible.

- **Remote Recipe KB reads and writes follow a single unified contract**: Remote
  mode now reads a single Recipe per session (selected by canonical ID), replays
  its combined config and patch timeline, and writes one final record at CLOSE.
  Previously, a missing or imprecise configuration donor could block the entire
  CLOSE write; it is now skipped individually so the remaining sections still
  publish. Local Recipe storage and non-Recipe GBrain integrations are unchanged.
  No user action is required unless you operate a custom remote KB integration.

- **`KERNEL_OPT_BACKEND_ORDER` is the single kernel-backend switch** *(breaking change — remove `KERNEL_OPT_BACKENDS` from your environment)*:
  The GEAK gate no longer falls back to the persisted `shared_state.kernel_optimizer`
  field, so the backend choice is now identical on a resume. The
  `KERNEL_OPT_BACKENDS` environment variable is removed; remove it from any
  launcher scripts or `.env` files.

- **Action metadata consolidated into code** *(breaking change — affects tools that read action YAML files directly)*:
  `actions/_meta/*.yaml` and `orchestrator/actions/registry.py` are replaced by
  `ACTION_CATALOGUE` in `inference_optimizer/protocol/action_surfaces.py`. The
  `preferred_backend`, `preferred_model`, `max_turns`, and `params_schema` fields
  are dropped because no runtime code read them; the operational `verdict_class`
  is kept. If your tooling reads the action YAML files directly, migrate to
  `ACTION_CATALOGUE`.

- **Three unexecutable kernel actions removed** *(breaking change — sessions containing these actions cannot be resumed)*:
  `vendor_kernel_config`, `operator_tuning`, and `deep_kernel_analysis` never had
  an executor, so every request for them was answered with `unknown_kernel_kind`.
  Sessions recorded under the old build that carry these names in `state.json` or
  `coordinator.db` cannot be resumed after upgrading, and no migration is
  provided. Start a fresh session for any affected workloads.

- **Write-only artifacts no longer produced** *(breaking change — pipelines that consume these files will stop receiving them)*:
  `agent_transcript.jsonl`, `orchestration_turns.jsonl`, `mn_input_params_*.json`,
  and the work_dir copy of `semantic_audit.json` are no longer written. The first
  three persisted secrets or raw LLM transcripts past a redactor that inspected
  values but not keys. If any downstream pipeline reads these files, remove that
  dependency.

- **Magpie leak salvage is now opt-in** *(breaking change — scripts that hardcode `/workspace/` as the result directory will now fail)*:
  Salvage no longer defaults to `/workspace/` and runs only when
  `$INFERENCE_OPTIMIZER_RESCUE_PATHS` is set. The generic
  `{framework}_{gpu_type}.sh` scripts respect `$RESULT_DIR` and are unaffected,
  but a script pinned through `params.benchmark_script` that hardcodes
  `/workspace/` will now fail the task with `no_report`. Fix: set
  `INFERENCE_OPTIMIZER_RESCUE_PATHS=/workspace/` in your launcher, or update the
  script to write to `$RESULT_DIR`.

- **`kernel_optimization.py` drops `--test-command` and `--test-harness-path`** *(breaking change — callers that pass these flags will now fail at startup)*:
  The unittest-harness contract they fed had no reachable caller. Remove both
  flags from any script or tool that invokes `kernel_optimization.py` directly;
  argparse will exit with an error if either is still present.

### 1.0.0b1 highlights

- **Remote Recipe KB Store cutover**: Remote Recipe reads and CLOSE writes use
  the KB Store Recipe View with verified artifacts and combined config,
  ordered Explore/Framework overlay, and Kernel replay. Local Recipe storage
  and non-Recipe GBrain consumers are unchanged.

- **`--no-eval` session-wide accuracy opt-out**: The accuracy eval can be turned
  off for a whole run as an explicit choice, anchoring the baseline on throughput
  instead of halting on the missing reference. It persists across `--resume` and
  is refused once the session has anchored an accuracy. Runs made with the flag
  are not accuracy-validated.

- **Claude subscription OAuth support**: `CLAUDE_CODE_OAUTH_TOKEN` is now a
  first-class Anthropic-side credential, so Claude Max/Pro subscribers can run
  Hyperloom through the `claude` CLI without buying separate API credits. Install,
  preflight, specialist subprocesses, and Ray-backed kernel work preserve the
  token without mirroring it into API-key slots.

- **Enterprise LLM gateway setup and headers**: The install docs now show how to
  configure Anthropic-compatible enterprise gateways and custom auth headers,
  including AMD APIM's `Ocp-Apim-Subscription-Key`. `.env` loading, setup
  persistence, Ray runtime environments, and specialist secret forwarding preserve
  `ANTHROPIC_CUSTOM_HEADERS` / `OPENAI_CUSTOM_HEADERS`, so header-authenticated
  gateways work from a fresh shell.

### 1.0.0a3 highlights

- **Recipe-KB write traceability**: Writes to the cross-session recipe KB are now
  mirrored as Langfuse spans. KEEP/REVERT decisions, framework-PR results, and
  CLOSE writes are now auditable in Langfuse without having to diff local history.

- **Remote Cortex KB removed**: The obsolete remote Cortex KB integration is
  removed end to end, including CLI wiring, critic assessment calls, prompt
  injection, bundle fields, env vars, and the specialist Cortex KB MCP server.
  If you were using `CORTEX_KB_*` variables or the Cortex MCP server, remove them —
  they are no longer read.

- **Recipe-KB naming realignment**: Internal recipe knowledge-base paths are
  renamed to use `recipe_*` prefixes consistently across Python APIs, CLI flags,
  emitted state, breakdown data, stop reasons, warm-recipe source tags, sweep grid
  sources, and session runtime directories. If your tooling or scripts reference
  the old KB path names, update them to the `recipe_*` equivalents.

### 1.0.0a2 highlights

- **Long-horizon Forge kernel optimization**: The KernelForge long-horizon CLI is
  integrated end-to-end into the kernel-optimization path, with
  `forge_experiments/best_result.json` promoted to the top-priority keep/revert
  authority — rewritten atomically on every KEEP, correctness-gated, and naming an
  already-committed workspace, so tuned results survive soft-budget exhaustion and
  hard kills. Forge hardens deadline recovery, artifact export, and KB identity,
  decouples the Fusion stage from GEMM tuning, and reaps timed-out process groups.

- **Trace-driven GEMM shape capture and block-FP8 tuning**: Real vLLM GEMM shapes
  are captured before Forge tuning (explicit failures instead of silent skips),
  routed to the vLLM-AITER blockscale tuner and preferred over stale specialist
  CSVs. Block-FP8 tuning reuses steady-state Roofline/TraceLens shapes only when
  provenance and normalized runtime match, falling back cleanly otherwise.

- **Enablement subsystem for non-runnable combos**: A new path lets a non-runnable
  (model, backend) combination repair itself and earn KEEP by actually launching
  the model, using attempt-scoped runtimes in isolated venvs, source localization of
  merged-PR/vendored closures behind a compiled-closure gate, and off-loop compiled
  builds (AITER, sgl-kernel, vLLM-from-source) on a dedicated `build_lane`.

- **Long-horizon orchestration, budgets, and resume fidelity**: Every macro-cycle
  gets a fresh directive with cycle-scoped plateau/transient counters; SWEEP and
  EXPLORE stop testing grid variants once the wall-clock budget is exhausted;
  FRAMEWORK outcomes reconcile across resumes without fabricating deliveries; and
  `current_best` / `current_setting.sh` reproduce the complete accepted recipe.

- **Ray execution and multi-node safety**: Single-node execution again defaults to
  the Ray backend when `INFERENCE_OPTIMIZER_RAY_EXEC` is unset (multi-node stays
  gated off; pytest keeps the local subprocess path). create-rayjob idempotency
  scans all canonical state-file locations and reuses the first live `rayjob_id`,
  preventing duplicate RayJobs that waste a node set and deadlock scheduling.

- **Serving and inference correctness**: A quant-aware gate reads the checkpoint's
  `config.json` so Quark MXFP4 / W4A4 MoE models stay on sglang's aiter path
  instead of being forced onto `--moe-runner-backend triton`. Magpie client-trust
  patching extends to MI355X/MI300X local-path SGLang clients, the InferenceX pin
  is refreshed, and an author-time v4 breakdown model adds richer tracing.

- **Evaluation and install integrity**: Accuracy eval survives the refactored
  InferenceX `run_lm_eval` arg parser — the patcher recognizes the merged-case
  shape and install-time judgment is aligned with the runtime entry point using a
  shared concurrency-unblocked helper, ending false-positive install aborts
  (exit 5). Persistent baseline servers use per-session unique ports, and the
  installer downloads the release wheel and hotfix using public `curl`.

## Hyperloom 1.0.0a1 public release

The first public release of Hyperloom (1.0.0a1) combines features from the following versions:

### 1.0.0a1 highlights

- **Unified macro-cycle orchestration and budget accounting**: Short and long
  sessions now share the same cyclic optimization model. Phase budgets use
  consistent charge-back accounting, short runs stop dispatching after their
  phase budget is exhausted, and new macro-cycles open when sufficient budget
  remains (the effective floor scales with session length so short sessions are
  not unconditionally blocked by the 3-hour absolute floor). This removes legacy
  cyclic-mode branches and makes phase progression more predictable across
  bounded and long-horizon runs.

- **Ray-managed single-node serving and GPU execution**: The Ray path now places
  all serving and GPU-specialist operations under the whole-machine `serving_slot`
  mutex, including framework-agent benchmarks, `integrate_patch`, and concurrency
  sweeps. GPU specialists can queue without blocking the Coordinator, serving
  receives scheduling priority, and stale AITER JIT locks are cleaned before
  server launch. Single-node runs default to the Ray path; set
  `INFERENCE_OPTIMIZER_RAY_EXEC=0` to force local subprocess execution. Multi-node
  behavior is unchanged.

- **vLLM and serving-environment reliability**: Hyperloom now isolates co-located
  Ray heads, uses per-session free ports for persistent serving processes, and
  reaps orphaned vLLM/SGLang process groups safely. Package-root `MAGPIE_PATH`
  entries are kept out of `PYTHONPATH`, preventing main-environment Torch packages
  from shadowing isolated vLLM environments. TraceLens patching and dependency
  discovery also support isolated vLLM/AITER installations more reliably.
  Block-FP8 Forge tuning reuses a successful workload- and backend-matched
  TraceLens-selected steady-state Roofline trace instead of launching a duplicate profile.
  Sessions created before this metadata was introduced safely perform one
  standard Roofline shape capture because their existing traces cannot be
  verified.

- **GEAK-first kernel optimization**: GEAK is now the default owner of the complete
  KERNEL phase. Forge remains an explicit opt-in and runs only when
  `KERNEL_OPT_BACKEND_ORDER=forge` is set. GEAK candidates remain provisional until
  Hyperloom revalidates them with its benchmark harness; successful revalidation can
  now complete before SWEEP begins, while failed or inconclusive validation exits
  cleanly without blocking the session.

- **Baseline, evaluation, and reporting integrity**: Relative evaluation-result paths
  are resolved against the benchmark output directory, preventing false baseline-accuracy
  failures. Persistent servers use unique ports across retries, framework-phase gains
  are attributed correctly in session breakdowns, and concurrency-sweep internal tasks
  are no longer rejected by their own singleton policy.

- **Security and policy hardening**: Multi-node restart arguments are validated and
  shell-quoted before execution. Explicit framework source roots must remain inside the
  configured allowlist, and queued tasks are revalidated by PolicyGate before dispatch
  to prevent forged task rows from bypassing normal authorization and Critic gates.

- **CI and release engineering**: Every PR targeting `main` now runs a single-GPU
  Qwen3 smoke test on the dedicated Hyperloom E2E runner. The sharded Python 3.10/3.11 test
  suite has stronger failure visibility and completeness checks. Release version reporting
  now comes from installed package metadata, keeping
  `hyperloom.inference_optimizer.__version__`, wheel metadata, and the release version
  aligned.

### 0.9.0 highlights

- **Opt-in Ray-managed single-node execution**: When
  `INFERENCE_OPTIMIZER_RAY_EXEC=1` is set, GPU/serving units (baseline,
  profile/roofline, explore, sweep, conc_sweep, and `needs_gpu` specialists)
  run on a Ray-managed GPU lease, with Ray owning GPU queueing, device
  isolation, and a whole-machine serving mutex. The path fails fast on an
  infeasible cluster, times out stuck specialist scheduling, detects dead
  actors, and adds safeguards around Ray actor failure and server lifecycle
  cleanup. When unset, single-node serving uses the local subprocess path.
  Multi-node is unchanged (gated off).

- **Accuracy-gate and eval-result integrity**: `lm-eval` output is wired to a
  session-scoped `EVAL_RESULT_DIR`; a baseline that produces no accuracy
  verdict now hard-stops instead of optimizing an unvalidated baseline; leaked
  eval results are salvaged back from the InferenceX checkout / local-disk
  mirror; and session-breakdown attribution no longer fabricates credit from a
  seeded stack.

- **Provider-direct LLM configuration**: Hyperloom connects directly to Anthropic (provider-only paths), with env-driven gateway auth/endpoint resolution, case-insensitive Anthropic-endpoint handling, and support for running the Critic over the native provider endpoint.

- **Model-path and workload-default consistency**: A single `--model` value (local
  path or HF repo id) resolves identically across baseline, roofline, and the
  kernel agent; prompt-stated ISL/OSL/CONC/TP are honored as flags instead of
  silently defaulting; and the `model_arch` freshness guard is org-aware across
  HF-cache snapshot paths.

- **Kernel / Forge / GEAK**: Forge-fusion is adopted end-to-end with hardened
  subprocess timeouts, GEAK v4 installs via `pip`/one-click, advertised kernel
  backends are aligned with the runtime, and a TraceLens-free bypass benchmark
  harness ships as a Magpie drop-in for text-gen + xDiT.

- **Framework agent**: The FRAMEWORK phase gains cross-framework rating + PR-KB
  discovery, a flag-gated config-exploration subphase, and a candidate-free
  local-exploration arm so a dry PR feed no longer wastes the phase.

### 0.8.0 highlights

- **Kernel-optimization integrity and GEAK faithfulness**: Patch-only kernel
  wins are no longer discarded by the FULL_BENCHMARK verifier (full source is
  reconstructed from the patch); multi-file (L3) kernel optimizations are
  preserved end-to-end via recorded kernel-artifact bundles; GEAK always runs
  against a freshly re-profiled TraceLens snapshot; and the kernel-candidate
  pool cap is decoupled from the dispatch budget.

- **Profiling / roofline / TraceLens**: GPU information is restored in
  profiler traces (torch-trace `"kernel"` category), and a stale `TRACELENS_ROOT`
  inherited from the kernel-agent env file no longer breaks TraceLens discovery.

- **Orchestrator reliability and long-run durability**: `reports/final.json` is
  now written crash-safe even on non-graceful/`time_exhausted` exits, and
  orchestrator LLM calls survive slow heavy-reasoning models (for example, Kimi-K2.6)
  via idle-timeout + amplified retry.

- **Server config and Local-Mode portability**: SGLang `--context-length` is
  clamped to the run's `--max-model-len` (no more contradictory server config),
  and Local Mode portability groundwork removes Core42 / WekaFS hard-coding (docs).

### 0.7.0 highlights

- **Forge: a third autonomous kernel-optimization backend (new track)**:
  0.7's headline is **Forge** (**Kernel-Forge**) — a self-driving kernel-optimization
  backend that joins GEAK and OOB. It runs an autonomous edit→build→bench loop
  with kernel_kind-aware kernel backend routing (Triton / HIP / CK / aiter / hipBLASLt / FlyDSL),
  an aiter compiled-kernel closed loop, honest compile-only skips for
  non-rewritable kernels, and its own session-breakdown lane. Forge already
  produces the majority of detected kernels on MI300X runs.

- **Deterministic GEMM tuning (new track)**: A standalone `forge-gemm-tune`
  backend lands as the default GEMM-tuning prelude: it auto-detects MoE / dense +
  precision / quant, selects the applicable tuners, and each tuner's tuned config
  is **independently E2E-validated and stacked** (per-tuner KEEP / REVERT, like
  kernel-opt) instead of bundled — so one bad tuner can't drag down the set.

- **Knowledge Base: GBrain-backed, 7-tuple canonical identity**: The Recipe KB
  extends its canonical identity from a 5-tuple to a **7-tuple** with config-donor
  warm-replay, and Forge kernel backends now read cross-KB knowledge directly from the
  unified **GBrain** (KernelForge + GEAK + PTAO), with KB-usage provenance surfaced
  in the session breakdown.

- **Long-horizon durability + specialist autonomy**: Long-run optimization is
  hardened end-to-end: crash-window recovery, integrate-fault retries instead of
  first-crash discard, duplicate-optimizer corruption protection, orchestrator / CLI
  decoupling, and **opened-up GPU specialist exploration** with serving-disjoint
  leases — so multi-day runs stay productive and self-correcting.

- **GEAK / kernel-dispatch reliability, end-to-end**: A full RCA sweep on the
  trace→shape→dispatch pipeline: correct GEAK dispatch attribution (no more
  mis-bucketing non-GEAK attempts), faithful harnesses with real per-arg dtypes,
  trace-anchored shape pinning, candidate-artifact retry on shared-storage
  visibility lag, empty-queue clean skip, and forwarding of GEAK scoring / profiler
  knobs across the Ray boundary.

- **Profiling / roofline + TraceLens 0.7**: A deterministic (no-LLM) trace-
  analysis route, TraceLens 0.7.0, profile-scoped OSL control with a
  serialization-safe capture cap, expert-parallel flag handling for MoE roofline,
  and eager-boot fallback for the SGLang profile-cuda-graph path.

- **Reliability: sandbox-hang elimination and real-cluster hardening**: LLM streaming
  reads are now bounded client-side and timed-out subprocess trees are reaped by
  process group, eliminating the "pod Running but idle" sandbox hang. Plus setup_env
  race / USER_DATA_PATH corruption fixes, user-uncommitted-change protection before
  destructive reverts, and invalid / premature zero-gain rejection.

- **CI: structural model pre-filter and throughput**: A shared model-compatibility
  pre-flight (multimodal / Gemma2 / Phi3-longrope / dual-chunk / ModelOpt-FP8 /
  FlashInfer / gated / missing-tokenizer) skips doomed models before a session is
  created, alongside HF-token rotation with 429 backoff, larger daily pools, and 48h
  long-horizon budgets.

### 0.6.0 highlights

- **Search Space: looser orchestration + long-horizon runs**: 0.6 turns the
  Orchestration theme from 0.5's "search efficiency" to **search space**: mechanical
  guardrails are downgraded to advisory so the optimizer drives itself, free-form
  and cross-domain specialist dispatch lets it explore beyond the fixed action
  catalog, and 2–3 day long-horizon optimization with finer-grained start / stop
  / checkpoint / resume keeps long runs productive and recoverable.

- **Quantization agent (new track)**: A prompt-driven Quark quantization sub-agent
  lands as an optimization prelude, with Quark enhancements and quantization-agent
  proposals — quantization joins kernel-opt as a first-class optimization lever.

- **Knowledge Base: unified and knowledge-graph-backed**: Cortex-KB and GBrain
  converge behind a single Recipe KB interface, GBrain is integrated, a Knowledge
  Graph is wired in, and a KB-evaluation harness is added so warm-start priors can
  be measured rather than assumed.

- **GEAK / kernel-optimization reliability, end-to-end**: The trace → kernel-shape
  → GEAK pipeline is hardened so genuinely-good kernels actually reach GEAK and are
  not silently dropped: cache-invalidation by target type (aiter cpp_itfs / Triton / inductor),
  recovery of high-GPU-time kernels missed by analysis.md-only extraction,
  server-patcher idempotency, and parallel GEAK / OOB ladders with per-attempt cache
  isolation. rocprof-compute roofline and kernel-level roofline detail land alongside.

- **TraceLens 0.6, CLI + WebUI, and budget-aware roofline**: TraceLens 0.6 ships with
  open-source MAF backfill (GPU microbenchmark), a TraceLens CLI, WebUI standalone / comparative
  analysis, and roofline that is time-boxed against the total budget.

- **Multi-node and scale**: Multi-node Optimus support comes online and the Arbor
  mechanism migration is completed.

- **Observability overhaul**: Live Langfuse tracing, full-trace token and conversation
  logging, phase / step-level observability, a per-session token-consumption breakdown,
  and an author-time session-breakdown recorder make long runs inspectable in real time.

- **Reliability and real-cluster hardening**: A broad sweep driven by large-scale
  cluster-run analysis: fast-fail for immediate arg / config errors, log line-buffering
  so healthy runs no longer look frozen, pod-local dependency roots decoupled from WekaFS,
  Ray raylet fd-limits, MI308X detection, attention-backend argument hygiene, GEAK container
  network path to the LLM gateway, and clearer setup / baseline failure classes.

- **Docs, licensing, and coverage**: Repo-wide Google-style docstrings with a published Sphinx
  documentation site, the license relicensed **Apache → MIT**, requesting-access / SSO docs, and
  Python test coverage raised to ~91.5%.

### 0.5.0 highlights

- **Orchestration: vocabulary unification + search efficiency**: 0.5 advances the "search
  efficiency" theme ([#272](https://github.com/AMD-AGI/Hyperloom/issues/272)): the optimizer's
  action/state vocabulary converges onto the unified explore grid-runner and the EXPLORE/specialist
  fan-out gains parallel headroom. New **Atom framework support**
  ([#336](https://github.com/AMD-AGI/Hyperloom/issues/336)) and a soft **Dynamic Action** cross-domain
  deep-dive ([#335](https://github.com/AMD-AGI/Hyperloom/issues/335)) widen the search surface.

- **GEAK and kernel optimization, deeper**: **GEAK GEMM tuning**
  ([#331 ](https://github.com/AMD-AGI/Hyperloom/issues/331)) and **FlyDSL kernel-optimization
  integration** ([#211](https://github.com/AMD-AGI/Hyperloom/issues/211)) land, and **kernel-level
  roofline support + quality fixes** ([#330](https://github.com/AMD-AGI/Hyperloom/issues/330))
  sharpen targeting. Input quality to GEAK is tightened across the board: hot-kernel candidates
  are filtered to backend-routable kernels only ([#314](https://github.com/AMD-AGI/Hyperloom/issues/314)),
  the kernel-opt prompt drops the bloated full analysis.md
  ([#307](https://github.com/AMD-AGI/Hyperloom/issues/307)), the GEAK budget no longer forces quick-mode
  timing under GEAK_RUN_MODE=full ([#337](https://github.com/AMD-AGI/Hyperloom/issues/337)), and kernel
  batch parallelism adapts to smaller pods ( [#338](https://github.com/AMD-AGI/Hyperloom/issues/338) ).

- **Knowledge Base productization**: The 0.4 Knowledge Base Service moves toward operations with **KB
  Productization and Data Maintenance** ([#333](https://github.com/AMD-AGI/Hyperloom/issues/333)) and **KB Recipe
  Ingestion** ([#332](https://github.com/AMD-AGI/Hyperloom/issues/332)).

- **Profiling, TraceLens, and Dashboard**: **TraceLens 0.5** ([#358](https://github.com/AMD-AGI/Hyperloom/issues/358));
  a patched profiler docker image that captures HipGraphLaunch kernels so optimization-loop traces are
  complete ([#352](https://github.com/AMD-AGI/Hyperloom/issues/352)); **profiling information for all Hyperloom
  models** ([#346](https://github.com/AMD-AGI/Hyperloom/issues/346)); **kernel roofline on the dashboard**
  ([#345](https://github.com/AMD-AGI/Hyperloom/issues/345)); and a **Session Breakdown enhancement** spanning
  auto-collection, alerting, TraceLens/GEAK detail capture, and kernel roofline
  ([#334](https://github.com/AMD-AGI/Hyperloom/issues/334)).

- **Stability and bug fixes**: 0.5 closes a batch of orchestration / runtime defects surfaced by 0.4 runs:
  local-mode KERNEL phase failing to dispatch GEAK plus TP variants leaking past visible-device scope
  ([#341](https://github.com/AMD-AGI/Hyperloom/issues/341)); integrate_handler early-out on a missing
  base_tput that was already in SharedState ([#319](https://github.com/AMD-AGI/Hyperloom/issues/319));
  per-cluster call_timeout_s for the Claude/Codex backends ([#318](https://github.com/AMD-AGI/Hyperloom/issues/318));
  kernel-agent misreading a mini-swe-agent step-header $X.XX as a budget cap
  ([#317](https://github.com/AMD-AGI/Hyperloom/issues/317)); arbor zombie-process cleanup on failure
  ([#268](https://github.com/AMD-AGI/Hyperloom/issues/268)); and the README cert-install script on
  RHEL/CentOS hosts ([#328](https://github.com/AMD-AGI/Hyperloom/issues/328)).

### 0.4.0 highlights

- **Hyperloom v2 architecture lands**: 0.4 is a substantial v2 leap: end-to-end Model Auto-Optimize,
  Framework Agent integration, Agent Kernel Arena, the Self-Evolving Skills and Memory layer ramping past
  the 0.3 proposal-only stage, and the first-iteration Hyperloom Knowledge Base Service. Multi-Node
  CI/CD comes online as a first-class capability.

- **Robustness Agent overhaul**: A foundation rewrite ships 13 independent signal detectors (preflight,
  kernel-pipeline, gpu-leak, decision-audit, critic-health, budget, state-integrity, repeated-payload,
  aiter-jit, external-deps, progress, event, local-health), an action ladder for graduated response, run
  finalize / postmortem, and a persistent state store. Closes long-standing issues like SQLite corruption
  on multi-day runs, validate_stack retry / TP miscalibration, and arbor orchestrator silent exits.

- **TraceLens ↔ GEAK integration tightened**: Multi-root InferenceX patcher with post-profile trace
  structure validation; SGLang 0.5.11 patch parity (alongside 0.5.9); TraceLens prose + source-function
  aggregation surfaced to GEAK during kernel rewrite; duplicate-markdown and standalone-upload bugs
  resolved. Server patcher robustness landed in two follow-up rounds.

- **GEAK Quick vs Full mode**: GEAK now supports a Quick (latency-first) vs Full (deeper search) mode,
  and on the recipe side learns to recommend kernel-fusion opportunities. Old 1.50x early-exit and
  hard-coded homogeneous-mode hints in the kernel-optimization prompt are obsoleted.

- **Critic Agent (first iteration)**: A new Critic Agent backend ships with decision reviewer, prompt
  builder, robustness priors, and web-search support for evidence retrieval during decision review.

- **CI / Automation**: Issue ball-tracker and stale-issue auto-management workflows ship; CI scheduling
  is hardened (top-2000 candidate pool, GLM5 remote mode, Windows-safe `NFS_ROOT`, per-run optimization-
  result publishing).

### 0.3.0 highlights

- **Multi-Agent architecture: Sprint + Marathon unified**: The optimizer's execution backbone is rebuilt:
  the previous single-agent harness is replaced by a multi-agent pipeline that unifies Sprint and Marathon
  under one orchestration model. Workloads now flow through a P0+P1+P2 + kernel-agent layout with each
  agent in its own Claude CLI process and JSONL-based IPC, inheriting the 24h-stable, context-isolated
  runtime first introduced by Marathon in 0.2 — but now applied to Sprint as well, with shared scheduling,
  memory, and trace plumbing.

- **TraceLens enters the E2E optimization loop**: 0.2 introduced TraceLens as a standalone analysis
  surface; 0.3 takes the next step and wires TraceLens directly into the end-to-end inference-optimization
  loop. Profiling, trace splitting, agent invocation, and output parsing all happen inline now — the optimizer
  reasons over fresh trace evidence between actions instead of relying on stale or summarized data. The
  integration is pinned against `release/hyperloom_integration_v0.3`. A new TraceLens Agent Debug Mode also
  exposes the full `StreamJSON` event stream for offline replay and diff against local runs.

- **GEAK gets memory: RAG + Cross-Sessions**: Two long-requested capabilities land together. **GEAK RAG
  enablement** lets GEAK retrieve past optimization knowledge during kernel-rewrite reasoning. **Across-Sessions
  Memory for GEAK** carries learned heuristics, kernel patterns, and outcome data from one optimization session
  to the next, breaking the "every run starts from scratch" pattern. A new internal **Memory Service** provides
  the storage primitives both features sit on, with a Long-Term Memory layer for cross-session knowledge
  retention.

- **Self-Evolving Skills (first iteration)**: Skills are no longer fully static. The first version of self-
  evolving skills ships in 0.3: skills accumulate session-level evidence and propose their own incremental
  updates. The 0.3 surface is intentionally narrow (proposal + manual review); the deeper regression-aware
  auto-update loop is scoped to 0.4.

- **Roofline-aware kernel optimization priority**: The kernel-classification heuristic that decides "what to
  optimize first" is overhauled. Instead of a fixed 5-tier kernel-class priority (`triton > aiter_ck > framework >
  comm > hipblaslt`), 0.3 picks targets via roofline analysis combined with E2E time share. This directly
  addresses the regression where MoE / aiter kernels were either silently skipped (vendor mis-classification) or
  mis-prioritized away from real bottlenecks. Hyperloom now also passes complete kernel metadata to GEAK — shape,
  dtype, backend, runtime args, env vars, and kernel-specific parameters — replacing the previous path-only contract.

- **Multi-Node support comes online**: Hyperloom now supports optimization across multiple nodes, with an
  accompanying Multi-Node CI/CD pipeline that exercises the same A/B testing mechanism single-node runs already use.
  This unlocks training and large-model workloads that don't fit on a single node and keeps the multi-node path
  continuously validated.

### 0.2.0 highlights

- **TraceLens/OOB/Magpie standalone comes online**: Hyperloom now ships dedicated standalone workflows for
  TraceLens, OOB, and Magpie — each usable independently of the full end-to-end optimization pipeline. The
  TraceLens Standalone UI supports three input modes — Default (training / non-vLLM/SGLang eager), Inference with
  Eager (vLLM/SGLang), and Inference with Graph Capture (vLLM/SGLang with a capture folder) — and produces
  structured analysis reports under `/workspace/hyperloom/standalone_analysis.md.` The OOB and Magpie standalone
  paths follow the same invocation pattern, each running its own agent with the appropriate CLI parameters and
  collecting per-run artifacts accordingly.

- **Sandbox Queue for user-facing workload scheduling**: When a user's Hyperloom request exceeds available
  cluster resources, the run is no longer rejected — it is placed into a Sandbox Queue, and the user sees their
  live queue position directly in the UI. This replaces the previous "resource full → fail fast" behavior with
  deterministic FIFO scheduling against the finite sandbox pool (currently enabled on the oci-slc cluster).

- **Agent runtime hardening for long-running sessions**: The executor's permission model has been rebuilt:
  the previous wildcard allow-list (which did not match Claude Code built-in tools) is replaced by an explicit
  allow-list covering Bash, Write, Edit, and other required tools. Agent invocation prompts now enforce strict
  step order, mandate independent Task subagents for Step 6 and Step 7 (context isolation), and require findings
  files to follow the shared template. As a result, LLM-heavy categories such as kernel fusion are no longer
  silently skipped, and subagent hangs caused by permission denial are resolved.

- **CI/CD with inference A/B testing**: Hyperloom ships an inference A/B test workflow integrated into its
  single-node CI/CD pipeline, allowing optimization proposals to be validated against a baseline automatically
  before promotion. Combined with the new auto-labeling system (GitHub Models LLM plus a rule-based engine) and
  project-board automation, the full optimization loop from issue filing to validated CI results is now
  end-to-end traceable.

- **Marathon Inference Launcher (3-pane tmux architecture)**: Hyperloom introduces a new Marathon Inference
  Launcher that runs the inference-optimization pipeline as three independent Claude CLI agents in tmux panes
  (orchestrator, kernel-manager, watchdog), coordinating via JSONL files on shared NFS. This replaces the previous
  Python harness, provides independent context windows per agent (surviving 24h runs), and supports auto-restart
  through the CLI `--continue` loop.

- **White-box visibility: Root Cause and Pending Cause Agents**: Two new supervisory agents make Hyperloom white-box
  at both ends of the user experience. The **Root Cause Agent** watches the Marathon optimization loop for failures,
  diagnoses the cause, and writes actionable guidance back into the next retry prompt — failures stop being binary
  pass-or-discard and become constrained retries. The **Pending Cause Agent** does the analogous thing on the queue
  side: when a user's run is waiting, it surfaces a concrete reason why. Together they replace two previously opaque
  states — "something failed" and "still waiting" — with explainable, user-controllable answers.
