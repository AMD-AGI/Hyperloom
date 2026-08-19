# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **GEAK same-harness revalidation dispatch.** KERNEL-phase GEAK wins now enqueue
  rebench tasks that survive non-`CLOSE` phase boundaries and can be re-dispatched
  after a cancelled attempt, instead of reusing one key per session and reading
  the settled row back as `rebench_unavailable`. Duplicate dispatch is skipped
  while a rebench is already in flight. `CLOSE` stops any remaining rebench and
  settles the slot, so the phase that only writes reports cannot starve the
  post-opt roofline or rewrite the headline after the report is generated;
  pruning the explore family settles the slot the same way. A rebench result is
  applied only when ``geak_pending`` still tracks it — orphaned and late results
  no longer promote, drop the candidate, or replay through the GEAK harness, and
  are recorded as an observation rather than dropped silently.

## [v1.0.0b2] - 2026-08-19
Current packaged version (`pyproject.toml`). See
[release notes](docs/release-notes.md) and the
[GitHub release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0b2)
for the user-facing summary.

### Removed

- **BREAKING — `kernel_optimization.py` no longer accepts `--test-command` or
  `--test-harness-path`.** The unittest-harness contract they fed had no
  reachable caller; an external invoker still passing either flag now fails in
  argparse rather than being silently ignored.

- **BREAKING — four write-only artifacts are no longer produced**:
  `agent_transcript.jsonl`, `orchestration_turns.jsonl`,
  `mn_input_params_*.json`, and the work_dir copy of `semantic_audit.json`.
  None had a reader. The first three also persisted secrets or raw LLM
  transcripts past a redactor that inspected values but not keys.

- **BREAKING — Magpie leak salvage no longer defaults to `/workspace/`.** It
  runs only when `$INFERENCE_OPTIMIZER_RESCUE_PATHS` is set. Note the blast
  radius: the generic `{framework}_{gpu_type}.sh` scripts respect `$RESULT_DIR`
  and never needed salvage, but a script pinned through
  `params.benchmark_script` that hardcodes `/workspace/` was previously rescued
  and now fails the task with `no_report`. Set the env explicitly to keep the
  old behaviour.

- **BREAKING — the `vendor_kernel_config`, `operator_tuning` and
  `deep_kernel_analysis` actions are gone.** None of them ever had an executor
  or a `KERNEL_REQUEST_HANDLERS` kind, so every request for them was answered
  with `unknown_kernel_kind`; they were authored for the `kernel_agent` LLM
  role that PR #1095 retired. Sessions recorded under the old build may carry
  these names in `state.json` / `coordinator.db`; they are no longer resumable
  and no migration is provided.

- **BREAKING — `actions/_meta/*.yaml` and `orchestrator/actions/registry.py`
  are removed.** Action metadata is now `ACTION_CATALOGUE` in
  `inference_optimizer/protocol/action_surfaces.py`. Editing a yaml no longer
  changes anything because there is no yaml. The `preferred_backend`,
  `preferred_model` and `max_turns` fields are dropped outright: no runtime
  code ever read them, so changing them never had an effect. The
  `params_schema` blocks are dropped for the same reason. `verdict_class`,
  which the old docs described as advisory, is genuinely operational and is
  kept.

- Kernel-owned actions no longer get a no-op executor. A delegate or
  `propose_action` naming one was already denied by PolicyGate
  (`rule=kernel_owned_by_kernel_agent`); the stub only stood ready to report an
  unexecuted action as `succeeded`.

- `run_fusion` is no longer registered in `KERNEL_REQUEST_HANDLERS`. It is
  invoked directly by `KernelPhase`, so no request ever carried that kind.

- The `KERNEL_OPT_BACKENDS` environment variable is gone. No production code
  read it; `KERNEL_OPT_BACKEND_ORDER` is the sole backend switch, and only an
  exact `forge` opts out of the default GEAK phase.

### Changed

- The recommended vLLM container image is now the official upstream
  `vllm/vllm-openai-rocm:v0.27.1` instead of
  `rocm/hyperloom:vllm-v0.27.1-rocm7.2.3`, because AMD deprecated `rocm/vllm`
  and `rocm/vllm-dev`. The tag is a 1:1 replacement, but its entrypoint is
  `vllm serve`, so a long-running Hyperloom container has to override it (for
  example `--entrypoint tail`). SGLang images are unchanged.

- The default Magpie benchmark dependency is upgraded from v0.1.0 to v0.2.0.
  Both the installer and runtime preflight remain pinned to the immutable
  v0.2.0 release commit for reproducible installs.

- **Remote Recipe knowledge now uses one current KB Store contract.** Remote
  mode reads one identity-addressed inference Recipe containing replay config,
  the ordered patch timeline, and nested kernel columns, then publishes one
  final CLOSE session with verified artifacts under the same throughput
  champion. Local Recipe storage and non-Recipe GBrain integrations remain
  unchanged.

- Degraded configuration donors now require exact precision, and a permanently
  missing owner patch is dead-lettered without blocking publication of the
  remaining Recipe sections.

- `_geak_enabled` no longer falls back to the persisted
  `shared_state.kernel_optimizer` field, so `KERNEL_OPT_BACKEND_ORDER` is the
  single source of truth for the kernel backend on a resume as well. The field
  itself is unchanged and still feeds the session breakdown.

## [v1.0.0b1] - 2026-08-11
See [release notes](docs/release-notes.md) and the
[GitHub release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0b1)
for the user-facing summary.

### Added

- **`--no-eval` turns the accuracy eval off for a whole run.** Setting
  `RUN_EVAL=false` by hand leaves the baseline with no accuracy reference, which
  the baseline guard rejects, so the run stopped before it optimized anything.
  The flag makes that an explicit session-wide choice instead: the baseline
  anchors on throughput rather than halting on the missing reference, and every
  candidate lands on the existing `baseline_accuracy == 0` path that already
  degrades to a throughput-only KEEP. A *measured* regression still blocks — the
  scriptable (xDiT) `quality_gate` is computed by the benchmark run itself and
  never consulted `RUN_EVAL`.

  The choice is session state (`shared_state.eval_disabled`), not just a parsed
  arg, so it also reaches the lanes that template their own benchmark config
  rather than inheriting the baseline's: the framework-agent bench, eval-origin
  enablement, the multi-node `lm_eval` preflight install, and the GEAK GEMM
  shape capture. It persists across `--resume`, and is refused with a warning
  once the session has anchored an accuracy, because every KEEP up to that point
  was graded against it.

  Default-off is byte-for-byte today's behaviour. Runs made with the flag are
  not accuracy-validated.

### Fixed

- **Enablement dispatch evidence reaches the specialist again**: the Coordinator
  computes the source lines near the offending site — and, on a weight-init
  failure, the checkpoint's per-layer weight inventory — plus a ranked list of
  bridging PR refs, but since the mandate stopped being passed as free-text
  `notes` none of it was delivered: the mandate was re-rendered downstream from
  a bare request, so the agent was told to find a bridge while the candidates
  already discovered for it were withheld. Both now travel as structured
  `enablement_source_context` / `enablement_candidate_refs` params and are
  folded into the §1b mandate at the point of use.

- **An LLM outage during forge-fusion no longer disables fusion for the rest of the
  session.** forge-fusion reports `verdict: llm_unavailable` (manifest schema v2)
  when discovery never reached the model, which is a fact about the gateway and not
  about the kernel. The wrapper's `_normalize_manifest` had no case for it, so it
  fell through to the generic no-KEEP shape — `status: complete`,
  `micro_decision: no_improvement`, `decision: REVERT`. That was wrong twice over.
  It recorded an outage as an optimization result, and because
  `_fusion_required_before_kernel_opt` skips fusion once `last_fusion.status` is
  `ok`/`complete`/`kept`, a single gateway blip marked fusion "done" and the model
  was never fusion-optimized again in that session.

  It is now shaped like the existing subprocess-timeout result — `status: failed`,
  `micro_decision: failed`, `kept: false`, `error_class: llm_unavailable`, with the
  manifest's error kind, attempt count and message carried through — which is how
  Hyperloom already says "infrastructure failed, this is retryable". A real
  `no_opportunity` (the model was asked and found nothing) is unchanged and still
  suppresses a pointless re-run. The verdict is matched tolerantly and only honoured
  when the manifest reports no KEEP, so it can never discard a validated fusion.

- **vLLM roofline runs no longer launch an unbounded torch profiler**: the
  profile path injects `--profiler-config.delay_iterations/max_iterations` into
  `EXTRA_VLLM_ARGS`, but three later steps could each drop them — a candidate
  carrying `args_mode="replace"` (which `writeback` sets automatically as soon as
  a KEEP needs `remove_args`) overwrote the whole flag string, `extra_envs` could
  override it outright, and `remove_args` strips flags by name — taking the
  `--profiler-config.ignore_frontend True` from the profile YAML with them. vLLM
  reads a missing `max_iterations` as "profile until `stop_profile`", so the
  worker accumulated every profiler event in host anonymous memory — measured at
  60 MiB/s with the production option set — until the cgroup OOM-killer took the
  engine or worker process out mid-roofline, at 107–137 GiB RSS. Because
  `args_mode` is sticky on `current_best`, one such KEEP turned *every* later
  roofline in that session into an OOM candidate.

  `materialize_config_with_envs` now re-asserts the profiler flags as the LAST
  write to `EXTRA_VLLM_ARGS` — after the `extra_server_args`/`extra_envs` merges
  and after `remove_args`/`unset_envs` — restoring only the flags that went
  missing, warning about exactly which ones, and re-running the shell-safety
  guard on the result. `ignore_frontend` is stated alongside the bounds, since the
  AsyncLLM-side profiler tracks no iterations and would otherwise capture the
  entire `start_profile`..`stop_profile` range. Candidate flags still win for
  everything else, and the append path is unchanged apart from no longer relying
  on the YAML to carry `ignore_frontend`.

  The re-assertion checks flag VALUES, not just flag names, for the two flags that
  decide whether the capture is bounded at all: `max_iterations` has to parse as a
  positive integer within the computed serialization-safe cap (vLLM reads 0 as "no
  limit"), and `ignore_frontend` has to be true. A name-only check accepted
  `--profiler-config.max_iterations 0` and then logged that it had bounded the
  profiler — worse than not guarding, since the warning sends the next
  investigation the wrong way. The injected flags also keep overriding whatever the
  YAML pins, via the repeated-flag last-wins vLLM's argparse already applies: a
  hand-written `max_iterations 100000` is unbounded in practice and must not
  displace the computed budget (`HYPERLOOM_PROFILE_MAX_ITERS` is the override
  channel for that), and a stale `capture_torch_profiler_dir` must not send this
  run's traces to a previous session's directory.

  Scope: **vLLM only**. SGLang bounds its capture through `start_step`/`num_steps`
  inside `PROFILE_EXTRA_BODY`, which is written before the same `extra_envs`
  merge and is therefore droppable the same way, but it is not re-asserted here —
  whether a non-positive `num_steps` means "unbounded" or "no capture" needs a
  SGLang-side answer this layer does not have, and every OOM observed so far was
  vLLM. The exposure is called out in a comment at that write site.

### Removed

- **Kernel-agent LLM role retired** (breaking): the `kernel_agent` role has been
  removed from the role registry. All kernel work was already handled by
  programmatic Python handlers in `orchestrator/kernel/request_handlers.py`; the
  LLM role was a no-op heartbeat responder. These env vars are gone, and setting
  them now has no effect:
  - `INFERENCE_OPTIMIZER_KERNEL_AGENT_MAX_TURNS` — no kernel LLM backend.
  - `INFERENCE_OPTIMIZER_KERNEL_CLAUDE_CONVERSATIONAL` — no kernel LLM backend.

  The matching CLI flags **still parse, as accepted no-ops**, so a launcher or
  operator template that passes them keeps starting instead of dying in argparse
  before the run begins. They are hidden from `--help`, nothing reads them, and
  they will be deleted outright in a future release once the callers that pass
  them have been updated:
  - `--kernel-prompt PATH` — overriding the kernel system prompt is no longer
    meaningful. It still consumes its argument, so the path is swallowed rather
    than left behind as a stray positional.
  - `--kernel-codex` / `--kernel-claude` — there is no kernel LLM backend to select.
  
  `--no-kernel` continues to work: it sets `shared_state.kernel_enabled=False`,
  which causes the Coordinator's request router to auto-reject kernel REQUESTs
  with `agent_disabled`.

  The Slurm launcher's `HL_KERNEL_BACKEND` (`codex|claude`) selected the retired
  LLM backend and is removed with it. Use `KERNEL_OPT_BACKEND_ORDER`
  (`geak|forge`) to steer the kernel-opt rewrite ladder; the launcher forwards it
  into the container and every carrier defaults it to `geak`.

  `agents/kernel/SKILL.md` (561 lines, never loaded by Python) has been partially
  superseded by `docs/conceptual/kernel-execution-path.md`, which documents the
  programmatic dispatch flow and artifact layout. Operator sections from the
  original (Credentials, Ray head, Recovery, TraceLens Requirements, Proposal
  Rules) are not carried over; refer to the individual reference docs for those.

## [v1.0.0a3] - 2026-08-05
See [release notes](docs/release-notes.md) and the
[GitHub release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0a3)
for the user-facing summary.

### Added

- **Recipe-KB writes in the Langfuse trace**: every write to the cross-session
  recipe KB (`recipe.json`) is now mirrored as a `kb:recipe_write:<generator>`
  span under the `recipe_kb` agent, alongside the existing
  `kb:recipe_snapshot:<method>` read spans. Both write sites are covered — the
  session-opening T0 identity anchor (`generator=t0_anchor`) and the
  Coordinator's KEEP/REVERT/framework-PR/CLOSE amends
  (`generator=coordinator`), the latter carrying the session's lessons,
  pitfalls, `best_config`, `prs_tested`, `what_worked`/`what_failed` and
  `sessions` entries. Previously only reads were visible, so what a session
  sank into the KB could only be recovered by diffing `history/v*.json`.

  `RecipeKB.put_recipe` emits the audit event (reusing the existing
  `audit_hook` → `runtime/recipe_snapshot/.audit.jsonl` channel), so the
  offline `backfill_langfuse` CLI replays write spans too. Each event reports a
  per-field `delta` against the pre-write row — `put_recipe` rewrites the whole
  row, so absolute counts alone cannot distinguish an amend that appended a
  lesson from the T0 anchor, which round-trips the existing lists untouched.
  Read spans are unchanged, and audit rows predating this change (no `op`
  field) still replay as reads. On-disk recipe rows are untouched: this is a
  trace mirror, so warm-start reads the same data as before.

  `LocalRecipeStore.put_recipe` now additionally returns `prior_counts` and
  `counts` (per-field sizes before/after the write) to support the delta.

### Removed

- **Remote Cortex KB, end to end**: every path that could reach a remote Cortex
  KB is gone, not just its CLI wiring. `--cortex-kb-url` is removed, and the
  Critic's `/v2/reasoning/assess` client (`kb_assess_client.py`) is deleted
  along with the bundle fields (`kb_assess_by_proposal`, `kb_assess_trace`),
  the prompt injection, and the Langfuse `kb_assess` span. `CORTEX_KB_URL`,
  `CORTEX_KB_HTTP_TIMEOUT_SEC` and `CORTEX_KB_ASSESS_INJECT` are no longer read
  anywhere, so setting them in `.env` or the shell has no effect — previously
  the Critic would still call out if the variable reached its environment by
  any route. No Hyperloom code makes outbound requests to a Cortex KB.
- **Specialist `cortex_kb` MCP**: Specialists no longer receive
  `mcp__cortex_kb__*` tools or a `cortex_kb` MCP server in
  `specialist_mcp.json`. PR Monitor MCP remains available when configured.
- **IR-3 preflight**: Remote recipe-KB reachability probe removed; IR-3 now
  probes PR Monitor only. `--degraded-kb` no longer disables PR Monitor.
- **Recipe KB with `--degraded-kb`**: T0/T2/T3/T4 are skipped (`recipe_kb=None`).
- **PolicyGate R4 (`kb_write_unauthorized`)**: removed. `KB_WRITE_TOOL_NAMES`
  was empty, so the rule could never fire while its comment still claimed it
  guarded KB writes. Local Recipe KB writes go through direct Python calls
  (`writeback.py` / `proposals.py`), which R4 never covered. R5
  (`tool_whitelist_role`) is unchanged and still gates PR Monitor / Web tools.

### Changed

- **breaking: `cortex_*` renamed to `recipe_*`**. After the remote Cortex KB
  was removed, the names left behind held a *local* `RecipeKB` and no longer
  referred to anything called Cortex. Renamed across code, prompts and
  serialized data:
  - Python API: `Coordinator.cortex_kb` → `.recipe_kb`, `args.cortex_enabled` →
    `args.recipe_kb_enabled`, `_bootstrap_cortex_kb()` → `_bootstrap_recipe_kb()`,
    `cortex_finalize_recipe_and_journal()` → `finalize_recipe_and_journal()`,
    `_cortex_t4_hook()` → `_recipe_kb_t4_hook()`, and the module
    `orchestrator.knowledge.cortex_t0` → `.recipe_kb_t0`.
  - CLI: `--cortex-strict-fingerprint` → `--recipe-kb-strict-fingerprint`.
    No alias is kept; the legacy flag now fails argparse.
  - Emitted data: SharedState `cortex_session_id` / `cortex_session_summary` →
    `recipe_kb_*`; breakdown `kb_provenance.cortex_session_id` →
    `recipe_kb_session_id`; stop reasons `cortex_t0_failed` /
    `cortex_drain_failed` / `cortex_commit_failed` → `recipe_kb_*`; warm-recipe
    source tag `cortex-kb` → `recipe-kb`; sweep grid source `cortex_recipe` →
    `recipe_kb`. Consumers that parse these values need updating.
  - On disk: `<session>/runtime/cortex/` → `<session>/runtime/recipe_kb/`.
    No migration is provided and none is needed: this directory holds only
    derived bookkeeping, while the authoritative recipe store is the local KB
    root (mirrored to gbrain) outside the session tree. Resuming an older
    session regenerates the snapshots on its next T0 anchor.

  Note: this does **not** touch Primus Cortex (`agents/framework/sources/primus_cortex.py`,
  `PRIMUS_CORTEX_PR_API`), which shares only the word "Cortex" with the removed
  KB. It is the framework-agent's PR-candidate source and the backend behind
  PR Monitor (`--pr-monitor-url` defaults to `$PRIMUS_CORTEX_PR_API`), which
  this release keeps.

## [v1.0.0a2] - 2026-07-29
See [release notes](docs/release-notes.md) and the
[GitHub release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0a2)
for the user-facing summary.

- **breaking(inference_optimizer)**: rename the multi-node `optimize` CLI
  flags `--rayjob-image` → `--mn-image` and `--rayjob-gpus-per-node` →
  `--gpus-per-node`, covering both the `rayjob` and `infera` multi-node
  backends. No alias is kept; the legacy flags now fail argparse. The former
  `INFERENCE_OPTIMIZER_RAYJOB_IMAGE` env is no longer read — set the image via
  `--mn-image`. See the [upgrade guide](docs/reference/upgrade.md).
- **feat(orchestrator)**: absorb PR #461 free-form dynamic specialist
  dispatch. The orchestration agent can `delegate{action_name='dynamic_specialist'}`
  to spawn CPU-only, non-domain-locked specialist sub-agents (claude CLI
  subprocesses) in waves, plus `dynamic_specialist_check` / `_collect`.
  Adds the ActionRegistry `_meta` registration PR #461 omitted (so the
  delegate is no longer denied with `unknown_action` and renders in the
  prompt catalogue), wires the dispatch model to the blessed specialist /
  orchestration model, and adds a liveness reaper that kills timed-out /
  stale subprocesses (process-group SIGTERM/SIGKILL) so the run never
  leaks zombie agents.
- Add repository governance docs (LICENSE, SECURITY.md, CONTRIBUTING.md, CODE_OF_CONDUCT.md).
- Add structured Sphinx documentation under `docs/`: install guides, how-to
  guides, reference material, component pages, release notes, and compatibility
  docs.
- Refresh the optimization-loop documentation under
  `docs/conceptual/optimization-loop.md` and add
  `src/hyperloom/inference_optimizer/README.md` as a package-level entry point.
- README now links to the structured docs from its "Get Started" and
  "Documentation" sections.
- **fix(orchestrator)**: drop pre-M4 `select_kernels` request alias and the
  legacy `SharedState.last_select_kernels` / `record_select_kernels` mirror.
  Only the canonical `trace_analyze` kind / `last_trace_analyze` cache
  remain; readers had previously checked the removed mirror in
  `_kernel_phase_todos` TODO 3/5, which caused the KERNEL-phase
  `trace_analyze` request loop (RooflineExecutor populated only the
  canonical cache, so the guard never saw a fresh entry and forever
  instructed the LLM to re-emit the request). Resume of a stale
  `state.json` carrying `last_select_kernels` silently drops the slot
  via `_legacy_drop_fields`.

## [v1.0.0a1] - 2026-07-22
See [release notes](docs/release-notes.md) for the user-facing summary.

## [0.8.0]
Earlier packaged version. See [release notes](docs/release-notes.md) for the
user-facing summary.

## [v0.3] - 2026-05-14
### Added
- Opt-in PMC roofline action gated after `select_kernels`, deriving workload from materialized Magpie config.
- PMC roofline integration tests for Ray-based execution path.

### Fixed
- Enforce PMC roofline GPU work to run inside a Ray-owned worker while preserving local debug escape hatches.
- Resolve PMC roofline GPU spec handling for Ray contexts.

## [v0.2] - 2026-04-22
### Added
- Hardened optimization protocol with deep kernel analysis, KM feed pipeline improvements, micro-benchmarking, and GPU time-share handling.
- Vendor kernel configuration guidance and updated kernel-manager skills/actions (including local-test flow).
- Launcher scripts refinements for orchestrator/kernel manager panes.

[Unreleased]: https://github.com/AMD-AGI/Hyperloom/compare/v1.0.0b2...HEAD
[v1.0.0b2]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0b2
[v1.0.0b1]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0b1
[v1.0.0a3]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0a3
[v1.0.0a2]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0a2
[v1.0.0a1]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0a1
[0.8.0]: https://github.com/AMD-AGI/Hyperloom/blob/main/docs/release-notes.md
[v0.3]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v0.3
[v0.2]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v0.2
