# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

### Removed

- **Kernel-agent LLM role retired** (breaking): the `kernel_agent` role has been
  removed from the role registry. All kernel work was already handled by
  programmatic Python handlers in `orchestrator/kernel/request_handlers.py`; the
  LLM role was a no-op heartbeat responder. The following CLI flags and env vars
  are removed:
  - `--kernel-prompt` — overriding the kernel system prompt is no longer meaningful.
  - `--kernel-codex` / `--kernel-claude` — there is no kernel LLM backend to select.
  - `INFERENCE_OPTIMIZER_KERNEL_AGENT_MAX_TURNS` — no kernel LLM backend.
  - `INFERENCE_OPTIMIZER_KERNEL_CLAUDE_CONVERSATIONAL` — no kernel LLM backend.
  
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
Current packaged version (`pyproject.toml`). See
[release notes](docs/release-notes.md) and the
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

[Unreleased]: https://github.com/AMD-AGI/Hyperloom/compare/v1.0.0a3...HEAD
[v1.0.0a3]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0a3
[v1.0.0a2]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0a2
[v1.0.0a1]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0a1
[0.8.0]: https://github.com/AMD-AGI/Hyperloom/blob/main/docs/release-notes.md
[v0.3]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v0.3
[v0.2]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v0.2
