# Agent-Delegate Gap Analysis
## Goal: make `inference_optimizer` truly multi-agent (Arbor-style), and retire single-agent exploration.

Scope: read `./inference_optimizer` against `/wekafs/zgong/Arbor` (skills/SKILL.md, `src/arbor/dispatch.py`, `src/arbor/comms.py`, `src/arbor/prompt_builder.py`, `src/arbor/gpu_pool.py`, `launcher/orchestrator.md`) and the artifacts of the just-completed DeepSeek‑R1‑0528 run (`/workspace/hyperloom-KB/`).

Decision: the user has requested that the **single-agent exploration form factor must be fully discarded** and replaced by a real dispatch‑agent loop modelled on Arbor. This document enumerates every blocking gap.

---

## 0. Empirical evidence the dispatch path never fired

A 13.12% gain run just completed against DeepSeek-R1-0528 (`/workspace/hyperloom-KB/`):

| Source | Evidence |
|---|---|
| `manifest.json` | `research_lane_capacity = 1` → SpecialistRunner *was* registered. |
| `storage/coordinator.db` (`tasks` group-by kind) | `baseline×1, target_analysis×1, explore×4, sweep×2, report×2, session_breakdown×1`. **Zero tasks of `kind='specialist'`.** |
| `runs/` directory layout | `baseline/`, `explore/`, `sweep/`, `profile/`, `kernel_opt/`, … — **no `specialist/`**. |
| `agents/orchestration/system_prompt.snapshot.md` | EXPLORE allowed-actions block says `explore, recover, specialist`, but section "## 4. ACTIONS YOU MAY USE" lists **only** `target_analysis / baseline / explore / sweep / report / recover`. `specialist` has no entry, no EMIT hint, no payload schema. |
| Critic / Robustness workdirs | They are per‑tick evaluators, not dispatchers. Neither emits `delegate` for specialists. |
| Coordinator events / `delegate` intents over the entire session | Every `delegate` had `action_name='explore'` and was authored by the Orchestration LLM with `provenance='llm_direct'`. |

**Conclusion**: the wiring is half there; the calling protocol is not. The Orchestration LLM never asked for a specialist because the system prompt does not document the action.

---

## 1. Architectural diff at a glance

| Concern | Arbor (target) | inference_optimizer (current) |
|---|---|---|
| **Specialist dispatch unit** | OS subprocess; one per agent | In‑process Python call from the same orchestrator reactor |
| **Concurrency model** | `dispatch_batch` launches N specialists in parallel; pollable via `reap_completed` | `research_lane_capacity=1` (single concurrent specialist), no batch primitive |
| **GPU allocation** | Shared `GPUPool` (file-locked JSON) over a dynamic GPU set; CPU‑only specialists are unbounded | No GPU pool. Specialists never get GPUs. Serving GPUs are the only GPUs the orchestrator knows about |
| **Liveness protocol** | Per-agent dir with `task.json / heartbeat.json / results.jsonl / patches/ / done.json` | `runs/specialist/<task_id>/{prompt.md, transcript.jsonl, heartbeat.json, specialist_done.json}` (parsed but never produced because dispatcher never fires) |
| **Agent output unit** | Code patches under `patches/`, plus `config_changes`, plus `new_knowledge.md` | `specialist_done.proposal_set` — a list of *config variants* for the next `explore` round (no source-code patches at all) |
| **Tool surface for specialists** | Bash, Read, Write, Edit, MultiEdit, Grep, Glob, Task, WebSearch, WebFetch, TodoWrite, MCP tools | `emit_intent, Read, Grep, Glob, Bash, Web*, mcp__pr_monitor__*, mcp__cortex_kb__*` — **Edit/Write/MultiEdit are explicitly in `SPECIALIST_TOOL_DENYLIST`** |
| **Code-change capability** | First-class. Iron rule: "every dispatch round MUST include at least one code-patch agent" | None. Specialists physically cannot write source patches |
| **Domain catalogue** | Constructed dynamically: orchestrator decides what expertise is needed and crafts a prompt | Frozen 6-domain catalogue in `specialist_domains.py`; only `serving_specialist` is M5-active; the other 5 are M6 (no per-domain prompt template) |
| **Knowledge feed** | `arbor/skills/kb/{kernel,communication,compiler,framework,fusion,systems,pr_intelligence,recipes}/`, selected by `select_kb(domain)` and embedded into the agent prompt | `kb_subgraph` slot in `SpecialistPromptInputs`, fed from Cortex KB warmup — which was empty for the full DeepSeek-R1 run |
| **Recipe / warm-start** | `recipes.load_recipe(model, hw)` → `best_config + remaining_gaps + pitfalls + stack_fingerprint`, applied at session start, persisted across sessions | No equivalent. Manifest has a `stack_fingerprint` but no recipe lookup, no remaining_gaps, no pitfalls feed |
| **Integration loop** | Apply one agent's patches/configs → restart server → gate (throughput + accuracy) → keep/revert → re-profile | `specialist_done.proposal_set` is folded back into a future `explore` grid; no patch apply, no separate restart for specialist output |
| **Orchestrator authority** | Full Bash, Read, Write, Edit, …, Task tool — can also self-dispatch | Orchestration is `emit_intent`-only by policy; cannot launch tasks except via `delegate` intents — and **`delegate{action_name='specialist'}` is undocumented in its system prompt** |
| **Iron rules** | Every dispatch round must mix config + code; ≥1 code-patch agent; revert at ‑0.5% throughput OR accuracy ‑1pp | Inlined `explore` runs config grids only; no code/config mix policy |

---

## 2. Gap inventory (root causes & their failure modes)

### G1. The specialist action is invisible to the Orchestration LLM
**Symptom**: Orchestration emits `delegate{action_name='explore'}` exclusively.
**Root cause**:
1. No `actions/_meta/specialist.yaml` exists in `inference_optimizer/actions/_meta/` (verified — directory listing has 15 files; specialist not among them).
2. `prompt_builder._section_action_catalogue()` iterates the ActionRegistry; since specialist has no metadata, no entry is rendered. The LLM only knows specialist as a *provenance tag string* on explore-variant rows.
3. There is no `_format_emit_hint` payload schema for specialist, no `EMIT:` line, no per-domain example.

**Files**:
- `inference_optimizer/orchestrator/system_prompts/prompt_builder.py::_section_action_catalogue` (lines ~385‑425)
- `inference_optimizer/actions/_meta/` (missing `specialist.yaml`)
- `inference_optimizer/orchestrator/system_prompts/orchestration.md` (rules text mentions specialist only obliquely)

### G2. Even with G1 fixed, capacity is 1 → no parallelism
**Symptom**: even if the LLM tried, the lane would block.
**Root cause**: `--research-lane-capacity` defaults to 1. The CLI warns at >6. `_set_lane_capacity(self.db.raw, "research_lane", cap)` in `coordinator.py:467`.
**Implication**: Arbor's "dispatch 6 specialists across 4 gaps in parallel" pattern is impossible without raising the cap *and* having a batch-dispatch primitive on the Orchestration side.
**Files**: `cli.py::517-563`, `coordinator.py::458-471`, `policy.py::~94` (capacity-1 docstring).

### G3. Five of six domains are stub-only (no prompt template)
**Symptom**: even if Orchestration delegated `kernel_switch_specialist`, `comm_specialist`, `compiler_specialist`, `system_specialist`, `pr_intel_specialist`, the SpecialistRunner falls back to a *generic* template (see `SPECIALIST_DOMAINS_M5` in `specialist_domains.py:135`).
**Root cause**: `specialist_prompt_builder.py` only ships the framework-specialist template; M6 was never wired.
**Files**:
- `inference_optimizer/orchestrator/specialist_domains.py:131-137`
- `inference_optimizer/orchestrator/system_prompts/specialist_prompt_builder.py` (single template path)

### G4. Specialists cannot produce code patches
**Symptom**: even if specialists ran, their output is a `proposal_set` (list of config knob variants). Source patches are not collected, not applied.
**Root cause**:
1. `SPECIALIST_TOOL_DENYLIST = {Edit, Write, MultiEdit} | KB_WRITE_TOOL_NAMES` — strips write tools defensively (`specialist_runner.py:103`).
2. `_handle_specialist_done` routes `proposal_set` only — the `_record_specialist_result` path feeds future explore grids. There is no `collect_patches(...)` / `git apply` / `restart_server` / `run_gate` pipeline.
3. There is no per-specialist git worktree (Arbor's `dispatch.py::_setup_kernel_worktree` style isolation).

**Files**:
- `inference_optimizer/orchestrator/specialist_runner.py:100-106`
- `inference_optimizer/orchestrator/coordinator.py::_handle_specialist_done` (~line 4528), `_record_specialist_result` (~line 4588)

### G5. No subprocess isolation; same Python process, same Claude SDK
**Symptom**: a hung Claude call in a specialist stalls the orchestrator reactor.
**Root cause**: `SpecialistRunner.run()` is awaited inside the dispatcher event loop using the same in-process `ClaudeBackend` (`cli.py::_build_specialist_executor → _backend_factory`). No subprocess, no separate log fd, no per-agent OS-level liveness signal.
**Arbor contrast**: `_dispatch_via_cli` spawns `subprocess.Popen(["claude", "--print", "--system-prompt-file", ...])` writing to its own `process.log`. Reaping uses `proc.poll()` plus `done.json` existence. Hung agents cannot block the orchestrator.

**Files**:
- `inference_optimizer/orchestrator/specialist_runner.py::SpecialistRunner.run` (entire async body)
- `inference_optimizer/cli.py::_build_specialist_executor` (697-721)

### G6. No GPU pool / no dynamic GPU allocation
**Symptom**: specialists cannot run on-GPU micro-benchmarks, kernel compilation, profiling — even though Arbor's whole reason for GPU isolation is to keep specialists off the serving GPUs.
**Root cause**: `inference_optimizer` knows only `--serving-gpus` (mapped to TP). There is no concept of "dynamic GPUs 2-7" available to specialists. `gpu_pool.py` analogue does not exist; `resource_lock.py` allocates *lanes*, not GPUs.
**Implication**: every specialist is forced to be CPU-only research-only, which makes the "kernel-patch agent" branch in Arbor's design impossible.

**Files (missing)**:
- `inference_optimizer/orchestrator/gpu_pool.py` — does not exist.
- `inference_optimizer/orchestrator/resource_lock.py` only models lane semaphores.

### G7. No `dispatch_batch` primitive on the LLM-visible surface
**Symptom**: Arbor's orchestrator builds N tasks then calls `dispatch_batch(tasks, ..., gpu_pool=pool)` — one call, N agents. inference_optimizer's Orchestration LLM has to emit one `delegate{...}` per agent, sequentially.
**Root cause**: PolicyGate / Coordinator process intents one at a time and route to the `research_lane`. There is no "batch delegate" intent and no orchestrator-side multiplexer.
**Files**:
- `inference_optimizer/orchestrator/intent_parser.py` (no batch type)
- `inference_optimizer/orchestrator/coordinator.py::_handle_delegate` (~line 4027) processes one delegate per tick

### G8. No recipe / cross-session warm-start
**Symptom**: every session re-explores from scratch — no `best_config` carry-over, no `remaining_gaps`, no `pitfalls`.
**Root cause**: there is no `recipes.py` analogue. The manifest captures `stack_fingerprint` for breakdown only. Cortex KB is the *intended* persistence layer but the run shows it was effectively empty (HTTP 422 sync errors, `kb_subgraph={}`).
**Arbor contrast**: `arbor/recipes.py::{load_recipe, save_recipe, update_from_session, is_stale}`; persisted per `(model, hardware)` pair; applied as the prompt's "Prior Optimization History" block via `prompt_builder.build_recipe_summary`.

**Files (missing/inadequate)**:
- `inference_optimizer/orchestrator/recipes.py` — does not exist.
- `inference_optimizer/orchestrator/knowledge_plane.py` — partial; warm-up best-effort only.

### G9. No knowledge-base bundle for specialists
**Symptom**: `SpecialistPromptInputs.kb_subgraph` was `{}` for the entire DeepSeek‑R1 run; specialists would have started "from zero" even if dispatched.
**Root cause**: Arbor ships a curated, versioned KB tree (`arbor/skills/kb/{kernel,communication,compiler,framework,fusion,systems,pr_intelligence,recipes}/`). `select_kb(domain)` picks files; `load_kb_content` concatenates them into the prompt.
inference_optimizer routes KB through *Cortex KB MCP* (a remote service). When Cortex is unreachable, specialists see an empty KB. There is no on-disk fallback.

**Files**:
- `inference_optimizer/orchestrator/knowledge_plane.py`
- `inference_optimizer/cortex_kb_client.py`
- No on-disk `kb/` tree exists in the repo.

### G10. No "code vs config" intervention-mix policy
**Symptom**: every successful round was a config tweak (env vars + grid args). Throughput plateau hit at +13% because the deeper code-level wins are out of reach.
**Root cause**: Arbor enforces:
- "Every dispatch round MUST include at least one code-patch agent."
- "After 2 consecutive config-only iterations, MUST shift strategy to a code-patch agent."
- `record_action(..., change_type='config'|'code_patch')` to track the mix.

inference_optimizer has neither the typing (`change_type` is not a field on `record_action`) nor the gating logic.

**Files**:
- `inference_optimizer/orchestrator/coordinator.py::record_action` (no change_type)
- `inference_optimizer/orchestrator/policy.py` (no escalation rule)
- Decision framework in `prompt_builder._section_decision_framework` does not mandate code patches.

### G11. Orchestration prompt enumerates only `emit_intent` as a tool
**Symptom**: even when an LLM wants to "stop and search the codebase", the Orchestration role cannot. It cannot read a file, cannot run Bash, cannot call Task (sub-agent spawn). The repo design intentionally constrains it to *intents only*.
**Arbor contrast**: Arbor's orchestrator system prompt advertises `Bash, Read, Write, Edit, Grep, Glob, Task, WebSearch, WebFetch, TodoWrite` plus MCP. The orchestrator is itself an agent that can directly dispatch.
**Implication**: in the new design we must either (a) widen Orchestration's tool surface to include a `dispatch_agents` tool (like Arbor's `cli_dispatch_tools.cmd_dispatch`), or (b) introduce a "dispatcher specialist" intent type that does the batch fan-out.

**Files**:
- `inference_optimizer/orchestrator/backends/claude.py:: _OUTPUT_INSTRUCTIONS` — only documents `emit_intent`.
- `inference_optimizer/orchestrator/system_prompts/orchestration.md` — bans Bash/Read/etc. for the role.

### G12. No accuracy gate inside the specialist integration path
**Symptom**: the current `explore` action runs an internal accuracy guard, but a putative specialist code-patch would bypass that because there is no specialist→integrate→gate pipeline.
**Root cause**: G4 + missing `bench.run_gate` analogue. Arbor's gate runs throughput + GSM8K and reverts on either failure.
**Files**: `inference_optimizer/orchestrator/action_executors/_accuracy_gate.py` is per-variant inside `explore`; there is no orchestrator-level gate the dispatch loop can call.

### G13. PR Monitor / GitHub search is optional and was disabled
**Symptom**: even if specialists ran, they could not pull recent PRs from `ROCm/vllm`, `sgl-project/sglang`, etc.
**Root cause**: `--degraded-pr` path strips the `mcp__pr_monitor__*` tool block on specialist dispatch. Arbor treats GitHub PR search as a *core* tool (it ships `github_search.py` baked in).
**Files**:
- `inference_optimizer/orchestrator/knowledge_plane.py` (pr_monitor_enabled flag)
- `inference_optimizer/orchestrator/specialist_runner.py::_resolve_tools` (strips on degraded)
- Missing: `inference_optimizer/orchestrator/github_search.py` (Arbor has it; we don't)

### G14. The Critic / Robustness roles are not the dispatcher
**Symptom**: One might assume Critic or Robustness could dispatch specialists. They cannot — they only emit `review_verdict` / `alert` / `escalate_strategy_change`. PolicyGate blocks them from `delegate`.
**Files**: `inference_optimizer/orchestrator/agent_role.py`, `policy.py` (role-scoped intent allowlist).
**Implication**: Orchestration is the ONLY role that can dispatch. So fixing G1 (action visibility) and G11 (tool surface) is on Orchestration alone — Critic/Robustness do not need new dispatch capability.

### G15. There is no "single-agent exploration" feature flag to retire
**Symptom**: the current explore loop is the *only* path. There is no `--multi-agent on/off` flag; the system simply ignores the specialist path.
**Implication**: discarding single-agent exploration means deleting code, not toggling a flag. Concretely:
- `action_executors/explore.py` keeps its LLM-direct grid path (`provenance='llm_direct'`).
- The Orchestration prompt + decision framework needs to *forbid* `provenance='llm_direct'` and *require* `provenance='specialist:<domain>'` for every grid variant.
- Or: every explore round must be preceded by ≥1 specialist dispatch that yields the `proposal_set` that fills the grid (and the orchestrator must refuse to call `explore` with `llm_direct` provenance once multi-agent mode is on).

---

## 3. The single-agent path that must be discarded (concretely)

The "single-agent exploration form" the user wants to retire is:
- Orchestration LLM (Claude) reasons about gaps in-prompt → emits `delegate{action_name='explore', params={grid:[...], provenance:'llm_direct'}}` directly.
- All variants are authored by the same orchestration model from the same prompt window in the same process.
- No sub-agent ever spawns. The `explore` executor just runs Magpie + bench on each variant.

**Files to gut / re-route**:
| File | Today | After refactor |
|---|---|---|
| `prompt_builder._section_decision_framework` | step 5d: "specialist proposal_set (M5+) — when an explore round just finished, the proposal_set drives the next explore grid" — optional | mandatory: `provenance='llm_direct'` denied; every explore grid must trace to a specialist's proposal_set |
| `prompt_builder._format_grid_injection_hint('explore')` | lists `provenance ∈ {'default_grid','llm_direct','specialist:<domain>'}` | drop `'llm_direct'` from the LLM-visible options |
| `policy.py` PolicyGate | accepts any provenance | new rule `explore_provenance_required_specialist` denies `llm_direct` (warm-period grace for PRELUDE→EXPLORE handoff if needed) |
| `action_executors/explore.py::run` | accepts variants with provenance='llm_direct' | reject (or treat as `default_grid` fallback)
| `coordinator.py::_handle_delegate(action='explore')` | enqueues even if no specialist ran in this phase | gate: deny if `SharedState.specialist_runs == 0` in the current EXPLORE entry |
| `orchestration.md` rules fragment | does not mention specialist-first | new rule: "PRE-explore you MUST `delegate{action_name='specialist', params={domain,gap}}` for each top-K gap; only after ≥1 specialist_done may you propose `explore`" |

---

## 4. The minimum work to make dispatch actually fire

In dependency order. Each item is sized so it can land independently behind a feature flag (`--dispatch-mode=arbor`), with `single-agent` removal as the final step.

1. **Author `actions/_meta/specialist.yaml`** + a per-domain markdown body under `actions/specialist.md`. Schema must match `ActionMetadata`: family=creative/analysis, requires_lanes=[research_lane], preferred_backend=claude, allowed_tools=DEFAULT_SPECIALIST_TOOLS minus denylist, max_turns from `DEFAULT_SPECIALIST_MAX_TURNS`. Once present, `_section_action_catalogue` will render an EMIT line.
2. **Extend `_format_emit_hint('specialist')`** in `prompt_builder.py` with the concrete payload schema: `delegate{action_name='specialist', params={domain:<one-of>, gap_canonical_id, gap_symptom, gap_layer, gap_evidence?, kb_subgraph?, pr_feed?, source_hint_directories?, max_turns?}}`.
3. **Update `orchestration.md`** rules fragment: in EXPLORE you MUST dispatch ≥1 specialist before any `explore` proposal; cite the specialist's `task_id` in the resulting `explore.params.kb_evidence` per variant.
4. **Implement parallel dispatch**: a new intent shape `delegate_batch` (or accept a list under `delegate.params.tasks`). Coordinator fans out into N TaskRegistry rows in a single tick. PolicyGate R2 validates each.
5. **Raise default `research_lane_capacity`** to 4 (parity with Arbor's "6 dynamic GPUs" minus the serving-GPU-blocking specialists). Cap at MAX_PARALLEL_SPECIALISTS.
6. **Add subprocess isolation** in `SpecialistRunner._run_backend` — replace in-process `ClaudeBackend.run` with `subprocess.Popen(["claude", "--print", "--system-prompt-file", ..., "--mcp-config", ...])` piping to per-agent `process.log`. Mirror Arbor's `_dispatch_via_cli`. Per-agent `heartbeat.json` written from within the subprocess via `claude-agent-sdk` hook OR an injected harness.
7. **Introduce a `GPUPool`** module — port `arbor/gpu_pool.py` near-verbatim. Detect serving GPUs via `/proc` scan (Arbor already has the pattern). Wire `--dynamic-gpus` CLI flag.
8. **Add per-domain prompt templates** in `specialist_prompt_builder.py` for kernel/comm/compiler/system/pr_intel. Each template embeds the matching KB content + PR feed + source hint directories.
9. **Bundle the KB**: copy or hard-link Arbor's `skills/kb/{kernel,communication,compiler,framework,fusion,systems,pr_intelligence,recipes}/` into `inference_optimizer/kb/` (subject to license review). Add `kb.select_kb / load_kb_content` helpers.
10. **Build the code-patch integration path**:
    - Remove `Edit, Write, MultiEdit` from `SPECIALIST_TOOL_DENYLIST` (gate behind `--allow-specialist-patches`).
    - Per-specialist git worktree (`runs/specialist/<task_id>/worktree`).
    - On `specialist_done.patches_written != []`, enqueue an `integrate` task that: applies patches → restarts server → runs throughput + accuracy gate → KEEP or REVERT.
11. **Recipe layer**: a `recipes.py` keyed on `(model_name, gpu_type, framework)` persisted to `~/.cache/inference_optimizer/recipes/` AND mirrored into Cortex KB. Load at session start, surface as the orchestration prompt's "Prior Optimization History" block.
12. **Intervention-mix policy**: add `change_type ∈ {config, code_patch}` to `record_action`; PolicyGate denies a second consecutive `explore` (config) if no specialist code-patch has succeeded in the current EXPLORE phase.
13. **Discard the single-agent path**: once 1–12 land and tests/integration prove stable, flip `DISPATCH_MODE` default from `single_agent` to `multi_agent`, then delete the `provenance='llm_direct'` branch in `explore.py`, the LLM-direct grid hints in `prompt_builder`, and the orchestration prompt language that allows it.

---

## 5. Test/observability obligations

- `breakdown.specialist_runs` already exists as a schema slot (`coordinator.py::_record_specialist_result`); the report tail must render it as a first-class section so we can verify multi-agent dispatch on every run.
- Add a session-level invariant test: `assert tasks_by_kind['specialist'] >= 1 when phase==EXPLORE has ever been entered AND --dispatch-mode=arbor`.
- Tests already exist: `tests/test_v08_m5_specialist_integration.py`, `tests/test_v08_m5_specialist_lifecycle.py`. They cover the wiring; they do NOT cover the *prompt visibility* gap (G1). Add a test that asserts the rendered orchestration system prompt contains `EMIT: delegate{action_name='specialist'` whenever `research_lane_capacity > 0` and phase is EXPLORE.
- Add a regression test that asserts SubAgentRunner cannot run any executor of kind `kernel_opt / integrate / …` from inside a SpecialistRunner subprocess (defense in depth for G4 once write tools are re-enabled).

---

## 6. Summary of why current state is "wiring without calling"

The repo carries five well-formed pieces:
- `SpecialistRunner` (635 lines, complete state machine)
- `SubAgentRunner.register_executor('specialist', …)` (wired in `cli.py:804`)
- `PolicyGate` accepts `delegate{action_name='specialist'}` in EXPLORE
- `phase_state.PHASE_ALLOWED_ACTIONS['EXPLORE']` includes `specialist`
- `research_lane_capacity` manifests propagate through Coordinator

…but exactly **one missing piece** stops it from ever firing: **the Orchestration LLM does not know the action exists, because no `actions/_meta/specialist.yaml` is loaded and `prompt_builder._section_action_catalogue` renders nothing for it**. G1 is the single highest-leverage fix.

The other 14 gaps (G2–G15) are the difference between *firing once* and *Arbor-level multi-agent throughput*. They must all be closed before single-agent exploration is retired, because today's single-agent path is also the only safety net.

---

## 7. Suggested PR slicing

Smallest reviewable PRs, in order:

1. **PR-1 (G1)**: ship `actions/_meta/specialist.yaml` + `actions/specialist.md` + `_format_emit_hint('specialist', …)` + a prompt snapshot test. After this, the LLM can finally emit specialist delegates. End-to-end smoke test: dispatch 1 specialist, confirm `tasks` table has `kind='specialist'`.
2. **PR-2 (G3 + G9)**: ship the 5 per-domain prompt templates + bundle the on-disk KB tree.
3. **PR-3 (G6 + G5)**: introduce `gpu_pool.py` + subprocess specialist dispatch.
4. **PR-4 (G7 + G2)**: add `delegate_batch` intent; raise `research_lane_capacity` default to 4.
5. **PR-5 (G4 + G12)**: lift write-tool denylist behind a flag; add patch→integrate→gate pipeline.
6. **PR-6 (G8)**: recipes layer.
7. **PR-7 (G10 + G15)**: intervention-mix policy; deprecate `provenance='llm_direct'`; remove single-agent path.
8. **PR-8 (G11)**: optionally widen Orchestration's tool surface with a `dispatch_agents` tool, only if PR-4's batch intent is judged insufficient.
9. **PR-9 (G13)**: ship `github_search.py` + make PR Monitor a hard dependency.

PR-1 is the unblocker. PR-7 is the user's stated end-state.
