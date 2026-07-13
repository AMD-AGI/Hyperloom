# Hyperloom Token Optimization Plan

> Goal: reduce unnecessary token spend without changing product behavior or
> forcing hard output caps. This plan replaces the stricter parts of
> `token_eff.plan.md` with token-efficiency changes that are observable,
> reversible, and compatible with Python 3.10.

## Decisions

### Claude SDK Version

Use:

```text
claude-agent-sdk>=0.2.110
```

Rationale:

- `0.2.110` is the minimum baseline that supports Python `>=3.10` and publishes
  Linux wheels for `manylinux_2_17_x86_64` / `aarch64`.
- `ClaudeAgentOptions` from `0.2.110` onward exposes the fields needed here:
  `thinking`, `effort`, `max_budget_usd`, `resume`, `strict_mcp_config`,
  `tools`, and `output_format`.
- A lower bound (`>=0.2.110`) is preferable to an exact pin: operators can pick
  up SDK patch/minor fixes without editing Hyperloom, while still excluding
  pre-`0.2.110` builds that lack the required options.

Note that `>=0.1.65` already resolves to `0.2.110+` today, so raising the floor
buys **capability guarantees**, not merely a newer install: `thinking` /
`effort` are already installable and simply unused by current code. Because
P0.2/P3.2 introduce new `ClaudeAgentOptions` kwargs, the compatibility test
that instantiates the options with Hyperloom's required kwargs belongs in P0.1
now (see P0.1).

### No Hard Length Limits

Do **not** add hard `maxLength` fields to `emit_intent` schemas.

Do **not** reject intents because `body_md`, `reasoning`, `summary`, `detail`,
or `reason` are too long.

Do **not** introduce output truncation that can silently remove model intent
content.

The optimization strategy is to reduce repeated input context, prefer compact
summaries, configure thinking/effort conservatively, narrow tool surfaces, and
measure cache behavior accurately. Output guidance may ask the model to be
concise, but it must remain advisory rather than a validator-enforced limit.

The only non-advisory backstop is the optional `max_budget_usd` operator knob
(P0.2): it bounds spend per turn without inspecting or truncating content, so it
guards against a runaway turn while preserving the no-truncation guarantee.

## Priority Plan

### P0 - Dependency and High-Frequency Prompt Spend

#### P0.1 Raise the Minimum Claude SDK Version (Python 3.10+)

Files:

- `pyproject.toml`
- `src/hyperloom/inference_optimizer/cli/preflight.py`
- `src/hyperloom/inference_optimizer/assets/install.sh`
- `src/hyperloom/inference_optimizer/references/troubleshooting.md`
- `src/hyperloom/inference_optimizer/SKILL.md`

Changes:

- Replace `claude-agent-sdk>=0.1.65` with `claude-agent-sdk>=0.2.110`.
- Replace the optional `claude_agent_sdk>=0.1` extra with
  `claude-agent-sdk>=0.2.110`, or remove the duplicate extra if it is no longer
  needed.
- Update preflight auto-install messages and troubleshooting docs so operators
  install a build that satisfies the new minimum.
- Add/adjust a unit test that verifies the effective requirement string.
- Add a compatibility test that instantiates `ClaudeAgentOptions` with every
  kwarg P0.2/P3.2 will pass (`thinking`, `effort`, `strict_mcp_config`, `tools`,
  `resume`) and asserts the SDK accepts them. This is the same class of guard as
  the `resume=` downgrade in P1.2, applied up front instead of at runtime.

Why:

- Hyperloom needs `thinking` / `effort` support on a known-good SDK baseline while
  still allowing newer compatible releases.

#### P0.2 Configure Claude Thinking/Effort Softly

Files:

- `src/hyperloom/orchestrator/roles/claude.py`
- `src/hyperloom/inference_optimizer/cli/backends.py`
- `src/hyperloom/inference_optimizer/tests/test_claude_backend.py`
- `src/hyperloom/inference_optimizer/tests/test_claude_backend_branches_unit.py`

Changes:

- Add env-driven Claude SDK options in `ClaudeBackend._build_options()`:
  - `INFERENCE_OPTIMIZER_CLAUDE_THINKING`, default `adaptive`.
  - `INFERENCE_OPTIMIZER_CLAUDE_EFFORT`, default `medium`.
  - Per-role overrides: `INFERENCE_OPTIMIZER_CLAUDE_ORCHESTRATION_EFFORT`
    (default `medium`) and `INFERENCE_OPTIMIZER_CLAUDE_KERNEL_EFFORT` (default
    `low`, since kernel reactor turns are high-frequency and mechanical).
- Use `thinking={"type": "adaptive"}` rather than fixed `budget_tokens=N` by
  default; let the model size its own reasoning instead of forcing a floor.
- Default `effort="medium"` for orchestration. Orchestration is the core
  decision loop that picks the next optimization and judges specialist output;
  defaulting it to `low` risks degrading the exact thing the product exists to
  do. Reserve `low` for high-frequency mechanical turns (kernel polling).
- Allow operators to set `effort=high|xhigh|max` when a run needs deeper
  reasoning, or `low` when a run is explicitly cost-bound.
- Keep `max_budget_usd` off by default, but wire it as an opt-in operator knob
  now: since this plan deliberately adds no hard output cap, `max_budget_usd` is
  the intended hard backstop against a runaway verbose turn — the escape valve
  the token-usage reviewers can point to without Hyperloom ever truncating
  model intent.
- Keep feature detection around SDK kwargs so a broken environment surfaces a
  structured warning instead of failing obscurely.

Why:

- This guides the model toward lower reasoning spend without enforcing a hard
  token ceiling or truncating useful output.
- Effort changes must be judged on decision quality, not token count alone: gate
  any effort reduction on a run where final `cumulative_gain` does not regress
  versus the medium-effort baseline (see Success Metrics guardrail).

#### P0.3 Stop Inlining Full `analysis.md` by Default

Files:

- `src/hyperloom/orchestrator/state/_shared_state/render.py`
- `src/hyperloom/orchestrator/roles/mcp_context_tools.py`
- `src/hyperloom/inference_optimizer/tests/test_roofline_executor.py`
- `src/hyperloom/inference_optimizer/tests/test_roofline_snapshot_unit.py`

Changes:

- Change `INFERENCE_OPTIMIZER_PROMPT_ANALYSIS_MD_INLINE` default from `"1"` to
  off.
- Keep `profiler_digest` in the prompt.
- Keep `show_analysis_md` available as an explicit pull tool for cases that
  need the full TraceLens report.
- Preserve the current opt-in escape hatch:
  `INFERENCE_OPTIMIZER_PROMPT_ANALYSIS_MD_INLINE=1`.
- Add tests for both default digest-only behavior and explicit inline behavior.

Why:

- This removes the largest repeated input block while preserving access to the
  original data.

#### P0.4 Replace Hard Output Limits with Soft Output Hygiene

Files:

- `src/hyperloom/orchestrator/roles/claude.py`
- `src/hyperloom/orchestrator/roles/codex.py`
- `src/hyperloom/orchestrator/roles/critic_agent.py`
- `src/hyperloom/orchestrator/prompts/orchestration.md`
- `src/hyperloom/orchestrator/prompts/kernel_agent.md`

Changes:

- Update output instructions to prefer concise intent payloads and avoid
  restating context already present in `SharedState`, inbox, or `analysis.md`.
- Do not add `maxLength`.
- Do not add runtime rejection for long strings.
- Do not truncate model output.
- Add tests that the schema remains permissive while the prompt guidance is
  present.

Why:

- The user-visible behavior remains flexible, while routine turns stop paying
  for repeated context summaries in model-authored fields.

#### P0.5 Configure OpenAI Reasoning Effort for Codex/Critic

Files:

- `src/hyperloom/orchestrator/roles/codex.py`
- `src/hyperloom/orchestrator/roles/critic_agent.py`
- `src/hyperloom/orchestrator/scoring/proposal_scorer.py`
- `src/hyperloom/inference_optimizer/tests/test_codex_backend.py`

Changes:

- Add an optional env-driven `reasoning_effort` to the OpenAI-protocol calls for
  Codex, Critic, and the proposal scorer, defaulting to `medium` and only passed
  for models that accept it (feature-detect; omit for models that reject it).
- Bias high-frequency simple calls (Critic reviews) toward `low` behind a
  model-compat allowlist; keep authoring/scoring at `medium`.
- Keep the existing `max_completion_tokens` limits (2000/4096) unchanged; this
  addresses the shared reasoning+output budget noted for the scorer without
  touching output caps.

Why:

- These backends also spend reasoning tokens, and the scorer's 4096 budget is
  explicitly shared between reasoning and the final JSON; steering effort is the
  matching lever for the non-Claude side.

### P1 - Cache Metrics and Conversation Continuity

#### P1.1 Fix Cache Hit Rate Measurement

Files:

- `src/hyperloom/inference_optimizer/experiments/_roofline_audit_common.py`
- `src/hyperloom/inference_optimizer/experiments/verify_roofline_v2.py`
- `src/hyperloom/inference_optimizer/experiments/audit_roofline_decisions.py`
- `src/hyperloom/inference_optimizer/breakdown/collectors/decision.py`
- `src/hyperloom/inference_optimizer/breakdown/schema.py`
- `src/hyperloom/inference_optimizer/tests/test_verify_and_audit_scripts.py`

Changes:

- Stop reading `state["tick_cache_metrics"]` as the authoritative source.
- Read `reports/trace/llm_calls.jsonl` and ext shards, using the same token
  keys already used by the breakdown collector:
  `cache_creation_input_tokens` and `cache_read_input_tokens`.
- Compute `cache_hit_rate = cache_read / (cache_read + cache_creation)`.
- Expose the derived rate in `session_breakdown.json`.
- Keep backward compatibility for old sessions that happen to have
  `tick_cache_metrics`, but do not prefer it over trace rows.

Why:

- The current audit quality gate can report 0% even when real Claude prompt
  caching is working.

#### P1.2 Make Claude Resume Downgrade Visible

Files:

- `src/hyperloom/orchestrator/roles/claude.py`
- `src/hyperloom/orchestrator/trace/llm_trace.py`
- `src/hyperloom/orchestrator/loop/coordinator.py`
- `src/hyperloom/inference_optimizer/tests/test_claude_backend_branches_unit.py`

Changes:

- When `ClaudeAgentOptions` rejects `resume=`, keep the current stateless
  fallback but emit a structured trace/warning field such as
  `resume_supported=false` or `resume_downgraded=true`.
- Surface the warning in `llm_calls.jsonl` metadata and coordinator
  observations.
- Add a test that this downgrade appears in telemetry.

Why:

- Persistent conversation and prompt caching are central to token efficiency;
  silent stateless fallback hides the root cause of bad cache behavior.

#### P1.3 Extend Delta Conversation to Claude Kernel Agent

Files:

- `src/hyperloom/inference_optimizer/cli/backends.py`
- `src/hyperloom/orchestrator/loop/conversation.py`
- `src/hyperloom/orchestrator/loop/maintenance.py`
- `src/hyperloom/orchestrator/state/orchestration_memory.py`
- `src/hyperloom/inference_optimizer/tests/test_coordinator_async_batch2_unit.py`

Changes:

- Add a Claude-kernel conversational mode behind an env flag first, for example
  `INFERENCE_OPTIMIZER_KERNEL_CLAUDE_CONVERSATIONAL=1`.
- Reuse the orchestration SEED/DELTA pattern for `kernel_agent` when its backend
  is Claude.
- Keep Codex kernel backend stateless.
- Add kernel-specific checkpoint state only if long-running kernel
  conversations prove useful; do not overload `orchestration_memory` names.

Why:

- The kernel agent can be high-frequency and currently gets full state on every
  turn.

### P2 - Input Payload Compression

#### P2.1 Compact Prompt JSON That Enters LLM Calls

Files:

- `src/hyperloom/orchestrator/roles/critic_agent.py`
- `src/hyperloom/orchestrator/prompts/specialist_prompt_builder.py`
- `src/hyperloom/inference_optimizer/breakdown/reporters/llm_prompt.py`
- `src/hyperloom/orchestrator/framework/client.py`

Changes:

- For prompt-bound JSON, replace `json.dumps(..., indent=2)` with compact JSON
  or markdown tables where readability matters.
- Keep pretty JSON for files written to disk; disk artifacts are not prompt
  token spend.
- Prioritize:
  - Critic `judge_bundle`.
  - Specialist `kb_subgraph`.
  - Specialist `warm_start_recipe`.
  - Breakdown reporter prompt payload.

Why:

- This saves input tokens without changing semantics or restricting output.

#### P2.2 Remove Duplicate Raw Recipe Text from Prompt Inputs

Files:

- `src/hyperloom/orchestrator/knowledge/cortex_t0.py`
- `src/hyperloom/orchestrator/prompts/specialist_prompt_builder.py`
- `src/hyperloom/inference_optimizer/tests/test_specialist_prompt_builder_coverage_unit.py`

Changes:

- Keep the on-disk recipe snapshot unchanged.
- For prompt injection, omit the `"raw"` field when it duplicates structured
  recipe data already present in the same payload.
- Add a test that the prompt-rendered warm-start recipe no longer carries the
  duplicate raw JSON string.

Why:

- This removes duplicated input without weakening the KB warm-start signal.

#### P2.3 Prefer Digests for Large Tool Outputs

Files:

- `src/hyperloom/orchestrator/actions/executors/report.py`
- `src/hyperloom/orchestrator/state/_shared_state/render.py`
- `src/hyperloom/orchestrator/prompts/specialist_prompt_builder.py`

Changes:

- Keep full artifacts on disk.
- Put compact digests in prompts.
- Provide explicit pull paths/tools for full content.
- Avoid automatic repeated replay of raw tool output unless a phase needs it.

Why:

- The model should see the decision-relevant facts by default, not the whole
  artifact every turn.

### P3 - Tool Surface and Prompt Shape

#### P3.1 Narrow Specialist Tool Defaults by Domain

Files:

- `src/hyperloom/orchestrator/specialists/runner.py`
- `src/hyperloom/orchestrator/specialists/domains.py`
- `src/hyperloom/orchestrator/policy/gate.py`
- `src/hyperloom/inference_optimizer/tests/test_per_domain_prompts.py`

Changes:

- Keep the current broad toolset as an explicit fallback.
- Add domain-to-tool profiles:
  - Source/patch specialists: file tools + Bash + minimal KB.
  - PR research specialists: PR monitor + Web tools + read-only file tools.
  - KB/recipe specialists: Cortex KB read tools + read-only file tools.
  - System/env specialists: Bash + Read, no PR/KB by default.
- Let `Task.allowed_tools` continue to override/narrow the profile.
- Keep PR/KB stripping when the corresponding MCP server is disabled.

Why:

- Tool schemas are repeated prompt context. Narrowing them saves tokens and
  reduces accidental tool exploration.

#### P3.2 Use SDK `tools` / `strict_mcp_config` Where Safe

Files:

- `src/hyperloom/orchestrator/roles/claude.py`
- `src/hyperloom/orchestrator/specialists/subprocess_.py`
- `src/hyperloom/orchestrator/specialists/runner.py`

Changes:

- For SDK-backed Claude roles, prefer `tools=[]` or a small explicit tool list
  where the agent should only use MCP tools.
- Set `strict_mcp_config=True` for Hyperloom-managed SDK calls after verifying
  no required user/global MCP settings are expected.
- Keep this behind tests because SDK docs distinguish `tools` from
  `allowed_tools`.

Why:

- `allowed_tools` controls auto-approval, not necessarily tool availability.
  Using the correct SDK surface can reduce tool context more reliably.

#### P3.3 Make Prompt Brevity Conditional, Not Punitive

Files:

- `src/hyperloom/orchestrator/prompts/orchestration.md`
- `src/hyperloom/orchestrator/prompts/kernel_agent.md`
- `src/hyperloom/orchestrator/prompts/critic.md` or generated critic prompt
  builder if applicable

Changes:

- Replace unconditional "be aggressive" / broad restatement patterns with
  conditional guidance:
  - Summarize only new information.
  - Reference existing session facts instead of restating them.
  - Ask for full artifacts only when digest evidence is insufficient.
- Do not penalize or reject longer content when it carries necessary evidence.

Why:

- This is token efficiency through better prompting, not hard output policing.

### P4 - Lower-Level Efficiency

#### P4.1 Reuse SharedState in Kernel Handlers Where Practical

Files:

- `src/hyperloom/orchestrator/kernel/request_handlers.py`
- `src/hyperloom/orchestrator/loop/coordinator.py`

Changes:

- Avoid repeated `SharedState.load_or_init(session_dir)` calls when the
  coordinator already owns a fresh state object.
- Start with dependency injection for hot handler paths rather than a large
  refactor.

Why:

- This is not a direct LLM token fix, but it reduces runtime overhead and makes
  prompt state ownership clearer.

#### P4.2 Add Token Optimization Regression Tests

Files:

- `src/hyperloom/inference_optimizer/tests/test_claude_backend.py`
- `src/hyperloom/inference_optimizer/tests/test_verify_and_audit_scripts.py`
- `src/hyperloom/inference_optimizer/tests/test_specialist_prompt_builder_coverage_unit.py`
- `src/hyperloom/inference_optimizer/tests/test_per_domain_prompts.py`

Changes:

- Test SDK option construction for `thinking` / `effort`.
- Test default `analysis.md` digest-only rendering.
- Test cache hit rate reads trace rows.
- Test no hard `maxLength` is introduced.
- Test specialist tool profiles are narrower than the current full default for
  representative domains.

Why:

- Token regressions are easy to reintroduce when prompt code evolves.

## Rollout Order

0. Capture a reference-session token baseline from `llm_calls.jsonl` (per-role
   input/output plus cache read/creation) before any change, so every later item
   has a fixed before/after comparator.
1. Land P0.1-P0.5 together.
2. Run focused tests for Claude backend, Codex backend, render, and prompt
   builders.
3. Land P1.1-P1.2 to make savings observable.
4. Enable P1.3 behind an env flag and compare token traces across a real
   session before making it default.
5. Land P2/P3 as incremental PRs, each with before/after prompt-size or token
   rollup evidence from `llm_calls.jsonl`.

## Success Metrics

- `session_breakdown.json` reports nonzero cache hit rate when Claude cache
  reads are present in `llm_calls.jsonl`.
- Default orchestration prompt no longer repeats full `analysis.md`.
- Specialist prompts carry fewer tool schemas for domain-specific tasks.
- Claude SDK options include `thinking={"type": "adaptive"}` and a `medium`
  `effort` default for orchestration under Python 3.10.
- No `emit_intent` hard string length rejection is added.
- No model output is silently truncated by Hyperloom.
- Final `cumulative_gain` on a reference workload does not regress under the new
  `effort`/thinking defaults versus the prior medium-effort baseline (quality
  guardrail, not a token metric).
- Each landed item carries a before/after token delta measured from
  `llm_calls.jsonl` on the same reference session, with P0.3 (`analysis.md`) and
  P1.3 (kernel delta) expected to be the two largest input-token reductions.

