# Token efficiency: measurable, reversible LLM spend reductions

## Summary

Reduces unnecessary token spend across the multi-agent optimizer **without
changing product behavior or adding hard output caps**. Every change is
env-gated or additive, reversible, and Python-3.10 compatible. Merged up to
date with `origin/main`.

The work responds to a token-usage audit. It targets the three real hotspots —
the full TraceLens `analysis.md` re-injected every tick, reasoning spend that
was never bounded on the Claude path, and a cache-hit-rate metric that was
permanently reporting 0% — plus a set of smaller, safe input-size wins.

## What changed

### P0 — dependency + high-frequency prompt spend
- **Raise minimum `claude-agent-sdk` to `>=0.2.110`** so installs are guaranteed
  to expose `thinking` / `effort` and related `ClaudeAgentOptions` fields while
  still allowing newer compatible SDK releases. Four hardcoded spec strings
  updated.
- **Env-driven reasoning effort + adaptive thinking on the Claude backend.**
  `effort` defaults to `medium` for orchestration and `low` for the
  high-frequency kernel reactor, with per-role and shared env overrides;
  `thinking` defaults to adaptive. Role is inferred from `conversational`, so no
  CLI plumbing. Gracefully degrades if an SDK build rejects the kwargs.
- **`analysis.md` no longer inlined by default.** The largest repeated input
  block is now replaced by the existing `profiler_digest` + a pointer to the
  `show_analysis_md` pull tool. Opt back in with
  `INFERENCE_OPTIMIZER_PROMPT_ANALYSIS_MD_INLINE=1`.
- **Soft output hygiene.** Prompts ask the model to send only new information
  and not restate context already in SharedState / inbox / analysis.md. No
  `maxLength`, no truncation.
- **Env-gated `reasoning_effort` for the OpenAI backends** (codex / critic /
  proposal scorer). Injected only when `HYPERLOOM_REASONING_EFFORT` (or
  `OPENAI_REASONING_EFFORT`) is set; a no-op otherwise, so models/gateways that
  reject the field are unaffected.

### P1 — cache metrics + conversation continuity
- **Fixed the cache-hit-rate metric.** It read `state["tick_cache_metrics"]`,
  which no production code ever writes, so the rate was always 0% and the
  `>=50%` gate always failed. It now aggregates the real per-call ledger
  (`reports/trace/llm_calls.jsonl` + ext shards) via a small public wrapper over
  the existing breakdown collector, and surfaces a derived `cache_hit_rate` in
  `session_breakdown.json`.
- **Resume-downgrade telemetry.** When the SDK rejects `resume=` and the backend
  falls back to a stateless turn, a `resume_downgraded` flag is now carried on
  the call metadata and written to `llm_calls.jsonl`, so silent cache loss is
  diagnosable.
- **Opt-in Claude session resume for the kernel agent** behind
  `INFERENCE_OPTIMIZER_KERNEL_CLAUDE_CONVERSATIONAL` (default off). Resume-only:
  prompt composition is unchanged (still full state per turn, no delta), because
  kernel requests are heterogeneous and a thin delta could leak stale
  cross-request context.

### P2 — input payload compression
- **Compact prompt-bound JSON** (`indent=2` → compact separators) for the critic
  judge bundle, specialist `kb_subgraph` / `warm_start_recipe`, and the
  breakdown reporter payload. On-disk artifacts keep pretty JSON.
- **Dropped the duplicate `raw` recipe** from the prompt-injected warm-start dict
  (it re-serialized the same `recipe`); the on-disk snapshot keeps it for
  envelope-shape compatibility.

## Deliberately deferred (not in this PR)
- **P3.1 (narrow specialist tools by domain)** — behavior-changing; land
  separately after the P0/P1 savings are measured.
- **P3.2 (`strict_mcp_config`)** — production specialists run as a
  `claude --print` subprocess, not through the SDK options path, so it buys
  nothing where it matters.
- **P4.1 (reuse SharedState in kernel handlers)** — the repeated
  `load_or_init` calls are deliberate cross-process freshness reads; injecting a
  cached instance would risk stale reads in adopt/rebench decisions.

## Safety / rollout
- No hard output cap is added; `max_budget_usd` remains available as the opt-in
  operator backstop.
- New behavior is env-gated (reasoning effort, kernel resume) or default-off
  where it changes what the model sees; the `analysis.md` default flip keeps the
  full report reachable via the pull tool and a `=1` opt-in.
- Suggested validation: on a reference session, compare per-role token deltas
  from `llm_calls.jsonl` before/after, and confirm final `cumulative_gain` does
  not regress under the new effort/thinking defaults (quality guardrail).

## Merge note
Merged `origin/main` (CI refactor: `ci/optimize_submit.py` → `optimize_submit_lib`
package, obsolete workflow/script cleanup). No conflicts — main's changes are
confined to `ci/` and `.github/`, disjoint from this branch's `src/hyperloom`
changes; `pyproject.toml` auto-merged (SDK minimum + main's coverage config both
apply).

🤖 Generated with [Claude Code](https://claude.com/claude-code)
