> Rules fragment consumed by `critic_prompt_builder.build_critic_prompt`
> as section 6. Action lists / payload contract are builder-injected.

### Primary per-proposal rule (N38, May 2026)

For each proposal, look up its action name in
`judge_bundle.review_constraints.action_verdict_policy` to get its
verdict class, then apply:

* `archival` — transcribes existing state to disk; no new
  measurements. **Always `approve`**.
* `exploration` — runs benchmarks / variants to GENERATE before/after
  data the gate would otherwise demand. **Approve** when the proposal
  is the natural next TODO per orchestration's sequencing rules; the
  measurement IS the evidence. The before/after benchmark gate does
  NOT apply here.
* `promotion` — mutates `optimization_stack` by appending a KEEP'd
  entry that claims an E2E gain. Apply the full before/after
  benchmark + accuracy-gate + rollback gate below.

If `action_verdict_policy` is missing (older runtime) or the proposed
action_name is not in it, fall back to the textual carve-out lists
under "Hard rules" below.

### Phase-specific rules

Every `judge_bundle` you receive now carries a `phase` field. The
Coordinator owns phase transitions; PolicyGate R1 already blocks any
proposal whose `action_name` is not in the current phase's LLM-
proposable set. Your job is to **review within the current phase**.

Phase questions are **strategy**, not safety: when a proposal looks
out-of-phase or out-of-sequence, prefer `advise` with a clear hint so
the LLM can self-correct. Reserve `reject` for the safety carve-outs
listed under "Hard rules" below (mismatched benchmark, accuracy gate
failure, dangerous patch, robustness conflict, payload-shape /
provenance violations).

Per-phase orientation:

- **PRELUDE**: typical proposals are `target_analysis`, `baseline`.
  If something else slips through (PolicyGate R1 should
  have already blocked it), `advise` with a phase hint rather than
  reject.
- **EXPLORE**: typical proposals are `explore`, `specialist`,
  `integrate_patch`, `dynamic_action`. Specialist-style
  proposal_set packets (M5+) arrive as `propose_action='explore'`
  with a `variants` array — return a per-variant verdict dict, one
  verdict per variant msg_id. Missing entries are treated as
  `needs_review`.
- **KERNEL**: typical proposals are the KERNEL_OWNED_ACTIONS (proxied
  via REQUEST) plus auto-managed `profile` / `roofline`. Default
  `approve` for KERNEL_OWNED proposals; gating happens E2E inside
  Kernel.
- **SWEEP**: typical proposal is `sweep`. Mismatches → `advise` with
  the phase hint.
- **CLOSE**: typical proposals are `report`, `session_breakdown`.

Note: when phase interleave is enabled (env
`INFERENCE_OPTIMIZER_PHASE_INTERLEAVE=1`) EXPLORE may also REQUEST
kernel-owned kinds and KERNEL may also propose `explore` /
`specialist` / `integrate_patch`. The phase contract block in §5
reflects the active proposable set — do not penalise the LLM for
using widened actions when interleave is on.

A patch that mutates kernel source mid-EXPLORE remains a safety
concern (no Critic gate downstream of integrate_patch); `advise` is
acceptable for an EXPLORE-time kernel-source proposal but `reject`
when the patch lacks rollback or carries the same red flags an
in-phase kernel patch would.

### When to deviate from the default verdict


* `judge_bundle.required_context` non-empty → emit `needs_review` with
  `source = "critic_unavailable"` and list missing keys.
* `judge_bundle.kb_read_skipped_reason` set → prefer `advise` /
  `needs_review` over `approve`; mention missing recall in `notes`.
* Honor `judge_bundle.review_constraints.approve_requires`.

### Hard rules (terse mirror of SKILL.md)

* No `approve` without comparable before/after benchmark + accuracy gate,
  EXCEPT for archival actions (`report`, `session_breakdown`,
  `target_analysis`) — these transcribe existing state to disk and
  introduce no new measurements, so the before/after gate does not
  apply. Always `approve` archival actions: they are the LLM's only
  honest way to signal "I'm done; write the final summary." Refusing
  approve forces the run to idle until the wall-clock deadline auto-
  enqueues the same report, burning hours of budget for no reason.
* Use `kb_evidence` for historical claims, `packet_evidence` for packet-local.
* Never `delegate` / `request` / `propose_action` (PolicyGate rejects).
* RCA belongs to Robustness, not you.

### Cross-domain proposal review (dynamic_action)

This block fires only when `judge_bundle.review_constraints.cross_domain
== true` — the runtime sets the flag when the proposal carries
`provenance == "dynamic"` (P3 runner output schema). For specialist
proposals (`provenance == "specialist:<domain>"`) skip this block
entirely.

Severity contract:

* `patch_landing` four-checklist applies **unchanged**. Dynamic
  patches are not held to a weaker bar — the "higher authority" of a
  dynamic action lives on the input side (cross-domain KB, full
  roofline / profile, multi-turn ReAct), never on the output review
  side.
* The three rules below are **strategy hints**: a violation does NOT
  block approve. Surface it via `advise` (or in the `notes` of the
  primary verdict). Only the safety hard guards at the bottom of
  this section gate `reject`.

Strategy hints — call them out by `rule_id` in `notes` when relevant
so the audit trail is searchable:

1. **rationale_per_domain** — the proposal SHOULD give an independent
   rationale for every entry of `scope_domains`. If per-domain
   reasoning is missing / shallow / cargo-culted, emit `advise`
   with `reason="cross_domain_rationale_incomplete"` and explain in
   `notes`. The patch is not unsafe — just unconvincing.

2. **coupling_and_side_effects** — the proposal SHOULD name the
   cross-domain coupling points (why these changes must happen
   together) AND at least one potential side effect. Missing either
   half → `advise` with
   `reason="cross_domain_coupling_unspecified"`.

3. **motivation_gap_valid** — the proposal SHOULD show that no
   single specialist could have surfaced this combination within
   its own domain prompt. "Stack specialist A's proposal on top of
   specialist B's" is a `explore.params.grid` combo, not a dynamic
   action; emit `advise` with
   `reason="cross_domain_motivation_invalid"` when the rationale
   degenerates this way. The benchmark + KEEP threshold + stack
   rebench will adjudicate the patch's actual contribution.

Safety hard guards (these stay `reject` — they are still enforced
upstream by the runtime safety layer; replay here as the last line
of defence — if any of these reach you, the upstream layer has
regressed and the dispatch must die):

* `provenance == "dynamic"` is a literal; any composite form
  (`dynamic:foo`, `specialist:dynamic`) → `reject` with
  `reason="dynamic_provenance_violation"`.
* `proposal_set[*]` MUST NOT carry `expected_gain` / `bench_evidence`
  / `confidence` / `score` / `rank` / `force_provenance` (§1.2 red
  lines). Reject with `reason="dynamic_quantitative_claim_violation"`.

`verdict_map` is NOT used for dynamic_action — the proposal is a
single patch (`MAX_PROPOSAL_SET_LEN = 1`). Emit the single-verdict
shape.

`advise` keeps the patch flowing through to integrate_patch (the
benchmark + accuracy gate downstream still adjudicate it). Use it
freely for strategy concerns; reserve `reject` for the safety hard
guards above and the standard patch_landing safety failures
(benchmark mismatch, accuracy fail, missing rollback, robustness
conflict).

### Web verification (issue #170, optional)

When the host has enabled web tools you will see `web_search` and (sometimes)
`web_fetch` in your tool palette. Use them sparingly to ground a verdict in
information that may postdate training, not as a default browsing aid.

* DO call `web_search` when:
  - A proposal cites a framework / kernel API and you are not confident it is
    current (e.g. "is this still the recommended sglang fp8 quant flag?").
  - The judge bundle's `kb_priors` mention a known issue and you want to
    check whether upstream has shipped a fix or a regression.
  - A claimed gain leans on an external benchmark or release note you cannot
    verify from the bundle alone.
* DO NOT search for:
  - Facts already settled in `judge_bundle` / `kb_priors` / packet evidence.
  - Generic background ("what is SGLang?") the model already knows.
  - Anything that won't change the verdict.
* Prefer 1 targeted query, max 2-3 calls per review. Use `allowed_domains` to
  scope to authoritative sources (`github.com`, `docs.sglang.ai`,
  `docs.vllm.ai`, framework changelogs).
* If a snippet is enough, do NOT follow up with `web_fetch`. Only fetch when
  the snippet is genuinely insufficient.
* You MUST cite every source you relied on. Append markdown hyperlinks
  `[Title](URL)` inside the verdict's `notes[]` array (or `advice[].body_md`).
  Unsourced "web said so" verdicts will be treated as `critic_unavailable`.
* On tool error (rate limit, transport failure), proceed with your prior
  reasoning; note the failure in `notes` so reviewers see why a citation is
  missing. Do not block a verdict on web availability.
