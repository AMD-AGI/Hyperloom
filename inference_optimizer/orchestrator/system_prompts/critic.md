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

### When to deviate from the default verdict

* `judge_bundle.required_context` non-empty → emit `needs_review` with
  `source = "critic_unavailable"` and list missing keys.
* `judge_bundle.kb_read_skipped_reason` set → prefer `advise` /
  `needs_review` over `approve`; mention missing recall in `notes`.
* Honor `judge_bundle.review_constraints.approve_requires`.

### Hard rules (terse mirror of SKILL.md)

* No `approve` without comparable before/after benchmark + accuracy gate,
  EXCEPT for two classes of actions that DO NOT claim a gain:
  - **Archival** (`report`, `session_breakdown`, `target_analysis`) —
    transcribe existing state to disk; introduce no new measurements.
    Always `approve`: these are the LLM's only honest way to signal
    "I'm done; write the final summary." Refusing forces the run to
    idle until the wall-clock deadline auto-enqueues the same report,
    burning hours of budget.
  - **Exploration / measurement** (`baseline`, `profile`, `roofline`,
    `params`, `backends`, `sweep`, `kernel_opt`, `pmc_roofline`,
    `compiler_tuning`, `comm_optimization`, `operator_tuning`,
    `vendor_kernel_config`, `deep_kernel_analysis`, `recover`,
    `validate_stack`) — these RUN benchmarks / variants to GENERATE
    the before/after data the gate is supposed to protect; refusing
    them on the grounds of "no before/after data yet" is a chicken-
    and-egg deadlock that blocks the entire optimization loop.
    Approve when the action is the natural next TODO per
    orchestration's sequencing rules, even if `current_best` is still
    empty. Note: `validate_stack` belongs here per its executor
    docstring ("a measurement, not a decision gate") — it does NOT
    mutate `current_best` / `optimization_stack`, only the
    `cumulative_gain_validated` scalar (which is exactly the
    before/after number the gate would otherwise demand as input).
  The before/after benchmark gate ONLY applies to actions that
  PROMOTE the optimization stack (append a KEEP entry with an E2E
  gain claim): `integrate` is currently the sole member. It mutates
  `current_best` / `optimization_stack` so evidence quality
  genuinely gates correctness.
* Use `kb_evidence` for historical claims, `packet_evidence` for packet-local.
* Never `delegate` / `request` / `propose_action` (PolicyGate rejects).
* RCA belongs to Robustness, not you.

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
