---
myst:
  html_meta:
    "description": "Hyperloom release notes: headline capabilities for version 0.9.0, including the agentic optimization loop, multi-agent runtime, TraceLens integration, and session artifacts."
    "keywords": "Hyperloom, release notes, LLM inference, AMD GPU, ROCm, agentic optimization, TraceLens, GEAK, Primus-Claw, bare metal, kernel optimization"
---

# Hyperloom release notes

The current packaged version is `0.9.0` (`pyproject.toml`). For the
per-change history since the initial snapshot, see
[`CHANGELOG.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/CHANGELOG.md),
or view a detailed breakdown of all previous Hyperloom pre-release versions under
[Releases](https://github.com/AMD-AGI/Hyperloom/releases); this page
summarizes the headline capabilities.

## Hyperloom public snapshot

The first public snapshot of Hyperloom combines features from the following
per-release versions:

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

- **Accuracy-gate & eval-result integrity**: lm-eval output is wired to a
  session-scoped `EVAL_RESULT_DIR`; a baseline that produces no accuracy
  verdict now hard-stops instead of optimizing an unvalidated baseline; leaked
  eval results are salvaged back from the InferenceX checkout / local-disk
  mirror; and session-breakdown attribution no longer fabricates credit from a
  seeded stack.

- **Provider-direct LLM configuration**: Hyperloom runs against Anthropic
  directly (provider-only paths), with env-driven, consistent gateway
  auth/endpoint resolution and case-insensitive Anthropic-endpoint handling;
  the Critic can run over the native provider endpoint.

- **Model-path & workload-default consistency**: A single `--model` value (local
  path or HF repo id) resolves identically across baseline, roofline, and the
  kernel agent; prompt-stated ISL/OSL/CONC/TP are honored as flags instead of
  silently defaulting; and the `model_arch` freshness guard is org-aware across
  HF-cache snapshot paths.

- **Kernel / Forge / GEAK**: forge-fusion is adopted end-to-end with hardened
  subprocess timeouts, GEAK v4 installs via `pip`/one-click, advertised kernel
  backends are aligned with the runtime, and a TraceLens-free bypass benchmark
  harness ships as a Magpie drop-in for text-gen + xDiT.

- **Framework agent**: The FRAMEWORK phase gains cross-framework rating + PR-KB
  discovery, a flag-gated config-exploration subphase, and a candidate-free
  local-exploration arm so a dry PR feed no longer wastes the phase.

### 0.8.0 highlights

- **Kernel-optimization integrity & GEAK faithfulness**: Patch-only kernel
  wins are no longer discarded by the FULL_BENCHMARK verifier (full source is
  reconstructed from the patch); multi-file (L3) kernel optimizations are
  preserved end-to-end via recorded kernel-artifact bundles; GEAK always runs
  against a freshly re-profiled TraceLens snapshot; and the kernel-candidate
  pool cap is decoupled from the dispatch budget.

- **Profiling / roofline / TraceLens**: GPU information is restored in
  profiler traces (torch-trace `"kernel"` category), and a stale TRACELENS_ROOT
  inherited from the kernel-agent env file no longer breaks TraceLens discovery.

- **Orchestrator reliability & long-run durability**: `reports/final.json` is
  now written crash-safe even on non-graceful/`time_exhausted` exits, and
  orchestrator LLM calls survive slow heavy-reasoning models (e.g. Kimi-K2.6)
  via idle-timeout + amplified retry.

- **Server config & Local-Mode portability**: sglang `--context-length` is
  clamped to the run's `--max-model-len` (no more contradictory server config),
  and Local Mode portability groundwork removes Core42 / WekaFS hard-coding (docs).

### 0.7.0 highlights

- ***Forge: a third autonomous kernel-optimization backend (new track)**:
  0.7's headline is **Forge** (**Kernel-Forge**) — a self-driving kernel-optimization
  backend that joins GEAK and OOB. It runs an autonomous edit→build→bench loop
  with kernel_kind-aware fellow routing (Triton / HIP / CK / aiter / hipBLASLt / FlyDSL),
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
  warm-replay, and Forge fellows now read cross-KB knowledge directly from the
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

- **Reliability: sandbox-hang elimination & real-cluster hardening**: LLM streaming
  reads are now bounded client-side and timed-out subprocess trees are reaped by
  process group, eliminating the "pod Running but idle" sandbox hang. Plus setup_env
  race / USER_DATA_PATH corruption fixes, user-uncommitted-change protection before
  destructive reverts, and invalid / premature zero-gain rejection.

- **CI: structural model pre-filter & throughput**: A shared model-compatibility
  pre-flight (multimodal / Gemma2 / Phi3-longrope / dual-chunk / ModelOpt-FP8 /
  FlashInfer / gated / missing-tokenizer) skips doomed models before a session is
  created, alongside HF-token rotation with 429 backoff, larger daily pools, and 48h
  long-horizon budgets.

### 0.6.0 highlights

- **Search Space: looser orchestration + long-horizon runs**: 0.6 turns the
  Orchestration theme from 0.5's "search efficiency" to **search space**: mechanical
  guardrails are downgraded to advisory so the optimizer drives itself, free-form
  and cross-domain specialist dispatch lets it explore beyond the fixed action
  catalogue, and 2–3 day long-horizon optimization with finer-grained start / stop
  / checkpoint / resume keeps long runs productive and recoverable.

- **Quantization agent (new track)**: A prompt-driven Quark quantization sub-agent
  lands as an optimization prelude, with Quark enhancements and quantization-agent
  proposals — quantization joins kernel-opt as a first-class optimization lever.

- **Knowledge Base: unified and knowledge-graph-backed**: cortex-KB and gbrain
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

- **Multi-node & scale**: Multi-node Optimus support comes online and the Arbor
  mechanism migration is completed.

- **Observability overhaul**: Live Langfuse tracing, full-trace token and conversation
  logging, phase / step-level observability, a per-session token-consumption breakdown,
  and an author-time session-breakdown recorder make long runs inspectable in real time.

- **Reliability & real-cluster hardening**: A broad sweep driven by large-scale
  cluster-run analysis: fast-fail for immediate arg / config errors, log line-buffering
  so healthy runs no longer look frozen, pod-local dependency roots decoupled from WekaFS,
  Ray raylet fd-limits, MI308X detection, attention-backend argument hygiene, GEAK container
  network path to the LLM gateway, and clearer setup / baseline failure classes.

- **Docs, licensing & coverage**: Repo-wide Google-style docstrings with a published Sphinx
  documentation site, the license relicensed **Apache → MIT**, requesting-access / SSO docs, and
  Python test coverage raised to ~91.5%.

### 0.5.0 highlights

- **Orchestration: vocabulary unification + search efficiency**: 0.5 advances the "search
  efficiency" theme ([#272](https://github.com/AMD-AGI/Hyperloom/issues/272)): the optimizer's
  action/state vocabulary converges onto the unified explore grid-runner and the EXPLORE/specialist
  fan-out gains parallel headroom. New **Atom framework support**
  ([#336](https://github.com/AMD-AGI/Hyperloom/issues/336)) and a soft **Dynamic Action** cross-domain
  deep-dive ([#335](https://github.com/AMD-AGI/Hyperloom/issues/335)) widen the search surface.

- **GEAK & kernel optimization, deeper**: **GEAK GEMM tuning**
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
  Productization & Data Maintenance** ([#333](https://github.com/AMD-AGI/Hyperloom/issues/333)) and **KB Recipe
  Ingestion** ([#332](https://github.com/AMD-AGI/Hyperloom/issues/332)).

- **Profiling, TraceLens & Dashboard**: **TraceLens 0.5** ([#358](https://github.com/AMD-AGI/Hyperloom/issues/358));
  a patched profiler docker image that captures HipGraphLaunch kernels so optimization-loop traces are
  complete ([#352](https://github.com/AMD-AGI/Hyperloom/issues/352)); **profiling information for all Hyperloom
  models** ([#346](https://github.com/AMD-AGI/Hyperloom/issues/346)); **kernel roofline on the dashboard**
  ([#345](https://github.com/AMD-AGI/Hyperloom/issues/345)); and a **Session Breakdown enhancement** spanning
  auto-collection, alerting, TraceLens/GEAK detail capture, and kernel roofline
  ([#334](https://github.com/AMD-AGI/Hyperloom/issues/334)).

- **Stability & bug fixes**: 0.5 closes a batch of orchestration / runtime defects surfaced by 0.4 runs:
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
  Framework Agent integration, Agent Kernel Arena, the Self-Evolving Skills & Memory layer ramping past
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

- **White-box visibility: Root Cause & Pending Cause Agents**: Two new supervisory agents make Hyperloom white-box
  at both ends of the user experience. The **Root Cause Agent** watches the Marathon optimization loop for failures,
  diagnoses the cause, and writes actionable guidance back into the next retry prompt — failures stop being binary
  pass-or-discard and become constrained retries. The **Pending Cause Agent** does the analogous thing on the queue
  side: when a user's run is waiting, it surfaces a concrete reason why. Together they replace two previously opaque
  states — "something failed" and "still waiting" — with explainable, user-controllable answers.
