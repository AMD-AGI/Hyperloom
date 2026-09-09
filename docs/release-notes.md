---
myst:
  html_meta:
    "description": "Hyperloom release notes: headline capabilities for version 1.1.0, including the vendored KernelForge kernel-optimization agent, the merged framework optimization phase, agentic-replay grading on total token throughput, compute-partition awareness, and the single concurrency sweep."
    "keywords": "Hyperloom, release notes, LLM inference, AMD GPU, ROCm, agentic optimization, TraceLens, GEAK, KernelForge, Primus-Claw, bare metal, kernel optimization"
---

# Hyperloom release notes

The current packaged version is 1.1.0 (`pyproject.toml`). For the
per-change history since the initial snapshot, see
[`CHANGELOG.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/CHANGELOG.md),
or view a detailed breakdown of all previous Hyperloom pre-release versions under
[Releases](https://github.com/AMD-AGI/Hyperloom/releases); this page
summarizes the headline capabilities.

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
are marked below; the full per-change list is in
[`CHANGELOG.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/CHANGELOG.md).

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
  typos and renames — seven shipped examples kept passing a retired `--fellow`
  flag and ran an inferred backend while exiting 0. Both commands now fail with
  click's own error and exit 2 before any GPU work starts.

  Alongside it: `$FORGE_PATH` is removed and **not** honoured as an override
  (use `$KERNELFORGE_PROJECT_ROOT`, which defaults to
  `$USER_DATA_PATH/kernelforge`, else `~/.cache/hyperloom/kernelforge`);
  `forge-gemm-tune` is gone as a console script and as a distribution, invoked
  now as `kernelforge gemm-tune`; the `fellow` vocabulary is retired in favour of
  `kernel_backend`, and a campaign config carrying the retired key fails at load
  rather than migrating silently; and `FORGE_MAX_ITERS` /
  `FORGE_COMPILED_MAX_ITERS` are gone, having fed a cap that was never applied.
  `FORGE_` stays on the dotenv prefix allowlist, so a stale `FORGE_PATH` or
  `FORGE_DISABLE_COMPILED_FELLOWS` is still forwarded into the run and then
  ignored — the latter is now detected and warned about once per run, because an
  operator who had switched compiled kernel backends off would otherwise
  silently get them back.

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
