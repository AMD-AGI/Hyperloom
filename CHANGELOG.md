# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Codex sandbox bypass uses a single env var.** Set
  `HYPERLOOM_CODEX_SANDBOX_MODE=bypass` when an external sandbox already
  enforces isolation. `HYPERLOOM_CODEX_EXTERNAL_SANDBOX` is removed from
  Hyperloom and KernelForge.

- **PR Monitor now shares the KB Store endpoint.** Hyperloom derives REST
  `${KB_STORE_URL}/pr-monitor/v1` and MCP
  `${KB_STORE_URL}/pr-monitor/mcp/` URLs for Framework discovery,
  KernelForge priors, IR-3, and specialist tools. The independent
  `PRIMUS_CORTEX_PR_API`, `--pr-monitor-url`, and `--pr-monitor-mcp-url`
  configuration paths are removed.

- **An AgentX run now grades on total token throughput under an interactivity
  constraint.** The SemiAnalysis CC corpus an agentic replay runs averages ~114k
  prompt tokens against ~810 output tokens per request, so grading on output
  throughput alone optimises about 1% of the token budget: a measured Kimi-K3
  baseline read 25978 tok/s total against 183 tok/s output. Total token
  throughput is the objective and interactivity p90 (E2E normalized
  interactivity, `OSL/E2EL`) is a veto rather than a weighted term, which is the
  shape InferenceX ranks a submission by. It is default-on under
  `HYPERLOOM_AGENTX=1`; `HYPERLOOM_PERF_METRIC` overrides in both directions
  (`composite_v1` opts a non-AgentX run in, any other value opts an AgentX run
  out), and `HYPERLOOM_PERF_NOISE_PCT` (default `5.0`) sets the veto band.
  Either AgentX signal is enough: the ambient `HYPERLOOM_AGENTX` or the session's
  persisted `benchmark_mode`, so a re-baseline or integrate round driven from a
  subprocess that never inherited the env var still grades on the agentic axis.
  Scriptable frameworks keep output-throughput grading. Candidate and reference
  are always read off the same axis: a lane whose measurement cannot supply
  both graded axes degrades to output throughput on both sides and logs the
  reason. The final report names the grading mode.

- **An AgentX measurement the scenario judged invalid is no longer selectable.**
  `submission_valid=False` always rejects. An undetermined verdict (`None`)
  rejects too unless `HYPERLOOM_ALLOW_UNVERIFIED_SUBMISSION` is set. The gate
  covers every measurement the run accepts -- baseline, explore, kernel, sweep
  -- not the baseline alone, so an unverified measurement cannot become the
  denominator of every gain that follows it.

- **The server-boot timeout default is 7200s, up from 2700s.** A 1.56 TB MXFP4
  MoE checkpoint reads for ~37 minutes before the first aiter JIT, so the
  baseline died to a timeout unrelated to the workload unless the operator
  pinned `INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC` by hand. A genuinely
  wedged server is still stopped by the per-phase and session budgets.

- **`--extra-env` reaches the benchmark for every framework, and now outranks
  the config.** The pins were copied into `benchmark.envs` only on the `custom`
  path; every other framework left them in the orchestrator's own environment,
  where Magpie -- which forwards `benchmark.envs` and nothing else -- never saw
  them, so a vLLM Ray worker booted without them.<br/>
  **Operator note**: on the `custom` path the pins used to be applied with
  `setdefault`, so a value already present in the YAML won. They are now written
  last and win outright, which is what an explicit CLI pin should do but is a
  change in precedence for a `custom` workload whose YAML sets the same key.
  Names on the untrusted-env denylist (`PATH`, `PYTHONPATH`, `LD_PRELOAD`,
  credential-shaped names) are still refused on this route and logged.

- **A shell-quoted `--flag=value` operand no longer keeps its wrappers.**
  Quote-preserving tokenization keeps a JSON blob intact, and a fully-wrapped
  operand (`--tool-call-parser 'kimi_k3'`) is already unwrapped; the `=` form
  (`--tool-call-parser='kimi_k3'`) was not, so the quotes survived into Magpie's
  unquoted `EXTRA_*_ARGS` expansion and reached argv literally. The unwrap is
  applied to the right-hand side of the first `=` only, so the flag name is never
  altered and token boundaries cannot shift; a JSON value is left verbatim in
  both positions.

- **A published Recipe now carries three columns instead of five.**
  `config`/`explore`/`framework`/`kernel`/`patch_timeline` collapse to
  `config`/`patch`/`kernel`, each owned end to end by one SDK facade
  (`ConfigKB`, `PatchKB`, `KernelAgentKB`). The `explore` and `framework`
  source overlays merge into a single `patch` column; replay order is the
  lexicographic order of the zero-padded stack/member indices in each
  `patch/overlays/<stack>/<member>-<name>.patch` ref, so `patch_timeline` is
  gone. Each overlay carries a `provenance` row.

- **The recorded apply root is now the sole authority for warm replay.**
  Each overlay records `provenance[].host_origin.apply_roots` (`{ref:
  absolute_root}`) and each kernel item records `host_origin.apply_root`, read
  back at replay to place the change into the checkout it was measured on. The
  env/allowlist root search is removed: a record that cannot name its checkout
  is skipped whole rather than applied to a tree the gain was never measured
  on. `host_origin` is the one sanitizer-exempt subtree allowed to carry
  absolute paths (secret-named keys are still dropped there).

### Fixed

- **SWEEP is one concurrency sweep, and it produces the chart a submission is
  read on.** The workload sweep over `(CONC, ISL, OSL)` is deleted. Two of its
  three axes carried nothing under an agentic replay — request shapes come from
  the trace corpus, so ISL and OSL are inert placeholders — and the concurrency
  axis is what `conc_sweep` already swept. `conc_sweep` is now the only sweep,
  on by default for both workloads, and every rung carries `intvty_p90`,
  `input_throughput` and `tpot_p90_ms` alongside the output-axis figures. The
  chart it renders follows the payload's `benchmark_mode`: an agentic run is
  plotted on p90 interactivity against token throughput per chip — the pair
  InferenceX ranks a submission by — and anything else keeps the previous
  output-throughput pair unchanged.<br/>
  **Operator note**: the default ladder is now per workload —
  `256,128,64,32,16,8,4,2` synthetic, `1,4,8,10,14,20,28` under
  `HYPERLOOM_AGENTX`, where a request carries a measured ISL p50 near 108k
  tokens and the same card saturates two orders of magnitude lower.
  `--conc-sweep-concs` still overrides both. The sweep is no longer off by
  default under AgentX, and its budget default is sized at the ladder it has to
  fund (seven rungs on each of two arms); `--conc-sweep-total-budget-sec` is
  still a ceiling the session's own remaining time clamps. The `sweep` action
  is gone from the LLM catalogue, the executor registry and the phase contract,
  and the SWEEP exit reasons `conc_sweep_done` / `conc_sweep_failed` collapse
  into `sweep_done` / `sweep_failed` with no alias for the old spelling — a
  resumed session carrying one will not map to a clean exit code.

- **Each concurrency-sweep rung is bounded by its own concurrency.** The inner
  benchmark cap, the client's `--warmup-grace-period` and the variant
  subprocess cap all derived from the session's `CONC`, so a ladder rung at 64
  was given the bound of a session sitting at 8 while having to drain eight
  times the warmup. All three now take the rung's own concurrency, and the five
  budget gates that admit a rung price it at the same number. Inert unless the
  operator has declared both `AGENTX_WARMUP_GRACE_PERIOD` and
  `AGENTX_WARMUP_GRACE_CONC`.

- **The AgentX baseline overhead is derived from the warmup bound instead of a
  flat constant.** `AGENTX_BASELINE_OVERHEAD_SEC` was a single measured number
  (7200s, calibrated on GLM-5.2/Qwen3.8) covering setup, corpus load, warmup and
  first-compile. Warmup is the share that actually varies by model, and it
  already has an operator-visible bound in the client:
  `AGENTX_WARMUP_GRACE_PERIOD`. A model whose warmup runs long is therefore a
  model whose operator has already raised that knob — a raw aiperf run against
  Kimi-K3 at concurrency 64 measured warmup alone at ~12075s, past the entire
  flat cap. The overhead is now `5400s non-warmup + AGENTX_WARMUP_GRACE_PERIOD`,
  and every input is logged at INFO so a field timeout can be read back to the
  values that produced it.<br/>
  **Operator note**: at canonical settings the cap is unchanged
  (5400 + 1800 = 7200), so nothing moves for existing synthetic or GLM-5.2-class
  runs. Raising `AGENTX_WARMUP_GRACE_PERIOD` now also raises the baseline
  timeout by the same amount — which is the point, but it means the round's
  worst-case wall clock grows with that knob. `AGENTX_BASELINE_OVERHEAD_SEC`
  still overrides the derivation outright, and the "nothing has been tuned for
  this model" warning now fires only when *neither* knob is set.

- **Overriding `HYPERLOOM_PROFILE_MAX_ITERS` under AgentX no longer lifts the
  host-RAM capture bound silently.** The AgentX branch clamps captured profile
  steps to 8 because an agentic step carries orders of magnitude more profiler
  events than the synthetic shape the normal cap is sized against — at the stock
  cap a DeepSeek-V4 round was OOM-killed mid-capture three times in a row. The
  operator override is applied afterwards and wins, which is intended, but the
  two existing warnings could not report it: `cap` defaults to 128, so the
  obvious `HYPERLOOM_PROFILE_MAX_ITERS=128` was neither below the steady-state
  floor nor above the cap and restored the full exposure without printing
  anything. The override is still honoured verbatim; it now warns.

- **`AIPERF_HTTP_TCP_USER_TIMEOUT` is re-stated after the `AIPERF_*` scrub.**
  `TCP_USER_TIMEOUT` bounds how long Linux tolerates an established connection
  making no progress, and an agentic turn against a long-context model makes
  none for as long as the server is prefill-bound. aiperf's stock 30s therefore
  aborts otherwise-live connections mid-prefill, surfacing as a warmup failure
  with no server-side error to match it. Upstream's Kimi-K3 and DSv4 recipes all
  export `900000` (15 min); Hyperloom scrubs every inherited `AIPERF_*` except
  `AIPERF_BIN`, so an operator setting it had no effect and the client ran on
  the stock bound. Now exported after the scrub, tunable via
  `AGENTX_HTTP_TCP_USER_TIMEOUT`.

- **A loosened `AGENTX_FAILED_REQUEST_THRESHOLD` is flagged as a non-canonical
  workload.** Raising the abort ratio keeps alive a run that upstream's 0.10
  would have aborted, and the surviving requests are then mapped as an ordinary
  measurement. aiperf stamps no scenario marker for it — the threshold is the
  client's own safety net, not part of the scenario — so the round came back
  `submission_valid=true`. Only a *larger* ratio is flagged; tightening it
  measures a strictly cleaner run.<br/>
  **Operator note**: a run that raises this knob is now stamped
  `submission_valid=false` with `failed_request_threshold=<v>(canonical 0.10)`
  in `submission_invalid_reasons`, and `benchmark_result.py` will refuse the
  measurement. Rounds that previously passed on a raised threshold will now be
  rejected — which is the intended correction, not a regression.

- **The AgentX warmup bound scales with concurrency, and both layers read the
  same number.** The client builds warmup as `CANON_WARMUP_PER_LANE` requests
  per lane across `CONC` lanes, so the work is linear in concurrency by
  construction, while `AGENTX_WARMUP_GRACE_PERIOD` is one flat number — a grace
  measured at one concurrency under-budgets every higher one (measured on
  Kimi-K3: conc=8 → 87 warmup requests ~3000s; conc=16 → 177 requests ~5000s).
  The grace is now scaled by `CONC / AGENTX_WARMUP_GRACE_CONC`, and the scaling
  lives in one function that both consumers call: this process derives the
  subprocess cap from it, and `apply_agentx_switch` exports its result into the
  benchmark env so the client's `--warmup-grace-period` — the thing that
  actually stops the warmup — cannot disagree with the cap.<br/>
  **Operator note**: `AGENTX_WARMUP_GRACE_CONC` declares the concurrency the
  grace was measured at and defaults to 8, so every existing configuration
  derives exactly what it derived before. Declare it when you measured
  elsewhere — the scaling is a ratio, and a 14400s grace measured at conc=16
  passed in without the anchor is read as an 8-anchored number and doubled.
  The floor only ever raises a bound.

- **Budget admission prices a variant at the cap it will actually be granted.**
  Four gates (`_skip_rest_for_budget` and three in the conc sweep) plus the
  sweep's session soft deadline compared the remaining budget against the
  *declared* `variant_timeout_sec`. Under AgentX the round is granted the raised
  cap instead, so a variant was admitted that the budget could not pay for, had
  its timeout clamped back to the remaining time, and died mid-warmup — the
  exact failure the cap-raise exists to prevent. All five now use the raised
  cap; with AgentX off the helper is the identity and the synthetic path prices
  and paces exactly as before.

- **An AgentX benchmark timeout is never lowered below what the config
  declared.** The inner-timeout raise was an unconditional assignment, so a
  config declaring more than the AgentX derivation had its timeout cut
  (`profile_sglang.yaml`'s 14400s became 10800s). It now takes the maximum and
  logs when the config's own number wins.

- **The AgentX client holds the server connection open, and validates its
  numeric knobs.** `AIPERF_HTTP_TCP_USER_TIMEOUT` gave the client a 900s
  tolerance, but nothing raised the server's keep-alive (vLLM defaults to 5s),
  so the server closed idle connections mid-warmup and the round failed with
  `ServerDisconnectedError` after a full weight load. The wrapper now defaults
  the framework's own knob (`VLLM_HTTP_TIMEOUT_KEEP_ALIVE` /
  `SGLANG_TIMEOUT_KEEP_ALIVE`) to the same tolerance, overridable via
  `AGENTX_HTTP_KEEP_ALIVE_S` and never overwriting an explicit setting.
  Separately, `AGENTX_FAILED_REQUEST_THRESHOLD` was interpolated into an awk
  program body, making its value executable; the three measurement knobs are now
  validated as numbers and the comparison passes them through `awk -v`.

### Added

- **Session breakdown exports now include the additive V6 startup contract.**
  The existing V5 payload remains intact while `metadata`, `outcome`,
  `timeline`, and `close` provide the V6 read model. Install and model-gate
  source events use one ordered timeline ledger that preserves fresh and resume
  attempts, and write failures are surfaced through `metadata.warnings`.

- **KernelForge now ships inside Hyperloom as the built-in kernel-opt agent.**
  Its source was snapshotted from `AMD-BRAIN-Internal/KernelForge` at
  `85b49f2f` (upstream `main`, PR #53 included) into `src/kernelforge/`;
  Hyperloom is the sole source from here on. The three former top-level
  packages collapsed into one: `kernel_agents` -> `kernelforge`, `forge_llm` ->
  `kernelforge.llm` / `kernelforge.agent_backends`, `forge_gemm_tune` ->
  `kernelforge.gemm_tune`. forge keeps its own CLI (`kernelforge`, invoked as
  `python -m kernelforge.cli`), and the orchestrator's kernel-agent dispatch
  path is unchanged, including `KERNEL_OPT_BACKEND_ORDER`, which still selects
  between the forge and geak backends exactly as before.

  Its knowledge base, examples and serving patches moved inside the package as
  `kernelforge/data/` and now ship in the wheel, so `resource_path()` resolves
  them from an installed distribution rather than from a checkout. It raises
  `FileNotFoundError` on a missing resource instead of returning a path that
  does not exist, and runtime state that used to be written next to those
  resources goes to a writable root instead of into `site-packages`.

  Two things in the snapshot did not come across. The `intellikit` kernel
  backend is removed: nothing in Hyperloom could reach it -- `infer_kernel_backend`
  has no arm for it and the dispatch path only ever passes triton/flydsl/ck/aiter
  -- and its author confirms it is no longer needed. Its `languages/asm/`
  knowledge tree (117 files, a vendored copy of `ROCm/intellikit-asm-skills`
  plus CDNA4 ISA extracts) went with it, being reachable from no other backend.
  Eight kernel backends remain: CK, FlyDSL, Triton, Gluon, AITER, HIP,
  hipBLASLt, and the fusion backend. `deploy/` is also absent -- every file in
  it targets the retired repository.

- **`scripts/partition_mode_sweep.py` measures which compute-partition mode a
  workload wants.** Sets each mode on one card in turn, runs the same benchmark
  on every partition that mode creates, sums the throughput, and restores the
  card's entry mode on the way out — including after a failure or a Ctrl-C.
  Modes whose partitions provably cannot hold the configured streams are skipped
  with the arithmetic shown rather than run into an out-of-memory failure.<br/>
  The fan-out is the substance of it. A benchmark that loads one partition and
  ignores the rest measures a fraction of the card, which reports `CPX` as eight
  times worse than it is; every figure here is the sum over a mode's partitions
  with all of them loaded together, and a mode is reported only when every one of
  its partitions returned a measurement. Partitions are selected by matching CU
  count within the swept card's PCI bus, never by device index: `amd-smi` orders
  by PCI address while HSA/HIP enumerates whole cards first, so on an 8-card
  MI355X node with card 0 in `CPX` the two tools disagree about which devices the
  partitions are — 0-7 against 7-14.<br/>
  This is where the privileged `amd-smi set` lives, and the only place it does.
  A card-wide mutation that evicts every GPU context is reasonable between
  benchmarks in a script an operator ran on purpose, and unreasonable inside an
  optimization loop that also runs agent-authored code, so `optimize` continues
  to only read the mode. Together the two halves are a boundary: the sweep
  chooses the shape, the session asserts it.<br/>
  Because that set evicts work, the check standing in front of it fails closed:
  an `amd-smi` process listing in a shape the parser does not model is a refusal,
  not an empty one, since the only wrong answer that destroys anything is reading
  a busy node as free. It is scoped to the card being swept, so a neighbour's
  benchmark on a shared node no longer forces `--allow-busy` and with it the loss
  of the guard on the target card. Every exit from a started sweep runs the
  restore and the report, including on an error the script does not model — which
  exits `4`, keeps the modes already measured, and still yields `3` if the card
  could not be put back.
- **The card's compute-partition shape is now recorded, checked, and published.**
  An MI300-series card can be split into independent partitions (`SPX`, `DPX`,
  `QPX`, `CPX`), and splitting one trades per-request latency for aggregate
  throughput. Until now nothing in a session recorded which shape a
  number came from, so two runs of the same configuration on the same card in
  `SPX` and in `CPX` were indistinguishable in the history — different
  experiments filed under one name.<br/>
  The observed mode now goes into the platform fingerprint alongside NPS, the
  session report names it on partitioned runs, and the shape is published to
  the environment for the benchmark entrypoint to fan work out across
  partitions. That entrypoint lives outside this repository, so until it reads
  them a session on a split card measures one partition rather than the total;
  the recorded shape is still what stops a `CPX` number being filed as though
  it were `SPX`. The published variables are set only for the scriptable
  frameworks whose benchmarks can fan out; a serving session records the shape
  but is handed no fan-out contract, and its report says the figure cannot be
  read as an aggregate.<br/>
  Two optional flags configure it. `--compute-partition-mode` **asserts** the
  mode the card is already in and refuses the session if it is in another one,
  or if the card cannot be read — the flag exists to catch an external set that
  did not take, so an unverifiable assertion is treated as a failed one.
  `--streams-per-partition` (default `2`) is how many concurrent streams go on
  each partition; a value below `1` is refused rather than quietly replaced by
  the default, since `0` is far more likely to be a mistake than a request.<br/>
  **The optimizer does not change the mode.** Setting it is privileged and
  disrupts every process holding a GPU context, which is not something an
  optimization loop should do between benchmark rounds. The card must be in its
  mode before `optimize` starts: the shape is checked and recorded at launch, so
  a mode applied later — by the benchmark entrypoint, for instance — is too late
  to be either. Nothing added here needs privilege: every probe is an
  unprivileged read, and a host without `amd-smi` behaves exactly as before.<br/>
  **Operator note**: launch now refuses a session whose streams provably will
  not fit one partition, sized from the checkpoint's weight bytes as a lower
  bound. The arithmetic costs milliseconds and the failure it replaces is an
  out-of-memory crash hours in. When the checkpoint cannot be sized the session
  runs and says so. The refusal applies where streams will actually share a
  partition — a scriptable framework, or an operator who named the flags — and
  not to a serving session that merely happens to start on a card someone else
  left split. Multi-node sessions record no shape, since the readable card is
  not the benchmark's.

### Changed

- **BREAKING: `forge-loop` and `forge-rewrite-by-flydsl` now reject an undeclared
  option instead of dropping it.** These two were the only tolerant entry points
  in the forge CLI: an option they did not declare was discarded, named on
  stderr, and recorded as `ignored_cli_options` on the result document, and the
  run proceeded on the defaults. The exemption existed because a consumer in a
  *separate repository* drove them and could ship ahead of the installed
  producer; vendoring put producer and consumer in one tree and one wheel, so
  that skew can no longer occur. What the tolerance still absorbed was typos and
  renames — silently. Seven shipped examples kept passing a `--fellow` flag after
  the `fellow` -> `kernel_backend` rename and ran an inferred backend instead of
  the intended one, exiting 0 the whole time; contrast the fusion wrapper's
  `--llm-model` -> `--model` rename, which `forge-fuse` rejected outright and
  which was therefore found and fixed. Both commands now behave like every other
  forge subcommand — click's own error, exit 2, before any GPU work starts, with
  a "Did you mean" suggestion. `kernelforge/cli_forward_compat.py` and the
  `ignored_cli_options` result field are removed; nothing in Hyperloom read that
  field. The retired `--max-iters`, previously accepted and ignored, is now
  rejected too.

- **BREAKING: `$FORGE_PATH` is removed, not demoted.** Installing Hyperloom
  installs forge, so there is no checkout to point at and nothing to clone:
  `local_setup.sh` no longer clones the private KernelForge repo (and the
  quick-start Dockerfile no longer needs an SSH mount for it), and `install.sh`
  no longer pip-installs forge as a separate distribution from a checkout — it
  verifies that `kernelforge.cli` and `kernelforge.fusion` import instead.
  Vendor-playbook resolution, the serving-patch root and the gemm-tune root now
  read the packaged copy, where they previously failed or skipped.<br/>
  **No code reads `$FORGE_PATH` any more.** An earlier draft of this entry said
  it still worked as a deliberate override; that was true of an intermediate
  revision and is not true of what shipped. Every value it could hold pointed at
  the pre-inlining repository layout, so honouring it would have shadowed the
  packaged tree with an archived one. Because `FORGE_` remains on env_safety's
  dotenv prefix allowlist, a stale setting is still forwarded into the run and
  then ignored — silently, which is why it is called out here. The dev override
  that replaces it is **`$KERNELFORGE_PROJECT_ROOT`**: a writable root holding
  `knowledge_base/`, `serving_patches/` and the other resource trees, taking
  precedence over the packaged copy when the tree it names exists. It defaults
  to `$USER_DATA_PATH/kernelforge`, else `~/.cache/hyperloom/kernelforge`.

- **BREAKING: `forge-gemm-tune` is gone as a console script and as a
  distribution.** The tuner is now the `kernelforge.gemm_tune` subpackage of the
  Hyperloom wheel, invoked as `kernelforge gemm-tune` (or
  `python -m kernelforge.cli gemm-tune run`). There is no subtree left to
  `pip install` on its own, and `FORGE_GEMM_TUNE_ROOT` no longer resolves one.
  `install.sh` now treats a missing `gemm-tune` subcommand as a fatal incomplete
  install rather than a warning, because it ships in the same wheel as
  everything else the script just verified.

- **BREAKING: the `fellow` vocabulary is retired.** "Kernel backend" in prose,
  `kernel_backend` in code. Concretely: the CLI flag is `--kernel-backend`
  taking a bare name (`triton`, not `triton-fellow`); the campaign-config key is
  `kernel_backend`, and a config carrying the retired key **fails loudly at
  load** rather than migrating silently; the environment variable is
  `FORGE_DISABLE_COMPILED_KERNEL_BACKENDS`.<br/>
  The CLI flag was the one place where the failure was *not* loud on its own:
  `forge-loop` still tolerated unknown options at the time, so `--fellow
  triton-fellow` was dropped with a warning and the campaign proceeded on an
  inferred backend. That tolerance is removed in this same release (see above),
  so the flag now fails like the config key does. The seven
  shipped `run_example.sh` that still passed it are fixed, and the rename guard
  that should have caught them — its exemption globbed `data/*` rather than
  `data/*.md`, so it was exempting runnable scripts along with the prose it
  meant to protect — is narrowed.<br/>
  `FORGE_DISABLE_COMPILED_FELLOWS` has the same forwarded-then-ignored hazard as
  `$FORGE_PATH`, and a worse consequence: an operator who had switched compiled
  kernel backends off would silently get them back. It is not honoured, but it
  is now detected and warned about once per run.

- **BREAKING: the post-KEEP confirmation round is removed.** An `explore`
  variant and an `integrate_patch` candidate were each re-benched once more
  after they had already been graded, and the second measurement overwrote the
  first as the reported number. Both now report the round that graded them.
  - `explore` measured that round as a third run on the server its warmup and
    decision rounds had already warmed, so it carried more cache than the round
    it overwrote — and the inflated value became the anchor the next in-batch
    variant was graded against. Removing it takes the bias out of the reported
    gain and saves a full benchmark per KEEP.
  - `integrate_patch` measured it on a server of its own, so removing it costs
    two things and they are worth stating: a patch that only cleared the bar on
    one measurement is no longer asked to clear it again before being committed
    to the framework tree, and `delta_pct` is now read off the same measurement
    that selected the patch, which reads higher than an independent re-measure
    would.
  - GEAK's same-harness revalidation dispatched an `explore` that inherited the
    confirmation round. It now measures like every other explore, so its
    throughput is graded colder against the engagement and current-best gates:
    expect more `fallback` (2a harness replay) and `no_promote` verdicts.
  - **Removed from the session record:** the `KEEP_UNSTABLE` outcome, the
    `keep_unstable_in_stack` result key, and the `stack_rebench_tput` /
    `stack_rebench_workspace` / `stack_rebench_warnings` fields. Readers of
    `keep_unstable_count` stay so a session recorded before this change still
    renders. `cumulative_gain_validated` now records `e2e_decision_round` as
    its measurement basis for explore promotions.
  - `enable_stack_rebench` and `rebench_stable_threshold_pct` are no longer
    read from task params.

- **BREAKING: the EXPLORE phase is merged into FRAMEWORK_AGENT.** The chain is
  now `PRELUDE → FRAMEWORK_AGENT → KERNEL_AGENT → SWEEP → CLOSE`. Configuration
  search and source/upstream landing are two arms of one phase, worked in
  parallel; the phase advances only when both are dry. One arm plateauing
  raises `switch_bottleneck` for the next macro-cycle instead of ending the
  phase while the other lever still pays.
  - **`--no-explore` is removed** rather than aliased. The two arms cannot be
    disabled separately, so the flag's new meaning would be strictly wider
    than the one an operator script asked for; an unrecognised argument says
    so where a silent widening would not. Use `--no-framework-agent`.
  - `--max-minutes-explore-pct` / `--phase-budget-explore-pct` are aliases for
    the framework budget option. The merged phase's default share is `0.40`,
    against `0.50` for KERNEL_AGENT.
  - Exit reasons `explore_*` and `framework_agent_*` are replaced by
    `optimize_no_more_leverage`, `optimize_phase_budget_exhausted` and
    `optimize_budget_cap`.
  - **A session recorded at `EXPLORE` cannot be resumed by this build.** Its
    phase names a machine that no longer exists, and starting over would
    re-run PRELUDE on top of its baseline and KEPT stack, so the Coordinator
    refuses at startup. Archived sessions still *read* — the attribution and
    recorder paths understand the old labels — they just cannot be continued.

- **BREAKING: the `framework_agent` action is retired.** Upstream PRs land
  through `integrate_patch` with `patch_source='upstream_pr'`, the same action
  and the same apply / vet / bench / KEEP-REVERT pipeline every other patch
  source uses. `runs/framework_agent/<task_id>/` is no longer produced; PR
  candidate workspaces are under `runs/integrate_patch/<task_id>/`.

- **BREAKING: `pr_intel_specialist` is replaced by
  `candidate_discovery_specialist`,** which owns finding, ranking and judging
  upstream candidates rather than being an occasional PR top-up.

- **Gain is attributed by lever, not by phase.** Both arms run inside one
  phase, so the phase that was live when a KEEP landed no longer says which
  lever moved it. `lever_kind` is the attribution key, read from what a
  specialist delivered rather than from what its mandate asked for, and
  `attribution.lever_breakdown` splits validated gain by it. The values are:
  - `config` — server args / envs; nothing on disk is touched.
  - `source_patch` — a diff a specialist wrote for this session.
  - `upstream_pr` — a diff fetched from an upstream pull request.
  - `enablement` — graded on runnability and the accuracy floor, not throughput.
  - `kernel` — a tuned or authored kernel, graded on the end-to-end bench.

  Gain that carried no stamp lands under `unattributed`; a non-zero figure
  there is a tagging gap, not a category.

### Fixed

- **A baseline round no longer OOMs against a server a prior sweep/explore
  round's timeout left orphaned.** `BaselineExecutor`'s pre-start cleanup
  ran only on the double-run path, and only when the reuse port answered
  `/health` with no matching pid/json metadata (a "zombie" heuristic). That
  heuristic was unreliable either way: an eligible `server_lifecycle` port
  is a freshly OS-assigned ephemeral port confirmed free at assignment time,
  so it always reported unhealthy and the cleanup never fired even when a
  same-port zombie was in fact present; an ineligible port fell back to a
  fixed default that could coincide with an unrelated co-tenant's server,
  risking the opposite failure. Pre-start cleanup now runs unconditionally
  before every baseline round (double-run and single-round alike -- the
  latter being the common way the kernel phase re-establishes its
  baseline), reaping any lingering server via the same `_kill_stale_servers()`
  `/proc` scan already used elsewhere: Hyperloom's own scheduling
  (`gpu_research_lane`, capacity 1) guarantees at most one server-holding
  task runs at a time, so nothing matching should be alive at this point
  regardless of port health. `conc_sweep` and `explore` -- the two actions
  whose timed-out rounds most often leave one of these orphans -- now also
  reap any lingering server once they themselves finish, shrinking the
  window an orphan can sit on the GPU before the next baseline attempt.
  `_kill_stale_servers()` itself is now scoped to our own GPU allocation
  when one is known (`ROCR_VISIBLE_DEVICES` et al set by an operator that
  carved us a subset of the machine's cards): a matching process is only
  reaped when its own visible-GPU mask overlaps ours, and a candidate whose
  mask cannot be read or declares none at all is left alone rather than
  reaped, so it can no longer touch a co-tenant's server parked on a
  different subset of the same machine. (AMD-AGI/Hyperloom#1354)
- **GEMM tuning no longer discards the MoE dispatch key.** `gemm-tune run`
  derived its demand file only when the serving log carried dense tuned-config
  misses, so a MoE-only model -- or one whose dense tables all hit while
  `fused_moe` missed -- threw away the dispatch tuple the log had recorded.
  `fmoe_ck` then skipped itself for want of evidence that was in the log all
  along. A log with either kind of demand now produces a demand file. (Ported
  from KernelForge #53.)
- **Dense GEMM shape selection reads the demand file, not the precision label.**
  The router was handed a boolean saying a demand file existed and inferred the
  operator set from the precision label instead; it now receives the parsed
  report, which names the tables the runtime actually consulted. The file is
  parsed once and shared with the coverage-gap report. (Ported from
  KernelForge #53.)
- **A token-restricted tuner now gets `token_hint` as well as `tokens`.**
  Setting only `tokens` erased the distinction between "this is the allowed
  set" and "this is the coverage sweep", which every run has, so paths starting
  from runtime-observed tokens could not tell the two apart. (Ported from
  KernelForge #53.)

- **rocprof-compute's Python dependencies were never installed.** `install.sh`
  claimed they arrived with the KernelForge root install; they were in that
  project's `profiling` extra, which the install never requested. They now ship
  as the `forge-profiling` extra and are installed explicitly. The same step was
  gated on the presence of a KernelForge checkout, which after vendoring would
  have become a permanent skip — it is unconditional and fail-soft now.

- **`COVERAGE_RELAX_FAIL_UNDER` never did anything.** `tests-coverage.yml` read
  the variable in two scripts but never mapped `vars.*` into their step
  environments, so the coverage gate was always strict regardless of the
  setting. Both steps now map it.

- **Test trees were shipping in the wheel.** setuptools defaults
  `include-package-data` to true for `pyproject.toml` config, which sweeps every
  file under a package directory — so `packages.find.exclude` dropped `*.tests`
  from the package list and the sweep re-added the same files as package data
  (627 test entries before this change). Explicit `package-data` declarations
  are now the only source of shipped non-module files.

- **The upstream-PR arm was gated shut at dispatch.** A PR candidate is
  pre-screened by the Critic before any specialist exists, so its task carries
  a candidate id and no `specialist_task_id` — and every enforcement point read
  only the latter. The verdict was never recorded, PolicyGate denied the
  dispatched row as if its params had been forged, and the dispatch reconcile
  could not re-queue it. A patch's review subject is now resolved in one place
  (the specialist task id for an authored patch, the candidate id for a
  pre-screen) and `specialist_patch_verdicts` is keyed by it. The executor's
  upstream-PR lane also gained the pre-side-effect verdict check the specialist
  lane already ran.
- **The framework accuracy gate never passed.** `_bench_candidate` read a
  `result_dir` field that does not exist on `VariantResult`, so the eval parse
  searched the process CWD, found nothing, and blocked every KEEP with
  `accuracy_unavailable_reject` whenever a baseline accuracy existed.
- **Untrusted diffs reached `git apply` unvetted.** `vet_patches` runs at
  authoring time inside the specialist runner, so patches supplied directly —
  including every upstream PR diff — were never structurally checked.
  `patch_escapes_tree` also missed absolute paths in headers without the
  conventional `a/` prefix.
- **The authored-lane retry state did not survive a resume**, letting the
  re-author cap be re-spent once per resume.
- **Seven Coordinator-internal enqueues took no lane lease,** launching servers
  and benchmarks without `server_lifecycle` / `benchmark_lane`; the enablement
  build probe took the research lane instead of its own kind's.
- **The stack-rebench floor could exceed the KEEP gate it confirmed** from
  macro-cycle 2 onward, rejecting variants the same round had admitted.
- **A session's `--no-eval` was silently overridden** on the framework patch
  lane.

- **`canonical_fingerprint` now uses pair-aware arg normalization.**
  The previous implementation sorted all arg tokens as a flat list, which
  destroyed the flag→value binding: `--max-num-seqs 128 --max-model-len 4096`
  and `--max-num-seqs 4096 --max-model-len 128` produced the same fingerprint
  and were incorrectly treated as duplicates by the `explore_search` dedup
  ledger.  Args are now parsed into sorted `(flag, value)` pairs with
  last-wins semantics for repeated flags, matching the semantics of
  `_shell_safe_dedupe`.<br/>
  **Operator note**: this changes the hash for any variant whose `extra_args`
  contains at least one flag with a value.  All fingerprint keys already
  persisted in `explore_search.tested`, `accepted`, `rejected`, and
  `name_index` inside `state.json` are invalidated.  On the next resume the
  session will re-bench its full explored history.
- **`--max-latency-ms`: a latency constraint on every KEEP.** The optimizer
  maximized `output_throughput` and nothing else; latency was measured,
  reported and fed to the prompts, but no latency number could block a
  promotion. That is survivable for a lever that raises throughput without
  touching per-request latency, and unsafe for any lever that raises throughput
  *by* making each stream slower — against a throughput-only gate, such a lever
  does not merely tolerate a latency regression, it selects for the largest one
  available.<br/>
  The flag names a ceiling on mean end-to-end latency in milliseconds. It is a
  constraint rather than a target, so it sits outside the `--target-*`
  mutually-exclusive group and combines with them. Enforcement is at
  `_lift_to_current_best`, the single choke point that writes `current_best`, so
  it holds for explore, kernel, framework, specialist and integrate winners
  alike rather than only for the lane that happened to be wired first. Explore
  also applies it a round earlier, which spares an over-budget variant a
  stack-rebench it cannot survive. Where a rebench supersedes the decision
  round, the gate grades the rebench's latency, because that is the measurement
  the headline throughput comes from.<br/>
  **Off by default**, which leaves KEEP behaviour exactly as it was. When set,
  the gate **fails closed**: a candidate that reported no end-to-end latency is
  refused, since an unmeasured constraint is not a satisfied one. Operator note:
  this makes end-to-end latency effectively mandatory for promotion under a
  budget — a lane that never times its candidates will not promote one. Refused
  winners are recorded in `latency_refusals` and listed in the report, so a
  constrained session that ends near its baseline can be told apart from one
  that simply found no headroom. A baseline already over budget warns at
  promotion time rather than failing, since it is the reference the run is
  measured against, but it does mean nothing will be kept until a candidate
  comes in under the ceiling.

## [v1.0.0] - 2026-08-26
Current packaged version (`pyproject.toml`). See
[release notes](docs/release-notes.md) and the
[GitHub release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0)
for the user-facing summary.

### Removed

- **`stop_ray_if_owned` and the ownership return value of `ensure_ray_cluster` are gone.**
  `stop_ray_if_owned` was introduced alongside `parallel_e2e_runner.py` and was
  called exclusively by `_stop_ray_via_helper` in that script. When
  `parallel_e2e_runner.py` was retired in `c92784cbf`, the helper was deleted but
  `stop_ray_if_owned` was left behind with zero production call sites, no test
  references, and no `__all__` or documentation contract. `ensure_ray_cluster`
  returned the ownership flag only for that pair; with the pair gone the return
  value had no consumer. The function is deleted and the signature narrowed to
  `-> None`. The standard deployment path starts a long-lived shared head via
  `install.sh`; `ensure_ray_cluster` connects to it and returns immediately, so
  nothing that previously ran after a `False` return changes behaviour.

- **The `reference_envs` filter inside `materialize_config_with_envs` is gone.**
  The only writer of that mapping is `cli/bootstrap.py` via
  `reference_script.parse_reference_script`, which already filters every key
  through `is_allowed_external_env_key` — strictly stronger than the
  `valid_env_key` shape check applied here. The pass dropped nothing in
  production and its warning log could never fire.

- **BREAKING — the two tool-free LLM source tiers are gone**, along with
  `HYPERLOOM_LLM_SOURCE_PROVIDER` and `HYPERLOOM_LLM_SOURCE_PREVIEW` and the
  `llm_fallback_*` reason codes they produced. A per-kernel fallback picking a
  path off a grep shortlist and a whole-table pass auditing the result were both
  shown a prompt assembled in advance, so neither could see what a kernel
  actually is — and the deterministic tiers' failure mode is not coming up empty
  but coming up confidently wrong, which ranking paths by keyword cannot tell
  apart. One tool-enabled review session on the agent analysis route replaces
  both: it is handed locations rather than contents and opens what the evidence
  leads it to. `HYPERLOOM_LLM_SOURCE_MODEL` still selects the model.
  **Operators on `--analysis-route deterministic` should note that route now has
  no model assistance of any kind** — it keeps its no-model guarantee by not
  reaching the stage, so a kernel the curated, trace-launcher and grep tiers all
  miss stays unresolved instead of being completed by a model.

- **`FORGE_MAX_ITERS` and `FORGE_COMPILED_MAX_ITERS` are gone**, along with the
  `--max-iters` this repository put on every `forge-loop` and
  `forge-rewrite-by-flydsl` argv. KernelForge deleted the option: its campaigns
  are bounded by `--max-hours`, and the flag had already been documented there
  as accepted-and-ignored. The compiled/ASM fellow cap those variables fed was
  therefore a no-op that logged a cap it never applied. `--max-hours` and the
  hard-kill timeout remain the only budget controls, exactly as before.

### Changed

- **`canonical_fingerprint` now uses pair-aware arg normalization.**
  The previous implementation sorted all arg tokens as a flat list, which
  destroyed the flag→value binding: `--max-num-seqs 128 --max-model-len 4096`
  and `--max-num-seqs 4096 --max-model-len 128` produced the same fingerprint
  and were incorrectly treated as duplicates by the `explore_search` dedup
  ledger.  Args are now parsed into sorted `(flag, value)` pairs with
  last-wins semantics for repeated flags, matching the semantics of
  `_shell_safe_dedupe`.<br/>
  **Operator note**: this changes the hash for any variant whose `extra_args`
  contains at least one flag with a value.  All fingerprint keys already
  persisted in `explore_search.tested`, `accepted`, `rejected`, and
  `name_index` inside `state.json` are invalidated.  On the next resume the
  session will re-bench its full explored history.

- **`force_restart_local_cluster` now routes its `ray stop` through `_stop_ray_force`.**
  The function previously inlined its own `subprocess.run(["ray", "stop", "--force"], ...)`
  without a timeout or `OSError` guard, meaning a hung `ray stop` on the
  version-mismatch recovery path would block indefinitely. `_stop_ray_force`
  already enforces `DEFAULT_RAY_STOP_TIMEOUT_SEC` (30 s, overridable via
  `HYPERLOOM_RAY_STOP_TIMEOUT_SEC`) and swallows both `TimeoutExpired` and
  `OSError`, so the timeout constant now covers all three stop sites instead of
  only one. Log output is unchanged: `_stop_ray_force` appends the stop command
  and any timeout note to `log_path` in the same order as before.

- **Multi-node SSH forwarding now uses the shared env-safety definitions.**
  `multi_node/_internal/env_safety` declared its own nine-name `_DENY_KEYS` set
  and its own copy of the POSIX key-shape regex. The denylist was missing
  `CDPATH`, `GIT_SSH_COMMAND`, `NODE_OPTIONS`, `PERL5OPT`, `PYTHONSTARTUP`,
  `PYTHONINSPECT`, `PYTHONUSERBASE` and `SHELLOPTS`, all of which a
  shell-launched remote pod is exposed to. Both local definitions are deleted in
  favour of `BLOCKED_UNTRUSTED_ENV_NAMES` and `valid_env_key`. Forwarding is
  unaffected: `_collect_forward_env` builds its mapping from a prefix allowlist
  plus four hardcoded names, none of which are in the blocked set.

- **`BLOCKED_UNTRUSTED_ENV_NAMES` and `BLOCKED_CHILD_ENV_NAMES` no longer list
  `DYLD_INSERT_LIBRARIES`, `DYLD_LIBRARY_PATH`, or `RUBYOPT`.** This is a
  ROCm/Linux-only repository with no macOS platform code and no Ruby tooling, so
  those three blocked nothing real. Every remaining name corresponds to a process
  this repository actually spawns: bash benchmark wrappers, Python subprocesses,
  the glibc dynamic loader, git, and the Node.js-based agent CLIs. `PERL5OPT`
  stays because `moreutils` (`ts`) is a perl program the benchmark wrapper's
  timestamped logging shim pipes through.

- **`--continue-kernel-after-gemm` is now `--auto-kernel-opt`.** The switch gates
  the KERNEL-entry source-level `kernel_opt` dispatch, which runs on both entry
  routes — tuning GEMM shape tables and rewriting kernel source are unrelated —
  so the old name described a dependency that does not exist and read as a no-op
  on a run that never tunes GEMM. The old spelling still works and still opts
  out, with a `DeprecationWarning`; the current flag wins when both are passed.
  `SharedState.continue_kernel_after_gemm` became `auto_kernel_opt_enabled`
  (state schema v6, migrated on load, so a resumed opt-out keeps opting out).
  The switch covers that dispatch only: orchestration can still request
  `kernel_opt`, and the forge-fusion and collective lanes keep their own gates.

- **The hot-kernel dispatch floor defaults to 5% of GPU time, was 10%.** On a
  decode trace with a flat kernel distribution nothing but a graph-launch
  wrapper reaches double digits, so the 10% floor admitted no real kernel and
  left the batch dispatcher idle while the orchestrator picked candidates one at
  a time. Expect more candidates dispatched per run, and correspondingly more
  GPU time spent in KERNEL. `HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT` overrides it.

- **The fusion wrapper passes `--model` to `forge-fuse`, not `--llm-model`.**
  KernelForge renamed the option to match the spelling the rest of its CLI
  already used, and `forge-fuse` rejects an unknown option outright rather than
  ignoring it, so every fusion run was exiting 2 before it started and
  surfacing as a missing `fusion_manifest.json`. The `llm_model` key in the
  wrapper's own input JSON is unchanged.

### Fixed

- **The recorded framework version now comes from the interpreter preflight
  resolved, not from whatever the orchestrator's own process happens to have.**
  `--framework-env isolated` is the default for vLLM, whose ROCm wheel pins its
  own torch, so the framework is installed where `importlib.metadata` in this
  process cannot see it — and `detect_stack_fingerprint` probed this process
  first, recording `unknown` on the default bare-metal vLLM path, or the version
  of a shared install the run never served with when one happened to be present.
  `_resolve_framework_build` already walks the candidate interpreters and imports
  the framework to find the right one, but `_check_serving_framework` only
  printed the winner; it is now published as `$HYPERLOOM_RESOLVED_FRAMEWORK_PYTHON`
  (paired with `$HYPERLOOM_RESOLVED_FRAMEWORK`, since the scan answers for one
  framework and `sglang` is the default) and the fingerprint reads its
  `site-packages`. The installer-written `$VLLM_VENV_ROOT` is no longer read by
  the fingerprint directly: it is only ever written, never cleared, so on its own
  it cannot say whether the tree it names still holds vLLM. It still leads
  preflight's candidate list and is probed there, which is what the recorded
  version now follows. A prefix that yields no `site-packages` — a system Python keeps its
  packages in `dist-packages` — is treated as a failed derivation and falls back
  to this process, not as an authoritative "not installed".<br/>
  **Operator note**: the framework check returns before publishing when
  `$HYPERLOOM_SKIP_FRAMEWORK_CHECK` is set, when `$BENCHMARK_BASE_URL` points at
  a remote server, on external multi-node, and for scriptable frameworks (xDiT,
  custom) that own their entrypoint — serving is not local on those paths, so
  the fingerprint falls back to this process rather than reading a venv root
  that describes some other host.

- **Shell and loader hijack names are rejected from the `extra_envs` argument to
  `materialize_config_with_envs` before the config is persisted.** The predicate
  was `valid_env_key`, a key-shape check that let `LD_PRELOAD`, `PYTHONPATH` and
  `PATH` through into the rendered YAML and from there into the benchmark
  subprocess. It is now `is_allowed_variant_env_key`, the predicate `GridVariant`
  already uses for per-variant overrides. The credential filter that runs
  immediately before the YAML is written is unchanged; it still covers the
  operator `--extra-env` channel, which has no upstream filtering.

- **A specialist's `config_changes` / `extra_envs` proposal is filtered where it
  enters `integrate_patch`, so the benchmarked configuration and the recorded one
  can no longer differ.** The raw mapping was assembled with no key validation and
  then took two paths: the gate bench went through `GridVariant` (which filters)
  while an `advanced` verdict persisted the unfiltered mapping into
  `accepted_config` and on into the revalidation baseline. Filtering once at
  assembly collapses both paths onto the same value. Dropped key names are logged
  and reported as `dropped_env_overrides` on the gate verdict and in the
  enablement `round.json`.

- **A reproduced warm replay is recorded as an adopted optimization.** The
  replay was mirrored into the canonical recorder streams before
  `_promote_warm_replay` reached its keep decision; because a replay's executor
  settles on `succeeded` either way, every replay was recorded as `discarded`
  with no adoption. A reproduced one was then pushed onto the stack and moved
  `cumulative_gain_validated` while the canonical streams held no adoption for
  it, so `optimizations.entries` came back empty on a session that had
  measurably gained and the whole gain surfaced as a `reconciliation_gap_pct`.
  The replay is now mirrored after the ruling: a reproduced one records a keep,
  chained from the recorded session baseline (not an enqueue-time anchor) so the
  ledger and `cumulative_gain_validated` are a single number; drift/failed
  replays stay discarded but their attempt row now carries the measured gain,
  the threshold, and the reason. `validated` keys off a present accuracy score
  rather than merely whether an eval ran, so an admitted-but-unscored replay
  records `keep_verdict_unscored` instead of a fabricated `accuracy_pass`. This
  is a forward fix: breakdowns already exported without the adoption are not
  retroactively repaired.

- **`best_result.json` is read again.** `_validated_forge_best_result` gated on
  `schema_version == 1`; KernelForge has stamped `2` into that file since
  2026-08-13. Every published best was therefore rejected and the kernel
  backend fell through to the caller checkpoint or the stdout sentinel, losing
  the one record that survives a hard kill — the case it exists for. The gate
  is gone rather than corrected: every field the evidence is read for is
  already checked on its own — the commit against the workspace history, the
  timings for being positive, the score for actually improving — so a version
  number decided nothing those checks do not, and was the only part that could
  fail closed on a bump that changed none of them. The eight tests that already
  covered this salvage path were passing only because their fixtures carried
  the same wrong version.

## [v1.0.0] - 2026-08-26
Current packaged version (`pyproject.toml`). See
[release notes](docs/release-notes.md) and the
[GitHub release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0)
for the user-facing summary.

## [v1.0.0b2] - 2026-08-19
See [release notes](docs/release-notes.md) and the
[GitHub release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0b2)
for the user-facing summary.

### Removed

- **BREAKING — the robustness agent's remote cluster data path is gone**, along
  with the flags that fed it: `--robustness-server-url`,
  `--robustness-workload-uid`, `--robustness-enable-cluster-pod-metrics` /
  `--no-...`, and `--robustness-pod-metrics-categories`. Callers still passing
  any of them now fail in argparse. `$ROBUSTNESS_SERVER_URL` and
  `$ROBUSTNESS_ENABLE_CLUSTER_POD_METRICS` are no longer read, and the startup
  probe that tried `http://robustness-server:8000` and `http://localhost:8000`
  on every tick is gone. No robustness-server is deployed and none of the five
  workload-uid env keys was ever set, so the endpoints could only 404; the
  `cluster_fault` and `pod_not_running` symptoms went with them, having had no
  other producer.

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

- `agents/kernel/tools/parallel_e2e_runner.py` is gone. It was the
  self-validation harness written alongside the original kernel-agent, back when
  no KERNEL phase existed to prove the toolkit end to end; its own first step
  (running the SGLang baseline) was removed in May, leaving a driver with no
  caller whose `--backends` default was empty, so it raised on any plain
  invocation. Its `load_env_file` duplicated the credential-alias derivation that
  `tools/backends/ray_runtime.py` still performs under wider test coverage.

### Changed

- **Multi-node runs now use the real robustness agent instead of the heartbeat
  mock.** `--nodes >= 2` previously forced `--robustness-mock`, which produced
  no symptoms at all — including `deadline_imminent`, the signal that drives the
  `delegate(report)` wind-down. The downgrade guarded against LocalProbe false
  positives, but `disable_local_probe` already defaults to True on multi-node
  and swaps the probe for a silent stub, so the signals the agent reads straight
  off the Coordinator prompt and inbox were being discarded for no reason. Those
  now fire: the deadline and budget ladder, `gain_plateau`, `no_levers_found`,
  crash escalation, `phase_budget_nearly_exhausted`,
  `conversation_no_progress`, and the inbox-driven `agent_stall` /
  `repeated_failure` / `repeated_policy_denied` family. Expect alerts on
  multi-node where there were none; pass `--robustness-mock` for the old
  behaviour.

- `ReactorBundle.aclose()` now closes the RCA engine's provider client. It
  previously closed only the robustness-server client, leaking the HTTP client
  the LLM RCA engine owns.

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
  superseded by `docs/reference/kernel-execution-path.md`, which documents the
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

[Unreleased]: https://github.com/AMD-AGI/Hyperloom/compare/v1.0.0...HEAD
[v1.0.0]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0
[v1.0.0b2]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0b2
[v1.0.0b1]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0b1
[v1.0.0a3]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0a3
[v1.0.0a2]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0a2
[v1.0.0a1]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.0.0a1
[0.8.0]: https://github.com/AMD-AGI/Hyperloom/blob/main/docs/release-notes.md
[v0.3]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v0.3
[v0.2]: https://github.com/AMD-AGI/Hyperloom/releases/tag/v0.2
