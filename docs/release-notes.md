---
myst:
  html_meta:
    "description": "Hyperloom release notes: headline capabilities for version 1.2.0, which runs every round through InferenceSuite, removes Magpie, adds the hyperloom-multibranch command and sandboxed specialists, and cannot resume sessions started by earlier releases; plus the 1.1.5 release, which runs AgentX optimization on Magpie's native AgentX contract, holds framework and kernel work to the phase budgets, adds GPU power budgets, an operable Experience KB and the PrimaTune predictor, and cannot resume sessions started by earlier releases; plus the 1.1.4 release, which makes `hyperloom <command>` the only entry point, runs every specialist as a Claude CLI subprocess, ships the Experience KB and GPU power measurement, and removes several options, environment variables and launch paths 1.1.3 accepted; the 1.1.3 patch release, the 1.1.2 release, which validates ROCm 10, makes ENABLEMENT its own phase, retires robustness supervision in favour of explicit resume and recovery, and supports ATOM end to end, and the 1.1.1 and 1.1.0 releases."
    "keywords": "Hyperloom, release notes, LLM inference, AMD GPU, ROCm, agentic optimization, TraceLens, GEAK, KernelForge, Primus-Claw, bare metal, kernel optimization"
---

# Hyperloom release notes

The current packaged version is 1.2.0 (`pyproject.toml`). For the
per-change history since the initial snapshot, and for a detailed breakdown of
all previous Hyperloom pre-release versions, see
[Releases](https://github.com/AMD-AGI/Hyperloom/releases); this page
summarizes the headline capabilities and carries the changes on `main` that no
release has shipped yet.

## Unreleased

Merged to `main` and not yet carried by a tagged release. Each entry moves
into the [release](https://github.com/AMD-AGI/Hyperloom/releases) that ships
it.

- **The hosted application image's `Dockerfile` and `build.sh` are removed.**
  `scripts/hosted_app_image/` keeps only the hosting layer (`build_layer.sh`
  and `Dockerfile.layer`), which starts a hosted run on any framework image
  unchanged. No CI job built or tested the application image. An image built
  from the old files keeps working; build new deployments on the layer. See
  [Upgrade](reference/upgrade.md).
- **`python -m hyperloom.measure.runner.cache_build` takes the wheel target
  only as `--wheel-target`.** `--python-version`, `--implementation`, `--abi`
  and `--platform` are removed and rejected with `unrecognized arguments`;
  `--wheel-target cpXY-manylinux_2_N_ARCH` names the same target with every
  platform tag the image's glibc accepts. See [Upgrade](reference/upgrade.md).
- **A result manifest states `runner.rule_pack` and `timing.phases`.** The
  `measurement_result.v2` schema and the reader now require both, with
  `timing.phases` an array (`[]` for none). Every runner that writes a v2
  manifest has written both; a manifest without them is refused as invalid
  instead of being read as one an older runner signed.
- **A measurement sheet's `environment.runtime` must be null.** The
  `measurement_spec.v2` schema and the reader no longer accept a runtime
  snapshot, and the runner no longer checks one: no producer, inside
  Hyperloom or out, ever wrote a non-null value. The key stays in every sheet
  as `null`, so spec fingerprints and JIT cache keys are unchanged.
- **`HYPERLOOM_SANDBOX_TEMPLATES` no longer accepts `runner` for the
  `measurement` role.** A hosted run reads only `measurement` and `dev`; a
  list that gives only `runner` is refused as having no `measurement` role.
  No host sends `runner`. See [Upgrade](reference/upgrade.md).
- **A hosted run no longer checks host inputs it never used.**
  `HYPERLOOM_MCP_PUBLIC_URL`, `HYPERLOOM_SANDBOX_KEY_SUBRUN`,
  `HYPERLOOM_RESULT_ARCHIVE` and the S3 archive's `AWS_ACCESS_KEY_ID`,
  `AWS_SECRET_ACCESS_KEY`, `AWS_SESSION_TOKEN`, `AWS_ENDPOINT_URL_S3` and
  `AWS_REGION` are no longer read or validated, so a malformed value no longer
  refuses the run; they are still removed from the environment before any
  child process starts. Results are archived on the shared storage, as
  before: archiving to `s3://` was never implemented.
- **The sandbox socket's `Role` is `measurement` or `dev`.** The `subrun`
  role is gone from `hyperloom.sandbox.socket.Role`, so a local template
  declared with it is refused. No Hyperloom code ever created a sub-run
  sandbox; a host may still list a `subrun` role in
  `HYPERLOOM_SANDBOX_TEMPLATES`, which a hosted run ignores as before.
- **`campaign_spec.v1` drops fields nothing read.**
  `objective.target_gain_pct`, `options.kernel`, `options.framework_agent`,
  `options.enablement` and `placement.specialist_capacity` are gone from the
  schema and the validator. A spec that still sets them parses as before,
  since unknown fields are ignored, but a malformed value in one of them is no
  longer refused. `options.eval` stays and still refuses `false`.
- **`hyperloom session report` is removed.** The command rendered a
  `session_breakdown.json` as Markdown (`session_report.md`), optionally with
  an LLM-written summary. It ran only by hand: no session wrote
  `session_report.md` on its own. Read `reports/final.md` for a session's
  summary and `session_breakdown.json` for the recorded facts; neither
  changes. `HYPERLOOM_REPORT_LLM_BACKEND`, `HYPERLOOM_REPORT_MODEL` and
  `HYPERLOOM_REPORT_MAX_TOKENS` are no longer read. See
  [Upgrade](reference/upgrade.md).
- **The hosted single-session flow is retired.** `hyperloom hosted run` now
  only checks the campaign spec, runs the installer and hands the run to
  `hyperloom-multibranch hosted run`. A spec must set `"flow": "multibranch"`
  or carry a `multibranch` section; an unchanged 1.2.0 single spec (neither,
  which ran the single-session flow, measured locally) and a spec that still
  says `"flow": "single"` end the run with a final `exit.json` (return code 2)
  naming the retired flow and what to add. `placement.measurement` now
  defaults to `hosted`, and a single run in flight does not resume. See
  [Upgrade](reference/upgrade.md). The single flow's MCP host tools
  (status, stop request), its progress events and its `hosted-gpu` /
  `hosted-cpu` plug profiles are removed, and with them the
  `hosted/control/stop.json` stop request the Coordinator read.
  `HYPERLOOM_EVENTS_DIR` and `HYPERLOOM_MCP_HOST` / `_PORT` / `_TOKEN` are no
  longer read or checked, and are still removed from the environment of every
  child. `optimize --session-dir` stays. `options.conc_sweep` is no longer
  part of the spec (it is ignored like any unknown field); a multi-branch run
  takes its sweep switch from `multibranch.conc_sweep`.
- **The plug kernel is removed.** `optimize` drops `--plug-profile`,
  `--manifest`, `--plug` and `--plug-config` (each now exits 2 as an
  unrecognized argument), no longer reads `HYPERLOOM_MANIFEST` or
  `HYPERLOOM_MANIFEST_OVERRIDES`, and no longer writes or reads back
  `plugs.resolved.json`. Its only filled sockets were the progress events,
  whose hosted writer went with the single flow, and the native builds of a
  checkout-built server, which `optimize` now opens directly. See
  [Upgrade](reference/upgrade.md).
- **forge-loop runs every round as lanes and has no planning stage.** Every
  iteration is one round: one lane per focus the Analysis Agent divides its
  evidence into (at most `--lanes`, and one lane with no focus when it divides
  nothing or `--lanes` is 1), each lane a self-directed Implementer session,
  then one candidate measured on the canonical tree. A round's only lane works
  in the workspace itself; lanes that run side by side each work in their own
  copy of it. When several lanes changed the kernel, an Implementer session on
  the canonical tree merges them, starting from the lane that scored best. The
  lanes of such a round share the campaign's GPU through a queue: a lane's
  commands reach the device through `forge_device.py` in its copy and wait
  their turn, and its stop check waits in the same queue. `--lanes` above 1 is
  refused on a workspace whose repository is selected through `GIT_DIR` or
  `GIT_WORK_TREE`, which a copy cannot carry. A lane that runs beside others
  may stop on a correct candidate that is no faster overall; a round's only
  lane still has to beat the current best. The search turns to a different mechanism after
  one round without a new best and is reviewed after two, at every `--lanes`.
  **Breaking:** the dispatch, specialist, synthesis and Plan Critic sessions,
  the per-iteration lane queue and the stacking of archived candidates are
  removed, with `--merge-stacking`, `--specialist-probe` and
  `--specialist-probe-{max,budget-sec,scratch-root}`; passing one of these
  options is an error. The `FORGE_SPECIALIST_PROBE*` variables are no longer
  read: one that is still set is ignored, and the run starts normally. A
  campaign created before this change is not resumed: its `run_state.json` carries
  fields the loader rejects. `kernelforge export-runs` no longer emits
  `plan_file`, `plan` from a plan file or `analysis_report`, and writes one
  unit per round at every `--lanes`. The `kernelforge-run-v1` declaration and
  the `kernelforge-v1` mapping drop `rationale.plan`, `rationale.plan_file`,
  `reflection.analysis_report` and the decision label `ORCHESTRATION_ERROR`,
  so Experiences collected from here on carry a new `schema_ref`; those
  already in a KB keep theirs.
- **The DeepSeek Harness reaches the gateway through the `llm-pi-ai` route.**
  The `deepseek-official` adapter rejected the `redacted_thinking` blocks a
  gateway returns with `UNSUPPORTED_CONTENT`; the `anthropic-messages` route
  keeps and replays them. With `HYPERLOOM_DSH_MAX_TOKENS` unset, a request's
  output cap is now 128000, the `llm-pi-ai` default; it was 256000.
- **A hosted multi-branch test closes its sandbox when it ends.**
  Make-before-break, which kept a finished test's sandbox until the next
  test's ran where the grant fit at least three tests, is removed. A hosted
  session now runs `min(max_gpus // tp, max_sandboxes)` tests at once, one
  more than before on such a grant, and each test's sandbox queues for a
  node. `hyperloom-multibranch run` no longer takes
  `--[no-]make-before-break`. `multibranch.make_before_break` in a campaign
  spec is accepted and ignored, with a warning in the log.
- **A hosted multi-branch run sets the AgentX window and the server boot
  budget.** The campaign spec's `multibranch` block takes
  `agentx_duration_sec` and `agentx_objective` (for `workload: agentx` only),
  forwarded as `--agentx-duration-sec` and `--agentx-objective`; before, a
  hosted AgentX run always replayed the default 3600 s window. A new
  `--boot-budget-sec` on `hyperloom-multibranch run` (`boot_budget_sec` in the
  spec, at most 21600) is how long, at least, a test's server may take to be
  ready; the round raises the framework's own ready timeout to the budget plus
  3600 s when that is above 10800 s. It only lengthens: a test runs for its
  trace window, if any, plus 1800 s plus that budget when that is longer than
  its default timeout, and the boot limit is the budget when that is longer
  than half the test's timeout. The session manifest records it as
  `settings.workload.boot_budget_sec` (null when unset). Without it nothing
  changes.
- **A single-node session no longer uses Ray.** Serving rounds run in the
  Coordinator's process, one at a time behind the resource lanes, and GPU
  specialists take their cards from the session's GPU pool only. The installer
  and preflight no longer install, certify or start a Ray head, nor raise the
  open-files limit for one; `RAY_VERSION`, `SKIP_RAY_START` and the
  `INFERENCE_OPTIMIZER_RAY_*` settings are no longer read. The failure classes
  `ray_cluster_infeasible`, `ray_actor_error` and `ray_round_unplaced`, the
  return codes -913 and -916, and `serving_lease_held` / `ensure_ray` in
  `session_breakdown.json` are gone. A cancelled GPU specialist now gets 13.5 s
  to stop instead of 23.75 s. Multi-node (RayJob, Infera) is unchanged, except
  that GEAK no longer receives a `RAY_ADDRESS` nothing read. Stop the head an
  older install started with `ray stop --force` once; see
  [Upgrade](reference/upgrade.md).
- **An AgentX KEEP that removes a recipe flag no longer strips the recipe from
  every later round.** Writeback turned such a KEEP (one with `remove_args` or
  `unset_envs`) into `args_mode=replace`, so each later round launched only the
  KEEP's own flags: the recipe's `--tensor-parallel-size` and the rest were
  removed, servers started at TP1 and died in model init with an out-of-memory
  error, and the session ended on that KEEP. The KEEP now stays relative to the
  recipe, as the explore and integrate executors already record it, and stacks
  on what earlier KEEPs accepted: their flags, and their removals and unsets
  too, so a later KEEP no longer silently brings back a flag or variable an
  earlier one took out (unless it sets that flag or variable itself). Outside
  AgentX, and for a KEEP that is itself `replace`, nothing changes.
- **AgentX runs on ATOM.** `HYPERLOOM_AGENTX=1 hyperloom optimize --framework
  atom` now starts: the base comes from InferenceX's ATOM agentic entries, with their forced
  acceptance length (`--spec-decode-acceptance-length`) as the golden layer.
  Before, every AgentX entry point refused ATOM. Entries that offload KV cache
  (`extra-kv-connectors`) are still refused as unported. ATOM's own catalog
  still serves fixed workloads only.
- **The runner names an ATOM server's model.** An ATOM round now passes
  `--served-model-name`, as SGLang and vLLM rounds do, so ATOM needs v0.1.6 or
  later: an older server is refused before launch as `flag_unknown`. A layer or
  a frozen ATOM base whose own args set `--served-model-name` is refused as
  setting a runner-owned flag; for a frozen base, start a new session.
- **An ATOM AgentX baseline with its prefix cache on is kept.** The check that
  refuses a cacheless AgentX baseline took ATOM's cache as off unless the launch
  passed `--enable_prefix_caching`; ATOM caches by default. It now refuses only
  a launch whose argv, as the round's `launch.json` records it, turns the cache
  off: its last prefix-cache switch is `--no-enable_prefix_caching` (either
  spelling).
- **An ATOM profile round** sets `ATOM_PROFILER_TIMEOUT` to the 600 s the round
  waits on `/stop_profile`, so a long capture is not cut off at ATOM's 300 s
  default. An annotated ATOM AgentX capture turns on ATOM's own light knobs
  (`ATOM_ENABLE_DETAILED_ANNOTATION`, `ATOM_PROFILER_RECORD_SHAPES`) instead of
  SGLang's (v0.1.6 records no tensor shapes). Unless the point sets
  `capture_timeout_seconds`, it records 30 s rather than 300 s: ATOM's profiler
  takes no step bound, and a long capture slows the replay it runs inside.
- **Bump validated vLLM stack to 0.34.0 (ROCm 10 docker and source build).**
  Switches the recommended vLLM image to `docker.io/vllm/vllm-openai-rocm:v0.34.0-rocm100`
  (docs, examples, `models.tsv`, installer hint) and bumps the bare-metal defaults
  (`VLLM_VERSION=0.34.0`, `VLLM_SOURCE_REF` at the `v0.34.0` release commit). vLLM 0.34.0
  publishes only a ROCm 10 wheel, so the ROCm 7.2.x wheel route now stops before any
  change when the wheel index has no such release, instead of failing after the venv is
  created. **Upgrade note:** move from `vllm/vllm-openai-rocm:v0.33.0-rocm100` to
  `v0.34.0-rocm100`; on a ROCm 7.2.x bare-metal host use Docker, a ROCm 10 host, or
  `VLLM_VERSION=0.33.0 VLLM_ROCM_VARIANT=rocm72`.
- **A multi-branch session reads and writes the workspace's Experience KB.**
  With `HYPERLOOM_KB_URL` and `HYPERLOOM_KB_TOKEN` exported (the values
  `hyperloom-setup` writes into `.env`), a session's briefs recall past
  sessions' tests and learnings and its learnings tool their learnings, after
  this session's and the RSI lesson pool's; a claim reads none. CLOSE writes the
  session's tests and learnings there, spooling beside the service's data what
  it does not take. An InferenceX session reads both schemas its tests are
  written under, so a kernel change is recalled too. A loopback service is
  started before the session, outside the session's process tree, and left
  serving; one serving with other settings is not restarted
  (`experience_kb_service ensure --no-restart`). Before, a session read and
  wrote no Experience KB, whatever the environment named.
- **The Experience KB service plans reads through Hyperloom's gateway
  clients.** Its planner calls the Anthropic gateway when it has a key, else
  the OpenAI one, with the clients the rest of Hyperloom uses: their custom
  headers (`ANTHROPIC_CUSTOM_HEADERS`, `OPENAI_CUSTOM_HEADERS`), connect
  retries and usage ledger, and it asks for the plan as JSON text instead of
  forcing a tool call. The model is `LOCAL_KB_PLANNER_MODEL`, else
  `CLAUDE_MODEL` (or `CODEX_MODEL`), else the backend's default;
  `ANTHROPIC_MODEL` no longer names it. Before, every read failed on an AMD
  gateway, which refuses a call without its subscription header and a forced
  tool choice, and an OpenAI-only workspace planned no reads.
- **The EXPLORE screen pre-filter is removed.** The opt-in reduced-GPU probe
  that 1.2.0 added behind `HYPERLOOM_EXPLORE_SCREEN=1` is gone: on the one grid
  it was measured against, the order of variants did not transfer to fewer
  GPUs, and it was off by default. `HYPERLOOM_EXPLORE_SCREEN` and its
  `_MARGIN_PCT`, `_GPUS`, `_MODEL`, `_LAYERS`, `_ATTENTION_BACKEND` and
  `_TIMEOUT_SEC` settings are no longer read, and no variant is skipped as
  `screen_decisively_slower`. The measurement store and the InferaSim
  projection are unchanged, and so is a run that never set the switch.
- **A session reads its GPU from the machine; no board table is kept.** Every
  start and resume reads the visible cards and prints the one it runs on (`GPU
  card        : AMD Instinct MI325X (gfx942, 304 CUs, 262128 MiB, 2100 MHz)`).
  Its arch, compute units, memory, bandwidth and top engine clock replace the
  per-board rows, so a card needs no registration. The session asks `amd-smi
  static --json` first, then `rocm-smi`, then torch's device properties
  (`gcnArchName`); the last two give only the name, arch and memory, so such a
  card has no vendor peak or AITER CSV check. A multi-node
  run asks a remote node the same way. Child processes read the card from
  `HYPERLOOM_GPU_CARD` and the manifest archives it as `gpu_card`. A host none
  of the three can describe prints `GPU card        : none read (<reasons>)`
  and runs without arch-specific launch rows or gates. When the visible cards
  are not one kind (arch, compute units, memory) the launch warns, listing each
  kind, and takes the first; a device mask may name cards by ROCm number or
  UUID.
- **`--gpu-type` accepts any name; the card's own name still wins.** The GPU
  type is the label results, recipes and KB rows are filed under. It is the
  model designation in the card's market name (`AMD Instinct MI355X` is
  `mi355x`; `MI300XL` and `R9700S` are labels of their own), and it overrides a
  given name that differs, with a `WARN` naming both, as the detected board
  did. `--gpu-type`, then `TARGET_GPU_TYPE`, then `GPU_TYPE`, name the GPU when
  the driver gives the card no model name or no card was read, and any name is
  accepted there (the closed list of ten boards is gone); with none of them an
  unnamed card is filed under its arch. A session with neither a card nor a
  name exits 2 asking for `--gpu-type`. No framework is refused at launch as
  not validated on a Radeon board (R9700, R9600D, W7900, W7900D).
- **A recipe serves only the board it names.** InferenceX points, SGLang
  cookbook cells and headings, and ATOM catalog runners and headings match the
  session's GPU type by name. Boards sharing an arch and memory size no longer
  borrow each other's recipes: an MI355X session no longer takes ATOM catalog
  entries tuned on `mi350x`, nor MI350X and MI308X sessions the InferenceX
  entries of MI355X and MI300X; they fall through to the next source or the
  default table. A frozen `base_launch.json` serves another board only when
  both queries record the same `gfx` and `memory_mib` (new query fields,
  outside the digest), so one frozen by an earlier release serves its own board
  only. To borrow another board's launch, pass it as `--reference-script`.
- **The default table is keyed by the card's arch and has a fallback row per
  framework** (`--trust-remote-code` only). An arch without a tuned row, or a
  session without a card, runs the fallback with a warning (`the default table
  has no sglang launch tuned for gfx1201, so its fallback launch runs it`)
  instead of exiting with `it has no sglang row for r9700 (gfx1201)`.
- **Arch-dependent behaviour follows the card's arch, not the board name.**
  ENABLEMENT attempts targeted builds on every arch (Radeon boards were
  refused). `mxfp4` schemes run on any gfx950 card, so an MI350X can use them.
  `SGLANG_USE_AITER_FP8_PER_TOKEN` is injected on any gfx942 card. The AITER
  model-config check compares CSV rows with the card's arch and compute units,
  and is skipped without a card. KernelForge GEMM tuning is given the card's
  arch (`auto` without one) instead of the GPU type. The GEAK handoff no longer
  writes `expected_target` and `expected_gfx`. Compute partitions are sized by
  what `amd-smi` reports for the partitioned GPU, never by a board table; where
  it reports no compute units a declared `--compute-partition-mode` is refused.
- **Roofline peaks and bandwidth come from the card and TraceLens.** Bandwidth
  is the driver's `vram.max_bandwidth` (6144 GB/s on an MI325X, was 6000). The
  compute peak is the measured `max_achievable_tflops` in TraceLens'
  `<LABEL>.json` when the checkout has one, else the card's vendor peak
  (compute units × top engine clock × the arch's rate); before, only MI300X and
  MI325X had measured peaks. A card that reports no bandwidth, which includes
  the Radeon boards, has no analytic ceiling, where the board table gave one;
  the roofline TraceLens computes from a profile is unaffected. On Radeon the
  driver's top clock level is below the boost clock AMD's published peak uses,
  so the vendor peak computed there is 19% to 30% under the published figure.
  `hyperloom conc-sweep` run on its own reads the card the session archived in
  its manifest, else the machine's. **Upgrade note:** in a `TRACELENS_ROOT`
  checkout you maintain, delete any `<LABEL>.json` an earlier Hyperloom wrote
  with vendor peaks; it is now read as measured.
- **Nothing defaults to MI355X.** `tracelens_analysis --target-platform` is
  required, and a `trace_analyze` request with no GPU type fails with
  `gpu_label_missing`. A custom workload no longer looks for `custom_mi355x.sh`
  when nothing names the machine. `MLPERF_AGENTIC_HARDWARE` defaults to the
  session's GPU type. An InferaSim projection with no GPU to price is refused,
  and its HBM capacity defaults to the card's memory.
- **Recipe KB borrows across GPU types only between rows that record the same
  `gfx_arch`** and topology suffix, locally and remotely. Rows written before
  this release match only their own GPU type until a session on it writes them
  again. Remote searches no longer send `hardware_in`.
- **The hosted runner checks the card by its facts.** A measurement sheet's
  `hardware` gains `card` (null when the session read none). The runner
  compares each visible GPU's arch, compute units and memory with it, instead
  of matching the GPU type against a device-id table and the market name.
  **Upgrade note:** a runner from an earlier release refuses the new sheets
  with `spec.hardware: unknown key 'card'`; move the app and its runners
  together.
- **Multi-branch sessions take any `--gpu-type` and read their card**: locally
  from the machine, hosted from the recording of `amd-smi static --json` that
  `HYPERLOOM_RUNNER_GPU_CARD` names on the app host. The session is filed under
  the `--gpu-type` it is given and warns when the card names itself otherwise;
  the list of six boards is gone, and `r9700` and `w7900d` are no longer
  limited to `--framework llamacpp`. The session stops before any test when the
  variable is unset or the recording describes no one kind of card. The
  roofline and TraceLens tools default to the session's GPU type. **Upgrade
  note:** set `HYPERLOOM_RUNNER_GPU_CARD` on every hosted app host before
  upgrading; see [Run hosted multi-branch on a new
  cluster](how-to/hosted-multibranch-new-cluster.md).
- **KernelForge maps only gfx ids and its four stored labels** (`mi300x`,
  `mi355x`, `w7900`, `radeon8060s`) to an arch: pass `gfx942` or `gfx1201` for
  an MI308X, MI325X or R9700. Any other name keeps its own spelling as a KB
  key.
- **The bare-metal installer prints the GPU as `amd-smi` reports it** (`GPU:
  AMD Instinct MI325X (gfx942)`) and no longer warns `GPU arch <gfx> untested
  (supported: gfx942/gfx950)`.
- **`hyperloom_common.gpu_identity` is removed**, with no alias. **Upgrade
  note:** read `hyperloom_common.gpu_card` instead: `session_card()`,
  `read_cards()`, `cards_of(document)`, `GpuCard` and `board_label(name)`.

## Hyperloom 1.2.0 release

The [1.2.0 release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.2.0)
follows 1.1.5. Every round now runs through InferenceSuite: Magpie and the
`bypass` backend are removed, and a session resolves its base launch from
InferenceX or the framework's cookbook and freezes it as `base_launch.json`.
The release adds the `hyperloom-multibranch` command, sandboxes specialists
with bubblewrap where the host allows it, records each round's energy in
`power.json`, runs the agent roles on the DeepSeek Harness on request, and
recognises the MI350X, Radeon AI PRO R9600D, Radeon PRO W7900 / W7900D and
Radeon 8060S. The validated stacks move to SGLang 0.5.23 and vLLM 0.33.0.
Sessions started by 1.1.5 or earlier cannot be resumed, and several options,
environment variables, module paths and report fields move; see "Before
upgrading from 1.1.5" below. The 1.1.5 known issue (a removal KEEP on a Magpie
AgentX recipe that pins golden acceptance) does not apply: that Magpie path is
gone.

### Before upgrading from 1.1.5

Finish in-flight sessions under the build that started them. Every session's
`storage/coordinator.db` now records schema version 8 and there are no
in-place migrations, so `optimize --resume-from` on a session started by 1.1.5
or earlier exits `2` before it touches the session, with
`coordinator.db is at schema version 7, older than this build's 8: sessions created before this release cannot be resumed`.
`kernelforge forge-loop --resume` likewise refuses a 1.1.5 checkpoint that has
finished a round, with
`run state round_costs.recent[0] missing fields: review_sec`.
See [Upgrade Hyperloom version](reference/upgrade.md).

Recipe KB rows that 1.1.5 filed under a framework build with a local label,
`v` prefix or `.dev` segment (for example `0.30.0+rocm723`) are no longer exact
warm-start hits: identities now key on the release (`0.30.0`), and those rows
are reached only through the `compatible_framework_version` tier (confidence
0.72). Nothing migrates them; the next CLOSE writes the release row.

Options and module paths 1.1.5 accepted are removed or refused. The parsers
are strict, so a command that still carries one exits with
`unrecognized arguments` or `No module named` before it runs.

| Removed | What to do |
|---|---|
| TraceLens kernel tool `--use-llm-orchestrator` / `--no-llm-orchestrator` | Use `--route agent` or `--route non-llm`. |
| TraceLens kernel tool `--num-denoise-steps`, `--model-path`, `--precision` | Delete them; they fed the diffusion latency roofline, which is removed. |
| `python -m hyperloom.orchestrator.trace_analysis.bypass_trace_analysis` | Use `tracelens_analysis --route non-llm`. |
| `python -m hyperloom.inference_optimizer.framework_deps` | Nothing replaces it; its installer assets were already gone, so it did nothing. |
| A `--benchmark-config` YAML whose `profiler.torch_profiler.num_profiles` is not `1` | **Refused** at launch: a profile round captures one window. Drop the key. |
| Imports of `hyperloom.common.{aiter_jit_cache, checkpoint_quant, coerce, env, fs_utils, git_safety, gpu_identity, io, jsonio, pr_monitor_urls, proctree, timeutil, unified_diff}` and of `hyperloom.common.{llm_config, llm_headers, llm_attribution, llm_request_hooks, llm_stability_env, token_usage, reasoning_effort, claude_oneshot}` | Import them from `hyperloom_common` and `hyperloom_common.llm`; no aliases are kept. `kernelforge.tools.durable_io` is removed: use `hyperloom_common.io`. |
| KernelForge modules under `kernelforge.loop`, `kernelforge.gemm_tune`, `kernelforge.fusion`, `kernelforge.orchestrator`, `kernelforge.assembly`, `kernelforge.llm`, `kernelforge.tracker`, `kernelforge.tools`, `kernelforge.kernel_rewrite_controller` and `kernelforge.rewrite_by_flydsl` | They move under `kernelforge.workflows`, `kernelforge.kernel_backends`, `kernelforge.agent_backends`, `kernelforge.utils` and `kernelforge.config`, with no compatibility modules. A hand-written MCP configuration that runs `python -m kernelforge.orchestrator.specialist_probe_server`, the path 1.1.5 named, needs `kernelforge.workflows.forge_loop.roles.specialist_probe_server`; `kernelforge.evaluation.benchmark.runner` and `kernelforge.knowledge.pr_knowledge.mcp_server` keep their paths. No `kernelforge` subcommand is removed. |

Environment variables fail differently: nothing refuses most of them, so a box
that still exports one starts normally and behaves as though it were unset.
Find them by reading launch scripts, `.env` and the kernel-agent env file. The
settings that do stop the run are marked below.

| Removed or changed variable | What to do |
|---|---|
| `MAGPIE_EVAL_LIMIT`, `MAGPIE_EVAL_TASKS`, `MAGPIE_EVAL_TOKENIZED_REQUESTS` | Rename them to `HYPERLOOM_EVAL_LIMIT`, `HYPERLOOM_EVAL_TASKS` and `HYPERLOOM_EVAL_TOKENIZED_REQUESTS`. The old names are ignored, so a smoke box that capped the eval with `MAGPIE_EVAL_LIMIT` now runs the full GSM8K. |
| `MAGPIE_PATH`, `MAGPIE_REF`, `MAGPIE_REPO`, `MAGPIE_PACKAGE_SPEC`, `MAGPIE_INFERENCEX_PATH`, `MAGPIE_EVAL_FLAG_STRICT`, `PATCH_MAGPIE` and every other `MAGPIE_*` name | Delete them; Magpie is no longer installed or imported (see *Installation prerequisites*). |
| `HYPERLOOM_BENCHMARK_BACKEND` | Delete it; every round runs through InferenceSuite. `--framework custom` no longer needs `HYPERLOOM_BENCHMARK_BACKEND=bypass`; `--benchmark-scripts-dir` (`HYPERLOOM_BYPASS_SCRIPTS_DIR`) still names the script directory. |
| `INFERENCEX_REF`, `INFERENCEX_DEFAULT_DIR` | Delete them; no InferenceX checkout is configured any more. The client is staged at the commit `inference_suite.pins` pins, and the base launch is read from InferenceX `main` unless `--inferencex-commit` names one. `INFERENCEX_PATH` is optional and no longer set by `install.sh`; the research scout reads reference launch scripts under it. |
| `HYPERLOOM_EVAL_DERIVE_STOP`, `HYPERLOOM_EVAL_STOP_STRINGS` | Delete them; nothing reads them. `HYPERLOOM_EVAL_MAX_TOKENS` is still read. |
| `HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass`, or `analysis_route: bypass` in a `trace_analyze` request | **Fails** the request with `invalid_analysis_route`. Set `non-llm`, which needs a TraceLens checkout and writes `analysis.json` instead of `analysis.md`. `HYPERLOOM_BYPASS_STEADY_STATE`, `HYPERLOOM_BYPASS_CORR_WARN_PCT`, `HYPERLOOM_BYPASS_OTHERS_WARN_PCT`, `HYPERLOOM_TRACE_SHAPE_MANIFEST`, `HYPERLOOM_TRACE_SHAPE_MANIFEST_MAX_CAPTURES`, `HYPERLOOM_NUM_DENOISE_STEPS` and `KERNEL_AGENT_USE_LLM_ORCHESTRATOR` are ignored. |
| `HYPERLOOM_SPECIALIST_PERMISSION_MODE` | Delete it; a Claude specialist always runs with `--permission-mode bypassPermissions`. |
| `INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES`, `HYPERLOOM_ROCPROF_ROOFLINE_ENRICH`, `HYPERLOOM_ROCPROF_ROOFLINE_TIMEOUT_SEC` | Delete them; nothing reads them. |
| `GEAK_TERM_GRACE_S` | Delete it; the KERNEL phase stops every process its GEAK run started, and `GEAK_FLUSH_GRACE_S` is the window GEAK gets to flush its result. |
| `RAY_CLI_CLICK_MAX_VERSION` | Delete it; the Ray CLI is certified by its own `ray start` parser. `RAY_VERSION` is still honoured. |
| `HYPERLOOM_GPU_POWER_INTERVAL_S` | The default drops from 2 s to 1 s. A value above 1.5 s is refused: the round samples nothing, so a power budget refuses it as unmeasured. |
| `KERNEL_AGENT_ENV` naming `$USER_DATA_PATH/runtime/kernel-agent.env.sh` (a 1.1.5 `.env` or shell) | Ignored and derived again; the file now lives at `${TMPDIR:-/tmp}/hyperloom-<uid>-<key>/kernel-agent.env.sh` on the container's own disk, and install removes the old copy. A value in a directory that the group or others can write **is refused** by `install.sh` and by preflight. |
| `TRACELENS_ROOT` naming an earlier install's `TraceLens@<sha>` | **Exits 2** at preflight (`is an earlier install's checkout`). Delete the line and rerun `install.sh`. |
| A malformed numeric `HYPERLOOM_MN_*` setting (for example `HYPERLOOM_MN_HEALTH_POLL_S=10s`) | **Fails** the first multi-node restart (`invalid multi-node restart setting`) and makes a `hyperloom multi-node` subcommand exit 3, where 1.1.5 fell back to a default on some paths. |

Defaults and gates that move, so the same command behaves differently:

- **The base launch comes from a recipe, not from Magpie's scripts.** Without
  `--reference-script`, a new session resolves its server launch from InferenceX
  `main`, then the framework's cookbook, then (for a fixed workload only) a
  default table, and freezes it as `base_launch.json`. The sources are read from
  GitHub at launch; a fixed session that cannot read one takes the default
  table, and `--inferencex-commit SHA` pins the InferenceX read. An omitted
  `--tp` / `--ep` (and no `$TP` / `$EP`) now follows the recipe. A native AgentX
  session that no recipe serves, or a single-node base that needs more GPUs than
  the host sees, exits `2`.
- **`--reference-script` is the session's base launch.** It wins over every
  recipe source and is now accepted under native AgentX. A script that sets a
  flag from a shell variable (1.1.5 dropped it silently), carries a
  `# hyperloom-launch-controls:` line, cannot be read or does not launch the
  session's framework exits `2`, and a base whose server command is not the
  framework's default entrypoint (for example a hand-written interpreter path)
  is refused as `config_invalid` at its first round.
- **Native AgentX accuracy is a GSM8K round.** The baseline, and each
  candidate that clears the KEEP bar, runs GSM8K on a server of its own;
  `--no-agentx-accuracy` turns it off. The request error rate no longer grades
  accuracy, only the measurement's validity.
- **AgentX keeps on a 2% P50 interactivity gain, not 3%.** The guards are P90
  interactivity and total token throughput inside `HYPERLOOM_PERF_NOISE_PCT`;
  an output-throughput drop alone no longer refuses, `output_regressed` is no
  longer given and `total_regressed` is new.
- **A `custom` workload's script is named after the board and its gate is
  read on raw scores.** `GPU_TYPE` is exported as the board, so
  `--benchmark-scripts-dir` is searched for `custom_mi308x.sh`,
  `custom_mi325x.sh` or `custom_r9700.sh` where 1.1.5 looked for
  `custom_mi300x.sh` or `custom_gfx12.sh`; keep the script the only `.sh` in
  the directory or rename it. A baseline whose report carries no
  `quality_gate` stops the run with `baseline_accuracy_failed`, and a gate
  without `passed` is judged against the limits it names, else `lpips` 0.15,
  `ssim` 0.85 and `mse` 0.006.
- **Specialists run sandboxed where bubblewrap works.** With a usable,
  non-setuid `bwrap` on `PATH`, every specialist runs in a bubblewrap sandbox,
  with no switch to turn it off: it reaches only the LLM endpoints and MCP
  servers its launch names, downloads no model or package (a Hugging Face model
  must already be in the cache), and `codex` ignores
  `HYPERLOOM_CODEX_SANDBOX_MODE`. A `claude` CLI whose login exists only in the
  app's home cannot read it there: it runs unconfined with a WARNING until the
  environment holds `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN` or
  `CLAUDE_CODE_OAUTH_TOKEN`, and is refused where isolation is required (a
  hosted run, or `HYPERLOOM_SPECIALIST_USER` set). Claude specialists also stop
  loading the operator's `~/.claude` settings (`--setting-sources=`).
- **A session on part of an NPM-enabled node is refused.** With node power
  management enabled, a launch or resume whose visible-device mask leaves out
  any GPU on the node exits `2`; pass `--allow-shared-npm-node` on every launch
  and resume to measure anyway.
- **`ROCR_VISIBLE_DEVICES` must hold GPU indices.** A round names its GPUs by
  index, so a UUID mask, which Magpie passed through, fails the baseline with
  `the round's ROCR_VISIBLE_DEVICES is not a list of GPU indices`.
- **A launch needs `install.sh` in its own container.** The kernel-agent env
  file is container-local, so `optimize` in a new container that skipped
  `install.sh` exits `2` at preflight. Source the kernel-agent env file with
  the line given in [Upgrade Hyperloom version](reference/upgrade.md).
- **A session that holds a credential keeps its database outside the session
  directory.** When `--extra-env` or `--server-args` carry a credential, the
  coordinator database lives below `${TMPDIR:-/tmp}/hyperloom-ledger-<uid>/`,
  so resuming it in a new container is refused unless `HYPERLOOM_LEDGER_DIR`
  names a persistent private directory that holds it.

Installation prerequisites move:

- Magpie is no longer installed and no InferenceX checkout is configured. `install.sh`
  stages the InferenceX client at the pinned commit, the GSM8K task and the
  AgentX harness into `INFERENCE_SUITE_TOOL_CACHE` (default
  `${HYPERLOOM_CACHE_DIR:-$REPO_ROOT/.cache}/inference-suite-tools`) and
  installs `lm-eval` 0.4.13, hash-locked from PyPI, into `$PYTHON`; preflight
  no longer installs `lm_eval`. A round fetches nothing and refuses with
  `staging_missing` when the cache or `$PYTHON` lacks a pinned tool, and
  `install.sh --check-only` fails with `behind the pinned revisions`. Rerun
  `install.sh` after upgrading, on every node and container that runs rounds.
- TraceLens is pinned to `bda57c6d97cb0ec6810d41f9eba8bb10a0315d52` (was
  `614c597bc0c66cabf6a15152037324acc7386090`); rerun `install.sh`.
- A GEAK run that selects an agent backend (`GEAK_AGENT_BACKEND`,
  `GEAK_AGENT_PROFILE`, or the Codex backend an OpenAI-only configuration
  implies) needs Node 20 or newer and that agent's CLI. `install.sh` reuses a
  Node on `PATH` or fetches Node 22.23.3 from nodejs.org, and pins
  `GEAK_NODE_BIN` and `GEAK_CODEX_BIN`; preflight exits `2` when a binary GEAK
  will run is not runnable. Forge runs and `--no-kernel` skip the check.
- The validated SGLang stack moves from 0.5.21 to 0.5.23 (`SGLANG_REF`
  `297e54fd105e`): use `lmsysorg/sglang-rocm:v0.5.23-rocm10-mi30x-20261111` or
  `-mi35x-20261111`. The 20261118 builds answer `/start_profile` with
  `403 Forbidden` because Hyperloom starts SGLang without an admin API key, so
  their sessions get no roofline profile.
- The validated vLLM stack moves from 0.29.0 to 0.33.0: the image is
  `vllm/vllm-openai-rocm:v0.33.0-rocm100` (was
  `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`), and on
  bare metal `VLLM_ROCM_VARIANT` defaults to `rocm72` (was `rocm723`) and
  `VLLM_SOURCE_REF` to `4ba787ec9959`. An Ubuntu 22.04 host that pins
  `VLLM_VERSION=0.27.1` must now also set `VLLM_ROCM_VARIANT=rocm723`.
- `install.sh` commits the served state of each framework checkout an image
  ships (SGLang, aiter), uncommitted and untracked files included, as
  `Hyperloom image baseline` under `refs/hyperloom/image-baseline`, and stops
  on a checkout with unmerged paths or more than 256 MiB of untracked files.
- `install.sh` stops writing `TRACELENS_ROOT`, `GEAK_ROOT`, `GEAK_E2E_RUNNER`,
  `KERNEL_AGENT_ENV`, `HYPERLOOM_RUNTIME_DIR` and `HYPERLOOM_ROOT` into `.env`,
  removes the installer-written copies, and removes `INFERENCEX_PATH` even
  when written by hand; export them in the shell to keep an override. On a checkout another account can write, it keeps the
  Anthropic-side credentials out of `.env`, rebuilds the file (mode `0600`,
  comments dropped) from the assignments such a file may lend, and leaves
  `HYPERLOOM_DOTENV_CREDENTIALS=kernel-agent-env`; export credentials in the
  shell that runs `install.sh`.
- New optional extras: `deepseek`, `hosted`, `runner` and `inferasim`; `forge`
  adds `pydantic>=2.0`. No console script is removed; `hyperloom-multibranch`
  is new.

Session and report consumers have to follow these changes:

- Each round directory is an InferenceSuite run directory with `report.json`,
  `launch.json` and `power.json`; Magpie's result files are gone, `failed_step`
  names the step `report.json` names, and grid results no longer carry
  `benchmark_backend`. `power.json` replaces `gpu_power.json` and
  `gpu_samples.csv`, `report.json` gains a `power` section, and
  `gpu_power_avg_w` and `gpu_power_by_gpu_w` are `null` when `power_valid` is 0.
- `session_breakdown.json` moves from v6.0 to v6.1: `throughput_tok_s_per_gpu`
  in `outcome.baseline`, `outcome.final` and the `baseline` event's
  `measurement` is renamed `output_throughput` (same value; the old key is not
  written), and re-exporting a 6.0 session yields a null
  `outcome.baseline.output_throughput`. Keys no producer populated, keys that
  only carried a constant, and keys only the diffusion latency roofline or the
  `bypass` route wrote are removed (read `ceiling_available` instead of
  `close.roofline_progress.ceiling_kind`, and `selected_role` instead of the
  trace certificate's `production_*` keys); `attributed_gain_pct`,
  `unattributed_gain_pct` and run `empty` are `null` when unmeasured. A GEAK
  candidate that got no rebench verdict now reports `gain_pct` `null` and
  verdict `no_improvement` instead of GEAK's own figure. The full list is
  under Versioning in
  [`session_breakdown.json` integration](reference/session-breakdown.md).
- The workflow-evaluation contract moves from v2 to v3, so new sessions stamp
  `e0d44b1ababa0febbcf0bf1b416b11f62dd52d4ff3a8877cfc8f8c709e774516`. Its stop
  reasons drop `no_kernel_skipped` and gain `host_gpus_unavailable` and
  `prelude_budget_exhausted`; `global` drops `baseline_tput`,
  `cycle_reloop_min_remaining_sec`, `session_remaining_sec` and
  `sweep_closeout_status`; predicate inputs drop `phase_budget_pct` and the
  `skip_to_close` hint; `sweep_reloop` drops `max_cycles`; and ENABLEMENT's
  `integrate_patch` and `specialist` move from LLM-proposable to
  Coordinator-reserved.
- `manifest.json` drops `dependencies.magpie`, `research_lane_capacity` and
  `gpu_specialist_capacity`. `state.json` drops `benchmark_backend`,
  `active_inferencex_path`, `reference_server_args`, `reference_envs`,
  `reference_launch_controls`, `reference_model`, `agentx_epoch`,
  `kernel_opt_task_attempts` and `last_kernel_opt`, among others, and gains
  `base_launch_digest`, `measurement_epoch`, `agentx_accuracy`,
  `baseline_accuracy_definition_digest` and `roofline_runtime_sec`.
- `kernel_optimization_summary.json` moves to `schema_version` 3: `totals`
  counts `claimed`, `adopted`, `rejected`, `failed`, `needs_review` and
  `not_dispatched`, `rejection_breakdown` is replaced by
  `failure_reason_breakdown`, and `by_kernel` has one row per candidate.
- Failure records gain `engine_execute_timeout` and `engine_runtime_death`, so
  a death after the server was ready is no longer filed as `engine_core_init`.

### Added

- **A recipe can bound the SGLang draft its deployment runs.** A benchmark YAML
  may declare `speculative_draft_limits` (`draft_length`, `draft_tokens`,
  `topk`, each with `min` and/or `max`). An explore variant outside them is
  dropped before the round with reason `capability_unsupported`, and a launch
  outside them is refused before its server boots. SGLang's
  `--speculative-num-steps`, `--speculative-num-draft-tokens` and
  `--speculative-eagle-topk` are now read for this check, and the
  speculative-decoding specialist is told the bounds.

- **The base launch can be resolved from InferenceX at a given commit.**
  `--inferencex-commit SHA` on `hyperloom optimize`, `hyperloom-multibranch
  run` and `python -m inference_suite resolve`, and
  `multibranch.inferencex_commit` in a hosted run's campaign spec, read
  InferenceX at that commit (a full sha, or a prefix of 7 or more hex digits; a pull request's
  head commit included) instead of the head of `main`, so a session can match
  a CI run; `base_launch.json` records the full sha. The cookbooks, vLLM
  recipes and ATOM catalog are
  still read at `main`, and a resumed session keeps its frozen base. A commit
  InferenceX does not have, or one that lacks the functions the resolver runs
  (such as `acceptance_length`), is refused before the session starts,
  naming what it lacks; resolve that
  base elsewhere and pass the frozen `base_launch.json` as
  `--reference-script`. A hosted run whose spec value is no commit ends for
  good with exit 2, and one whose commit is refused with exit 1, as `run`
  does. While a commit is pinned, a fixed session never falls
  back to the default table: a source that cannot be read (InferenceX, the
  model card, a release list, a cookbook) refuses it, naming the source and its
  error, and so does no recipe serving it, naming why each source missed. See
  [Base launch](reference/inference-suite.md#base-launch).

- **A fixed-workload session resolves its base launch once and freezes it
  as `base_launch.json`.** The base comes from `--reference-script`, else
  InferenceX `main`, else the framework's cookbook `main`, else a default
  table (with a warning); omitted `--tp` / `--ep` follow the base. Resume
  reads the frozen file and never refetches. A `--reference-script` flag set
  from a shell variable, which used to be dropped silently, now exits 2, as
  does a `# hyperloom-launch-controls:` line or a launch file or default row
  that needs more GPUs than the session has; `--host`, `--port`,
  `--served-model-name` and `--result-*` are recorded as stripped. A JSON
  `base_launch.json` from an earlier session is accepted as the launch file.
  **Upgrade note:** a session whose state predates `base_launch.json`
  cannot be resumed; start a new one. A `current_setting.sh` that records
  launch controls no longer re-feeds as `--reference-script`. See
  [Choose the base launch](how-to/optimize.md#choose-the-base-launch).
  A local checkpoint that records no Hugging Face id is looked up by the
  InferenceX model whose `org/repo` ends in its directory name (compared
  without case), and every later recipe source is read by that id.

- **`optimize --session-dir DIR` creates the new session in the directory you
  name.** The directory must be absolute and below `$USER_DATA_PATH`, and the
  flag cannot be combined with `--resume-from`. Without it the session
  directory is generated as before. A hosted run uses it to find its session
  again after a restart. The same change adds the campaign spec
  (`hyperloom.campaign-spec/v1`) a hosted run is given, with its JSON Schema
  and a field-by-field validator, and keeps a hosted session's
  `<session dir>/archive/` out of the session package.

- **`hyperloom hosted run` runs the legacy `optimize` flow for a hosting platform.** It reads the campaign spec and the host's
  inputs, runs `optimize` in a session directory derived from the run (a restart resumes it), and records how it ended in
  `$HYPERLOOM_STATE_DIR/exit.json`; the host restarts the run only when `final` is not true. Host-only inputs, including the
  new `HYPERLOOM_GRANT_LIMITS`, are removed from the environment before `optimize` starts. The entry runs the packaged `install.sh`
  before `optimize` on every leg; a failing installer ends the run finally. `src/hyperloom/hosted/app_bootstrap.sh`
  builds the wheel of a commit once and installs it, alone, into the image's framework Python environment; `install.sh` resolves the dependencies.

- **A single-lineage AgentX decision round that trails its reference at 1800 s is stopped.** Every AgentX measurement now records `e2e_norm_intvty_p50_1800s`, the median interactivity over the requests finished in the first 1800 s of profiling, and writes the ruling to `agentx_checkpoint.json` in the round's slot. Under `HYPERLOOM_AGENTX_EARLY_STOP=enforce`, the single-lineage default, an explore decision round that trails its reference by more than 2% there is stopped and recorded as REVERT; `shadow` only writes the ruling and `off` skips it. Replayed over 808 graded rounds, the rule stopped 196, all of them REVERT or NO_PROMOTE, and saved 99 of 812 measured hours. Before, every decision round measured for its full hour.

- **Hyperloom recognises the MI350X, the Radeon AI PRO R9600D and the Radeon PRO W7900 and W7900D.** `--gpu-type mi350x`, `r9600d`, `w7900` and `w7900d` are accepted, and each board is identified from its device id or product name at its own roofline figures; a multi-branch session also takes `mi350x`. R9600D, W7900 and W7900D run with `--framework vllm` only, as R9700 does, and a vLLM session on gfx1100 resolves its base launch from the default table. KernelForge runs forge-loop campaigns on gfx1100 with RDNA 3 hardware notes, refuses `aiter`, `flydsl`, `assembly`, `gluon`, `fusion` and `ave` campaigns there, and files gfx1100 experience-KB records under `w7900`. On gfx1201 the automatic AITER FP4/MXFP4 tuners are skipped.

- **KernelForge has an AveLang backend.** `--kernel-backend ave` selects it and backend inference recognises AveLang imports after the existing checks, so Hyperloom dispatches Ave tasks through its Forge path.

- **`HYPERLOOM_DSH_MAX_TOKENS` caps the output tokens of each DeepSeek Harness request.** It reaches the harness orchestration role and the headless `dsh` specialists; unset, the harness sends `max_tokens=256000` as before, which a gateway route with a lower output limit (for example 128000) rejected on every request. A value that is not a positive integer fails at start.

- **TraceLens fusion findings reach forge-fuse and the specialists.** When TraceLens reports any, forge-fuse's discovery prompt and the specialist prompts include them, the fusion workspace holds `fusion_hints.json`, and the fusion result records its path as `fusion_hints`. They are advisory: which chains are fused, and whether discovery runs, is decided as before.

- **A report whose validated gain holds ends with a link to the repository.** `reports/final.md` of such a session closes with one sentence asking for a star on https://github.com/AMD-AGI/Hyperloom. `final.json` and every other report are unchanged.

- **EXPLORE can drop variants before benchmarking them, on three opt-in pre-filters.**
  All are off by default. `HYPERLOOM_MEASUREMENT_STORE` names a store of
  reverted launches, and a variant whose exact launch it already reverted on
  the same stack is not measured again. `HYPERLOOM_EXPLORE_SCREEN=1` runs each
  vLLM variant on a cheap offline probe (one GPU by default) and drops one that
  reads well behind the stack. `HYPERLOOM_EXPLORE_PROJECTION=1` asks InferaSim,
  without a GPU, about variants that only move what it models (parallelism,
  concurrency, the running-batch cap, sequence lengths, dtypes) and drops those
  decisively behind. A dropped variant is a `SKIPPED_DEDUP` row naming its
  reason. The projection needs the `inferasim` extra
  (`pip install 'hyperloom-inference-optimizer[inferasim]'`, which pins
  `amd-infera[projection]==0.3.3`) and Origami, installed separately;
  `HYPERLOOM_INFERASIM_ROOT`, an Infera checkout, takes precedence over the
  package.

- **A `custom` workload can declare its own explore levers.** The
  `--benchmark-scripts-dir` may hold `levers.json`, a JSON list of variants
  that EXPLORE seeds ahead of the model's proposals (a model variant of the
  same name wins) and does not seed again once tested. A declaration no
  restart could apply, such as `extra_envs` given as a string, exits `2` at
  launch. Before, a `custom` workload gave EXPLORE nothing to sweep.

- **Orchestration can dispatch a speculative-decoding specialist.**
  `speculative_decoding_specialist` joins the specialist list and takes
  memory-bound decode; it proposes enabling or tuning speculative decoding as
  benchable variants and may download a matching drafter from Hugging Face.
  Specialists now see `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE` and
  `HF_ENDPOINT` (still not `HF_TOKEN`), so a non-default cache, offline mode or
  mirror applies to them too.

- **The agent roles can run on the DeepSeek Harness.**
  `INFERENCE_OPTIMIZER_AGENT_BACKEND=deepseek` runs orchestration, specialists
  (as `dsh` CLI subprocesses), TraceLens and quantization on the harness, while
  the critic stays on the side the credentials rank first, and Forge takes
  `--agent-backend deepseek` /
  `FORGE_AGENT_BACKEND=deepseek`. It needs the `deepseek` extra and the
  `OPENAI_BASE_URL` / `OPENAI_API_KEY` gateway; `DSH_MODEL` (also read from
  `.env`) picks the model, and the credential ranking never selects the
  harness on its own. With specialists enabled (`--research-lane-capacity`
  above 0), a run refuses to boot when no `dsh` is found.

- **Node power management is recorded and asserted, and a shared NPM node is refused.**
  On ROCm 7.2+ where `amd-smi` has the `node` command, the GPU power record in
  state and in the platform fingerprint gains a `node` block (`npm`,
  `node_power_limit_w`). `--node-power-management enabled|disabled` and
  `--node-power-limit-w W` refuse a launch or resume with exit `2` when the
  node differs or cannot be read; a resume that omits them re-asserts the
  launch's values. On an NPM-enabled node, a session whose visible-device mask
  leaves out some of the node's GPUs exits `2`
  (`node power management is enabled and GPU(s) ... are outside this session`)
  unless `--allow-shared-npm-node` is passed on every launch and resume, which
  records the cards left out as `shared_npm_node`. Each round's `power.json`
  says how much of the measured phase the node was held at its NPM limit, and
  logs it. **Upgrade note:** a session on part of an NPM-enabled node,
  including one launched under 1.1.5, now refuses on launch and resume until it
  is given every card or passes `--allow-shared-npm-node`.

- **`--gpu-sclk-max-mhz` asserts a graphics-clock limit as the power cap is asserted.**
  The limit is read with `amd-smi metric --clock`, recorded in the platform
  fingerprint, and a mismatch or a failed read refuses the launch like the cap
  and performance-level checks. Before, a clock limit set by hand left no
  trace in the records.

- **`optimize` can select plug implementations.** `--plug-profile NAME`
  (`local`, `hosted-gpu`, `hosted-cpu`), `--manifest PATH` (else
  `HYPERLOOM_MANIFEST`), repeatable `--plug SOCKET=IMPL` (over
  `HYPERLOOM_MANIFEST_OVERRIDES`) and `--plug-config SOCKET.KEY=VALUE` choose
  what fills each plug socket. A run that selects something writes
  `plugs.resolved.json` into the session, and `--resume-from` reads it back; a
  refused selection leaves the session untouched. Without these flags nothing
  changes.

- **Multi-node hand-off reads a topology file.** `TOPOLOGY_HANDOFF_FILE`
  names a `topology-handoff/v1` JSON file (`rayjob` or `infera` backend, nodes,
  GPUs per node, service URL and the active backend's block, with credentials
  as file paths) that is read instead of `INFERENCE_OPTIMIZER_MN_BACKEND` and
  `HYPERLOOM_MN_EXT_*`; when both are set and differ, the file wins with a
  warning. `INFERENCE_OPTIMIZER_NODES` and `INFERENCE_OPTIMIZER_GPUS_PER_NODE`,
  which `optimize` exports from `--nodes` and `--gpus-per-node`, win over the
  file, which fills them only when they are unset. A named file that cannot be
  used fails closed, naming the problem, with no fallback to the variables.

- **Specialists can run as a separate non-root account.**
  `HYPERLOOM_SPECIALIST_USER=<account>` runs the `claude`, `codex` and `dsh`
  specialist CLIs as that account, so they cannot read the app's files or
  credentials. Hyperloom then needs root, or `CAP_SETUID`, `CAP_SETGID` and
  `CAP_KILL`. When the account cannot be used, a specialist runs under
  bubblewrap with a warning where bubblewrap works, and fails with a specific
  error where it does not, instead of running with full access. Unset, nothing
  changes.

- **An xDiT roofline profile works on a stock xDiT image.** The profile
  round lays TraceLens's xDiT patch, chosen by the checkout's version, into the
  xDiT checkout (`/app/xDiT` by default). Before, profiling needed an image
  with the patch baked in, and a stock image only logged a warning.

- **The Radeon 8060S has a board identity.** `--gpu-type` and `GPU_TYPE`
  accept `radeon8060s` (gfx1151), detection names it instead of
  `unknown_gpu`, and its roofline uses the figures AMD publishes.

- **KernelForge runs can be collected into the Experience KB.**
  `kernelforge export-runs` writes forge-loop run logs as one document, and
  `hyperloom-kb-collect --mapping kernelforge-v1 --document <file>` collects it
  against the `kernelforge-run-v1` declaration. Each Experience carries the
  Analysis report its plan was built on (`reflection.analysis_report`; one over
  the 32 KB text limit is left out and flagged as
  `analysis_report_omitted`), and lane iterations write `lessons/iter_NNN.md`.
  Before, a later run could not read what an earlier run tried on the same
  kernel.

- **The Forge Implementer can check whether an upstream PR is already in a tree.**
  With `--pr-kb`, a resolved repository and an MCP-capable provider, the
  Implementer gets the `pr_check_inclusion` tool, answered from local git as
  `included` or `unknown` (never "absent"). The same probe runs as
  `python -m kernelforge.knowledge.pr_knowledge.inclusion --tree PATH --commit SHA --pr N [--merge-commit-sha SHA]`,
  exiting `0` for `included`, `3` for `unknown` and `2` on a usage error.

- **forge-fuse records its LLM spend.** A `forge-fuse` run writes
  `<output dir>/llm_usage.json` with the token counts, call count and cost of
  its discovery and authoring sessions, updated after each call; a nested
  forge-loop keeps its own ledger under `forge_experiments/`.

- **The installer provides Node and Codex for GEAK, and preflight checks them.**
  A GEAK session without either is refused at preflight, naming
  `GEAK_NODE_BIN` or `GEAK_CODEX_BIN`, instead of failing in KERNEL_AGENT hours
  in, and a failed GEAK run shows its error in the runner log. A run without a
  GEAK backend only gets a warning.

- **A hosted run can be stopped and watched by its host, and moved between containers.**
  `hyperloom hosted run` starts an MCP server for the host when given a port
  and `HYPERLOOM_MCP_TOKEN` (at least 16 characters, else the run is refused;
  `HYPERLOOM_MCP_HOST` defaults to `127.0.0.1`). Once `stop.json` exists, the
  coordinator stops at its next tick and every later start of the run ends at
  once with a final `exit.json`. The events plug writes the progress stream the
  host tails, a resume in a new container is confirmed before it proceeds, and
  the session package goes where `HYPERLOOM_SESSION_PACKAGE_DEST` points (the
  archive directory in a hosted run). `src/hyperloom/hosted/start.sh` starts a
  run on any framework image from a layer directory, without an image per
  framework.

### Multi-branch sessions

`hyperloom-multibranch` is new in 1.2.0; 1.1.5 had neither the command nor
its package. The entries after the first two record how it changed on `main`
before this release, so a "Before" in them describes an earlier `main` build,
not 1.1.5.

- **`hyperloom-multibranch` runs one optimization session as several branches at once.** The new command (also `python -m hyperloom_multibranch`) is installed with Hyperloom. `run` takes a fixed workload (`--workload inferencex` or `agentx`, `--model`, `--framework`, `--gpu-type` of `mi300x`, `mi325x`, `mi350x` or `mi355x`, `--tp`, `--ep`, `--conc`) and runs PRELUDE (baseline and profile), EXPLORE, FINAL and CLOSE. In EXPLORE up to `--max-branches` (default 4) specialist branches each work in a worktree of their own, a judge opens, holds and stops them, and an integration branch merges what they keep; every test is measured through InferenceSuite on cards the session leases (`--measurement local`, or `hosted` in a grant's sandboxes) and kept or refused by a fixed rule. Before a test gets a card, the Python its patches change is compiled and imported on CPU (`merge_smoke`; `HYPERLOOM_MERGE_SMOKE=off` skips it). `--autonomous-agent SOURCE` adds a third-party agent; an adapter package declared under the `hyperloom_multibranch.component_agents` entry-point group makes it a component agent, and Hyperloom ships none. `run --resume SESSION_DIR` continues a session that stopped before FINAL without a second PRELUDE, and `final-only SESSION_DIR` finishes one that cannot be resumed. A session needs `HYPERLOOM_RSI_ARCHIVE`, the shared archive it reads earlier sessions from and archives its search tree into; `python -m hyperloom_multibranch.blackboard.rsi.lessons` lists and edits that archive's lessons. The image check runs before PRELUDE as `HYPERLOOM_IMAGE_CHECK` says (`warn`, the default, `strict` or `off`). The session writes `MB-manifest.json`, `MB-decisions.jsonl` and `MB-report.md`, stops at `--max-hours` (default 2), `--max-gpu-hours` or `--max-llm-usd`, and exits 0, 1 or 2. See [Running a multi-branch session](how-to/run-multibranch.md).

- **A hosting platform can run a multi-branch session.** `hyperloom hosted run` on a multi-branch campaign spec runs the packaged `install.sh` first (`install.log` in `$HYPERLOOM_STATE_DIR`; a failing installer ends the run for good with exit 2) and then hands the run to `hyperloom-multibranch hosted run`, which reads the campaign spec and host inputs and measures in the grant's sandboxes or, with `placement.measurement: local`, on the app container's own cards. `multibranch.reference_script` becomes the session's `--reference-script` and `hardware.ep` its `--ep`. A restart resumes a session past its baseline, up to the deadline its first start set, or finishes it; a restart that finishes a preempted session whose FINAL chose ends with exit 0 and `stop_reason: finished_after_restart`. EXPLORE ends early enough for FINAL's validation to finish inside the grant. The run refuses a spec that sets `launch.server_args` or `launch.extra_env` (exit 2), a host where its specialists cannot be isolated (exit 1, `agents cannot be isolated on this host`), and a leg whose account cannot write the runner cache or the RSI archive. The hosting layer image now ships a static, non-setuid bubblewrap as `/layer/bin/bwrap`. See [Run hosted multi-branch on a new cluster](how-to/hosted-multibranch-new-cluster.md).

- **A multi-branch session runs on MI325X.** `hyperloom-multibranch run`
  accepts `--gpu-type mi325x`. Before, it refused MI325X, a limit left from
  Magpie, which had no benchmark script of MI325X's own; InferenceSuite
  measures MI325X as it measures the other boards. A Qwen3-14B-FP8 InferenceX
  session (SGLang, TP1) and a DeepSeek-V4.1-Flash AgentX session (vLLM, TP4)
  each ran its 12 hours to the end on an 8-GPU MI325X node.

- **A multi-branch InferenceX session caches its branches' prompts for an
  hour.** Each branch's Claude CLI now runs with `CLAUDE_CODE_PROMPT_CACHE_TTL`
  and `CLAUDE_CODE_SUBAGENT_PROMPT_CACHE_TTL` at `1h` on an InferenceX
  workload, and at the CLI's default `5m` on an AgentX one. Either variable set
  in the session's environment wins, and now reaches a single-lineage
  specialist too. Before, the specialist allowlist dropped both, so every
  specialist ran at `5m`, and a branch that waited past 5 minutes for its test
  wrote its whole conversation to the cache again. In a 12-hour Qwen3-14B
  InferenceX session on MI325X those rewrites were 74% of the cache writes, and
  the same calls priced at `1h` cost 20% less.

- **A component agent's own commits work in an agent sandbox.** An agent
  sandbox's home, bubblewrap or hosted, has no git config and the sandbox kept
  no git identity, so a component agent that commits its own checkpoints
  failed its first commit with `Author identity unknown`, and its share branch
  never started. The sandbox's commands now carry an agent git identity
  (`hyperloom-agent`), apart from the session's own (`hyperloom-multibranch`).

- **A multi-branch AgentX session with a component agent needs no launch
  flag.** `hyperloom-multibranch run --autonomous-agent` now serves every
  component agent (an agent an installed adapter recognises) the session's
  frozen base launch, resolved from InferenceX or the framework's cookbook,
  when neither `--agent-launch` nor `--reference-script` is given. Before, an
  AgentX branch without one of those flags was refused before PRELUDE.
  `--agent-launch` still replaces the base. A credential the base exports,
  which the session froze as its sha256, is served from the host's variable of
  the same name; a host that holds no value of it, or another one, refuses the
  agent at preflight.

- **MB records the AgentX checkpoint's ruling on its explores.** In a
  `hyperloom-multibranch` AgentX session graded on `p50_interactivity`, each
  explore test is now held to the single-lineage 1800 s checkpoint against its
  parent commit's median. By default (`HYPERLOOM_AGENTX_EARLY_STOP=shadow` for
  MB) nothing is stopped: the result carries the ruling and `MB-report.md` gets
  an "AgentX checkpoint" section with how many explores it would have stopped
  and how those ended, to decide on `enforce`. With `enforce` a trailing explore
  is stopped, not retried, and judged REGRESSED with the refusal
  `checkpoint_trailed`. Committed candidates now record
  `e2e_norm_intvty_p50_1800s`, the reference of their children. The mode is read
  once at session start and kept in `MB-manifest.json`. Confirm, integration,
  FINAL, baseline, correctness-fix and retraction tests, and sessions on
  `total_token_throughput`, are not judged. Before, an MB explore that trailed
  its parent ran its full 75-90 min with no early signal.

- **MB's AgentX bar drops to 2% and its guards follow the objective.** A
  `hyperloom-multibranch` AgentX session keeps a change on a 2% gain on its
  objective, not 3%. Under `p50_interactivity` the guards are P90 interactivity
  and total token throughput (new refusal `total_regressed`); under
  `total_token_throughput` they are P90 and P50 interactivity
  (`p50_regressed`). Output throughput no longer guards either, so
  `output_regressed` is no longer given; it is still recorded. A resumed
  session judges by the new rule. Before, a real 2-3% gain was refused, and a
  gain was held back when the replayed window's output-length mix lowered
  output throughput.

- **An MB AgentX session can be graded on total token throughput.**
  `hyperloom-multibranch run --workload agentx --agentx-objective
  total_token_throughput` keeps a change on the round's
  `total_token_throughput` instead of `e2e_norm_intvty_p50`; the default,
  `p50_interactivity`, leaves existing sessions as they were. The retest and
  the error-rate threshold are the same under either objective. The objective
  is recorded in the manifest and kept on resume; FINAL, the dashboard and
  `MB-report.md` name it, and the bar's refusal reads
  `total_throughput_gain_below_threshold`. It writes no Recipe KB record
  (`objective_not_partitioned`): the KB files AgentX records by workload, not
  objective, so its record would sit beside the interactivity champions of the
  same workload. `--agentx-objective` with `--workload inferencex` exits 2.
  Before, an operator had to patch the source to grade on total throughput.

- **An MB AgentX session's Recipe KB record carries both total throughputs.**
  The gate now records `total_token_throughput` on each AgentX judgement, so
  `total_throughput` and `baseline_total_throughput` are written beside the
  interactivity gain; before, they were always left out.

- **An autonomous agent on an AgentX session is given the gate's own round to
  measure with.** The client MB gives it is `python -m inference_suite run`
  against its server, on the session's frozen base, trace corpus, seed,
  concurrency and trace window, writing `report.json` per run. A session whose
  base or staged tools cannot run that round now refuses to start, naming why.
  **Upgrade note:** an `--agent-launch` file no longer takes a `client` key,
  which the gate's client always overrode; a file that names one is refused
  (`keys a launch does not: client`), so remove it.

- **`hyperloom-multibranch run` takes the operator's base launch as
  `--reference-script FILE`, with or without an autonomous agent.** It is the
  file `hyperloom optimize --reference-script` takes: a shell launch script or
  a `base_launch.json` frozen elsewhere, which the session freezes as its base
  without reading any recipe source. Before, only a session with an
  `--autonomous-agent` could take a base, as an `--agent-launch` record. Every
  component agent of such a session is served the frozen base. **Upgrade
  note:** an `--agent-launch` that is a base launch record, or any
  `--agent-launch` beside `--reference-script`, now exits 2; pass the record as
  `--reference-script`. See [Options of
  `run`](how-to/run-multibranch.md#options-of-run).

- **A local `hyperloom-multibranch run`, `run --resume` or `final-only`
  needs its
  model on the machine.** `--model` must be a checkpoint directory, or a hub id
  found under `HL_MODEL_BASE` or in the Hugging Face cache; otherwise `run`
  refuses before the session directory exists, and `run --resume` and `final-only`
  refuse a session recorded with such a model. Before, the local test's server, whose
  home is its sandbox directory, fetched the hub id from the hub.

- **A hosted `hyperloom-multibranch` run publishes its own inference_suite
  staging.** When it pins its runner cache entry, the run now also publishes
  the commit's inference_suite staging into the runner cache; a staging it
  cannot publish ends the run with exit 1 (`CacheBuildError`) before anything
  is measured. Before, a hosted runner found only the staging a `cache_build`
  run had published for the commit.

### Removed

- **Environment variables nothing honours any more.**
  `HYPERLOOM_SPECIALIST_PERMISSION_MODE`,
  `INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES`,
  `HYPERLOOM_ROCPROF_ROOFLINE_ENRICH`, `HYPERLOOM_ROCPROF_ROOFLINE_TIMEOUT_SEC`
  and `INFERENCE_OPTIMIZER_CLAUDE_KERNEL_EFFORT` are no longer read; a box
  that exports one starts normally and behaves as though it were unset.

- **`python -m hyperloom.inference_optimizer.framework_deps` is removed.**
  `install.sh` no longer runs that step, and the install event loses its
  `framework_deps` row, which was always `skipped`.

- **The Critic's unused request kinds and KB batch path are removed.**
  `request.json`, `judge_bundle.json` and `emit.json` lose `kind`
  (`judge_bundle.json` also `messages`, `decision` and
  `kb_priors_for_decision`), and the critic dead-letter queue no longer writes
  `batch_insert.jsonl`. **Upgrade note:** a leftover `batch_insert.jsonl` stays
  queued as an unknown endpoint, so clear it before upgrading a live
  critic-KB deployment.

### Changed

- **A new specialist waits while the untested-proposal queue is full.** In
  FRAMEWORK_AGENT, a `specialist` delegate is refused with `policy_denied`
  rule `specialist_untested_backlog_full` while 8 proposals, twice the
  Coordinator's 4-variant automatic grid, wait for a bench: the
  `untested proposals` count of Orchestration's queued backlog. Proposals in a
  queued or running explore grid, or carried by a finished automatic one, do
  not count. Nothing is refused on the way into the queue: a specialist
  already queued or running still records every proposal it returns. Before,
  Orchestration could keep dispatching specialists faster than the benchmark
  lane benched their proposals, paying a full agent run for each while the
  proposals waited unmeasured until the macro cycle ended. The Coordinator's
  stalled-domain specialist is held back the same way; the operator-defined
  custom specialist is not.

- **Every round runs through InferenceSuite; Magpie and the `bypass` backend
  are removed.** Baseline, explore, grid, KEEP and profile rounds are
  inference_suite run directories with a `report.json`, and `failed_step`
  follows the step it names; a GEAK replay runs its own script and is not one.
  Magpie is no longer installed or imported, and no InferenceX checkout is
  configured any more: `install.sh` fetches the pinned client commit into
  `INFERENCE_SUITE_TOOL_CACHE`, and base-launch resolution reads InferenceX
  itself. `MAGPIE_*`, `HYPERLOOM_BENCHMARK_BACKEND` and
  `INFERENCEX_REF` are no longer read; `MAGPIE_EVAL_LIMIT` is now
  `HYPERLOOM_EVAL_LIMIT`. **Upgrade note:** hosted measurement sheets and
  result manifests are v2 and nothing reads v1 ones any more: a session
  started before this change cannot be resumed, nor its accepted results
  verified again. Build the runner cache (`cache_build`) for the new commit
  before starting hosted sessions on it; a runner whose published staging
  differs from the sheet's refuses with `preflight_mismatch`.

- **Native AgentX accuracy is a GSM8K round.** The baseline, and each
  candidate that clears the KEEP bar, runs GSM8K on a server of its own,
  graded like any lm-eval score; `--no-agentx-accuracy` turns it off. The
  request error rate is no longer an accuracy score, only the measurement's
  validity. The MLPerf backend still grades on its inline score.

- **A custom workload's quality gate may carry raw scores alone.** A
  `quality_gate` without `passed` is judged against the limits it names, else
  `lpips` 0.15, `ssim` 0.85 and `mse` 0.006; before, such a gate failed. A
  baseline whose report carries no `quality_gate` now stops the run with
  `baseline_accuracy_failed` instead of rejecting every candidate.

- **Each round records its power in one `power.json`, with energy and J per token.**
  An InferenceSuite round (and a GEAK replay) now writes `power.json` in place
  of `gpu_power.json` and `gpu_samples.csv`: every card's sysfs power,
  temperatures, clocks, voltage, busy and memory each second from the start of
  the round to its end; each card's energy accumulator at every phase mark;
  every `amd-smi metric` field every 10 s and at every mark; caps, performance
  level and DPM tables at both ends; and the node's NPM power and the CPU's
  RAPL or `amd_energy` counters where the node has them (`null` with a reason
  where it does not). The measured window's energy is computed as InferenceX
  does: `total_gpu_energy_j`, p75/p90 power, `joules_per_input_token`,
  `joules_per_output_token`, `joules_per_total_token`,
  `joules_per_successful_query`, and `power_valid` with InferenceX's reasons. The
  window now opens at the instant the phase mark carries; on an AgentX round it
  opened up to one sample interval late. `report.json` gains a `power` section,
  and a round's measurement gains `gpu_energy_j`, `gpu_power_p75_w`,
  `gpu_power_p90_w`, the four `joules_per_*` figures and `power_valid` (0/1),
  which MB rounds publish. `gpu_power_avg_w` and `gpu_power_by_gpu_w` are now
  the window's energy over its duration and are `null` when `power_valid` is 0,
  so the power budget refuses such a round as unmeasured. A card `amd-smi`
  reads as `N/A` is measured from sysfs. With configured cards, only those are
  asked whether they hold a model; another server on the node's other cards
  does not invalidate the round. `HYPERLOOM_GPU_POWER_INTERVAL_S` defaults to
  1 s; above 1.5 s it is refused and the round samples nothing. AgentX counts
  gain `input_tokens` and `output_tokens`; a report written before reads them,
  and its `power` section, as `null`.

- **Single-lineage AgentX keeps on MB's rule.** `hyperloom optimize` on an
  AgentX replay now keeps a change on a 2% P50 interactivity gain, not 3%, with
  P90 interactivity and total token throughput inside the noise band
  (`HYPERLOOM_PERF_NOISE_PCT`). An output-throughput drop alone no longer
  refuses: `output_regressed` is no longer given, `total_regressed` is new, and
  a correctness fix's guard refusal names `total_throughput` instead of
  `output_throughput`. The GEAK handoff's `workload_spec.acceptance` carries
  `min_gain_pct` 2 and the P90 guard only, since GEAK records no total-throughput
  guard. The 1800 s checkpoint's 2% early stop is unchanged. A resumed session
  judges by the new rule. Before, a real 2-3% median gain was reverted, and a
  gain was reverted when the window's output-length mix lowered output
  throughput.

- **A serving round's server runs in the interpreter its framework checks
  read.** The server now starts as `<framework_python> -m sglang.launch_server`
  (vLLM: `-m vllm.entrypoints.cli.main serve`; ATOM: `-m
  atom.entrypoints.openai_server`) instead of the base's `python3 -m ...` or
  `vllm serve` through `PATH`, which could run another install than the one the
  version probe and flag parser had validated. `launch.json` records the command
  as `entrypoint` and in `argv`, so launch digests of new rounds differ from
  earlier ones; base digests and earlier `launch.json` files read as before.
  **Upgrade note:** a frozen base whose entrypoint is not its framework's
  default (for example a hand-edited interpreter path) is refused as
  `config_invalid`; set `framework_python` to that interpreter instead.

- **A `custom` workload's entrypoint is named after the board itself.**
  `--benchmark-scripts-dir` is searched for `custom_<gpu-type>.sh` with the
  board as given, so `mi308x`, `mi325x` and `mi350x` now look for
  `custom_mi308x.sh`, `custom_mi325x.sh` and `custom_mi350x.sh`; they were
  folded into `custom_mi300x.sh` and `custom_mi355x.sh` before. **Upgrade
  note:** on those boards, rename the script, or keep it the only `.sh` in
  the directory; an unresolved entrypoint exits 2 at launch. See
  [Optimize a custom workload](how-to/optimize-custom-workload.md#the-entrypoint).

- **An explore variant that moves a lever its base launch pins is dropped
  before the round spends a slot on it.** When the base launch simulates
  speculative-decoding acceptance (`rejection_sample_method: synthetic`,
  `synthetic_acceptance_length`, `SGLANG_SIMULATE_ACC_LEN`), its draft and its
  simulated acceptance are pinned: a proposal that changes, removes or unsets
  one of them, or that declares an acceptance length on a base that verifies a
  real draft, is now refused by the round's pre-flight filter as a `serve_flag`
  failure naming the golden-owned key and listed in the round's skipped variants,
  where before it was admitted and then failed as the launch was materialized,
  having already taken a variant slot. The refusal at materialization stays as
  the backstop for launches the filter does not compose.

- **Bump validated vLLM stack to 0.33.0 (ROCm 10 docker) and accept the relocated graph-capture marker.**
  Switches the recommended vLLM image to `docker.io/vllm/vllm-openai-rocm:v0.33.0-rocm100`
  (docs, examples, `models.tsv`, installer hint), bumps the bare-metal defaults
  (`VLLM_VERSION=0.33.0`, `VLLM_ROCM_VARIANT=rocm72`, `VLLM_SOURCE_REF` at the `v0.33.0`
  release commit), and clears the image entrypoint in the Slurm `docker run` branch.
  The TraceLens patch check now looks for `capture_traces` in `vllm/profiler/wrapper.py`
  from vLLM 0.32, where the patch writes it, so a correctly patched tree is no longer
  rolled back. **Upgrade note:** move from
  `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0` to
  `vllm/vllm-openai-rocm:v0.33.0-rocm100` (or match `VLLM_VERSION` and `VLLM_SOURCE_REF` on bare metal).

- **Bump validated SGLang stack to 0.5.23 (ROCm 10 docker).** Updates
  `SGLANG_REF`, `SGLANG_PRETEND_VERSION`, the recommended
  `lmsysorg/sglang-rocm:v0.5.23-rocm10-*-20261111` image tags, `models.tsv`,
  the quick-start `Dockerfile` and the hosted app image `Dockerfile`. These
  builds ship `fastokens`, which the DeepSeek-V4.1-Flash InferenceX recipes
  pass as their tokenizer backend, and answer `/start_profile` without an
  admin API key, so the roofline profile captures a trace. **Upgrade note:**
  move from `v0.5.21-rocm10-mi30x|mi35x-20261008` to
  `v0.5.23-rocm10-mi30x|mi35x-20261111` (or match `SGLANG_REF` on bare metal).

- **A Ray install is certified by Ray's own `ray start` parser**, not by the
  version of `click` it is built on. Preflight and the kernel-tools installer
  hand `ray start` a representative argv of the one Hyperloom runs on the head
  and accept the CLI exactly when it parses, so a `click` release that leaves
  the CLI working no longer triggers a reinstall or a downgrade.
  `RAY_CLI_CLICK_MAX_VERSION` is removed, superseding the 1.1.4 entry that
  introduced it; `RAY_VERSION` is still honoured verbatim.

- **Code Hyperloom and KernelForge share lives in the `hyperloom_common`
  package.** `aiter_jit_cache`, `checkpoint_quant`, `coerce`, `env`,
  `fs_utils`, `git_safety`, `gpu_identity`, `io`, `jsonio`,
  `pr_monitor_urls`, `proctree`, `timeutil` and `unified_diff` move from `hyperloom.common`
  to `hyperloom_common`; `llm_config`, `llm_headers`, `llm_attribution`,
  `llm_request_hooks`, `llm_stability_env`, `token_usage`,
  `reasoning_effort` and `claude_oneshot` to `hyperloom_common.llm`. `kernelforge.tools.durable_io` is removed:
  `hyperloom_common.io` writes with `options=DURABLE` and provides
  `fsync_directory`, `fsync_tree` and `fsync_tree_directories`. Code that
  imports one of these modules must import it from its new place; no aliases
  are kept. An atomic write that asks for `fsync` or `fsync_dir` now fails
  when the flush fails instead of continuing without it. KernelForge
  recognises Radeon 8060S (gfx1151) and keys its experience-KB records under
  `radeon8060s`; gfx1201 records stay under the ISA name, which more than one
  board shares.

- **Specialists run in a bubblewrap sandbox where the host allows it.** On such a host a specialist can no longer read the app's home, credentials or state, nor reach anything but the LLM endpoints and MCP servers its launch names; in a multi-branch session agent and dev sandboxes are confined too. A native build that cannot run as user 65534 compiles under bubblewrap (`bwrap+netns`). The start-up log says once how agents are isolated and the session report carries an `agent isolation:` line. Where neither the specialist account nor bubblewrap works, a run logs one WARNING and agents run unconfined, unless isolation is required (a hosted run, or `HYPERLOOM_SPECIALIST_USER` set), which refuses; a hosted container that maps only its root now starts, with its specialists under the layer's bubblewrap. A specialist that swaps its worktree for a link gets no patch collected (`worktree_refused`), and a host whose user namespace does not map the specialist account refuses the switch (`RunAsError`) instead of running it as root. `python -I -m hyperloom_common.bwrap probe` says whether a host can sandbox agents and why not. **Upgrade note:** a sandboxed `claude` CLI cannot read a login kept only in the app's home: such a launch runs unconfined with a WARNING, or is refused where isolation is required, until its environment carries `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN` or Bedrock credentials.

- **The kernel-agent env file and a shared `.env` no longer hand out credentials.** `install.sh` writes `kernel-agent.env.sh` to `${TMPDIR:-/tmp}/hyperloom-<uid>-<key>/` instead of `$USER_DATA_PATH/runtime/`, removes a copy left there, and stops when `KERNEL_AGENT_ENV` points into a directory other accounts can write; preflight, the documented source line and the slurm job refuse such a file too. A `.env` that is not private to the account, or that sets `HYPERLOOM_DOTENV_CREDENTIALS=kernel-agent-env`, lends no credential; one that another account can write also exports only listed non-credential settings with listed values. Before, every pod of the user could read the credentials in that file and replace what the next launch sourced. **Upgrade note:** run `install.sh` in every new container before `optimize`; one that did not now fails preflight with exit 2 instead of sourcing a file another container wrote.

- **A session on shared storage can be continued by another account of its group.** Below a setgid, group-writable directory closed to others (`2775`), the session's state, database, lock, logs, blackboard, git objects and refs, runner cache and archive are written group-writable, so `--resume-from` or a hosted restart as another account of the group reads and continues what an earlier leg left; git's own config, `HEAD` and hooks stay the creating account's. A session whose `--extra-env` or `--server-args` hold a credential keeps `state.json`, `manifest.json` and its journal owner-only, and a resume by another account is refused naming the file. A directory a leg cannot write is named up front with its owner and mode. Elsewhere modes are unchanged. **Upgrade note:** create the shared directories group-owned with mode `2775` once.

- **The orchestration role no longer stalls inside its own turn.** On the Claude backend an orchestration call may run 32 Claude Code turns instead of 12, so a model that talks between tool calls reaches its intent. On Codex the orchestration session runs without Codex's shell, so it cannot `sleep` in its turn while nothing is dispatched, and a timed-out turn is not retried. On DeepSeek every round opens a fresh conversation, and every DeepSeek Harness runtime compacts at a quarter of the model's window instead of 80%; a 12-hour run had spent about 403M input tokens against 167M on Claude. Codex runs no longer offer the hosted `web_search` tool, whose result a gateway fronting Claude rejected with a 400.

- **An agent backend's launch probe waits up to five minutes.** It waited 60 s, so on a gateway that took most of a minute to answer, a campaign configured for Claude or Codex switched at launch to the fallback provider's default model. A genuinely hung gateway now takes up to 300 s (or the session timeout, if shorter) to fail over; a probe the gateway rejects fails as fast as before.

- **TraceLens is pinned at `bda57c6` (1.1.5 pinned `614c597`).** Analyses gain the speculative-decoding graph merge and perf models for GLM-5.x, DeepSeek-V4-Pro and DeepSeek-V4.1-Flash kernels, and a report whose compute findings lack their Data tables is sent back for repair instead of reaching Hyperloom with no hot kernels. TraceLens's SGLang graph-capture patch now applies to SGLang 0.5.18 through 0.5.21 only: on 0.5.22 and later a profile keeps detailed annotations (before, the patch was reported `unavailable` and annotations were turned off), and the analysis reads SGLang's own `graph_capture_profile/` when `capture_traces/` holds no capture trace.

- **forge-fuse and forge-rewrite name the host's GPU.** On a host other than MI355X, the rewrite controller's operator id and task directory, forge-loop's `--gpu-type` and `--gpu-target`, and the KB identities forge-loop and forge-rewrite file under carry that host's SKU and arch (for example `mi300x`, gfx942) instead of `mi355x`, gfx950, and forge-fuse counts Gemma-style norm kernels toward `launch_bound_share`. **Upgrade note:** KB records such hosts filed under `mi355x` are no longer read back, and `forge-rewrite-by-flydsl` without `--gpu-target` refuses to start where neither `GPU_TARGET` nor rocminfo names exactly one GPU (it assumed gfx942).

- **A degraded trace analysis says so.** A fallback or poor-quality analysis, one with no located kernel source, a phase split the splitter could not confirm, and a roofline computed on vendor peaks after the arch microbenchmark failed (`gpu_arch_json_vendor_peak`) now appear in `trace_health_warnings`, and the roofline result reads `degraded: true`. Vendor peaks go to the session's own `gpu_arch/` directory instead of the TraceLens checkout, where one failed microbenchmark used to leave them for every later session. The trace certificate now certifies the file the analysis reads: `<trace_dir>.selfcert.json` loses its `production_*` keys, and `session_breakdown.json` v6.1 drops `trace_quality.production_selected_role` and `production_would_analyze_split_chunk`.

- **forge-loop plans each round on the decisions in force.** A round plans under the search mode and the Supervisor ruling its own iteration returned, not the previous ones. A resumed plan revision runs under its synthesis session's prompt, which roughly halves revision spend within the cache TTL and gives Codex and DeepSeek revisions their instructions. A lane candidate the tree moved past is filed under `forge_experiments/unmeasured_lanes/` and listed to later planning as `unmeasured_lane_candidates`. The plan critic answers in one structured object, and a review given only in prose is `NOT_REVIEWED` (the draft stands) instead of being read as `REVISE`. `pydantic>=2.0` is a declared dependency of the `forge` extra.

- **`session_breakdown.json` is schema 6.1, and `kernel_optimization_summary.json` is version 3.**
  `throughput_tok_s_per_gpu` in `outcome.baseline`, `outcome.final` and the
  `baseline` event is now `output_throughput` (the same whole-server value; the
  old key is no longer written), keys no producer populated are removed, and
  two unmeasured figures are `null` instead of `0`. A GEAK candidate is
  recorded at close (`close.geak_candidate`), and a kernel's
  `e2e.e2e_gain_pct` comes from the rebench, with the self-reported figure kept
  as `e2e.self_reported_e2e_gain_pct`. In `kernel_optimization_summary.json`,
  `totals` counts `claimed`, `adopted`, `rejected`, `failed`, `needs_review`
  and `not_dispatched`, `rejection_breakdown` is gone and `by_kernel` has one
  row per candidate. **Upgrade note:** readers must accept both throughput
  keys, since 6.0 documents and those exported by 1.1.x keep the old one.

- **A candidate whose patches the tree already holds ends as `already_present`, unmeasured.**
  When every patch of a FRAMEWORK upstream-PR candidate, a framework or
  explore specialist patch, or an enablement round is already in the framework
  tree and the round changes nothing else, `integrate_patch` ends with
  `status: already_present` (one `{patch, check}` row per patch) and no
  restart, benchmark or eval runs. It settles as `failed`, counts as a stall in
  the enablement lane, and writes a framework-KB row with
  `outcome: already_present` when the candidate carries a PR identity. Before,
  such a candidate was applied again and benchmarked as if it were new.

- **Preflight stops when the framework probe gives no clean answer.** Only
  a probe that exits `0` passes. A crash, a timeout or any other exit now exits
  `2` with the probe's stderr, the GPU count `rocm-smi` reports beside the
  masks in force, and `To proceed anyway, set HYPERLOOM_SKIP_FRAMEWORK_CHECK=1`.
  Before, those outcomes warned and admitted a baseline the serving stack could
  not run on any GPU.

- **`hyperloom setup` configures the Codex (OpenAI) side.** Setup offers
  Codex and asks for the OpenAI endpoint, key and model, and the bare-metal
  installer keeps a complete OpenAI side (`OPENAI_BASE_URL`, `OPENAI_API_KEY`,
  `OPENAI_CUSTOM_HEADERS`) on another host than the Anthropic side. In
  Anthropic-only mode it still drops an OpenAI side that is incomplete or whose
  URL equals `ANTHROPIC_BASE_URL`. Before, setup wrote only the Anthropic side
  and removed a separate OpenAI side; a `.env` that still holds a complete
  OpenAI route from an earlier trial now keeps it.

- **Orchestration pays for fewer uncached turns.** Session-constant blocks
  stay in the cached prompt context and the per-tick ones move into the turn
  (`tick` and `agent_last_active` now sit in an `=== Tick ===` block), so Claude
  cache reads carry across turns, and a KERNEL_AGENT tick with nothing to
  decide skips the Orchestration turn; a resumed session always gets one turn
  first.

- **The orchestration model can no longer close a run.** An escalate hint
  outside the vocabulary is refused (`policy_denied`,
  `escalate_hint_not_in_vocab`), and the workflow-evaluation v3 schema drops
  the `skip_to_close` hint and the `sweep_closeout_status`,
  `session_remaining_sec`, `cycle_reloop_min_remaining_sec` and
  `baseline_tput` inputs. `install.sh` now pins each dirty framework checkout
  to a detached `Hyperloom image baseline` commit, so `git status` is clean
  before a session patches it; it stops with the tree untouched on unmerged
  paths or more than 256 MiB of untracked files.

- **ENABLEMENT takes authored work only from its lane.** In ENABLEMENT, a
  `specialist` or `integrate_patch` proposal is refused with
  `phase_incompatible` (hint `ENABLEMENT proposes: target_analysis`), and an
  enablement round that reaches the executor after the lane succeeded or the
  phase moved on ends `skipped`
  (`enablement lane takes no rounds in phase <phase>`), applying nothing.
  Before, such proposals were graded as performance work inside ENABLEMENT and
  kept the phase open after the anchor.

- **A failed PRELUDE baseline stops at the optimization reserve, and baseline pricing leaves the eval out.**
  Once the baseline has failed in a leg and PRELUDE has spent everything
  outside the optimization reserve, the retry is refused before ignition
  (`budget_shortfall.reason = retry_past_the_optimization_reserve`) and the
  session closes with `stop_reason = prelude_budget_exhausted`; before, the leg
  stalled. The cold round's accuracy eval is measured once
  (`eval_runtime_sec`) and no longer priced into each hot pass or retry, so a
  run whose cold round had a long eval keeps its hot pass when the budget
  covers the benchmark alone.

- **A phase is bounded only by its share of the time left.**
  FRAMEWORK_AGENT and KERNEL_AGENT lose their cross-cycle absolute cap and the
  `optimize_budget_cap`, `kernel_budget_cap` and `sweep_budget_cap` exits; each
  entry gets its share of what is left of the session, so a first entry can run
  longer than before. With the concurrency sweep off (AgentX by default, or
  `--no-enable-conc-sweep`), SWEEP's share goes to the phases that run. Before,
  a long session's first cycle could spend the cap, and every later cycle left
  the phase on its first tick and closed as `global_converged` with hours
  unused.

- **The concurrency sweep schedules pairs.** Synthetic rungs run `--conc`
  first, then low to high (MLPerf agentic sweeps stay high to low); the
  baseline arm runs first and the optimized arm only the concurrencies the
  baseline reached; a concurrency neither arm could pair is `skipped` /
  `budget_exhausted` on both. `budget_skip_reason` can be
  `first_arm_share_exhausted`, and `insufficient_remaining_for_variant` is no
  longer given. Before, the most expensive rung ran first and could take a
  long-OSL sweep's whole budget.

- **A speculative draft the pinned stack has no kernel for is refused before launch.**
  On vLLM with the `ROCM_AITER_MLA` draft backend and an FP8 KV cache, a
  two-token draft whose head count AITER builds no kernel for (a 16-head MLA
  drafter, say) is dropped from explore, or refused at launch, with reason
  `capability_unsupported`, and the remedy (pin the draft to `TRITON_MLA`) is
  in the speculative-decoding prompt. Before, the engine booted and died in
  memory profiling, spending a variant slot.

- **The trace-analysis fallback is a `non-llm` route.**
  `HYPERLOOM_TRACE_ANALYSIS_ROUTE` and the `trace_analyze` key `analysis_route`
  take `agent` (default) or `non-llm`; any other value, `bypass` included,
  fails with `invalid_analysis_route`. `tracelens_analysis --route {agent,non-llm}`
  replaces `--use-llm-orchestrator` / `--no-llm-orchestrator`, which now exit
  with `unrecognized arguments`. The `non-llm` route needs a TraceLens
  checkout. **Upgrade note:** the `bypass_trace_analysis` module is gone, and
  `KERNEL_AGENT_USE_LLM_ORCHESTRATOR`, `HYPERLOOM_BYPASS_STEADY_STATE`,
  `HYPERLOOM_BYPASS_CORR_WARN_PCT`, `HYPERLOOM_BYPASS_OTHERS_WARN_PCT`,
  `HYPERLOOM_TRACE_SHAPE_MANIFEST` and
  `HYPERLOOM_TRACE_SHAPE_MANIFEST_MAX_CAPTURES` are ignored.

- **Recipe KB rows key on the framework release, not the build.**
  `0.30.0+rocm723` and `v0.30.0` file under `0.30.0`, and an SGLang nightly
  under its release (`0.5.15.post1`), so a session on a rebuilt image gets an
  exact warm-start hit. Rows written under `+local` ids are reached only
  through the `compatible_framework_version` tier.

- **Install keeps installer-managed paths out of `.env`, and runtime paths follow `USER_DATA_PATH`.**
  After the next install, `.env` no longer holds installer-written
  `TRACELENS_ROOT`, `GEAK_ROOT`, `GEAK_E2E_RUNNER`, `HYPERLOOM_RUNTIME_DIR`,
  `KERNEL_AGENT_ENV` or `HYPERLOOM_ROOT`; sessions read them from the
  kernel-agent env file, so several sessions can install and run from one
  checkout by exporting their own `USER_DATA_PATH`. A
  path the operator set in `.env` is kept, and a `.env` written by
  `hyperloom setup` stays authoritative. `install.sh --check-only` exits
  non-zero with `behind the pinned revisions: <keys>` when a pinned checkout
  is not installed.

- **forge-loop skips the plan critic on rounds that continue a kept route.**
  **Upgrade note:** `forge-loop --resume` of a campaign whose `run_state.json`
  predates this release fails on the missing
  `round_costs.recent[].review_sec`; finish it under the build that started it.

- **Forge's Claude sessions spend fewer tokens.** Sessions make no title
  request, stream summarized thinking (so long planning turns no longer time
  out behind a gateway that drops silent streams), run the lesson-summary turn
  as a read-only continuation of the Implementer's conversation, and in a
  multi-lane round share the lane prompt prefix, starting the other lanes once
  the first answers.

- **KernelForge modules move to the restructured layout.** Workflows live
  under `kernelforge.workflows` (forge-loop in
  `kernelforge.workflows.forge_loop`, with `run_forge_loop` in its `api`
  module), `python -m kernelforge.assembly` is now
  `python -m kernelforge.kernel_backends.assembly`, and `kernelforge.llm` is
  dissolved into the packages that own its modules. Logger names follow the
  modules, so a log filter on an old name stops matching, and
  `kernelforge.__version__` is gone (use
  `importlib.metadata.version("hyperloom-inference_optimizer")`). No alias is
  kept at the old paths.

- **A profile round no longer writes into the installed vLLM or SGLang.**
  Its server imports a patched copy, so other sessions on the host keep the
  stock framework; where the patch cannot be laid, the round runs without it
  and says so (`profile_patch_not_laid`,
  `HYPERLOOM_TRACELENS_PATCH_STATUS=unavailable`). AgentX profile rounds get
  the same TraceLens patching and SGLang kernel-shape setup as fixed-sequence
  ones, and lm-eval is installed once per environment instead of per test.

- **Failures that used to be swallowed now fail their step.** A run whose
  close sequence fails exits as failed after releasing everything, a defect
  while stopping in-flight actions fails the CLI after the session lock is
  released, a raising compute-bound re-profile fails the roofline action, a
  raising round recorder fails the round, and the vLLM ROCm probe fails when
  vLLM's own platform check raises.

### Fixed

- **The automatic bench moves past a grid that ended without a verdict.**
  When the automatic explore grid of the untested-proposal queue's head ended
  without a verdict for any of its proposals (its round failed or was
  cancelled, or it refused every variant), the Coordinator computed the same
  head on every later tick and enqueued nothing, so every proposal behind it
  waited until a new specialist round reordered the queue or the macro cycle
  ended. The next automatic grid now takes the proposals behind it. The ones
  the ended grid carried are not benched again, and Orchestration's
  untested-proposal section names them on a line of their own instead of
  listing them as waiting.

- **A GEAK or Forge reproduction of the measured launch keeps settings whose
  names contain `TOKEN`.** The GEAK handoff's `baseline_env_spec`, the Forge
  handoff's serving context and the reference launch script now carry launch
  settings such as `SGLANG_SIMULATE_ACC_TOKEN_MODE`,
  `SGLANG_USE_AITER_FP8_PER_TOKEN` and `MAX_NUM_BATCHED_TOKENS`, and GEAK or a
  reference script may unset one. Before, any name containing `TOKEN` was taken
  for a credential: the reproduction ran without the setting the original
  launch had, so its baseline was not the measured configuration, and a GEAK
  result unsetting one was rejected. Credentials (`HF_TOKEN`,
  `AWS_SECRET_ACCESS_KEY`) are still never written into these artifacts.

- **A recipe image whose tag names no framework version holds no minimum
  version.** For an untagged image such as
  `lmsysorg/sglang-rocm:muse-spark-open`, the version InferenceSuite assumes
  (the framework's latest release at resolve time) is recorded but not held
  as a minimum, and `launch.json` reports `version_check: assumed_not_held`
  with the installed version. A version read from the image tag is held.

- **The pre-boot GPU check ignores a card another process only opened.**
  Every ROCm process opens all cards and keeps up to a few MiB on each, so the
  check counts a process as holding a card only above 1 GiB of VRAM there (or
  when `amd-smi` reports none); a scheduler or an idle server elsewhere on the
  host does not block every card. See
  [inference-suite](reference/inference-suite.md).

- **Single-lineage warm start replays a recipe that only removes launch
  options.** `hyperloom optimize` now replays a Recipe KB record whose only
  change is `remove_args`, as an MB branch writes one. Before, such a record
  was dropped as having nothing to replay.

- **A GEAK run no longer leaves processes running after KERNEL_AGENT.** The
  KERNEL phase marks every process its GEAK run starts and stops all of them
  when the run returns, times out or is cancelled, including A/B legs started
  with `setsid nohup`, so SWEEP no longer boots next to a GEAK server.
  `GEAK_TERM_GRACE_S` is removed: the outer kill is now only a backstop, and
  run_e2e's chance to flush `result.json` is the `GEAK_FLUSH_GRACE_S` window
  geak_runner already gives it before that kill.

- **`click` is held below 8.3 only where Ray 2.44.1 needs it.** 1.1.5
  installed `click<8.3.0` with Ray every time. Preflight and the kernel-tools
  installer now leave `click` as installed, and hold `click<8.3` once, then
  certify the CLI again, only when the default Ray 2.44.1 CLI fails to import
  with `click` 8.3 or newer (`ValueError: ... is not a valid Sentinel`). Every
  other failure, and any Ray set through `RAY_VERSION`, leaves `click`
  untouched. There is no setting.

- **A baseline the host has no idle GPUs for stops the run.** It fails with `host_gpus_unavailable`, naming each GPU turned down and why, and the run stops on that first refusal whatever `--enablement` says.

- **Ray no longer freezes the optimizer or tears down a working head.** Before connecting in-process, the optimizer probes the Ray head from a subprocess; a head that does not answer within `HYPERLOOM_RAY_CONNECT_PROBE_TIMEOUT_SEC` (60 s) is logged and serving and GPU work run locally, where the process used to freeze until its hard stop. A `ray status` that times out no longer stops a healthy head: the optimizer waits up to `HYPERLOOM_RAY_READY_TIMEOUT_SEC` (30 s) and restarts only a head that is down. `kill -TERM` on an optimizer that has used Ray now runs the whole CLOSE sequence and reaps the run's processes, instead of leaving servers on the GPUs and no close artifacts.

- **Boots no longer wait on the aiter JIT check.** The coverage check before the baseline server and a GEMM-tuning integrate takes seconds instead of up to 15 minutes, checks only the last tuned row of each dispatch key (on the current MI355X image warmup is back to 60-90 s from about 1100 s), and ignores tuned rows for another GPU target. The launch preflight runs under a lock in `/tmp/hyperloom-aiter-serving`, so the GEMM tuning lane and another session's preflight no longer move the same serving modules at once, and a server whose last log line is aiter's `start build` is not stopped by the silence window while it builds.

- **gemm-tune keeps its work when it runs long.** Shapes are tuned in chunks sized from the measured time per shape, finished chunks are kept, and a cut or failed chunk makes the run `partial_output`; before, a batch killed at its cap lost every finished shape. A run whose timings are all zero or infinite fails as `unmeasured_timings` instead of reporting a speedup, and a tuner uses every card it can see. Fused MoE is tuned at the tokens its lookups missed, dense demand is padded as aiter's lookup does, and a tier-3 result is written in the target table's schema.

- **GEAK reports what happened to its run.** A failed GEAK run logs its `error_class` and error, and the phase transition reads `kernel_geak_failed` instead of `kernel_no_more_leverage`. When Hyperloom's outer deadline kills a run, the result GEAK flushed is kept, and with none the run is recorded as `external_timeout`. On native AgentX the handoff declares the replayed workload (`workload_spec` of kind `agentx_trace_replay`, with the corpus and entry count the accepted round replayed), so GEAK searches on the workload Hyperloom grades.

- **Explore rounds are admitted on what they will measure.** An explore with no grid and no seed source is refused at admission instead of failing later as `empty_grid`. A new grid is queued whatever the model names it, and one that repeats queued, running or measured launches is refused naming them, so the agent no longer loops on `duplicate_idempotency_key_running`. Every turn shows each queued and running round's grid and progress, and a variant the server's argument parser refuses is recorded as `server_argv_invalid` with the parser's line. Token-named and `value: 0` framework switches now reach the bench.

- **Integrate and resume finish what they started.** A patch whose context an earlier KEEP only moved now merges instead of costing a re-author round. A resumed session finishes the KERNEL, integrate and PRELUDE work it was in, and dead specialists are retried. A bench that never reached its server ends `failed/bench_unmeasured` instead of a REVERT, and removal-only config levers are benched. A task seeded from `current_best` launches with the flags and environment the winner was measured with, and a warm-started recipe's `AITER_CONFIG_*` paths point at this run's copies of the tables instead of another session's.

- **FRAMEWORK_AGENT phases no longer stall.** After a harvested discovery round the next one is queued, and three empty or failed rounds retire discovery. A specialist whose straggler sweep runs past 60 s is harvested anyway, a slow LLM turn no longer delays booking finished tasks, and an engine that died while serving is classified `engine_execute_timeout` or `engine_runtime_death` instead of `engine_core_init`. The Critic is admitted only while a proposal awaits its verdict; before, its idle turns grew `session_breakdown.json` to 60-100 MB on a 12-hour leg. AgentX baselines now write `runtime_findings.json`, checklist hints follow the session's framework, and a resume that restores its budget no longer prints the 2.0 h default note.

- **Serving launches what was asked, on the cards it was given.** A benchmark point that sets `ROCR_VISIBLE_DEVICES` launches on those cards (`HYPERLOOM_ROCR_AUTOFILL` marks a filled mask), and a round's server is masked for ROCr as well as HIP, so parallel tests on disjoint cards no longer charge each other's cards. Deduping server arguments keeps a single-dash flag such as ATOM's `-tp 1`, which had broken every ATOM sweep baseline. A vLLM nightly that keeps its serve parser in `entrypoints.common` passes the parser check instead of stopping with `framework_version_unknown`. Applying or reverting a Python kernel patch invalidates vLLM's AOT compile cache, and integration and replay start vLLM with the TunableOp hook that produced the candidate. Servers no longer exit with `EADDRINUSE`, a second SIGTERM during shutdown no longer orphans servers, and install no longer refuses a ROCm pod as CUDA-built.

- **TraceLens analyses reach the gateway and read their report.** On a LiteLLM gateway the model id is resolved against the ids the gateway lists, so `claude-opus-5.5` and non-Claude ids such as `gpt-5.6-sol` are no longer rewritten to a form the gateway rejects. A session whose trace report is an `analysis.json` records its roofline snapshot again instead of raising `KeyError`.

- **forge-fuse and forge-loop keep work that was correct.** A forge_loop candidate whose import made the interpreter write bytecode under a protected directory is no longer rejected as `harness_tampered`, and Forge sessions that import benchmark helpers no longer abort as workspace violations. A timed-out fusion run no longer leaves a fused hook in the serving framework, a short fusion share ends with an exported keeper, and a salvaged fusion KEEP keeps its patch for integrate. FlyDSL assembly campaigns start on FlyDSL 0.3.4.1 and later.

- **A slow boot is not taken for a hung one.** Before ready, a server whose log is silent but whose process tree still uses CPU, grows its resident memory or reads its checkpoint counts as active, so a large-checkpoint boot is not stopped as `server_silent_before_ready` at the 1800 s silence bound. The launch holds the framework's own ready timeout (`VLLM_ENGINE_READY_TIMEOUT_S`, `SGLANG_WARMUP_TIMEOUT`) at the suite's. An AgentX round in which aiperf skipped requests as a context overflow is invalid, and a warmup refused for its length fails the candidate as `agentx_context_overflow`.

- **Smaller fixes.** A Codex session on a host where its sandbox cannot write the working roots fails at start naming them. A dsh session log of any schema version is read, so specialist token spend is not lost. Installer archives unpack as the installing account, so a one-id user namespace can install. An HSA queue abort is classified as a GPU fault (`hip_runtime_error`) instead of a missing kernel, and an `ImportError` quoted inside another message no longer files a crash as the candidate's `import_error`.

- **A resumed session reinstalls the kept source patches on a tree that lost them.**
  Before anything is benched, resume installs each kept `source_patch` layer
  from its snapshot, confirms every file, and on a git tree commits them so no
  candidate stashes the stack away; a layer it cannot install stops the resume
  with `patch_recovery_incomplete`, and one without a snapshot is reported as
  `source_layer_not_reproducible`. A promoted warm replay's overlays and a
  Controller KEEP's rewrite are now snapshotted and committed like an
  `integrate_patch` KEEP, and a refused warm replay whose promotion commit
  cannot be undone stops the session with `warm_replay_rollback_failed`.
  Before, a resume onto a fresh tree (a new sandbox, say) benched every later
  candidate against stock source, and even a live session benched
  `integrate_patch` candidates without the warm replay's overlays.

- **`--force-resume --target-gain <higher>` keeps optimizing.** A resume of
  a session that stopped on `target_reached` with a raised target now walks
  the phase chain until the new target or the budget is reached; before, the
  stale marker sent it from ENABLEMENT straight to SWEEP and CLOSE.

- **Explore judges a variant by what it changes about the running launch.**
  A row or warm recipe that restates the current stack is skipped
  (`baseline_noop` / `restates_baseline`) instead of costing a measurement
  (about 68 minutes under AgentX), a row repeating another row's launch is
  recorded as `round_dup` with `duplicate_of`, stack revalidations run again,
  and the automatic bench moves past restated proposals.

- **The operator's launch settings reach every candidate.** `--server-args`
  and `--extra-env` pins now apply to grid candidates benchmarked before a
  baseline is accepted; a KEEP that declares no `remove_args`, `unset_envs` or
  `args_mode` keeps the stack's replace-mode controls instead of reverting it
  to append mode; and cumulative KEEP args no longer drop a base flag that is a
  character prefix of a candidate flag (`--disable-cuda-graph` beside
  `--disable-cuda-graph-padding`). `workload_signature` now follows the AgentX
  switch, so a session with `HYPERLOOM_AGENTX=0`, or with AgentX selected only
  by `--benchmark-config`, measures its earlier explore rows once more after the
  upgrade.

- **`current_setting.sh` holds the whole launch, and `KB_STORE_TOKEN` stays out of launches.**
  `current_setting.sh` now carries every server arg the current best ran with,
  not only the optimizer's delta. `KB_STORE_TOKEN` and the
  `INFERENCE_OPTIMIZER_EXTRA_ENV` blob no longer reach the benchmark YAML, the
  launch evidence or the server's environment. Existing session directories
  are not rewritten; rotate the token if others could read them.

- **A shared recipe no longer carries the publishing run's `--context-length` and `--watchdog-timeout`.**
  A replay gets both from its own launch guards, sized for its shape; before,
  a replay at a larger shape inherited the publisher's cap and was refused.
  Records already in the KB keep the two flags until they are rewritten.

- **The local Recipe KB keeps what worked.** A CLOSE that does not beat a
  row's `best_throughput` keeps its `what_worked`, and `what_failed`
  accumulates across sessions (one row per name and reason), so the next warm
  start's `do_not_repeat` list grows. Remote mode is unchanged.

- **An AgentX benchmark's time limit grows with its concurrency.** The hard
  limit is a 1800 s boot allowance, plus (11 × `CONC` + 1) × 20 s of warmup,
  plus 3640 s of replay and profiling, plus 300 s, and never below 7800 s.
  Before, a sweep rung at `CONC` 28 was killed at 7800 s and recorded as a
  timeout.

- **Warmup and CLOSE roofline limits follow what the session measured.**
  A discarded warmup stops at the phase deadline as `phase_time_exhausted`
  instead of overrunning the phase, and CLOSE sizes its post-optimization
  roofline from the session's own roofline runtime, recording
  `insufficient_session_time` instead of a timeout on a slow board.

- **Profiles capture what they should.** A profile replays captured graphs
  node by node (`DEBUG_CLR_GRAPH_PACKET_CAPTURE=0` on ROCm 7.1.0–7.2.3, and each
  later runtime's own switch), so the trace holds the kernels of every decode
  replay instead of one in many; a speculative-decoding server's capture
  window is divided by the tokens it emits per step, so it writes rank traces
  instead of none.

- **A vLLM server on the Rust frontend is seen as ready.** With
  `VLLM_USE_RUST_FRONTEND=1`, readiness is read from
  `OpenAI server is ready to accept requests`, which arms the post-ready stall
  gate and moves the power and KV recorders into their ready phase.

- **AITER JIT preflight keeps prewarmed modules.** On an image whose
  prewarm wrote `hyperloom-prewarm.json`, baseline and integrate preflight no
  longer move its modules aside (the skip result gains `kept_image_own`), so
  the next SGLang start does not recompile them into the 300 s watchdog, and
  bpreshuffle cktile rows are checked against their own module.

- **Trace analysis runs on no-GPU and packaged hosts and on MI355X.** A
  no-GPU host can import it, a packaged install finds the dependency checkouts
  `install.sh` made, and TraceLens analysis no longer refuses MI355X.

- **Failures are named from their evidence.** A start that dies because
  the engine's capacity does not fit (vLLM's `max_num_seqs` above what its KV
  cache holds) is `engine_capacity` with its cause line instead of
  `server_init_dead`; a `hipError` other than a missing binary is
  `hip_runtime_error` instead of `hip_kernel_missing`; `worker_crash` is
  reported only on a death marker or a signal exit; and a grid round reaped
  before its server was ready names its `server_log_path`.

- **The session report counts and labels what was recorded.** Report lines
  render each field on the axis and scope it was recorded on; dispatches,
  kernel outcomes and critic passes are counted once each (`critic.iterations`
  is ordered by `ts`); `call_count` is filled, a rebound warm replay closes
  once, and kernel gains carry their `gain_axis`.

- **Leases and processes are released when a task ends.** A task with a
  terminal row gives back its lane and GPU leases even when its teardown is
  unconfirmed. After a specialist ends, nothing it started keeps running and
  `heartbeat.json` stops claiming it runs, and a stuck specialist is reaped as
  stale from the agent CLI's own output instead of running to its deadline.
  Queued work the next phase does not admit is no longer spawned in the tick
  that lifts the exit hold, and an expired framework allocation freezes
  admissions and cancels queued work outside the next phase.

- **ENABLEMENT ends once the session anchors another way.** A session that
  enters ENABLEMENT on a boot failure and then anchors a baseline by another
  route leaves ENABLEMENT when that anchor settles and grades later patches
  against the keep threshold, and the rooflines boot with the anchor's server
  args (and its custom benchmark script). With enablement admitted, a baseline
  failure that settles in PRELUDE enters ENABLEMENT instead of stopping the run
  with `baseline_failed`. Before, the phase lingered and the gain could be
  under-reported.

- **An authored patch that fails to apply is re-authored.** Up to the
  apply-fail cap, counted per PR candidate, with the conflict as feedback; a
  raw upstream-PR diff that does not apply starts an authoring specialist
  (`reauthor:1`). Before, the retry was dropped.

- **Long Anthropic calls finish.** Every Anthropic Messages call is
  streamed, so a reply that takes longer than a gateway's 180 s cut completes
  instead of returning 504 (the critic's long reviews were cut), and a
  connection that never came up is dialled again up to twice. Critic and
  proposal-scorer rows in `llm_calls.jsonl` now report `input_tokens` as the
  uncached share and fill the cache columns, like the other rows.

- **Quantization reruns start clean.** Each `--quantize` run works in
  `runs/<run_id>/` of the model's workspace with its own retry budget and fix
  hypotheses. Before, a rerun inherited an earlier run's exported model,
  thresholds and spent retry, so after an OOM the first failure ended it.

- **A malformed `HYPERLOOM_MN_*` restart setting fails before the restart.**
  The numeric multi-node restart settings are read by one rule, so a value
  such as `HYPERLOOM_MN_HEALTH_POLL_S=10s` is refused naming the variable
  instead of failing wherever it was first read.

- **gemm-tune ships better rows.** a8w8 blockscale bpreshuffle tuning finds
  asm split-K rows on the graph batch sizes, decode rows are re-timed in a CUDA
  graph and the fastest eligible candidate is shipped, a tuned row that beats
  the pre-run config below aiter's update bar is kept, tables gain a
  `relNormErr` column, and aiter's compare directory is found under the system
  temp directory instead of `/tmp`.

- **GPU detection ignores generic ISA names.** `forge-fuse`, `gemm-tune`
  and `roofline-ceiling` no longer take `gfx12-generic` or the first of several
  architectures rocminfo prints; they report none (gemm-tune then asks for
  `--gpu-type`). A probe string naming a longer product no longer resolves to a
  shorter board's target.

- **forge-loop and fusion gates.** A failed allclose verdict rejects a
  candidate whatever its SNR, a stop with no tracked change ends as
  `no_changes`, iteration admission honours absolute deadlines, and the
  workspace guard deletes the `.pyc` an interpreter writes under a protected
  directory instead of failing a forge_fusion author session with
  `protected ignored files changed`.

## Hyperloom 1.1.5 release

The [1.1.5 release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.1.5)
follows 1.1.4. AgentX optimization runs on Magpie's native AgentX contract,
new FRAMEWORK_AGENT and KERNEL_AGENT work must fit what is left of the phase
budget, and a phase transition lets running work finish instead of cancelling
it. The release adds GPU power budgets and opt-in power settings, an operable
Experience KB with the `hyperloom-kb` command, an operator-defined specialist
the Coordinator runs itself, the PrimaTune predictor, Radeon AI PRO R9700
support with vLLM, a Claude–Codex relay for a stalled forge-loop search and a
Claude subscription credential path, and explore no longer re-measures a
launch whose answer the session already has. Sessions started by 1.1.4 or
earlier cannot be resumed, a workspace Experience KB starts empty, the
accuracy eval can no longer be turned off, and several options, environment
variables and defaults move; see "Before upgrading from 1.1.4" below. The
per-change history is on the
[Releases](https://github.com/AMD-AGI/Hyperloom/releases) page.

### Before upgrading from 1.1.4

Finish in-flight sessions under the build that started them. Every session's
`storage/coordinator.db` now records schema version 7 and there are no
in-place migrations, so `optimize --resume-from` on a session started by 1.1.4
or earlier exits `2` before it touches the session, with
`coordinator.db is at schema version <N>, older than this build's 7: sessions created before this release cannot be resumed`.
`kernelforge forge-loop --resume` likewise refuses a 1.1.4 checkpoint with
`run state cumulative has unknown fields: api_errors, orchestration_errors`.
See [Upgrade Hyperloom version](reference/upgrade.md).

The workspace Experience KB starts over. Its service now keeps
`database.sqlite3`, `<kb_id>/records/` and `<kb_id>/files/` in its data home
and leaves a 1.1.4 home's `kb.sqlite3`, `sync.sqlite3` and `canonical/`
unread, so the first launch or `ensure` after the upgrade restarts it on a
new, empty KB and Experiences recorded under 1.1.4 stop reaching prompts.
Experiences are now `schema_version` 2 (with `hyperloom-kb.collect.v2`
mappings), which a 1.1.4 service rejects, as a 1.1.5 service rejects 1.1.4
records, so upgrade a global KB together with the workspaces that sync with
it; `POST /v1/pull` now requires a `schema_ref`. An attempt whose change
content exceeds 32 KiB is skipped at collection with
`schema validation failed`.

Options and module paths 1.1.4 accepted are removed or refused. The parser is
strict, so a launch, resume or install command that still carries one exits
with `unrecognized arguments` (or exit `2`) before the session starts.

| Removed | What to do |
|---|---|
| `--no-eval` | Delete it; the accuracy eval always runs (see *Removed*). |
| `--pr-feed-window-days` | Delete it; nothing read it. |
| `--local-kb-root` | Set `KNOWLEDGE_LOCAL_ROOT` instead; 1.1.4 already warned that the flag was deprecated. |
| `kernelforge forge-loop --return-after-read-kb` | Delete it. forge-loop no longer has a warm-start-only mode that returns before iteration 1 when a KB solution applies and improves. |
| `hyperloom multi-node kernel-bench` | Removed with no replacement; nothing in the optimizer called it. |
| `install.sh --only-aiperf` | Delete it; the installer no longer installs the AIPerf client. |
| Under native AgentX (`HYPERLOOM_AGENTX=1` without the MLPerf backend, or `--benchmark-config`): `--isl`, `--osl`, `--profile-osl`, `--max-model-len`, `--reference-script`, `--quantize`, `--quantize-scheme`, `--enable-conc-sweep`, `--nodes` other than `1`, a framework other than `sglang` / `vllm`, and `--extra-env` pins of `ISL`, `OSL`, `RANDOM_RANGE_RATIO`, `MAX_MODEL_LEN`, `EP_SIZE` or any `AIPERF_*` / `AGENTX_*` / `AGENTIC_*` key | **Exits 2** at launch. Request lengths come from the trace corpus: start one session per concurrency, pass expert parallelism as `--ep`, quantize the checkpoint beforehand, and put other Magpie settings in a `--benchmark-config` YAML. `--docker-image` without native AgentX also exits `2`. The MLPerf backend is unaffected. |
| `python -m kernelforge.mcp_server.tools.bench`, `python -m kernelforge.mcp_server.pr_stdio_server`, `python -m kernelforge.mcp_server.probe_stdio_server` | Use `python -m kernelforge.evaluation.benchmark.runner`, `kernelforge.knowledge.pr_knowledge.mcp_server` and `kernelforge.orchestrator.specialist_probe_server`. KernelForge builds its own server commands, so only an MCP configuration written by hand needs the new paths. Code that imports a moved module (for example `kernelforge.loop.scoring` or `kernelforge.durable_io`) must import it from its new place under `kernelforge.evaluation`, `kernelforge.tools` or `kernelforge.knowledge`. |
| `tools/kbmine` (`pip install -e tools/kbmine`, `python -m kbmine.cli`) | Nothing replaces it; a 1.1.4 checkout still carries it. It was never part of the `hyperloom` package and the optimizer never used it. |

Environment variables fail differently: nothing refuses most of them, so a box
that still exports one starts normally and behaves as though it were unset.
Find them by reading launch scripts, `.env` and the kernel-agent env file. The
settings that do stop the run are marked below.

| Removed or changed variable | What to do |
|---|---|
| `RUN_EVAL` exported in the shell, pinned through `--extra-env` or set in a benchmark YAML template | Delete it; it is ignored, and the GSM8K eval runs wherever 1.1.4 ran it by default (an AgentX template keeps its own value). A box that set `RUN_EVAL=false` to get past a broken eval must fix the eval: a baseline whose eval fails can stop the run with `baseline_accuracy_failed`. |
| `AGENTX_DATASET`, `AGENTX_CANONICAL_DATASET`, `AGENTX_PROFILE_WINDOW_S`, `AGENTX_TRACE_FIRST_FILE_TIMEOUT_S`, `AGENTX_TRACE_FLUSH_TIMEOUT_S`, `HYPERLOOM_ALLOW_UNVERIFIED_SUBMISSION` | Delete them; nothing reads them. Magpie's pinned InferenceX picks the AgentX corpus, the benchmark YAML sets profiling, and a native AgentX KEEP needs a valid Magpie report. Passed through `--extra-env` under native AgentX, the `AGENTX_*` names **exit 2**. |
| `AIPERF_BIN`, `INSTALL_AIPERF` and the other `AIPERF_*` installer variables | Delete them; AIPerf is no longer installed or run. |
| `HYPERLOOM_AGENTIC_BACKEND=aiperf` | Drop it: any value other than `mlperf` now selects native Magpie AgentX. |
| `INFERENCEX_PATH`, `INFERENCEX_REF` on a native AgentX run | No longer used there: Magpie resolves its own pinned InferenceX and preflight skips the InferenceX clone. Set `benchmark.inferencex_path` in the YAML or `MAGPIE_INFERENCEX_PATH` to use your own checkout. Synthetic and MLPerf runs still read both. |
| `HYPERLOOM_SGLANG_REPO_URL`, `HYPERLOOM_SGLANG_REF`, `HYPERLOOM_SGLANG_INDEX_URL`, `HYPERLOOM_VLLM_ROCM_INDEX_URL` | Delete them; nothing reads them. Enablement Rung 3 installs the runtime the specialist requests (see *Changed*). |
| `HYPERLOOM_ENABLEMENT_ORIGIN_ALLOWLIST`, `HYPERLOOM_ENABLEMENT_INDEX_ALLOWLIST` | Delete them; nothing reads them. Rung 3 no longer restricts the git origins or pip indexes a specialist names, and no setting replaces the two allowlists; `--enablement off` keeps enablement from running at all. |
| `HYPERLOOM_LOCAL_KB_ROOT` | Set `KNOWLEDGE_LOCAL_ROOT` instead. A box that still exports only the old name uses the default knowledge root. |
| `TRACELENS_SOURCE_ROOT` | Delete it; nothing reads it. |

Defaults and gates that move, so the same command behaves differently:

- **Phase budgets are hard limits for FRAMEWORK_AGENT and KERNEL_AGENT work.**
  Work that does not fit what is left of the phase is refused, skipped or
  stopped at the phase end (see *Changed*), so a session with less than an hour
  of usable time left no longer admits a patch specialist, and
  `HYPERLOOM_GEMM_TUNING_TIMEOUT_SEC` still sizes a GEMM tuning run but no
  longer forces one into a KERNEL slice that cannot fund it. The LLM's
  `extend_explore_budget` and `extend_kernel_budget` hints, each of which added
  5 points to a phase's share up to 80% of the session, no longer move budget.
- **The default Claude model is `claude-opus-5-5`.** A run that sets no
  `CLAUDE_MODEL` or `--claude-model` now orchestrates, reviews and writes
  reports on `claude-opus-5-5`, and preflight falls back to `claude-opus-5`
  only where the gateway catalog lacks it
  (`'claude-opus-5-5' not in gateway catalog; falling back to 'claude-opus-5'`),
  so results and token cost can differ from the same command under 1.1.4. Set
  `CLAUDE_MODEL=claude-opus-5` to keep the 1.1.4 model; the slurm
  `proxy.env.template` still does.
- **The default Codex model is `gpt-6-astra`.** A run that sets no
  `CODEX_MODEL` (Hyperloom's `--codex-model`, the Codex critic, the TraceLens
  Codex runner and KernelForge's Codex backend) now uses `gpt-6-astra`, with
  `gpt-5.6-sol`, `gpt-5.5` and `gpt-5.4` as preflight fallbacks, where 1.1.4
  used `gpt-5.6-sol`; the slurm templates set `CODEX_MODEL=gpt-6-astra` too.
- **New forge-loop campaigns switch agent providers on a stall.** When a second
  provider passes its startup check, a stalled search is handed to it instead
  of to the Supervisor (see *Added*). `FORGE_AGENT_CLI` is used only when it
  identifies as the selected provider's CLI and is otherwise ignored with a
  warning, and `forge-loop --resume` refuses an `--agent-backend` or
  `FORGE_AGENT_BACKEND` that differs from the campaign's primary.
- **Variants keep the operator's `--no-enable-prefix-caching`.** A session
  whose `--server-args` pass `--no-enable-prefix-caching` now measures every
  explore and sweep variant with prefix caching off, as its baseline runs,
  where 1.1.4 dropped the flag from every composed launch and credited the
  variants with the cache; such a session reports lower gains than the same
  command did under 1.1.4.
- **AgentX baselines need the prefix cache on.** An AgentX run whose baseline
  server runs with the prefix cache off now stops at the baseline with
  `stop_reason=baseline_failed` (`agentx_baseline_prefix_cache_off`). Drop
  `--disable-radix-cache` and `--no-enable-prefix-caching` from
  `--server-args`, and on ATOM pass `--server-args "--enable_prefix_caching"`.
- **Synthetic sessions no longer measure context-copying speculative drafts.**
  Explore refuses before launch a variant that introduces or changes a vLLM
  `ngram` / `ngram_gpu` / `suffix` or SGLang `NGRAM` draft
  (`speculative_context_copy_on_synthetic`), so a synthetic-benchmark session
  can no longer KEEP one, where 1.1.4 could keep it on a gain that does not
  hold on real text. Model-based drafts, a base launch that already runs a
  copying draft and AgentX sessions are unaffected.
- **Explicit context and memory limits are checked before the first
  baseline.** Preflight's new `validate_server_args` step stops a launch whose
  `--max-model-len` or `$MAX_MODEL_LEN` exceeds the context the model's
  `config.json` declares (`max_position_embeddings` or an equivalent key, read
  without RoPE scaling, so a RoPE-scaled model run above its raw context is
  refused), and on vLLM a `--server-args` `--gpu-memory-utilization` outside
  `0 < value <= 1` or `--max-model-len` above that context (`auto`, `-1` and
  unit suffixes still pass). It fails with
  `<flag> must satisfy 0 < value <= <N>; got '<value>'`, where 1.1.4 launched
  explicit values as given. `--tp` now takes only a positive integer.
- **Quark and compressed-tensors sessions take the checkpoint's format as
  their precision.** A new session started without `--precision` or
  `PRECISION` on such a checkpoint records and exports the format the
  checkpoint nests (for example `fp4` for Quark MXFP4, `fp8` for
  compressed-tensors FP8), where 1.1.4 used `quark` or `compressed-tensors`,
  so its workload fingerprint and Experience KB records no longer match those
  of earlier sessions on the same checkpoint.
- **GPU power checks and sampling follow the cards the mask names.** The
  `--gpu-power-cap-w` / `--gpu-perf-level` assertion and the measured-phase
  power sampler translate `ROCR_VISIBLE_DEVICES`, then `HIP_VISIBLE_DEVICES`
  (or `CUDA_VISIBLE_DEVICES`), to amd-smi card numbers through each card's KFD
  node, where 1.1.4 read mask numbers as amd-smi indices and so, on
  MI300X/MI355X, checked and sampled other cards than the session's; 1.1.5
  power readings are not comparable with those. A mask that is set but empty,
  is not plain comma-separated numbers (for example GPU UUIDs), repeats a card
  or names one that does not exist now makes a declared assertion exit `2`,
  where 1.1.4 fell back to every card; the sampler then samples every card and
  logs `gpu_power: <reason>; sampling every card`. `--gpu-perf-level` takes
  only the levels `amd-smi set` accepts, so the
  `AMDSMI_DEV_PERF_LEVEL_<LEVEL>` spelling 1.1.4 normalized now fails argument
  parsing.

Installation prerequisites move:

- `install.sh` pins Magpie to `6a1ee26f9053363d3750b43e3d5afb5d1c4ce84e` and
  replaces an installed Magpie at any other revision unless `MAGPIE_PATH`
  names your own tree, and preflight likewise reinstalls a Magpie whose
  revision does not match (`Preflight: Magpie revision is missing; installing ...`),
  where 1.1.4 reinstalled only one that would not import. The minimum
  supported revision rises from `e6833b81` (Magpie v0.2.0) to
  `1eb3df69de9cb26cb5c8109e43371dd1abb9bbed`, the first whose vLLM client
  scripts carry the `--tokenizer-mode` hook: with an older `MAGPIE_REF` or
  `MAGPIE_PATH`, a DeepSeek-V4 round whose checkpoint the installed
  transformers cannot resolve fails with `client_tokenizer_unpatchable`. The
  AIPerf client is no longer installed.
- TraceLens is pinned to `614c597bc0c66cabf6a15152037324acc7386090`, and
  preflight and the kernel-tool installer now require
  `TraceLens_generate_perf_report_from_config` on `PATH` instead of
  `TraceLens_generate_perf_report_pytorch_inference`. Rerun `install.sh`;
  otherwise a session that runs kernel work or the roofline refuses to start
  (`ERROR: TraceLens CLI(s) not on PATH`, exit `2`).
- An interactivity-graded AgentX session that delegates to GEAK needs a GEAK
  that accepts `E2E_METRIC=e2e_norm_intvty_p50`, and KERNEL's wait for the
  handed-off GPUs relies on a GEAK that frees them on exit; GEAK `main`, the
  default `GEAK_REF`, has both. Rerun `install.sh` and move a pinned
  `GEAK_REF`: on an older GEAK every replay of such a session fails with
  `measured <basis>, not the requested e2e_norm_intvty_p50` and the session
  gets no GEAK candidate.
- Multi-node preflight always probes `lm_eval[api]` and installs the pinned
  build when it is missing, where 1.1.4 skipped that step for a session with
  the eval off.

Session and report consumers have to follow these changes:

- `state.json` drops `custom_specialist_dispatched` and
  `enablement.build_novelty`, and gains `custom_specialist_only`,
  `analysis_failures` and `predictor_asked_keys`; roofline task params gain
  `recipe_generation`.
- For AgentX sessions, `manifest.json` `workload` keeps
  `benchmark_mode: agentx` and records `benchmark_config` (the archived
  `benchmark.yaml`) in place of `corpus_loader`, `corpus_entries`,
  `corpus_duration_s`, `isl_distribution`, `osl_distribution` and
  `prefix_cache_hit`.
- `session_breakdown.json` drops the top-level `robustness` key and
  `close.robustness`, and `robustness_escalated` leaves the stop-reason
  vocabulary; all were always empty. On interactivity-graded AgentX sessions
  the stack-ledger `baseline_tput`, `contribution_pct`, `cumulative_gain_pct`,
  `attributed_gain_pct`, `chain_total_gain_pct` and `reconciliation_gap_pct`
  are now measured on the graded axis (see *Fixed*) and are not comparable
  with 1.1.4 exports.
- Reports, summaries and the roofline snapshot label serving throughput
  `tok/s` instead of `tok/s/GPU`; `throughput_tok_s_per_gpu` in
  `session_breakdown.json` keeps its name and holds the same server aggregate
  as in 1.1.4.
- On a multi-node kernel apply, the apply result and `manifest.json` carry
  `jit_build_backup` as the bare `{"status": "remote"}` marker, without
  `per_node` / `per_host`, and `multinode` no longer carries `per_node`; read
  `multinode.records_by_host` instead. Backup-ledger records for a rename
  source carry `revert_action: "restore"` instead of `"restore_old"`, and
  snapshot `manifest.json` entries no longer carry `binary`.
- `forge_experiments/pr_refs/index.md` renames its `merged` column to `state`,
  and new `snapshot.json` entries record `state`, `draft`, `merged_at` and
  `base_ref` in place of `is_merged`.
- The v1 workflow-evaluation contract and its schema are removed. The v2
  contract gains the `baseline_over_power_budget` stop reason and drops
  `robustness_escalated`, so its digest changes and new sessions stamp
  `9887b33c30aeada4359b3b3d38b817b9dec1655df84e84a98911e5a1a08871c7`, and
  the v2 schema's `hint` enum drops `extend_explore_budget` and
  `extend_kernel_budget`.
- The session breakdown's roofline trace-validation summary drops
  `recommended_steady_state_mode`, `recommended_splitter_mode`,
  `modes_that_would_fail`, `steady_state_forecast` and
  `hot_kernel_list_would_be_suppressed`, trace self-certification
  certificates (`<trace_dir>.selfcert.json`) no longer carry a split forecast,
  and a selected chunk with no GPU work is reported as
  `steady_state_chunk_low_quality` instead of `steady_state_chunk_empty`.
- Provenance built from the environment reads the product from
  `TARGET_GPU_TYPE`, so MI325X and MI308X sessions record `gpu_type`
  `mi325x` / `mi308x` where 1.1.4 recorded the runner label `mi300x`, and a
  session on a `setup.py develop` checkout or an isolated vLLM venv records
  its real framework version (for example `0.29.0+rocm100`) instead of
  `unknown_version`; their KB keys no longer match knowledge 1.1.4 filed for
  the same setup.
- Log lines and error classes that move:
  `explore dedup: payload=N → runnable=N (round_dup=N baseline_noop=N)` is now
  `explore grid: payload=N → runnable=N (leading baseline_noop=N)` and counts
  only the restatements dropped before the round (read the result's
  `skipped_dup` for every skip); KernelForge's
  `[kb] experience write skipped (...)` is now
  `experience KB publication failed for <commit>: ...`; and enablement no
  longer emits `localization_rung5_deferred` or `localization_fetch_failed`.
- The enablement constraints file is `rocm_constraints.txt` (was
  `torch_constraints.txt`).
- A Recipe KB still only in the legacy `$USER_DATA_PATH/kb` (or
  `/workspace/hyperloom/kb`) location, which 1.1.4 copied into the knowledge
  root on its first local start, is no longer read or migrated.

### Added

- **Power budgets: `--max-power-w` and `--max-per-gpu-power-w`.**
  `--max-power-w N` caps the sum of the serving GPUs' measured-phase mean
  power, and `--max-per-gpu-power-w GPU_ID W`, repeated per card, caps one
  card. Every KEEP path (explore, `integrate_patch`, warm replay, the lift and
  GEAK replays, which are now power-sampled) refuses a candidate over a budget
  or without a per-card reading (`power_budget_exceeded`,
  `gpu_power_budget_exceeded`, `power_unmeasured`), and a baseline that breaks
  a budget or has no reading stops the run with `baseline_over_power_budget`.
  A launch or resume exits `2` when a listed GPU is not one the session can
  use, the per-GPU limits add up to more than `--max-power-w`, power cannot be
  sampled (multi-node, `xdit` or `custom`, `HYPERLOOM_GPU_POWER_SAMPLING=0`,
  no `amd-smi`), or a resume changes either value. Rounds record
  `gpu_power_by_gpu_w` beside `gpu_power_avg_w`, and `gpu_power.json` lists
  cards that may be serving but could not be read under `unread_gpus`.
- **`--apply-gpu-power-settings`.** The session sets the declared
  `--gpu-power-cap-w` / `--gpu-perf-level` itself, once at launch and only on
  the cards of its integer `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES`
  mask, and restores the originals when it ends, including on SIGTERM or
  SIGHUP; the platform fingerprint records them as applied by Hyperloom. It
  needs root, and refuses with exit `2` and nothing set without root, on
  multi-node, without an integer mask, on a card not in SPX mode or already
  holding more than 2 GB of VRAM, on a fractional cap, and on a perf-level
  change from `determinism` or `manual`. The originals are kept under
  `/var/tmp/hyperloom/gpu_power_settings/` (mount the host's
  `/var/tmp/hyperloom` into each container): after a SIGKILL the next launch
  with the flag restores those cards first, and a launch without it warns.
  Pass the flag again on `--resume-from`; a resume without it refuses and says
  so.
- **Each round reports its power-throttle share and sustained clock.**
  `gpu_power.json` gains per-card `sclk_mean_mhz` (mean graphics clock while
  busy), `sclk_max_mhz` (the clock limit) and `power_throttle_fraction` (the
  share of the measured phase held back by the power limit, from
  `amd-smi metric --violation`), and top-level `power_throttled_gpus`,
  `throttle_query_errors` and `throttle_last_error`. A round with a serving
  card throttled for at least 10% of its measured phase logs
  `gpu_power: GPU(s) <ids> were power throttled for <share> of the measured phase, at a mean clock of <MHz>`.
  The fields are additive (`schema_version` stays `1`), a share not read
  across every measured window is `null`, and no gate or decision reads them.
- **Operable Experience KB state, and the `hyperloom-kb` command.** Each
  Experience KB service has a `kb_id`, and the new `hyperloom-kb` console
  script (`health`, `push`, `rebind`, `pull --schema`, `labels`, `label`,
  `restore`, `exclude`, `include`, `exclusions`, `list`, `export`) operates
  the service `HYPERLOOM_KB_URL` names, printing JSON and exiting `1` on a
  refused request or an incomplete sync and `2` when the URL or token is
  unset. `exclude --reason` hides an Experience from every read with the
  reason kept in its history, `restore` returns to a labelled state, a pull
  first labels the state it replaces (`before_pull`) so restoring that label
  undoes it, and a sync with a URL that now answers as another KB is refused
  until `rebind`. Every service answers `/livez`, `/readyz` and `/metrics`
  without a token and logs JSON lines carrying the `X-Request-ID` it returns,
  and prompts get only an Experience's knowledge fields, with a change's
  content inline. `HYPERLOOM_KB_HOME` names the local service's data home
  (unset, `$USER_DATA_PATH/experience-kb`), so runs on different hosts that
  share a `USER_DATA_PATH` can each start their own service. The
  `hyperloom-global-kb` skill is no longer installed and `hyperloom-setup` no
  longer asks about a global KB, which stays opt-in through
  `HYPERLOOM_GLOBAL_KB_URL` / `HYPERLOOM_GLOBAL_KB_TOKEN`.
- **PrimaTune predictor for FRAMEWORK_AGENT.** `--primatune-endpoint URL` (or
  `HYPERLOOM_PREDICTOR_ENDPOINT`) asks a PrimaTune service for launch
  configurations once per decision point of an `sglang` or `vllm` session,
  and `--primatune-mode` (`HYPERLOOM_PREDICTOR_MODE`) picks `shadow`, the
  default once an endpoint is set, which only logs the answers, `active`,
  which queues up to six rows per answer ahead of the specialists' proposals
  under the same KEEP threshold and accuracy gate, or `off`. In active mode
  FRAMEWORK's first orchestration turn waits up to five minutes for an
  answer, a proposed source change becomes a patch-specialist mandate, and
  credited variants carry provenance `primatune`; a request times out after
  `HYPERLOOM_PREDICTOR_TIMEOUT_SEC` (default 900 s). The start-up banner
  prints a `predictor` line with the mode and endpoint, and neither flag is
  persisted, so a `--resume-from` relaunch needs them again unless the shell
  exports the variables. Without an endpoint nothing changes. See
  [PrimaTune predictor](reference/primatune-predictor.md).
- **Radeon AI PRO R9700 (gfx1201) with vLLM.** An R9700, detected by its
  product name or given as `r9700`, runs single-node vLLM through Magpie's
  gfx12 runner (`runner_type: gfx12`, `vllm_gfx12.sh`), reusing one server
  across rounds as the MI boards do; any other framework exits `2` at launch
  with `Magpie runner 'gfx12' is currently validated only for vllm`. The
  roofline and bypass roofline use AMD's published peaks, labelled `vendor`
  (an MXFP4 model gets the memory ceiling only), no targeted build is
  attempted on the board, and the GEAK hand-off carries `expected_target` /
  `expected_gfx`. MI325X and MI308X runs now materialize
  `runner_type: mi300x` and pin `vllm_mi300x.sh`, where 1.1.4 pinned a
  per-board script Magpie does not ship.
- **forge-loop relays a stalled search between agent profiles.** Every new
  campaign builds one profile per registered provider (`claude`, `codex`),
  with the `--agent-backend` provider as primary, and checks each with one
  minimal model call before the `--max-hours` budget starts. After three
  non-improving iterations from the active profile (infrastructure, API,
  device-contention and integrity failures do not count), the next clean
  iteration boundary hands the whole search to another available profile, so
  with Claude and Codex both available the search alternates between them
  until a KEEP; the Supervisor runs only when no other profile is available. A
  profile without a gateway of its own borrows the other provider's (Codex
  appends `/v1` to the `ANTHROPIC_*` URL), and the result's `agent_relay`
  field and the `relay_*` events in `forge_experiments/events.jsonl` record
  every switch.
- **A Claude Pro or Max subscription can be the Anthropic credential.**
  `hyperloom-setup` asks whether to use an Anthropic API key or a
  subscription. The subscription path has you run `claude setup-token`
  yourself, writes `CLAUDE_CODE_OAUTH_TOKEN` and
  `ANTHROPIC_BASE_URL=https://api.anthropic.com` to `.env`, and warns that an
  `ANTHROPIC_API_KEY` or `ANTHROPIC_AUTH_TOKEN` already in `.env` or the shell
  outranks the token and bills API credits. A launch whose only Anthropic
  credential is `CLAUDE_CODE_OAUTH_TOKEN` now starts, where 1.1.4 stopped in
  preflight with `critic model ... did not answer ... cannot run inside an active event loop`
  (exit `2`) unless `--critic-mock` was passed.
- **The session breakdown counts the framework phase's bench runs.**
  `session_breakdown.json` gains `outcome.variants` (`bench_runs`,
  `distinct`, `kept`, `reverted`, `failed`, `keep_rate_pct`), and each attempt
  row on a `framework_agent` event carries `benched`. A run is kept when it
  was adopted into the stack, reverted when it was measured and not adopted,
  and failed when it reached the bench but failed or left no measurement; an
  attempt that never reached the bench, such as a patch that did not apply,
  is not counted, and neither are kernel candidates. `hyperloom session
  report` shows the counts as a **Variants** line in its key facts and
  executive summary, with the final stack and validated gain when known, and
  derives the line from the timeline for a breakdown exported before this
  release.
- **Experience KB summaries carry each record's identity.** Every item in
  `POST /v1/read` `experiences` and `GET /v1/list` `items`, and so in
  `hyperloom-kb list` output and in the `experiences` of `state.json`
  `experience_kb_injections` rows, gains an `identity` object: the conditions
  the Experience was measured under. A reader can tell which records match its
  own conditions without paging through `GET /v1/export`.

### Removed

- **The accuracy eval can no longer be turned off.** `--no-eval`, the
  `RUN_EVAL` process-environment override and the `disable_run_eval` task
  param are removed, so the eval runs on every round that ran it by default.
  A kernel-stack validation, kernel-lane re-baseline or warm-recipe replay
  whose eval fails now fails that round instead of being retried with
  `RUN_EVAL=false`; a genuine baseline, which 1.1.4 already did not retry
  that way, is unchanged.
- **Resume of sessions started before 1.1.5, and several unused or deprecated
  options.** Hyperloom no longer carries readers for session formats older
  than this release and refuses those sessions up front instead of failing
  mid-run (see "Before upgrading from 1.1.4"). `hyperloom multi-node
  kernel-bench`, `--pr-feed-window-days`, `--local-kb-root`,
  `forge-loop --return-after-read-kb` and the `robustness_escalated` stop
  reason are gone, along with the always-empty `robustness` sections the
  retired robustness agent left in `session_breakdown.json`.
- **`kbmine`.** The standalone estimator under `tools/kbmine` that 1.1.4 added
  is removed with its tests, docs and CA provisioning script. It was never
  part of the `hyperloom` package, and nothing in the optimizer used it.

### Changed

- **AgentX optimization runs on Magpie's native AgentX contract.** Under
  `HYPERLOOM_AGENTX=1` Hyperloom turns `--model`, `--framework`,
  `--precision`, `--tp`, `--ep` and `--conc` into a Magpie AgentX config, or
  takes one from `--benchmark-config PATH` (start from
  `assets/configs/agentx_sglang.yaml` in the package), archives it as
  `benchmark.yaml` in the session and resumes from that copy;
  `--docker-image IMAGE` names the serving image to run inside. Magpie owns
  server start, trace replay, profiling and cleanup, and a KEEP needs a valid,
  unprofiled Magpie report for the session's own workload (otherwise
  `agentx_evidence_invalid` or `agentx_workload_mismatch`). Native AgentX is
  single-node SGLang or vLLM at one concurrency per session, takes request
  lengths and context from the trace corpus, and exits `2` at launch on a flag
  it cannot honour. Hyperloom's own AIPerf client, which drove 1.1.4's AgentX
  replay, is removed with its installer and repair path; the MLPerf agentic
  backend (`HYPERLOOM_AGENTIC_BACKEND=mlperf`) keeps its entry point.
- **New FRAMEWORK_AGENT and KERNEL_AGENT work must fit the phase's remaining
  budget, and the LLM can no longer lengthen a phase.** The
  `extend_explore_budget` and `extend_kernel_budget` hints of
  `escalate_strategy_change` are dropped like any unknown hint, and work whose
  estimate does not fit what is left of the phase is refused as a policy
  denial with `rule=phase_time_budget` (for example
  `action='explore' denied: needs ~30 min but only 12 min of the FRAMEWORK_AGENT phase budget is left`),
  or cancelled before dispatch with `reason: phase_time_budget` if it no
  longer fits when its turn comes. Running work stops at the phase end:
  explore starts no variant that cannot fit (a round that measured nothing
  reports `phase_time_exhausted`), a specialist's hard stop is the earliest of
  its shape budget, the session end and the phase end (`extend_lease` cannot
  move it past), the rewrite Controller keeps one measurement round in
  reserve and leaves later patches unapplied as `phase_time_exhausted`, and
  GEMM E2E validation, vLLM shape capture and the KERNEL-entry re-profile
  start nothing that cannot fit. In a bounded session, Forge GEMM tuning and
  fusion return `skipped` with `error_class: insufficient_budget` when their
  share of the KERNEL phase cannot fund one target, where 1.1.4 ran them on
  their 5 h / 2 h defaults. A specialist is admitted only when its shortest
  worthwhile run fits (10 min research, 60 min patch, a baseline round plus
  10 min for a bench specialist), where 1.1.4 used a 6-minute catalogue cost,
  and a kernel `integrate` / `apply_patch` request is refused when 25 min, or
  one measured baseline round if longer, does not fit.
- **A phase transition lets running work finish instead of cancelling it.**
  The transition waits, logging
  `phase_machine: holding phase_transition:<FROM>-><TO> (reason=<reason>) until <N> running task(s) finish and are booked`,
  and the wait is charged to the phases that follow, so a phase can run past
  its budget while its last running task finishes. A GEAK rebench still
  running when KERNEL ends on `kernel_phase_budget_exhausted` or
  `kernel_budget_cap` is kept or settled instead of ending as
  `rebench_cancelled`. Session exhaustion, shutdown and a terminal move to
  CLOSE (a recorded stop reason, `time_exhausted`, a skip-to-close escalation
  or a failed PRELUDE baseline) still cancel running work, and only a terminal
  move leaves `stopped_task_ids` non-empty in the
  `tasks_cancelled_on_phase_transition` observation.
- **The Coordinator runs the operator-defined specialist, and can run it
  alone.** With `--custom-specialist-prompt-file` and
  `--custom-specialist-description` set, the Coordinator queues
  `custom_specialist` on every FRAMEWORK_AGENT entry, once per macro cycle and
  ahead of other queued work, benches its proposals next in their own grid,
  and holds a plateau exit, never a budget exit, until both have run; 1.1.4
  left this to orchestration, so a session could end with the specialist never
  run or its proposals never benched. Orchestration is no longer offered the
  domain, a delegate naming it as domain or tag is refused with
  `specialist_custom_coordinator_owned`, and `--orch-prompt` no longer
  conflicts with the pair. `HYPERLOOM_CUSTOM_SPECIALIST_ONLY=1` (by `export`
  or `--extra-env`; it needs the pair, and a non-boolean value exits `2`)
  makes FRAMEWORK_AGENT run only that specialist and the bench and review of
  its output, with no orchestration turn and up to 4 h of wall budget, then
  hand over to KERNEL_AGENT when enabled, else SWEEP; it is persisted, and a
  resume can turn it on but not off.
- **Explore skips a variant whose answer the session already has.** Before
  each variant runs, explore skips one that only restates the launch it would
  run on (`reason=baseline_noop`) or that already has a KEEP or REVERT in
  `explore_search` on the same base launch, anchor throughput and workload
  (`reason=already_measured`, logged as
  `explore: skipping already-measured variant <name> (...)`), recording it as
  `SKIPPED_DEDUP` and in the result's `skipped_dup` instead of spending a
  benchmark round reproducing a known number. Restatements are judged against
  the stack the variant would launch on, so one that restates the opening
  launch's value of a setting an earlier KEEP in the same round changed now
  runs, where 1.1.4 dropped it as `baseline_noop`; a throughput REVERT
  (`gain_below_threshold`) runs again once the decaying KEEP bar no longer
  exceeds its recorded gain; and FAILED rows, kernel-overlay variants and
  stack revalidations always run. A delegated explore grid whose every variant
  is already measured, queued or running on the current stack is refused with
  a `policy_denied` observation, rule `explore_grid_already_covered`, and a
  round whose every runnable variant was skipped at its turn ends `succeeded`
  instead of `failed`. A PRELUDE warm replay without patches or kernel items
  that only restates the baseline launch is skipped (`warm_replay_outcome`
  reads `status=skipped`, `reason=restates_baseline`) unless the session has
  reference launch controls.
- **OPTIMIZE turns to source changes when configuration stops paying.** When
  the configuration arm plateaus in FRAMEWORK_AGENT, a second read-only
  `static_recon_specialist` pass runs once per macro cycle
  (`reason=config_arm_plateau`, `gap_canonical_id=gap.static_recon.cycle<N>`)
  and refreshes the `gap.static_recon.*` rows in `=== Current gaps ===`, where
  1.1.4 ran recon only at PRELUDE; `--no-static-recon` turns off both passes.
  On consecutive config-only rounds the orchestration prompt now points at a
  `mode='patch'` specialist for a source change rather than another grid, and
  patch-mode dispatches into the kernel-switch, comm, compiler and system
  domains get the source-patch guidance 1.1.4 gave only the serving
  specialist.
- **Enablement Rung 3 provisions the runtime the specialist requests.** An
  enablement specialist names it with `needs_runtime` in its done payload (a
  git ref given as PR URL, `PR:N`, tag, branch or SHA, with optional packages
  and index URL; packages only on vLLM), and Rung 3 builds it with no operator
  setting; once a round keeps a runtime, later requests are logged as
  `needs_runtime ignored` and the kept one stays. Rung 3, Rung 5 and selfcheck
  installs pin torch, torchvision, torchaudio, triton and every ROCm-labelled
  distribution through `rocm_constraints.txt`, so a request that needs a
  different torch or triton ends the round `reverted` with `provision_failed`
  and pip's conflict.
- **Every framework's trace goes through the TraceLens splitter.** On the
  TraceLens route, scriptable frameworks (`xdit`, `custom`) no longer skip
  splitting: their traces are cut into one `steady_state_*` chunk with
  `--steady-state-mode generic`, so their kernel breakdowns rest on a
  steady-state chunk instead of the whole trace, and a trace that cannot be
  split fails with `trace_split_no_steady_state` where 1.1.4 analysed it
  unsplit. Serving frameworks run the splitter in its LLM-inference mode, so a
  trace without serving annotations is still split into prefill and decode
  phases by shape. The capture-time self-certification no longer lists
  `too few step annotations for the splitter to cut a steady window` as a
  blocking reason, and start-up warm-up is trimmed by the TraceLens splitter
  itself.
- **The TraceLens Agent runs from a checked YAML run config.** Hyperloom
  writes the agent's inputs to `run.yaml` in the analysis output directory and
  checks it with TraceLens's own validator before the agent starts, so a
  config TraceLens rejects (an unknown platform with the internal extension
  on, or a scriptable workload with a capture folder, which 1.1.4 ran with the
  folder silently ignored) fails in seconds with the `ConfigError` text in the
  failure JSON, instead of inside a paid agent session as a missing
  `analysis.md`. The recorded `analysis_mode` now matches what TraceLens ran:
  `inference` for serving and unknown frameworks, `default` for scriptable
  ones, where 1.1.4 recorded `default` for ATOM runs the agent analysed as
  `inference` and gave unknown frameworks `default`.
- **KernelForge's evaluation tooling and MCP servers move.** The exploratory
  sweep command is now `python -m kernelforge.evaluation.benchmark.runner`,
  which the backend prompts and the shipped wiki card give. The specialist
  probe and upstream-PR MCP servers start as
  `python -m kernelforge.orchestrator.specialist_probe_server` and
  `python -m kernelforge.knowledge.pr_knowledge.mcp_server`, keeping their
  server names, tool names, schemas and replies, and the
  `kernelforge.mcp_server` package is gone. KernelForge Click commands,
  KEEP/REVERT scoring, persisted data and benchmark output are unchanged.
- **Critic reviews over the Anthropic protocol reuse a cached prompt prefix.**
  The Critic sends its fixed instructions as a cacheable block ahead of each
  review's changing bundle, so reviews that follow each other within the
  provider's cache window bill most of their input at the cached-read rate.
  Review text, model choice, output limits and retries are unchanged, and the
  OpenAI-protocol and CLI paths send the same prompt as before.
- **Errors that 1.1.4 caught and stepped past now surface.** Handlers across
  the optimizer and KernelForge that caught every exception now catch only
  the errors their code can raise. Expected I/O, network, subprocess and
  import failures behave as before; anything else reaches the task, tick or
  CLOSE step above it and is recorded there instead of turning into a quiet
  fallback, and outermost best-effort handlers (per-patch E2E validation, a
  concurrency-sweep arm, task preflight stages, critic dead-letter replay, the
  multi-node pod scripts) now log what they used to drop. In particular:
  preflight stops on an exception in its eval-concurrency or multi-node
  client-trust compat patch, where 1.1.4 printed `compat patch failed to run`
  and went on; an `install-geak` that raises on the Infera multi-node backend
  stops the launch instead of logging `install-geak skipped`; a target-analysis
  crash fails its task instead of returning `status=succeeded` with
  `reason=analyzer_crash`; `integrate_patch` provision, bench and parity still
  turn environment failures into a reverted attempt, but any other error now
  surfaces instead of counting as a revert; and a `state.json` save still logs
  and continues on `OSError`, but a serialization error raises instead of
  leaving `state.json` stale. At CLOSE, a recipe KB read or write that fails
  leaves `recipe_finalize_status=failed` instead of overwriting the live
  recipe from an empty read or reporting it `written`, a failed per-variant
  write is logged as `recipe KB write for task <id> failed: <type>: <message>`
  and later variants are still recorded, and a defect in the roofline-progress,
  baseline-progress, stack-ledger or final-recipe step is recorded as a failed
  close step, so the close reads `degraded`. In KernelForge, a failing aiter
  preflight fails `gemm-tune` instead of reporting no blocking problems,
  `forge-loop --resume` stops when it cannot restore its scoring state, a
  kernel-backend prompt that cannot be built fails the forge agent's start-up
  instead of falling back to the generic prompt, and a failed experience-KB
  write is logged as `experience KB publication failed for <commit>: ...`.

### Fixed

Resume and recovery:

- **A resumed session runs its rounds from the InferenceX checkout again.**
  Only a tree that carries `benchmarks/benchmark_lib.sh` is recorded as the
  session's InferenceX checkout or re-exported as `INFERENCEX_PATH` on
  `--resume-from`, where 1.1.4 could record the patched framework tree that
  PRELUDE's warm replay adopted, so every later sweep, explore and integrate
  round of the resumed session failed as `magpie_nonzero_invalid_measurement`,
  `detokenizer_stall` or `magpie_timeout`. A baseline or grid round whose
  InferenceX root lacks that file now fails before launch with
  `<path> is not an InferenceX checkout`, as `inferencex_checkout_invalid` on a
  baseline and return code `-918` on a grid variant.
- **A refused `--resume-from` names the confirmation that unblocks it.** The
  `resume blocked` message now asks first for
  `hyperloom recover --session-dir S --confirm-scope-gone OWNER_SCOPE --confirmation-reason TEXT`
  when another owner scope's environment is gone, then for
  `--confirm-stopped TASK_ID` for a listed task that names no owner scope
  (such as a lane-less CLOSE-phase step), where 1.1.4 offered
  `--confirm-stopped` for legacy empty-scope ownership only.
  `hyperloom recover --help` now describes both confirmation modes and the
  exit codes (`0` success, `1` breakdown rebuild failed, `2` invalid arguments
  or a refused confirmation), and the crash-recovery runbook in
  [Operations](reference/operations.md) follows the same order.

Scheduling, phases and the LLM backend:

- **An orchestrator `skip_to_close` that the session's own facts contradict no
  longer ends the run.** While a baseline is measured and the remaining budget
  still funds another macro cycle, the hint is refused with `policy_denied`
  rule `skip_to_close_contradicted` (the reason names the measured baseline
  and the minutes left against the minutes a cycle needs), recorded in
  `state.json` `policy_denial_history`, and the run continues in its phase,
  where 1.1.4 closed on the spot and booked it `global_converged`. A
  `skip_to_close` with no measured baseline (`baseline_failed`) or below the
  cycle floor (`time_exhausted`) closes as before. While PRELUDE or
  ENABLEMENT waits for its initial roofline, the orchestration `=== Phase ===`
  block carries a `hold` line naming the roofline and how long the hold can
  last, so the agent no longer reads the phase's spent share
  (`remaining_sec=0`) as a spent session. **Upgrade note:** a run whose
  orchestrator asks to close early with a baseline measured and a cycle's
  budget left now keeps optimizing, so its length, `stop_reason` and result
  differ from a 1.1.4 run that closed at that point.
- **Orchestration can no longer run an action its phase does not offer, and a
  validated gain keeps its baseline.** A `propose_action` or `delegate`
  outside the phase's proposable set (for example `baseline` in
  FRAMEWORK_AGENT, `explore` in ENABLEMENT, KERNEL_AGENT or SWEEP, `report`
  outside CLOSE, or an action name no phase offers) is refused before the
  Critic sees it, with a `policy_denied` observation `rule=phase_incompatible`
  and the hint `<PHASE> proposes: ...`, where 1.1.4 passed it to the Critic
  and ran it once approved. Once a cumulative gain has been validated, a
  higher re-baseline no longer replaces `baseline_tput`: the log reads
  `baseline anchor: keeping <prior>; re-baseline measured <new> (task=<id> reason=anchor_load_bearing)`
  and the baseline attempt record carries
  `anchor_held: {reason, prior, measured}`, so the report no longer flags the
  validated gain `inconsistent`. An ENABLEMENT revalidation still moves the
  anchor.
- **A `null` field in an agent's intent means "not supplied".** Hyperloom
  drops JSON `null` values from an intent's payload and its `params` before
  validating it, so a proposed action whose optional parameters the agent left
  `null` runs on the session's defaults instead of failing the round with a
  `TypeError`. A `delegate` or `propose_action` whose required field is `null`
  is refused (`missing required payload field`), and on the Claude backend
  `emit_intent` returns that refusal, and a bad `alert` severity,
  `extend_lease` or `review_verdict`, as a tool error naming the field, where
  1.1.4 answered `ok`. **Upgrade note:** an optional param sent as `null` now
  inherits the stack's value instead of clearing it (`extra_server_args`,
  `remove_args`, `unset_envs` and `args_mode` take `current_best`'s, and
  `baseline_double_run` falls back to on), so the same proposal can launch a
  different server command than under 1.1.4.
- **Reviewed patches no longer starve behind explore grids in
  FRAMEWORK_AGENT.** A queued `integrate_patch` dispatches ahead of tasks
  queued earlier on the lanes it needs (explore grids and, through lane
  expansion, queued `profile`, `roofline` and GPU specialists), oldest first
  among patches; where 1.1.4 dispatched oldest-first, a Critic-approved patch
  waited behind every earlier grid, and on AgentX, where a serving round takes
  about an hour, the phase could end and cancel every queued patch unmeasured.
  A running task is not preempted, and in ENABLEMENT a queued
  `integrate_patch` still waits behind the revalidation `baseline` queued
  before it.
- **A specialist domain holds at most one queued or running single-domain
  specialist.** A second `scope=domain` delegate for that domain is refused
  with `policy_denied` rule `specialist_domain_in_flight`, whose hint names
  the holding task, and an approved specialist proposal for it is dropped as
  `proposal_materialize_skipped` with that reason, where 1.1.4 queued a fresh
  copy whenever a later tick delegated the domain again and, on a one-GPU
  host, the duplicates pushed other domains back by hours. Orchestration's
  in-flight block gains a
  `queued specialists (each holds its domain): <domain> task_id=<id>, ...`
  line; `domains` and `freeform` dispatches hold no domain. A
  `scope=domains` dispatch naming fewer than two tags is re-inferred from its
  tags, so a one-tag one runs as a single-domain specialist.
- **A specialist proposal whose `AITER_CONFIG_*` file is missing is refused
  at admission.** A proposal whose `extra_envs` point an `AITER_CONFIG_*`
  variable, or any element of its `:`-joined list, at a file that is not on
  disk is dropped before it is benched, listed under
  `proposals_env_ungrounded`, and recorded as a `proposal_env_file_missing`
  specialist failure that the next SEED turn sees, with a pointer to the
  `gemm_tuning` action. Where 1.1.4 booted the server, which died in aiter's
  config loader and then timed out on `/health`, a grid variant that still
  carries such a path now fails before boot with `capability_unsupported` on
  every framework.
- **Orchestration's `send_message` reaches every recipient in a list.** A
  `to` holding several names writes one bus message per distinct name, in
  order, and puts a copy in each `specialist:<task_id>` recipient's inbox. A
  `to` that is an empty list, holds an empty or non-string name, or is any
  other non-string value is refused with `policy_denied` rule `payload`; 1.1.4
  passed it straight to the events insert, which raised
  `sqlite3.ProgrammingError` and failed the running phase.
- **KERNEL no longer keeps waiting on kernel work that a fusion KEEP already
  landed.** A `fusion` row on `optimization_stack` counts as integrated work,
  so KERNEL exits to SWEEP sooner and admits less work, and its exit evidence
  reads `pending_work.kernel_work_pending: false` in those rounds. A
  benchmark-round recorder that raises is logged as
  `round recorder <name>.<method> failed` and the round still reports its
  measurement, where 1.1.4 failed a single-recorder round over a good number,
  and a multi-node restart that fails before the relaunch restores the
  profiler trace path and forwarded-environment variables it set instead of
  leaving them for the next round's launch.
- **A specialist patch whose first proposal is malformed is still reviewed.**
  When the first `proposal_set` entry of a specialist's patch deliverable is
  not an object, its `integrate_patch` review is submitted with an empty patch
  name, where 1.1.4 raised `AttributeError` before submitting it. The
  `framework_config_autosubmitted_for_review` observation now carries
  `patch_name`, as `specialist_patch_autosubmitted_for_review` does, and trace
  analysis omits a boolean or non-finite `num_denoise_steps`,
  `num_inference_steps` or tensor-parallel size instead of passing
  `--num-denoise-steps 1` / `--tensor-parallel-size 1` or failing with
  `OverflowError`.
- **A gap row counts each failure once.** The gap ledger that orchestration
  reads shows the real number of failures behind a gap (`attempts=1` for one
  failure), where 1.1.4 counted the same failure again on every gap refresh
  until the row reached its 20-attempt cap, so gap ranking by attempt count is
  no longer skewed.
- **Orchestration is offered only the built-in tools it may use.** The Claude
  backend passes the declared built-ins (`Read`, `WebFetch`, `WebSearch`) as
  the SDK tool set, so orchestration no longer sees Bash, Write or Edit and
  stops spending turns on Bash calls that were denied while it inspected logs.
  MCP tools (`emit_intent` among them), approval settings, filesystem
  permissions and specialist permissions are unchanged.

Measurement, profiling and AgentX:

- **An AgentX baseline measured with the prefix cache off is refused.** When
  the baseline server's own launch account shows the cache off (SGLang
  `disable_radix_cache=True`, vLLM `enable_prefix_caching=False`, or ATOM
  server args without `--enable_prefix_caching`), the baseline fails with
  `error_class=agentx_baseline_prefix_cache_off` naming that evidence, and the
  run stops with `stop_reason=baseline_failed` on the first occurrence,
  re-baselines included, without opening enablement. In 1.1.4 such a
  cold-cache baseline became the anchor, so any candidate that turned the
  cache back on was credited with the cache itself as a gain. A launch whose
  account cannot be read, synthetic baselines and candidates are not judged.
- **A candidate can no longer change the draft under a simulated acceptance.**
  When the base launch simulates speculative-decoding acceptance
  (`SGLANG_SIMULATE_ACC_LEN`, a vLLM `--speculative-config` with
  `rejection_sample_method: "synthetic"` or a `synthetic_acceptance_length`,
  or an acceptance-length flag such as ATOM's
  `--spec-decode-acceptance-length`), an explore variant, `integrate_patch`
  bench or concurrency-sweep arm that changes the draft, its method or the
  simulated acceptance fails before boot with
  `error_class: speculative_decoding_pinned`, naming each changed setting,
  where 1.1.4 benchmarked it against the stale acceptance and could KEEP a
  gain that does not exist. A base that verifies a real draft may still tune
  it but may not switch to simulated acceptance, and a base with no
  speculative decoding may enable it either way. The check covers synthetic
  and MLPerf-backend sessions; native Magpie AgentX candidates go through
  Magpie's launch overrides and skip it.
- **A context-copying speculative draft is no longer kept on a synthetic
  benchmark.** In a synthetic session, explore refuses before launch any
  variant that introduces, switches to or tunes a vLLM `ngram`, `ngram_gpu` or
  `suffix` draft or an SGLang `--speculative-algorithm NGRAM` draft, recorded
  as `SKIPPED_DEDUP` with `reason=speculative_context_copy_on_synthetic` and a
  message naming the method: random prompts held to full length drive the
  model into repetition loops such a draft accepts almost entirely, so its
  gain does not carry over to real text. A round whose eval-start marker lands
  in Magpie's nested `benchmark_*/benchmark_stderr.log` now gets its `eval`
  phase in `kv_metrics.json`, so the measured phase's KV and GPU-power figures
  no longer include eval traffic.
- **A roofline or trace analysis that failed the same way on the same input
  is not re-run.** After a roofline fails with `trace_split_no_steady_state`
  or a TraceLens analysis timeout (`analysis_timeout`), the cycle-start,
  gain-watermark, kernel-entry and CLOSE re-arms skip it until a KEEP changes
  the recipe, logging
  `internal-analysis (<reason>): not run -- the roofline on recipe-<N> already failed with <class> ...`
  and leaving a `prelude_arm_dropped` marker, and a `trace_analyze` request
  identical to one that already failed that way on the same trace returns at
  once with `status=failed` and
  `error_class=trace_analysis_failure_remembered`. 1.1.4 ran them again,
  holding the benchmark, profile and server lanes for up to an hour each time
  to fail the same way. Other failures, another trace or another
  `budget_minutes` still run, and the kernel-entry `forge_reprofile` record
  says `skipped_reason=repeated_failure` when a remembered failure declined
  the analysis, instead of always `gpu_trace_unsupported`.
- **Explore and sweep variants launch with the baseline's harness flags.** A
  variant keeps `--no-enable-prefix-caching` when the operator's
  `--server-args` carry it, where 1.1.4 dropped it from every composed launch
  and compared prefix caching off in the baseline with it on in every variant;
  persisted recipes still drop the flag. A specialist patch that matches
  several identical copies of the source tree goes on to review and
  integration when it applies cleanly to the session's own copy, instead of
  being refused as ambiguous.
- **vLLM runs with CUDA graphs off keep torch.compile.** When a baseline
  retries after a CUDA-graph capture failure, or a roofline profile runs with
  CUDA graphs disabled, vLLM gets `--compilation-config.cudagraph_mode NONE`
  where 1.1.4 passed `--enforce-eager`, so inductor compilation stays on and a
  profile shows the kernels the measured runs use. The mode is merged into a
  JSON `--compilation-config` the server arguments already carry, and nothing
  is added when they already pass `--enforce-eager`; SGLang keeps
  `--disable-cuda-graph` and ATOM `--enforce-eager`. **Upgrade note:** such
  vLLM baselines and profiles now run with inductor on, so their throughput
  and kernel breakdowns differ from 1.1.4's.
- **A quote-stripped spec-decode config whose draft is an absolute path
  boots.** A recorded `--speculative-config` that lost its JSON quotes and
  names its draft model by path (`{model:/models/X,...}`) is repaired into
  valid JSON on every launch path, warm replay included. 1.1.4 left it
  unparsable, so the server failed at boot with `server_argv_invalid`, the
  warm replay was lost and the session fell back to cold exploration.
- **A bypass boot-only round boots the server and stops.** Under
  `HYPERLOOM_BENCHMARK_BACKEND=bypass` the round that boots a concurrency
  sweep's server now starts it and runs no benchmark client or GSM8K eval,
  where 1.1.4 ran a full round first (about 7 minutes per boot on GPU). The
  round writes no `benchmark_report.json` and is filed as
  `server_lifecycle_boot_only`; the boot concurrency is still measured by the
  first reuse round.
- **Rounds stop warning that the tokenizer-mode patch could not be applied.**
  Hyperloom no longer rewrites Magpie's vLLM client scripts to add a
  `--tokenizer-mode` hook, which the pinned Magpie ships, so the
  `generic vLLM client block not found ... tokenizer-mode patch could not be applied`
  warning logged for the multimodal `*_mm.sh` scripts is gone. A round for a
  `deepseek_v4` checkpoint the installed transformers cannot resolve is still
  refused with `client_tokenizer_unpatchable` when its client script lacks the
  hook, and the message now names the Magpie revision to use.

Enablement:

- **Enablement no longer loses rounds, its kept stack or the session to its
  own apply and revert.** Whether an integrate is an enablement round is
  decided when it runs, so a stale one is graded as an ordinary integrate
  instead of growing the enablement stack; a kept patch is no longer
  duplicated by its harvested diff or removed by the low-disk `runs/` trim,
  and a `kept_inert` result is committed so the next attempt keeps it. A
  failed apply leaves the framework tree untouched (no `git apply -3` retry),
  and the revert resets to the recorded HEAD and cleans the tree, so it no
  longer ends the session with `patch_recovery_incomplete`. Rung 4 backports
  only files under the framework package, with no file-count cap; a backport
  that does not apply is dropped and logged as
  `integrate_patch: localization ... skipped`, and the round runs on the
  specialist's own patch, where 1.1.4 reverted the whole round. A switched
  patch whose switch-off parity leg has no base throughput to compare against
  reverts as `switch_off_parity_inconclusive`.
- **After enablement promotes a runtime, later phases patch the tree the
  bench imports.** The promoted checkout,
  `enablement/stacks/<framework>/<task>/src`, becomes the session's framework
  tree for specialists and `integrate_patch` (`state.json`
  `framework_repo_path`, restored on `--resume-from` unless `--framework-path`
  is given), where 1.1.4 kept patching the image's tree (for example
  `/sgl-workspace/sglang`) to no effect; a runtime accepted by an advanced
  round is re-activated like a kept one. Each attempt logs its git and pip
  steps with exit codes to
  `enablement/stacks/<framework>/<task>/provision.log`, kept after a failed
  provision, and the round error carries pip's own error. Attempt venvs see
  the image's eggs and editable installs (`sgl_kernel`, editable aiter), and
  an SGLang checkout installs with the `srt_hip*` extra the image's ROCm torch
  satisfies, keeping that torch. Orchestration naming `enablement_specialist`
  is refused with `specialist_domain_coordinator_dispatched` instead of
  crashing the dispatch.
- **Enablement rounds finish what they start.** Rung 5 no longer fails with a
  spurious `abi_mismatch` when the coordinator runs from a virtualenv, a
  failed targeted build no longer ends the round that requested it,
  ENABLEMENT exits once its initial roofline, run under the accepted launch
  config, is booked, and Rung 4 localization covers every code gap on sglang
  and vLLM.

Kernel:

- **A GEAK hand-off ends in a result the session can keep.** On an
  interactivity-graded AgentX session the hand-off's `e2e_metric`, GEAK's
  runner `E2E_METRIC` and `workload_spec.metric_basis` name
  `e2e_norm_intvty_p50` where 1.1.4 handed total token throughput,
  `workload_spec.acceptance` carries the session's KEEP rule (+3% on p50, with
  the p90 tail and output throughput inside the noise band), and sweep rows
  gain `measured_value` and `metric_basis`. Before delegating, KERNEL stops the
  session's leftover servers, waits up to 120 s for the hand-off's cards to
  fall below 1% VRAM and records `gpus_busy_at_handoff`, naming the busy cards
  and up to eight serving process groups, instead of starting GEAK on cards
  that stay busy. Each hand-off holds back the time its rebench needs (one
  baseline round, its warm pass when the baseline ran one, and a quarter
  pass), logged as
  `GEAK: holding back <N>s of the KERNEL budget to re-measure GEAK's result`,
  where 1.1.4 gave GEAK the rest of the KERNEL budget, and GEAK is skipped as
  `insufficient_budget` when less than `GEAK_MIN_RUN_S` (default 600 s) would
  remain. Before re-measuring a result GEAK has just returned, KERNEL again
  stops any server the session left running and waits up to 2 minutes for the
  handed-off GPUs to free their memory, then measures either way, logging
  `geak rebench: GPUs still hold VRAM after GEAK returned: ...` if they stay
  busy. A GEAK result is no longer refused its rebench for time or lost to a
  server boot on occupied GPUs, which left the session at 0% validated gain.
  **Upgrade note:** GEAK's deadline on every hand-off is shorter by the
  rebench reserve, so GEAK outcomes are not comparable with 1.1.4 runs; see
  "Installation prerequisites move" for the GEAK revision this needs.
- **GEAK serves the engine an AgentX baseline ran.** Under AgentX with the
  GEAK backend, `geak/handoff.json` names the session's server builtin as
  `launch_server_script` and leaves `bench_launcher` unset, so GEAK boots its
  benchmark server through Magpie with the baseline's engine flags and
  environment, and the hand-off's `baseline_env_spec.config.server_launch_flags`
  and `extra_envs` carry the launch the baseline recorded in
  `server_command.txt` and `benchmark_stderr.log`. 1.1.4 forced
  `bench_launcher: native`, so GEAK served its adapter's defaults instead of
  the recipe's flags and exports; `native` remains only when the builtin
  cannot be found on disk.
- **GEMM tuning tunes the experts of Quark and compressed-tensors checkpoints
  and reads aiter's fused-MoE dispatch lines again.** When the precision
  handed to forge GEMM tuning is empty, `auto` or only the checkpoint's
  toolkit name, the router reads the formats nested in `quantization_config`:
  the routed experts' format picks the MoE tuner and the other linear layers'
  format the dense one. `fmoe_ck` therefore runs on Quark MoE checkpoints,
  which 1.1.4 skipped as `Unsupported MoE precision/quant combo: quark/none`,
  and when no serving log names the dense kernel a GLM-5.3 MXFP4 checkpoint's
  fp8 attention gets `a8w8_blockscale`. With `--quant-type auto` the MoE
  tuners route first on the weight dtype and `QuantType` of the most-missed
  fused-MoE dispatch in `--kernel-signature-log`, and fall back to the nested
  format when the log shows none or one no tuner handles. From aiter commit
  `e7d2453` on, which prints the activation unquoted, Hyperloom matched none of
  aiter's fused-MoE dispatch log lines, so no untuned fused-MoE CSV came from
  the serving log and gemm-shape coverage reported tuned MoE shapes as
  untuned; Hyperloom now reads those lines with Forge's own parser. An
  explicit `fp8`, `mxfp4` or `bf16` precision routes as before.
- **Forge's upstream PR references report a PR closed without merging as
  `closed`.** The "Upstream PR references" block Forge adds to kernel prompts
  under `--pr-kb`, `forge_experiments/pr_refs/index.md` and the
  `pr_find_references` / `pr_get_reference` tools label each PR `open`,
  `closed`, `merged` or `unknown`, where 1.1.4 called every unmerged PR
  `open`. Drafts read `open, draft`, and the tool rows also carry `draft`,
  `merged_at` and `base_ref`, with `null` for anything the PR Monitor service
  did not report.
- **A multi-node kernel apply names the pods that did not invalidate the
  AITER JIT cache.** The failure (error class `aiter_jit_invalidation_failed`,
  unchanged) reads
  `AITER JIT cache invalidation failed: pods ['<host>'] did not invalidate it exactly once`,
  where 1.1.4 printed `one or more pods did not invalidate AITER JIT` or
  `each pod must invalidate AITER JIT exactly once`, and it carries
  `jit_build_backup` as a single-node failure does. The single-file and
  snapshot apply paths hold every pod to the same rule: exactly one JIT backup
  per pod.
- **A kernel request whose `target_agent` carries surrounding whitespace
  runs.** The router reads the name stripped, as the policy gate that admitted
  the request does, so a request for `" kernel_agent "` reaches the kernel
  agent's handler instead of failing with `unknown_target_agent` and being
  recorded as a failed action.
- **The workspace guard leaves forge-loop's own output tree alone.** The
  protected-file inventory skips every `forge_experiments/` directory, as it
  skips `.git`, so a profiling venv staged there no longer fails the analysis
  phase with `protected ignored files changed`, and a plan revision or lesson
  summary resumed read-only no longer fails with "the read-only session
  changed the workspace" or rolls back the `llm_usage.json` spend ledger and
  `events.jsonl` the loop wrote meanwhile. A read-only session that edits a
  workspace file is still rejected and restored.

Install, preflight and environment:

- **vLLM source-overlay runs install, keep their framework version and start
  under Ray.** `hyperloom setup` verifies an overlay by importing
  `vllm.entrypoints.openai.api_server` where 1.1.4 ran `pip check`, which
  reported the host image's own pin conflicts, so it never passed on SGLang
  ROCm images, and missed a host `openai` the server could not import. The
  framework version of a `setup.py develop` checkout or an isolated vLLM venv
  is read from the serving interpreter, so the stack fingerprint and KB key
  carry it instead of `unknown_version`. Ray worker and serving-actor
  environments re-index a node-wide `HIP_VISIBLE_DEVICES` into the devices
  Ray's `ROCR_VISIBLE_DEVICES` exposes, so Ray-launched servers start on such
  hosts. A failed backend turn is logged at WARNING as
  `Coordinator: <agent> backend turn failed (<n> in a row): <error>`.
- **`--reference-script` reads an `sglang serve` command.** An SGLang
  cookbook launch written as `sglang serve ...` now yields its server flags as
  written. 1.1.4 found no flags in it, keeping only the script's exports or
  exiting `2` with "yields neither a flag nor an export", unless it was
  rewritten to `python3 -m sglang.launch_server`.
- **A baseline that fails `CacheConfig` validation stops as an argument
  error.** Server output carrying `validation error for CacheConfig` (or
  `errors`) is classified as an argument failure however long the server ran
  first, so the second such baseline stops the run with
  `stop_reason=baseline_arg_error` and the failure is not used as enablement
  evidence; 1.1.4 recognised argument errors only within 30 s of start and
  treated a slower one as `subprocess_nonzero`. A KV-cache OOM is still
  classified first.

Reports and exports:

- **Serving throughput is labelled `tok/s`, not `tok/s/GPU`.** Reports,
  summaries and the roofline snapshot print vLLM, SGLang and ATOM throughput
  as `tok/s`: the number is the server's aggregate over all its TP GPUs, so
  1.1.4's label overstated per-GPU throughput by the TP degree. The values are
  unchanged, and genuine per-GPU figures (InferenceX targets, the
  concurrency-sweep efficiency plot) keep `/GPU`.
- **AgentX stack-ledger adoptions are measured on the graded axis.** On an
  AgentX session graded on interactivity (`e2e_norm_intvty_p50`), each
  adoption row of the stack ledger in `session_breakdown.json` carries the
  interactivity baseline as `baseline_tput` and computes `contribution_pct`
  and `cumulative_gain_pct` against it, so the ledger reconciles with the
  stack validation; 1.1.4 divided an interactivity gain by the
  output-throughput baseline, which showed as a large
  `reconciliation_gap_pct`. When that baseline was not measured the row
  carries `null` and the ledger is `degraded`. Sessions graded on output
  throughput are unchanged.
- **The concurrency-sweep chart follows the concurrency ladder.**
  `conc_sweep_curve.png` joins each arm's points in concurrency order, labels
  each point `c=<conc> <value>`, and draws a concurrency only one arm measured
  as a hollow marker with a "hollow: concurrency measured by one arm only"
  legend entry. 1.1.4 joined the points in interactivity order, so a line
  could run 8, 4, 14 and a rung the baseline never measured read as the
  optimized arm doing worse.
- **A failed remote Recipe KB publication keeps its cause.** The publish row
  CLOSE appends to `runtime/recipe_snapshot/.audit.jsonl` now records, beside
  `error.type`, the wrapped cause's type (`error.cause_type`) and, when known,
  the HTTP status (`error.http_status`) or OS errno (`error.errno`, with
  `error.reason_type` for a connection failure), where 1.1.4 recorded only the
  exception class, such as `KBStoreError`. Exception messages, URLs and
  response bodies stay out of the row.

Other:

- **The non-git patch channel reports and records what it actually did.**
  When a patch is applied without git, the *Rejected hunks* section of the
  apply feedback shows the hunks `patch` actually rejected, where 1.1.4 always
  left it empty, and the apply leaves no `.orig` files behind and no longer
  deletes the `.rej` files of a concurrent git-channel apply. A rename with no
  content change is recorded and reverted whichever diff prefix style
  produced it (`diff.noprefix`, `diff.mnemonicPrefix`), and a section with
  `/dev/null` on both sides is refused.
- **The low-disk trim of `runs/` no longer deletes live work.** It fires only
  below 20 GB free and skips workspaces a live task still reads, where 1.1.4
  also trimmed above 85% used and could delete a running specialist's
  workspace. TraceLens falls back to `CLAUDE_MODEL` when `ANTHROPIC_MODEL` is
  unset, and specialists receive `INFERENCEX_PATH`.

### Known issues

- **On a Magpie AgentX session, a KEEP that removes recipe options can make
  every later bench fail.** Such a KEEP (one carrying `remove_args` or
  `unset_envs`, for example an explore winner that drops the
  hierarchical-cache flags) switches `current_best.args_mode` to `replace`, so
  later candidates reach Magpie with `launch_overrides.replace_args: true` and
  lose every recipe option Magpie does not protect. On a recipe that pins
  golden acceptance, such as DeepSeek-V4-Pro with speculative decoding, every
  later explore, roofline and `integrate_patch` bench then fails within
  seconds as `magpie_nonzero_invalid_measurement`
  (`Changing AgentX golden acceptance inputs requires a new resolved recipe`).
  The fix is on `main` and ships in the next release.

## Hyperloom 1.1.4 release

The [1.1.4 release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.1.4)
follows 1.1.3. Every command now runs as `hyperloom <command>`, every
specialist runs as a Claude CLI subprocess in its own worktree, and the kernel
tools move into the packages that own them. The release adds the Experience
KB, per-GPU power measurement, operator-defined specialists, runtime findings
from server logs and an opt-in MLPerf agentic backend, and a session whose
container is gone can now be released and resumed. A 1.1.3 `state.json` still
loads: keys this release no longer carries are dropped. Several options,
environment variables and launch paths 1.1.3 accepted are removed, and some
defaults move; see "Before upgrading from 1.1.3" below. The per-change history
is on the [Releases](https://github.com/AMD-AGI/Hyperloom/releases) page.

### Before upgrading from 1.1.3

Console scripts, options and module paths 1.1.3 accepted are removed. The
parser is strict, so a launch, resume or install command that still carries
one exits with `unrecognized arguments` (or exit `2`) before the session
starts.

| Removed | What to do |
|---|---|
| `inference_optimizer` and `quantization-agent` console scripts; `python -m hyperloom.inference_optimizer.{cli,multi_node,setup}` and `python -m hyperloom.inference_optimizer.tools.*` | Run `hyperloom <command>`. `recover-session` becomes `hyperloom recover`, and `--verbose` goes after the command. The full old-to-new table is in [Upgrade Hyperloom version](reference/upgrade.md). |
| `--kernel-codex`, `--kernel-claude`, `--kernel-prompt` | Delete them; they were ignored. |
| `--max-minutes-{prelude,framework,explore,kernel,sweep,close}-pct`, `--phase-budget-explore-pct` | Use `--phase-budget-<phase>-pct` (`--phase-budget-framework-pct` covers what the explore spelling set). |
| `recover --backfill-trace` | Run `hyperloom session backfill --session-dir S`. |
| `--specialist-dispatch-mode`, `--specialist-per-turn-max-seconds` | Delete them. Specialists always run as a Claude CLI subprocess; a session that finds no claude CLI stops at startup. |
| `--target-summary`, `--warm-replay-min-reproduce-pct`, `--plateau-kernel-revert-streak`, `--plateau-kernel-keep-gain`, `--plateau-kernel-lookback` | Delete them; none changed a decision (see *Removed*). |
| `install.sh --skip-kernel-agent`; running `install_kernel_tools.sh` on its own | Delete the option. The kernel tools install as part of `install.sh`; the standalone script exits `2` and points at it. |
| `forge-rewrite-by-flydsl --op-name` | Use `--logical-op-name`. |
| `hyperloom.agents.kernel.tools.*` modules, including the `forge_gemm_tuning` and `forge_fusion` wrappers; the TraceLens and bypass trace-analysis options `--runtime-env`, `--roofline-json`, `--height`, `--width`, `--cfg-batch`, `--top-k` | Run the tools as `python -m hyperloom.orchestrator.trace_analysis.<tool>` or `python -m hyperloom.orchestrator.kernel.<tool>`. The kernel request handlers now build the forge GEMM tuning and fusion commands themselves, and nothing passed the removed analysis options. |
| `src/kernelforge/data/examples/...` | Example tasks live under `src/kernelforge/examples/` (`resource_path('examples')`). |

Environment variables fail differently: nothing refuses most of them, so a box
that still exports one starts normally and behaves as though it were unset.
Find them by reading launch scripts, `.env` and the kernel-agent env file. Two
settings do stop the run, and are marked below.

| Removed or changed variable | What to do |
|---|---|
| `HYPERLOOM_KERNEL_AGENT_ROOT`, `KERNEL_AGENT_ROOT` | Delete them. Preflight reads the kernel-agent env file only from `$KERNEL_AGENT_ENV` or `$USER_DATA_PATH/runtime/kernel-agent.env.sh`, and **exits 2** when that file yields no supported variable; rerun `install.sh` to regenerate it. |
| `AGENTX_SERVER_SCRIPT` naming an InferenceX `agentic/` recipe | **Exits 2** at server boot. Move the recipe's server arguments into `--server-args` and its environment into `--extra-env`, and drop `INFERENCEX_PATH` so Hyperloom clones its own InferenceX. Runs that leave `AGENTX_SERVER_SCRIPT` unset are unaffected. |
| `INFERENCEX_PATH` as a patch target | No change needed. The InferenceX checkout is no longer a framework source root, so integrate, enablement and specialists can no longer patch it, in any session. |
| `AGENTX_PROFILE_WARMUP_S`, `INFERENCE_OPTIMIZER_INLINE_FAST_ACTIONS`, `INFERENCE_OPTIMIZER_INLINE_ACTION_TIMEOUT_S`, `INFERENCE_OPTIMIZER_CLAUDE_FOLLOWS_CODEX` | Delete them. |
| `KERNEL_AGENT_SKIP_MODEL_LOAD`, `KERNEL_AGENT_BUILD_GEAK_RAG_INDEX`, `KERNEL_AGENT_NUM_GPUS` | Delete them; nothing read them. |
| `KERNEL_OPT_BACKEND_BUDGET_MIN`, `KERNEL_OPT_MAX_PARALLEL`, `INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL`, `HYPERLOOM_FORGE_NOMINATION_AUTO`, `HYPERLOOM_LLM_SOURCE_MODEL`, `INFERENCE_OPTIMIZER_ENABLEMENT_ON_EVAL_FAIL`, `INFERENCE_OPTIMIZER_ENABLEMENT_ACCURACY_FLOOR`, `HYPERLOOM_FORGE_REWRITE_BY_FLYDSL`, `GEMM_TUNING_BACKEND`, `INFERENCE_OPTIMIZER_REQUIRE_FRAMEWORK_ACCURACY`, `INFERENCE_OPTIMIZER_MODEL_PATH_ROOTS`, `HYPERLOOM_REAP_BACKEND` | Delete them; nothing read them, and they are gone from `.env.template` and the env reference. A kernel-agent env file still carrying `HYPERLOOM_FORGE_REWRITE_BY_FLYDSL` now draws preflight's `ignoring unsupported kernel-agent env key` warning. |
| `SGLANG_FP8_BLOCKSCALE_CK_MAX_M` | Delete it. The SGLang 0.5.12 CK fp8 block-scale serving patch it triggered is retired (it was already inert on 0.5.20 and later). |
| `HYPERLOOM_CLAUDE_CODE_VERSION` as an npm version spec | Use `stable`, `latest` or an exact `X.Y.Z`. |
| `ISL`, `OSL`, `CONC`, `TP`, `EP`, `PRECISION`, `MAX_MODEL_LEN` pinned through `--extra-env` or exported in the shell | Every `--extra-env` pin is now exported under its own name, so a launch script that pinned `TP=4` or `ISL=4096` that way now runs at that shape, where 1.1.3 ran the default and recorded the pin. On a fresh launch each name resolves as flag, then `$NAME`, then the resumed session's value, then the default, so a shell export of `ISL`, `OSL`, `CONC`, `TP`, `EP` or `PRECISION`, which 1.1.3 ignored, now takes effect; unset any you did not mean. On `--resume-from` the session's recorded `ISL` / `OSL` / `CONC` / `TP` / `EP` / `MAX_MODEL_LEN` win over stale shell exports unless a flag or `--extra-env` re-pins them, but an exported `PRECISION` still overrides the recorded precision: unset it before resuming. |
| A malformed numeric value for a strictly parsed variable | Fix it: the run now stops with `EnvValueError` instead of using the default (list under *Changed*). |

Defaults and gates that move, so the same command behaves differently:

- **AgentX sessions skip the concurrency sweep unless asked.**
  `--enable-conc-sweep` now defaults to unset and resolves to off under AgentX,
  so no sweep runs and no `reports/conc_sweep_summary.json` is written unless
  you pass `--enable-conc-sweep`; `AGENTX_DEFAULT_CONCS` applies only then.
  Synthetic sessions are unchanged, and a resumed session keeps its persisted
  value.
- **Single-node `--ep N` reaches the server.** sglang now gets `--ep-size N`
  and atom `--enable-expert-parallel`, where 1.1.3 ran them TP-only while
  recording `ep=N`; vLLM keeps its flag through replace-mode variants. Server
  arguments that pin an sglang EP size contradicting `--ep` now fail at
  materialize instead of launching with the pin.
- **AgentX grading.** Default aiperf AgentX `integrate_patch` and enablement
  candidates grade accuracy on the request error rate, so a patch above the 10%
  error ceiling is now reverted where 1.1.3 kept it. The comparability gate now
  admits a candidate whose error rate is up to 0.5 percentage points above the
  anchor's instead of refusing any extra failed request, and REVERT reasons
  name the refusing gate (`p50_gain_below_threshold`, `p90_regressed`,
  `output_regressed`, `duration_drift`, `extra_failed_requests`,
  `comparability_inputs_missing`) instead of `median_or_guard_failed`; update
  log scrapers that match the old reason.
- **GLM-5.3 AgentX baselines move.** A GLM-5.3 model now replays the uncapped
  `semianalysis_cc_traces_weka_062126` corpus (mean ISL about 180k, as
  InferenceX does) instead of the 256k-capped one, so its baselines and gains
  are not comparable with 1.1.3 runs. To keep the old corpus, declare it
  canonical with
  `AGENTX_CANONICAL_DATASET=semianalysis_cc_traces_weka_062126_256k`; pinning
  it through `AGENTX_DATASET` instead stamps every result
  `submission_valid=false`, so nothing can KEEP.
- **A failing LLM backend ends the session.** After one hour without a
  successful turn, or after 50 consecutive agent turns that emit no intent, the
  session stops with `stop_reason=backend_unhealthy` (outcome `aborted`; CLOSE
  still publishes the best stack). 1.1.3 retried every tick until the budget
  ran out.

Installation prerequisites move:

- `install.sh` installs Claude Code with the native installer into
  `~/.local/bin` and needs `curl` and access to `claude.ai`; it no longer
  installs Node.js, and an install that cannot provide claude fails.
- The recommended SGLang image is
  `lmsysorg/sglang-rocm:v0.5.21-rocm10-mi30x|mi35x-20261008`.
- During kernel-tool install, variables exported in the shell now win over
  `.env`.
- The kernel agent's bypass route needs an importable TraceLens and fails with
  `tracelens_root_missing` without it.

Session and report consumers have to follow these changes; no aliases are kept:

- `state.json` drops `target_gap_pct`, `budget_extensions`,
  `framework_agent_authoring_enabled`, `last_specialist`,
  `last_consumed_escalate_hint` / `_ts`, `reference_source`,
  `last_profile_kernel_breakdown`, `orchestration_memory_history`,
  `cycle_directive_history`, `current_action`, `target_summary`,
  `warm_replay_min_reproduce_pct` and `resume_pending_revalidation`;
  `orchestration_memory` shrinks to `{next_cycle_directive, for_cycle,
  parse_error}`. `optimization_stack` entries no longer carry
  `stack_member_identities` or `stack_validation_started_at`, and an in-flight
  stack checkpoint written by 1.1.4 cannot be resumed by 1.1.3.
- SBD `phase_action.decision` takes `adopted` / `refused` / `reverted` /
  `failed` / `recorded`. `optimization_journal.json` outcomes are derived from
  that decision: `KEEP` for adopted, `REVERT` for reverted, `no_promote` for
  refused and failed, and the new `recorded` for steps with nothing to adopt
  (analysis, specialist, report). The workflow-evaluation
  `prelude_inputs.double_run` key is gone from both schemas. Both workflow
  contract digests change, and new sessions stamp the v2 workflow-evaluation
  contract (older sessions keep v1).
- Orchestration prompt snapshots are named
  `system_prompt.cN.PHASE.snapshot.md`, and the orchestration agent's
  `UPDATE_STATE` intent is gone: an intent that still carries it is refused
  with `intents[<i>].intent_type '<type>' not in allowed set`.
- Enablement writes `reports/enablement/enablement_setup.sh` and
  `setup_files/` instead of `reports/enablement/patches/` and
  `reports/enablement/artifacts/`.
- `tool_versions[kernel_agent]` reports the package version instead of a git
  SHA. A failed roofline profile is no longer retried on the same config, so
  `retry_after_*` reasons and the `all 3 profile attempts failed` prefix are
  gone, and the CUDA-graph diagnosis is written as
  `diagnostics/roofline_cuda_graph_capture_<task_id>.json`.
- KernelForge workspace JSON artifacts and the forge-loop checkpoint drop
  `schema_version`, and `forge-loop --resume` on a workspace written by an
  older kernelforge fails instead of migrating it.
- A session persisted by v1.1.0 or earlier with a pending non-git warm replay
  now stops on resume with `warm_replay_rollback_failed`.
- `session_breakdown.json` exported by earlier builds carried credential-shaped
  `--extra-env` values in plaintext under `metadata.task_config.launch_env`;
  1.1.4 writes them as `[REDACTED]`, including on re-export. Rotate any
  credential passed that way.

### Added

- **Experience KB.** The wheel ships the `hyperloom_kb` Experience KB.
  `hyperloom-setup` configures a per-workspace loopback KB service and asks
  about a global KB, and FRAMEWORK_AGENT orchestration and specialist prompts
  of throughput-graded runs receive measured Experiences from earlier
  sessions. New: the `hyperloom-kb-serve` and `hyperloom-kb-collect` console
  scripts, `experience_kb_service init-env|ensure|push|pull`, the
  `hyperloom-global-kb` skill, the `HYPERLOOM_KB_URL` / `HYPERLOOM_KB_TOKEN` /
  `HYPERLOOM_GLOBAL_KB_URL` / `HYPERLOOM_GLOBAL_KB_TOKEN` /
  `HYPERLOOM_KB_AUTO_PUSH` / `LOCAL_KB_PLANNER_*` settings,
  `state.json` `experience_kb_injections` and `reports/experience_collect.json`.

- **Per-GPU power measurement and declared power state.** Each round samples
  power through `amd-smi` over its measured phase on the serving cards only and
  writes `gpu_power.json` and `gpu_power_avg_w` (`HYPERLOOM_GPU_POWER_SAMPLING=0`
  disables it, `HYPERLOOM_GPU_POWER_INTERVAL_S` sets the cadence). The platform
  fingerprint records the observed power cap and performance level, and
  `--gpu-power-cap-w` / `--gpu-perf-level` refuse a launch or resume with exit
  `2` when the GPUs do not match the declaration or it cannot be checked.

- **Operator-defined specialist.** `--custom-specialist-prompt-file` and
  `--custom-specialist-description`, passed together, define a
  `custom_specialist` that orchestration can dispatch and that runs at least
  once in FRAMEWORK_AGENT. It persists in state and can be replaced on
  `--resume-from`. Combining it with `--orch-prompt`, `--no-framework-agent`,
  `--research-lane-capacity 0` or `--reset-state` fails at launch.

- **Runtime findings from server logs.** Every measured slot writes
  `runtime_findings.json` from a server-log scan (rules such as
  `vllm.unknown_env`, `feature_disabled`, `aiter.tuned_miss`,
  `runtime.traceback`), surfaced through the `get_runtime_findings` context
  tool, the specialist prompt and `trace-evidence.md`. A specialist patch that
  declares `resolves_finding` can KEEP with up to a 3% drop on the graded axis,
  recorded as `keep_reason=correctness_fix`, and the stack ledger reports
  `attributed_gain_pct_excluding_fixes`.

- **MLPerf agentic backend for AgentX.** `HYPERLOOM_AGENTIC_BACKEND=mlperf`
  (with `HYPERLOOM_AGENTX=1`) runs the MLCommons `run_agentic.sh` harness,
  graded on output throughput with inline-accuracy KEEP gating and a
  trajectory-sized default benchmark timeout. It measures Kimi-K3 only and
  refuses any other model. On that model it accepts every registered serving
  framework (sglang, vllm, atom) and refuses scriptable and unregistered ones,
  and `AGENTX_KEEP_ALIVE_ENV` names the server keep-alive
  variable (default 900 s). **Upgrade note:** the same change makes default
  aiperf AgentX candidates grade accuracy on the request error rate (see
  "Before upgrading from 1.1.3").

- **Release a vanished container's scope.**
  `hyperloom recover --session-dir S --confirm-scope-gone OWNER_SCOPE --confirmation-reason TEXT`
  cancels the unfinished tasks of an execution scope whose container is gone
  and releases its execution and GPU leases, so `optimize --resume-from` can
  resume the session. It refuses the current scope, unexpired leases of another
  scope and tasks that span two scopes.

- **`--max-latency-ms` for scriptable frameworks.** For xdit and custom
  frameworks it vetoes any KEEP whose mean end-to-end latency exceeds the
  ceiling or is unmeasured (`latency_budget_exceeded` / `latency_unmeasured`),
  and a baseline already over it stops the run with
  `baseline_over_latency_budget`. Serving frameworks reject the flag with exit
  `2`.

- **Reproducible enablement setup.** Enablement writes
  `reports/enablement/enablement_setup.sh` plus `setup_files/`, which
  reproduce the setup per root with sha256, mode and package-version checks
  without starting a server; `enablement_setting.sh` runs it before launching,
  and the recipe carries `setup_script: {path, sha256, standalone}`.

- **KernelForge roofline ceiling.** A new `kernelforge roofline-ceiling`
  subcommand writes `performance_ceiling.json` and
  `performance_ceiling_analysis.md`. `forge-loop --roofline-ceiling on|off`
  (default off) feeds per-case attainment to the planner, `--roofline-target`
  stops a campaign on mean attainment, and `forge-rewrite-by-flydsl
  --roofline-ceiling` forwards to its nested forge-loop.

- **`kbmine`.** A standalone CLI under `tools/kbmine` (installed separately
  with `pip install -e tools/kbmine`; nothing in the optimizer imports it)
  estimates a target's inference uplift from Recipe KB and Pulse evidence
  without running a session, configured by `KB_STORE_URL` / `PULSE_URL` /
  `KB_STORE_TOKEN` and `--ca-bundle`.

- **`HYPERLOOM_CLAUDE_CLI_PATH`.** Specialist dispatch and the Claude backend
  resolve the CLI as `HYPERLOOM_CLAUDE_CLI_PATH`, then the installer-recorded
  `GEAK_CLAUDE_BIN`, then `PATH`, so a claude installed outside `PATH` is
  found.

- **Explore skips no-op variants.** Before booting a server, explore drops a
  variant whose every changed setting and environment value already matches
  the running stack's observed launch record, recorded as `SKIPPED_DEDUP`
  `reason=baseline_noop` and counted as `baseline_noop=N` in the dedup log
  line; `launch_evidence` gains `observed_server_config`.

- **Kernel tuning tool and GEAK backend visibility.** Preflight's warn-only
  `check_kernel_tuning_clis` step prints
  `Preflight: WARNING — <cli> not on PATH` for a missing `hipblaslt-bench` or
  `ckProfiler` (skipped with `--no-kernel`), and GEAK's `absent_backends`
  appear in the log, in `state.json` `geak_result.absent_backends` and as a
  kernel-summary takeaway.

- **Richer workflow evaluation.** New sessions stamp the manifest with the v2
  workflow-evaluation contract identity and digest. The Session Breakdown
  records stage milestones, outcome source-event IDs, integration-to-stack
  adoption links and CLOSE receipts, and the final breakdown and package are
  refreshed after CLOSE settles. The SBD wire version stays v6.0.

### Removed

- **`--target-summary`.** The flag advertised a free-text goal summary
  "surfaced in prompts" and no prompt ever read it. The value — the operator's
  string, or one composed from `--target-gain` / `--target-tput` /
  `--target-roofline` and `--max-hours` when the flag was absent — only ever
  landed in the `target_summary` key of the session state record, where the
  orchestration role could also rewrite it to no effect. Launching with the
  flag now fails with an unrecognised-argument error instead of accepting a
  sentence that bought nothing, and `target_summary` is gone from the state
  record: a reader parsing that key must drop it. The objective itself is
  unchanged and still comes from the target flags.

- **`--warm-replay-min-reproduce-pct`.** Its help named the bar that decided
  `status=reproduced` against `status=drift`, and that was never what it did.
  A warm replay is adopted when it clears the session's keep threshold against
  the baseline and wins the lift to `current_best`; the flag's fraction of the
  recipe's *historical* gain — a number measured on another pod — was advisory
  only, annotating the settled outcome with `below_historical_reproduce_pct`
  and `historical_reproduce_bar_pct` and rejecting nothing. Those two keys are
  gone from `warm_replay_outcome` and from the SBD V6 `warm_replay` verdict
  row; removing the flag leaves the reproduced/drift decision unchanged (a
  local legacy replay is now held to the keep threshold rather than any
  positive gain, but that comes from warm replay honouring graded adoption,
  below), and `--no-warm-replay` and `--warm-replay-min-confidence` are
  untouched. The
  flag's cost was a threshold operators could tune for hours without moving an
  adoption.

- **`--plateau-kernel-revert-streak` / `--plateau-kernel-keep-gain` /
  `--plateau-kernel-lookback`.** The three thresholds tuned one advisory
  paragraph in the orchestration prompt — "KERNEL_AGENT plateau detected:
  REVERT streak or low recent KEEP gain" — and nothing else. KERNEL leaves when
  its kernel work drains, when the rewrite controller settles the macro cycle,
  when the agent has gone the idle window without progress, or on its phase
  budget — never on that verdict, so the knobs changed what the model was told
  and no exit. The paragraph and the computation behind
  it go with the flags, and the plateau advisory now renders for OPTIMIZE only,
  where its text reads the same predicate the exit rule does. The
  `--plateau-explore-*` flags are untouched.

- **The old command-line entry points.** `inference_optimizer`,
  `quantization-agent`, `python -m hyperloom.inference_optimizer.{cli,multi_node,setup}`
  and the `python -m ...tools.*` session tools are gone; every command now runs
  as `hyperloom <command>` (see *Changed*). Removed with them:
  `--kernel-codex` / `--kernel-claude` / `--kernel-prompt` (ignored no-ops),
  the `--max-minutes-*-pct` and `--phase-budget-explore-pct` aliases,
  `recover --backfill-trace` (use `hyperloom session backfill`),
  `INFERENCE_OPTIMIZER_CLAUDE_FOLLOWS_CODEX`, and the critic runtime's
  `init-session`, `close-session`, `list-priors`, `write-verdict`,
  `write-kb-drafts` and `add-contradiction` subcommands, which nothing called.
  A removed flag now fails argument parsing instead of being ignored.

- **In-process specialist dispatch.** `--specialist-dispatch-mode` and
  `--specialist-per-turn-max-seconds` are no longer accepted: every specialist
  runs as a Claude CLI subprocess in its own worktree. A session that finds no
  claude CLI (`HYPERLOOM_CLAUDE_CLI_PATH`, `GEAK_CLAUDE_BIN` or `PATH`) stops at
  startup instead of silently running every specialist in-process without
  source roots or tools. A specialist that ends without `specialist_done.json`
  is recorded as `no_specialist_done_emitted`.

- **Kernelforge readers for its own historical workspace formats.**
  `forge-loop --resume` on a workspace written by an older kernelforge now fails
  with a missing/unknown-field error instead of migrating it; start a fresh
  campaign. Workspace JSON artifacts and the forge-loop result's `checkpoint`
  no longer carry `schema_version`. `forge-rewrite-by-flydsl --op-name` is gone
  (use `--logical-op-name`), agent providers registered only under the
  `kernel_agents.agent_providers` entry-point group are no longer loaded
  (register under `kernelforge.agent_providers`), and the already-ignored
  `KERNEL_AGENTS_MAX_TURNS` no longer warns.

- **The kernel agent's standalone install and run paths.** `install.sh` runs
  the kernel-tool steps with its own helpers, lock and environment, so
  `--skip-kernel-agent` is gone and `install_kernel_tools.sh` run on its own
  exits `2`. The kernel request handlers run forge GEMM tuning and forge fusion
  as one child process each, so the `forge_gemm_tuning` / `forge_fusion`
  wrappers and their `forge_*_input.json` files are gone, as are the
  trace-analysis options no handler passed.

- **`HYPERLOOM_KERNEL_AGENT_ROOT` / `KERNEL_AGENT_ROOT`.** The kernel tools now
  live in `hyperloom.orchestrator.trace_analysis` and
  `hyperloom.orchestrator.kernel` and are started as
  `sys.executable -m <module>`, so nothing reads a kernel-agent source root and
  the `hyperloom.agents.kernel.tools.*` paths are gone. The installers stop
  writing the two variables and still scrub them from the shell.

- **The AgentX agentic-recipe launch path.** Pointing `AGENTX_SERVER_SCRIPT` at
  an InferenceX `single_node/agentic/*.sh` recipe, which ran its own replay,
  could not be profiled and depended on uncommitted patches in a shared
  InferenceX checkout. AgentX now has one launch path: the generic InferenceX
  launcher with the model's arguments from `--server-args` and its environment
  from `--extra-env`. The recipe renderer, the `*.hl-<digest>.sh` recipe
  copies and the `recipe_lever_unavailable` error class go with it, and the
  InferenceX checkout stops being a patchable source root.

- **The orchestration delegation leftovers.** The `UPDATE_STATE` intent,
  `INFERENCE_OPTIMIZER_INLINE_FAST_ACTIONS` /
  `INFERENCE_OPTIMIZER_INLINE_ACTION_TIMEOUT_S` and the `state.json` fields
  listed under "Before upgrading from 1.1.3" are removed. `extend_*_budget`
  hints and resume phase budgets now take effect, warm replay honours graded
  adoption, and vLLM profile runs no longer build the profiler summary table.

- **The SGLang 0.5.12 CK fp8 block-scale serving patch.** Forge + SGLang + FP8
  block-scale runs on gfx942 no longer enqueue the synthetic
  `ck_blockscale_backend_switch` candidate or spend a full E2E round on it, and
  the `serving_patches/` resource tree is gone. No supported stack loses an
  optimization: the patch was already inert on SGLang 0.5.20 and later.

- **Twelve environment variables nothing read** are dropped from
  `.env.template` and the env reference (listed under "Before upgrading from
  1.1.3"), so operators stop setting knobs that did nothing. The orchestration
  prompt no longer describes a PARTIAL kernel-opt retry cap that never existed.

### Changed

- **One `hyperloom <command>` entry point.** Setup, preflight, optimization,
  recovery, quantization, the session tools and the multi-node CLI all run as
  `hyperloom <command>` (or `python -m hyperloom <command>`), and
  `hyperloom --help` lists them: `hyperloom setup`, `hyperloom check`,
  `hyperloom optimize`, `hyperloom recover` (was `recover-session`),
  `hyperloom quantize`, `hyperloom session breakdown|report|backfill|events|state`
  and `hyperloom multi-node <sub>`. `--verbose` now goes after the command, and
  a failed `--quantize` prelude exits `4` (exit `3` is only "session lock
  held"). Invalid PD combinations on rayjob now fail at startup instead of being
  dropped. External launchers must switch; the old-to-new table is in
  [Upgrade Hyperloom version](reference/upgrade.md).

- **The installer owns the one claude CLI every consumer runs.** It installs
  Claude Code with the native installer into `~/.local/bin`, puts that first on
  `PATH` before GEAK's bootstrap runs, and writes the same file into
  `GEAK_CLAUDE_BIN`, so GEAK, the specialists and Forge resolve one copy. An
  install that cannot provide claude (no curl, or `claude.ai` unreachable) now
  exits with an error instead of a warning. `HYPERLOOM_CLAUDE_CODE_VERSION`
  takes the native installer's targets (`stable`, `latest` or an exact
  `X.Y.Z`) rather than an npm version spec, and the installer no longer installs
  Node.js.

- **Bump validated SGLang stack to 0.5.21 (ROCm 10 docker).** Updates
  `SGLANG_REF`, `SGLANG_PRETEND_VERSION`, recommended
  `lmsysorg/sglang-rocm:v0.5.21-rocm10-*` image tags, `models.tsv`, and the
  quick-start `Dockerfile`. **Upgrade note:** move from
  `v0.5.20-rocm10-mi30x|mi35x-20260920` to
  `v0.5.21-rocm10-mi30x|mi35x-20261008` (or match `SGLANG_REF` on bare metal).

- **A malformed environment value stops the run instead of being silently
  replaced by the default.** `INFERENCE_OPTIMIZER_CYCLE_HOURS`,
  `INFERENCE_OPTIMIZER_BACKEND_ERROR_STREAK_THRESHOLD`,
  `INFERENCE_OPTIMIZER_REACTOR_TURN_TIMEOUT_SEC`,
  `INFERENCE_OPTIMIZER_KERNEL_IDLE_MAX_TICKS`,
  `INFERENCE_OPTIMIZER_KERNEL_IDLE_MIN_SECONDS`,
  `INFERENCE_OPTIMIZER_CYCLE_RELOOP_MIN_REMAINING_SEC`, `SWEEP_VARIANT_TIMEOUT_SEC`,
  `HYPERLOOM_FUSION_KEEP_PCT`, `HYPERLOOM_GEMM_PAIRED_PAIRS`, `GEAK_E2E_TIMEOUT_S`,
  `GEAK_BUDGET_MARGIN_S`, `GEAK_MIN_RUN_S` and `GEAK_TERM_GRACE_S`, and
  `MAX_MODEL_LEN` and `GPU_MEMORY_UTILIZATION` where they stand in for a
  `--max-model-len` / `--gpu-memory-utilization` the baseline server arguments
  do not carry, now raise `EnvValueError` naming the variable and the value when
  they are set to something that is not a number.
  A typo used to be swallowed: the run proceeded on the default, so a session
  launched with `GEAK_MIN_RUN_S=600s` or `SWEEP_VARIANT_TIMEOUT_SEC=40m` spent
  its hours under limits nobody had chosen, and the report gave no sign the
  setting had been ignored. The two KERNEL idle variables are read at import,
  so a bad value there fails the launch outright. Out-of-range is still not the
  same as malformed: a non-positive or non-finite reactor turn timeout, and a
  KERNEL idle tick count below `1` or an idle window at or below `0`, continue
  to fall back to the default.

- **Specialists run alongside the tick.** The tick no longer waits for
  dispatched specialists: orchestration and the Critic take turns while work
  runs, so long specialists mean more LLM calls. FRAMEWORK candidates are
  authored in parallel up to `research_lane_capacity`, untested specialist
  proposals are benched four at a time as `auto-explore-c<cycle>-<hash>`
  explore grids (also on a single node), CPU specialists see no GPUs and use
  workspace-local compiler caches, the specialist wall budget is 60 minutes for
  a patch and 10 minutes for research, and specialists honour
  `CLAUDE_CONFIG_DIR`.

- **The kernel agent reads TraceLens directly.** It reads TraceLens
  `analysis.json` and resolves kernel sources through TraceLens
  `resolve_kernel_source`, so `kernel_candidates.json` source paths come from
  the installed framework tree (fewer resolve when it is not mounted).
  Tensile, CK and MIOpen kernels land in `skipped_kernels` with a dispatcher
  `source_file`, Triton-named wrappers over native kernels are no longer routed
  as patchable, and bypass rows gain `identity_route` / `op_to_source_*`.

- **Kernel-opt dispatch order.** The kernel_opt gate orders untried hot kernels
  by TraceLens priority rank and impact score instead of raw `gpu_pct`, which
  changes the forge-route queue and the order of `untried_hot_kernels` in the
  mission summary and report. `HYPERLOOM_KERNEL_OPT_GATE_TOP_N` now caps ranked
  and unranked (for example NCCL) rows separately, so the gate can return up to
  `2N` ids.

- **Explore no longer pre-drops variants by probing `--help`.** Variants
  carrying `--enable-flashinfer-mla`, `--enable-deepep-moe` or
  `--enable-ep-moe` now launch and, on a framework that lacks the flag, fail at
  argument parsing (recorded as failed with `server_log_path`) instead of
  being skipped as "likely too old"; the "compatibility probe ... produced no
  help text" warning is gone.

- **RULE F2 no longer points orchestration at
  `INFERENCE_OPTIMIZER_RESCUE_PATHS`**, which it could not set.

- **KernelForge packaged resources move out of `kernelforge/data`.** Example
  tasks live under `src/kernelforge/examples/` (`resource_path('examples')`)
  and the shipped knowledge base under
  `kernelforge/knowledge/local_wiki/resources/`. **Upgrade note:** the 1.1.3
  quick-start path `src/kernelforge/data/examples/...` no longer exists;
  `resource_path('local_knowledge')` and `KERNELFORGE_PROJECT_ROOT` overrides
  keep working.

- **A rejected roofline profile logs one warning.** The profile runs once and
  a rejected attempt logs
  `roofline profile failed (<error class>): <message>`, and profile timeline
  rows' `duration_sec` now include the post-attempt liveness probe and
  capture-evidence collection.

### Fixed

Resume and recovery:

- **A session resumed in SWEEP with no recorded sweep outcome** (for example
  one cancelled by `recover`) re-fires the SWEEP entry hook once and measures
  its sweep instead of idling until the lost sweep's grant runs out.
- **Resuming on the same node after a container restart.** The lease owner
  scope adds the start time of the PID namespace's PID 1 (two-part scopes are
  still recognised), so `recover --confirm-scope-gone` and the resume guard no
  longer mistake a lost container's scope for the current one.
- **A container lost mid-`integrate_patch`** no longer always stops the resume
  with `patch_recovery_incomplete`: once its tasks are confirmed stopped the
  session resumes, the resume report carries `rolled_back_pending_integrate`
  (`tree_replaced: true`, `unrestored_stash_oid`; the lost checkout's
  uncommitted stash is not recovered), and an env-only or server-args-only KEEP
  that reached the task row is replayed into the stack
  (`replayed_pending_integrate`).
- **Every sweep, explore and baseline round stops the server its AgentX client
  booted** (read from `agentx_server.pid`) however the round ended, so a round
  killed at the benchmark cap or cancelled no longer leaves a server holding
  the port and GPU memory for later rungs to measure against or fail on with
  `magpie_nonzero_invalid_measurement`. A server that outlives SIGKILL is
  reported as `server_survived_teardown:<pid>` in `nonfatal_warnings`.
- **On `--resume-from`, recorded workload shape outranks stale shell exports.**
  The session's `ISL` / `OSL` / `CONC` / `TP` / `EP` / `MAX_MODEL_LEN` win over
  exports left in the shell (for example by sourcing a generated repro script)
  unless a flag or `--extra-env` re-pins them.
- **`optimize --resume-from` without `--framework`** runs preflight against
  the session's persisted framework instead of the sglang default; a
  conflicting explicit `--framework`, an unregistered persisted framework, or a
  manifest without `state.json` is refused with exit `2` before any install
  step.
- **An interrupted stack validation is unwound from its apply rows** even when
  the ledger cannot attribute them, where resume used to halt with
  `stack recovery lacks structured member evidence` and a patched tree; a
  decided KEEP it cannot bind stops with `patch_recovery_incomplete` instead of
  an unhandled `KeyError`. **Upgrade note:** see the persisted-state changes
  under "Before upgrading from 1.1.3".
- **A SWEEP entry or resume that cannot settle or bind a stack checkpoint**
  stops with `patch_recovery_incomplete` instead of logging and continuing, an
  `optimization_stack` row naming no members stops at load, and a guard-only
  interrupted attempt left by v1.1.2 is recovered at resume with its members
  selectable again.
- **Non-git warm-replay teardown restores through the backup ledger** and
  verifies `pre_image_sha256`, so a backup corrupted after apply leaves the
  tree untouched with a `baseline_executor: nogit patch restore failed` warning
  instead of copying corrupt bytes and reporting success. **Upgrade note:** a
  session persisted by v1.1.0 or earlier with a pending non-git warm replay
  stops on resume with `warm_replay_rollback_failed`.
- **A cancelled `integrate_patch` with a restored critic verdict** always gets
  its `-reconcile<n>` child: the check uses exact case-sensitive prefix
  matching instead of SQL `LIKE`, which skipped near-identical keys.
- **A resumed session leg no longer re-pushes earlier spans and Scores** into
  the shared Langfuse trace, and a flush after CLOSE ships rows recorded after
  CLOSE; `langfuse_receipt.json` gains a `rows_sent` cursor map (the old keys
  are still read) and `decision_trace.jsonl` rows gain `decision_id`.
- **A session started under workflow contract v1 and resumed on current code**
  exports a breakdown that validates against the v1 schema, where every
  FRAMEWORK_AGENT, KERNEL_AGENT or SWEEP exit used to fail export validation.

Scheduling, phases and the LLM backend:

- **A failing LLM backend is retried with exponential backoff** (2 s doubling
  to 5 min) instead of every tick. **Upgrade note:** after an hour without a
  successful turn the session stops with `stop_reason=backend_unhealthy`, and
  both workflow contract digests change.
- **An orchestrator Claude turn is cut** after
  `INFERENCE_OPTIMIZER_CLAUDE_TURN_TIMEOUT_SEC` (default 1500 s), with leftover
  CLI processes killed and a `backend_error` recorded, instead of stalling the
  tick loop for 70-110 minutes behind CLI `api_retry` frames; CLI retries are
  logged.
- **The `emit_intent` MCP tool is offered when Claude Code's remote flags are
  unavailable** (for example under `CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC`,
  `DISABLE_TELEMETRY` or `DO_NOT_TRACK`); `agents/orchestration/mcp_setup.json`
  records the CLI version, MCP tools and server status, and 50 consecutive
  intent-less turns stop the session with `backend_unhealthy` after a
  high-severity `no_intent_streak` observation.
- **Long runs no longer fall into a permanent "Prompt is too long" loop.** The
  specialist findings, inbox, priors-match and `current_best=` prompt sections
  are bounded, with notes for what was omitted or held for the next turn.
- **GPU action leases** (baseline, roofline, profile, explore,
  `integrate_patch`) last at least one measured baseline round plus 10%
  instead of the static catalogue TTL, so `lease_expires_in_sec` no longer goes
  negative on healthy slow rounds and orchestration stops needing routine
  `extend_lease` calls.
- **A cancelled concurrency sweep** stops at its current rung and reports
  `status: "cancelled"` with `cancel_reason`, so `stop_reason` names the cause
  (for example `time_exhausted`) instead of `sweep_failed`, and SWEEP no longer
  cancels its own sweep at the per-phase cap while it is inside the budget SWEEP
  granted it.
- **PRELUDE and ENABLEMENT wait for the initial roofline** (`reason=prelude_initial`)
  instead of killing it at the phase-exit barrier, so the kernel phase gets its
  analysis and candidates, and a cancelled roofline stops after one attempt
  instead of booting two more servers.
- **After a GEAK run ends with status `timeout`**, the GEAK-owned KERNEL phase
  exits to the next phase instead of staying open on per-kernel pending work.
- **A `skip_to_kernel` hint emitted outside FRAMEWORK_AGENT** is dropped with a
  log line instead of ending FRAMEWORK_AGENT, or the whole optimize phase when
  the kernel phase is disabled, on entry.
- **`--no-kernel` and `--no-framework-agent` budgets.** With `--no-kernel`
  FRAMEWORK_AGENT dispatches for its full redistributed share (79.5% instead of
  pausing at 38%); with `--no-framework-agent` KERNEL_AGENT dispatch, lane TTL,
  GEAK kill budget and rewrite timeouts are sized from 81.3% instead of 47%.
  The enablement stall cap closes the lane `failed` / `stalled` with
  `enablement_attempts_exhausted` instead of raising `AttributeError`.
- **A session that closes on `skip_to_close` without ever measuring a
  baseline** ends with `stop_reason=baseline_failed` instead of the
  success-shaped `time_exhausted` or `global_converged`.
- **The SWEEP status `cycle_reloop_feasible`** shown to orchestration reports
  the same decision the transition takes, and the baseline warm-patch HEAD read
  timeout rises from 5 s to 30 s.
- **On terminal fallback and deadline paths CLOSE is recorded before** the
  report and breakdown tasks are dispatched, so their phase attribution matches
  the CLOSE receipt.

Measurement, profiling and AgentX:

- **AgentX roofline profiles run again.** `AGENTX_PHASE_WAIT_TIMEOUT_S` is
  exported as an integer (`'7800'`, not `'7800.0'`, which made
  `aiperf_client.sh` exit `2`), and the critic preflight's 60 s timeout reaches
  the HTTP transport, so a slow gateway no longer stops the session.
  **Upgrade note:** a failed roofline profile is no longer retried on the same
  config (`max_attempts` 1; the `retry_after_*` reasons and
  `PROFILE_ATTEMPT_AFTER_*` exports are gone), and the CUDA-graph diagnosis is
  renamed to `diagnostics/roofline_cuda_graph_capture_<task_id>.json`.
- **Server-bounded AgentX profile captures** (SGLang `num_steps`, vLLM
  `--profiler-config.max_iterations`) end on the server's own step bound
  instead of the client's 20 s window and a redundant `/stop_profile`, so
  complete traces are no longer recorded `stop_profile_failed` or come back
  empty. New failure reason `bounded_capture_incomplete`, a one-time safety
  stop at `AGENTX_TRACE_FIRST_FILE_TIMEOUT_S` (default 900 s), and
  `AGENTX_PROFILE_WINDOW_S` applies only to unbounded captures.
- **The roofline records the last profile only with a successful trace
  analysis**, so a failed analysis keeps the previous good record instead of
  showing an unanalysed trace as `succeeded` (which drew false "roofline stuck
  in trace_analyze" alerts).
- **vLLM profile runs produce GPU kernel and `cuda_runtime` events.** The
  profile start-up shim drops `ROCPROFILER_REGISTER_LIBRARY` before the
  EngineCore child loads HIP, so traces are no longer CPU-only.
- **SGLang 0.5.18-0.5.21 per-batch-size CUDA-graph capture works on profiling
  rounds.** The TraceLens `sglang_gc_patch` is applied, so DSA models no longer
  crash during capture, capture traces are named `bs_<n>[_dense|_sparse]`, and
  the shape manifest keeps dense, sparse and draft variants apart.
  Multi-node DSA runs need `HYPERLOOM_SGLANG_VERSION_PIN` when the version
  cannot be detected, and `HYPERLOOM_ENABLE_PATCH=0` disables the patch.
- **Profile and roofline servers import the same trees as baseline.** Every
  launch-path prepend site builds `PYTHONPATH` / `PATH` / `LD_LIBRARY_PATH` on
  the inherited value when the config lacks the key, instead of dropping the
  operator's and the image's entries.
- **A vLLM GEMM tuning request records the model's GEMM shapes** (the
  `PYTORCH_TUNABLEOP_*` recording variables are no longer unset in the
  shape-capture config) instead of always failing with `shape_capture_failed`.
- **AIPerf KV and server metrics are filtered by `phase_time_ranges`**, so
  whole-run counters are no longer attributed to both warmup and profiling.
- **Under `HYPERLOOM_AGENTX=1` an explore KEEP graded on interactivity**
  promotes to `current_best` instead of being dropped as
  `candidate_axes_missing` or turned into a lift REVERT.
- **GLM-5.3 AgentX sessions replay the uncapped corpus**, matching InferenceX.
  **Upgrade note:** baselines and gains shift; see "Before upgrading from
  1.1.3".
- **Guard-refused explore variants render with their graded gain and refusal
  reason** in orchestration and critic prompts (new `graded_gain_pct` and
  `reason` keys on tested rows) instead of `no_meas`.
- **A baseline boot no longer writes `runs/baseline/<id>/aiter_jit_backup/`**
  (about 3.5 GB per sglang session) into the session directory; the dropped
  aiter modules are parked in `<aiter>/jit/hyperloom_jit_backup`. Explore
  tested and winner rows carry every graded axis, adding
  `output_tput_per_gpu`.
- **MoE PerfModel ceilings follow a runtime `--quantization` / `--dtype`
  override** for models whose `config.json` declares `expert_dtype` at the
  same precision as the other weights.
- **An atom baseline's `EXTRA_ATOM_ARGS` are read**, so KB workload tags appear
  on atom recipe, lesson and pitfall records; measured results are unaffected.

Enablement:

- **A non-install setup command proposed by an enablement round** ends the
  round `apply_failed` (`setup_commands_refused`) naming every refused command,
  and the next mandate carries a `SETUP COMMAND REFUSED` block, instead of the
  command being dropped silently. The run-provisioned venv `pip` is accepted,
  stopped or Ray-infeasible baselines no longer become boot evidence, and a
  kept round no longer banks a duplicated `extra_server_args` flag.
- **Enablement ladder rungs 3 and 4 run.** Mandates carry the resolved failure
  class, PR discovery searches for the failure, a runtime-only round is
  benched, patches apply to the tree the attempt imports, and later rounds boot
  on a kept runtime. A queued `enablement_specialist` task from an older
  `coordinator.db` costs one failed round on `--resume-from`.
- **Enablement rounds that move a boot failure to a new root cause** are kept
  as advanced instead of reverted as stalled; PR discovery no longer fails with
  GitHub HTTP 422, roofline profile runs skip the gsm8k eval (about 40 minutes
  shorter), and repeated watermark crossings get a fresh roofline.
- **A passing enablement revalidation baseline** that finishes between the
  dispatcher pump and the next reconcile is no longer lost, so the lane ends
  `succeeded` with its `setup_script` record.
- **vLLM enablements with JSON- or enum-valued arguments** (for example
  `--compilation-config`) are compared structurally from vLLM's
  `non-default args:` record, so `replay_sufficiency` can reach `sufficient`;
  KEEP assertions always include the serving framework version.
- **Enablement KEEP capture fails closed.** A KEEP whose framework tree,
  materialized config or build output cannot be read is recorded as
  `levers_unverified` / `build_carry_unverified`, and a baseline with a
  missing, invalid or empty materialized config fails with `ConfigUnreadable`.
- **A patch already bound to a patch root by an earlier round** keeps that
  tree at KEEP instead of being re-pointed by the KEEP round's done payload.
- **Credentialed URLs in rejected setup commands are redacted** to
  `<index_url>` / `<vcs_url>` instead of being echoed verbatim.

Install, preflight and environment:

- **Specialists, the Critic and GEAK authenticate to a gateway.** Preflight
  expands `${ANTHROPIC_API_KEY}`, `${ANTHROPIC_AUTH_TOKEN}` and
  `${OPENAI_API_KEY}` references in `ANTHROPIC_CUSTOM_HEADERS` /
  `OPENAI_CUSTOM_HEADERS` instead of every call failing with HTTP 401.
- **ROCm 10 vLLM images.** Both installers find the image Python in
  `/opt/python/bin` instead of refusing with "no torch" or "pip is required".
- **`install.sh` run as a background job completes.** Every `ray start`
  detaches the Ray daemons into their own session with stdin from `/dev/null`.
- **A hand-started local Ray head without the `serving_slot` resource** is
  restarted with it when that is safe; otherwise the error says why and the
  session stops `baseline_failed` on the first `ray_cluster_infeasible`
  baseline. Two identical `subprocess_nonzero` baseline failures stop at two
  attempts.
- **Preflight honours exported `RAY_VERSION` and `RAY_CLI_CLICK_MAX_VERSION`**,
  and an explicit Ray pin must match exactly and is never replaced by the
  automatic fallback.
- **Forge GPU-target detection ignores rocminfo's generic ISA names** (for
  example `amdgcn-amd-amdhsa--gfx12-generic`).
- **xDiT sessions resolve their framework tree**, atom's eager fallback passes
  `--enforce-eager`, and an unregistered framework now fails visibly (resume
  exits `2`) instead of silently using sglang defaults.
- **Integrate refuses a git tree whose HEAD cannot be read** with an `OSError`
  naming the tree before applying any patch, instead of reporting patches
  reversed while they stayed on the tree.

Reports and exports:

- **Credentials in the exported launch environment are masked.**
  Credential-shaped variables in `metadata.task_config.launch_env` of
  `session_breakdown.json` are written as `[REDACTED]`, also on re-export of
  older sessions. **Upgrade note:** rotate any credential passed through
  `--extra-env` before this release.
- **Session Breakdown keeps rows whose external identifier contains a colon**
  (for example `aiter:paged_attention_ragged`).
- **Config-arm explore attempts in SBD v6 carry `attempt.proposal_ref`**, and
  Langfuse Generation metadata includes `call_id`.
- **A failing inbox, recent-outcomes or running-tasks context tool** logs its
  traceback under `context tool <name> failed`.
- **Bypass failure reports** carry the configured `profiling_enabled`, a
  `phase=server` never-ready report measures `execution_time` from boot start,
  and in-progress `conc_sweep_summary.json` lists `concs_requested` in caller
  order.

## Hyperloom 1.1.3 release

The [1.1.3 release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.1.3)
is a patch release on top of 1.1.2. The per-change history is on the
[Releases](https://github.com/AMD-AGI/Hyperloom/releases) page.

### Added

- **`--repo-scope` / `--no-repo-scope` for `forge-fuse`.** Give discovery and
  authoring the whole framework repository instead of one resolved file.
  Requires `--discover llm` or `--discover anchored`. Use when the fusion chain
  is not in the arch-class model file and you do not want to name its location.

- **`--fuse-kernel` for `forge-fuse`.** Name the GPU kernel a fusion is built
  around (full trace spelling); neighbours are taken from the trace and
  aggregated by op category. Implies `--discover anchored`. With `--dry-run`,
  resolution writes `fusion_anchor.json` and stops with the new
  `anchor_resolved` verdict. The manifest gains an `anchor` key (`null` when
  unused).

### Fixed

- **Fusion discovery no longer proposes folding work into a tuned library GEMM's
  epilogue.** Prefer the consumer's prologue so a dtype change does not knock a
  tuned GEMM off its table (anchored discovery names that direction). This is a
  discovery-time constraint, not a post-fusion gate.

- **`forge-fuse` forwards a concrete agent provider to `forge-loop`.** Default
  `--agent-backend auto` was passed through unchanged; `forge-loop` rejects
  `auto`, so campaigns died after discovery. The resolved provider is now
  forwarded.

## Hyperloom 1.1.2 release

The [1.1.2 release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.1.2)
is a patch release on top of 1.1.1. ROCm 10 becomes a validated stack, bring-up
gets a phase of its own, and the automatic supervision layer is retired in
favour of explicit resume and recovery. The session record resumes across the
upgrade: `ensure_schema` migrates a 1.1.1 `coordinator.db` on the way in. The
command line, the environment contract and several report fields do move; see
"Before upgrading from 1.1.1" below.

### Before upgrading from 1.1.1

Two optimizer options and one console script 1.1.1 accepted are removed. The
parser is strict, so a launch or resume command that still carries an option
exits with `unrecognized arguments` before the session starts.

| Removed | What to do |
|---|---|
| `--conc-sweep-timeout-sec` | Delete it. Each benchmark spawn is bounded by `INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC` (default `7800`) and, once a ready marker is seen, `INFERENCE_OPTIMIZER_BENCHMARK_SILENCE_TIMEOUT_SEC` (default `600`); sweep admission uses measured durations and the session deadline. |
| `--recipe-kb-strict-fingerprint` | Delete it. It was read nowhere; partition mode and `ep` are now part of the KB key, so the mismatch it promised to catch cannot occur. |
| `robustness-agent` console script, `tools/robustness_monitor.sh.example` | Nothing restarts a session automatically any more. Resume explicitly with `--resume-from`. |

Resume now refuses, before any write or dispatch, when the session holds
execution ownership it cannot verify; ownership recorded by an earlier build is
diagnosed rather than assumed dead. After confirming that the task's process
tree, remote workers and Ray actor have stopped, release it with
`inference_optimizer recover-session --session-dir PATH --confirm-stopped TASK_ID --confirmation-reason TEXT`.

Environment variables fail differently: nothing refuses them, so a box that
still exports one starts normally and behaves as though it were unset. Find
them by reading launch scripts and `.env`.

| Removed or changed variable | What to do |
|---|---|
| `LLM_GATEWAY_KEY` as the only OpenAI-side credential | Rename it to `OPENAI_API_KEY`; value and endpoint are unchanged. `OPENAI_API_KEY` and the Anthropic-side names are the only credentials read, and `LLM_API_KEY`, `AMD_LLM_API_KEY`, `LLM_GATEWAY_KEY`, `AMD_API_KEY`, `LLM_PROXY_BASE_URL`, `LLM_PROXY_API_KEY` and `GEAK_WORK_DIR` are no longer forwarded to workers. |
| `HYPERLOOM_SUPERVISOR`, `HYPERLOOM_SUPERVISOR_ENFORCE`, `HYPERLOOM_SUPERVISOR_TICK_STALL_SEC`, `ROBUSTNESS_AGENT_ROOT`, `ROBUSTNESS_LLM_RCA_DISABLED` | Delete them. The Supervisor, the Monitor and the Robustness agent are gone. |
| `INFERENCE_OPTIMIZER_DISABLE_LOCAL_INFERENCEX`, `INFERENCE_OPTIMIZER_LOCAL_INFERENCEX_ROOT` | Delete them. Baseline benchmarks run from `$INFERENCEX_PATH` as set; a network-filesystem checkout warns once and is recorded as `network_fs`. |
| `MAGPIE_PATCH_STRICT` | Delete it. It had no effect once the atomic-write patch was dropped. |

Installation prerequisites move with the stack:

- TraceLens and its dependencies must be pre-installed in the coordinator's
  Python interpreter; analysis no longer runs `pip install -e .` per request.
  SGLang >= 0.5.18 profiles through TraceLens `kernel_shape_tool` under
  `TRACELENS_ROOT` instead of git-applied roofline patches.
- The recommended SGLang image is
  `lmsysorg/sglang-rocm:v0.5.20-rocm10-mi30x|mi35x-20260920`, and the vLLM image
  is `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0`.
- Bare-metal vLLM accepts ROCm 7.2.x (the `0.29.0+rocm723` wheel) or ROCm 10
  (a source build of `VLLM_SOURCE_REF`); any other ROCm stack is rejected, and
  `VLLM_INSTALL_METHOD` can only confirm the detected route.
- The Slurm sbatch no longer forwards `ROCR_VISIBLE_DEVICES`,
  `HIP_VISIBLE_DEVICES` or `CUDA_VISIBLE_DEVICES` into the container; isolation
  comes from the cgroup.

Report consumers have to follow these field changes; no aliases are kept:

- SBD V6 `conc_sweep` comparison rows: `baseline_throughput` → `baseline_value`
  and `optimized_throughput` → `optimized_value`. V6 failure rows are keyed by
  `stage` instead of `phase`.
- `reports/kernel_optimization_summary.json` moves to `schema_version` 2 with a
  `lane_totals` block, and `kernel_opt_outcome` can be `unvalidated`.
- A concurrency sweep that produced no comparable pair stops as `sweep_failed`
  rather than `sweep_done`.
- `plateau_overrides.explore_empty_streak` counts benched rounds rather than
  variants, and the GEMM and fusion E2E KEEP threshold defaults to 1% instead of
  3%.

### 1.1.2 highlights

- **ROCm 10 is a validated stack.** Docker runs SGLang 0.5.20 and vLLM on
  ROCm 10 user space, bare-metal vLLM builds from source on a ROCm 10 host, and
  `rocprof-compute` is installed and found on TheRock's pip-packaged ROCm. A
  prebuilt SGLang image whose torch fails under `rocprofv3` is marked
  profiling-unavailable instead of hanging the round.

- **ENABLEMENT is the sixth phase of the loop.** Bring-up leaves
  FRAMEWORK_AGENT and gets its own entry and exit, `phase_history` rows and
  report section, bounded by the wall clock and `ENABLEMENT_MAX_ATTEMPTS`
  rather than a budget share. A KEEP now ships an ordered replay recipe with a
  `sufficient` / `insufficient` verdict under `enablement.recipe` in
  `session_breakdown.json`.

- **Automatic supervision is retired; limits and recovery are explicit.** The
  Robustness agent, the runtime `recover` action and the Monitor/Supervisor
  auto-resume are removed. Each benchmark spawn has a hard deadline plus a
  silence limit armed only by a ready marker, resume refuses unverifiable
  ownership instead of guessing, and `recover-session --confirm-stopped`
  releases a stopped task's residue under the session lock.

- **ATOM is supported end to end.** Its recovery workers are reaped, its kernel
  backend defaults to forge, and detailed profiling adds `--mark-trace` and
  ATOM's annotation switches only when the installed build supports them.

- **KernelForge gains an assembly backend and keeps more of its work.**
  `forge-loop --kernel-backend assembly` optimizes compiler-emitted AMDGPU
  assembly behind a numerical contract. Warm starts fall back to a fuzzy KB
  match across framework versions and GPUs, a lane's patch is merged against
  the KEEPs landed before it, and a killed campaign's proven fusions are
  salvaged instead of reported as REVERT.

- **SBD V6 says which axis a session graded on.** `metadata.grading`, per-figure
  `perf` and `graded_on` make AgentX and synthetic sessions distinguishable, and
  the timeline records the events it used to drop, closing an event cut short by
  a recording error as `interrupted`.

### Other changes

- **Bump validated SGLang stack to 0.5.20 (ROCm 10 docker).** Updates
  `SGLANG_REF`, `SGLANG_PRETEND_VERSION`, recommended
  `lmsysorg/sglang-rocm:v0.5.20-rocm10-*` image tags, `models.tsv`, and the
  quick-start `Dockerfile`. Kernel-shape profiling for SGLang >= 0.5.18 uses
  TraceLens `kernel_shape_tool` rather than git-applying SGLang roofline
  patches. **Upgrade note:** move off `v0.5.18-rocm724-*` images or bare-metal
  0.5.19 pins to `v0.5.20-rocm10-mi30x|mi35x-20260920` (or match `SGLANG_REF`
  on bare metal).

- **Repair kernel-tuning promotion and analysis handoffs.** GEMM integration
  and fusion siblings now default to a 1% E2E KEEP threshold instead of 3%,
  preserving explicit fusion overrides. MoE tuning collects and persists valid
  candidate tables and reports missing artifacts instead of exporting unusable
  paths. TraceLens uses the coordinator's Python interpreter and distinguishes
  dependency or splitter failures from successful splits with no steady-state
  output. **Upgrade note:** pre-install TraceLens and its dependencies in that
  interpreter; analysis no longer runs `pip install -e .` on each request.

- **Simplify optimizer lifecycle and benchmark limits.** Remove the Robustness
  agent/runtime RCA, runtime `recover` action, Monitor/Supervisor automatic
  supervision and resume, and task/lease age expiry. Each actual benchmark spawn
  uses a 7800-second hard deadline (including boot and accuracy), plus a
  600-second output-silence limit only after a ready marker is observed in this
  round's logs; both are finite positive settings. A warm-reuse hint alone does
  not arm silence, avoiding false kills when original Magpie buffers client
  output and the reused server writes its previous round's log. Reuse rounds
  without a current ready marker remain bounded by the hard deadline, session
  budget, and cancellation. Output cannot extend the hard deadline. Session
  cancellation and admission/phase budgets remain, as do explicit `--resume-from`,
  offline `recover-session`, process cleanup, and historical SBDv6 readers.

- **Drop the redundant Magpie `benchmarker.py` atomic-write patch.** The default
  `MAGPIE_REF` already copies benchmark scripts atomically upstream; install-time
  patching is limited to SGLang trust, eval-concurrency, and the client tokenizer
  hook. `MAGPIE_PATCH_STRICT` no longer exists (setting it had no effect after the
  atomic patcher was removed).

- **Scheduler stalls found in the ROCm 10 six-leg run are fixed.** KERNEL entry
  hooks no longer block the coordinator tick, conc-sweep boot rounds are no
  longer counted as measured variants, vLLM roofline reads the real rank trace,
  and a trace with zero GPU events stops repeat roofline scheduling with a
  recorded reason. Final reports keep GEAK and GEMM-tuning outcomes, and
  `kernelforge.gemm_tune run` exits non-zero on a hard AITER tune/serve
  alignment mismatch.

- **One rule picks every agent CLI.** Orchestration goes through the shared
  `preferred_agent_backend` ranking, so a host with both sides configured and
  only the Codex extra installed now orchestrates on Codex. Preflight and the
  kernel agent no longer mirror the credential into other variable names.

- **One kernel's KB records share one address.** `normalize_operator_name`
  splits camelCase, so `KdaPackedDecodeKernel` and `kda_packed_decode_kernel`
  meet, and framework versions reduce to their release (`0.24.0+rocm723` and
  `v0.24.0` are both `0.24.0`). Task publication refuses a backend outside
  `KERNEL_BACKENDS` instead of filing it as flydsl. Records stored under the old
  spellings are reached once the KB Store migration has run.

- **The Explore phase is dissolved into one attempt ledger.** The configuration
  arm gives up after five benched rounds without a win rather than five
  variants, a grid that kept an earlier variant is no longer read as dry, and a
  resumed session extends `tested` / `rejected` / `name_index` instead of
  replacing them. Apply-conflict retries no longer end FRAMEWORK_AGENT.

- **The rewrite pipeline's caller owns its task shape.** A KEEP requires the
  task's `compile_command` / `correctness_command` only on the assembly backend;
  elsewhere the driver's suite decides. `--no-applyback` skips the framework
  apply-back stage and returns its reserve to the search.

- **Roofline on GPUs without a Hyperloom achievable-peak entry uses AMD vendor
  theoretical peaks.** MI355X keeps its per-op PerfModel breakdown on vendor
  peaks and MI308X moves to PerfModel; per-kernel bypass rows record the peak's
  source. MI300X and MI325X are unchanged.

- **A benchmark that served its whole protocol is kept despite a non-zero
  wrapper exit.** When `completed_requests` reaches `num_prompts`, the round
  succeeds with warning `nonzero_rc_after_complete_protocol:<rc>`; a round that
  served fewer requests still fails.

- **Profiling follows InferenceX's move of `benchmark_serving.py` under
  `infx/bench_serving/`.** Every AgentX roofline had been failing with
  `benchmark_serving_ok=False` on current checkouts.

- **SGLang patches under `sglang/kernels/jit/` no longer trigger an editable
  reinstall.** They rebuild as `content_addressed_jit`, which kept the candidate
  server importable and the framework version equal to the baseline's; a
  framework reinstall is followed by an import check that reverts the patch on
  failure.

- **An agentic AgentX recipe satisfies the server-phase pid contract.**
  `aiperf_client.sh` accepts a recipe that already ran the lifecycle and wrote
  `inferencex_result.json`, and flat metric keys sit beside the nested schema so
  a healthy run no longer reads as 0.00 req/s.

- **The kernel installer preserves operator-supplied GEAK checkouts.** Only the
  installer-managed `GEAK@*` caches are re-cloned; a missing or invalid operator
  root fails clearly instead of being overwritten.

### Added

- **An enablement session now ships an ordered replay recipe, and a verdict on
  whether it can be replayed at all.** The session's durable state said what was
  kept and nothing about how to reproduce it: an operator holding a KEEP had a
  list of patches, no order to apply them in, no record of which tree each was
  written against, and no way to tell a complete stack from one whose evidence
  was never captured.

  `recipe_steps` projects the state onto an ordered array — setup, build, patch —
  each step naming the root it applies to, the targets its own diff declares and
  the identity of what each install consumed. `replay_sufficiency` judges that
  array and reports `sufficient` or `insufficient` against a closed vocabulary of
  reasons, each naming what it blocks: replay, assertion validation, or both. A
  patch whose targets nobody recorded, a build nothing replayed, a root with no
  base commit and a credential the recipe cannot supply are all refusals, not
  assumptions. Both appear in `session_breakdown.json` under `enablement.recipe`,
  and the session package now carries the KEEP's source overlay the steps
  reference, so a `sufficient` recipe does not ship with its own evidence
  missing.

  The KEEP records what the two need: each root's identity and base commit taken
  before the round's first mutation, byte-exact snapshots of every declared
  target, the environment closure and installed versions read through the
  interpreter the accepted bench launched, and an append-only ledger of the setup
  commands as they ran. Credentials are classified and sanitised on emission.

- **`ray.init`'s connect is bounded.** It had no timeout at all, so an
  unreachable head node hung the leg until the session clock ran out instead of
  failing it.

### Fixed

- **KernelForge results could be lost or integrated in the wrong order.** KB
  warm-start commits now force only their approved pathspecs, so tracked files
  matched by a repository's ignore rules cannot reject the whole commit. Task
  preparation stages tracked edits and only newly created files, preventing
  pre-existing JIT caches and generated artifacts from leaking into exported
  patches. Forge-loop stdout and stderr are retained beside each task result,
  and Controller patches are integrated by `task.json.priority` rather than
  encoded directory-name order so cumulative E2E validation follows the
  opportunity analyst's ordering.

- **Inline MCP actions no longer block the coordinator's event loop.** The
  `run_action_now` context tool awaits the action's result without blocking the
  loop or occupying the thread pool needed by database operations. Action
  execution, timers and caller cancellation can proceed concurrently. Direct
  calls to the synchronous bridge on the coordinator loop now fail immediately
  without scheduling work. Existing inline wait limits and registered-action
  completion after a caller timeout remain unchanged.

- **A Slurm row declaring a workload shape was benchmarked at the defaults.**
  The optimizer resolves `tp` / `conc` / `ep` / `isl` / `osl` / `precision` as
  flag > persisted state > default and deliberately never reads them from the
  environment, but `_incontainer.sh.in` only exported them. A row declaring
  `tp=4` therefore materialised a `tp=1` baseline, where sglang's rank math
  divided by zero; a row declaring `isl=8192 osl=512` silently measured
  1024/1024, and `precision` came from the checkpoint sniffer rather than the
  row. The whole declared shape is now passed as flags in both the python and
  claude backends. The exports stay, because the framework recipes downstream
  do read them -- that split is now stated where the command is spelled out,
  since the old wording pointed the carrier at the ignored mechanism. `ep_size`
  also gains a producer: it is a new trailing `models.tsv` column, parsed and
  exported by `run_hyperloom.sbatch` and added to the docker backend's `-e`
  allowlist so enroot and docker agree on the shape. It trails `target_gain` so
  existing 13-column rows still parse, and an absent value keeps expert
  parallelism at 1.

- **A cancelled job's container kept its GPUs and broke the next job on that
  node.** `docker run --rm` cleans up when its client exits normally, but a
  `scancel` or NODE_FAIL kills the client and leaves the container running, so
  `--rm` never fires and the weights stay resident. The next job then failed in
  whichever way its framework noticed first: sglang sat in
  `wait_for_amd_gpu_clean` for the full fifteen minutes because that gate maxes
  VRAM% over every GPU on the box, while vLLM was refused outright with
  `Free memory on device cuda:N ... is less than desired GPU memory
  utilization`. Observed on two nodes at once, where a container from a job
  cancelled three hours earlier still held ~168 GiB on each of GPU 0-3 --
  exactly where the next tp=4 server wanted to land, because the container gets
  no ROCR mask and every framework starts from GPU 0. The launcher now reclaims
  those containers first, deciding ownership by liveness rather than by name or
  image: `docker run` carries `-e CLAW_SESSION_ID=`, so a container whose
  session id has no live client is ours and orphaned, and a co-tenant job that
  still has its client is left alone. `docker run` also gains `--init`, without
  which the container's pid 1 becomes an unreapable zombie once the client is
  killed and even the daemon refuses to remove it (`PID <n> is zombie and can
  not be killed. Use the --init option ...`); the reclaim keeps a cgroup-level
  SIGKILL fallback for containers already in that state. `--name
  hl-<key>-<jobid>` makes a running container traceable back to its job. The
  liveness check walks `/proc` and matches `CLAW_SESSION_ID=<sid>` literally
  rather than through a truncating, regex-reading `pgrep -f`, which had
  reclaimed a live neighbour's container, and a container younger than
  `HL_ORPHAN_MIN_AGE_S` (default `900`) is never reclaimed.

- **A vLLM profile round had its profiler bounds dropped before launch.** The
  argv preflight probe sees only `EXTRA_VLLM_ARGS`, while the launcher appends
  `--profiler-config.profiler torch` and a trace directory of its own
  afterwards. `ProfilerConfig` refuses the iteration bounds this layer injects
  unless both are present in the same fragment, so the probe rejected an argv
  that is valid once the launcher's flags are appended, and the round then ran
  with no bound on the capture window. Both flags are now asserted alongside
  the bounds so the probed fragment is self-consistent on its own. The trace
  directory is a placeholder: the launcher's own value has to win vLLM's
  last-wins dotted-flag merge, so the bypass backend now emits its profiler
  flags after `EXTRA_VLLM_ARGS` the way Magpie's launcher already does, rather
  than before it where the placeholder would have won and sent the trace
  somewhere trace discovery never looks. An operator-set profiler flag is left
  untouched.
- **A campaign the host killed cost the next task in the same repository.**
  Every in-place task is handed the same `forge_experiments` directory, and the
  release archives it -- but a run that was killed never reaches the release.
  The leftover then did two things: forge-loop refused the workspace outright
  ("already contains a Forge campaign; pass --resume to continue it"), failing
  the next task at dispatch, and recovery read the stale manifest as that
  task's own best result, reporting the dead campaign's commit as a missing
  base commit -- a reason with nothing to do with the task it lost. Measured: a
  session lost a `chunk_gated_delta_rule` task to a fusion campaign left behind
  at a timeout an hour earlier. A leftover is now archived on the way in as
  well as on the way out, kept rather than deleted because it is the only
  account of what that run did, and a trusted manifest has to name a commit the
  repository still has before it is read as this task's result. An archive that
  cannot be made says so, since the dispatch refusal that follows is otherwise
  undiagnosable.
- **A campaign killed mid-search lost a fusion it had already published.**
  `run_campaign` publishes each winning iteration to the shadow repo's
  `forge_experiments/best/` and points `forge_loop_<stem>.json` at it, but the
  exported patch and the aggregate manifest only land once the campaign
  returns. A wrapper timeout while it was still iterating therefore reported
  REVERT with `patch: null` even though correctness had passed. Measured: a
  session lost a 5.011x fusion of qkvgate split + QK norm + RoPE with its
  experiment still running when the 5400s timeout fired. Salvage now falls back
  to the per-campaign artifacts, and each salvaged row carries the env flag its
  fused path is gated behind -- read back from the driver the campaign wrote,
  since the patch does not carry it and without it the re-baseline server boots
  un-gated, measures the eager path and rejects the win one stage later. A
  campaign whose flag cannot be read is not salvaged at all, rather than queued
  to fail that way. Artifacts a previous run left in the same output directory
  are swept before the run starts, so they cannot be salvaged as its own.
- **A KB recipe carrying code overlays could not be replayed into a framework
  installed from a wheel.** A Recipe whose patch timeline is non-empty replays
  as required, and that path refused any tree without a git HEAD -- which a
  pip-installed framework never has. Every code-level optimization a session
  published therefore became unreplayable the moment it reached the KB: warm
  replay failed before booting a server, and the recorded gain could only be
  re-earned from scratch. What promotion needs is a way to unwind the tree if
  the replay is rejected, not a sha, so it now accepts either channel: a git
  checkout's snapshot, or the backups a nogit apply records as it writes. An
  overlay the tree already carried applies as a no-op and records neither,
  which is not the same as an apply whose artifacts were lost, so each tree now
  states whether the round wrote to it -- a no-op tree promotes and is skipped
  by the rollback, a written-to tree still has to answer with a channel, and a
  record persisted ahead of the apply reads as written-to so a resume restores
  it. Framework-agnostic; the shape of the checkout decides the channel.
- **A shipped aiter tuned CSV naming a kernel this host never compiled failed
  every boot of the session.** aiter resolves its tuned tables two ways: a
  pinned `AITER_CONFIG_*` env is taken as the exact `:`-joined list, and an
  unset one falls back to the shipped default plus every `model_configs`
  overlay whose name matches. A round that tunes one operator sets only that
  operator's variable, so the others take the unset branch and pull in overlays
  cut on another host -- tables whose `kernelName`s are absent from this
  machine's compiled `module_*.so`. Serving then aborted at load with a
  registry mismatch, and because the table is shipped rather than produced by
  the run, every retry hit the same wall: the whole session failed at boot with
  nothing to roll back. The CSV set a boot will actually load is now resolved
  by aiter's own two-branch rule and checked against the compiled modules
  before the config is materialized; an uncovered module is unlinked so the
  next boot rebuilds it. On integrate, a registry mismatch also drops the
  modules the error names, not only the ones the round's environment mapped.
  Framework-agnostic; it is the aiter install that is repaired, not the server.
- **An author whose transport died took the whole fusion lane with it.** A lane
  costs hours and an authoring call costs minutes, but a provider that never
  delivered an answer -- a stream stalled mid-response, a connection reset --
  ended the lane on the first failure. The classification that tells "the model
  never answered" apart from "the model answered nothing" already exists in
  `llm_failure`, so authoring now consults it where the exception is still in
  hand and retries only the transport, with backoff and against a deadline. A
  provider safety stop is still never retried, because retrying one is the
  anti-pattern the session-resume allowlist already refuses; neither is a
  timeout, which has just spent a full attempt's budget. The deadline defaults
  to what the configured attempts can legitimately cost, since the generic
  1800s LLM default is shorter than a single 7200s authoring attempt and would
  have made the retry unreachable.
- **A fusion campaign killed before it returned reported REVERT while proven
  work sat on disk.** `fusion_manifest.json` is the only artifact that points
  at a keeper, and it was written once every campaign had returned. `on_keep`
  exports the patch and smokes it the moment a recipe is kept, so a wrapper
  killed between the last keeper and the aggregate reported `patch: null` even
  though a sibling had already passed correctness and a serving smoke.
  Measured: a session lost a 5.011x fusion of qkvgate split + QK norm + RoPE
  this way, the iteration having published 44 minutes before the timeout fired.
  The manifest is now published as each keeper is proved, so it is never
  missing -- only as complete as the run got -- and the existing salvage path
  reads it unchanged. The end-of-run write still overwrites it with the final
  loop, compile-pass and error fields before any exit, and a sibling the smoke
  rejected has its patch unlinked so no reader can find work the run refused.
- **AgentX baselines retain request-quality grading with `RUN_EVAL=false`.**
  Missing AIPerf error-rate metrics are derived from profiling request counts;
  zero errors are inferred only with successful requests and an explicitly empty
  error summary. Warmup accounting is excluded, and invalid or unknown evidence
  remains fail-closed. Valid zero-error baselines no longer lose their quality
  signal merely because serving lm-eval is disabled.
- **Honor concurrency-sweep budgets without treating the hard cap as a start cost.**
  Admission uses the measured expected duration when available; unknown-duration
  work may start while budget remains. Boot retries, reuse and fallback share
  the earlier sweep/session deadline, and reports retain the actual stop source.
  The manual sweep driver no longer passes or advertises the retired
  `--variant-timeout-sec` option.
- **Complete cooperative build and specialist cancellation without accepting
  unconfirmed cleanup.** Cancellation reaches pending work and running workers;
  confirmed cleanup records a cancelled outcome, while unknown cleanup retains
  ownership. Completed outcomes remain in task history for diagnosis even when
  cleanup fails, without publishing them for promotion or retry. Completion
  callback failures after confirmed cleanup no longer retain execution entries,
  and repeated inline calls return stored terminal results instead of rerunning.
- **Refuse unsafe session resumes explicitly.** Legacy or foreign execution
  ownership and unproven historical cancellations now produce bounded task/lane
  diagnostics before resume writes or dispatch. Resume admission itself clears
  no ownership. After independently verifying that a task's complete process tree,
  remote workers and Ray actor have stopped, operators can use
  `recover-session --confirm-stopped TASK_ID --confirmation-reason TEXT` to record
  that confirmation, cancel an unfinished task, and release only its unattributed
  execution/GPU records under the POSIX session lock. This explicit operation preserves rounds
  and other tasks, rejects nonempty owner scopes, and neither stops workers nor
  accepts old results or starts a resume. Existing `--force` remains report-only.
  Resume admission and
  round reconciliation honor the latest recorded cleanup outcome, including
  results recorded after an earlier terminal transition. A Ray worker whose root
  exited is not treated as proof that detached descendants exited, and a missing
  stop acknowledgment no longer destroys the specialist actor's cleanup channel.
  Specialist cleanup makes one bounded follow-up confirmation on the same lease
  before retaining unconfirmed ownership; a late acknowledgment is consumed by
  the existing completion path rather than requiring a new background reaper.
- **Restore safe AITER lock cleanup at baseline startup.** Stale locks are cleaned
  only when compiler absence is established. Unreadable live-process identity
  leaves locks untouched; known zombies do not block cleanup.
- **The AITER version in `stack_fingerprint` is now the AITER that is actually
  installed.** Two independent faults made that field untrustworthy.

  The env tuple read `AITER_COMMIT` and `AITER_VERSION`, neither of which
  anything in this repo writes, so the env path never produced a value.
  `install_baremetal.sh` already resolves the exact tag it installs, exports it,
  and persists it to `.env` as `AITER_REF`, which the dotenv loader admits under
  its `AITER_` prefix — so the value was sitting one key away the whole time.
  `AITER_REF` joins the tuple, behind `AITER_COMMIT`. It also covers the default
  isolated vLLM path, where aiter lives in the framework venv and no in-process
  probe can see it under any name.

  The probe then looked up a distribution named `aiter`, but AITER renamed itself
  to `amd-aiter` at v0.1.8, so the lookup missed every host running v0.1.8 or
  newer. Worse, on PyPI `aiter` is an unrelated 2019 async-iterator library, so
  where that package happened to be installed the probe recorded its version —
  `0.13.20191203` — as the AITER version. The old name is corrected rather than
  kept as a fallback, precisely so that value can no longer be produced: nothing
  recorded is better than something that looks like an answer. Hosts older than
  v0.1.8 are covered by `AITER_REF`, which is exact.

  Only what gets *written* changes. No read path compares `rocm` or `aiter`
  against the pod today; that gap is tracked in #1507, and getting the recorded
  value right is a prerequisite for it — a comparison fed `0.13.20191203` would
  report a confident mismatch against every real AITER build.

- **AgentX grading failures no longer fall back to throughput KEEP.** When an
  AgentX session cannot grade on interactivity because either side is missing
  the axis pair, explore, ``_lift_to_current_best``, and
  ``resolve_graded_comparison`` fail closed instead of promoting on the
  diagnostic output figure. The removed ``ANCHOR_DEGRADED`` round-local output
  fallback is part of the same rule. ``degrade_reason`` still travels on the
  explore, stack-validation and integrate rows that record a refused
  comparison, so the breakdown can still name why a variant did not KEEP; an
  *adoption* row can no longer carry one, because the resolver now returns
  REVERT on a degraded pair and a REVERT never reaches the adoption writeback.

- **A partitioned card is now a different machine in the KB key, so a warm-start
  hit can no longer replay a config tuned on a differently shaped one.** The
  `canonical_id` is a seven-tuple of model, hardware, framework name, model type,
  architectures, framework version and precision. The compute-partition mode was
  not among those dimensions, and neither was expert parallelism on a single
  node, so `kb_hardware_slug` collapsed to the bare GPU type and a run in SPX and
  a run in CPX landed on one identity — `inference:qwen3-32b:mi355x:...` either
  way. The warm-start cascade only relaxes `conc`/`isl`/`osl`, so nothing
  downstream caught it either: an `exact` tier hit at confidence 1.0 could hand
  the auto-replay a config recorded with eight times the partitions, and the
  `--warm-replay-min-reproduce-pct` backstop only noticed after spending the
  verify round.

  `kb_hardware_slug` now suffixes the partition mode and `ep` at any node count,
  not just on a cluster: both are fixed at launch rather than explored, which is
  the argument `_tp{tp}` already makes for itself. A CPX pod therefore cannot
  read an SPX row because it is asking a different `canonical_id` — no flag, no
  demotion, and no second comparison that could be applied to a different row
  than the one that gets replayed, since `resolve_kb_topology` is the single call
  both the reader and the writer build the key from. Every suffix is omitted at
  its default value (`ep <= 1`, SPX, or a mode nobody published, including one
  this build does not recognise), so existing keys stay byte-identical and
  nothing in the corpus moves. `_TOPOLOGY_SUFFIX_RE` learned the single-node
  forms too, so `_hardware_fallback_values` still offers the same-ISA SKUs for
  exactly the rows these suffixes were added for.

  `workload_shape` still publishes `ep` and `partitions` as a description of the
  run, and `knowledge_to_warm_recipe` derives its projection allowlist from the
  publisher rather than restating it, which is what had been silently dropping
  keys the publisher emitted. Both are omitted at their default: `--ep` defaults
  to 1, so publishing it would have every dense run claim a formation it never
  chose, and one partition is the whole card. The count a launch published wins
  over one re-derived from the mode name, so there is only ever one derivation to
  keep in agreement.

### Removed

- **`--recipe-kb-strict-fingerprint`.** It was declared in the parser and read
  nowhere, and it promised to refuse rows whose `stack_fingerprint` disagreed
  with the pod — which was never the exposure, since framework version and
  precision are already identity dimensions. Encoding the partition mode in the
  key makes the mismatch it would have caught unrepresentable, so a read-side
  comparison has nothing left to do. `rocm_version` and `aiter_commit` are still
  written into every row's `stack_fingerprint` and compared nowhere at read time;
  that is a real gap and is tracked in #1507 rather than under a flag whose name
  says fingerprint and whose behaviour would have been workload shape.

### Changed

- **breaking: the SBD V6 concurrency-sweep comparison rows are named for the
  axis they carry rather than for throughput.** `baseline_throughput` →
  `baseline_value` and `optimized_throughput` → `optimized_value` in each
  `conc_sweep` event's `comparison` rows. The old names were a lie on an AgentX
  session: the figure they held is whatever `result.metric` names, which is the
  slow-tail interactivity percentile (`e2e_norm_intvty_p90`, milliseconds)
  whenever the session grades on one, so a consumer reading `*_throughput` was
  plotting latencies as throughputs against rungs measured in tokens/s. No
  alias is kept — an alias would preserve exactly the misreading the rename
  exists to stop. External parsers of `reports/sbd_v6/` must be updated.

  The rows also gained `baseline_guard`, `optimized_guard` and `guard_holds`,
  and the roll-up gained `guard_axis` and `best_conc_guard_holds`: the throughput
  the session would have held a promotion to, reported beside the ranked axis
  rather than enforced, so a rung that bought interactivity by giving up
  throughput is visible as such instead of reading as a clean win. All are null
  off the interactivity objective. `guard_holds` is tri-state — `null` means the
  framework never answered, which is not the same fact as `false`.

- **SBD V6 publishes the axis a session graded on, at the session level and on
  each settled figure.** Three additions to the wire shape, all of them facts no
  consumer could previously recover:

  - `metadata.grading` — `{benchmark_mode, objective, tput_guard: {enabled,
    noise_pct}}`. An AgentX replay is ranked on `e2e_norm_intvty_p90` with total
    throughput held as a guard; a synthetic run is ranked on output throughput
    alone. Every throughput field elsewhere in the document is the output axis by
    construction and `benchmark_mode` never reached the breakdown at all, so
    without this block the two kinds of session are indistinguishable — and on
    the canonical corpus the two axes differ by roughly two orders of magnitude,
    which is enough for a consumer to sort one against the other and never
    notice. Recorded from `SharedState.grading` rather than re-derived at export:
    the axis is settled once at seed, where the run can still see its own
    configuration, and re-deriving it here would read the exporting subprocess's
    environment. `tput_guard.noise_pct` is `null` on a session seeded before the
    band was recorded — the band that session applied is unknown, and today's
    default is not evidence of it.
  - `outcome.baseline.perf` and `outcome.final.perf` — the four graded axes the
    measurement reported (`e2e_norm_intvty_p90`, `total_throughput`,
    `input_throughput`, `tpot_p90_ms`), each an explicit `null` where nothing
    measured it. All four keys are always present: absent would be
    indistinguishable from an axis the framework failed to report, and zero reads
    as "measured, and it was zero". A synthetic run publishes four nulls.
  - `outcome.final.graded_on` and `outcome.validation.graded_on` — the axis the
    gain beside them is on. The stack reconciliation has to be single-axis, since
    an attributed figure on one axis against an unattributed figure on another
    makes the gap meaningless, and `graded_on` is what names it.
    `outcome.validation.perf` carries the settled measurement's own axes on the
    same row as the gain they produced, because a revalidation moves the
    cumulative figure without re-promoting the recipe.

- **One definition of the agent default model ids, and the backend picks its own
  last rung.** `DEFAULT_CLAUDE_MODEL` and `DEFAULT_CODEX_MODEL` lived in
  `orchestrator/roles/agent_role.py` and again in
  `kernelforge/agent_backends/{claude,codex}.py`, and the two values were also
  spelled as bare literals in the Forge provider registry beside the module that
  defined them, in the Claude allowlist head, in the GEAK model default, in the
  TraceLens Claude path and in the Critic's Codex field. Eight copies of two
  strings, each free to drift. They now live in `hyperloom/common/llm_config.py`
  next to `AGENT_BACKEND_CLAUDE` / `AGENT_BACKEND_CODEX`, which is the pair they
  are keyed by, following the `DEFAULT_REASONING_EFFORT` precedent that both
  packages already import from `common`.

  `resolve_forge_llm_model` lost its `default` parameter. Every caller passed the
  chosen backend's own default, and `request_handlers` reimplemented the
  per-backend branch the function already performs to work out what to pass;
  `patch_conflict_merge` carried a comment at each call site explaining that a
  default is mandatory because `CLAUDE_MODEL` is unset on OAuth-token runs and
  the resolver would otherwise post an empty model id. The function knows the
  backend, so it now answers that itself and the parameter that could be
  forgotten is gone.

  The test that asserted the allowlist head equals `DEFAULT_CLAUDE_MODEL` is
  gone with it: the head is now that constant by construction, so the drift it
  watched for is unrepresentable. Model knobs that merely share a value today
  are deliberately untouched — the narrative report model, the RCA model, the KB
  synthesis model and the quantization driver model each have their own
  override and their own reason to move, and collapsing them onto one constant
  would couple decisions that should stay free to diverge.

- **Roofline CUDA graph capture failures are classified instead of retried in
  eager mode.** When profiling cannot capture a graph, the executor records a
  structured failure category and writes a diagnosis artifact rather than
  rebooting the server in eager mode and retrying. Timeline rows now publish
  ``graph_capture_disabled`` instead of ``eager_fallback_applied``. Blocking
  filesystem and liveness probes run in ``asyncio.to_thread`` so the roofline
  action no longer stalls the coordinator event loop.

- **ENABLEMENT is the sixth phase of the optimization loop.** Bring-up used to
  run inside FRAMEWORK_AGENT, which left it a lane with no lifecycle of its
  own: it could not be entered, exited or reported on, and a phase that owned
  optimisation work was also carrying the work of making the combo run at all.
  It is now a phase of its own between PRELUDE and FRAMEWORK_AGENT, with entry
  and exit predicates in `compute_next_phase`, its own `phase_history` rows and
  a section in the Markdown session report. `PHASE_NAMES` is six long.

  **No wall-clock budget is apportioned to it.** `DEFAULT_PHASE_BUDGET_PCT` has
  no ENABLEMENT key, and an absent key means no cap rather than a zero one: a
  budget apportions optimisation effort, and a combo that cannot run has
  nothing to optimise yet. `PHASE_FRAMEWORK_AGENT` drops 0.40 → 0.38 and
  `PHASE_KERNEL_AGENT` 0.50 → 0.47, so the table now sums to 0.95 and bring-up
  is bounded by the run's wall clock and by `ENABLEMENT_MAX_ATTEMPTS` instead.
  The phase's terminal exit is `enablement_attempts_exhausted`, which
  `enablement/lane.py` sets.

  **Runnability is decided from the measurement, not from a log scan.** A combo
  counts as served once it has produced positive throughput and completed
  requests, which is a signal the baseline already carries; the `booted`
  property it replaces scanned the server log for bring-up milestones and could
  not witness one past the head of a chatty log. Both enablement origins — a
  boot failure and the accuracy gate — now open the same baseline revalidation
  window, and the baseline is Coordinator-owned while the phase is ENABLEMENT.

### Fixed

- **A KernelForge Controller KEEP now reaches the stack ledger in every
  benchmark mode, not only under AgentX.** `_run_kernel_rewrite_controller`
  handed `integrate_controller_patches` the session-owned writeback only when
  `agentx_active` held; every other mode fell back to a module-local recorder
  that wrote `optimization_stack`, `current_best` and
  `cumulative_gain_validated` directly. That writer bypassed
  `_lift_to_current_best`, which is the only caller of
  `stack_event.record_adoption`, so a controller KEEP in synthetic mode landed
  in `state.json` and never produced an adoption row — `session_breakdown` sat
  behind the state it was meant to describe.

  This is visible in the document: for a non-AgentX run the kernel bucket and
  its `by_backend.forge` split go from empty to populated, and
  `validated_total_gain_pct` / `at_head` now account for controller KEEPs. The
  KEEP result carries `backend: forge` / `engine: kernel_rewrite_controller`, so
  the adoption is attributed rather than folded into `unattributed`. The local
  recorder and the `record_keep is None` branch that selected it are gone and
  `record_keep` is now required, so a caller cannot silently reacquire the
  no-ledger path.

- **A measured Controller patch was dropped because the patch kept before it
  had moved its context.** Lanes run in parallel from one pinned base commit,
  so two lanes touching the same file each ship a diff written against that
  same base; integration applies them one at a time and commits every KEEP,
  which leaves the later diff stale by the time its turn comes. Measured in the
  Kimi-K3 session of 2026-09-13: `flydsl_moe_stage2` (1.1727x micro) was lost
  to `error: patch failed: aiter/ops/flydsl/moe_kernels.py:14` once
  `flydsl_moe_stage1` had landed, although the two touched disjoint functions
  and defined disjoint module-level symbols -- they collided only because each
  inserted its own sweep helpers at the same anchor. A refused diff is now
  rebuilt in escalating steps: `git apply -3` for pure line drift, then keeping
  both sides of every conflict region whose merge base is empty, then an LLM
  for the regions where the lanes genuinely edited the same lines -- one region
  at a time, carrying the surrounding source as context rather than the file to
  rewrite. Whatever the last two steps reconstruct is discarded unless it still
  contains every line the incoming patch and every landed KEEP added, parses,
  carries no conflict marker and redefines no module-level name; a lane that
  fails any of those is dropped exactly as it was before, with the worktree put
  back to HEAD. Nothing here decides a KEEP: the E2E gate downstream still
  measures and still reverts, so a bad merge costs what a dropped patch already
  cost and can never produce an unmeasured KEEP. A lane that landed as a merge
  rather than verbatim is reported as `merge_strategy` on the integration
  result and in `summary.json`. A patch that applies cleanly takes the path it
  always took, and the resolver runs on whichever backend
  `preferred_agent_backend` picks for this deployment -- Claude through the
  single-shot Anthropic transport, Codex through `achat_completion` -- and is
  skipped entirely when neither side is credentialed.

- **An accuracy eval that failed because the server was gone was read as a
  missing framework capability.** `run_eval` reports a vanished server and a
  model that scored badly the same way -- a non-zero exit -- so the eval-rooted
  branch stamped both as an eval-failure contract and handed them to the
  enablement lane. Measured: a baseline whose throughput pass had already
  completed lost its server mid-eval, the client's next request was refused,
  and the run then spent five specialist rounds hunting a capability gap the
  evidence never supported before stopping on the stall cap with a terminal
  reason that named enablement rather than the server. A refused connection is
  now separated out: the baseline still fails, nothing is salvaged and the
  accuracy gate is untouched, but it counts as an ordinary baseline failure so
  the existing total-failure backstop ends the run on the cause it actually
  had. Framework-agnostic; the eval path is shared by vLLM, SGLang and ATOM.

- **`--framework atom` defaults the kernel backend to forge.** The kernel phase
  runs on atom, but GEAK -- the backend every framework gets unless the
  environment opts into forge -- is on weaker ground there: its extraction rules
  forbid guessing a rewrite seam on a quantized, non-vLLM backend and require
  resolving one from the live server, which is unproven on atom. The default phase split gives that phase half the session, so
  defaulting to GEAK on atom meant defaulting to half a session of nothing,
  while the CLI printed that kernel-agent was "wired for atom". On atom an unset
  `KERNEL_OPT_BACKEND_ORDER` is now filled in with `forge` before the session
  records its backend, and the choice is reported at launch. A value the
  operator set is kept, so running GEAK on atom deliberately stays possible; the
  CLI warns that its seam resolution is unproven there. `--no-kernel` skips the
  defaulting entirely. Only the `framework == "atom"` branch is touched; SGLang
  and vLLM keep GEAK.

- **Recognize recorded ATOM servers during lifecycle teardown and recovery.**
  The serving-process checks now accept `atom.entrypoints`. Recovery records
  the members of a recognized ATOM process group before sending TERM, then
  checks each recorded PID, group and start time before sending KILL. This
  allows anonymous workers to be reaped after their leader exits without
  treating a reused PID or a newly discovered process as an owned worker.
  A rank that forked after the snapshot is in neither the recorded set nor any
  cmdline that still reads as an owner, so a confirmed group also receives a
  closing group KILL; measured on an 8-rank bring-up, those ranks otherwise
  survived holding their cards. That kill reaches members this pass never
  enumerated, so it is not treated as proof the group exited. Ownership must be
  confirmed first: when the recorded leader is absent from the group, no longer
  reads as an ATOM server, or no longer matches the group and start time
  recorded for it, nothing is signalled at all and the pidfile is kept for a
  later pass -- a recorded pgid the kernel has since recycled would otherwise
  take the teardown meant for ours.
  Recovery retains the pidfile while the group is still alive and reports the
  worker PIDs actually signalled. Measured on one MI355X serving
  Qwen3-14B-FP8: against a fully booted server recovery reaped the leader and
  three anonymous workers and the card went from 87% to 0% VRAM; fired mid-boot
  it left no engine process behind. Normal warmup/measure reuse and the existing
  vLLM/SGLang recovery paths are unchanged. This does not recover ownership of
  anonymous workers whose leader had already exited before recovery began;
  the generic subprocess teardown and third-party benchmark scripts are unchanged.

- **A bare `--resume-from` rebuilt the budget and the stop target from the
  flags.** `--max-hours` carried an argparse default, so a resume passing no
  flags at all was indistinguishable from one passing the default: a 24 h
  session was shortened to 2 h and closed as `time_exhausted` before its first
  action, and the objective was dropped on the way. The flag now
  defaults to `None` and resolves to `DEFAULT_MAX_HOURS` only after the archive
  has had its chance at the persisted budget, so an absent flag restores 8 h
  while an explicit `--max-hours 2` wins over it.

- **Argv preflight read every dotted vLLM flag as unrecognised.** It probed
  through `parse_known_args`, which is not what vLLM's parser uses to expand
  `--<group>-config.<field>`, so preflight spent its one repair dropping flags
  the server would have accepted. One of them bounded the profiler, and the
  roofline that followed recorded 25.7 GB of trace over the whole workload
  instead of over a steady-state window. The probe goes through the entry point
  that performs the expansion.

- **The IR-1 stale-process scan failed on an idle machine, and could not see an
  ATOM server.** It excluded only `os.getpid()`, so the launcher shell — whose
  argv quotes the whole command — matched the scan's own patterns and failed the
  gate; it now excludes its ancestry. Separately, the pattern list named only
  vLLM and SGLang, so a leftover ATOM server still holding every rank's VRAM
  read as a clean machine. Matched on `atom.entrypoints` alone: the per-rank
  workers are `multiprocessing.spawn` children carrying no identifying argv, so
  only descent from the wrapper reaches them, which teardown already covers.

### Removed

- **The orchestrator drops five mechanisms nothing read: the `kernel_agent`
  inbox subscription with `IntentType.RESPONSE`, `Message.priority`,
  `IntentSpec.builder`, the `PHASE_EXIT_REASONS` vocabulary, and 11 stop
  reasons no code path can produce.** Each had writers, or a schema column, or
  a test suite holding it up — everything except a production reader.

  `IntentType.RESPONSE` was dispatched, policy-gated and validated end to end
  while the `kernel_agent` process that would consume it is never instantiated.
  The `request` / `response` topics and `kernel_agent` as a routing target are
  untouched; only the subscription and the intent type go. `Message.priority`
  had 15 writers and no reader, no index and no `ORDER BY`.
  `IntentSpec.builder` kept two unwired builders looking alive — their
  validators stay, because the role may still emit both intent types, just not
  through a builder. `PHASE_EXIT_REASONS` was a closed 37-member vocabulary
  with no production reader at all, unlike its load-bearing twin
  `STOP_REASON_VOCAB`, which gates `machine.py`, `close.py` and
  `set_stop_reason`: a phase exit reason only reaches `phase_history` for a
  human to read, so a typo there cannot misroute anything.

  The 11 stop reasons were not "not yet triggered". The crash-threshold path
  sets `emergency`, and `compute_plateau_explore` / `compute_plateau_kernel`
  return booleans, so no exit rule ever named `plateau_explore` or
  `plateau_kernel`. The comment calling these legacy sentinels kept for
  resuming old sessions did not survive checking, and goes with them:
  `SharedState.from_dict` restores `stop_reason` as a dataclass field,
  bypassing `set_stop_reason`, and `_global_terminal` returns unrecognised
  values verbatim under `vocab: "unknown"` — vocabulary membership was never
  what let an old session report its terminal. `enablement_stalled` remains a
  legal value in the enablement timeline event namespace, which is a different
  vocabulary and is untouched.

  **Resuming across the change needs nothing from the operator.**
  `CREATE TABLE IF NOT EXISTS` leaves `priority INTEGER NOT NULL` on a
  `coordinator.db` written before this release, where a column with no default
  would refuse every append the new code writes, so `ensure_schema` drops it on
  the way in. An in-flight session resumes with its event history intact.

  The rewrite kernel lane is a **refactor, not a removal**. `allocate()` built
  a `LaneAllocation` for a lane whose budget the rewrite route never read — it
  sizes itself against the wall clock — but the `0.5` was load-bearing as a
  divisor, holding the other two lanes down to 0.3 and 0.2 of the phase. It
  becomes `REWRITE_RESERVE_SHARE`, taken off the top before the two real lanes
  divide the rest, and each lane's share of the phase is unchanged.

- **`--max-minutes-enablement-pct` / `--phase-budget-enablement-pct`.** The
  flag parsed and reached `DEFAULT_PHASE_BUDGET_PCT`, but no ENABLEMENT branch
  in `compute_next_phase` ever calls `phase_cap_exceeded`, so the cap it
  advertised was never enforced against anything. Wiring it up would have
  contradicted the phase having no budget by design. It was new in this
  release and nothing depended on it.

## Hyperloom 1.1.1 release

The [1.1.1 release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.1.1)
is a patch release on top of 1.1.0. The session record does not move, so a
session recorded by 1.1.0 resumes on this build. The command line and the
environment contract do: one optimizer option, one console script and eight
environment variables that 1.1.0 accepted are gone, one variable's accepted
values narrow, and one option is added. See "Before upgrading from 1.1.0"
below.

Most of it is the Recipe knowledge base telling the truth: the prior work a
session had actually earned was not reaching the model at all. The other
corrections are in agent-backend selection, which three places answered
differently, and in the supervisor watchdog, whose restarts are now resumable
and bounded.

### Before upgrading from 1.1.0

One optimizer option 1.1.0 accepted is removed. The parser is strict, so it is
not an ignored token: a launch or resume command that still carries it exits
with `unrecognized arguments` before the session starts. Check operator scripts
before upgrading.

| Removed option | What to do |
|---|---|
| `--breakdown-include-transcripts` | Delete it. The Session Breakdown section it inlined into is gone, so there is nothing left to inline; specialist transcripts are still written to disk and carried as `transcript_path`. |

The deprecated KernelForge console-script alias is also gone, so a script
invoking it fails with `command not found` rather than a parser error. Call
`kernelforge`, which has been the name since v1.0.0b2; the retired spelling is
`kernel-agents`.

Eight environment variables 1.1.0 read are also gone, and these fail differently
from the options above: nothing refuses them. A box that still exports them
starts normally and behaves as though they were never set, so they have to be
found by reading launch scripts and `.env` rather than by watching a run fail.

| Removed variable | What to do |
|---|---|
| `FORGE_CLAUDE_MODEL`, `FORGE_CODEX_MODEL`, `FORGE_AGENT_MODEL` | Set `CLAUDE_MODEL` / `CODEX_MODEL` instead. Forge now walks Hyperloom's ladder and nothing above it, so one spelling configures both. A box left on the old names does not fail; it falls through to the provider default. |
| `HYPERLOOM_SKIP_COLLECTIVE`, `HYPERLOOM_COLLECTIVE_ONLY`, `HYPERLOOM_COLLECTIVE_KEEP_PCT`, `FORGE_COLLECTIVE_TIMEOUT`, `FORGE_COLLECTIVE_AGENT_TIMEOUT` | Delete them. They steered the collective optimization lane, which is now part of the rewrite controller; communication operators are picked up as ordinary rewrite candidates and need no separate switches. |

One more variable survives with a narrower accepted set rather than being
removed. `HYPERLOOM_REASONING_EFFORT` no longer takes `minimal` or `none`; the
ladder is `low | medium | high | xhigh | max`. The two sides that read it
disagree about a value outside that ladder, which is why it needs naming here:
Forge refuses at startup (`'minimal' is not a reasoning effort`), while
Hyperloom's own `chat.completions` drops the field and takes the gateway
default, deeper and more expensive than `minimal` was. A deployment sitting on
either value has to move to `low` by hand, and only one half of the run will
say so.

### 1.1.1 highlights

- **The warm-start block, the KB's lessons, and its pitfalls reach the model
  again.** All three read field shapes no writer produces. An exact hit carrying
  a full config printed `(no recipe text — first session for this workload/hw)`,
  and sections 5b and 5c rendered `(none)` no matter how much a prior session had
  learned. The block now renders `warm_start_context`, the view `recipe_kb_t0`
  already persists on every anchor, and a borrowed config is labelled with the
  model it came from. No recorded knowledge was lost; until now none of it was
  being shown.

- **One rule picks the agent backend across both packages: a configured
  credential first, then an installed SDK, with Claude ahead of Codex.** Four
  places answered this and three disagreed, so `forge-loop` on an OpenAI-only
  host could resolve to Claude and then fail to authenticate. The Robustness
  Agent's RCA engine follows the same precedence instead of checking the OpenAI
  side unconditionally. One consequence worth naming: a deployment whose runtime
  is logged in by other means is no longer refused up front — `forge-fuse`
  dropped its `--agent-backend auto` usage error and forge-fusion dropped the
  `llm_provider_unconfigured` result, leaving the real authentication failure to
  the preflight that can see it.

- **Supervisor watchdog restarts are resumable and bounded.** A wedged
  coordinator takes SIGHUP and keeps its interrupted phase segment rather than
  recording a session outcome; three restart attempts, counted durably before
  the signal goes out, make the wedge terminal. Separately, the robustness
  monitor now reads the real stop-reason vocabulary — it had been importing a
  module that does not exist and silently falling back to a subset missing 16
  terminal reasons, so a finished session could be relaunched.

- **`--extend-hours` grants a resumed session more budget.** Elapsed time is
  summed forward across every leg and never reset, so this is the only way to
  continue a run that has already spent its budget. The grant is applied on
  `--resume-from` and recorded in the session state with its reason; it defaults
  to `0.0`, so an invocation that does not pass it behaves as before.

- **The bare-metal `vllm` default is `0.29.0+rocm723`, up from `0.27.1`.**
  `install_baremetal.sh`, the compatibility matrix, the install guide, the
  example `SKILL.md` recipes, and `assets/slurm/models.tsv` all name it, and the
  pinned TraceLens ref ships the matching profiler-config patch.
  `VLLM_VERSION` and `VLLM_ROCM_VARIANT` override it as before.

- **Dead surfaces are removed**: the pre-rename KernelForge
  console script kept as an alias since v1.0.0b2, superseded by `kernelforge`
  (the retired spelling is `kernel-agents`); and its `learning/` tuning
  database with the tracker's superseded scoring layer, about 1.4k lines whose
  writes had already been disabled. The one observable difference: a
  `forge-loop` run no longer writes lesson markdown under the writable
  knowledge base's `learned/` directory and no longer prints
  `Lessons learned: N`. Nothing read that directory.

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
are marked below; the full per-change list is in the
[1.1.0 release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.1.0).

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
  typos and renames — seven shipped examples kept passing a flag that had been
  renamed out from under them and ran an inferred backend while exiting 0. Both
  commands now fail with click's own error and exit 2 before any GPU work starts.

  Alongside it: `$FORGE_PATH` is removed and **not** honoured as an override
  (use `$KERNELFORGE_PROJECT_ROOT`, which defaults to
  `$USER_DATA_PATH/kernelforge`, else `~/.cache/hyperloom/kernelforge`);
  `forge-gemm-tune` is gone as a console script and as a distribution, invoked
  now as `kernelforge gemm-tune`; the kernel-backend vocabulary is normalized, so
  the CLI flag is `--kernel-backend` taking a bare name (`triton`), the
  campaign-config key is `kernel_backend`, and a config carrying the retired key
  fails at load rather than migrating silently; and `FORGE_MAX_ITERS` /
  `FORGE_COMPILED_MAX_ITERS` are gone, having fed a cap that was never applied.
  Because `FORGE_` stays on the dotenv prefix allowlist, a stale `FORGE_PATH` or
  a retired spelling of `FORGE_DISABLE_COMPILED_KERNEL_BACKENDS` is still
  forwarded into the run and then ignored; the latter is detected and warned
  about once per run, because an operator who had switched compiled kernel
  backends off would otherwise silently get them back. The retired spelling a
  migrating script will be carrying is `FORGE_DISABLE_COMPILED_FELLOWS`.

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

- **Retired `kernel_optimization.py` CLI** *(historical — the wrapper is gone)*:
  Kernel work is dispatched by the Coordinator KERNEL handlers and, for Forge,
  `kernelforge forge-loop`. Direct invocation of `kernel_optimization.py` is no
  longer a supported entry point.

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
