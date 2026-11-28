---
myst:
  html_meta:
    "description": "Hyperloom release notes: headline capabilities and breaking changes for versions 1.2.0."
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

### Release notes

Merged to `main` and not yet carried by a tagged release. Each entry moves
into the [release](https://github.com/AMD-AGI/Hyperloom/releases) that ships
it.

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

### Release notes

The [1.2.0 release](https://github.com/AMD-AGI/Hyperloom/releases/tag/v1.2.0)
follows 1.1.5. Every round now runs through InferenceSuite: Magpie and the
`bypass` backend are removed, and a session resolves its base launch from
InferenceX or the framework's cookbook and freezes it as `base_launch.json`.
The release adds the `hyperloom-multibranch` command, sandboxes specialists
with bubblewrap where the host allows it, records each round's energy in
`power.json`, runs the agent roles on the DeepSeek Harness on request, and
recognizes the MI350X, AMD Radeon AI PRO R9600D graphics, AMD Radeon PRO W7900 and W7900D, and AMD Radeon 8060S. The validated stacks move to SGLang 0.5.23 and vLLM 0.33.0.
Sessions started by 1.1.5 or earlier cannot be resumed, and several options,
environment variables, module paths and report fields move; see the
[upgrade guide](reference/upgrade.md#upgrading-from-115) for migration steps.
The 1.1.5 known issue (a removal KEEP on a Magpie
AgentX recipe that pins golden acceptance) does not apply: that Magpie path is
gone.

### Added

- **A recipe can bound the SGLang draft its deployment runs.** A benchmark YAML
  can declare `speculative_draft_limits` (`draft_length`, `draft_tokens`,
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
  table (with a warning); omitted `--tp` or `--ep` follow the base. Resume
  reads the frozen file and never refetches. A `--reference-script` flag set
  from a shell variable, which used to be dropped silently, now exits 2, as
  does a `# hyperloom-launch-controls:` line or a launch file or default row
  that needs more GPUs than the session has; `--host`, `--port`,
  `--served-model-name` and `--result-*` are recorded as stripped. A JSON
  `base_launch.json` from an earlier session is accepted as the launch file.
  Upgrade note: a session whose state predates `base_launch.json`
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

- **Hyperloom recognizes the MI350X, the AMD Radeon AI PRO R9600D graphics, and the AMD Radeon PRO W7900 and W7900D.** `--gpu-type mi350x`, `r9600d`, `w7900` and `w7900d` are accepted, and each board is identified from its device id or product name at its own roofline figures; a multi-branch session also takes `mi350x`. R9600D, W7900 and W7900D run with `--framework vllm` only, as R9700 does, and a vLLM session on gfx1100 resolves its base launch from the default table. KernelForge runs forge-loop campaigns on gfx1100 with RDNA 3 hardware notes, refuses `aiter`, `flydsl`, `assembly`, `gluon`, `fusion` and `ave` campaigns there, and files gfx1100 experience-KB records under `w7900`. On gfx1201 the automatic AITER FP4/MXFP4 tuners are skipped.

- **KernelForge has an AveLang backend.** `--kernel-backend ave` selects it and backend inference recognizes AveLang imports after the existing checks, so Hyperloom dispatches Ave tasks through its Forge path.

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
  `--benchmark-scripts-dir` can hold `levers.json`, a JSON list of variants
  that EXPLORE seeds ahead of the model's proposals (a model variant of the
  same name wins) and does not seed again once tested. A declaration no
  restart could apply, such as `extra_envs` given as a string, exits `2` at
  launch. Before, a `custom` workload gave EXPLORE nothing to sweep.

- **Orchestration can dispatch a speculative-decoding specialist.**
  `speculative_decoding_specialist` joins the specialist list and takes
  memory-bound decode; it proposes enabling or tuning speculative decoding as
  benchable variants and might download a matching drafter from Hugging Face.
  Specialists now see `HF_HOME`, `HF_HUB_CACHE`, `HF_HUB_OFFLINE` and
  `HF_ENDPOINT` (still not `HF_TOKEN`), so a non-default cache, offline mode or
  mirror applies to them too.

- **The agent roles can run on the DeepSeek Harness.**
  `INFERENCE_OPTIMIZER_AGENT_BACKEND=deepseek` runs orchestration, specialists
  (as `dsh` CLI subprocesses), TraceLens and quantization on the harness, while
  the critic stays on the side the credentials rank first, and Forge takes
  `--agent-backend deepseek` or
  `FORGE_AGENT_BACKEND=deepseek`. It needs the `deepseek` extra and the
  `OPENAI_BASE_URL` and `OPENAI_API_KEY` gateway; `DSH_MODEL` (also read from
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
  logs it. Upgrade note: a session on part of an NPM-enabled node,
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
  component agent (an agent an installed adapter recognizes) the session's
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
  Upgrade note: an `--agent-launch` file no longer takes a `client` key,
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
  `HYPERLOOM_EVAL_LIMIT`. Upgrade note: hosted measurement sheets and
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

- **A custom workload's quality gate can carry raw scores alone.** A
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
  Upgrade note: a frozen base whose entrypoint is not its framework's
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
  rolled back. Upgrade note: move from
  `rocm/vllm:rocm10.0.0_ubuntu24.04_py3.14_pytorch_2.12.0_vllm_0.27.0` to
  `vllm/vllm-openai-rocm:v0.33.0-rocm100` (or match `VLLM_VERSION` and `VLLM_SOURCE_REF` on bare metal).

- **Bump validated SGLang stack to 0.5.23 (ROCm 10 docker).** Updates
  `SGLANG_REF`, `SGLANG_PRETEND_VERSION`, the recommended
  `lmsysorg/sglang-rocm:v0.5.23-rocm10-*-20261111` image tags, `models.tsv`,
  the quick-start `Dockerfile` and the hosted app image `Dockerfile`. These
  builds ship `fastokens`, which the DeepSeek-V4.1-Flash InferenceX recipes
  pass as their tokenizer backend, and answer `/start_profile` without an
  admin API key, so the roofline profile captures a trace. Upgrade note:
  move from `v0.5.21-rocm10-mi30x|mi35x-20261008` to
  `v0.5.23-rocm10-mi30x|mi35x-20261111` (or match `SGLANG_REF` on bare metal).

- **A Ray install is certified by Ray's own `ray start` parser**, not by the
  version of `click` it is built on. Preflight and the kernel-tools installer
  hand `ray start` a representative argv of the one Hyperloom runs on the head
  and accept the CLI exactly when it parses, so a `click` release that leaves
  the CLI working no longer triggers a reinstall or a downgrade.
  `RAY_CLI_CLICK_MAX_VERSION` is removed, superseding the 1.1.4 entry that
  introduced it; `RAY_VERSION` is still honored verbatim.

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
  recognizes AMD Radeon 8060S graphics (gfx1151) and keys its experience-KB records under
  `radeon8060s`; gfx1201 records stay under the ISA name, which more than one
  board shares.

- **Specialists run in a bubblewrap sandbox where the host allows it.** On such a host a specialist can no longer read the app's home, credentials or state, nor reach anything but the LLM endpoints and MCP servers its launch names; in a multi-branch session agent and dev sandboxes are confined too. A native build that cannot run as user 65534 compiles under bubblewrap (`bwrap+netns`). The start-up log says once how agents are isolated and the session report carries an `agent isolation:` line. Where neither the specialist account nor bubblewrap works, a run logs one WARNING and agents run unconfined, unless isolation is required (a hosted run, or `HYPERLOOM_SPECIALIST_USER` set), which refuses; a hosted container that maps only its root now starts, with its specialists under the layer's bubblewrap. A specialist that swaps its worktree for a link gets no patch collected (`worktree_refused`), and a host whose user namespace does not map the specialist account refuses the switch (`RunAsError`) instead of running it as root. `python -I -m hyperloom_common.bwrap probe` says whether a host can sandbox agents and why not. Upgrade note: a sandboxed `claude` CLI cannot read a login kept only in the app's home: such a launch runs unconfined with a WARNING, or is refused where isolation is required, until its environment carries `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `CLAUDE_CODE_OAUTH_TOKEN` or Bedrock credentials.

- **The kernel-agent env file and a shared `.env` no longer hand out credentials.** `install.sh` writes `kernel-agent.env.sh` to `${TMPDIR:-/tmp}/hyperloom-<uid>-<key>/` instead of `$USER_DATA_PATH/runtime/`, removes a copy left there, and stops when `KERNEL_AGENT_ENV` points into a directory other accounts can write; preflight, the documented source line and the slurm job refuse such a file too. A `.env` that is not private to the account, or that sets `HYPERLOOM_DOTENV_CREDENTIALS=kernel-agent-env`, lends no credential; one that another account can write also exports only listed non-credential settings with listed values. Before, every pod of the user could read the credentials in that file and replace what the next launch sourced. Upgrade note: run `install.sh` in every new container before `optimize`; one that did not now fails preflight with exit 2 instead of sourcing a file another container wrote.

- **A session on shared storage can be continued by another account of its group.** Below a setgid, group-writable directory closed to others (`2775`), the session's state, database, lock, logs, blackboard, git objects and refs, runner cache and archive are written group-writable, so `--resume-from` or a hosted restart as another account of the group reads and continues what an earlier leg left; git's own config, `HEAD` and hooks stay the creating account's. A session whose `--extra-env` or `--server-args` hold a credential keeps `state.json`, `manifest.json` and its journal owner-only, and a resume by another account is refused naming the file. A directory a leg cannot write is named up front with its owner and mode. Elsewhere modes are unchanged. Upgrade note: create the shared directories group-owned with mode `2775` once.

- **The orchestration role no longer stalls inside its own turn.** On the Claude backend an orchestration call can run 32 Claude Code turns instead of 12, so a model that talks between tool calls reaches its intent. On Codex the orchestration session runs without Codex's shell, so it cannot `sleep` in its turn while nothing is dispatched, and a timed-out turn is not retried. On DeepSeek every round opens a fresh conversation, and every DeepSeek Harness runtime compacts at a quarter of the model's window instead of 80%; a 12-hour run had spent about 403M input tokens against 167M on Claude. Codex runs no longer offer the hosted `web_search` tool, whose result a gateway fronting Claude rejected with a 400.

- **An agent backend's launch probe waits up to five minutes.** It waited 60 s, so on a gateway that took most of a minute to answer, a campaign configured for Claude or Codex switched at launch to the fallback provider's default model. A genuinely hung gateway now takes up to 300 s (or the session timeout, if shorter) to fail over; a probe the gateway rejects fails as fast as before.

- **TraceLens is pinned at `bda57c6` (1.1.5 pinned `614c597`).** Analyses gain the speculative-decoding graph merge and perf models for GLM-5.x, DeepSeek-V4-Pro and DeepSeek-V4.1-Flash kernels, and a report whose compute findings lack their Data tables is sent back for repair instead of reaching Hyperloom with no hot kernels. TraceLens's SGLang graph-capture patch now applies to SGLang 0.5.18 through 0.5.21 only: on 0.5.22 and later a profile keeps detailed annotations (before, the patch was reported `unavailable` and annotations were turned off), and the analysis reads SGLang's own `graph_capture_profile/` when `capture_traces/` holds no capture trace.

- **forge-fuse and forge-rewrite name the host's GPU.** On a host other than MI355X, the rewrite controller's operator id and task directory, forge-loop's `--gpu-type` and `--gpu-target`, and the KB identities forge-loop and forge-rewrite file under carry that host's SKU and arch (for example `mi300x`, gfx942) instead of `mi355x`, gfx950, and forge-fuse counts Gemma-style norm kernels toward `launch_bound_share`. Upgrade note: KB records such hosts filed under `mi355x` are no longer read back, and `forge-rewrite-by-flydsl` without `--gpu-target` refuses to start where neither `GPU_TARGET` nor rocminfo names exactly one GPU (it assumed gfx942).

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
  row per candidate. Upgrade note: readers must accept both throughput
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
  replaces `--use-llm-orchestrator` or `--no-llm-orchestrator`, which now exit
  with `unrecognized arguments`. The `non-llm` route needs a TraceLens
  checkout. Upgrade note: the `bypass_trace_analysis` module is gone, and
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
  Upgrade note: `forge-loop --resume` of a campaign whose `run_state.json`
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

### Removed

- **Environment variables nothing honors any more.**
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
  `batch_insert.jsonl`. Upgrade note: a leftover `batch_insert.jsonl` stays
  queued as an unknown endpoint, so clear it before upgrading a live
  critic-KB deployment.

### Resolved Issues

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
  (`baseline_noop` or `restates_baseline`) instead of costing a measurement
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
  `no_changes`, iteration admission honors absolute deadlines, and the
  workspace guard deletes the `.pyc` an interpreter writes under a protected
  directory instead of failing a forge_fusion author session with
  `protected ignored files changed`.

