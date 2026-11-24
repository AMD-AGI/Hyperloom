---
myst:
    html_meta:
        "description": "Step-by-step migration guide for upgrading Hyperloom. Covers required and recommended changes from 0.5.x to 0.6.0 and the generic upgrade procedure."
        "keywords": "Hyperloom, upgrade, migration, version, release notes, 0.6.0, USER_DATA_PATH, GEAK, TraceLens, Ray, AMD GPU, ROCm, session, inference optimizer"
---
# Upgrade Hyperloom version

Per-version migration steps. This page is a companion to
[Releases](https://github.com/AMD-AGI/Hyperloom/releases): a release answers *what
changed*, this page answers *what you have to do about it*.

If you are starting fresh, skip this page and follow the
[installation instructions](../install/install.md).

---

## Conventions

These labels are used throughout this page to indicate urgency:

* **Required**: Your run will fail or behave incorrectly until you do
  this.
* **Recommended**: Your run will still work, but you will get a
  deprecation warning or suboptimal behavior.
* **Optional**: Strictly improves UX or unlocks new features.

Hyperloom does not mutate your `.env` on upgrade; all migrations
below are explicit.

```{note}
Shell paths on this page follow the recommended `pip install --target .` layout.
In a source checkout, replace the `hyperloom/` prefix with `src/hyperloom/`.
The command blocks assume `REPO_ROOT` points at that workspace. No installer
exports it for you, so run `export REPO_ROOT="$(pwd -P)"` from the workspace
first — it does not survive a new shell.

The `${USER_DATA_PATH:-/workspace/hyperloom}` fallback below is the last resort
in the chain and only correct where `/workspace` is writable; on a bare-metal
host the CLI defaults to `session/` under the current directory instead. Prefer
`KERNEL_AGENT_ENV`, which the installer writes for you.
```

---

## Upgrading from 1.1.5

Finish in-flight sessions under the build that started them. Every session's
`storage/coordinator.db` now records schema version 8 and there are no
in-place migrations, so `optimize --resume-from` on a session started by 1.1.5
or earlier exits `2` before it touches the session, with
`coordinator.db is at schema version 7, older than this build's 8: sessions created before this release cannot be resumed`.
`kernelforge forge-loop --resume` likewise refuses a 1.1.5 checkpoint that has
finished a round, with
`run state round_costs.recent[0] missing fields: review_sec`.

Recipe KB rows that 1.1.5 filed under a framework build with a local label,
`v` prefix or `.dev` segment (for example `0.30.0+rocm723`) are no longer exact
warm-start hits: identities now key on the release (`0.30.0`), and those rows
are reached only through the `compatible_framework_version` tier (confidence
0.72). Nothing migrates them; the next CLOSE writes the release row.

### Removed options and module paths

Options and module paths 1.1.5 accepted are removed or refused. The parsers
are strict, so a command that still carries one exits with
`unrecognized arguments` or `No module named` before it runs.

| Removed | What to do |
|---|---|
| TraceLens kernel tool `--use-llm-orchestrator` or `--no-llm-orchestrator` | Use `--route agent` or `--route non-llm`. |
| TraceLens kernel tool `--num-denoise-steps`, `--model-path`, `--precision` | Delete them; they fed the diffusion latency roofline, which is removed. |
| `python -m hyperloom.orchestrator.trace_analysis.bypass_trace_analysis` | Use `tracelens_analysis --route non-llm`. |
| `python -m hyperloom.inference_optimizer.framework_deps` | Nothing replaces it; its installer assets were already gone, so it did nothing. |
| A `--benchmark-config` YAML whose `profiler.torch_profiler.num_profiles` is not `1` | Refused at launch: a profile round captures one window. Drop the key. |
| Imports of `hyperloom.common.{aiter_jit_cache, checkpoint_quant, coerce, env, fs_utils, git_safety, gpu_identity, io, jsonio, pr_monitor_urls, proctree, timeutil, unified_diff}` and of `hyperloom.common.{llm_config, llm_headers, llm_attribution, llm_request_hooks, llm_stability_env, token_usage, reasoning_effort, claude_oneshot}` | Import them from `hyperloom_common` and `hyperloom_common.llm`; no aliases are kept. `kernelforge.tools.durable_io` is removed: use `hyperloom_common.io`. |
| KernelForge modules under `kernelforge.loop`, `kernelforge.gemm_tune`, `kernelforge.fusion`, `kernelforge.orchestrator`, `kernelforge.assembly`, `kernelforge.llm`, `kernelforge.tracker`, `kernelforge.tools`, `kernelforge.kernel_rewrite_controller` and `kernelforge.rewrite_by_flydsl` | They move under `kernelforge.workflows`, `kernelforge.kernel_backends`, `kernelforge.agent_backends`, `kernelforge.utils` and `kernelforge.config`, with no compatibility modules. A hand-written MCP configuration that runs `python -m kernelforge.orchestrator.specialist_probe_server`, the path 1.1.5 named, needs `kernelforge.workflows.forge_loop.roles.specialist_probe_server`; `kernelforge.evaluation.benchmark.runner` and `kernelforge.knowledge.pr_knowledge.mcp_server` keep their paths. No `kernelforge` subcommand is removed. |

### Removed and changed environment variables

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
| `HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass`, or `analysis_route: bypass` in a `trace_analyze` request | Fails the request with `invalid_analysis_route`. Set `non-llm`, which needs a TraceLens checkout and writes `analysis.json` instead of `analysis.md`. `HYPERLOOM_BYPASS_STEADY_STATE`, `HYPERLOOM_BYPASS_CORR_WARN_PCT`, `HYPERLOOM_BYPASS_OTHERS_WARN_PCT`, `HYPERLOOM_TRACE_SHAPE_MANIFEST`, `HYPERLOOM_TRACE_SHAPE_MANIFEST_MAX_CAPTURES`, `HYPERLOOM_NUM_DENOISE_STEPS` and `KERNEL_AGENT_USE_LLM_ORCHESTRATOR` are ignored. |
| `HYPERLOOM_SPECIALIST_PERMISSION_MODE` | Delete it; a Claude specialist always runs with `--permission-mode bypassPermissions`. |
| `INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_FAILURES`, `HYPERLOOM_ROCPROF_ROOFLINE_ENRICH`, `HYPERLOOM_ROCPROF_ROOFLINE_TIMEOUT_SEC` | Delete them; nothing reads them. |
| `GEAK_TERM_GRACE_S` | Delete it; the KERNEL phase stops every process its GEAK run started, and `GEAK_FLUSH_GRACE_S` is the window GEAK gets to flush its result. |
| `RAY_CLI_CLICK_MAX_VERSION` | Delete it; the Ray CLI is certified by its own `ray start` parser. `RAY_VERSION` is still honored. |
| `HYPERLOOM_GPU_POWER_INTERVAL_S` | The default drops from 2 s to 1 s. A value above 1.5 s is refused: the round samples nothing, so a power budget refuses it as unmeasured. |
| `KERNEL_AGENT_ENV` naming `$USER_DATA_PATH/runtime/kernel-agent.env.sh` (a 1.1.5 `.env` or shell) | Ignored and derived again; the file now lives at `${TMPDIR:-/tmp}/hyperloom-<uid>-<key>/kernel-agent.env.sh` on the container's own disk, and install removes the old copy. A value in a directory that the group or others can write **is refused** by `install.sh` and by preflight. |
| `TRACELENS_ROOT` naming an earlier install's `TraceLens@<sha>` | Exits 2 at preflight (`is an earlier install's checkout`). Delete the line and rerun `install.sh`. |
| A malformed numeric `HYPERLOOM_MN_*` setting (for example `HYPERLOOM_MN_HEALTH_POLL_S=10s`) | Fails the first multi-node restart (`invalid multi-node restart setting`) and makes a `hyperloom multi-node` subcommand exit 3, where 1.1.5 fell back to a default on some paths. |

### Changed defaults and gates

The same command behaves differently with these defaults and gates:

- **The base launch comes from a recipe, not from Magpie's scripts.** Without
  `--reference-script`, a new session resolves its server launch from InferenceX
  `main`, then the framework's cookbook, then (for a fixed workload only) a
  default table, and freezes it as `base_launch.json`. The sources are read from
  GitHub at launch; a fixed session that cannot read one takes the default
  table, and `--inferencex-commit SHA` pins the InferenceX read. An omitted
  `--tp` or `--ep` (and no `$TP` or `$EP`) now follows the recipe. A native AgentX
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
  the line given in the [Generic upgrade procedure](#generic-upgrade-procedure).
- **A session that holds a credential keeps its database outside the session
  directory.** When `--extra-env` or `--server-args` carry a credential, the
  coordinator database lives below `${TMPDIR:-/tmp}/hyperloom-ledger-<uid>/`,
  so resuming it in a new container is refused unless `HYPERLOOM_LEDGER_DIR`
  names a persistent private directory that holds it.

### Installation prerequisites

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
  comments dropped) from the assignments such a file can lend, and leaves
  `HYPERLOOM_DOTENV_CREDENTIALS=kernel-agent-env`; export credentials in the
  shell that runs `install.sh`.
- New optional extras: `deepseek`, `hosted`, `runner` and `inferasim`; `forge`
  adds `pydantic>=2.0`. No console script is removed; `hyperloom-multibranch`
  is new.

### Session and report consumer changes

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
  [`session_breakdown.json` integration](session-breakdown.md).
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

---

## Upgrading from 1.1.x

### Required: launch through `hyperloom <command>`

Every command now runs under one entry point, `hyperloom <command>`, or
`python -m hyperloom <command>` where no console script is installed. Update
launchers, skills and scripts:

| Before | After |
|---|---|
| `python -m hyperloom.inference_optimizer.cli --verbose optimize ...` | `python -m hyperloom optimize --verbose ...` |
| `inference_optimizer optimize ...` | `hyperloom optimize ...` |
| `... recover-session --session-dir S` | `hyperloom recover --session-dir S` |
| `python -m hyperloom.inference_optimizer.setup ...` / `hyperloom --check-only` | `hyperloom setup ...` / `hyperloom setup --check-only` |
| `quantization-agent ...` | `hyperloom quantize ...` |
| `python -m hyperloom.inference_optimizer.multi_node SUB ...` | `hyperloom multi-node SUB ...` |
| `python -m hyperloom.inference_optimizer.tools.dump_session_breakdown` | `hyperloom session breakdown` |
| `python -m hyperloom.inference_optimizer.tools.dump_session_report` | `hyperloom session report` |
| `python -m hyperloom.inference_optimizer.tools.backfill_langfuse` | `hyperloom session backfill` |
| `.../tools/event_counts.py` | `hyperloom session events` |
| `.../tools/read_optimizer_state.py` | `hyperloom session state` |
| `.../tools/preflight_optimizer.py MODEL_PATH` | `hyperloom check MODEL_PATH` |

`--verbose` now follows the command name. `hyperloom` without a command prints
its usage and exits 2 instead of running setup.

### Required: removed flags

These flags are rejected with `unrecognized arguments`:

* `--kernel-codex`, `--kernel-claude`, `--kernel-prompt` (no-ops since the
  kernel LLM role was retired): drop them.
* `--max-minutes-{prelude,framework,explore,kernel,sweep,close}-pct` and
  `--phase-budget-explore-pct`: use `--phase-budget-<phase>-pct`
  (`--phase-budget-framework-pct` covers what the explore spelling set).
* `recover --backfill-trace`: run `hyperloom session backfill --session-dir S`.
* On the rayjob backend, `--pd-prefill-ep`, `--pd-decode-ep` and
  `--pd-*-extra-args` now fail the server restart instead of being ignored;
  they are supported on infera only.
* `--pr-feed-window-days` and `--local-kb-root` (no reader left): drop them;
  select the knowledge root with `KNOWLEDGE_LOCAL_ROOT`.
* `kernelforge forge-loop --return-after-read-kb`, and the TraceLens kernel tool's
  `--skip-split` / `--source-root` (with `TRACELENS_SOURCE_ROOT`): drop them.
* `hyperloom multi-node kernel-bench` is removed; nothing in the optimizer calls it.
* The TraceLens kernel tool's `--use-llm-orchestrator` / `--no-llm-orchestrator`:
  use `--route {agent,non-llm}`. Its `--num-denoise-steps`, `--model-path` and
  `--precision` fed the removed diffusion latency roofline: drop them.
* `python -m hyperloom.orchestrator.trace_analysis.bypass_trace_analysis` is
  removed and fails with `No module named`; use `tracelens_analysis --route non-llm`.

### Required: the `bypass` trace-analysis route is replaced by `non-llm`

`HYPERLOOM_TRACE_ANALYSIS_ROUTE=bypass`, or `analysis_route: bypass` in a
`trace_analyze` request, now fails the request with `invalid_analysis_route`.
Set `non-llm` instead; it needs a TraceLens checkout and writes `analysis.json`
rather than `analysis.md`.

These are no longer read and are silently ignored, so drop them:

* `HYPERLOOM_BYPASS_STEADY_STATE`, `HYPERLOOM_BYPASS_CORR_WARN_PCT`,
  `HYPERLOOM_BYPASS_OTHERS_WARN_PCT`, `HYPERLOOM_TRACE_SHAPE_MANIFEST` and
  `HYPERLOOM_TRACE_SHAPE_MANIFEST_MAX_CAPTURES` (the bypass route).
* `HYPERLOOM_NUM_DENOISE_STEPS` and the `trace_analyze` keys `num_denoise_steps`,
  `model_path`, `precision` and `analysis_mode`.
* `KERNEL_AGENT_USE_LLM_ORCHESTRATOR`; the agent route always runs unless
  `--route non-llm` is given.

### Required: sessions created before this release do not resume

Every session's `storage/coordinator.db` now records schema version 8, and
there are no in-place migrations. `optimize --resume-from` on a session whose
database records an older version, which is every session started by 1.1.5 or
earlier, exits 2 at resume admission, before `state.json` or the database is
touched:

```text
ERROR: --resume-from coordinator.db is at schema version 7, older than this build's 8:
sessions created before this release cannot be resumed. Finish the session under the
build that started it, or start a new session.
```

Opening such a database through the coordinator storage raises the same
message without writing to it. The legacy shapes the old migrations handled
(`events.priority`, a `leases` table without `owner_scope`, `state.json`
without `elapsed_charged_sec` or `phase_elapsed_totals`, manifests without
`workflow_contract_version`) are no longer read. Finish in-flight sessions
before upgrading.

### Required: `session_breakdown.json` drops `robustness`

The top-level `robustness` key, `close.robustness` and the
`robustness_escalated` stop reason are gone; they were always empty since the
robustness agent was retired. Consumers that read them should stop.

### Required: `session_breakdown.json` drops the latency roofline and `bypass` keys

The schema stays `v6.1` but drops the keys only the diffusion latency roofline
or the `bypass` route produced; read `ceiling_available` instead of
`close.roofline_progress.ceiling_kind`. Each key and its disposition is under
Versioning in [`session_breakdown.json` integration in Hyperloom](session-breakdown.md).
The trace certificate's `production_*` keys go with them; read `selected_role`.

### Required: quantization failure exits 4

A failed `--quantize` prelude now exits 4 instead of 3; exit 3 means only
that another optimizer holds the session lock. Launchers that branch on the
exit code should treat 4 as a non-retryable failure. See also
[operations.md](operations.md) for the full exit-code table.

### Recommended: commands that stay outside `hyperloom.cli`

`hyperloom <command>` covers setup, check, optimize, recover, quantize,
multi-node, and `session {breakdown,report,backfill,events,state}`. These
entry points are intentionally separate (same as before the unification,
only the optimizer-family commands moved under `hyperloom`):

| Entry | Typical use |
|---|---|
| `kernelforge` / `python -m kernelforge.cli` | KernelForge campaigns (orchestrator dispatches `-m kernelforge.cli`) |
| `hyperloom-kb-serve`, `hyperloom-kb-collect`, `python -m hyperloom_kb` | Experience KB service and collect CLI |
| `hyperloom-kb` / `python -m hyperloom_kb.cli` | Operate an Experience KB service: health, labels, restore, exclusions, list, export |
| `python -m hyperloom.inference_optimizer.experience_kb_service {init-env,ensure,...}` | Per-workspace Experience KB sidecar, with every `hyperloom-kb` command (setup skill, optimizer SKILL) |
| `python -m hyperloom.agents.critic.runtime.cli {prepare-review,commit-review,replay-dead-letter}` | Critic backend subprocess (not operator-facing) |
| Kernel agent tool modules (`tracelens_analysis`, …) | Orchestrator subprocess scripts with their own `__main__` |
| `multi_node/scripts/*.py` on remote pods | Payload scripts SSH'd by `hyperloom multi-node`, not local `-m` entry points |

There is no `hyperloom kb` subcommand in 1.1.x; keep using
`experience_kb_service`, `hyperloom-kb`, or the `hyperloom-kb-*` console scripts above.

---

## Upgrading from 0.5.x → 0.6.0

Apply the following changes in order. Required steps must be completed before running; recommended and optional steps improve behavior or unlock new features.

### Required: rename multi-node image / GPU flags and envs

The `optimize` CLI multi-node image and per-node GPU flags were renamed
and the legacy names are no longer accepted (no alias). Any launcher that
passes the old flags will fail with `unrecognized arguments`. The image
is platform-owned and `--mn-image` is accepted but ignored; the per-node GPU
count is set through `--gpus-per-node`. The previous
`INFERENCE_OPTIMIZER_RAYJOB_IMAGE` env is no longer read.

```diff
# run launchers, k8s ConfigMaps, .env
- --rayjob-image harbor/...
+ --mn-image harbor/...
- --rayjob-gpus-per-node 8
+ --gpus-per-node 8
- INFERENCE_OPTIMIZER_RAYJOB_IMAGE=harbor/...
```

The new flags cover both multi-node backends (`rayjob` head+workers and
`infera` worker/prefill/decode pods), which is why they dropped the
`rayjob`-specific prefix.

### Recommended: review `--model-class` if you relied on live classification

The live `classify` action was removed. Current Coordinator boot still infers
and persists `model_class` from model metadata or model-path family keywords
when possible, but launchers that know the class should pass it explicitly to
avoid a generic fallback:

```diff
python3 -m hyperloom optimize \
    --model /path/to/GLM-5-FP8 \
    --framework sglang \
    --gpu-type mi355x \
+   --model-class moe_mla_nsa \
    --isl 1024 --osl 1024 \
    --max-hours 2.0
```

Recognised `--model-class` values (case-insensitive, with `-`/`+`/space
tolerated; see [Inference Optimizer Skill](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/SKILL.md)):
`dense`, `moe_mla`, `moe_swa`, `moe_mla_nsa`.

If `--model-class` is omitted and inference cannot determine the family, the
Coordinator falls back to a generic dense prior, likely suboptimal for mixture of experts (MoE) /
multi-head latent attention (MLA) / native sparse attention (NSA) models.

### Required: pass `--compare-against-gpu` to opt into InferenceX reference fetching

The `classify` action used to derive this implicitly. Without
`--compare-against-gpu`, `target_analysis` writes a
`no_target_gpu_configured` marker and the run proceeds without an
external reference (the optimisation still works; you just don't get
the "vs B200" comparison number).

```diff
+ --compare-against-gpu B200
```

### Required: setup is no longer an in-loop action

Earlier launchers might have waited for the Coordinator to emit a
`setup` action. Move all setup work to before the
`python -m hyperloom optimize` call:

```diff
# launcher.sh
- python3 -m hyperloom.inference_optimizer.cli optimize ... # expects setup as first action
+ bash "$REPO_ROOT/hyperloom/inference_optimizer/assets/install.sh"
+ kernel_agent_env_file="$(. "$REPO_ROOT/hyperloom/inference_optimizer/assets/runtime_env.sh"; runtime_env_kernel_agent_env_path)" && . "$kernel_agent_env_file"
+ ray stop --force; ulimit -Sn "${RAY_MIN_NOFILE:-65536}" 2>/dev/null || true; ray start --head --num-gpus="$RAY_NUM_GPUS" --include-dashboard=false --resources='{"serving_slot":1}'
+ python3 -m hyperloom optimize ...
```

### Recommended: review the `KERNEL_OPT_BACKEND_ORDER` default

Bare-metal installs now default `KERNEL_OPT_BACKEND_ORDER` to `geak`. If you
had a custom order hard-coded, confirm it is still intentional.

### Recommended: set `INFERENCE_OPTIMIZER_RESCUE_PATHS` if you use model-specific benchmark scripts

When `--gpu-type` is set, Hyperloom pins
`benchmark_script=<framework>_<gpu_type>.sh` to stop InferenceX-native
scripts from leaking `result.json` outside the session dir. If you
intentionally use a model-specific script (for example, `dsr1_fp8_mi300x.sh`),
keep passing `benchmark_script=` explicitly — operator overrides
still win against the generic-script pin. You might additionally want
to set `$INFERENCE_OPTIMIZER_RESCUE_PATHS` so the harvest step can
recover any leaked `result.json` files written to hardcoded
`--result-dir` locations.

### Recommended: stop expecting `standalone_analysis.md` or `tracelens_report.md`

The kernel-agent no longer aliases the TraceLens v0.3 report. The
canonical analysis path is now surfaced by the `trace_analyze` request
handler as `trace_report_path` and forwarded to the lifecycle as
`analysis_report_path` (pointing at
`$SESSION_DIR/kernel-agent/runs/<session_id>/tracelens/analysis.md`).
The `--compat-report-path` argument was removed.

### Roofline is now on by default (no PMC env toggle)

There is no `HYPERLOOM_ENABLE_PMC_ROOFLINE` environment variable. Roofline
analysis (the composite `profile` + `trace_analyze` + `analysis.md` path) is
controlled by the CLI flag `--enable-roofline`, which defaults on. Pass
`--no-enable-roofline` for a profile-only run. See
[Environment variables](environment-variables.md).

### Schema compatibility

`session_breakdown.json` emits `hyperloom.session_breakdown.v6.1`. 6.1 changes
three things inside the 6 major, a declared exception to the major-bump rule: it
renames `throughput_tok_s_per_gpu` in `outcome.baseline`, `outcome.final` and
the `baseline` event's `measurement` to `output_throughput` (the old key is no
longer written), removes keys no producer populated, that only carried a
constant or whose producer was removed, and reports two unmeasured figures as null. Archived 6.0 documents and documents exported by 1.1.x keep
the old key, so readers must accept both keys or migrate.
Re-exporting a 6.0 session directory emits a null
`outcome.baseline.output_throughput` and the `contract_digest` recorded at
session creation, and the default output overwrites the archived
`<session_dir>/session_breakdown.json`. The removed keys (disposition: removed
in 6.1, never populated) are `policy.force_exit_budget_pct`,
`policy.variant_timeout_sec`, `policy.overtime_kill_ratio`,
`trace_validate_ref`, `critic_review.iteration`, `geak.product.cfg_hash`,
`geak.product.final_overlay_digest`, `entry.budget_remaining_sec` and the
roofline rows' `rocprof_roofline`; the roofline rows' `roofline_measured` and the
baseline action's `convergence` (disposition: removed in 6.1, constant) are
removed as well.
`attributed_gain_pct`, `unattributed_gain_pct` and run `empty` are null when
unmeasured or not provided.

V6 was the preceding cutover, for the timeline: each action records its own event as it
runs, so an event's `start_time` is when the work started rather than when its
artefacts were written, and the KERNEL and BASELINE projections are no longer
emitted. Ordering that relied on the old collapsed windows changes as a result.

V5 was the preceding cutover, for optimization results: adopted optimizations
are reported only through `optimizations`, and the `optimization_stack`,
`attribution`, GEAK invocation, Forge invocation, and GEMM-tuning projections
are no longer emitted. Consumers reading archived v2 / v3 / v4 / v5 documents
need a downstream migration, as described in
[`session_breakdown.json` integration in Hyperloom](session-breakdown.md).

---

## Upgrading from pre-0.5 (training and MLPerf-training)

Hyperloom is inference-only since v0.4. If you are still on a
training-mode build:

* The training-mode TraceLens CLI
  (`TraceLens_generate_perf_report_pytorch_training`) is no longer
  accepted by `install.sh`. Use the `_inference` variant.
* Training and MLPerf-training skills have been removed from this
  repo. There is no in-place migration; switch to the inference flow
  documented in the [setup and examples guide](../../examples/README.md).

---

## Required: migrate to exclusive KnowledgePlane modes

KnowledgePlane now selects exactly one Recipe backend. This is a breaking
change for deployments that previously got remote-first reads merely by
exporting GBrain credentials:

```diff
  GBRAIN_BASE_URL=https://gbrain.example
  GBRAIN_TOKEN=...
+ KNOWLEDGE_STORE_MODE=remote
```

Without the explicit mode, local/default ignores both credentials. Set
`KNOWLEDGE_STORE_MODE=local` (or leave it unset) for local-only operation.

The implicit local root changed from `$USER_DATA_PATH/kb` (or
`/workspace/hyperloom/kb`) to `$USER_DATA_PATH/knowledge` (or
`~/.cache/hyperloom/knowledge`). Data under the old root is neither read nor
migrated; `--local-kb-root` and `HYPERLOOM_LOCAL_KB_ROOT` are removed, so use
`KNOWLEDGE_LOCAL_ROOT` to select a root.

---

## Generic upgrade procedure

For any minor or patch upgrade:

1. Pull the new Hyperloom revision into `$REPO_ROOT`.
2. Re-run `bash "$REPO_ROOT/hyperloom/inference_optimizer/assets/install.sh"`. The
   installer is idempotent: it picks up new GEAK / TraceLens versions,
   refreshes generated LLM gateway aliases, and regenerates `kernel-agent.env.sh`.
3. Re-source the env file. `runtime_env.sh` prints it only when `install.sh` wrote it in this
   container and no other account could have changed it; otherwise it says why and nothing is sourced:
   ```bash
   kernel_agent_env_file="$(. "$REPO_ROOT/hyperloom/inference_optimizer/assets/runtime_env.sh"; runtime_env_kernel_agent_env_path)" && . "$kernel_agent_env_file"
   ```
4. To resume an ongoing session across the upgrade, first check the target
   release's upgrade section: a release that raises the `coordinator.db`
   schema version (1.1.x → this release does) refuses sessions started before
   it, so finish those under the build that started them. Otherwise verify
   `manifest.json`, `state.json` and `storage/coordinator.db` are intact, then run
   `python -m hyperloom optimize --resume-from "$SESSION_DIR"`.

Upgrades do not move storage or rewrite historical sessions.

---

## Related guides

Use these resources for related reference information:

* [Releases](https://github.com/AMD-AGI/Hyperloom/releases): Full per-release notes.
* [Hyperloom authentication and credentials](authentication.md): Credential and path env reference.
* [Environment variables](environment-variables.md): Every
  environment variable read by the runtime.
* [Hyperloom self-hosting and operations guide](operations.md): Self-host runbook.
