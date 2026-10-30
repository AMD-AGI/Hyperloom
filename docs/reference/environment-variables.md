---
myst:
    html_meta:
        "description": "Reference for the environment variables read by the Hyperloom runtime, grouped by purpose: credentials, paths, workload parameters, backend selection, and observability. Opt-in benchmark modes carry additional variables documented alongside them."
        "keywords": "Hyperloom, environment variables, configuration, OPENAI_API_KEY, USER_DATA_PATH, ROCm, AMD GPU, LLM inference, kernel optimization, LLM gateway, Langfuse, session"
---
# Environment variables

User-configurable environment variables for Hyperloom, grouped by purpose.
Runtime parameters such as framework, tensor parallelism, prompt lengths, and
phase toggles are configured with CLI flags; internal subprocess handoff envs
are intentionally not listed as user configuration.

Variables marked **Required** must be set (using shell or `$REPO_ROOT/.env`)
or the CLI will exit fast at startup. Variables marked **Optional** have
sensible defaults; the default is shown in the **Default** column.

Precedence rule (applies everywhere): shell-exported env wins over `.env`.
See [Hyperloom authentication and credentials](authentication.md).

---

## Credentials

These variables configure LLM gateway access and optional backend credentials.
See [Authentication and credentials](authentication.md) for the accepted
provider-side combinations and what each one enables.

| Variable               | Required | Default | Description                                                                                                                                                                                            |
|------------------------|----------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ANTHROPIC_BASE_URL`   | Conditional | —    | Anthropic-side endpoint. Required together with `ANTHROPIC_API_KEY` to enable Claude.                                                                                                        |
| `ANTHROPIC_API_KEY`    | Conditional | —    | Anthropic-side key. Pairs with `ANTHROPIC_BASE_URL`.                                                                                                                               |
| `ANTHROPIC_AUTH_TOKEN` | No       | —    | Claude CLI auth token alias, accepted in place of `ANTHROPIC_API_KEY`. Preflight never fills it; the Ray / e2e / forge-fusion env builders default it from the Anthropic-side key when they hand credentials to a subprocess.                                                                        |
| `ANTHROPIC`<br>`_CUSTOM_HEADERS` | No | —    | Extra request headers for the Anthropic side, for gateways that authenticate on a header of their own (for example Azure API Management). Newline-delimited `Name: value` as in the Anthropic SDK; a JSON object is accepted too. `${VAR}` references are expanded from the same environment, so a gateway header can reuse `ANTHROPIC_API_KEY` instead of duplicating the secret. |
| `CLAUDE_CODE`<br>`_OAUTH_TOKEN` | No | — | Claude Max/Pro subscription token from `claude setup-token`. Lowest-priority Anthropic credential: either API-key variable outranks it. On its own it implies `https://api.anthropic.com`. Passed to subprocesses verbatim and never copied into `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, or `~/.claude/config.json`, which would switch the run to API-credits billing. |
| `CLAUDE_CODE`<br>`_USE_BEDROCK` | No | — | Route the Claude runtime through AWS Bedrock, which carries the credential itself. Holds the Anthropic side for **agent backend selection only**, so a Bedrock box is not read as unconfigured and redirected to Codex. It names no key, so it does not satisfy `--critic-protocol=anthropic` or the credential preflight, and it never derives an `ANTHROPIC_BASE_URL`. |
| `CLAUDE_CODE`<br>`_USE_VERTEX` | No | — | Route the Claude runtime through Google Vertex AI. Holds the Anthropic side exactly as `CLAUDE_CODE_USE_BEDROCK` does, with the same limits. |
| `OPENAI_BASE_URL`      | Conditional | —    | OpenAI-side endpoint. Required together with `OPENAI_API_KEY` to enable Codex. An OpenAI-only configuration drives Orchestration through the Codex backend; Claude and GEAK stay disabled.                                                                        |
| `OPENAI_API_KEY`       | Conditional | —    | OpenAI-side key. Pairs with `OPENAI_BASE_URL`. Never borrowed from the Anthropic side.                                                                                                                               |
| `OPENAI`<br>`_CUSTOM_HEADERS` | No | —    | Extra request headers for the OpenAI side. Same shape as `ANTHROPIC_CUSTOM_HEADERS`; set it whenever you set `OPENAI_BASE_URL` against a gateway that authenticates on its own header. |
| `CLAUDE_MODEL`         | No       | Derived from `ANTHROPIC_BASE_URL` | Orchestration model id on the Anthropic side. Falls back to the endpoint default, then the project-wide `DEFAULT_CLAUDE_MODEL`. `GEAK_CLAUDE_MODEL` inherits from it, and Forge reads it directly for its Claude backend.                                                                    |
| `CODEX_MODEL`          | No       | Derived from `OPENAI_BASE_URL`    | Model id on the OpenAI side, used by the Codex backend, including Forge running on it. Model settings are never borrowed across providers.                                                                    |
| `INFERENCE_OPTIMIZER`<br>`_AGENT_BACKEND` | No | Ranked by credentials | Names the agent CLI for the agentic roles: `claude`, `codex` or `deepseek`. An explicit `claude` or `codex` pins the role to that CLI instead of the credential-ranked choice. The choice applies to orchestration, specialists, Forge and TraceLens; quantization honours only `deepseek`. `deepseek` (the DeepSeek Harness) is never picked by the credential ranking: Hyperloom's roles reach it only through this variable, and Forge also takes `--agent-backend deepseek` / `FORGE_AGENT_BACKEND=deepseek`. It needs the `deepseek` extra, and uses the `OPENAI_BASE_URL` / `OPENAI_API_KEY` gateway. With `deepseek`, specialists run on the `dsh` CLI with `DSH_PERMISSION_MODE=danger-full-access`, so dsh applies no sandbox of its own, and a run with specialists enabled (`--research-lane-capacity` above 0) refuses to boot when no `dsh` is found. |
| `DSH_MODEL`            | No       | `deepseek-v4-flash`               | Model id the DeepSeek Harness runs, for every role that selects it. |
| `GEAK_API_KEY`         | No       | —    | Internal alias, never derived from either side. GEAK runs on the Anthropic side (`ANTHROPIC_*` + `GEAK_CLAUDE_MODEL`); set this only to point GEAK elsewhere.                                                                                                                              |
| `GEAK_BASE_URL`        | No       | —    | Internal alias, never derived from either side. Set it only to point GEAK at a different endpoint than the Anthropic side.                                                                                                                          |
| `GEAK_CLAUDE_MODEL`   | No       | Inherits `CLAUDE_MODEL` | GEAKv4 Claude Code workflow model id.                                                                                                                                                           |
| `HYPERLOOM`<br>`_REASONING`<br>`_EFFORT` | No | Provider default | Reasoning effort for Hyperloom's own LLM calls, and the effort a Forge campaign runs at unless `FORGE_AGENT_REASONING_EFFORT` names one. One of `low` / `medium` / `high` / `xhigh` / `max`. `low`–`xhigh` are what the Claude CLI and the OpenAI-compatible gateway both accept; `max` is Claude's deepest and is sent as `xhigh` on the OpenAI protocol, which rejects the name. Set it once and every component runs at the depth you asked for. Hyperloom's own calls ignore an unrecognized value; a Forge campaign refuses to start on one. |
| `FORGE_AGENT`<br>`_REASONING`<br>`_EFFORT` | No | `high` | Forge-only reasoning effort, same five levels. Outranks `HYPERLOOM_REASONING_EFFORT`, and outranks whatever a Forge call site would have chosen — every agent session in a campaign runs at this effort. |
| `LANGFUSE_HOST`        | No (required <br> only <br> when `HYPER`<br>`LOOM_LA`<br>`NGFUSE`<br>`_ENABLE=1`) | Unset | Base URL of your Langfuse deployment (for example, `https://langfuse.<your-domain>`). Used by both the live trace push and the offline `backfill_langfuse` CLI. |
| `LANGFUSE`<br>`_PUBLIC_KEY`  | No (required <br> only <br> when `HYPER`<br>`LOOM_LA`<br>`NGFUSE`<br>`_ENABLE=1`) | Unset | Langfuse project public key (`pk-...`).                                                                                                                  |
| `LANGFUSE`<br>`_SECRET_KEY`  | No (required <br> only <br> when `HYPER`<br>`LOOM_LA`<br>`NGFUSE`<br>`_ENABLE=1`) | Unset | Langfuse project secret key (`sk-...`).                                                                                                                  |

---

## Path environment

The following variables configure filesystem paths for Hyperloom's runtime dependencies and session data.

| Variable                                  | Required             | Default                                                            | Description                                                                                                                                                                          |
|-------------------------------------------|----------------------|--------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `REPO_ROOT`                               | No (recommended)     | `$(pwd)`                           | This Hyperloom checkout. Used to locate `.env`, skills, scripts. Falls back to the current working directory when unset.                                                                                                                     |
| `INFERENCEX_PATH`                         | Conditional          | Auto-cloned by `install.sh`                                    | Path to the SemiAnalysisAI/InferenceX repo, used by baseline / target analysis. `install.sh` clones it when unset; only required if that auto-clone fails.                                                                                                                                          |
| `TRACELENS_ROOT`                          | No (installer auto-clones) | `${HYPER`<br>`LOOM_CA`<br>`CHE_DIR:-`<br>`$REPO_ROOT`<br>`/.cache}/Tr`<br>`aceLens@<resolved-sha>` (auto-clone of `AMD-AGI/TraceLens` pinned to a fixed SHA) | `src/hyperloom/inference_optimizer/assets/install_kernel_tools.sh` clones the public repo into the repo-local cache root when unset. Export it to opt into a pre-existing checkout you maintain — that is an explicit operator override and skips both the clone and the SHA pin. |
| `GEAK_CLAUDE_BIN`                          | No (installer auto-resolves) | `$(command -v claude)` after the installer has put the Claude Code CLI on `PATH`; written to `kernel-agent.env.sh` | Pins the Claude Code binary the GEAK SDK path uses to the same CLI the specialists run, so `claude_agent_sdk` doesn't fall back to its older bundled CLI. Hyperloom's own agents also run it when `HYPERLOOM_CLAUDE_CLI_PATH` is unset. |
| `HYPERLOOM_`<br>`CLAUDE_CLI_PATH`         | No | Unset | The Claude Code binary Hyperloom's agents run: the in-process backend and subprocess specialists alike. When unset they run `GEAK_CLAUDE_BIN`, then `claude` on `PATH`; the in-process backend leaves the choice to `claude_agent_sdk` only when none is found. |
| `USER_DATA_PATH`                          | No                   | `/workspace/hyperloom` if `/workspace` is writable, else `<cwd>/session` | Session directory root (logs, runs, mirrors, breakdown). Container images ship a writable `/workspace`; a bare-metal host that has neither falls back to the second form and the CLI logs which root it took.                                                |
| `HYPERLOOM_`<br>`RUNTIME_DIR`             | No                   | `$USER_DATA_PATH/runtime` (installer)                               | Private writable runtime state. Like `KERNEL_AGENT_ENV` and `HYPERLOOM_ROOT`, it is derived from the shell's `USER_DATA_PATH` when the shell sets one and never read from the checkout's `.env`, which several sessions share; `install.sh` keeps only `USER_DATA_PATH` there. Codex SDK turns create a unique mode-`0700` `CODEX_HOME` here and remove it after the SDK client closes. When unset, Codex uses the first safe declared output root, then a run-local working directory; it never falls back to `/tmp` or a source checkout. If launch config points this parent at storage unsuitable for SQLite, set it to a private local directory instead — see [Codex SDK turns stall](troubleshooting.md#codex-sdk-turns-stall-or-tracelens-roofline-times-out). |
| `INFERENCE_`<br>`OPTIMI`<br>`ZER_CU`<br>`RRENT_S`<br>`ESSION_DIR` | No (set by CLI) | Set at session boot | Absolute path to the active session directory. Written by the CLI when a session starts and inherited by every benchmark subprocess; session-path resolution prefers it over scanning `USER_DATA_PATH`. Do not set by hand. |
| `HYPERLOOM_ROOT`                          | No                   | `$HYPER`<br>`LOOM_R`<br>`UNTIME_`<br>`DIR/sou`<br>`rce-mirrors`                            | Legacy source-mirror root kept for compatibility. Current open-source dependency checkouts default to the repo-local cache root (`${HYPER`<br>`LOOM_CA`<br>`CHE_DIR:-`<br>`$REPO_ROOT`<br>`/.cache}`), not this path. |
| `HYPERLOOM`<br>`_CACHE_`<br>`DIR`                          | No                   | `$REPO_ROOT`<br>`/.cache`                      | Writable, repo-local base for auto-cloned open-source deps (TraceLens, Magpie, etc.), cloned per revision as `<name>@<sha>`. Not under `$TMPDIR` so a reaper cannot wipe it mid-run. |
| `KERNELFORGE`<br>`_PROJECT_`<br>`ROOT`              | No                   | `$USER_DATA_PATH/kernelforge`, else `~/.cache/hyperloom/kernelforge` | Writable root for forge's own state and for resource-tree overrides. Holds the learned knowledge base (`knowledge_base/<backend>/learned/`), the tuning DB, postmortems and `forge_experiments/`. An `examples/` directory placed here overrides the copy packaged inside `kernelforge`, which is the supported way to try a task without editing site-packages. Must be writable: it deliberately never resolves to the installed package directory or to the cwd. **This is the replacement for the removed `FORGE_PATH`**, which nothing reads any more — a stale `FORGE_PATH` is still forwarded (the `FORGE_` prefix is on the dotenv allowlist) and then ignored. |
| `SKIP_FORGE`<br>`_PROFILING`               | No                   | Unset (the extra is installed) | Set to `1` to make `install.sh` skip `pip install -e "$REPO_ROOT[forge-profiling]"`. That extra is rocprof-compute's own dependency set (~20 wheels, including the exact `kaleido==0.2.1` / `astunparse==1.6.2` pins ROCm 7.2.x requires); without it forge's profiler degrades to the lightweight PMC path instead of System Speed-of-Light + roofline. Installed by default on purpose — the previous gate made this a silent skip on every pod. |
| `ROCPC_VENV`                               | No                   | `/opt/rocprof-compute-venv` | Private venv that rocprof-compute's analyze mode runs in, created by `install.sh` from the tool's own `requirements.txt` so its exact pins cannot displace the serving image's numpy and pandas. `rocpc_profile.py` derives the same path. Without the venv, analyze degrades and profiling still collects. |
| `MAGPIE_PATH`                              | No                   | Resolved from installed `Magpie` package unless explicitly set                               | Magpie package root for benchmark wrappers and patch inspection.                                                                                                                                            |
| `INFERENCE_`<br>`OPTIMI`<br>`ZER_SES`<br>`SION_DIR` | No (multi-node) | Unset | Last-resort session root for multi-node crash-log collection. Point it at one session dir, never at `$USER_DATA_PATH`. |

---

## Workload configuration

Set with CLI flags. `ISL` / `OSL` / `CONC` / `TP` / `EP` resolve on a ladder,
and the ladder differs by branch. A fresh launch takes **an explicit flag, then
the environment (where an `--extra-env NAME=VALUE` pin lands), then the
default**. A resume takes **an explicit flag, then the pins in effect for this
resume — its own `--extra-env`, or the launch's persisted pins when it passes
none — then the session's recorded value, then the shell environment, then the
default** — so a value left over in the shell cannot displace the shape the
session was measured at. `PRECISION` is not on that ladder: on both branches it
is **`--precision`, then `$PRECISION`, then the session's recorded value, then
the checkpoint's own dtype, then the default**, so on a resume a `$PRECISION`
left in the shell still outranks the recorded value. `GPU_TYPE` is a fallback
when `--gpu-type` is omitted.

- **Model / workload shape:** `--model`, `--model-class`, `--framework`,
  `--framework-version`, `--precision`, `--tp`, `--ep`, `--isl`, `--osl`,
  `--conc`, `--max-model-len`, `--profile-osl`.
- **Goal / budget:** `--target-gain`, `--target-roofline`, `--max-hours`,
  `--target-tput`, `--compare-against-gpu`. The roofline target composes with
  the others: whichever is met first ends the run.
  `--max-latency-ms` is a constraint on KEEP rather than a target, and applies
  to scriptable frameworks (`xdit`, `custom`) only; the CLI refuses it for a
  serving framework, including on resume. `--max-power-w` is the same kind of
  constraint on the serving GPUs' total mean power, for the serving frameworks
  (refused on a scriptable one, which runs no server to sample), and
  `--max-per-gpu-power-w GPU_ID W` (repeatable) adds a limit for one card; the
  per-GPU limits must add up to no more than the total. That power is sampled
  by Hyperloom over the measured phase into each round's `gpu_power.json`;
  `HYPERLOOM_GPU_POWER_SAMPLING=0` turns the sampler off (the round then falls
  back to the benchmark report's reading, and either power budget is refused at
  launch, since it could not be measured) and `HYPERLOOM_GPU_POWER_INTERVAL_S`
  sets its cadence (default 2 s, minimum 0.5 s).
- **GPU power settings:** `--gpu-power-cap-w`, `--gpu-perf-level`,
  `--gpu-sclk-max-mhz`. By default asserted: set them with `amd-smi set` before
  launch; the session refuses to start or resume if a card it uses differs, and
  records what the cards report either way. `--apply-gpu-power-settings` lets
  the session set the cap and perf level itself (never the clock limit) on
  the cards of its integer visible-device mask for the whole session, and
  restore the originals when it ends; it needs root (records under
  `/var/tmp/hyperloom/gpu_power_settings/`).
- **Node power management (MI350 series):** `--node-power-management`
  (`enabled` / `disabled`), `--node-power-limit-w`. Always asserted against
  `amd-smi node -p`, never set by the session: NPM is set on the BMC. The
  session refuses to start or resume if the node differs, and records the
  observed state and limit either way where `amd-smi` reports them. With NPM
  enabled, a session whose visible-device mask leaves any of the node's GPUs
  out is refused; `--allow-shared-npm-node` (on every launch and resume)
  measures anyway and records the cards left out.
- **Cluster topology & multi-node backend:** `--nodes`, `--gpus-per-node`,
  `--gpu-type`, `--mn-backend` (`rayjob` / `infera`), `--server-args` (rayjob).
  Per-pod sizing, the pod image and pod-side env are the provisioning
  platform's inputs, not `optimize` flags — the cluster already exists by the
  time the optimizer runs.
- **PD disaggregation (infera):** `--pd-mode disaggregated`,
  `--pd-prefill-nodes` / `--pd-prefill-tp` / `--pd-prefill-ep` /
  `--pd-prefill-extra-args`, `--pd-decode-nodes` / `--pd-decode-tp` /
  `--pd-decode-ep` / `--pd-decode-extra-args`, `--pd-transfer-backend`,
  `--pd-ib-device`.
- **Phase toggles:** `--enable-roofline` / `--no-enable-roofline`,
  `--enable-conc-sweep` / `--no-enable-conc-sweep`, `--conc-sweep-concs`,
  `--conc-sweep-total-budget-sec` (whole-sweep budget), `--no-framework-agent`, `--no-framework-local-explore`, `--no-kernel`.
- **Agent models:** `--claude-model`, `--codex-model`.
- **Session / resume:** `--resume-from`, `--force-resume`, `--reset-state`,
  `--extend-hours`.
- **Quantization:** `--quantize`, `--quantize-scheme`.
- **Operator-defined specialist:** `--custom-specialist-prompt-file` (focus
  text, max 16 KiB) and `--custom-specialist-description` (one line, max 200
  chars), passed together; unset means no `custom_specialist`. Refused with
  `--no-framework-agent`, `--research-lane-capacity 0` or `--reset-state`.
  `HYPERLOOM_CUSTOM_SPECIALIST_ONLY=1` (settable by `export` or `--extra-env`;
  needs the pair; any other value than a boolean exits 2) makes FRAMEWORK_AGENT run only that specialist and the
  bench of its proposals and patches, then move on to KERNEL_AGENT when
  enabled, else SWEEP. All three are persisted in `state.json`; the env on a
  `--resume-from` turns only mode on, never off. Re-pass both flags on
  `--resume-from` only to replace the definition, or pass `--reset-state`
  alone to clear it.

Run `hyperloom optimize --help` for the exhaustive flag list.

---

## Accuracy gates

A candidate that clears the throughput bar must also hold accuracy before it is
kept. Grading runs only *after* the throughput bar is cleared, and reads the
score back from the run's own eval output, so a gate never costs an extra eval
and a regressing candidate never spends a verdict on itself.

In every lane a measured drop beyond the tolerance is a `REVERT`. A missing
verdict while a positive baseline accuracy is on record drops to
`NEEDS_REVIEW` — eval should have worked and didn't. No baseline accuracy at
all degrades to a throughput-only `KEEP` rather than blocking every candidate,
so eval-less environments still make progress. The accuracy eval always runs:
a baseline whose eval fails stops the run instead of falling back to a
throughput-only measurement.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_QUALITY_REF`<br>`HYPERLOOM_QUALITY_REF_WRITE` | Derived under the session dir | The scriptable quality gate's reference artifact: `_WRITE` establishes it on the baseline, the other compares against it on every later candidate. What the artifact holds is the workload's own business — xDiT stores an image, an operator-supplied `custom` workload stores whatever its script compares. Also emitted as `XDIT_QUALITY_REF` / `XDIT_QUALITY_REF_WRITE` for bench scripts written before the rename; either name is read, both are written. |
| `INFERENCE_OPTIMIZER`<br>`_REQUIRE_KERNEL`<br>`_ACCURACY` | On | Gates the `KEEP` for a kernel patch integrated by the kernel lane. Set to `0` / `false` / `no` / `off` to fall back to a throughput-only `KEEP`. Disable only when the eval lane is known-broken: this gate is what stops a faster-but-wrong kernel from being kept. |
| `MAGPIE_EVAL_LIMIT` | Unset (full task set) | Caps the number of eval problems (`lm_eval --limit`). Useful for smoke runs; see the noise caveat below before using it on a run whose `KEEP` decisions matter. |

The tolerance is deliberately **not** an env knob: `ACCURACY_THRESHOLD` in
`src/hyperloom/common/gain_math.py` is a fixed
`0.05`, that is, a candidate must stay within 5 percentage points of the recorded
baseline accuracy.

Note that the score is measured once per candidate, not averaged over repeats.
On a full GSM8K run (1319 problems) the 5-point tolerance sits several standard
errors away from the baseline, so single-run noise does not trip it. Capping the
eval with a small `MAGPIE_EVAL_LIMIT` shrinks that margin sharply and can make
the gate noise-sensitive — prefer the full task set whenever a gate decision
depends on the result.

### Eval generation bounds

InferenceX runs lm-eval with `max_tokens=min(16384, ctx-4096)`, so a sample that
does not converge spends that entire budget, and 1319 of them can consume the
whole baseline timeout. Every generation request is therefore capped, and the
terminators the model declares are supplied with it — lm-eval carries a single
`eos_string` and its concurrent request path does not send even that one, so a
model like Qwen3, which declares `eos_token_id` `[151645, 151643]`, would
otherwise run with no end-of-turn stop condition at all.

Both are applied inside the eval process rather than passed in, which is what
keeps them equal across the baseline and candidate arms. **That symmetry is the
whole point**: the gate compares a *difference* of two scores, so a bound or a
terminator that reaches only one arm biases the verdict instead of merely
limiting it. Prefer leaving these alone; if you do change one, change it for the
whole session rather than a single round.

Each run reports what it applied, to stderr as `HYPERLOOM_EVAL_BOUNDS_SUMMARY`
and to `hyperloom_eval_bounds.json` in the result dir, including how many
generations hit the ceiling. Check `truncated` there before concluding a score
is low for any other reason.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_EVAL_MAX_TOKENS` | `4096` | Per-request generation ceiling. Never raises a lower ceiling a task already asked for. `0` disables the cap and restores the full upstream budget — a degenerate model then costs the whole timeout again. An unparseable value falls back to the default rather than to "unbounded". |
| `HYPERLOOM_EVAL_DERIVE_STOP` | On | Whether to read the model's `generation_config.json` / `tokenizer_config.json` for its terminators. Resolution is cache-only and never downloads, so an uncached repo id simply derives nothing. Set to `0` / `false` / `no` / `off` to reproduce an upstream number exactly, or for a server that rejects `stop_token_ids` (vLLM and SGLang both accept it). |
| `HYPERLOOM_EVAL_STOP_STRINGS` | Unset (derived) | Explicit terminators, separated by ASCII unit separator `0x1f` — commas and newlines are themselves legitimate stop strings. Outranks the derived values; use it when a checkpoint's metadata is absent or wrong. |

Set explicit terminators like this, quoting so the separator is a real `0x1f`
byte:

```bash
export HYPERLOOM_EVAL_STOP_STRINGS=$'<|im_end|>\x1f<|endoftext|>'
```

Upstream keeps at most four stop strings, and the task's own `until` list is what
its answer extraction depends on, so that list is never displaced: an explicit
`HYPERLOOM_EVAL_STOP_STRINGS` goes first, the task's list next, and derived
terminators last. Derived token ids travel separately as `stop_token_ids`, which
has no such limit, so nothing is lost on a server that supports it.

---

## Experience KB

Every workspace runs a local Experience KB service. The global KB settings are
optional: unset, the workspace uses only its local KB. See
[Experience KB service](experience-kb.md).

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_KB_URL` | Written by `hyperloom-setup` (`http://127.0.0.1:<port>`, a port of the workspace's own in 20000–29999) | The workspace's local Experience service. A loopback URL is started by each optimize launch; unset disables Experience reads and writes. |
| `HYPERLOOM_KB_TOKEN` | Generated by `hyperloom-setup` | The local service's access token. |
| `HYPERLOOM_KB_HOME` | `$USER_DATA_PATH/experience-kb` | The local service's data home. One home serves one service at a time; runs on different hosts that share a `USER_DATA_PATH` each need their own. |
| `HYPERLOOM_GLOBAL_KB_URL` | Unset | A global Experience KB to share Experiences with; `experience_kb_service push` and `pull` sync with it. |
| `HYPERLOOM_GLOBAL_KB_TOKEN` | Unset | The global KB's access token; required with its URL. |
| `HYPERLOOM_KB_AUTO_PUSH` | Off | `1` pushes each run's Experiences to the global KB after they are written locally. |
| `LOCAL_KB_PLANNER_MODEL` | `CLAUDE_MODEL` | The model the Experience service plans reads with, through `ANTHROPIC_BASE_URL`. |
| `LOCAL_KB_PLANNER_TIMEOUT_SECONDS` | `20` | How long the Experience service waits on its planner before a read fails. A run waits 30 s per read, so a value above that makes every read fail. |
| `LOCAL_KB_PLANNER_MAX_OUTPUT_TOKENS` | `1400` | The most output tokens the Experience service's planner may spend on one read plan. A plan cut at this limit fails its read with `planner gateway truncated output at max_tokens`. |

---

## PrimaTune predictor

Set by `--primatune-endpoint` and `--primatune-mode`, or directly in the shell. See
[PrimaTune predictor](primatune-predictor.md).

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_PREDICTOR_ENDPOINT` | Unset (off) | Base URL of a PrimaTune service; the client posts to `<endpoint>/v1/predict`. |
| `HYPERLOOM_PREDICTOR_MODE` | `shadow` | `off`, `shadow` (ask and log, queue nothing) or `active` (queue the answer for benchmarking). An invalid value falls back to `shadow`. |
| `HYPERLOOM_PREDICTOR_TIMEOUT_SEC` | `900` | Per-request timeout in seconds, at least `1`. The request runs off the tick loop, so a long timeout costs no Coordinator time. |

---

## Kernel-opt backend selection

The following variables control the kernel optimization backend ladder.

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `KERNEL_OPT_BACKEND_ORDER`     | Unset (resolves to `geak`) | Selects the kernel-opt backend. An unset or blank value resolves to `geak` for every framework, ATOM included; no framework overrides it at launch. Bare-metal setup preserves nonempty choices (process env > `.env`) without filling or persisting a backend default. Existing `.env` values, including `geak`, are not automatically migrated; remove that assignment and unset the shell variable to use the shared default. Slurm launchers still export `${KERNEL_OPT_BACKEND_ORDER:-geak}`. **Only an exact, case-insensitive `forge` enables the per-kernel Forge backend** (`forge_explicitly_enabled` in `common/env.py`). Read once when a session is created and recorded as the session's `kernel_optimizer`; a resumed session keeps that backend and does not re-read the variable. Despite the historical name, a comma list is not parsed: `forge,geak` stays on `geak`, as does any other nonblank value, legacy alias, or payload override. |
| `HYPERLOOM_GEMM_SHAPE_CAPTURE` | `1`                           | Enables automatic runtime GEMM-shape capture for eligible single-node dense vLLM Forge tuning when no explicit shape input is available. Block-FP8 first reuses shapes from the TraceLens-selected steady-state trace of a successful Roofline with exactly matching model, workload, server arguments, environment, and backend controls. Missing or stale evidence triggers the same standard Roofline/ProfileExecutor/TraceLens steady-state pipeline as a fallback. Set to `0` to preserve the no-capture path. |
| `HYPERLOOM_GEMM_SHAPE_CAPTURE_TIMEOUT_SEC` | `1800`          | Timeout in seconds for the dense vLLM TunableOp recording benchmark. Block-FP8 fallback uses the standard Roofline/ProfileExecutor timeout. Values below `60` are clamped to `60`. |
| `HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT` | `5.0`                     | GPU-time share, in percent, at which a reusable hot kernel in the latest trace still counts as owed work until some backend has produced a patch for it (queued for integration, gated end to end, landed on the stack, or published by the rewrite controller). One gate reads it: KERNEL's work-pending check, which keeps the phase from settling while such a kernel exists, and the same list is printed as `untried_hot_kernels` in the mission summary and as the report's completeness annotation. It selects nothing for dispatch; the backends choose their own targets. `HYPERLOOM_KERNEL_OPT_GATE_TOP_N` (default `5`) caps how many ranked rows the gate considers. It was `10.0` until a 60-layer sparse-MoE decoder showed the assumption behind that number — that hot kernels concentrate — does not hold: nothing but a graph-launch wrapper reached double digits and the largest real operator sat at 9.47%. |
| `AITER_LOG_TUNED_CONFIG`       | `1` (set for every serving run) | Makes aiter log each tuned-config lookup it *hits*, not only the ones it misses. Two checks have no input without it: the GEMM demand list, which learns the shapes the runtime actually asks for (config-derived shapes covered 0.4% of them), and the apply verdict, which cannot tell "the tuned table was never read" from "it was read and did not help". A scan of 60 production logs found it set in none of them, so it is now injected by default. An operator value wins — set `0` to turn hit logging off, at the cost of both checks going inconclusive. Every miss already prints a line regardless of this setting; hit logging adds roughly one line per lookup that succeeds. |
| `HYPERLOOM_GEMM_PAIRED_PAIRS`  | `0` (off)                     | How many interleaved baseline/tuned pairs to re-measure before a GEMM tuning KEEP is reported as confirmed. One end-to-end measurement cannot separate a gain from drift on this fleet: three rounds of a single unchanged configuration spanned 58%, and one controlled repeat moved 16%. Each pair costs two extra benchmark rounds. When `0`, the gain is still promoted — it is the best number available — but recorded as an unpaired block comparison rather than presented as a paired one. |
| `GEAK_E2E_TIMEOUT_S` | `43200` | GEAK end-to-end timeout in seconds. Used as the fallback when no run deadline is set. A malformed value raises `EnvValueError`. |
| `GEAK_BUDGET_MARGIN_S` | `300` | Seconds subtracted from the remaining kill budget before passing the runner timeout to GEAK, so the runner self-stops before the hard subprocess kill. A malformed value raises `EnvValueError`. |
| `GEAK_MIN_RUN_S` | `600` | Minimum GEAK runner seconds required to attempt a run. If the remaining budget is below this, GEAK is skipped and the phase winds down to SWEEP. A malformed value raises `EnvValueError`. |

---

## Fusion lane

The fusion lane is Coordinator-owned and forge-only: it runs at KERNEL entry on
the forge branch, never as an agent request, and the default `geak` backend
returns before reaching it. Its gate needs a fusion-eligible framework
(`sglang`, `vllm` or `vllm-aiter`), a decode trace to discover from, and no
fusion that already succeeded this session.

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_SKIP_FUSION`        | Unset (lane enabled)          | Truthy (`1` / `true` / `yes` / `on`) disables the fusion lane outright, before any other gate is evaluated.                                                                                        |
| `FORGE_FUSION_TIMEOUT`         | `7200` (2h)                   | Wrapper timeout in seconds for one forge-fusion run. Unset or unparseable, the fusion lane's share of the KERNEL phase is used, else the default.                                                  |
| `FORGE_FUSION_MAX_TURNS`       | `100`                         | Agent turn cap handed to forge-fusion for one run.                                                                                                                                                 |
| `HYPERLOOM_FUSION_KEEP_PCT`    | `1.0`                         | Minimum throughput gain, in percent, that a queued fusion patch must reach to be KEPT when it is integrated. Every nominated patch is queued; the value is recorded on each queued record. A malformed value raises `EnvValueError`. |

---

## KERNEL lane behaviour

No variable toggles the behaviours below; `KERNEL_OPT_BACKEND_ORDER` only decides
which of them apply.

Globally, on every backend including the default `geak`:

- **`kernel_opt` and `gemm_tuning` are no longer model-issued.** Orchestration
  can neither propose either action nor request `run_optimization` /
  `run_gemm_tuning` of the kernel agent; both are refused as
  `phase_incompatible`. The Coordinator dispatches each lane once at KERNEL
  entry from a lane budget, the way the fusion lane already worked. `integrate` stays proposable, so draining the KEEP queue is still the
  model's job.

On the `forge` KERNEL path only (`KERNEL_OPT_BACKEND_ORDER=forge`; the default
`geak` backend hands to SWEEP before reaching any of these):

- **Fusion authors one independent sibling patch per confirmed recipe** rather
  than a single combined patch.
- **The fusion lane caps its recipe count** at 3 (`--max-recipes`), funded from
  its share of the phase. A session with no finite remaining time derives no
  share and stays uncapped.
- **GEMM reports carry a per-tuner `candidates[]` list**, so one tuner's table
  can be kept while another is reverted, instead of `recommended_env` landing
  all-or-nothing.
- **The fp8 → bf16 dense GEMM retry runs inside the same gemm call.** The
  follow-up subprocess Hyperloom used to launch is gone; the bf16 dense pass is
  selected up front as a fallback tuner and executes only when no earlier tuner
  produced a candidate, so a winning fp8 run pays nothing for it.
- **The gemm lane caps how many routed tuners run** (`--max-tuners`), priced on
  the router's own per-tuner estimates.
- **Kernel subprocess output is recorded as activity** for diagnostics. Activity
  does not prove useful progress and does not extend benchmark hard deadlines.
- **A `status: partial` invocation spec is declined on the rewrite route**
  rather than admitted because the file exists. A partial spec leaves the
  producer on its placeholder driver, which burns the whole budget and then
  exits non-zero.

---

## Collective candidate extraction

Communication operators are optimized by the rewrite controller like any other
operator. TraceLens still resolves them deterministically: it maps a mangled
NCCL/RCCL kernel name back to the framework device source that issued it and
publishes the row in `kernel_candidates.json` with
`candidate_source: nccl_summary`. Vendor RCCL/NCCL symbols are opaque binaries
and never resolve to an editable source, so they are never published.

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_COLLECTIVE_ALLOW_INFERRED_SHAPES` | Unset (disabled) | Truthy allows a source-resolved collective to borrow shapes from the trace's sole all-reduce workload family. The default rejects this inference because those shapes were not observed on that device symbol. |

---

## Kernel source resolution

A kernel candidate must resolve to a real source file before any backend can
rewrite it. Resolution runs as a ladder: curated dictionary, then the
trace-derived launcher frame, then a name grep. All three are deterministic and
require no configuration.

Every run writes `kernel_source_resolution.json` next to the candidate report.
It answers one question per hot kernel — which file defines it, and which tier
decided that — in a versioned schema (`schema_version`, currently `1.2.0`), so
consumers and triage read a contract rather than candidate internals.

---

## Single-node Ray execution

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_OPTIMIZER_RAY_EXEC` | Unset (`on` for single-node) | Controls whether single-node serving benchmarks and `needs_gpu` specialists run through Ray actors. When unset, single-node runs are routed through Ray-managed leases while multi-node stays on the multi-node backend. Set to `0` / `false` / `no` / `off` to force the local subprocess path, or `1` / `true` / `yes` / `on` to force Ray. |

---

## Codex (OpenAI) agent sandbox

Selects how a Codex agent session (TraceLens analysis and every future
Codex-based agent) is contained. The secure default is `workspace-write`.
Codex implements both contained presets with bubblewrap, so Hyperloom executes
a real namespace-and-mount capability probe before starting the SDK. Merely
finding a `bwrap` executable is insufficient: if the current kernel or
container prevents it from establishing the sandbox, `workspace-write` and
`read-only` fail closed before the app-server starts. There is no automatic
fallback to `bypass`.

`bypass` is an explicit operator opt-in. Set
`HYPERLOOM_CODEX_SANDBOX_MODE=bypass` when an external container or sandbox
already enforces the required isolation. Hyperloom does not create that
boundary. A confirmed bypass maps to Codex full access even when no writable
roots are declared, because the external sandbox is authoritative. Under the
contained modes, no writable roots remains `read-only`. Unknown modes fail
immediately.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_`<br>`CODEX_SANDBOX_MODE` | `workspace-write` | `workspace-write` restricts writes to the session directory plus declared output roots; `read-only` forbids writes; `bypass` selects Codex full access when an external sandbox already enforces isolation. |

---

## Single-node Ray GPU scheduling

These variables tune the single-node Ray execution path (active when
`INFERENCE_OPTIMIZER_RAY_EXEC=1` and `--nodes=1`). They have no effect on
multi-node runs or when the Ray backend is disabled.

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_OPTIMIZER_RAY_GPU_PENDING_LIMIT` | `4` | Maximum number of GPU specialists that can be simultaneously in-flight (pending Ray scheduling + running) on the single-node Ray path. Ray still serializes execution on the physical GPU(s) using `num_gpus`; this limit caps how many actors can queue behind the current one. Floored at `1`. **Reduce to `1` or `2` when GPU memory or per-process overhead is a concern** (each queued actor holds a Ray worker slot even while it waits). |
| `INFERENCE_OPTIMIZER_RAY_ROUND_WAIT_SEC` | Unset (the round's own timeout + 3600s) | Wall-clock ceiling on blocking for one Ray round to return. Without it a round Ray never schedules parks the waiting thread forever, leaving its task `running` and its lane lease never released — the 2026-09-21 stall. Set a number of seconds to override; `<= 0` disables the ceiling, which is also what a round with no cap of its own gets. A non-finite value (`nan`, `inf`) is **rejected**, not tolerated, because `nan > 0` is false and would silently switch the ceiling off. On timeout the round is abandoned and its actor is kept alive on purpose, so its GPUs stay reserved rather than being handed back while its server tree may live. |
| `INFERENCE_OPTIMIZER_RAY_SERVING_PRIORITY` | On | When enabled (default), the dispatcher defers admitting new GPU research specialists while a serving benchmark holds the whole-machine `serving_slot`, preventing research work from starving serving. The slot is probed immediately before each specialist is admitted so a serving start that races the dispatch pass is caught. Set to `0`, `false`, `no`, or `off` to disable. |

---

## Compute partitioning (AMD)

An MI300-series card can be split into independent partitions (`SPX`, `DPX`,
`QPX`, `CPX`). Splitting it trades per-request latency for aggregate throughput,
so a partitioned measurement is not comparable with a whole-card one.

**The optimizer does not set the mode.** Changing it is privileged, disruptive to
every process holding a GPU context, and not something an optimization loop
should be doing between benchmark rounds. The card is put in its mode before
launch — by the operator or the provisioning platform — and `optimize` only
observes what it is in, refuses a session that cannot work in that shape, and
hands the shape to the benchmark entrypoint that places work across partitions.

Two CLI flags configure this, both optional:

- `--compute-partition-mode {SPX,DPX,QPX,CPX}` **asserts** the mode the card is
  already in. It is a check, not a request: if the card is in a different mode
  the session is refused rather than silently measuring the wrong topology. If
  the card cannot be read at all, a declared mode is also a refusal — the flag
  exists to catch an external set that did not take, and an unverifiable
  assertion is not a satisfied one.
- `--streams-per-partition N` (default `2`) is how many concurrent streams the
  benchmark places on each partition. One stream leaves each partition idle
  through the fixed per-pass cost; beyond two, on the workloads measured so far,
  only queueing is added. A value below `1` is refused rather than replaced by
  the default, so `0` is a usage error instead of a silent `2`.

At launch the per-stream HBM footprint is checked against one partition's
memory. A workload that provably will not fit is refused in milliseconds instead
of failing out of memory hours in. The footprint is the checkpoint's weight bytes
— a lower bound, since each stream holds its own copy of the weights, which is
why a "does not fit" verdict from it is trustworthy and a "fits" verdict proves
nothing. When the checkpoint cannot be sized the session runs with a warning.

The check only applies where streams will actually share a partition: with a
serving framework and no partition flags, the shape is recorded and nothing is
refused, because nothing in that session places work per partition and, since
whole cards enumerate before partitions, its benchmark may not even land on one.

Multi-node sessions (`--nodes >= 2`) record no shape at all. The card this
process can read is not the card the benchmark runs on, and a shape recorded from
the wrong node is the exact mislabelling this feature exists to prevent. A
declared mode there is a usage error rather than a silently unchecked assertion.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_PARTITION_GPU` | `0` | Which GPU's partition state describes this session. A value that is not a GPU id falls back to `0` with a warning rather than silently — the fallback reads a different card, and every number the session files afterwards would carry that card's shape. |

### Runtime hand-off

Published once at launch by the CLI and read by the benchmark entrypoint. Do not
set these by hand: they are overwritten at every launch, and clearing them first
is what stops a second session in the same shell from inheriting a shape that
was not asked for.

The first three describe the card and are published for every session on a
readable card, because `platform_fingerprint()` reads them back from here — it
runs on the crash path, where spawning `amd-smi` is not acceptable. The last two
are instructions to a benchmark that places work on each partition, so they are
published only when one will.

| Variable | Published | Description |
|----------|-----------|-------------|
| `HYPERLOOM_PARTITION_MODE` | Always | The observed mode. Also recorded in the platform fingerprint, so a result is never filed under a topology it was not measured on. |
| `HYPERLOOM_PARTITION_COUNT` | Always | Partitions the card presents in that mode. |
| `HYPERLOOM_PARTITION_CU` | Always | Compute units per partition. The entrypoint selects partition devices by matching this exactly — HIP enumerates whole cards before partitions, so an index list computed at launch would be wrong in the one case that matters, and wrong invisibly. |
| `HYPERLOOM_PARTITION_STREAMS_PER_PARTITION` | Fan-out only | Streams to place on each partition. |
| `HYPERLOOM_PARTITION_TOTAL_STREAMS` | Fan-out only | `count × streams`; the total concurrency the entrypoint should drive if it fans out across every partition. |

Only scriptable frameworks (`xdit`, `custom`) place work per partition. A serving
session is handed no fan-out instruction rather than a concurrency nothing will
drive, and passing the flags with one warns. Its shape is still recorded in the
report and the fingerprint — that is provenance, not a hand-off — and the report
states plainly that the figure cannot be read as an aggregate.

### Choosing the mode: `scripts/partition_mode_sweep.py`

`optimize` treats the mode as fixed and asserts it. Deciding *which* mode to be
in is a separate job, done before the session, by
`python3 scripts/partition_mode_sweep.py`. It sets each mode on one card in
turn, runs the same benchmark on every partition that mode creates, sums the
result, and restores the card's entry mode on the way out.

```bash
# what it would do, nothing set
python3 scripts/partition_mode_sweep.py --benchmark-config bench.yaml --dry-run

# sweep every mode the card reports, skipping any that cannot hold the workload
python3 scripts/partition_mode_sweep.py \
    --benchmark-config bench.yaml --output-dir /shared/sweep \
    --per-stream-gib 20.7 --sudo
```

The fan-out is the point rather than a detail: a benchmark that loads one
partition and ignores the rest measures a fraction of the card, which makes
`CPX` look eight times worse than it is. The sweep therefore launches every
partition at once, and reports a mode only when all of its partitions returned
a measurement — a mode with six of eight reporting is unmeasured, not slow.

It publishes the same `HYPERLOOM_PARTITION_*` variables as a session, so a
benchmark entrypoint written against the table above works unchanged under
either. It pins each process with `ROCR_VISIBLE_DEVICES` and removes any
inherited `HIP_VISIBLE_DEVICES`, because two masks apply in sequence and the
second indexes into the first.

The privileged `amd-smi set` lives here and nowhere else. An operator-run script
between benchmarks is a reasonable place for a card-wide mutation that evicts
every GPU context; an optimization loop that also runs agent-authored code is
not. Before setting anything it refuses if a process holds a context on the card
being swept — and only that card, since no other card is repartitioned. A
neighbour's benchmark on a shared node is not a reason to stop. If `amd-smi`
reports its process list in a shape the script cannot read, that is also a
refusal rather than an assumption that the card is idle: `--allow-busy` is the
way past both, and `--dry-run` never asks.

Exit codes: `0` swept, `1` nothing measurable, `2` refused before anything
changed, `3` swept but the card could not be restored to its entry mode, `4`
stopped on an error it does not model. Every path out of a started sweep goes
through the restore and the report, so a mode that fails unexpectedly costs its
own result and nothing else.

---

## Multi-node / prefill-decode (PD)

Use CLI flags for multi-node topology and prefill-decode configuration:

`--nodes`, `--mn-backend`, `--gpus-per-node`, `--tp`, `--ep`,
`--pd-mode`, `--pd-prefill-nodes`, `--pd-decode-nodes`, `--pd-prefill-tp`,
`--pd-decode-tp`, `--pd-transfer-backend`, and `--pd-ib-device`.

`optimize` never creates or releases a multi-node cluster. The provisioning
platform (for example, Primus-Claw) creates the RayJob or InferaDeployment and hands it
over through the variables below; without a hand-off `--nodes >= 2` exits 2.

### Cluster hand-off variables

`TOPOLOGY_HANDOFF_FILE` is read first. The file carries `backend` (`rayjob` or
`infera`), `nodes`, `gpus_per_node`, `service_url`, and the block of that
backend: `ray` (`head_ip`, `dashboard_token_file`) or `infera` (`ssh_key_file`,
`ssh_port_base`, `known_hosts_file`, `prefill_ips`, `decode_ips`, `worker_ips`).
Credentials are file paths; the Ray Dashboard token is read from its file. A set
variable naming a file that is unreadable, not valid JSON, of an unsupported
schema major, or missing a required field is an error that names the file and
the problem; Hyperloom does not fall back to the legacy variables. If legacy
variables are also set and differ from the file, the file wins and a warning
names the differing fields. The exceptions are `--nodes`, `--gpus-per-node` and
`--mn-backend`: a launch flag always wins over the file, so a `--nodes 1` run is
single-node and a `--nodes 3` run is stored as three nodes whatever the file says.

Without `TOPOLOGY_HANDOFF_FILE`, `HYPERLOOM_MN_EXT_SERVICE_URL` is the only variable
that tells the optimizer a cluster is ready; the rest describe how to reach it.

| Variable | Backend | Required | Description |
|----------|---------|----------|-------------|
| `TOPOLOGY_HANDOFF_FILE` | both | No | Path to a TopologyHandoff v1 JSON file (`"schema": "topology-handoff/v1"`) the platform wrote. When set (non-empty) it is the hand-off and the `HYPERLOOM_MN_EXT_*` and `INFERENCE_OPTIMIZER_MN_BACKEND` variables are not read for the fields it carries. The launch flags win over it: `--mn-backend` over `backend`, `--gpus-per-node` over `gpus_per_node`, and `--nodes` over `nodes` (`optimize` exports `INFERENCE_OPTIMIZER_NODES` and `INFERENCE_OPTIMIZER_GPUS_PER_NODE` from the flags, and a value set there is never replaced by the file's); the file fills in `backend` and `gpus_per_node` when their flag is omitted. When unset, the variables below apply unchanged. |
| `HYPERLOOM_MN_EXT_SERVICE_URL` | both | **yes** | Benchmark frontend URL (`http(s)://…`; infera frontend typically `:8000`). Its presence triggers external mode. |
| `HYPERLOOM_MN_EXT_SSH_KEY` | infera | **yes** | Private SSH key already authorized on the pods (the platform installs the public half at create time). |
| `HYPERLOOM_MN_EXT_PREFILL_IPS` / `_DECODE_IPS` | infera | PD | Prefill / decode pod IPs (comma-separated) for PD-disaggregated runs. |
| `HYPERLOOM_MN_EXT_WORKER_IPS` | infera | aggregated | Worker pod IPs (comma-separated) for aggregated (non-PD) runs. At least one of `_PREFILL_IPS` / `_DECODE_IPS` / `_WORKER_IPS` is required. |
| `HYPERLOOM_MN_EXT_SSH_PORT` | infera | No (default `2233`) | SSH base port; decode role is offset `+10`. |
| `HYPERLOOM_MN_EXT_SSH_KNOWN_HOSTS` | infera | No | `known_hosts` path; else a relaxed host-key check is used. |
| `HYPERLOOM_MN_EXT_HEAD_IP` | rayjob | No (recommended) | Ray head IP (Dashboard `:8265`, GCS `:6379`). Enables per-round restarts; omit for benchmark-only. |
| `HYPERLOOM_MN_EXT_RAY_DASHBOARD_TOKEN` | rayjob | No | Ray Dashboard auth token, only if the dashboard is authenticated. |

Infera external mode requires `HYPERLOOM_MN_EXT_SSH_KEY` plus at least one
`*_IPS` list, or the run fails fast at startup. RayJob external mode ignores
the SSH / IP vars and uses `HYPERLOOM_MN_EXT_HEAD_IP` for restarts.

---

## Quantization prelude

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_QUANTIZE_ENABLED` | Unset | Primary switch (`1` to enable) for the AMD Quark PTQ quantization prelude driven by `--quantize` / `--quantize-scheme`. |
| `QUARK_ROOT` | Unset | AMD Quark checkout used by the quantization agent. Set this explicitly when quantization is enabled. |

---

## Enablement admission

Enablement is **not** configured through the environment. Both self-heal lanes
are admitted by the `--enablement {off,launch,eval,all}` CLI flag, which defaults
to `all`:

- `launch` — a baseline that cannot boot routes into patch authoring.
- `eval` — a baseline that boots and measures throughput but fails its accuracy
  eval (crashes, produces no result, or scores below the floor) routes into
  patch authoring. Single-node only; multi-node keeps the strict stop.
- `all` (default) — both lanes.
- `off` — neither lane engages, and a baseline that keeps failing terminates the
  run with `stop_reason='baseline_failed'` instead of opening an authoring loop.

The eval trigger and the enablement KEEP gate both judge a score against the
fixed constant `hyperloom.common.gain_math.DEFAULT_ENABLEMENT_ACCURACY_FLOOR` (`0.5`):
`score >= floor` passes, and a score that is not finite or not above zero fails.
The lanes differ only when a candidate's eval produced no score: an `eval`-lane
candidate then fails, while a `launch`-lane candidate that serves is kept as
provisional, with its correctness not verified.

---

## Benchmark deadlines and lifecycle

Each actual benchmark subprocess spawn uses the same finite, positive limits,
including baseline, explore, sweep, and rebench measurements:

| Variable | Default | Description |
|---|---|---|
| `INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC` | `7800`; under `HYPERLOOM_AGENTIC_BACKEND=mlperf`, sized to the trajectory count, and under native AgentX sized to the round's `CONC` (see [MLPerf agentic backend](#mlperf-agentic-backend)) | Hard wall-clock seconds per actual spawn, including server boot and accuracy evaluation. Output cannot extend it. An explicit value always wins. |
| `INFERENCE_OPTIMIZER_BENCHMARK_SILENCE_TIMEOUT_SEC` | `600` | Output-silence seconds after real server readiness. Not armed before readiness or for server-less scriptable workloads. |

Session `--max-hours` and cancellation apply independently of benchmark limits.
Task and lease age do not expire work; admission and phase budgets, ordinary
release, and dead-PID cleanup remain. Session exhaustion is cooperative, not a
guarantee that a frozen Coordinator will terminate. There is no out-of-band
supervision or automatic resume. SIGHUP uses the normal terminal drain; an
operator may later explicitly use `--resume-from` on the same session.
Historical stop reasons remain interpretable and are not restart requests.

`PYTHONUNBUFFERED=1` affects Python output only, not shell/native buffering or
upstream subprocess capture. Active logs do not prove useful progress. The
Magpie streaming fix is local-only and unpublished; the installed dependency pin
must not be assumed to include it, and this policy is not evidence of a validated
end-to-end run.

Profile, KernelForge, GEAK, LLM-call, and SGLang's own watchdog budgets are separate
from these benchmark limits and retain their existing contracts.

## Reactor budgets

| Variable | Default | Description |
|---|---|---|
| `INFERENCE_OPTIMIZER_REACTOR_TURN_TIMEOUT_SEC` | `1800` | Total wall-clock limit for each reactor stage, including backend startup, streamed output, retries, backoff, and cleanup. This is independent of backend `*_CALL_TIMEOUT_SEC` settings: for streamed Claude turns those settings bound idle time between SDK messages, and activity resets that idle timer. Reaching this total limit cancels the stage and records a crash; a shorter remaining session bound still ends the stage without recording a crash. A malformed value raises `EnvValueError`; a non-positive or non-finite value falls back to the default. |
| `INFERENCE_OPTIMIZER_CLAUDE_TURN_TIMEOUT_SEC` | `1500` | Total wall-clock limit for one Claude Agent SDK call made by the orchestration backend (reactor turns, including the SWEEP cycle-directive handoff), covering every retry and the backoff between them; tearing the CLI down afterwards adds at most 30 s. Unlike the idle bound of `INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC`, the CLI's own API-retry messages do not extend it. Past it the call is cancelled, the CLI and every process it started are killed, and the turn fails as a backend error the loop retries on its next tick. |
| `INFERENCE_OPTIMIZER_DEEPSEEK_CALL_TIMEOUT_SEC` | `300` | Wall-clock limit for one DeepSeek Harness turn made by the orchestration backend. Past it the harness runtime is closed and the next turn starts a new runtime and a new conversation. |

## TraceLens analysis route

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_TRACE_ANALYSIS_ROUTE` | `agent` | `agent` runs the TraceLens skill under a model. `non-llm` runs TraceLens without a model and writes `analysis.json` from the TraceLens report; it shares the gates, candidates and reports with `agent` but adds no collective candidates and writes no prose, and the report's trace-quality warnings appear in `trace_health_warnings`. A `trace_analyze` request's `analysis_route` key overrides it. Any other value fails the request. |

## TraceLens analysis budgets

On the Claude backend, the TraceLens analysis session bounds each wait for the
next SDK message: the in-process SDK has no read timeout, so a stalled gateway
would otherwise hold the session. Both limits measure inactivity, not total
time. The Codex runner is one-shot and exposes no stream to bound.

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_TRACELENS_STREAM_IDLE_TIMEOUT_SEC` | `300` | Seconds to wait for the next SDK message while no tool call is running. Values below `30` are raised to `30`; `0` or a negative value removes the bound. |
| `HYPERLOOM_TRACELENS_TOOL_IDLE_TIMEOUT_SEC` | `3600` | The same bound while a tool call is running, when the SDK is silent by design: a TraceLens report over a large trace can run for many minutes. When unset it is never tighter than the stream bound. Values below `30` are raised to `30`; `0` or a negative value removes the bound while a tool runs. |

---

## Framework / source-tree discovery

The following variables configure framework source discovery and path overrides.

| Variable                                          | Default                                                                | Description                                                                                                                                            |
|---------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `INFERENCE_`<br>`OPTIMIZER_`<br>`FRAMEWORK_`<br>`SOURCE_ROOTS`      | Union with `/sgl-workspace`<br>`/{aiter,sglang`<br>`,vllm}`                        | Colon-separated supplement to the search roots returned by `hyperloom.inference_optimizer.framework_paths.resolve_kernel_search_roots`, used for source search and flag discovery. Populated automatically by `src/hyperloom/inference_optimizer/assets/install.sh`'s `_probe_framework_source_roots` step (using `hyperloom.inference_optimizer.framework_paths.probe_framework_source_roots_for_env`).   |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_RESCUE_PATHS`                | Unset                                                                  | Colon-separated list of extra directories the harvest step scans for stray `result.json` files written outside the session dir (InferenceX-native scripts that hardcode `--result-dir`). |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_AITER_JIT_DIR`               | Aiter default                                                          | Override the aiter just-in-time (JIT) cache root. See [Targeted builds (Rung 5)](#targeted-builds-rung-5).  |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_STRICT_PATHS`                | `1` when CLI bootstraps                                                | When `1`, missing path env raises instead of falling back to discovery. Set by the CLI at session start; do not override unless debugging.              |
| `HYPERLOOM_`<br>`SGLANG_PA`<br>`TCH_EXACT`<br>`_VERSIONS`           | Unset                                                                  | Pin the sglang server-patch step to specific upstream versions; advanced compatibility option.                                                          |
| `HYPERLOOM_`<br>`ENABLE`<br>`_PATCH`                          | `1`                                                                    | Set to `0` to skip the in-place server patch step (useful when the upstream is already pre-patched).                                                    |
| `HYPERLOOM_`<br>`SKIP_FRAME`<br>`WORK_CHECK`                  | Unset (check enabled)                                                  | Truthy skips the `optimize` preflight gate that requires the selected serving framework to be importable and a ROCm build. Last resort: when the server runs elsewhere, set `BENCHMARK_BASE_URL` instead, which exempts the check and configures the supported path. The gate already stays out of the way for `xdit`/`custom` (server-less) and external multi-node. |
| `AITER_REF` | Unset | Optional bare-metal AITER install pin. When unset, the installer selects the newest tag compatible with the installed torch/triton stack. |
| `ATOM_REF` | `fe1099b15ddc52e6873e1932935f722864cdeee0` | Full commit SHA that `--install-framework atom` checks out from `ATOM_REPO` (default `https://github.com/ROCm/ATOM.git`) into `ATOM_ROOT` (default `<deps root>/atom`). The default is the commit `rocm/atom-dev:v0.1.7-rc0` was built from. |

---

## Targeted builds (Rung 5)

These variables control the Rung-5 off-loop compiled-component acquisition
step (AITER FP4/MLA/NSA kernels, sgl-kernel, and vLLM from source).  All are
optional; defaults are safe for standard single-node deployments.

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_ENABLEMENT_DISABLE_TARGETED_BUILD` | Unset (`0`) | Set to `1` to completely disable Rung-5 auto-escalation.  When set, compiled-gap failures proceed to the stall gate without attempting a build.  Useful when the compile toolchain is unavailable or the session budget is too tight. |
| `INFERENCE_`<br>`OPTIMIZER_`<br>`AITER_JIT_DIR` | Aiter default | Per-attempt override set automatically to `<attempt_root>/aiter_jit` by each targeted build.  Override manually only when you need the global JIT cache to point at a pre-built location; leaving it unset lets each build use its own isolated directory. |
| `PYTORCH_ROCM_ARCH` | Detected | Explicit GPU target architecture (e.g. `gfx942`, `gfx950`) injected into each compile.  Set automatically from the session `--gpu-type`; operator-override applies to bare-metal installs outside the session.  **Compile target only — it does not participate in architecture detection.**  It names the archs a wheel is *built* for, not the installed device, so provenance ignores it entirely and resolves `gfx_arch` from `HYPERLOOM_GFX_ARCH`, then `--gpu-type`, then `rocminfo`. |
| `MAX_JOBS` | `8` | Parallelism cap for cmake/hipcc compile steps inside a targeted build.  Reduce on memory-constrained nodes (`MAX_JOBS=4` for a 64 GB compile node).  The default `8` is conservative enough for MI300X/MI355X nodes with 512 GB+. |
| `HYPERLOOM_`<br>`FRAMEWORK_PYTHON` | Unset | Explicit interpreter that launches the server for a from-source build (the venv Python the artifact was compiled against).  Set automatically from `FrameworkRuntime.runtime_python_exe` through `apply_runtime_override` into the per-variant YAML `benchmark.envs`.  Both backends export that mapping to the server env; the bypass backend additionally uses this value as the `python -m` interpreter.  Operators normally do not set this by hand. |

---

## Security compatibility switches

These switches keep production-compatible behavior by default while still
allowing operators to turn off credential/env persistence in hardened
deployments.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV` | Unset (`1`) | Specialist subprocesses inherit the limited provider credential set by default: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_CUSTOM_HEADERS`, `CLAUDE_CODE_OAUTH_TOKEN`, `OPENAI_API_KEY`, `OPENAI_CUSTOM_HEADERS`, and AWS Bedrock credential/config vars. Set to `0` only when the `claude` CLI is authenticated through its own config and env credentials must be suppressed. Unrelated secrets such as GitHub and KB tokens remain blocked. |
| `HYPERLOOM_SPECIALIST_USER` | Unset (no change) | Account name to run specialist subprocesses as (the `claude`, `codex` and `dsh` CLIs), so they cannot read the app's files, `/proc/<pid>/environ` or key files. Needs a Hyperloom process that can start a child as another user and later stop it: root, or all of `CAP_SETUID`, `CAP_SETGID` and `CAP_KILL` (`CAP_KILL` is what lets the deadline, stale-heartbeat, done-file and cancel reaps signal the child; without it none of them could be confirmed). The launch fails with a clear error, and no specialist runs, if one of those is missing, the account does not exist, is uid 0, is the app's own user, or belongs to group 0; so does a GPU specialist on the Ray actor path, which cannot be demoted. The child gets that account's `HOME`, `USER` and `LOGNAME`, supplementary groups, and `umask 007` (group full access, others none); the Hyperloom process's own umask is left alone, and each launch re-shares the workspace and worktree, so files the app rebuilt in between become group-writable again. The child starts through `setpriv` (util-linux, required), which clears its inheritable and ambient capabilities and sets `no_new_privs`, so a non-root app that holds `CAP_SETUID` as an ambient capability does not hand it to the child. Each task's workspace and worktree stay owned by the app but get the account's group with group read-write (setgid directories); the app must be root or a member of that group to do so, and the account must be able to traverse every parent directory of the session directory (the app never changes the modes of those parents), read the checkout the worktree is branched from, and execute the agent CLI. The worktree's `.git` file and its admin directory under the repository's `.git/worktrees` are shared read-only (git inside the worktree works for status, diff and log, not for writes), and a copy of the `--mcp-config` file is staged in the workspace. The app reads and writes the paths the child can rewrite (`specialist_done.json`, `patches/`, the harvested patch, the Codex config, `prompt.md`, `process.log` and its redaction copy, the `dsh` home and session logs) without following symlinks. The multi-branch specialist launcher (`LocalAgentRunner`) demotes the same way, sharing the session's working directory as the worktree; there an unusable account fails the launch with `AgentConfigError`. |
| `HL_ALLOW_DANGEROUS_AGENT_PERMISSIONS` | Unset (`0`) | Slurm carrier only. Set to `1` only in dedicated internal containers to re-enable legacy Claude/Codex approval and sandbox bypass flags. |

---

## Critic / knowledge base (KB)

The following variables configure the Critic and knowledge base components.

| Variable                              | Default                | Description                                                                                                                          |
|---------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `KNOWLEDGE_STORE_MODE`                | `local`                | Exclusive Recipe backend: `local` or `remote`. Ambient KB Store or GBrain credentials do not select remote mode. |
| `KNOWLEDGE_LOCAL_ROOT`                | `$USER_DATA_PATH/knowledge`, otherwise `~/.cache/hyperloom/knowledge` | Local Recipe/KG root. It is not used for Recipe data in remote mode. |
| `INFERENCE_OPTIMIZER_`<br>`FA_KB_PATH` | `$USER_DATA_PATH/framework-kb`, otherwise `/workspace/hyperloom/framework-kb` | Framework KB root, holding the lessons ledger the FRAMEWORK phase reads and writes. The orchestrator's KB reader and writeback both resolve through it. |
| `KB_STORE_URL`                        | Local Recipe mode: `https://global.primus-safe.amd.com/knowledge-base`; remote mode: unset | Unified KB Service endpoint. Local Recipe mode uses the default only for co-hosted PR Monitor access; IR-3 disables PR Monitor without disabling local Recipe storage when the endpoint is unreachable. Remote Recipe mode requires an explicit value, selects the current Recipe View, replays its Config, Patch, and Kernel columns, then writes one final session at CLOSE. PR Monitor is available at `${KB_STORE_URL}/pr-monitor/v1` (REST/IR-3/Framework/KernelForge) and `${KB_STORE_URL}/pr-monitor/mcp/` (specialist MCP). |
| `KB_STORE_TOKEN`                      | Unset                  | KB Store bearer token. Required when `KNOWLEDGE_STORE_MODE=remote`; transport failures during the final write are non-fatal. |
| `KB_DRAFT_DIR`                        | Runtime-generated      | Internal remote-mode handoff where each Recipe column (`config`, `patch`, `kernel`) stages its knowledge and files before CLOSE merges them. Hyperloom creates and exports it; operators must not set it. A Recipe cannot be built without it. |
| `KB_WARM_START_DIR`                   | Runtime-generated      | Internal remote-mode handoff pointing agents at the downloaded `recipe.json + files/` selected Recipe View. Hyperloom creates and exports it; operators must not set it. |
| `CRITIC_AGENT_ROOT`                   | Derived from `REPO_ROOT` | Override location of the critic-agent runtime.                                                                                    |
| `CRITIC_AGENT_`<br>`MAX_COMPLETION_TOKENS` | `32000`           | Output-token cap for one critic review call. A reply cut off at the cap is retried once at twice this value and then fails the turn, so the cap is a ceiling rather than a budget: unused headroom is never billed, while a truncated reply bills the whole call and yields nothing. Lower it for a model whose own output limit is smaller. A non-positive or unparseable value logs a warning and falls back to the default. |

---

## Session / observability hand-off

These are read by `src/hyperloom/inference_optimizer/session/manifest.py` and the `src/hyperloom/inference_optimizer/breakdown/collectors/`
package to populate `session_breakdown.json` for downstream consumers.

| Variable          | Description                                                                                |
|-------------------|--------------------------------------------------------------------------------------------|
| `CLAW_SESSION_ID` | Hosted SaFE / Claw session id, written to `session.claw_session_id` in `session_breakdown.json`. Set by the Primus-Claw sandbox; unset for local runs. |
| `SANDBOX_USER_ID` | Hosted SaFE / Claw user id, written to `session.sandbox_user_id`. Set by Primus-Claw; unset for local runs.                                            |
| `HYPERLOOM_LANGFUSE_ENABLE` | Primary switch (default **off**) for live Langfuse trace push. See details below. |
| `HYPERLOOM_LLM_ATTRIBUTION` | Names the gateway whose attribution headers every LLM call should carry (default **unset**, emitting none). See details below. |

**`HYPERLOOM_LANGFUSE_ENABLE`** details:

Primary switch (default **off**) for live Langfuse trace push.

- **SDK install**: when this flag is on, `src/hyperloom/inference_optimizer/assets/install.sh` auto-installs the optional `langfuse` SDK on demand and skips it entirely when off — no separate `pip install '...[trace]'` is required.
- **Live push**: when set to `1/true/yes/on` and the three `LANGFUSE_*` credentials are present, every in-process LLM call is mirrored into Langfuse while the run is live. A session-end flush backfills calls still missing their token or text half, the recipe-KB, specialist-intel, Forge-step, GEMM-tuning and trajectory logs, and KEEP/REVERT decision Scores.
- **Local ledger**: `reports/trace/*.jsonl` is always written regardless of this flag. If the SDK is unavailable, live push degrades to a no-op.
- **Correlation**: the Langfuse trace ID and `session_id` grouping are derived from `claw_session_id` (env `CLAW_SESSION_ID`), falling back to the internal session ID for standalone runs. Live push and the offline `backfill_langfuse` CLI collapse onto one trace per Primus-Claw session.
- **Span layout**: `trace → phase span (PRELUDE/ENABLEMENT/FRAMEWORK_AGENT/KERNEL_AGENT/SWEEP/…) → agent span (component: orchestration/kernel/specialist/critic/geak/forge/…) → Generation`. Each KEEP/REVERT/`gain_pct` Score attaches to the agent span that produced the decision, with a trace-level fallback when no matching span exists.
- **Recipe-KB spans**: under the `recipe_kb` agent span, local reads/writes and remote KB Store publish attempts are recorded from `runtime/recipe_snapshot/.audit.jsonl`. Read spans use `kb:recipe_snapshot:<method>`; write spans use `kb:recipe_write:<generator>`, where the generator distinguishes normal `close` from `t4_fallback`. Remote rows report `written`, `skipped`, or `error` without recording credentials or payload bodies.
- **Receipt**: every session records a `langfuse` section in `session_breakdown.json` (and `reports/trace/langfuse_receipt.json`) noting:
  - Whether push was enabled (or the `disabled_reason`)
  - The redacted connection config (host and key-presence booleans — never the keys themselves)
  - The derived `trace_id` and `session_id`
  - How many generations, scores, and spans were sent

  This lets an operator confirm post-hoc whether a run reached Langfuse.

### Langfuse and artifact-package — security and known limitations

* **Sensitive data surface**: When live push is on, `conversations.jsonl`
  (and Langfuse Generations) carry full prompt/response text. `redact_secret_values`
  scrubs common token shapes (Bearer, `sk-`/`pk-`, GitHub tokens, some
  `KEY=value`) but is not a complete data loss prevention (DLP) filter — bare keys without a
  recognizable prefix (for example, raw AWS `AKIA…`) can slip through. The artifact
  packager also copies `reports/trace/*.jsonl` and, with the loose mode on by
  default (`HYPERLOOM_SESSION_PACKAGE_LOOSE`), drops them under the package
  destination `HYPERLOOM_SESSION_PACKAGE_DEST`: `/workspace` for the Claw sync when
  it is unset, and `<session dir>/archive/` (mode 0700, on the shared storage) in
  a hosted run, where the entry sets it whatever the host passed. If a session might
  contain customer code or secrets, define an explicit retention + access-control
  policy for both the Langfuse project and the package destination, and consider
  disabling live push or loose packaging for those runs.
* **`live push` + `backfill_langfuse` overlap**: Both derive the same
  `trace_id` from `claw_session_id`, so running the offline backfill *after* a
  live run re-emits the generations and decision scores onto the same trace and can
  duplicate observations. Use one path per session, or treat backfill as a
  recovery tool only when live push did not run.
* **`flush_session` is idempotent, and retries only what failed**: the
  session-end reconcile runs as named steps (leftover halves,
  recipe-KB audit, specialist intel, forge steps, GEMM tuning, trajectory,
  decision scores, span close, final SDK flush). Every call runs every step,
  and each step sends only what is still owed, so a duplicated CLOSE step
  won't double-push, a row that failed is retried by the next call, and a
  later call (the shutdown flush after CLOSE) pushes whatever was recorded
  since the earlier one. `counts_final` is `true` when the latest call
  completed every step — a `false` there means the push is still incomplete,
  not that the session was short-lived. Every leg of a resumed session
  (a crash plus a `--resume`, or two shutdown paths racing) reports into the
  same trace. `rows_sent` maps each
  append-only log, by its path under the session directory, to how many of its
  rows were sent: the `trajectory/*.jsonl` shards, the
  recipe-KB audit, specialist intel, forge steps and GEMM tuning logs. A call
  sends only the rows past that count, advancing it one row at a time and
  stopping at the first row it could not send, so that row is retried by the
  next call or the next leg. A v1.1.3 receipt carries the trajectory shard
  counts as `trajectory_rows_sent`, which is still read. `decision_trace.jsonl` is rewritten ts-sorted on
  every export, so decision scores are tracked by the `decision_id` its writer
  stamps on each row instead (`decisions_sent`). Both record what was handed to
  the Langfuse SDK: its flush does not report a failed export, so a row lost in
  export is not re-pushed. The `*_read` counters count only what the current leg
  read past those cursors. The
  one-shot `session_start` / `session_breakdown` pushes are claimed through an
  exclusive marker file (`reports/trace/.session_start.claim`) rather than
  through the receipt read. The receipt also carries `payload_sha256` over its own body; a
  receipt whose hash does not match is ignored on read, so a torn file cannot
  suppress or replay the one-shot `session_start` / breakdown pushes.
* **Package completeness**: `PACKAGE_MANIFEST` describes what was actually
  written, and `included_files` never names a file the package lacks. Check
  `complete` before treating a package as the whole selection; it is `false`
  whenever anything selected is absent, and the reason is itemized in
  `dropped_files` (the bundle caps at 5000 files / 256 MB and a very long
  session can stop it short, alongside `truncated: true`), `failed_files`
  (the write failed) or `refused_files` (the entry was not a regular file
  inside the session — see below). The zip and the loose tree are written
  independently and each carries a manifest describing its own contents.
* **Session boundary**: a session directory is shared-filesystem state that
  agents write into, so the packager only bundles entries that resolve
  inside the session. A symlink pointing out of the session is refused
  rather than followed, which keeps unrelated file content from being
  copied to the dest root and synced onward.
* **Generation duration is ~0**: Both live and backfill stamp a single
  timestamp (`end == start`), so Langfuse shows no meaningful per-Generation
  duration — counts/usage are accurate, latency is not captured.

### Gateway attribution (`HYPERLOOM_LLM_ATTRIBUTION`)

The per-call ledger (`reports/trace/llm_calls.jsonl`) is Hyperloom's own account
of its spend, assembled from calls that report themselves. The gateway keeps a second account,
from the calls it actually metered, and the two disagree whenever a component
spends without a ledger producer. This variable makes the gateway's account
readable along the same axes, by tagging every outbound call with who made it.

Set it to the name of the gateway in front of the deployment; `litellm` is the
only preset today. Leave it unset (the default) and nothing is emitted.

```bash
export HYPERLOOM_LLM_ATTRIBUTION=litellm
```

Each call then carries, on headers that gateway understands:

| Field | Value |
|-------|-------|
| `application` | Always `hyperloom`, to separate this product's spend on a shared gateway. |
| `session` | `CLAW_SESSION_ID`, the same id `session_breakdown.json` records — this is what joins the two accounts. |
| `component` | The producer, e.g. `orchestration`, `kernel`, `critic`, `specialist`. |
| `phase` | The phase the run had reached. |
| `type` | The action executing inside that phase, e.g. `kernel_opt`. |
| `operation` | What the individual call was for, e.g. `generate_candidate`. |

For LiteLLM these arrive as `x-litellm-tags` (a comma-separated list landing in
the `request_tags` column of `LiteLLM_SpendLogs`) and `x-litellm-trace-id` (the
session id, which sets the `session_id` column and propagates to nested calls).
A spend report can then be grouped by component or phase, or reconciled against
the call ledger for one session.

Fields with per-call cardinality (a task or kernel id) are deliberately not in
the tag list: one tag per task would give the rollup as many buckets as there
are tasks. Supporting another gateway means adding a preset in
`src/hyperloom/common/llm_attribution.py`, not setting a different string here.

---

## AgentX performance metric

Grading defaults to output throughput alone. On an agentic replay that is the
wrong objective: the canonical corpus averages ~114k prompt tokens against ~810
output tokens per request, so output-only grading optimises about 1% of the
token budget, and a variant can lift decode tok/s while degrading user-perceived
latency with no visible cost.

`HYPERLOOM_AGENTX=1` (native YAML generated from Hyperloom flags), or an explicit
Magpie AgentX YAML passed with `--benchmark-config`, adopts the 2-D grading shape InferenceX uses. Upstream
sweeps a concurrency ladder, keeps TTFT / ITL / TPOT percentiles separately, and
publishes a **Pareto frontier** with E2E normalised interactivity as the x-axis
and token throughput per chip as the y-axis. It never collapses the axes into one
weighted number, and it has **no fixed interactivity target** — interactivity is
a frontier coordinate, not a constraint, so a point that trades interactivity for
throughput moves along the frontier rather than violating a rule.

Hyperloom's local KEEP rule (fixed concurrency, no ladder) approximates the
per-concurrency arm selection maintainers apply before submitting:

- **KEEP** — E2E normalised interactivity P50 (median) gain ≥ 3%, **and** the P90
  slow tail and output throughput each no worse than the noise band, **and** the
  pair is comparable: benchmark windows within 5% of each other and no more failed
  requests than the anchor.
- **REVERT** — anything short of all of that.

The median bar is the fixed `AGENTX_KEEP_P50_THRESHOLD_PCT` (3%), not
`keep_threshold_pct`: it is a property of the objective rather than of how far
into the session a round lands, so the decaying session threshold does not apply
to it. Total token throughput no longer participates in the verdict — on this
corpus it is almost entirely prefill, so it cannot see an output collapse.

Hyperloom reads `e2e_norm_intvty_p90` from the accepted `current_best` for both
grading and advisory comparison. This is aiperf's summary **P10** of the
per-request rate `OSL / E2EL_s`, representing the slow tail for a
`LARGER_IS_BETTER` metric. Comparison does not recompute request-level metrics.
The external reference uses `1 / P90(E2EL_s / OSL)`; finite-sample linear
interpolation means the estimators need not be numerically identical.
The comparison is advisory and does not change KEEP/REVERT.

TTFT is included in E2EL, unlike per-user `1/ITL`; on a ~114k-prompt replay TTFT
is most of what a user waits for so grading on `1/ITL` would miss it.

Default-on for AgentX runs, explicit opt-in via `HYPERLOOM_PERF_METRIC=intvty_v1`
otherwise. Any AgentX signal turns it on: `HYPERLOOM_AGENTX=1`, the bound Magpie YAML, or
`benchmark_mode=agentx` stamped at seed — so a round in a subprocess that never
inherited the env var still grades on the agentic axis. Serving frameworks only;
scriptable frameworks (xDiT, custom) keep output-throughput grading.

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_PERF_METRIC`        | `intvty_v1` for Magpie AgentX, else output tput | `intvty_v1` grades E2E normalised interactivity P50 (median) as the primary objective, with the P90 slow tail and output throughput each held as a guard. Reported in the final summary as `grading mode`. |
| `HYPERLOOM_PERF_NOISE_PCT`     | `5.0`                         | Noise band in percent applied to the guards. A candidate whose slow tail or output throughput sits within this band of the anchor is not considered worse on that axis; the median has its own fixed bar and is not subject to the band. The default is the top of the 1–5% run-to-run noise upstream records for this workload. An unparseable value raises `EnvValueError` naming the variable, rather than grading against a band nobody chose. |
| `INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC` | `7200` | Initial server-boot budget written by the persistent-server lifecycle configuration helper. Actual benchmark launches synchronize this field to `INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC`; this variable is not an additional benchmark deadline or a way to extend one. Non-benchmark lifecycle callers that do not perform that synchronization retain their own boot budget. |

### MLPerf agentic backend

`HYPERLOOM_AGENTIC_BACKEND=mlperf` (with `HYPERLOOM_AGENTX=1`) keeps Magpie's
server lifecycle and swaps the AgentX client for the MLCommons
`inference-endpoint` harness (`utility/run_agentic.sh`), replaying
`agentic_combined_v6` trajectories instead of aiperf Weka traces. The backend is
recorded at seed; resuming a session under the other backend is refused.

The harness publishes no per-request OSL/E2EL series, so MLPerf sessions grade
on **output throughput** (`HYPERLOOM_PERF_METRIC` is not honoured here) and gate
every KEEP on the harness's inline accuracy (`scores.json` `score`) against the
baseline's smoke score, within the usual accuracy tolerance. A round with
unscored turns, or over the 10% error-rate ceiling, fails that gate. A baseline
without a fully scored inline accuracy stops the session
(`baseline_accuracy_failed`), so the gate is never left without a reference.

Search runs the 150-trajectory smoke set. The canonical 613-trajectory
submission is not run by Hyperloom: run `run_agentic.sh full` and upstream
`check_compliance.py` on the published recipe.

The AgentX switch pins `PORT=30000`, `MLPERF_AGENTIC_MODEL=kimi-k3` and
`--served-model-name kimi-k3`, and settles `MLPERF_AGENTIC_FLOW`,
`AGENTIC_NUM_TRAJECTORIES` and `AGENTIC_CONCURRENCY` (from the round's `CONC`)
into the recipe. The client refuses to start if port 30000 is already
answering, checks that the listener belongs to the server it booted, and checks
`/v1/models` lists `kimi-k3`.

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_AGENTIC_BACKEND` | — | `mlperf` selects the separate MLPerf client together with `HYPERLOOM_AGENTX=1`. Otherwise `HYPERLOOM_AGENTX=1` selects native Magpie AgentX; an explicit Magpie AgentX YAML also selects it. |
| `MLPERF_ENDPOINTS_DIR` | `/opt/mlperf-endpoints` | Checkout of `mlcommons/endpoints` holding `utility/run_agentic.sh`. Checked at preflight. |
| `AGENTIC_DATASET_PATH` | — (required) | Path to `agentic_combined_v6.jsonl`. Checked at preflight; its stem is recorded as the corpus. |
| `MLPERF_TOKENIZER_DIR` | — (required) | Kimi-K3 fast-tokenizer directory the harness counts tokens with. Checked at preflight. |
| `MLPERF_AGENTIC_FLOW` | `smoke_test` | `run_agentic.sh` flow: `smoke_test` (150 trajectories) or `full` (613). |
| `AGENTIC_NUM_TRAJECTORIES` | Derived from the flow | Overrides the trajectory count. Sizes the benchmark timeout. |
| `MLPERF_AGENTIC_HARDWARE` | `mi355x` | Hardware key passed to `run_agentic.sh`. |

Preflight also refuses a non-SGLang framework and a model path that is not
Kimi. With `INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC` unset, the benchmark cap
defaults to `trajectories × 26 s × 2 + 1800 s` (26 s per trajectory measured on
8×MI355X at concurrency 16, doubled for a cold first round, plus boot), and never
below the stock 7800 s: about 9600 s for smoke and 33,700 s for `full`.

With `HYPERLOOM_AGENTX=1`, Hyperloom generates and archives the native Magpie YAML
from its ordinary workload flags. `--docker-image` supplies an explicit serving
image identity, required for custom models.

Canonical replay warms up 11 requests per lane plus one probe before it opens its
fixed profiling window, so its round grows with the concurrency being measured
while the window does not. That shape is Magpie's, at the ref native AgentX pins
it to (see [Magpie](../components/magpie.md)); Hyperloom composes only `CONC`, and
the numbers below have to be re-measured when the pin moves. With
`INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC` unset and a `CONC` pinned, the cap
defaults to `(11 × CONC + 1) × 20 s + 3640 s + 300 s + 1800 s` (20 s per warmup
request, sized to the slowest rung of one 8×MI355X sweep -- 18.8 s at `CONC` 20 --
with about 6% over it, plus the `metric_window_s` window, metrics finalisation and
boot), and never below the stock 7800 s: about 10,200 s at `CONC` 20. A
concurrency sweep sizes this per rung from the environment the rung's own config
is launched with -- whether the `CONC` in it was pinned by the rung, inherited
from the base config or cleared by the rung -- so the rung that measures more
lanes is granted more time rather than the ladder sharing one grant. The
override itself stays the operator's: it is read from the process environment
that preflight validated, never from a rung's config. Sizing reads a rung's
workload pins as its launch would, so a rung whose config spells one the reader
refuses (`HYPERLOOM_AGENTX=maybe`) is recorded as failed with error class
`benchmark_cap_error` and the sweep carries on with the next rung.

Native AgentX profiling shares TraceLens patch and kernel-shape-tool preparation with
fixed-sequence profiling. Configure capture steps, count, delay and interval through
`benchmark.profiler.torch_profiler` (`num_steps`, `num_profiles`, `start_seconds`,
`interval_seconds`). The diagnostic copy preserves those settings and explicit
`detailed_annotations`. When profiling is enabled, a missing capture setting becomes
one capture of 256 steps, starting 300 seconds after measurement, with a 900-second
interval. A missing `detailed_annotations` becomes true, and false when the TraceLens
runtime patch was attempted and did not apply: Magpie probes the server whenever the field
is true and refuses to launch when the server lacks it.
Hyperloom adds the shape-tool environment through `agentx.launch_overrides` while
preserving the server's runtime paths. Magpie owns profiling flags, starts after replay
reaches measurement, and manages capture/flush. Each window remains a separate TraceLens input.
Profile reports are diagnostic only and cannot supply KEEP evidence.
See [Optimize an AgentX workload](../how-to/optimize.md#optimize-an-agentx-workload).

---

## Phase tuning

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `INFERENCE_OPTIMIZER_CYCLE_HOURS` | Optional | `24.0` | Duration of one macro-cycle in hours. Multiplied by 60 to obtain `cycle_minutes` on the first session start. A malformed value raises `EnvValueError`. |
| `INFERENCE_OPTIMIZER_BACKEND_ERROR_STREAK_THRESHOLD` | Optional | `5` | Consecutive backend errors per agent role before a `backend_unhealthy` event is recorded. A malformed value raises `EnvValueError`. |
| `INFERENCE_OPTIMIZER_CYCLE_RELOOP_MIN_REMAINING_SEC` | Optional | `10800` | Absolute minimum remaining session seconds to justify opening a new macro-cycle. For bounded sessions the effective floor is `min(this, max_minutes * 60 * 0.15)` so shorter sessions are not unconditionally blocked. |
| `INFERENCE_OPTIMIZER_KERNEL_IDLE_MAX_TICKS` | Optional | `3` | Consecutive no-work KERNEL_AGENT ticks before the phase winds down to SWEEP. Values below `1` are treated as `3`. A malformed value raises `EnvValueError`. |
| `INFERENCE_OPTIMIZER_KERNEL_IDLE_MIN_SECONDS` | Optional | `600` | Wall-clock seconds a KERNEL idle streak must last before winding down to SWEEP. Values at or below `0` are treated as `600`. A malformed value raises `EnvValueError`. |
| `SWEEP_VARIANT_TIMEOUT_SEC` | Optional | `2400` | Per-variant wall-clock timeout in seconds for GEAK-backed SWEEP runs. A malformed value raises `EnvValueError`. |

---

## Plug selection

Which implementation serves each plug socket (today only `events`, progress reporting) is chosen by layering, lowest
first: the built-in `local` profile, the `plugs.resolved.json` of the session being resumed, `--plug-profile`,
the manifest (`--manifest` or `HYPERLOOM_MANIFEST`), `HYPERLOOM_MANIFEST_OVERRIDES`, then `--plug` and `--plug-config`.
With none of these set, a run uses the `local` profile, writes nothing to the session and prints nothing. See
[Select plug implementations](../how-to/optimize.md#select-plug-implementations).

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_MANIFEST` | Unset (the `local` profile) | Path of a JSON plug manifest (`"schema": "hyperloom.plugs/v1"`) laid over the profile. `--manifest` wins over it. An exported value changes which implementations every run in that shell uses, and a selection is recorded in the session and re-applied on resume. |
| `HYPERLOOM_MANIFEST_OVERRIDES` | Unset (no overrides) | Path of a JSON file of the same shape laid over the manifest; only `--plug` and `--plug-config` win over it. Meant for a host that must adjust a run without editing the manifest. |

A path that does not exist, a malformed file, an unknown socket or implementation, or a `${VAR}` that is unset and has no
default stops the run with exit code 2 before any session is opened or changed.

---

## Hosted run inputs

A hosted run is configured by the host that launches it. `src/hyperloom/hosted/host_info.py` is the only reader of these variables:
it checks them once at start-up and names **every** missing or malformed variable in one error (variable names and reasons, never
values), so a bad input stops the run before it starts. Which variables are required depends on what the run hosts; the run, state
and input variables are always required except for a session started by hand on a grant handed to it
(`hyperloom-multibranch run --measurement hosted`), which needs the sandbox URLs and templates, `HYPERLOOM_SANDBOX_KEY_MEASUREMENT`,
`HYPERLOOM_GRANT_LIMITS`, `HYPERLOOM_RUN_CREATED_AT` and the source pair. A variable that is set is checked even when it is not
required.

| Variable | Required | Description |
|----------|----------|-------------|
| `HYPERLOOM_INPUT_FILE` | Always | Absolute path of the campaign request file; must exist. |
| `HYPERLOOM_RUN_ID` | Always | Run identifier, no whitespace. |
| `HYPERLOOM_RUN_CREATED_AT` | Always | Run creation time, ISO 8601 with a UTC offset or `Z`. |
| `HYPERLOOM_STATE_DIR` | Always | Absolute path of the run's state directory on shared storage; must exist. |
| `HYPERLOOM_EVENTS_DIR` | No | Directory of the progress-event segments (see "Progress events" below). Default `$HYPERLOOM_STATE_DIR/events`. |
| `HYPERLOOM_MCP_PORT`<br>`HYPERLOOM_MCP_TOKEN` | When the MCP server runs (default) | Port (1 to 65535) of the run's MCP server and the token the host presents. The token is a credential and must be at least 16 characters; a shorter one refuses the run. |
| `HYPERLOOM_MCP_HOST` | No | Address the run's MCP server binds to, no whitespace. Default `127.0.0.1`; a hosted run is given `0.0.0.0`. |
| `HYPERLOOM_MCP_PUBLIC_URL` | When sub-runs call back into the run | `http(s)` URL under which sub-runs reach the run's MCP server. No credentials, query or fragment. |
| `HYPERLOOM_SANDBOX_TEMPLATES` | When a sandbox is hosted | JSON object of each role's templates by card count (a decimal key; the name is opaque), for example `{"measurement": {"1": "t1", "8": "t8"}, "dev": {"1": "d1", "8": "d8"}}`. `runner` is accepted as the older name of `measurement`. |
| `E2B_API_URL`<br>`E2B_SANDBOX_URL` | When a sandbox is hosted | Control-plane and data-plane `http(s)` URLs of the sandbox service, without credentials, query or fragment. |
| `HYPERLOOM_SANDBOX_KEY_MEASUREMENT`<br>`HYPERLOOM_SANDBOX_KEY_SUBRUN`<br>`HYPERLOOM_SANDBOX_KEY_DEV` | Per hosted role | The sandbox API key of each role. Credentials. |
| `HYPERLOOM_RESULT_ARCHIVE` | No | Unset means results are archived on the shared storage, in a directory the application chooses. Otherwise `s3://<bucket>/<prefix>`; the prefix may be empty and has no empty, `.` or `..` segment. A local path (`/x`, `file://...`) is refused: local means unset. |
| `AWS_ACCESS_KEY_ID`<br>`AWS_SECRET_ACCESS_KEY` | When `HYPERLOOM_RESULT_ARCHIVE` is `s3://`, and always as a pair | Archive credentials. `AWS_SESSION_TOKEN`, `AWS_ENDPOINT_URL_S3` (an `http(s)` URL) and `AWS_REGION` (not a credential) are optional. |
| `HYPERLOOM_GRANT_LIMITS` | No | JSON object of the totals the host granted the run: `max_sandboxes`, `max_gpus`, `max_runtime_sec` (whole numbers) and `gpu_hours` (a number), each at least 0 and each optional; other keys are ignored. The legacy `optimize` flow only logs it. Removed from the environment once read. |
| `HYPERLOOM_SOURCE_REPO`<br>`HYPERLOOM_SOURCE_SHA` | When the run starts measurement sandboxes (the application bootstrap always needs them) | The repository (`OWNER/NAME` or a git URL, no whitespace) and the full lowercase 40-hex commit id of the version the run executes; the sandbox runners fetch the same version. Checked when set; removed from the environment once read (the bootstrap script has already used them). |
| `NODE_NAME`<br>`CLAW_SESSION_ID` | No | Physical node name and session id, read when set. |

Once read, every host-only input above is removed from the process environment (`take_host_info`), so a child process started
without an explicit `env=` does not inherit it; `HostInfo` holds the values and code passes them on. `NODE_NAME` and `CLAW_SESSION_ID`
stay, because other code reads them. The sandbox keys, the MCP token and the archive credentials are held in an object that prints as
`Secret(<redacted>)`, and are also on the credential list in `hyperloom.common.env_safety`.

### The hosted entry and its bootstrap

`hyperloom hosted run` (`src/hyperloom/hosted/entry.py`) reads the variables above and the campaign spec in
`HYPERLOOM_INPUT_FILE` (`hyperloom.campaign-spec/v1`, schema `src/hyperloom/contracts/campaign_spec.v1.schema.json`), runs the
packaged `src/hyperloom/inference_optimizer/assets/install.sh` (no flags; the installer of the legacy launchers, from the same
wheel as the running code; it provisions TraceLens, GEAK, ray and `runtime/kernel-agent.env.sh`) on every leg, fresh or resumed, into the interpreter that runs the entry (`PYTHON`, `VIRTUAL_ENV` and `INFERENCE_OPTIMIZER_FORCE_PYTHON=1` are set for it, and that interpreter's `bin` directory leads `PATH` for the installer and `optimize`), with its
output in `$HYPERLOOM_STATE_DIR/install.log`, then runs `optimize` in the same process. An installer that exits non-zero ends the run
with `{"final": true, "returncode": 2, "error": ...}` naming `install.sh` and the log. The session directory is
`$USER_DATA_PATH/<model directory name>/<HYPERLOOM_RUN_CREATED_AT as YYYYMMDDTHHMMSSZ>-<first 8 hex of sha256(HYPERLOOM_RUN_ID)>`; a
restart finds the same directory and resumes it when it holds `manifest.json` and `state.json`. The entry writes two files in
`$HYPERLOOM_STATE_DIR`: `session.json` (`{"session_dir": ...}`, as soon as the directory exists) and `exit.json`
(`{"final": true, "returncode": N, "stop_reason": ...}` when `optimize` ends for any reason, and `{"final": true, "returncode": 2,
"error": ...}` when the request is refused). A host restarts the run only when `exit.json` is missing or its `final` is false.
`HYPERLOOM_IMAGE` is set from the spec's `framework.image`, and the entry needs `USER_DATA_PATH` set to an absolute path. The result
archive directory of the run is `<session dir>/archive/`, mode 0700, owned by the application user; the entry sets `HYPERLOOM_SESSION_PACKAGE_DEST` to it (whatever the host passed), so the session artifact package and its loose copies are written to `archive/hyperloom-session-packages/` on the shared storage, not to the container's `/workspace`.

`src/hyperloom/hosted/app_bootstrap.sh` starts a hosted run in an image without Hyperloom. It installs into the framework Python environment instead of a venv of its own, because a venv made from the image's interpreter cannot see the torch and ROCm stack that `install.sh` requires. The script is baked into the image: a change to it needs a rebuild.

`scripts/hosted_app_image/build.sh --tag <ref> [--base <image>] [--claude-version V] [--push]` builds that image: a framework base image (`--base`), the script as the image's command, the `hl-specialist` account for `HYPERLOOM_SPECIALIST_USER`, and the claude CLI that specialists spawn (`/usr/local/bin/claude` -> `/opt/claude-code`). The CLI's version is `--claude-version`; when it is not passed the build takes `latest`, so two builds of one tag can carry different CLI versions. Pass a version to pin it. The codex CLI is not installed separately: it comes with the wheel (`openai-codex-cli-bin`).

The image is not required. `scripts/hosted_app_image/build_layer.sh --out <dir> [--claude-version V] [--ca-file PEM]` builds a layer directory `<dir>/<short sha>-claude<version>/` (`start.sh`, `app_bootstrap.sh`, `bin/claude`, and `ca.pem` with `--ca-file`), and `scripts/hosted_app_image/Dockerfile.layer` makes it a small image (the directory is its build context; the content is under `/layer`). An application pod copies `/layer` into a volume with an init container (`cp -a /layer/. /opt/hyperloom-host/` into an `emptyDir`), and the main container runs `sh /opt/hyperloom-host/start.sh` on any framework image, unchanged. `start.sh` runs as the container's root and reads only its own directory (the layer may be mounted read-only): it creates `hl-specialist` (uid/gid 2000) when missing, puts `bin/claude` on `PATH`, writes a CA bundle (the image's plus `ca.pem`) to `${TMPDIR:-/tmp}/hyperloom-host/ca-bundle.pem` (directory 0755, file 0644, local to the container; nothing goes to the state dir) for `SSL_CERT_FILE`, `REQUESTS_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS` and `GIT_SSL_CAINFO`, and execs `app_bootstrap.sh`. The image must already have `git`, `setpriv` and `bash`; `start.sh` refuses by name when one is missing and installs nothing. uid 2000 must not belong to another account.

| Variable | Required | Description |
|----------|----------|-------------|
| `HYPERLOOM_SOURCE_REPO` | Yes | `OWNER/NAME` on GitHub, or a git URL or path, of the Hyperloom source to run. |
| `HYPERLOOM_SOURCE_SHA` | Yes | Full 40-character commit id of that source. The wheel for it is built once into `$USER_DATA_PATH/.hyperloom/wheels/<sha>/` and only that wheel is installed (`--no-deps --force-reinstall`, so a new commit replaces an old one) into the image's framework Python environment (`/opt/venv` when it exists, else `$PYTHON`, else `python3`, the order `install.sh` uses), which also runs the entry. The runtime dependencies are resolved there by the `install.sh` the entry runs before `optimize`, as in the legacy flow. The environment records the installed commit in `.hyperloom-install`; a container that already holds it installs nothing. |
| `HYPERLOOM_CODE_REVISION` | Set by the script | The commit id, exported before the entry starts: a wheel has no `.git` to read it from. |

### The host's guarantee about containers, and the resume that relies on it

The host guarantees that one run has at most one application container at a time, and starts a new leg only after it has
confirmed that the previous container no longer exists. The entry relies on that guarantee when it resumes a session:

* It compares the owner scopes recorded in the session's ownership ledger with the scope of its own container. With no other scope
  (a restart in the same container) the resume goes through the preflight's own checks, unchanged.
* Another scope means the previous container is gone. The entry then does what `hyperloom recover --confirm-scope-gone` does for each
  such scope, with the same code: the executions that scope left running are marked interrupted, their leases are released, their
  work is redone, and the resume proceeds. A confirmation that the code refuses (for example another owner's live lease) ends the run
  with `{"final": true, "returncode": 2, "error": ...}`.
* A host that starts a new container while the previous one is only unreachable breaks the guarantee: two containers would write
  one session directory. Such a host cannot use this hosting. Supporting one would need an explicit signal, in which the host
  states at the start of a leg that the previous container no longer exists; the entry does not read one today.

### Progress events

With the hosted `events` implementation (`src/hyperloom/hosted/events.py`, selected by the `hosted-*` plug profiles) the run writes
CloudEvents 1.0 structured JSON, one event per line, to `<events dir>/<10-digit segment>.jsonl`; a new segment starts before the
current one would pass 4 MiB. The directory and the run id (`source` is `hyperloom/<run id>`) come from `HostInfo`, never from the
environment, so a child process started by the run has no events directory and never emits.

* `id` is a decimal string that strictly increases within the run. After a restart it continues from the last complete line of the
  last segment; a half-written last line is cut off first.
* Every line ends with a newline and is written whole, so a line without one is an unfinished write, not an event. A reader that
  tails by byte offset takes only complete lines.
* One event is at most 16 KiB; a larger one is sent with `data` replaced by `{"oversized": true, "bytes": N, "keys": [...]}` and a
  warning is logged. Put large content in a file and send its path.
* `fsync` runs at most once a second, and on `flush()` and close. Emitting never raises into the run: a failure is logged.

Types (v1): `hyperloom.campaign_started`, `plugs_lock_mismatch`, `phase_changed`, `measurement_started`, `measurement_finished`,
`node_quarantined`, `candidate_settled`, `best_updated`, `budget`, `warning`, `campaign_finished`, `campaign_failed`. The `data`
schema of each is `src/hyperloom/contracts/hyperloom_events.v1.schema.json`; a test keeps the emitters consistent with it.
`hyperloom.budget` is sent after a state save, at most every 5 minutes; its `tokens` and `gpu_sec` are null in the legacy flow.
With the local implementation (the default) an event is only a debug log line.

## Measurement card guard

What is done about a process outside a test that holds one of the test's GPU cards. The local toolbox reads it once, when
it is built (the `card_guard` option wins over it), applies it to the cards it hands out to tests and leases, and passes
the same value to every measurement runner it starts.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_CARD_GUARD` | `enforce` | `enforce`: a card some process holds is not handed out (the test waits for a free one; the log names the card, the process and the memory it holds, once per holder), and a test whose card a process outside it holds is ended at the first look that finds it (about once a second) and refused as the infrastructure failure `foreign_process_on_card`, which is retried on other cards. `observe`: cards are handed out and tests run to their end; the signed result says `foreign` with the process, the card and when, and carries a warning. Any other value is refused at start-up. Where the cards cannot be read, both modes go on and the result says `unknown` and why. |

## Variables intentionally not exposed

These are read by `os.environ` somewhere in the codebase but are
internal-only — do not set them by hand:

* `HYPERLOOM_HOST_PROBE`, `HYPERLOOM_HOST_PROBE_DEEP`,
  `HYPERLOOM_HOST_PROBE_DIR`, `HYPERLOOM_HOST_PROBE_ROOTS` (and the
  `..._MAX_SITES` / `..._ARG_SAMPLES` caps): internal handoff to the host-stall
  evidence probe, armed by the profile leg that collects the evidence. The deep
  tier inflates host time by design, so setting it by hand distorts any trace
  collected alongside it.
* Any `_INFERENCE_OPTIMIZER_*_INTERNAL_*` symbol: internal toggles for
  the test suite.

If you find one of these in a log message, treat it as diagnostic
detail rather than something you should tune.

---

## More info

Use these resources for related configuration and reference information:

* [Hyperloom authentication and credentials](authentication.md): Credential precedence and direct upstream gateway wiring.
* [Troubleshooting Hyperloom](troubleshooting.md): Symptom → variable reverse-lookup for common failures.
