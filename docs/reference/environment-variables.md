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
| `TRACELENS_ROOT`                          | No (installer auto-clones) | `${HYPER`<br>`LOOM_CA`<br>`CHE_DIR:-`<br>`$REPO_ROOT`<br>`/.cache}/Tr`<br>`aceLens@<resolved-sha>` (auto-clone of `AMD-AGI/TraceLens` pinned to a fixed SHA) | `src/hyperloom/agents/kernel/scripts/install.sh` clones the public repo into the repo-local cache root when unset. Export it to opt into a pre-existing checkout you maintain — that is an explicit operator override and skips both the clone and the SHA pin. |
| `GEAK_CLAUDE_BIN`                          | No (installer auto-resolves) | First of `$HOME/.local/bin/claude`, `/usr/local/bin/claude`, `$(command -v claude)`; written to `kernel-agent.env.sh` | Pins the Claude Code binary the GEAK SDK path uses, so `claude_agent_sdk` doesn't fall back to its older bundled CLI. Export to force a specific build. |
| `USER_DATA_PATH`                          | No                   | `/workspace/hyperloom` if `/workspace` is writable, else `<cwd>/session` | Session directory root (logs, runs, mirrors, breakdown). Container images ship a writable `/workspace`; a bare-metal host that has neither falls back to the second form and the CLI logs which root it took.                                                |
| `HYPERLOOM_`<br>`RUNTIME_DIR`             | No                   | `$USER_DATA_PATH/runtime` (installer)                               | Private writable runtime state. Codex SDK turns create a unique mode-`0700` `CODEX_HOME` here and remove it after the SDK client closes. When unset, Codex uses the first safe declared output root, then a run-local working directory; it never falls back to `/tmp` or a source checkout. If launch config points this parent at storage unsuitable for SQLite, set it to a private local directory instead — see [Codex SDK turns stall](troubleshooting.md#codex-sdk-turns-stall-or-tracelens-roofline-times-out). |
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

Set with CLI flags or an explicit `--benchmark-config`, not ambient workload
env vars. Pre-set `ISL` / `OSL` / `CONC` / `PRECISION` / `TP` / `EP` env vars
are ignored and overwritten (`GPU_TYPE` is a fallback when `--gpu-type` is
omitted).

- **Model / workload shape:** `--benchmark-config`, `--model`, `--model-class`, `--framework`,
  `--framework-version`, `--precision`, `--tp`, `--ep`, `--isl`, `--osl`,
  `--conc`, `--max-model-len`, `--profile-osl`.
- **Goal / budget:** `--target-gain`, `--target-roofline`, `--max-hours`,
  `--target-tput`, `--compare-against-gpu`. The roofline target composes with
  the others: whichever is met first ends the run.
  `--max-latency-ms` is a constraint on KEEP rather than a target, and applies
  to scriptable frameworks (`xdit`, `custom`) only; the CLI refuses it for a
  serving framework, including on resume.
- **GPU power settings (asserted, never set):** `--gpu-power-cap-w`,
  `--gpu-perf-level`. Set them with `amd-smi set` before launch; the session
  refuses to start or resume if a card it uses differs, and records what the
  cards report either way. Per-round GPU power is sampled by Hyperloom over
  the measured phase into each round's `gpu_power.json`;
  `HYPERLOOM_GPU_POWER_SAMPLING=0` turns the sampler off (the round then falls
  back to the benchmark report's reading) and `HYPERLOOM_GPU_POWER_INTERVAL_S`
  sets its cadence (default 2 s, minimum 0.5 s).
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
  `--conc-sweep-total-budget-sec` (whole-sweep budget), `--no-framework-agent`, `--no-framework-local-explore`, `--no-kernel`,
  `--no-eval`.
- **Agent models:** `--claude-model`, `--codex-model`.
- **Session / resume:** `--resume-from`, `--force-resume`, `--reset-state`,
  `--extend-hours`.
- **Quantization:** `--quantize`, `--quantize-scheme`.

Run `inference_optimizer optimize --help` for the exhaustive flag list.

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
so eval-less environments still make progress. Pass `--no-eval` to turn the
eval off for the whole run: the baseline anchors on throughput instead of
halting on a missing accuracy reference, and every candidate then lands on
that degraded path.

| Variable | Default | Description |
|----------|---------|-------------|
| `RUN_EVAL` | `true` | Whether a serving benchmark runs the GSM8K eval. Turning it off removes the per-candidate accuracy signal entirely — accuracy regressions stop being caught. Ignored by scriptable workloads, whose correctness signal is the `quality_gate` in `benchmark_report.json`. |
| `HYPERLOOM_QUALITY_REF`<br>`HYPERLOOM_QUALITY_REF_WRITE` | Derived under the session dir | The scriptable quality gate's reference artifact: `_WRITE` establishes it on the baseline, the other compares against it on every later candidate. What the artifact holds is the workload's own business — xDiT stores an image, an operator-supplied `custom` workload stores whatever its script compares. Also emitted as `XDIT_QUALITY_REF` / `XDIT_QUALITY_REF_WRITE` for bench scripts written before the rename; either name is read, both are written. |
| `INFERENCE_OPTIMIZER`<br>`_REQUIRE_KERNEL`<br>`_ACCURACY` | On | Gates the `KEEP` for a kernel patch integrated by the kernel lane. Set to `0` / `false` / `no` / `off` to fall back to a throughput-only `KEEP`. Disable only when the eval lane is known-broken: this gate is what stops a faster-but-wrong kernel from being kept. |
| `MAGPIE_EVAL_LIMIT` | Unset (full task set) | Caps the number of eval problems (`lm_eval --limit`). Useful for smoke runs; see the noise caveat below before using it on a run whose `KEEP` decisions matter. |

The tolerance is deliberately **not** an env knob: `ACCURACY_THRESHOLD` in
`src/hyperloom/orchestrator/actions/executors/_accuracy_gate.py` is a fixed
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

Every workspace runs a local Experience KB service; a global KB is optional.
See [Experience KB service](experience-kb.md) for how each is used.

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_KB_URL` | Written by `hyperloom-setup` (`http://127.0.0.1:<port>`, a port of the workspace's own in 20000–29999) | The workspace's local Experience service. A loopback URL is started by each optimize launch; unset disables Experience reads and writes. |
| `HYPERLOOM_KB_TOKEN` | Generated by `hyperloom-setup` | The local service's access token. |
| `HYPERLOOM_GLOBAL_KB_URL` | Unset | The global Experience KB the local service pushes to and pulls from. |
| `HYPERLOOM_GLOBAL_KB_TOKEN` | Unset | The global KB's access token; required with its URL. |
| `HYPERLOOM_KB_AUTO_PUSH` | Off | `1` pushes after every run's Experiences are written locally. |
| `LOCAL_KB_PLANNER_MODEL` | `CLAUDE_MODEL` | The model the Experience service plans reads with, through `ANTHROPIC_BASE_URL`. |
| `LOCAL_KB_PLANNER_TIMEOUT_SECONDS` | `20` | How long the Experience service waits on its planner before a read fails. A run waits 30 s per read, so a value above that makes every read fail. |
| `LOCAL_KB_PLANNER_MAX_OUTPUT_TOKENS` | `1400` | The most output tokens the Experience service's planner may spend on one read plan. A plan cut at this limit fails its read with `planner gateway truncated output at max_tokens`. |

---

## Kernel-opt backend selection

The following variables control the kernel optimization backend ladder.

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `KERNEL_OPT_BACKEND_ORDER`     | Unset (resolves to `geak`) | Selects the kernel-opt backend. An unset or blank value resolves to `geak` for every framework, ATOM included; no framework overrides it at launch. Bare-metal setup preserves nonempty choices (process env > `.env`) without filling or persisting a backend default. Existing `.env` values, including `geak`, are not automatically migrated; remove that assignment and unset the shell variable to use the shared default. Slurm launchers still export `${KERNEL_OPT_BACKEND_ORDER:-geak}`. **Only an exact, case-insensitive `forge` enables the per-kernel Forge backend** (`forge_explicitly_enabled` in `common/env.py`). Despite the historical name, a comma list is not parsed: `forge,geak` stays on `geak`, as does any other nonblank value, legacy alias, or payload override. |
| `HYPERLOOM_GEMM_SHAPE_CAPTURE` | `1`                           | Enables automatic runtime GEMM-shape capture for eligible single-node dense vLLM Forge tuning when no explicit shape input is available. Block-FP8 first reuses shapes from the TraceLens-selected steady-state trace of a successful Roofline with exactly matching model, workload, server arguments, environment, and backend controls. Missing or stale evidence triggers the same standard Roofline/ProfileExecutor/TraceLens steady-state pipeline as a fallback. Set to `0` to preserve the no-capture path. |
| `HYPERLOOM_GEMM_SHAPE_CAPTURE_TIMEOUT_SEC` | `1800`          | Timeout in seconds for the dense vLLM TunableOp recording benchmark. Block-FP8 fallback uses the standard Roofline/ProfileExecutor timeout. Values below `60` are clamped to `60`. |
| `HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT` | `5.0`                     | GPU-time share a reusable hot kernel must clear to be worth a dispatch, in percent. Three readers share it and must agree, or the report explains a skip the dispatcher never made: the batch filter that selects candidates, the phase-advance gate that decides KERNEL still owes work, and the report's unattempted-reason breakdown. It was `10.0` until a 60-layer sparse-MoE decoder showed the assumption behind that number — that hot kernels concentrate — does not hold: nothing but a graph-launch wrapper reached double digits, the largest real operator sat at 9.47%, and the batch dispatcher selected nothing for six hours. Lower it when a trace's rewritable candidates cluster below the default and the operators above them are vendor binaries; a dropped candidate is reported as `below_min_gpu_pct=<value>` rather than as a failed attempt. |
| `AITER_LOG_TUNED_CONFIG`       | `1` (set for every serving run) | Makes aiter log each tuned-config lookup it *hits*, not only the ones it misses. Two checks have no input without it: the GEMM demand list, which learns the shapes the runtime actually asks for (config-derived shapes covered 0.4% of them), and the apply verdict, which cannot tell "the tuned table was never read" from "it was read and did not help". A scan of 60 production logs found it set in none of them, so it is now injected by default. An operator value wins — set `0` to turn hit logging off, at the cost of both checks going inconclusive. Every miss already prints a line regardless of this setting; hit logging adds roughly one line per lookup that succeeds. |
| `HYPERLOOM_GEMM_PAIRED_PAIRS`  | `0` (off)                     | How many interleaved baseline/tuned pairs to re-measure before a GEMM tuning KEEP is reported as confirmed. One end-to-end measurement cannot separate a gain from drift on this fleet: three rounds of a single unchanged configuration spanned 58%, and one controlled repeat moved 16%. Each pair costs two extra benchmark rounds. When `0`, the gain is still promoted — it is the best number available — but recorded as an unpaired block comparison rather than presented as a paired one. |
| `GEAK_E2E_TIMEOUT_S` | `43200` | GEAK end-to-end timeout in seconds. Used as the fallback when no run deadline is set. A malformed value raises `EnvValueError`. |
| `GEAK_BUDGET_MARGIN_S` | `300` | Seconds subtracted from the remaining kill budget before passing the runner timeout to GEAK, so the runner self-stops before the hard subprocess kill. A malformed value raises `EnvValueError`. |
| `GEAK_MIN_RUN_S` | `600` | Minimum GEAK runner seconds required to attempt a run. If the remaining budget is below this, GEAK is skipped and the phase winds down to SWEEP. A malformed value raises `EnvValueError`. |
| `GEAK_TERM_GRACE_S` | `180` | Grace period in seconds between SIGTERM and SIGKILL for the GEAK subprocess tree. A malformed value raises `EnvValueError`. |

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
| `FORGE_FUSION_TIMEOUT`         | `7200` (2h)                   | Wrapper timeout in seconds for one forge-fusion run. A payload `timeout` / `timeout_sec` takes precedence over the env; an unparseable value falls back to the default.                            |
| `FORGE_FUSION_MAX_TURNS`       | `100`                         | Agent turn cap handed to forge-fusion for one run. A payload `max_turns` takes precedence.                                                                                                         |
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
  than a single combined patch. `fuse_all_confirmed` is now the explicit opt-in
  for the combined shape instead of the default.
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
decided that — in a versioned schema (`schema_version`, currently `1.0.0`), so
consumers and triage read a contract rather than candidate internals.

---

## Single-node Ray execution

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_OPTIMIZER_RAY_EXEC` | Unset (`on` for ordinary single-node runs) | Controls whether single-node serving benchmarks and `needs_gpu` specialists run through Ray actors. When unset, ordinary single-node runs are routed through Ray-managed leases while multi-node stays on the multi-node backend. Set to `0` / `false` / `no` / `off` to force the local subprocess path, or `1` / `true` / `yes` / `on` to force Ray. Native AgentX is the exception: it always bypasses the outer Ray actor, and explicitly forcing Ray is rejected. |

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

`HYPERLOOM_MN_EXT_SERVICE_URL` is the only variable that tells the optimizer a
cluster is ready; the rest describe how to reach it.

| Variable | Backend | Required | Description |
|----------|---------|----------|-------------|
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

Multi-node SSH fanout creates session-scoped keys under the active session
directory. Treat `mn_id_ed25519` and `mn_id_ed25519.pub` as sensitive session
artifacts: keep the session directory on an access-controlled filesystem and
do not publish it unchanged in support bundles.

---

## Quantization prelude

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_QUANTIZE_ENABLED` | Unset | Primary switch (`1` to enable) for the AMD Quark PTQ quantization prelude driven by `--quantize` / `--quantize-scheme`. |
| `QUARK_ROOT` | Unset | AMD Quark checkout used by the quantization-agent. Set this explicitly when quantization is enabled. |

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
fixed constant `_accuracy_gate.DEFAULT_ENABLEMENT_ACCURACY_FLOOR` (`0.5`):
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
| `INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC` | `7800`; under `HYPERLOOM_AGENTIC_BACKEND=mlperf`, sized to the trajectory count (see [MLPerf agentic backend](#mlperf-agentic-backend)) | Hard wall-clock seconds per actual spawn, including server boot and accuracy evaluation. Output cannot extend it. An explicit value always wins. |
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
| `INFERENCE_OPTIMIZER_CLAUDE_TURN_TIMEOUT_SEC` | `1500` | Total wall-clock limit for one Claude Agent SDK call made by the orchestration backend (reactor turns, including the SWEEP cycle-directive handoff), covering every retry and the backoff between them; tearing the CLI down afterwards adds at most 30 s. Unlike the idle bound of `INFERENCE_OPTIMIZER_CLAUDE_CALL_TIMEOUT_SEC`, the CLI's own API-retry messages do not extend it. Past it the call is cancelled, the CLI and every process it started are killed, and the turn fails as a backend error the loop retries on its next tick. In-process Claude specialists use `--specialist-per-turn-max-seconds` instead. |

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
| `HYPERLOOM_`<br>`VLLM_ROCM_`<br>`INDEX_URL` | Unset | ROCm pip index URL used as the default vLLM adapter wheel index; also seeds the index allowlist. |
| `HYPERLOOM_`<br>`ENABLEMENT_`<br>`INDEX_ALLOWLIST` | Unset | Comma-separated allowlist of pip index URL prefixes; a candidate wheel index must match one of these prefixes or provisioning is refused (supply-chain safety). |
| `HYPERLOOM_`<br>`ENABLEMENT_`<br>`ORIGIN_ALLOWLIST` | Unset | Comma-separated allowlist of git origin URL prefixes; a candidate repo origin must match one of these prefixes or provisioning is refused (supply-chain safety). |
| `HYPERLOOM_`<br>`SGLANG_REPO_URL` | Unset | Override the SGLang source repo URL for the sgl-kernel / SGLang-from-source enablement build. |
| `HYPERLOOM_`<br>`SGLANG_REF` | Unset | Pin the SGLang source ref (tag/branch/sha) for the enablement build. |
| `HYPERLOOM_`<br>`SGLANG_INDEX_URL` | Unset | SGLang wheel index URL for the enablement build. |

> **Supply-chain security:** `HYPERLOOM_ENABLEMENT_INDEX_ALLOWLIST` and
> `HYPERLOOM_ENABLEMENT_ORIGIN_ALLOWLIST` are security controls.  When set, only
> pip index / git origin URLs matching one of the listed prefixes are accepted
> for runtime provisioning; any non-matching candidate is refused.

---

## Security compatibility switches

These switches keep production-compatible behavior by default while still
allowing operators to turn off credential/env persistence in hardened
deployments.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV` | Unset (`1`) | Specialist subprocesses inherit the limited provider credential set by default: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_CUSTOM_HEADERS`, `CLAUDE_CODE_OAUTH_TOKEN`, `OPENAI_API_KEY`, `OPENAI_CUSTOM_HEADERS`, and AWS Bedrock credential/config vars. Set to `0` only when the `claude` CLI is authenticated through its own config and env credentials must be suppressed. Unrelated secrets such as GitHub and KB tokens remain blocked. |
| `HYPERLOOM_SPECIALIST_PERMISSION_MODE` | `bypassPermissions` | `--permission-mode` passed to the `claude` CLI for specialist subprocesses. Controls the Claude runtime approval-prompt behaviour only. Codex containment is resolved independently through `HYPERLOOM_CODEX_SANDBOX_MODE`. The default `bypassPermissions` is required for unattended operation; change only in setups where an external interactive approval flow is intended. |
| `HL_ALLOW_DANGEROUS_AGENT_PERMISSIONS` | Unset (`0`) | Slurm carrier only. Set to `1` only in dedicated internal containers to re-enable legacy Claude/Codex approval and sandbox bypass flags. |

---

## Critic / knowledge base (KB)

The following variables configure the Critic and knowledge base components.

| Variable                              | Default                | Description                                                                                                                          |
|---------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `KNOWLEDGE_STORE_MODE`                | `local`                | Exclusive Recipe backend: `local` or `remote`. Ambient KB Store or GBrain credentials do not select remote mode. |
| `KNOWLEDGE_LOCAL_ROOT`                | `$USER_DATA_PATH/knowledge`, otherwise `~/.cache/hyperloom/knowledge` | Local Recipe/KG root. It is not used for Recipe data in remote mode. |
| `HYPERLOOM_`<br>`LOCAL_KB_ROOT`       | Unset                  | Deprecated explicit local Recipe root compatibility input, overridden by `--local-kb-root`; explicit use skips automatic legacy migration. |
| `INFERENCE_OPTIMIZER_`<br>`FA_KB_PATH` | `$USER_DATA_PATH/framework-kb`, otherwise `/workspace/hyperloom/framework-kb` | Framework KB root, holding the lessons ledger the FRAMEWORK phase reads and writes. The orchestrator's KB reader and writeback both resolve through it. The withdrawn `FRAMEWORK_AGENT_KB_DIR` is ignored with a warning naming the resolved root. On first start-up an existing partition under the legacy `$USER_DATA_PATH/kb` is copied across once; a copy that fails warns and leaves the phase to cold-start. |
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
- **Live push**: when set to `1/true/yes/on` and the three `LANGFUSE_*` credentials are present, every in-process LLM call is mirrored into Langfuse while the run is live. A session-end flush backfills out-of-process children (geak, forge, specialist) and KEEP/REVERT decision Scores.
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
  (and Langfuse Generations) carry full prompt/response text. `redact_secrets`
  scrubs common token shapes (Bearer, `sk-`/`pk-`, GitHub tokens, some
  `KEY=value`) but is not a complete data loss prevention (DLP) filter — bare keys without a
  recognizable prefix (for example, raw AWS `AKIA…`) can slip through. The artifact
  packager also copies `reports/trace/*.jsonl` and, with the loose mode on by
  default (`HYPERLOOM_SESSION_PACKAGE_LOOSE`), drops them under `/workspace`
  for the Claw sync. If a session might contain customer code or secrets, define
  an explicit retention + access-control policy for both the Langfuse project
  and the `/workspace` package destination, and consider disabling live push
  or loose packaging for those runs.
* **`live push` + `backfill_langfuse` overlap**: Both derive the same
  `trace_id` from `claw_session_id`, so running the offline backfill *after* a
  live run re-emits the out-of-process children onto the same trace and can
  duplicate observations. Use one path per session, or treat backfill as a
  recovery tool only when live push did not run.
* **`flush_session` is idempotent, and retries only what failed**: the
  session-end reconcile runs as named steps (leftover halves, `ext/` shards,
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
  rows were sent: the `ext/*.jsonl` and `trajectory/*.jsonl` shards, the
  recipe-KB audit, specialist intel, forge steps and GEMM tuning logs. A call
  sends only the rows past that count, advancing it one row at a time and
  stopping at the first row it could not send, so that row is retried by the
  next call or the next leg. A receipt written by v1.0.0 through v1.1.3 carries
  the ext shard counts as `ext_rows_sent`, and a v1.1.3 receipt also carries the
  trajectory shard counts as `trajectory_rows_sent`; both are still read. `decision_trace.jsonl` is rewritten ts-sorted on
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

### `token_usage` section (in `session_breakdown.json`)

Every breakdown carries a top-level `token_usage` section: a promoted,
discoverable rollup of LLM token spend derived from the per-call ledger
(`reports/trace/llm_calls.jsonl` + `ext/*.jsonl`). It is purely derived from
`decision_trace.token_rollup`, so it always reconciles with that section. No
env var controls it; it is always present (zeroed on pre-trace sessions).

* `session_total`: whole-session total across every call. Three counter
  families are kept apart because they are billed and interpreted differently:
  visible tokens (`total_in` / `total_out`), cache tokens
  (`total_cache_creation` / `total_cache_read`), and hidden reasoning output
  (`total_reasoning_out` — reported by reasoning models, absent from the reply
  text and therefore *not* part of `total_out`). Two convenience figures sit on
  top: `total_in_out` (visible prompt + completion only) and `grand_total`
  (visible + all cache tokens + reasoning output — the all-in figure).
* `by_component`: per-agent breakdown (orchestration / kernel / critic /
  specialist / proposal_scorer / geak / forge / …), each with the same
  convenience totals.
* `by_phase`: per-phase breakdown (PRELUDE / ENABLEMENT / FRAMEWORK_AGENT / KERNEL_AGENT / SWEEP / CLOSE).
* `attribution`: `attributed_to_decisions` vs `unattributed` split plus
  `attributed_calls_pct`. Only calls that carry a `task_id` / `dyn_id` joining
  to a KEEP/REVERT or dynamic_action decision (for example, specialist subprocess
  turns) are attributed; orchestration / kernel / critic / proposal_scorer
  turns are LLM-internal and land in `unattributed` (this is expected, not a
  gap in the data).
* `timeline`: each `action_timeline` row annotated with the tokens that join
  to it on `task_id`. Rows whose action has no LLM spend show `tokens: null`
  (rather than a zero bucket) to make the sparsity explicit.

To get the single "total tokens for this run" number, read
`token_usage.session_total.grand_total` (all-in: visible + cache + reasoning)
or `.total_in_out` (visible prompt+completion only). Read
`.total_reasoning_out` on its own when comparing a reasoning model against a
non-reasoning one — the latter reports `0` there.

### Gateway attribution (`HYPERLOOM_LLM_ATTRIBUTION`)

`token_usage` above is Hyperloom's own account of its spend, assembled from
calls that report themselves to the ledger. The gateway keeps a second account,
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
| `component` | The producer, from the same vocabulary as `token_usage.by_component`. |
| `phase` | The phase the run had reached, matching `token_usage.by_phase`. |
| `type` | The action executing inside that phase, e.g. `kernel_opt`. |
| `operation` | What the individual call was for, e.g. `generate_candidate`. |

For LiteLLM these arrive as `x-litellm-tags` (a comma-separated list landing in
the `request_tags` column of `LiteLLM_SpendLogs`) and `x-litellm-trace-id` (the
session id, which sets the `session_id` column and propagates to nested calls).
A spend report can then be grouped by component or phase, or reconciled against
`token_usage` for one session.

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

An input YAML with `benchmark.agentx: enable`, passed through
`--benchmark-config`, automatically adopts the 2-D grading shape InferenceX
uses and persists AgentX session mode. Upstream InferenceX submissions sweep a
concurrency ladder, keep TTFT / ITL / TPOT percentiles separately, and
publish a **Pareto frontier** with E2E normalised interactivity as the x-axis
and token throughput per chip as the y-axis. It never collapses the axes into one
weighted number, and it has **no fixed interactivity target** — interactivity is
a frontier coordinate, not a constraint, so a point that trades interactivity for
throughput moves along the frontier rather than violating a rule.

Native AgentX measurements are materialized from that source YAML as Magpie
`agentx: enable` runs.
The canonical identity and launcher are intentionally separate from the local
checkpoint path. The pinned pair is Magpie v0.3.0 plus native launch overrides, custom-model replay, and client/eval
compatibility fixes at commit `d80eb4d3dad7fabe01ce81d049e2983adf2c86dd` and InferenceX commit
`408c015be4b22d14c69518643609669405507077`:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `HYPERLOOM_BENCHMARK_CONFIG` | No | None | Legacy environment alias for CLI `--benchmark-config`. Prefer the CLI flag. Fresh launches only. Resume restores the accepted materialized config, or the snapshotted source config when baseline was not accepted yet. |
| `HYPERLOOM_AGENTX` | No | Off | Enables Magpie native AgentX for fresh sessions, including Hyperloom optimization. Persisted epoch-1 sessions keep their legacy client and epoch-2 sessions keep their native measurement-only contract. Native `benchmark.agentx: enable` sets AgentX session mode automatically; an explicitly false environment value conflicts and is rejected. |
| `HYPERLOOM_IMAGE` | No | Effective resolved image | Optional strict consistency assertion for native local mode. `benchmark.docker_image` overrides/pins the effective recipe image; otherwise the recipe default is used. If this variable already exists, it must match exactly. It does not attest the running container. |
| `AGENTX_MODEL_ID` | Native: only without `benchmark.model` | None | For native AgentX, fallback for the exact model id from the selected InferenceX recipe (for example `amd/GLM-5.2-MXFP4`). A separate CLI `--model` may name the local checkpoint and is emitted as `MODEL_PATH`; omitting it uses the source model id remotely. |
| `AGENTX_SERVER_SCRIPT` | No | Resolved by Magpie | For managed AgentX, the resolved client is `srt_agentic.sh`; serving comes from Magpie's resolved specification. Persisted shell-launcher sessions retain their original launcher identity and restrictions. |
| `INFERENCEX_PATH` | No | Installer pin | Legacy fallback for source `benchmark.inferencex_path`. The source path nominates a preferred writable checkout. Preflight replaces a missing or wrong-revision path with a pinned clone; an explicit correct but non-writable checkout fails. A simultaneously supplied, different ambient path conflicts with the source before preflight. Managed AgentX accepts the repository root or `inferencex-e2e/` project directory, including spaces. Saved shell-launcher sessions retain their original path restrictions. |
| `AGENTX_MODE` | No | `canonical` | Environment equivalent for enabled native AgentX of `benchmark.agentx.mode`. `canonical` uses the 3600-second native protocol and is required by `optimize`. `fast` uses 1200 seconds and is a direct-Magpie diagnostic; its non-publishable result is rejected as a Hyperloom baseline. |
| `AGENTX_RECIPE` | When inference is ambiguous | Inferred by Magpie | Environment equivalent for enabled native AgentX of `benchmark.agentx.recipe`: an exact recipe key from the pinned InferenceX config. |
| `AGENTX_CONFIG_FILE` | No | InferenceX `configs/amd-master.yaml` or `nvidia-master.yaml` | Environment equivalent for enabled native AgentX of `benchmark.agentx.config_file`: an explicit recipe-config path understood by Magpie. |
| `AGENTX_SELECTOR` | When one recipe/concurrency has multiple arms | `{}` | Environment equivalent for enabled native AgentX of `benchmark.agentx.selector`, encoded as a JSON object, for example `{"tp":4,"kv_offloading":"dram","kv_offload_backend":"hicache"}`. Prefer the YAML object with `--benchmark-config`. |
| `AGENTX_FAILED_REQUEST_THRESHOLD` | No | `0.10` | Environment equivalent for enabled native AgentX of `benchmark.agentx.failed_request_threshold`, from `0` through `1`. |

The source YAML can provide `benchmark.model`, `benchmark.precision`,
`benchmark.framework`, `benchmark.runner_type`, `benchmark.run_mode`,
`benchmark.benchmark_script`, optional `benchmark.inferencex_path`, effective
`benchmark.docker_image`, `benchmark.envs.CONC`, optional topology assertions,
and either scalar `benchmark.agentx: enable` or the object form:

```yaml
agentx:
  enabled: true
  recipe: glm5.2-fp4-mi355x-sglang-agentic-mtp  # only if inference is ambiguous
  selector:                                     # only if this CONC has multiple arms
    tp: 4
    kv_offloading: dram
    kv_offload_backend: hicache
```

Hyperloom checks those source fields before session state is seeded. The image
field overrides/pins the effective recipe image and is included in its
fingerprint; Hyperloom does not inspect the runtime to prove the actual
container. The selected recipe owns logical TP/PP/PCP/EP, `MODEL_PREFIX`,
`KV_OFFLOADING`, `KV_OFFLOAD_BACKEND`, and `TOTAL_CPU_DRAM_GB`; users do not copy
those fields into `benchmark.envs`. In particular, `envs.TP` is not an arm
selector. A concurrency not present in the recipe, or an ambiguous point
without `agentx.selector`, fails before launch. Hyperloom pre-resolves each
point through the same benchmark interpreter that will run Magpie. Native
InferenceX owns the duration (3600 seconds in canonical mode, 1200 in fast
mode) and configures a 393-trace dataset-entry cap. That value is a loader
ceiling, not a guarantee that 393 traces, sessions, or requests survive
availability and context-length filters. Native `AGENTX_DATASET` and
`WEKA_LOADER_OVERRIDE` overrides are
rejected. A pre-existing `HYPERLOOM_IMAGE` must exactly match the effective
resolved image. In contrast,
`AGENTX_DURATION` and `AGENTX_NUM_ENTRIES` still control legacy measurements
and saved compatibility-profiler sessions. They do not change managed AgentX measurement
duration or corpus size. Legacy `AGENTX_DATASET` and `WEKA_LOADER_OVERRIDE`
remain supported on the legacy backend.

Hyperloom owns the fixed concurrency of native measurement rounds through
CLI `--conc` or source/materialized `benchmark.envs.CONC`. It removes
`benchmark.agentx.concurrency` from the source YAML because Magpie gives that
field precedence over `envs.CONC`. New native sessions retain the optimizer's
concurrency sweep. Saved epoch-2 sessions keep sweep disabled; explicitly
enabling it on those sessions fails preflight.

When omitted, Hyperloom fills outer `--tp` with the resolved `TP×PP×PCP`
physical GPU count and `--ep` with recipe EP. Explicit values are exact
consistency assertions. Topology-changing AgentX sweeps are rejected. The
pinned launchers copy physical ROCR values back into logical
`HIP_VISIBLE_DEVICES`, so Hyperloom derives `gpu_selection.auto=false` and
`ROCR_VISIBLE_DEVICES=0,...,N-1` when omitted. Explicit masks are assertions;
nonzero, reordered, short, or long masks are rejected rather than remapped.

Hyperloom's native bridge currently supports local, single-node SGLang and
vLLM. It automatically bypasses Hyperloom's outer Ray actor; leave
`INFERENCE_OPTIMIZER_RAY_EXEC` unset or set it to `0`. An explicit true value is
rejected. Multi-node/disaggregated AgentX, `server_lifecycle`, and Atom are not
supported.

Fresh native sessions use the Magpie-managed launch-overrides contract.
Candidate arguments, environment controls, and source overlays have distinct
execution identities, while the canonical workload fingerprint stays fixed.
Unsupported controls fail closed. Existing epoch-2 sessions preserve the old
measurement-only contract and immutable materialized execution fingerprint.

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

Hyperloom reads `e2e_norm_intvty_p90` from native Magpie's
`agentx_metrics.latency_seconds`. InferenceX computes it as
`1 / P90(E2EL_s / OSL)`, so it is already the slow-tail rate for a
`LARGER_IS_BETTER` metric; Hyperloom does not multiply it by 1000 or invert it
again. Older profiler-compatibility artifacts carry AIPerf's P10 of the
per-request `OSL / E2EL_s` rate. The two finite-sample estimators can differ;
profile output is diagnostic and does not replace the native measurement used
for KEEP/REVERT.

TTFT is included in E2EL, unlike per-user `1/ITL`; on a ~114k-prompt replay TTFT
is most of what a user waits for so grading on `1/ITL` would miss it.

Default-on for AgentX runs, explicit opt-in via `HYPERLOOM_PERF_METRIC=intvty_v1`
otherwise. Source `benchmark.agentx: enable` stamps `benchmark_mode=agentx` at
seed; the fresh-launch `HYPERLOOM_AGENTX=1` switch does the same — so a round in a
subprocess that never inherited the env var still grades on the agentic axis.
Serving frameworks only; scriptable frameworks (xDiT, custom) keep
output-throughput grading.

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_PERF_METRIC`        | `intvty_v1` in AgentX session mode, else output tput | `intvty_v1` grades E2E normalised interactivity P50 (median) as the primary objective, with the P90 slow tail and output throughput each held as a guard. Reported in the final summary as `grading mode`. |
| `HYPERLOOM_PERF_NOISE_PCT`     | `5.0`                         | Noise band in percent applied to the guards. A candidate whose slow tail or output throughput sits within this band of the anchor is not considered worse on that axis; the median has its own fixed bar and is not subject to the band. The default is the top of the 1–5% run-to-run noise upstream records for this workload. An unparseable value raises `EnvValueError` naming the variable, rather than grading against a band nobody chose. |
| `HYPERLOOM_ALLOW_UNVERIFIED_SUBMISSION` | Unset (fail closed) | Truthy accepts a measurement whose submission verdict is absent or undetermined (`submission_valid=None`). A measurement the scenario explicitly judged invalid (`submission_valid=False`) is always rejected regardless of this flag. Applies to every measurement the run accepts (baseline, explore, kernel, sweep), not only the baseline — an unverified measurement makes every gain derived from it unverifiable. |
| `INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC` | `7200` | Initial server-boot budget written by the persistent-server lifecycle configuration helper. Actual benchmark launches synchronize this field to `INFERENCE_OPTIMIZER_BENCHMARK_TIMEOUT_SEC`; this variable is not an additional benchmark deadline or a way to extend one. Non-benchmark lifecycle callers that do not perform that synchronization retain their own boot budget. |

Epoch-4 native profiling derives a diagnostic Magpie run from the accepted
candidate configuration. Magpie owns server lifecycle, measured-phase gating,
step capture, and framework profiler arguments. The diagnostic torch settings
are `num_steps`, `num_profiles`, `start_seconds`, `interval_seconds`,
`capture_timeout_seconds`, `flush_timeout_seconds`, and `detailed_annotations`.
The first delay is measured from AIPerf measurement start; each later interval
starts after the previous trace flush. Magpie computes the capture upper bound
from the remaining measurement duration and reduces oversized requests.

Hyperloom consumes `agentx_metrics.profile_capture` and chooses one complete
capture for kernel/roofline analysis. It preserves all per-round artifacts and
the requested, planned, effective, and completed counts. Normal count reduction
is not a capture failure. A failed run, cancellation, or incomplete trace is
still a failure. Detailed annotations require a compatible instrumented
framework and are configured by Magpie; an accepted runtime/source overlay is
preserved, and installed framework files are not silently patched.

Diagnostic runs deliberately set `benchmark_valid=false` and
`publishable=false`; successful capture does not promote their performance
numbers to baseline or KEEP. System profiling and gap analysis remain unsupported
for native AgentX. Saved epoch-3 sessions keep their earlier compatibility
profiler and its controls. Epoch-2 sessions keep their measurement-only contract.
GEAK proposals still require Critic-reviewed source integration and canonical
native AgentX remeasurement before acceptance.

For native results, Hyperloom requires both `benchmark_valid=true` and
`publishable=true`, an `agentic-coding` scenario, and a valid recipe fingerprint,
including exact recipe-to-launch and recipe-to-raw fingerprint matches. It
accepts GPU count only from a trusted, fingerprint-bound report, Magpie config
snapshot, materialized workload record, or InferenceX raw aggregate. The raw
aggregate retains measured token-length distributions and cache metrics.
Magpie's `publishable` bit attests the canonical AgentX protocol and recipe
fingerprint. Hyperloom separately binds the selected recipe, resolved server
specification, client sources, and pinned checkout. It does not cryptographically attest the
actual outer image. The execution identity covers the resolved
`BenchmarkConfig` and the effective, scrubbed launcher environment for the
audited server/framework/runtime controls. It intentionally excludes
credentials, cache routing, output paths, and unrelated login-shell variables.

### MLPerf agentic backend

The explicit MLPerf backend keeps its existing client path. Fresh AIPerf
sessions use native Magpie AgentX. Combining a native AgentX source YAML with
`HYPERLOOM_AGENTIC_BACKEND=mlperf` is rejected because they select different
workloads.

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
| `HYPERLOOM_AGENTIC_BACKEND` | `aiperf` | `mlperf` selects the MLPerf agentic client. Any other value is aiperf. |
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

### Saved compatibility-profiler sessions

For saved epoch-3 AIPerf sessions, profiling starts when AIPerf reports its measured phase. The legacy
`AGENTX_PROFILE_WARMUP_S` delay is ignored. `AGENTX_PROFILE_WINDOW_S` controls
the capture window and defaults to 20 seconds; phase waiting is bounded by the
materialized benchmark timeout. Capture lifecycle status is
written to a per-invocation `capture-status.json`; the adjacent
`trace-manifest.json` records the selected primary and per-rank traces.
Benchmark measurement success and trace-capture success are reported
independently. AgentX multi-node profiling is currently rejected because its
legacy fixed-delay capture is not aligned with the AIPerf phase signal.

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

## Variables intentionally not exposed

These are read by `os.environ` somewhere in the codebase but are
internal-only — do not set them by hand:

* `HYPERLOOM_KERNEL_AGENT_ROOT`: internal CLI-only handoff to the
  kernel subprocess (Python constant `_KERNEL_AGENT_ROOT_ENV`).
* `HYPERLOOM_HOST_PROBE`, `HYPERLOOM_HOST_PROBE_DEEP`,
  `HYPERLOOM_HOST_PROBE_DIR`, `HYPERLOOM_HOST_PROBE_ROOTS` (and the
  `..._MAX_SITES` / `..._ARG_SAMPLES` caps): the same for the host-stall
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
