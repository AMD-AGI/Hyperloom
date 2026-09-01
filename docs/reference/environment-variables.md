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
| `OPENAI_BASE_URL`      | Conditional | —    | OpenAI-side endpoint. Required together with `OPENAI_API_KEY` to enable Codex. An OpenAI-only configuration drives Orchestration through the Codex backend; Claude and GEAK stay disabled.                                                                        |
| `OPENAI_API_KEY`       | Conditional | —    | OpenAI-side key. Pairs with `OPENAI_BASE_URL`. Never borrowed from the Anthropic side.                                                                                                                               |
| `OPENAI`<br>`_CUSTOM_HEADERS` | No | —    | Extra request headers for the OpenAI side. Same shape as `ANTHROPIC_CUSTOM_HEADERS`; set it whenever you set `OPENAI_BASE_URL` against a gateway that authenticates on its own header. |
| `CLAUDE_MODEL`         | No       | Derived from `ANTHROPIC_BASE_URL` | Orchestration model id on the Anthropic side. Falls back to the endpoint default, then the project-wide `DEFAULT_CLAUDE_MODEL`. `GEAK_CLAUDE_MODEL` and `FORGE_CLAUDE_MODEL` inherit from it.                                                                    |
| `CODEX_MODEL`          | No       | Derived from `OPENAI_BASE_URL`    | Model id on the OpenAI side, used by the Codex backend. `FORGE_CODEX_MODEL` inherits from it. Model settings are never borrowed across providers.                                                                    |
| `GEAK_API_KEY`         | No       | —    | Internal alias, never derived from either side. GEAK runs on the Anthropic side (`ANTHROPIC_*` + `GEAK_CLAUDE_MODEL`); set this only to point GEAK elsewhere.                                                                                                                              |
| `GEAK_BASE_URL`        | No       | —    | Internal alias, never derived from either side. Set it only to point GEAK at a different endpoint than the Anthropic side.                                                                                                                          |
| `GEAK_CLAUDE_MODEL`   | No       | Inherits `CLAUDE_MODEL` | GEAKv4 Claude Code workflow model id.                                                                                                                                                           |
| `FORGE_CLAUDE_MODEL`  | No       | Inherits `CLAUDE_MODEL` | Forge Claude backend model id (fusion, rewrite, collective). Set when Forge should use a different Claude model than orchestration.                                                                                   |
| `FORGE_CODEX_MODEL`   | No       | Inherits `CODEX_MODEL`  | Forge Codex backend model id (fusion, rewrite, collective). Set when Forge should use a different Codex model than the OpenAI-side default.                                                                          |
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
| `HYPERLOOM_`<br>`RUNTIME_DIR`             | No                   | `$USER_DATA_PATH/runtime` (installer)                               | Private writable runtime state. Codex SDK turns create a unique mode-`0700` `CODEX_HOME` here and remove it after the SDK client closes. When unset, Codex uses the first safe declared output root, then a run-local working directory; it never falls back to `/tmp` or a source checkout. |
| `INFERENCE_`<br>`OPTIMI`<br>`ZER_CU`<br>`RRENT_S`<br>`ESSION_DIR` | No (set by CLI) | Set at session boot | Absolute path to the active session directory. Written by the CLI when a session starts and inherited by every benchmark subprocess; session-path resolution prefers it over scanning `USER_DATA_PATH`. Do not set by hand. |
| `HYPERLOOM_ROOT`                          | No                   | `$HYPER`<br>`LOOM_R`<br>`UNTIME_`<br>`DIR/sou`<br>`rce-mirrors`                            | Legacy source-mirror root kept for compatibility. Current open-source dependency checkouts default to the repo-local cache root (`${HYPER`<br>`LOOM_CA`<br>`CHE_DIR:-`<br>`$REPO_ROOT`<br>`/.cache}`), not this path. |
| `HYPERLOOM`<br>`_CACHE_`<br>`DIR`                          | No                   | `$REPO_ROOT`<br>`/.cache`                      | Writable, repo-local base for auto-cloned open-source deps (TraceLens, Magpie, etc.), cloned per revision as `<name>@<sha>`. Not under `$TMPDIR` so a reaper cannot wipe it mid-run. |
| `KERNELFORGE`<br>`_PROJECT_`<br>`ROOT`              | No                   | `$USER_DATA_PATH/kernelforge`, else `~/.cache/hyperloom/kernelforge` | Writable root for forge's own state and for resource-tree overrides. Holds the learned knowledge base (`knowledge_base/<backend>/learned/`), the tuning DB, postmortems and `forge_experiments/`. A subtree placed here also **overrides the copy packaged inside `kernelforge`** — a `serving_patches/` or `examples/` directory under this root wins over the shipped one, which is the supported way to try a patch or a task without editing site-packages. Must be writable: it deliberately never resolves to the installed package directory or to the cwd. **This is the replacement for the removed `FORGE_PATH`**, which nothing reads any more — a stale `FORGE_PATH` is still forwarded (the `FORGE_` prefix is on the dotenv allowlist) and then ignored. |
| `SKIP_FORGE`<br>`_PROFILING`               | No                   | Unset (the extra is installed) | Set to `1` to make `install.sh` skip `pip install -e "$REPO_ROOT[forge-profiling]"`. That extra is rocprof-compute's own dependency set (~20 wheels, including the exact `kaleido==0.2.1` / `astunparse==1.6.2` pins ROCm 7.2.x requires); without it forge's profiler degrades to the lightweight PMC path instead of System Speed-of-Light + roofline. Installed by default on purpose — the previous gate made this a silent skip on every pod. |
| `MAGPIE_PATH`                              | No                   | Resolved from installed `Magpie` package unless explicitly set                               | Magpie package root for benchmark wrappers and patch inspection.                                                                                                                                            |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_MODEL_PATH_ROOTS` | No | Built-in model roots such as `/models` and `/shared_nfs` | `os.pathsep`-separated allowlist for absolute model paths restored from `state.json` during a resume. HuggingFace-style repo IDs remain allowed. Set this when production models live outside the built-in roots. |
| `SESSION_DIR`                             | No (robustness-agent)| Scan known paths                                                   | Path containing `storage/coordinator.db`; the robustness FindingSink writes under `{session_`<br>`dir}/ag`<br>`ents/ro`<br>`bustne`<br>`ss/fin`<br>`dings/`<br>`{sess`<br>`ion_id}.jsonl`.                                       |
| `INFERENCE_`<br>`OPTIMI`<br>`ZER_SES`<br>`SION_DIR` | No (monitor / multi-node) | Unset                                                   | Explicit session directory for the Robustness Monitor (`tools/robustness_`<br>`monitor.sh.example`), which prefers it over `.session_dir` in the launch-info JSON. Multi-node crash-log collection reads it as a last-resort session root. Point it at one session dir, never at `$USER_DATA_PATH`. |

---

## Workload configuration

Set with CLI flags, not env vars. Pre-set `ISL` / `OSL` / `CONC` / `PRECISION` /
`TP` / `EP` env vars are ignored and overwritten (`GPU_TYPE` is a fallback when
`--gpu-type` is omitted).

- **Model / workload shape:** `--model`, `--model-class`, `--framework`,
  `--framework-version`, `--precision`, `--tp`, `--ep`, `--isl`, `--osl`,
  `--conc`, `--max-model-len`, `--profile-osl`.
- **Goal / budget:** `--target-gain`, `--max-hours`, `--target-summary`,
  `--target-tput`, `--compare-against-gpu`.
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
  `--no-framework-agent`, `--no-framework-local-explore`, `--no-kernel`,
  `--no-eval`.
- **Agent models:** `--claude-model`, `--codex-model`.
- **Session / resume:** `--resume-from`, `--force-resume`, `--reset-state`.
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
| `INFERENCE_OPTIMIZER`<br>`_REQUIRE_FRAMEWORK`<br>`_ACCURACY` | On | Same gate for a framework source patch authored by a specialist. Same disable spellings. |
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

## Kernel-opt backend selection

The following variables control the kernel optimization backend ladder.

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `KERNEL_OPT_BACKEND_ORDER`     | Unset (resolves to `geak`)    | Selects the kernel-opt backend. Unset resolves to `geak` (whole-pipeline GEAK owns the KERNEL phase); the bare-metal installer and the Slurm launchers export `${KERNEL_OPT_BACKEND_ORDER:-geak}` on top of that. **Only an exact, case-insensitive `forge` opts into the per-kernel forge backend** (`forge_explicitly_enabled` in `common/env.py`). Despite the historical name, a comma list is not parsed: `forge,geak` silently stays on `geak`, as does any other value, legacy alias, or payload override. |
| `KERNEL_OPT_MAX_PARALLEL`      | `8` (GPU-adaptive cap)        | Max parallel kernel-opt attempts per request (per-kernel race fan-out). The runtime caps this by visible GPUs and per-attempt GPU reservation when it can detect them.                                                                                                                            |
| `HYPERLOOM_GEMM_SHAPE_CAPTURE` | `1`                           | Enables automatic runtime GEMM-shape capture for eligible single-node dense vLLM Forge tuning when no explicit shape input is available. Block-FP8 first reuses shapes from the TraceLens-selected steady-state trace of a successful Roofline with exactly matching model, workload, server arguments, environment, and backend controls. Missing or stale evidence triggers the same standard Roofline/ProfileExecutor/TraceLens steady-state pipeline as a fallback. Set to `0` to preserve the no-capture path. |
| `HYPERLOOM_GEMM_SHAPE_CAPTURE_TIMEOUT_SEC` | `1800`          | Timeout in seconds for the dense vLLM TunableOp recording benchmark. Block-FP8 fallback uses the standard Roofline/ProfileExecutor timeout. Values below `60` are clamped to `60`. |
| `INFERENCE_OPTIMIZER`<br>`_KERNEL_OPT_MAX_PARTIAL` | Unset           | Cap on how many `PARTIAL` kernel-opt verdicts an action can yield before it short-circuits to `NEEDS_REVIEW`. Useful for keeping budget contained when GEAK is consistently timing out.            |
| `KERNEL_OPT_BACKEND_BUDGET_MIN` | `90`                         | Wall-clock budget in minutes for one optimization, mirrored by the `kernel_optimization.py` wrapper. The env deliberately wins over the payload `budget_minutes`, which is LLM-authored from a prompt template, so an operator raising the budget is not silently overridden. forge-loop reserves half the window for finalize, so `90` leaves roughly 45 minutes of real iteration. |
| `HYPERLOOM_KERNEL_OPT_MIN_GPU_PCT` | `5.0`                     | GPU-time share a reusable hot kernel must clear to be worth a dispatch, in percent. Three readers share it and must agree, or the report explains a skip the dispatcher never made: the batch filter that selects candidates, the phase-advance gate that decides KERNEL still owes work, and the report's unattempted-reason breakdown. It was `10.0` until a 60-layer sparse-MoE decoder showed the assumption behind that number — that hot kernels concentrate — does not hold: nothing but a graph-launch wrapper reached double digits, the largest real operator sat at 9.47%, and the batch dispatcher selected nothing for six hours. Lower it when a trace's rewritable candidates cluster below the default and the operators above them are vendor binaries; a dropped candidate is reported as `below_min_gpu_pct=<value>` rather than as a failed attempt. |
| `AITER_LOG_TUNED_CONFIG`       | `1` (set for every serving run) | Makes aiter log each tuned-config lookup it *hits*, not only the ones it misses. Two checks have no input without it: the GEMM demand list, which learns the shapes the runtime actually asks for (config-derived shapes covered 0.4% of them), and the apply verdict, which cannot tell "the tuned table was never read" from "it was read and did not help". A scan of 60 production logs found it set in none of them, so it is now injected by default. An operator value wins — set `0` to turn hit logging off, at the cost of both checks going inconclusive. Every miss already prints a line regardless of this setting; hit logging adds roughly one line per lookup that succeeds. |
| `HYPERLOOM_GEMM_PAIRED_PAIRS`  | `0` (off)                     | How many interleaved baseline/tuned pairs to re-measure before a GEMM tuning KEEP is reported as confirmed. One end-to-end measurement cannot separate a gain from drift on this fleet: three rounds of a single unchanged configuration spanned 58%, and one controlled repeat moved 16%. Each pair costs two extra benchmark rounds. When `0`, the gain is still promoted — it is the best number available — but recorded as an unpaired block comparison rather than presented as a paired one. |

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

---

## Collective optimization lane

The collective lane is Coordinator-owned: it is dispatched directly at KERNEL
entry, never as an agent request. It requires `TP > 1`, a latest-snapshot
`Exposed Communication %` of at least 1% as parsed from the TraceLens executive
summary, a `trace_analyze` snapshot, and a source-resolved custom collective
candidate (`all_reduce`, `reduce_scatter` or `all_gather`) — vendor RCCL/NCCL
symbols are opaque binaries and never qualify.

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_SKIP_COLLECTIVE`    | Unset (lane enabled)          | Truthy (`1` / `true` / `yes` / `on`) disables the collective lane outright, before any gate is evaluated.                                                                                          |
| `HYPERLOOM_COLLECTIVE_ONLY`    | Unset                         | Truthy runs ONLY the collective lane at KERNEL entry — GEAK, fusion, and per-kernel `kernel_opt` are all skipped — and hints `skip_to_sweep` once the lane settles. Also the way to reach the lane while `KERNEL_OPT_BACKEND_ORDER` selects `geak`, which otherwise owns the whole phase. Mirrored into the `collective_only_mode` SharedState field. |
| `HYPERLOOM_COLLECTIVE_KEEP_PCT` | `1.0`                        | E2E `KEEP` threshold in percent for the collective integrate. Must parse as a finite, non-negative float, otherwise the integrate fails loudly rather than defaulting.                             |
| `HYPERLOOM_COLLECTIVE_ALLOW_INFERRED_SHAPES` | Unset (disabled) | Truthy allows a source-resolved collective to borrow shapes from the trace's sole all-reduce workload family. The default rejects this inference because those shapes were not observed on that device symbol. |
| `FORGE_COLLECTIVE_TIMEOUT`     | `14400` (4h)                  | Wrapper timeout in seconds for one forge-collective campaign; a collective iterates over N ranks per benchmark, hence the wide default. A payload `timeout` takes precedence over the env.          |
| `FORGE_COLLECTIVE_AGENT_TIMEOUT` | Unset (wrapper default)     | Per-agent timeout in seconds, forwarded to forge-collective as `--agent-timeout-sec`. A payload `agent_timeout_sec` takes precedence.                                                              |

---

## Kernel source resolution

A kernel candidate must resolve to a real source file before any backend can
rewrite it. Resolution runs as a ladder: curated dictionary, then the
trace-derived launcher frame, then a name grep. All three are deterministic and
require no configuration. Agent analysis might add the review stage below.

Every run writes `kernel_source_resolution.json` next to the candidate report.
It answers one question per hot kernel — which file defines it, and which tier
decided that — in a versioned schema (`schema_version`, currently `1.0.0`), so
consumers and triage read a contract rather than candidate internals.

### Candidate review

One agent session may audit the finished candidate table when
`--analysis-route agent` is used. The `deterministic` route never runs it, and
keeps its no-model guarantee by not reaching the stage at all.

The stage is tool-enabled rather than a completion call because the deterministic
tiers' real failure mode is not coming up empty but coming up confidently wrong,
and a model ranking paths by keyword off a prompt assembled in advance cannot
tell the difference — it never sees what a kernel actually *is*.

**What it is handed: paths, not contents.** The raw table, the resolution audit,
the TraceLens report, the model directory and the framework source roots — as
locations. The session reads with `Read`, `Grep` and `Glob`, so it opens what the
evidence leads it to instead of what was guessed to be relevant, and can confirm
a file defines the kernel it is credited with. A mangled vendor symbol is
demangled by the host before the session starts and arrives in the table as
`device_kernel_name_demangled` — that was the one job a shell was granted for,
so granting one is no longer necessary.

**It cannot write to the code under optimization.** That is the whole guarantee,
and there is no detection layer behind it. On the Claude backend `Write` is the
answer channel and is refused outside the run directory; every other tool,
including any shell, is denied by a default-deny callback, so a tool introduced
by a later SDK arrives refused rather than pre-authorised — and an SDK that will
not accept that callback gets no session at all. On the Codex backend the
containment is the OS sandbox: this stage states `workspace-write` rather than
inheriting the deployment's mode, so it cannot run under a configured
`bypass` (see [Codex (OpenAI) agent sandbox](#codex-openai-agent-sandbox)). The
two are not equivalent — Codex has a shell and its own file tools, confined by
the sandbox rather than refused per call — and the mode in force is logged,
because the backend is chosen by credentials rather than by an operator.

**What it costs.** One session per analysis, not one call per candidate. 900
seconds per attempt and two attempts by default. On the Claude backend each wait
for the next SDK message is bounded separately, because a stalled gateway would
otherwise hold the full window (see
`HYPERLOOM_TRACELENS_STREAM_IDLE_TIMEOUT_SEC` and
`HYPERLOOM_TRACELENS_TOOL_IDLE_TIMEOUT_SEC`, which this stage reuses). The Codex
entry point is one-shot and exposes no stream to bound, so a stalled Codex review
costs the full 900 seconds. `--dry-run` skips the stage entirely: it plans, and
nothing dispatches from the table it publishes.

**Model and backend.** `HYPERLOOM_LLM_SOURCE_MODEL` overrides everything. With
it unset the backend follows the configured credentials — Anthropic-only selects
Claude, an OpenAI side selects Codex — and the model comes from `CLAUDE_MODEL`
or `CODEX_MODEL` respectively. Model settings are never borrowed across
backends.

<div class="callout warn">

**Authority: proposals, bounded four ways.** The session may revise where a
kernel lives, supply the operand dims a graph-launched kernel never recorded,
correct a harness list, and refuse a candidate as not worth a session. It may
not touch what the trace measured — GPU share, durations, call counts, the keys
a row is joined by — because the impact ranking and the closing gain figure are
computed from those.

- A revised path must resolve under a known framework root; symlinks cannot
  escape it. TraceLens-style `path.py(247): function` answers are split into an
  openable path plus line and function metadata, and an unverifiable path is
  rejected with the original left standing. A proposed harness path is held to
  the same test — existence alone would let a proposal point the measurement at
  a file outside the traced tree, and a backend runs what that list names.
- A candidate the active finder already resolved is not overridable: reading the
  tree cannot beat knowing which symbol the binary exports. Nor is one matched to
  a vendor operator playbook, whatever tier resolved the path the playbook anchor
  replaced — its `source_file` points at a task bundle, and pointing it back at
  framework source would route the candidate to a backend with nothing to
  rewrite there.
- A restrictive routability hint is honoured, a permissive one is not —
  `classify_patchability` stays the only gate that admits a kernel, so a hint
  cannot talk it into dispatching something the deterministic rules rejected. A
  refusal carrying no stated reason is dropped as an echo of the input.
- Operand dims are taken only where the trace recorded none, and carry their own
  provenance: a session cannot claim `torch_trace` for a dim it derived.

</div>

The stage cannot fail a run. No model configured, a gateway error, a timeout, an
unparseable reply or an unforeseen fault all leave the deterministic table
standing and record an error-severity trace-health warning: losing the audit
costs some candidates, while failing the run costs the hours of benchmarking
behind the trace.

**Artifacts.** `kernel_candidates.raw.json` is the deterministic table,
`kernel_candidates.json` the reviewed one, and
`kernel_candidates_revisions.json` records what changed and why — so a bad
dispatch can be traced to the stage that caused it. Only the reviewed table is
resolvable as a backend's candidate source. A completed review also rebuilds
`kernel_source_resolution.json`, so the artifact a human reads and the table a
backend is handed cannot name different files for the same kernel.

### Source egress

The session reads the tree on the host rather than being handed a prompt
assembled from file contents up front, so what leaves is whatever it quotes back
to the model. It cannot write to that tree — see the tool scope above — so this
section is about disclosure only. Two boundaries apply to the material the prompt
does carry.

**The serving command line is never forwarded verbatim.** The stage needs
backend flags — the same MoE operator dispatches differently under
`--moe-runner-backend triton` and `aiter` — but `EXTRA_*_ARGS` also carries
credentials, model paths and user data. It is therefore tokenised, and only
flags on an explicit allowlist of backend selectors survive. A denied flag
consumes its value too, so the value cannot reappear as a stray token. Every
surviving value is dropped unless it is a short selector token. URL userinfo or
queries, authorization headers, JWTs, control characters, non-finite numbers,
vendor prefixes such as `sk-`, and long opaque strings are rejected. An
unbalanced quote discards the whole line rather than risking a partial parse.

**Environment variables** follow the same discipline: an explicit allowlist of
path-selecting names, with the secret-name pattern applied on top.

**Model config is allowlisted too.** Only fields that select architecture,
expert layout or kernel format are included. Inside `quantization_config`, only
explicit quantization selectors survive; arbitrary vendor fields, nested
metadata and credential-shaped values are dropped.

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_`<br>`LLM_SOURCE`<br>`_MODEL` | Unset | Optional candidate-review model override. With it unset the model comes from `CLAUDE_MODEL` or `CODEX_MODEL`, whichever matches the backend the configured credentials select; there is no cross-backend fallback. |

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

`bypass` is a deliberate double opt-in. Set both
`HYPERLOOM_CODEX_SANDBOX_MODE=bypass` and
`HYPERLOOM_CODEX_EXTERNAL_SANDBOX=1`; the second variable confirms that an
external container or sandbox already enforces the required isolation. It does
not create that boundary. A confirmed bypass maps to Codex full access even
when no writable roots are declared, because the external sandbox is
authoritative. Under the contained modes, no writable roots remains
`read-only`. Unknown modes and incomplete bypass configuration fail
immediately.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_`<br>`CODEX_SANDBOX_MODE` | `workspace-write` | `workspace-write` restricts writes to the session directory plus declared output roots; `read-only` forbids writes; `bypass` selects Codex full access only when the external-sandbox confirmation below is also set. |
| `HYPERLOOM_`<br>`CODEX_EXTERNAL_`<br>`SANDBOX` | Unset | Set exactly to `1` only when an external isolation boundary is already active and `HYPERLOOM_CODEX_SANDBOX_MODE=bypass`. Setting this alone has no effect and never weakens the default sandbox. |

---

## Single-node Ray GPU scheduling

These variables tune the single-node Ray execution path (active when
`INFERENCE_OPTIMIZER_RAY_EXEC=1` and `--nodes=1`). They have no effect on
multi-node runs or when the Ray backend is disabled.

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_OPTIMIZER_RAY_GPU_PENDING_LIMIT` | `4` | Maximum number of GPU specialists that can be simultaneously in-flight (pending Ray scheduling + running) on the single-node Ray path. Ray still serializes execution on the physical GPU(s) using `num_gpus`; this limit caps how many actors can queue behind the current one. Floored at `1`. **Reduce to `1` or `2` when GPU memory or per-process overhead is a concern** (each queued actor holds a Ray worker slot even while it waits). |
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

The accuracy floor shared by the eval trigger and the enablement KEEP gate is the
fixed constant `_accuracy_gate.DEFAULT_ENABLEMENT_ACCURACY_FLOOR` (`0.05`). It is
a collapse guard rather than a quality bar: a score of exactly `0.0` always fails,
otherwise `score >= floor` passes.

---

## Framework / source-tree discovery

The following variables configure framework source discovery and path overrides.

| Variable                                          | Default                                                                | Description                                                                                                                                            |
|---------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `INFERENCE_`<br>`OPTIMIZER_`<br>`FRAMEWORK_`<br>`SOURCE_ROOTS`      | Union with `/sgl-workspace`<br>`/{aiter,sglang`<br>`,vllm}`                        | Colon-separated list of source roots used by PolicyGate and flag discovery. Populated automatically by `src/hyperloom/inference_optimizer/assets/install.sh`'s `_probe_framework_source_roots` step (using `hyperloom.orchestrator.framework.paths.probe_framework_source_roots_for_env`).   |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_RESCUE_PATHS`                | Unset                                                                  | Colon-separated list of extra directories the harvest step scans for stray `result.json` files written outside the session dir (InferenceX-native scripts that hardcode `--result-dir`). |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_AITER_JIT_DIR`               | Aiter default                                                          | Override the aiter just-in-time (JIT) cache root. See [Targeted builds (Rung 5)](#targeted-builds-rung-5).  |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_STRICT_PATHS`                | `1` when CLI bootstraps                                                | When `1`, missing path env raises instead of falling back to discovery. Set by the CLI at session start; do not override unless debugging.              |
| `HYPERLOOM_`<br>`SGLANG_PA`<br>`TCH_EXACT`<br>`_VERSIONS`           | Unset                                                                  | Pin the sglang server-patch step to specific upstream versions; advanced compatibility option.                                                          |
| `HYPERLOOM_`<br>`ENABLE`<br>`_PATCH`                          | `1`                                                                    | Set to `0` to skip the in-place server patch step (useful when the upstream is already pre-patched).                                                    |
| `HYPERLOOM_`<br>`SKIP_FRAME`<br>`WORK_CHECK`                  | Unset (check enabled)                                                  | Truthy skips the `optimize` preflight gate that requires the selected serving framework to be importable and a ROCm build. Last resort: when the server runs elsewhere, set `BENCHMARK_BASE_URL` instead, which exempts the check and configures the supported path. The gate already stays out of the way for `xdit`/`custom` (server-less), external multi-node, and any framework `install_baremetal.sh` cannot install (`atom`), where it warns instead of blocking. |
| `AITER_REF` | Unset | Optional bare-metal AITER install pin. When unset, the installer selects the newest tag compatible with the installed torch/triton stack. |
| `INFERENCE_`<br>`OPTIMIZER_`<br>`FRAMEWORK_`<br>`AUDIT_USE_LLM`      | `auto`                                                                 | Controls the FRAMEWORK phase semantic-audit LLM deep-read. `off` keeps the hermetic static verdict only; `on` always runs the evidence-gated LLM refine; `auto` (default) escalates to the LLM only when the static verdict is `unknown` or `confidence < 0.5`. The refine never upgrades to an `already_*` status the static layer did not already back with evidence. |

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
| `HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV` | Unset (`1`) | Specialist subprocesses inherit the limited provider credential set by default: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `ANTHROPIC_CUSTOM_HEADERS`, `CLAUDE_CODE_OAUTH_TOKEN`, `LLM_GATEWAY_KEY`, and AWS Bedrock credential/config vars. Set to `0` only when the `claude` CLI is authenticated through its own config and env credentials must be suppressed. Unrelated secrets such as GitHub and KB tokens remain blocked. |
| `HYPERLOOM_SPECIALIST_PERMISSION_MODE` | `bypassPermissions` | `--permission-mode` passed to the `claude` CLI for specialist subprocesses. Controls the Claude runtime approval-prompt behaviour only. Codex containment is resolved independently through `HYPERLOOM_CODEX_SANDBOX_MODE`. The default `bypassPermissions` is required for unattended operation; change only in setups where an external interactive approval flow is intended. |
| `HL_ALLOW_DANGEROUS_AGENT_PERMISSIONS` | Unset (`0`) | Slurm carrier only. Set to `1` only in dedicated internal containers to re-enable legacy Claude/Codex approval and sandbox bypass flags. |

---

## Critic / Robustness / knowledge base (KB)

The following variables configure the Critic, Robustness, and knowledge base components.

| Variable                              | Default                | Description                                                                                                                          |
|---------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `KNOWLEDGE_STORE_MODE`                | `local`                | Exclusive Recipe backend: `local` or `remote`. Ambient KB Store or GBrain credentials do not select remote mode. |
| `KNOWLEDGE_LOCAL_ROOT`                | `$USER_DATA_PATH/knowledge`, otherwise `~/.cache/hyperloom/knowledge` | Local Recipe/KG root. It is not used for Recipe data in remote mode. |
| `HYPERLOOM_`<br>`LOCAL_KB_ROOT`       | Unset                  | Deprecated explicit local Recipe root compatibility input, overridden by `--local-kb-root`; explicit use skips automatic legacy migration. |
| `INFERENCE_OPTIMIZER_`<br>`FA_KB_PATH` | `$USER_DATA_PATH/framework-kb`, otherwise `/workspace/hyperloom/framework-kb` | Framework-agent KB root, holding the lessons ledger the FRAMEWORK phase reads and writes. The only supported override: the `fa` reader and the orchestrator's writeback both resolve through it, so it moves both halves at once. The withdrawn `FRAMEWORK_AGENT_KB_DIR` is ignored with a warning naming the resolved root. On first start-up an existing partition under the legacy `$USER_DATA_PATH/kb` is copied across once; a copy that fails warns and leaves the phase to cold-start. |
| `KB_STORE_URL`                        | Local Recipe mode: `https://global.primus-safe.amd.com/knowledge-base`; remote mode: unset | Unified KB Service endpoint. Local Recipe mode uses the default only for co-hosted PR Monitor access; IR-3 disables PR Monitor without disabling local Recipe storage when the endpoint is unreachable. Remote Recipe mode requires an explicit value, selects the current Recipe View, replays its Config, Patch, and Kernel columns, then writes one final session at CLOSE. PR Monitor is available at `${KB_STORE_URL}/pr-monitor/v1` (REST/IR-3/Framework/KernelForge) and `${KB_STORE_URL}/pr-monitor/mcp/` (specialist MCP). |
| `KB_STORE_TOKEN`                      | Unset                  | KB Store bearer token. Required when `KNOWLEDGE_STORE_MODE=remote`; transport failures during the final write are non-fatal. |
| `KB_DRAFT_DIR`                        | Runtime-generated      | Internal remote-mode handoff where each Recipe column (`config`, `patch`, `kernel`) stages its knowledge and files before CLOSE merges them. Hyperloom creates and exports it; operators must not set it. A Recipe cannot be built without it. |
| `KB_WARM_START_DIR`                   | Runtime-generated      | Internal remote-mode handoff pointing agents at the downloaded `recipe.json + files/` selected Recipe View. Hyperloom creates and exports it; operators must not set it. |
| `GBRAIN_BASE_URL`                     | Unset                  | Optional GBrain endpoint for Framework PR capabilities. It never enables or satisfies Recipe remote mode. |
| `GBRAIN_TOKEN`                        | Unset                  | Optional GBrain bearer token for Framework PR capabilities. It never enables or satisfies Recipe remote mode. |
| `CRITIC_AGENT_ROOT`                   | Derived from `REPO_ROOT` | Override location of the critic-agent runtime.                                                                                    |
| `CRITIC_AGENT_`<br>`MAX_COMPLETION_TOKENS` | `32000`           | Output-token cap for one critic review call. A reply cut off at the cap is retried once at twice this value and then fails the turn, so the cap is a ceiling rather than a budget: unused headroom is never billed, while a truncated reply bills the whole call and yields nothing. Lower it for a model whose own output limit is smaller. A non-positive or unparseable value logs a warning and falls back to the default. |
| `ROBUSTNESS_AGENT_ROOT`               | Derived from `REPO_ROOT` | Override location of the robustness-agent runtime.                                                                                |
| `ROBUSTNESS_LLM_RCA_DISABLED`         | Unset                  | Set to `1` to forcibly disable the LLM root cause analysis (RCA) engine even when credentials are present.                                                 |

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
- **Live push**: when set to `1/true/yes/on` and the three `LANGFUSE_*` credentials are present, every in-process LLM call is mirrored into Langfuse while the run is live. A session-end flush backfills out-of-process children (geak, forge, robustness, specialist) and KEEP/REVERT decision Scores.
- **Local ledger**: `reports/trace/*.jsonl` is always written regardless of this flag. If the SDK is unavailable, live push degrades to a no-op.
- **Correlation**: the Langfuse trace ID and `session_id` grouping are derived from `claw_session_id` (env `CLAW_SESSION_ID`), falling back to the internal session ID for standalone runs. Live push and the offline `backfill_langfuse` CLI collapse onto one trace per Primus-Claw session.
- **Span layout**: `trace → phase span (PRELUDE/FRAMEWORK_AGENT/KERNEL_AGENT/SWEEP/…) → agent span (component: orchestration/kernel/specialist/critic/geak/forge/…) → Generation`. Each KEEP/REVERT/`gain_pct` Score attaches to the agent span that produced the decision, with a trace-level fallback when no matching span exists.
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
  recipe-KB audit, specialist intel, forge steps, GEMM tuning, decision scores,
  span close, final SDK flush). Each step runs at most once **per process**, so
  a duplicated CLOSE step won't double-push; a step that raised is retried by
  the next call. The receipt reports `flush_steps_done` (the steps that
  succeeded, for this process) and `counts_final`, which is `true` only once
  *every* step has completed — a `false` there means the push is still
  incomplete, not that the session was short-lived. Across processes (a crash
  plus a `--resume`, or two shutdown paths racing), the durable unit is finer
  than a step: `ext_rows_sent` records how far each `ext/*.jsonl` shard was
  drained so its rows are never re-pushed while later ones still are, and the
  one-shot `session_start` / `session_breakdown` pushes are claimed through an
  exclusive marker file (`reports/trace/.session_start.claim`) rather than
  through the receipt read. The audit backfills (recipe-KB, specialist intel,
  forge steps, GEMM tuning, decision scores) are *not* cursor-tracked: a second
  process that reaches CLOSE for the same session re-emits those spans. The receipt also carries `payload_sha256` over its own body; a
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
* `by_phase`: per-phase breakdown (PRELUDE / FRAMEWORK_AGENT / KERNEL_AGENT / SWEEP / CLOSE).
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

## Phase tuning

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `INFERENCE_OPTIMIZER_CYCLE_RELOOP_MIN_REMAINING_SEC` | Optional | `10800` | Absolute minimum remaining session seconds to justify opening a new macro-cycle. For bounded sessions the effective floor is `min(this, max_minutes * 60 * 0.15)` so shorter sessions are not unconditionally blocked. |

---

## Variables intentionally not exposed

These are read by `os.environ` somewhere in the codebase but are
internal-only — do not set them by hand:

* `HYPERLOOM_KERNEL_AGENT_ROOT`: internal CLI-only handoff to the
  kernel subprocess (Python constant `_KERNEL_AGENT_ROOT_ENV`).
* Any `_INFERENCE_OPTIMIZER_*_INTERNAL_*` symbol: internal toggles for
  the test suite.

If you find one of these in a log message, treat it as diagnostic
detail rather than something you should tune.

---

## More info

Use these resources for related configuration and reference information:

* [Hyperloom authentication and credentials](authentication.md): Credential precedence and direct upstream gateway wiring.
* [Troubleshooting Hyperloom](troubleshooting.md): Symptom → variable reverse-lookup for common failures.
