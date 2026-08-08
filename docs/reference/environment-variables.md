---
myst:
    html_meta:
        "description": "Complete reference for all environment variables read by the Hyperloom runtime, grouped by purpose: credentials, paths, workload parameters, backend selection, and observability."
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

| Variable               | Required | Default | Description                                                                                                                                                                                            |
|------------------------|----------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `OPENAI_BASE_URL`      | Conditional | —    | OpenAI-side endpoint. Required together with `OPENAI_API_KEY` to enable the OpenAI side (Codex). Example: `https://<your-gateway-host>/api/v1/llm-proxy/v1`.                                                                                                                  |
| `OPENAI_API_KEY`       | Conditional | —    | OpenAI-side key, format `ak-...`. Pairs with `OPENAI_BASE_URL`; may omit it only when the Anthropic side has no base URL either, in which case the official OpenAI endpoint is implied. Also the source for the internal LLM aliases. Never used for the Anthropic side.                                                        |
| `ANTHROPIC_BASE_URL`   | Conditional | —    | Anthropic-side endpoint. Required together with `ANTHROPIC_API_KEY` to enable Claude. Never derived from `OPENAI_BASE_URL`.                                                                                                        |
| `ANTHROPIC_API_KEY`    | Conditional | —    | Anthropic-side key. Pairs with `ANTHROPIC_BASE_URL`; may omit it only when the OpenAI side has no base URL either, in which case the official Anthropic endpoint is implied. Never derived from `OPENAI_API_KEY`; setting it alongside an `OPENAI_BASE_URL` but without its own base URL fails preflight.                                                                                                                                |
| `ANTHROPIC_AUTH_TOKEN` | No       | —    | Claude CLI auth token alias, accepted in place of `ANTHROPIC_API_KEY`. Preflight never fills it; the Ray / e2e / forge-fusion env builders default it from the Anthropic-side key when they hand credentials to a subprocess.                                                                        |
| `GEAK_API_KEY`         | No       | —    | Internal alias, never derived from either side. GEAK runs on the Anthropic side (`ANTHROPIC_*` + `GEAK_CLAUDE_MODEL`); set this only to point GEAK elsewhere.                                                                                                                              |
| `GEAK_BASE_URL`        | No       | —    | Internal alias, never derived from either side. Set it only to point GEAK at a different endpoint than the Anthropic side.                                                                                                                          |
| `GEAK_CLAUDE_MODEL`   | No       | Inherits `CLAUDE_MODEL` | GEAKv4 Claude Code workflow model id.                                                                                                                                                           |
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
| `USER_DATA_PATH`                          | No                   | `/workspace/hyperloom`                                             | Session directory root (logs, runs, mirrors, breakdown). Replaces the retired `INFERENCE_OPTIMIZER_SESSION_DIR` and `WORKSPACE_PATH`.                                                |
| `INFERENCE_`<br>`OPTIMI`<br>`ZER_CU`<br>`RRENT_S`<br>`ESSION_DIR` | No (set by CLI) | Set at session boot | Absolute path to the active session directory. Written by the CLI when a session starts and inherited by every benchmark subprocess; session-path resolution prefers it over scanning `USER_DATA_PATH`. Do not set by hand. |
| `HYPERLOOM_ROOT`                          | No                   | `$HYPER`<br>`LOOM_R`<br>`UNTIME_`<br>`DIR/sou`<br>`rce-mirrors`                            | Legacy source-mirror root kept for compatibility. Current open-source dependency checkouts default to the repo-local cache root (`${HYPER`<br>`LOOM_CA`<br>`CHE_DIR:-`<br>`$REPO_ROOT`<br>`/.cache}`), not this path. |
| `HYPERLOOM`<br>`_CACHE_`<br>`DIR`                          | No                   | `$REPO_ROOT`<br>`/.cache`                      | Writable, repo-local base for auto-cloned open-source deps (TraceLens, Magpie, etc.), cloned per revision as `<name>@<sha>`. Not under `$TMPDIR` so a reaper cannot wipe it mid-run. |
| `MAGPIE_PATH`                              | No                   | Resolved from installed `Magpie` package unless explicitly set                               | Magpie package root for benchmark wrappers and patch inspection.                                                                                                                                            |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_MODEL_PATH_ROOTS` | No | Built-in model roots such as `/models` and `/shared_nfs` | `os.pathsep`-separated allowlist for absolute model paths restored from `state.json` during `--resume`. HuggingFace-style repo IDs remain allowed. Set this when production models live outside the built-in roots. |
| `SESSION_DIR`                             | No (robustness-agent)| Scan known paths                                                   | Path containing `storage/coordinator.db`; the robustness FindingSink writes under `{session_`<br>`dir}/ag`<br>`ents/ro`<br>`bustne`<br>`ss/fin`<br>`dings/`<br>`{sess`<br>`ion_id}.jsonl`.                                       |
| `ROBUSTNESS_SERVER_URL`                   | No (robustness-agent)| Scan known DNS                                                     | M1 primary data source; empty disables the primary path and forces local-only probes.                                                                                                |
| `WORKSPACE_PATH` *(legacy)*               | No                   | Unset                                                              | Legacy path variable. Still consumed in two narrow spots: the CLI `setdefault`s it to the repo root for the critic subprocess's static assets, and TraceLens uses it as a `USER_DATA_PATH` fallback. Prefer `USER_DATA_PATH`. See [Upgrade Hyperloom version](upgrade.md).                            |
| `INFERENCE_`<br>`OPTIMI`<br>`ZER_SES`<br>`SION_DIR` *(deprecated)* | No            | Unset                                                              | **Retired** — replaced by `USER_DATA_PATH`. No longer read.                                                                                                                       |

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
  `--no-explore`, `--no-eval`.
- **Agent models:** `--claude-model`, `--codex-model`.
- **Session / resume:** `--resume`, `--resume-from`, `--force-resume`,
  `--reset-state`.
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
`0.05`, i.e. a candidate must stay within 5 percentage points of the recorded
baseline accuracy.

Note that the score is measured once per candidate, not averaged over repeats.
On a full GSM8K run (1319 problems) the 5-point tolerance sits several standard
errors away from the baseline, so single-run noise does not trip it. Capping the
eval with a small `MAGPIE_EVAL_LIMIT` shrinks that margin sharply and can make
the gate noise-sensitive — prefer the full task set whenever a gate decision
depends on the result.

---

## Kernel-opt backend selection

The following variables control the kernel optimization backend ladder.

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `KERNEL_OPT_BACKEND_ORDER`     | Unset                         | Comma-separated override for the kernel-opt backend. Bare-metal defaults to `geak` (whole-pipeline GEAK). Use `forge` to opt into the forge backend.                    |
| `KERNEL_OPT_MAX_PARALLEL`      | `8` (GPU-adaptive cap)        | Max parallel kernel-opt attempts per request (per-kernel race fan-out). The runtime caps this by visible GPUs and per-attempt GPU reservation when it can detect them.                                                                                                                            |
| `HYPERLOOM_GEMM_SHAPE_CAPTURE` | `1`                           | Enables automatic runtime GEMM-shape capture for eligible single-node dense vLLM Forge tuning when no explicit shape input is available. Block-FP8 first reuses shapes from the TraceLens-selected steady-state trace of a successful Roofline with exactly matching model, workload, server arguments, environment, and backend controls. Missing or stale evidence triggers the same standard Roofline/ProfileExecutor/TraceLens steady-state pipeline as a fallback. Set to `0` to preserve the no-capture path. |
| `HYPERLOOM_GEMM_SHAPE_CAPTURE_TIMEOUT_SEC` | `1800`          | Timeout in seconds for the dense vLLM TunableOp recording benchmark. Block-FP8 fallback uses the standard Roofline/ProfileExecutor timeout. Values below `60` are clamped to `60`. |
| `INFERENCE_OPTIMIZER`<br>`_KERNEL_OPT_MAX_PARTIAL` | Unset           | Cap on how many `PARTIAL` kernel-opt verdicts an action can yield before it short-circuits to `NEEDS_REVIEW`. Useful for keeping budget contained when GEAK is consistently timing out.            |

---

## Kernel source resolution

A kernel candidate must resolve to a real source file before any backend can
rewrite it. Resolution runs as a ladder: curated dictionary, then the
trace-derived launcher frame, then a name grep. All three are deterministic and
require no configuration. Agent analysis may add the model-backed tiers below.

Every run writes `kernel_source_resolution.json` next to the candidate report.
It answers one question per hot kernel — which file defines it, and which tier
decided that — in a versioned schema (`schema_version`, currently `1.0.0`), so
consumers and triage read a contract rather than candidate internals.

Two model-backed tiers may sit on top of the deterministic ladder when
`--analysis-route agent` is used. The `deterministic` route never invokes either
tier. Agent-route network calls require an explicit
`HYPERLOOM_LLM_SOURCE_PROVIDER`; a model name alone never implies a provider or
endpoint. The tiers differ in scope, authority and data exposure; the
constraints of one do not apply to the other.

Neither can fail a run: no model configured, a gateway error, a timeout or an
unparseable reply all leave the deterministic result standing.

### Fallback tier

**When it runs.** Only for a candidate whose `source_file` is still empty after
all three deterministic tiers, and whose GPU share is at least 5%.

**What it sends.** One chat completion per such candidate, containing the kernel
symbol and every shortlisted path. The shortlist comes from a relaxed grep over
the known framework roots. **File contents are not sent unless
`HYPERLOOM_LLM_SOURCE_PREVIEW` authorises it** (see [Source egress](#source-egress));
with it, each path is accompanied by its first 40 lines, capped at 2000
characters.

**What it costs.** One call per qualifying candidate, 60-second ceiling, no
retry. `HYPERLOOM_LLM_SOURCE_MODEL` overrides the selected provider's model
setting. Claude uses `CLAUDE_MODEL`, then the project-wide
`DEFAULT_CLAUDE_MODEL`; OpenAI-compatible routing uses `OPENAI_MODEL`, then
`CODEX_MODEL`. Model settings are never borrowed across providers.

**Authority: selection only.** The model may return one of the exact shortlist
strings and nothing else. An invented path is rejected, as is any answer below
0.7 confidence. This is deliberate — an LLM-produced sentinel written into
`source_file` is what broke this pipeline originally.

### Review tier

The fallback only fires on an empty `source_file`, so it cannot catch the
deterministic tiers' actual failure mode: not coming up empty, but coming up
*confidently wrong*. Measured across historical sessions, only 59% of
verifiable resolutions mention the kernel they claim to define, and
`aten::fill_` alone has been resolved to four unrelated business files — each a
real, existing, root-resident source file passing every mechanical check.

**When it runs.** On the whole resolution table, including entries already
filled in by the deterministic tiers. Entries below 1% GPU share are skipped.

**What it sends.** A single chat completion carrying up to 40 entries at once.
For each entry it includes the kernel symbol, GPU share, current path and
deciding tier. File contents follow the same rule as the fallback tier: nothing
is sent unless `HYPERLOOM_LLM_SOURCE_PREVIEW` authorises it. When it does, one
call can ship up to 40 file heads, considerably more than the fallback tier
sends per call — which is why the switch is global rather than per-tier.

**What it costs.** One call per run (not per candidate), 180-second ceiling, no
retry. Same provider and model resolution as the fallback tier. The response
must include every sent `kernel_id` exactly once. A missing, duplicate or extra
ID rejects the whole batch so a truncated response cannot masquerade as a
complete review.

<div class="callout warn">

**Authority: it may rewrite, and it has no confidence threshold.** Unlike the
fallback tier, this one is not restricted to a shortlist — it can replace any
entry's path with any path, or drop a resolved entry back to unresolved. There
is no 0.7 confidence gate. The mechanical limit is that a rewritten path must
exist on disk **and** its resolved target must sit under a known framework root.
Symlinks cannot escape that boundary. TraceLens-style
`path.py(247): function` answers are split into a bare, openable path plus line
and function metadata. An unverifiable path is rejected and the original
stands. Curated `op_to_source` verdicts — including `non_rewritable` and
`no_kernel` — are authoritative and cannot be replaced by model review.

</div>

Every revision records `previous_source_file` and `previous_method`, so a bad
review is auditable and reversible, and `review_notes` lists every applied and
rejected change. The batch is staged before it is committed, so an exception
while validating one revision leaves every entry untouched. Failures — no model
configured, gateway error, timeout, unparseable reply — leave the deterministic
table untouched and are recorded in `review_notes`.

Accepted revisions are folded back into `hot_kernels`, all metadata derived from
the old path is cleared, and patchability is recomputed. The resolution JSON is
the audit view of the same effective candidate state, not a detached suggestion.

### Source egress

Both tiers call an external model provider, so what leaves the host is a
deliberate boundary rather than a side effect of building a useful prompt.

**Provider routing is explicit.** Set `HYPERLOOM_LLM_SOURCE_PROVIDER` to
`claude_agent_sdk` or `openai_compatible`. Claude requests use the native Claude
Agent SDK with all repository, shell and web tools denied; OpenAI-compatible
requests use chat completions. `kernel_source_resolution.json` records the
provider, model, source-preview decision, outcome and endpoint hostname. It
never records keys, custom headers, URL userinfo, query parameters or the full
prompt.

**Repository source is not sent by default.** The file heads described above are
withheld unless `HYPERLOOM_LLM_SOURCE_PREVIEW` is set to `1`/`true`/`yes`/`on`.
Without it both tiers still see candidate paths, which carry most of the
selection signal; with it, a review call can ship up to 40 file heads.

**The serving command line is never forwarded verbatim.** The tiers need backend
flags — the same MoE operator dispatches differently under
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
| `HYPERLOOM_`<br>`LLM_SOURCE`<br>`_PROVIDER` | Unset (no network call) | Required provider for source fallback/review: `claude_agent_sdk` (native Claude SDK, tools denied) or `openai_compatible` (chat completions). Common provider aliases are normalized to the canonical audit value. |
| `HYPERLOOM_`<br>`LLM_SOURCE`<br>`_MODEL` | Unset | Optional source-resolution model override. Otherwise resolves only from the selected provider's own model variables; no cross-provider fallback. |
| `HYPERLOOM_`<br>`LLM_SOURCE`<br>`_PREVIEW` | Unset (off) | Authorise sending the first 40 lines of candidate source files to the model provider. Applies to both the fallback and review tiers. Leave unset unless the provider is an approved destination for repository content. |

### How fallback failures surface

The fallback tier is advisory and never fails a run. Every
outcome is recorded on the candidate as `source_resolution_reason`, so a skip
can be told apart from a genuine failure:

| `source_resolution_reason` | Meaning |
|----------------------------|---------|
| *(absent)* | Resolved before fallback, so the tier was not reached |
| `llm_fallback_skipped: deterministic route` | Deterministic analysis explicitly prohibited model tiers |
| `llm_fallback_skipped: gpu_pct ...` | Candidate below the 5% GPU-share floor; no call made |
| `llm_fallback_skipped: no provider configured` | No `HYPERLOOM_LLM_SOURCE_PROVIDER`; settled before the shortlist grep, so an unconfigured tier costs nothing |
| `llm_fallback_no_shortlist` | Grep found nothing to choose from; no call made |
| `llm_fallback_declined: ...` | Model answered but the pick was rejected (invented path, low confidence, or refusal) |
| `llm_fallback_error: ...` | Call failed — import error, gateway rejection, or timeout |

Accepted answers are stamped `source_resolution_method="llm_fallback"` alongside
a `source_resolution_confidence`, so they can be audited separately from
deterministic resolutions. Failures in the trace-launcher tier are recorded the
same way under `trace_resolver_error: ...`, and both are logged at `WARNING`.

---

## Single-node Ray execution

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_OPTIMIZER_RAY_EXEC` | Unset (`on` for single-node) | Controls whether single-node serving benchmarks and `needs_gpu` specialists run through Ray actors. When unset, single-node runs are routed through Ray-managed leases while multi-node stays on the multi-node backend. Set to `0` / `false` / `no` / `off` to force the local subprocess path, or `1` / `true` / `yes` / `on` to force Ray. |

---

## Codex (OpenAI) backend web search

The following variables enable OpenAI's built-in server-side web search for the
Codex (GPT-style) backend. When enabled, every Codex turn is issued through the
OpenAI **Responses API** with the built-in `web_search` tool instead of
`chat.completions`; the search resolves server-side in one call and the model's
reply still carries the intent envelope. Only affects deployments whose Codex
endpoint is OpenAI-compatible and supports the Responses API `web_search` tool.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_CODEX_WEB_SEARCH` | Unset (off) | Set to `1`/`true` to route every Codex turn through the OpenAI Responses API with the built-in `web_search` tool. Default keeps the existing `chat.completions` path unchanged. |
| `HYPERLOOM_`<br>`CODEX_WEB_SEARCH`<br>`_CONTEXT_SIZE` | `medium` | Passed through as the `web_search` tool's `search_context_size` (`low` / `medium` / `high`). Ignored unless `HYPERLOOM_CODEX_WEB_SEARCH` is on. |

---

## Codex (OpenAI) agent sandbox

Selects how a Codex agent session (TraceLens analysis and every future
Codex-based agent) is contained. Codex implements its `read-only` and
`workspace-write` presets with bubblewrap; a host without `bwrap` fails every
shell command the agent issues with `bwrap: Failed to make / slave: Permission
denied`, so the default is `bypass` and containment rests on the container
Hyperloom already runs inside. Hosts that provide `bwrap` can hand containment
back to Codex. A session that declares no writable output directory is
read-only under every mode, and an unrecognized value fails the session
immediately rather than picking a preset for you.

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_`<br>`CODEX_SANDBOX_MODE` | `bypass` | `bypass` runs a writing session with Codex's filesystem restrictions lifted (required where `bwrap` is unavailable); `workspace-write` restricts writes to the session directory plus its declared output roots; `read-only` forbids writes entirely. |

---

## Single-node Ray GPU scheduling

These variables tune the single-node Ray execution path (active when
`INFERENCE_OPTIMIZER_RAY_EXEC=1` and `--nodes=1`). They have no effect on
multi-node runs or when the Ray backend is disabled.

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_OPTIMIZER_RAY_GPU_PENDING_LIMIT` | `4` | Maximum number of GPU specialists that may be simultaneously in-flight (pending Ray scheduling + running) on the single-node Ray path. Ray still serialises execution on the physical GPU(s) via `num_gpus`; this limit caps how many actors can queue behind the current one. Floored at `1`. **Reduce to `1` or `2` when GPU memory or per-process overhead is a concern** (each queued actor holds a Ray worker slot even while it waits). |
| `INFERENCE_OPTIMIZER_RAY_SERVING_PRIORITY` | On | When enabled (default), the dispatcher defers admitting new GPU research specialists while a serving benchmark holds the whole-machine `serving_slot`, preventing research work from starving serving. The slot is probed immediately before each specialist is admitted so a serving start that races the dispatch pass is caught. Set to `0`, `false`, `no`, or `off` to disable. |

---

## Multi-node / prefill-decode (PD)

Use CLI flags for multi-node topology and prefill-decode configuration:

`--nodes`, `--mn-backend`, `--gpus-per-node`, `--tp`, `--ep`,
`--pd-mode`, `--pd-prefill-nodes`, `--pd-decode-nodes`, `--pd-prefill-tp`,
`--pd-decode-tp`, `--pd-transfer-backend`, and `--pd-ib-device`.

`optimize` never creates or releases a multi-node cluster. The provisioning
platform (e.g. Primus-Claw) creates the RayJob or InferaDeployment and hands it
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
| `PYTORCH_ROCM_ARCH` | Detected | Explicit GPU target architecture (e.g. `gfx942`, `gfx950`) injected into each compile.  Set automatically from the session `--gpu-type`; operator-override applies to bare-metal installs outside the session. |
| `MAX_JOBS` | `8` | Parallelism cap for cmake/hipcc compile steps inside a targeted build.  Reduce on memory-constrained nodes (`MAX_JOBS=4` for a 64 GB compile node).  The default `8` is conservative enough for MI300X/MI355X nodes with 512 GB+. |
| `HYPERLOOM_`<br>`FRAMEWORK_PYTHON` | Unset | Explicit interpreter that launches the server for a from-source build (the venv Python the artifact was compiled against).  Set automatically from `FrameworkRuntime.runtime_python_exe` via `apply_runtime_override` into the per-variant YAML `benchmark.envs`.  The bypass backend honors it by launching `python -m`; the Magpie backend re-exports it from the YAML `benchmark.envs` to the server env.  Operators normally do not set this by hand. |
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
| `HYPERLOOM_SPECIALIST_INHERIT_SECRET_ENV` | Unset (`1`) | Bash-enabled specialist subprocesses inherit the limited provider credential set by default: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `OPENAI_API_KEY`, `ANTHROPIC_CUSTOM_HEADERS`, and AWS Bedrock credential/config vars. Set to `0` only when the `claude` CLI is authenticated through its own config and env credentials must be suppressed. Unrelated secrets such as GitHub and KB tokens remain blocked. |
| `HL_ALLOW_DANGEROUS_AGENT_PERMISSIONS` | Unset (`0`) | Slurm carrier only. Set to `1` only in dedicated internal containers to re-enable legacy Claude/Codex approval and sandbox bypass flags. |

---

## Critic / Robustness / knowledge base (KB)

The following variables configure the Critic, Robustness, and knowledge base components.

| Variable                              | Default                | Description                                                                                                                          |
|---------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `KNOWLEDGE_STORE_MODE`                | `local`                | Exclusive Recipe/KG backend: `local` or `remote`. Ambient GBrain credentials do not select remote mode. |
| `KNOWLEDGE_LOCAL_ROOT`                | `$USER_DATA_PATH/knowledge`, otherwise `~/.cache/hyperloom/knowledge` | Shared knowledge root. Remote mode uses only `.remote-locks/recipes` beneath it. |
| `HYPERLOOM_`<br>`LOCAL_KB_ROOT`       | Unset                  | Deprecated explicit local Recipe root compatibility input, overridden by `--local-kb-root`; explicit use skips automatic legacy migration. |
| `INFERENCE_OPTIMIZER_`<br>`FA_KB_PATH` | `$USER_DATA_PATH/framework-kb`, otherwise `/workspace/hyperloom/framework-kb` | Framework-agent KB root, holding the lessons ledger the FRAMEWORK phase reads and writes. The only supported override: the `fa` reader and the orchestrator's writeback both resolve through it, so it moves both halves at once. The withdrawn `FRAMEWORK_AGENT_KB_DIR` is ignored with a warning naming the resolved root. On first start-up an existing partition under the legacy `$USER_DATA_PATH/kb` is copied across once; a copy that fails warns and leaves the phase to cold-start. |
| `GBRAIN_BASE_URL`                     | Unset                  | GBrain endpoint; required with `KNOWLEDGE_STORE_MODE=remote` and ignored in local mode. |
| `GBRAIN_TOKEN`                        | Unset                  | GBrain bearer token; required with `KNOWLEDGE_STORE_MODE=remote` and ignored in local mode. |
| `RECIPE_KB_MIRROR_MODE`               | Obsolete               | Ignored. Remove it and select `KNOWLEDGE_STORE_MODE=local` or `remote`. |
| `CRITIC_AGENT_ROOT`                   | Derived from `REPO_ROOT` | Override location of the critic-agent runtime.                                                                                    |
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

**`HYPERLOOM_LANGFUSE_ENABLE`** details:

Primary switch (default **off**) for live Langfuse trace push.

- **SDK install**: when this flag is on, `src/hyperloom/inference_optimizer/assets/install.sh` auto-installs the optional `langfuse` SDK on demand and skips it entirely when off — no separate `pip install '...[trace]'` is required.
- **Live push**: when set to `1/true/yes/on` and the three `LANGFUSE_*` credentials are present, every in-process LLM call is mirrored into Langfuse while the run is live. A session-end flush backfills out-of-process children (geak, forge, robustness, specialist) and KEEP/REVERT decision Scores.
- **Local ledger**: `reports/trace/*.jsonl` is always written regardless of this flag. If the SDK is unavailable, live push degrades to a no-op.
- **Correlation**: the Langfuse trace ID and `session_id` grouping are derived from `claw_session_id` (env `CLAW_SESSION_ID`), falling back to the internal session ID for standalone runs. Live push and the offline `backfill_langfuse` CLI collapse onto one trace per Primus-Claw session.
- **Span layout**: `trace → phase span (PRELUDE/FRAMEWORK_AGENT/EXPLORE/KERNEL_AGENT/SWEEP/…) → agent span (component: orchestration/kernel/specialist/critic/geak/forge/…) → Generation`. Each KEEP/REVERT/`gain_pct` Score attaches to the agent span that produced the decision, with a trace-level fallback when no matching span exists.
- **Recipe-KB spans**: under the `recipe_kb` agent span, both directions of the cross-session recipe KB are recorded — `kb:recipe_snapshot:<method>` for reads (`get_recipe` / `search`) and `kb:recipe_write:<generator>` for writes. The generator suffix separates the session-opening `t0_anchor` identity stamp from the `coordinator` KEEP/REVERT/PR/CLOSE amends. Because a write rewrites the whole row, each write span carries `<field>_delta` metadata (`lessons_delta`, `pitfalls_delta`, …) reporting what that write actually contributed — a restamp that adds nothing shows no delta keys, so it is distinguishable from a real amend. The full audit row is attached as span output.
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
* **`flush_session` is idempotent**: A second flush only re-writes the receipt
  (no re-emit), so a duplicated CLOSE step won't double-push.
* **Package truncation**: The bundle caps at 5000 files / 256 MB. On a very
  long session the cap can stop the bundle short; the `PACKAGE_MANIFEST` then
  sets `truncated: true` and lists `dropped_files`, so consumers must not treat
  a truncated package as complete.
* **Generation duration is ~0**: Both live and backfill stamp a single
  timestamp (`end == start`), so Langfuse shows no meaningful per-Generation
  duration — counts/usage are accurate, latency is not captured.

### `token_usage` section (in `session_breakdown.json`)

Every breakdown carries a top-level `token_usage` section: a promoted,
discoverable rollup of LLM token spend derived from the per-call ledger
(`reports/trace/llm_calls.jsonl` + `ext/*.jsonl`). It is purely derived from
`decision_trace.token_rollup`, so it always reconciles with that section. No
env var controls it; it is always present (zeroed on pre-trace sessions).

* `session_total`: whole-session total across every call, with two
  convenience figures: `total_in_out` (prompt + completion only) and
  `grand_total` (in + out + all cache-creation + cache-read tokens).
* `by_component`: per-agent breakdown (orchestration / kernel / critic /
  specialist / proposal_scorer / geak / forge / …), each with the same
  convenience totals.
* `by_phase`: per-phase breakdown (PRELUDE / FRAMEWORK_AGENT / EXPLORE / KERNEL_AGENT / SWEEP / CLOSE).
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
`token_usage.session_total.grand_total` (all-in) or `.total_in_out`
(prompt+completion only).

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
