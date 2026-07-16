---
myst:
    html_meta:
        "description": "Complete reference for all environment variables read by the Hyperloom runtime, grouped by purpose: credentials, paths, workload parameters, backend selection, and observability."
        "keywords": "Hyperloom, environment variables, configuration, SAFE_API_KEY, USER_DATA_PATH, ROCm, AMD GPU, LLM inference, kernel optimization, LLM gateway, Langfuse, session"
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
| `SAFE_API_KEY`         | Conditional | —    | AMD primus-safe large language model (LLM) gateway key. Format `ak-...`. Required for the single-gateway setup; split-gateway deployments can instead provide provider-specific keys. Source for GEAK / Claude / Codex / Critic / Robustness credentials downstream (auto-aliased).                                                        |
| `OPENAI_BASE_URL`      | Conditional | —    | LLM gateway URL. Required for the single-gateway setup; split-gateway deployments can provide provider-specific base URLs instead. Production: `https://global.primus-safe.amd.com/api/v1/llm-proxy/v1`.                                                                                                                  |
| `ANTHROPIC_BASE_URL`   | No       | Derived from `OPENAI`<br>`_BASE_URL` | Claude-side base URL for split-gateway deployments.                                                                                                        |
| `ANTHROPIC_AUTH_TOKEN` | No       | Inherits `SAFE_API_KEY` | Claude CLI auth token alias; set explicitly only for split-gateway deployments.                                                                        |
| `GEAK_API_KEY`         | No       | Inherits `SAFE_API_KEY` | Only set explicitly to override the default inheritance.                                                                                                                              |
| `GEAK_BASE_URL`        | No       | Inherits `OPENAI`<br>`_BASE_URL` | Only set explicitly to override the default inheritance.                                                                                                                          |
| `GEAK_CLAUDE_MODEL`   | No       | Inherits `CLAUDE_MODEL`; DeepSeek-only defaults to `deepseek-chat` | GEAKv4 Claude Code workflow model id.                                                                                                                                                           |
| `ANTHROPIC_API_KEY`    | No       | Inherits `SAFE_API_KEY` (using preflight alias fan-out) | Only set explicitly to override.                                                                                                                                |
| `OPENAI_API_KEY`       | No       | Inherits `SAFE_API_KEY` (using preflight alias fan-out) | Only set explicitly to override.                                                                                                                                |
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
| `TRACELENS_ROOT`                          | No (installer auto-clones) | `${HYPER`<br>`LOOM_OP`<br>`EN_SOU`<br>`RCE_ROOT:-`<br>`/opt/hyperloom/`<br>`open-source`<br>`-repos}/Tr`<br>`aceLens` (auto-clone of `AMD-AGI/TraceLens` pinned to a fixed SHA) | `src/hyperloom/agents/kernel/scripts/install.sh` clones the public repo into the pod-local open-source checkout root when unset. Export it to opt into a pre-existing checkout you maintain — that is an explicit operator override and skips both the clone and the SHA pin. |
| `GEAK_CLAUDE_BIN`                          | No (installer auto-resolves) | First of `$HOME/.local/bin/claude`, `/usr/local/bin/claude`, `$(command -v claude)`; written to `kernel-agent.env.sh` | Pins the Claude Code binary the GEAK SDK path uses, so `claude_agent_sdk` doesn't fall back to its older bundled CLI. Export to force a specific build. |
| `USER_DATA_PATH`                          | No                   | `/workspace/hyperloom`                                             | Session directory root (logs, runs, mirrors, breakdown). Replaces the retired `INFERENCE_OPTIMIZER_SESSION_DIR` and `WORKSPACE_PATH`.                                                |
| `INFERENCE_`<br>`OPTIMI`<br>`ZER_CU`<br>`RRENT_S`<br>`ESSION_DIR` | No (set by CLI) | Set at session boot | Absolute path to the active session directory. Written by the CLI when a session starts and inherited by every benchmark subprocess; session-path resolution prefers it over scanning `USER_DATA_PATH`. Do not set by hand. |
| `HYPERLOOM_ROOT`                          | No                   | `$HYPER`<br>`LOOM_R`<br>`UNTIME_`<br>`DIR/sou`<br>`rce-mirrors`                            | Legacy source-mirror root kept for compatibility. Current open-source dependency checkouts default to the pod-local open-source root (`HYPER`<br>`LOOM_OP`<br>`EN_SOU`<br>`RCE_ROOT` / `$TMPDIR`), not this path. |
| `HYPERLOOM`<br>`_OPEN_`<br>`SOURCE`<br>`_ROOT`              | No                   | `/opt/hyperloom/`<br>`open-source`<br>`-repos`                      | Pod-local root for dependency checkouts. Decoupled from `USER_DATA_PATH` so shared session storage does not collocate concurrent pods' checkouts. |
| `MAGPIE_PATH`                              | No                   | Resolved from installed `Magpie` package unless explicitly set                               | Magpie package root for benchmark wrappers and patch inspection.                                                                                                                                            |
| `SESSION_DIR`                             | No (robustness-agent)| Scan known paths                                                   | Path containing `storage/coordinator.db`; the robustness FindingSink writes under `{session_`<br>`dir}/ag`<br>`ents/ro`<br>`bustne`<br>`ss/fin`<br>`dings/`<br>`{sess`<br>`ion_id}.jsonl`.                                       |
| `ROBUSTNESS_SERVER_URL`                   | No (robustness-agent)| Scan known DNS                                                     | M1 primary data source; empty disables the primary path and forces local-only probes.                                                                                                |
| `WORKSPACE_PATH` *(legacy)*               | No                   | Unset                                                              | Legacy path variable. Still consumed in two narrow spots: the CLI `setdefault`s it to the repo root for the critic subprocess's static assets, and TraceLens uses it as a `USER_DATA_PATH` fallback. Prefer `USER_DATA_PATH`. See [Upgrade Hyperloom version](upgrade.md).                            |
| `INFERENCE_`<br>`OPTIMI`<br>`ZER_SES`<br>`SION_DIR` *(deprecated)* | No            | Unset                                                              | **Retired** — replaced by `USER_DATA_PATH`. No longer read.                                                                                                                       |

---

## Workload configuration

Use CLI flags for workload shape and runtime behavior:

`--model`, `--framework`, `--gpu-type`, `--model-class`, `--tp`, `--ep`,
`--conc`, `--isl`, `--osl`, `--max-model-len`, `--precision`,
`--profile-osl`, `--enable-roofline` / `--no-enable-roofline`, and
`--enable-conc-sweep` / `--no-enable-conc-sweep`.

The CLI might still materialize internal envs for benchmark subprocesses, but
those are not stable user configuration and should not be pre-set by launchers.

---

## Kernel-opt backend selection

The following variables control the kernel optimization backend ladder.

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `KERNEL_OPT_BACKEND_ORDER`     | Unset                         | Comma-separated override for the kernel-opt backend. Bare-metal defaults to `geak` (whole-pipeline GEAK). Use `forge` to opt into the forge backend.                    |
| `KERNEL_OPT_MAX_PARALLEL`      | `8` (GPU-adaptive cap)        | Max parallel kernel-opt attempts per request (per-kernel race fan-out). The runtime caps this by visible GPUs and per-attempt GPU reservation when it can detect them.                                                                                                                            |
| `INFERENCE_OPTIMIZER`<br>`_KERNEL_OPT_MAX_PARTIAL` | Unset           | Cap on how many `PARTIAL` kernel-opt verdicts an action can yield before it short-circuits to `NEEDS_REVIEW`. Useful for keeping budget contained when GEAK is consistently timing out.            |

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

## Multi-node / prefill-decode (PD)

Use CLI flags for multi-node and prefill-decode configuration:

`--nodes`, `--mn-backend`, `--rayjob-image`, `--rayjob-gpus-per-node`,
`--pd-mode`, `--pd-prefill-nodes`, `--pd-decode-nodes`, `--pd-prefill-tp`,
`--pd-decode-tp`, `--pd-transfer-backend`, and `--pd-ib-device`.

The optimizer writes the resolved values into internal handoff envs when it
creates RayJob / Dynamo workloads; callers should not depend on those env names
as a public configuration API.

---

## Quantization prelude

| Variable | Default | Description |
|----------|---------|-------------|
| `HYPERLOOM_QUANTIZE_ENABLED` | Unset | Primary switch (`1` to enable) for the AMD Quark PTQ quantization prelude driven by `--quantize` / `--quantize-scheme`. |
| `QUARK_ROOT` | `/primus/hyperloom/Quark` | AMD Quark checkout used by the quantization-agent. |

---

## Framework / source-tree discovery

The following variables configure framework source discovery and path overrides.

| Variable                                          | Default                                                                | Description                                                                                                                                            |
|---------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `INFERENCE_`<br>`OPTIMIZER_`<br>`FRAMEWORK_`<br>`SOURCE_ROOTS`      | Union with `/sgl-workspace`<br>`/{aiter,sglang`<br>`,vllm}`                        | Colon-separated list of source roots used by PolicyGate and flag discovery. Populated automatically by `src/hyperloom/inference_optimizer/assets/install.sh`'s `_probe_framework_source_roots` step (using `hyperloom.orchestrator.framework.paths.probe_framework_source_roots_for_env`).   |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_RESCUE_PATHS`                | Unset                                                                  | Colon-separated list of extra directories the harvest step scans for stray `result.json` files written outside the session dir (InferenceX-native scripts that hardcode `--result-dir`). |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_AITER_JIT_DIR`               | Aiter default                                                          | Override the aiter just-in-time (JIT) cache root for cold-cap sizing.                                                                                                  |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_STRICT_PATHS`                | `1` when CLI bootstraps                                                | When `1`, missing path env raises instead of falling back to discovery. Set by the CLI at session start; do not override unless debugging.              |
| `HYPERLOOM_`<br>`SGLANG_PA`<br>`TCH_EXACT`<br>`_VERSIONS`           | Unset                                                                  | Pin the sglang server-patch step to specific upstream versions; advanced compatibility option.                                                          |
| `HYPERLOOM_`<br>`ENABLE`<br>`_PATCH`                          | `1`                                                                    | Set to `0` to skip the in-place server patch step (useful when the upstream is already pre-patched).                                                    |

---

## Critic / Robustness / knowledge base (KB)

The following variables configure the Critic, Robustness, and knowledge base components.

| Variable                              | Default                | Description                                                                                                                          |
|---------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_`<br>`LOCAL_KB_ROOT`             | `$USER_DATA_PATH/kb`   | Filesystem root for the local recipe-snapshot KB store (always the write target). Overridden by `--local-kb-root`. See [Integrate Recipe/Cortex knowledge base in Hyperloom](integrate-kb.md).             |
| `GBRAIN_BASE_URL`                     | Unset                  | Base URL of the gbrain recipe-snapshot page store — the **read** side of the recipe KB. When unset, recipe reads are local-only.       |
| `GBRAIN_TOKEN`                        | Unset                  | Bearer token for `GBRAIN_BASE_URL`.                                                                                                   |
| `RECIPE_KB_MIRROR_MODE`               | `external`             | `external` (default): an out-of-band CronJob ingests the local store into gbrain. `inline`: best-effort mirror each local write into gbrain in-process (local write stays authoritative). |
| `CORTEX_KB_URL`                       | Unset                  | Optional Cortex KB URL used **only** by the Critic agent's per-proposal assess enrichment (`/v2/reasoning/assess`) — *not* the recipe KB. Also set by `--cortex-kb-url`. No Cortex call is made unless configured. |
| `CRITIC_AGENT_ROOT`                   | Derived from `REPO_ROOT` | Override location of the critic-agent runtime.                                                                                    |
| `ROBUSTNESS_AGENT_ROOT`               | Derived from `REPO_ROOT` | Override location of the robustness-agent runtime.                                                                                |
| `ROBUSTNESS_LLM_RCA_DISABLED`         | Unset                  | Set to `1` to forcibly disable the LLM root cause analysis (RCA) engine even when credentials are present.                                                 |

---

## Session / observability hand-off

These are read by `src/hyperloom/inference_optimizer/session/manifest.py` and the `src/hyperloom/inference_optimizer/breakdown/collectors/`
package to populate `session_breakdown.json` for downstream consumers
(`claw-stats-service`).

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

* `session_total` — whole-session total across every call, with two
  convenience figures: `total_in_out` (prompt + completion only) and
  `grand_total` (in + out + all cache-creation + cache-read tokens).
* `by_component` — per-agent breakdown (orchestration / kernel / critic /
  specialist / proposal_scorer / geak / forge / …), each with the same
  convenience totals.
* `by_phase` — per-phase breakdown (PRELUDE / FRAMEWORK_AGENT / EXPLORE / KERNEL_AGENT / SWEEP / CLOSE).
* `attribution` — `attributed_to_decisions` vs `unattributed` split plus
  `attributed_calls_pct`. Only calls that carry a `task_id` / `dyn_id` joining
  to a KEEP/REVERT or dynamic_action decision (for example, specialist subprocess
  turns) are attributed; orchestration / kernel / critic / proposal_scorer
  turns are LLM-internal and land in `unattributed` (this is expected, not a
  gap in the data).
* `timeline` — each `action_timeline` row annotated with the tokens that join
  to it on `task_id`. Rows whose action has no LLM spend show `tokens: null`
  (rather than a zero bucket) to make the sparsity explicit.

To get the single "total tokens for this run" number, read
`token_usage.session_total.grand_total` (all-in) or `.total_in_out`
(prompt+completion only).

---

## Variables intentionally not exposed

These are read by `os.environ` somewhere in the codebase but are
internal-only — do not set them by hand:

* `HYPERLOOM_KERNEL_AGENT_ROOT` — internal CLI-only handoff to the
  kernel subprocess (Python constant `_KERNEL_AGENT_ROOT_ENV`).
* Any `_INFERENCE_OPTIMIZER_*_INTERNAL_*` symbol — internal toggles for
  the test suite.

If you find one of these in a log message, treat it as diagnostic
detail rather than something you should tune.

---

## More info

Use these resources for related configuration and reference information:

* [Hyperloom authentication and credentials](authentication.md) — Credential precedence and direct upstream gateway wiring.
* [Troubleshooting Hyperloom](troubleshooting.md) — Symptom → variable reverse-lookup for common failures.
