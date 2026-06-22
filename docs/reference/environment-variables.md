---
myst:
    html_meta:
        "description": "Complete reference for all environment variables read by the Hyperloom runtime, grouped by purpose: credentials, paths, workload parameters, backend selection, and observability."
        "keywords": "Hyperloom, environment variables, configuration, SAFE_API_KEY, USER_DATA_PATH, ROCm, AMD GPU, LLM inference, kernel optimization, auth-proxy, Langfuse, session"
---
# Environment variables

Every environment variable read by the Hyperloom runtime, grouped by
purpose. This page is the *exhaustive* reference; the root README,
`.env.template`, and each agent SKILL file are *convenience excerpts*.

Variables marked **required** must be set (using shell or `$REPO_ROOT/.env`)
or the CLI will exit fast at startup. Variables marked **optional** have
sensible defaults; the default is shown in the **Default** column.

Precedence rule (applies everywhere): shell-exported env wins over `.env`.
See [Hyperloom authentication and credentials](authentication.md) §1.

---

## Credentials

These variables configure LLM gateway access and optional backend credentials.

| Variable               | Required | Default | Description                                                                                                                                                                                            |
|------------------------|----------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SAFE_API_KEY`         | Yes      | —       | AMD primus-safe large language model (LLM) gateway key. Format `ak-...`. Source for GEAK / Claude / Codex / Critic / Robustness credentials downstream (auto-aliased).                                                        |
| `OPENAI_BASE_URL`      | Yes      | —       | LLM gateway URL. Production: `https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1`.                                                                                                                  |
| `CURSOR_API_KEY`       | No       | Unset   | Cursor software development kit (SDK) key (prefix `crsr_...`) for the out-of-box (OOB) `cursor` kernel-opt backend. Never inherited from `SAFE_API_KEY`. When unset, Hyperloom auto-drops `cursor` from the default kernel-opt ladder.       |
| `CURSOR_`<br>`DEFAULT`<br>`_MODEL` | No       | `claude-`<br>`opus-4-7` | Override the default Cursor model id.                                                                                                                                                          |
| `CLAUDE_MODEL`         | No       | `claude-`<br>`opus-4-7` | Claude model ID for OOB Claude attempts.                                                                                                                                                       |
| `CODEX_MODEL`          | No       | `gpt-5.4` | Codex model ID for OOB Codex attempts.                                                                                                                                                               |
| `GEAK_API_KEY`         | No       | Inherits `SAFE_API_KEY` | Only set explicitly to override the default inheritance.                                                                                                                              |
| `GEAK_BASE_URL`        | No       | Inherits `OPENAI`<br>`_BASE_URL` | Only set explicitly to override the default inheritance.                                                                                                                          |
| `GEAK_MODEL_NAME`      | No       | `claude-`<br>`opus-4-7` | GEAK preprocessor / solver model id.                                                                                                                                                           |
| `ANTHROPIC_API_KEY`    | No       | Inherits `SAFE_API_KEY` (using auth-proxy) | Only set explicitly to override.                                                                                                                                |
| `OPENAI_API_KEY`       | No       | Inherits `SAFE_API_KEY` (using auth-proxy) | Only set explicitly to override.                                                                                                                                |
| `LANGFUSE_HOST`        | No (required <br> only <br> when `HYPER`<br>`LOOM_LA`<br>`NGFUSE`<br>`_ENABLE=1`) | Unset | Base URL of your Langfuse deployment (e.g., `https://langfuse.<your-domain>`). Used by both the live trace push and the offline `backfill_langfuse` CLI. |
| `LANGFUSE`<br>`_PUBLIC_KEY`  | No (required <br> only <br> when `HYPER`<br>`LOOM_LA`<br>`NGFUSE`<br>`_ENABLE=1`) | Unset | Langfuse project public key (`pk-...`).                                                                                                                  |
| `LANGFUSE`<br>`_SECRET_KEY`  | No (required <br> only <br> when `HYPER`<br>`LOOM_LA`<br>`NGFUSE`<br>`_ENABLE=1`) | Unset | Langfuse project secret key (`sk-...`).                                                                                                                  |

---

## Path environment

The following variables configure filesystem paths for Hyperloom's runtime dependencies and session data.

| Variable                                  | Required             | Default                                                            | Description                                                                                                                                                                          |
|-------------------------------------------|----------------------|--------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `REPO_ROOT`                               | Yes (local mode)     | `$(pwd)` when invoked <br> from the repo root                           | This Hyperloom checkout. Used to locate `.env`, skills, scripts.                                                                                                                     |
| `OOB_SRC`                                 | Yes for OOB backends | —                                                                  | Path to the `OOB/` subdirectory inside the Primus-Claw clone.                                                                                                                        |
| `INFERENCEX_PATH`                         | Yes for baseline / target analysis | —                                                    | Path to the SemiAnalysisAI/InferenceX repo.                                                                                                                                          |
| `TRACELENS_ROOT`                          | No (installer auto-clones) | `${HYPER`<br>`LOOM_OP`<br>`EN_SOU`<br>`RCE_ROOT:-` <br> `${TMPDIR:-/tmp}`<br>`/hyperloom/` <br> `open-source`<br>`-repos}/Tr`<br>`aceLens` (auto-clone of `AMD-AGI/TraceLens` pinned to a fixed SHA) | `kernel-agent/scripts/install.sh` clones the public repo into the pod-local open-source checkout root when unset. Export it to opt into a pre-existing checkout you maintain — that is an explicit operator override and skips both the clone and the SHA pin. |
| `USER_DATA_PATH`                          | No                   | `/workspace/hyperloom`                                             | Session directory root (logs, runs, mirrors, breakdown). Replaces the retired `INFERENCE_OPTIMIZER_SESSION_DIR` and `WORKSPACE_PATH`.                                                |
| `HYPERLOOM_ROOT`                          | No                   | `$HYPER`<br>`LOOM_R`<br>`UNTIME_`<br>`DIR/sou`<br>`rce-mirrors`                            | Legacy source-mirror root kept for compatibility. Curnrent open-source dependency checkouts default to the pod-local open-source root (`HYPER`<br>`LOOM_OP`<br>`EN_SOU`<br>`RCE_ROOT` / `$TMPDIR`), not this path. |
| `HYPERLOOM`<br>`_OPEN_`<br>`SOURCE`<br>`_ROOT`              | No                   | `${TMPDIR:`<br>`-/tmp}/hy`<br>`perloom/`<br>`open-sou`<br>`rce-repos`                      | Pod-local root for auto-cloned open-source dependencies such as Magpie, TraceLens, GEAK, and OOB. Decoupled from `USER_DATA_PATH` so shared session storage does not collocate concurrent pods' checkouts. |
| `MAGPIE_DIR`                              | No                   | `$HYPER`<br>`LOOM_`<br>`OPEN_SO`<br>`URCE_`<br>`ROOT`<br>`/Magpie`                               | Magpie source root for benchmark wrappers.                                                                                                                                            |
| `SESSION_DIR`                             | No (robustness-agent)| Scan known paths                                                   | Path containing `storage/conductor.db`; the robustness FindingSink writes under `{session_`<br`>dir}/ag`<br>`ents/ro`<br>`bustne`<br>`ss/fin`<br>`dings/`<br>`{sess`<br>`ion_id}.jsonl`.                                       |
| `ROBUSTNESS_SERVER_URL`                   | No (robustness-agent)| Scan known DNS                                                     | M1 primary data source; empty disables the primary path and forces local-only probes.                                                                                                |
| `WORKSPACE_PATH` *(deprecated)*           | No                   | Unset                                                              | **Retired** during the all-artifacts-under-`USER_DATA_PATH` migration. Logged with a warning when set; do not rely on it. See [Upgrade Hyperloom version](../reference/upgrade.md).                            |
| `INFERENCE_`<br>`OPTIMI`<br>`ZER_SES`<br>`SION_DIR` *(deprecated)* | No            | Unset                                                              | **Retired** — replaced by `USER_DATA_PATH`. No longer read.                                                                                                                       |

---

## Auth-proxy

The following variables configure the local OOB auth-proxy.

| Variable           | Required | Default | Description                                                                                                  |
|--------------------|----------|---------|--------------------------------------------------------------------------------------------------------------|
| `AUTH_PROXY_PORT`  | No       | `4002`  | Bind port for the OOB auth-proxy on `127.0.0.1`. Change only if 4002 is occupied.                            |
| `OOB_API_KEY`      | No       | Inherits `SAFE_API_KEY` | Internal — only used inside the auth-proxy subprocess.                                              |

---

## Workload parameters (set by the CLI; can be pre-set for resume)

These are the canonical envs the Coordinator reads. The CLI sets them
from `--model` / `--framework` / `--isl` etc., but agents may also
read them when invoked standalone.

| Variable          | Default     | Description                                                          |
|-------------------|-------------|----------------------------------------------------------------------|
| `MODEL_PATH`      | —           | Path or Hugging Face (HF) id of the model to optimize.                              |
| `FRAMEWORK`       | `sglang`    | `sglang`, `vllm`, or single-node-only `atom`. A session cannot mix.   |
| `GPU_TYPE`        | Auto-detect | `mi300x` / `mi325x` / `mi355x`.                                      |
| `TARGET_GPU_TYPE` | Mirrors `GPU_TYPE` | Set by the CLI; used by Magpie YAML rendering for script pinning. |
| `MODEL_CLASS`     | Unset       | Optional launcher hint. When unset, Coordinator boot infers and persists it from model metadata or model-path family keywords; the old live `classify` action is removed. |
| `TP`              | `1`         | Tensor-parallel size.                                                |
| `CONC`            | `8`         | Benchmark concurrency.                                               |
| `ISL`             | `256`       | Input sequence length.                                               |
| `OSL`             | `256`       | Output sequence length.                                              |
| `MAX_MODEL_LEN`   | `8192`      | Server-side max sequence length.                                     |
| `PRECISION`       | `bf16`      | Model precision (`bf16`, `fp8`, `mxfp4`, ...).                       |
| `RANDOM_RANGE_RATIO` | Unset    | Optional Magpie random-range jitter.                                 |
| `ROCR_VISIBLE_DEVICES` | Inherited | Standard ROCm visible-device mask.                                  |
| `HIP_VISIBLE_DEVICES` | Inherited | Standard HIP visible-device mask.                                   |
| `RUN_EVAL`        | Unset       | When set to a non-empty value, runs the accuracy eval step inside the workload runner. |

---

## Kernel-opt backend selection

The following variables control the kernel optimization backend ladder.

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `KERNEL_OPT_BACKEND_ORDER`     | Unset                         | Comma-separated override for the kernel-opt backend ladder. Values: `forge`, `geak`, `claude`, `codex`, `cursor`. Honoured before the auto-derived default `forge,geak`; OOB backends (`claude`, `codex`, `cursor`) require an explicit override.                    |
| `KERNEL_OPT_MAX_PARALLEL`      | `2`                           | Max parallel kernel-opt attempts per request (per-kernel race fan-out).                                                                                                                            |
| `INFERENCE_OPTIMIZER`<br>`_KERNEL_OPT_MAX_PARTIAL` | Unset           | Cap on how many `PARTIAL` kernel-opt verdicts an action can yield before it short-circuits to `NEEDS_REVIEW`. Useful for keeping budget contained when GEAK is consistently timing out.            |

---

## Framework / source-tree discovery

The following variables configure framework source discovery and path overrides.

| Variable                                          | Default                                                                | Description                                                                                                                                            |
|---------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `INFERENCE_`<br>`OPTIMIZER_`<br>`FRAMEWORK_`<br>`SOURCE_ROOTS`      | Union with `/sgl-workspace`<br>`/{aiter,sglang`<br>`,vllm}`                        | Colon-separated list of source roots used by PolicyGate and flag discovery. Populated automatically by `kernel-agent/scripts/install.sh`'s probe step.   |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_SGLANG`<br>`_SERVER_ARGS`          | Derived from sglang source                                             | Path override for the file used to enumerate sglang server flags.                                                                                       |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_VLLM_ARG`<br>`_UTILS`              | Derived from vllm source                                               | Path override for the file used to enumerate vllm CLI flags.                                                                                            |
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
| `HYPERLOOM_`<br>`LOCAL_KB_ROOT`             | `$USER_DATA_PATH/kb`   | Filesystem root for the local recipe-snapshot KB store. Overridden by `--local-kb-root`. See [Integrate Recipe/Cortex knowledge base in Hyperloom](../reference/integrate-kb.md).             |
| `CORTEX_KB_URL`                       | Unset                  | Optional remote Cortex KB service URL. Also set by `--cortex-kb-url`. No remote KB is contacted unless this is configured.            |
| `RECIPE_KB_REMOTE`                    | Unset                  | Advanced remote-read mode selector. Writes remain local.                                                                              |
| `RECIPE_KB_MIRROR_MODE`               | Unset                  | Advanced mirroring mode for remote KB integrations.                                                                                   |
| `CRITIC_AGENT_ROOT`                   | Derived from `REPO_ROOT` | Override location of the critic-agent runtime.                                                                                    |
| `ROBUSTNESS_AGENT_ROOT`               | Derived from `REPO_ROOT` | Override location of the robustness-agent runtime.                                                                                |
| `ROBUSTNESS_LLM_RCA_DISABLED`         | Unset                  | Set to `1` to forcibly disable the LLM root cause analysis (RCA) engine even when credentials are present.                                                 |
| `ROBUSTNESS_`<br>`AGENT_ENABLE`<br>`_HARD_ACTIONS`| Unset                  | M4 milestone gate for scheduling-police hard actions (`prune_branch`, `force_dispatch`, ...). Default keeps them disabled.           |
| `LLM_MODEL`                           | `claude-opus-4-7`      | RCA model name for robustness-agent.                                                                                                 |
| `ROBUST_ANALYZER_URL`                 | Scan known DNS         | Optional hybrid-provider endpoint used by robustness-agent local/server data-source discovery.                                      |

---

## Session / observability hand-off

These are read by `manifest.py` and `breakdown/collectors.py` to
populate `session_breakdown.json` for downstream consumers
(`claw-stats-service`).

| Variable          | Description                                                                                |
|-------------------|--------------------------------------------------------------------------------------------|
| `CLAW_SESSION_ID` | Hosted SaFE / Claw session id, written to `session.claw_session_id` in `session_breakdown.json`. Set by the PrimusClaw sandbox; unset for local runs. |
| `SANDBOX_USER_ID` | Hosted SaFE / Claw user id, written to `session.sandbox_user_id`. Set by PrimusClaw; unset for local runs.                                            |
| `HYPERLOOM_LANGFUSE_ENABLE` | Master switch (default **off**) for live Langfuse trace push. See details below. |

**`HYPERLOOM_LANGFUSE_ENABLE`** details:

Master switch (default **off**) for live Langfuse trace push.

- **SDK install**: when this flag is on, `scripts/install.sh` auto-installs the optional `langfuse` SDK on demand and skips it entirely when off — no separate `pip install '...[trace]'` is required.
- **Live push**: when set to `1/true/yes/on` and the three `LANGFUSE_*` credentials are present, every in-process LLM call is mirrored into Langfuse while the run is live. A session-end flush backfills out-of-process children (geak, oob, robustness, specialist) and KEEP/REVERT decision Scores.
- **Local ledger**: `reports/trace/*.jsonl` is always written regardless of this flag. If the SDK is unavailable, live push degrades to a no-op.
- **Correlation**: the Langfuse trace ID and `session_id` grouping are derived from `claw_session_id` (env `CLAW_SESSION_ID`), falling back to the internal session ID for standalone runs. Live push and the offline `backfill_langfuse` CLI collapse onto one trace per PrimusClaw session.
- **Span layout**: `trace → phase span (PRELUDE/EXPLORE/KERNEL/SWEEP/…) → agent span (component: orchestration/kernel/specialist/critic/geak/oob/…) → Generation`. Each KEEP/REVERT/`gain_pct` Score attaches to the agent span that produced the decision, with a trace-level fallback when no matching span exists.
- **Receipt**: every session records a `langfuse` section in `session_breakdown.json` (and `reports/trace/langfuse_receipt.json`) noting:
  - Whether push was enabled (or the `disabled_reason`)
  - The redacted connection config (host and key-presence booleans — never the keys themselves)
  - The derived `trace_id` and `session_id`
  - How many generations, scores, and spans were sent

  This lets an operator confirm post-hoc whether a run reached Langfuse.

#### Langfuse and artifact-package — security and known limitations

* **Sensitive data surface**: When live push is on, `conversations.jsonl`
  (and Langfuse Generations) carry full prompt/response text. `redact_secrets`
  scrubs common token shapes (Bearer, `sk-`/`pk-`, GitHub tokens, some
  `KEY=value`) but is not a complete data loss prevention (DLP) filter — bare keys without a
  recognizable prefix (e.g. raw AWS `AKIA…`) can slip through. The artifact
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
  specialist / proposal_scorer / geak / oob / …), each with the same
  convenience totals.
* `by_phase` — per-phase breakdown (PRELUDE / FRAMEWORK_PR / EXPLORE / SWEEP / …).
* `attribution` — `attributed_to_decisions` vs `unattributed` split plus
  `attributed_calls_pct`. Only calls that carry a `task_id` / `dyn_id` joining
  to a KEEP/REVERT or dynamic_action decision (e.g. specialist subprocess
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

* `_KERNEL_AGENT_ROOT_ENV` — Internal CLI-only handoff to the kernel
  subprocess.
* `WORKSPACE_PATH` — Kept for legacy launcher warnings only; never
  read for behavior.
* `ANTHROPIC_BASE_URL` — Set by the auth-proxy at process launch.
* Any `_INFERENCE_OPTIMIZER_*_INTERNAL_*` Symbol — internal toggles for
  the test suite.

If you find one of these in a log message, treat it as diagnostic
detail rather than something you should tune.

---

## More info

Use these resources for related configuration and reference information:

* [Hyperloom authentication and credentials](authentication.md) — Credential precedence and the auth-proxy in detail.
* [Integrate Recipe/Cortex knowledge base in Hyperloom](../reference/integrate-kb.md) — Local recipe KB and optional Cortex KB setup.
* [Troubleshooting Hyperloom](../troubleshooting.md) — Symptom → variable reverse-lookup for common failures.
