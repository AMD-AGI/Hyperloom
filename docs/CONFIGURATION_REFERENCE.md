# Configuration Reference

Every environment variable read by the Hyperloom runtime, grouped by
purpose. This page is the *exhaustive* reference; the root README,
`.env.template`, and each agent SKILL file are *convenience excerpts*.

Variables marked **required** must be set (via shell or `$REPO_ROOT/.env`)
or the CLI will exit fast at startup. Variables marked **optional** have
sensible defaults; the default is shown in the "Default" column.

Precedence rule (applies everywhere): **shell-exported env wins over `.env`**.
See [ENV_AND_AUTH.md](ENV_AND_AUTH.md) §1.

---

## 1. Credentials

| Variable               | Required | Default | Description                                                                                                                                                                                            |
|------------------------|----------|---------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `SAFE_API_KEY`         | yes      | —       | AMD primus-safe LLM gateway key. Format `ak-...`. Source for GEAK / Claude / Codex / Critic / Robustness credentials downstream (auto-aliased).                                                        |
| `OPENAI_BASE_URL`      | yes      | —       | LLM gateway URL. Production: `https://core42.example-internal-host.invalid/api/v1/llm-proxy/v1`.                                                                                                                  |
| `CURSOR_API_KEY`       | no       | unset   | Cursor SDK key (prefix `crsr_...`) for the OOB `cursor` kernel-opt backend. **Never inherited from `SAFE_API_KEY`.** When unset, Hyperloom auto-drops `cursor` from the default kernel-opt ladder.       |
| `CURSOR_DEFAULT_MODEL` | no       | `claude-opus-4-7` | Override the default Cursor model id.                                                                                                                                                          |
| `CLAUDE_MODEL`         | no       | `claude-opus-4-7` | Claude model id for OOB Claude attempts.                                                                                                                                                       |
| `CODEX_MODEL`          | no       | `gpt-5.4` | Codex model id for OOB Codex attempts.                                                                                                                                                                |
| `GEAK_API_KEY`         | no       | inherits `SAFE_API_KEY` | Only set explicitly to override the default inheritance.                                                                                                                              |
| `GEAK_BASE_URL`        | no       | inherits `OPENAI_BASE_URL` | Only set explicitly to override the default inheritance.                                                                                                                          |
| `GEAK_MODEL_NAME`      | no       | `claude-opus-4-7` | GEAK preprocessor / solver model id.                                                                                                                                                           |
| `ANTHROPIC_API_KEY`    | no       | inherits `SAFE_API_KEY` (via auth-proxy) | Only set explicitly to override.                                                                                                                                |
| `OPENAI_API_KEY`       | no       | inherits `SAFE_API_KEY` (via auth-proxy) | Only set explicitly to override.                                                                                                                                |
| `LANGFUSE_HOST`        | no (required only when `HYPERLOOM_LANGFUSE_ENABLE=1`) | unset | Base URL of your Langfuse deployment (e.g. `https://langfuse.<your-domain>`). Used by both the live trace push and the offline `backfill_langfuse` CLI. |
| `LANGFUSE_PUBLIC_KEY`  | no (required only when `HYPERLOOM_LANGFUSE_ENABLE=1`) | unset | Langfuse project public key (`pk-...`).                                                                                                                  |
| `LANGFUSE_SECRET_KEY`  | no (required only when `HYPERLOOM_LANGFUSE_ENABLE=1`) | unset | Langfuse project secret key (`sk-...`).                                                                                                                  |

---

## 2. Path environment

| Variable                                  | Required             | Default                                                            | Description                                                                                                                                                                          |
|-------------------------------------------|----------------------|--------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `REPO_ROOT`                               | yes (local mode)     | `$(pwd)` when invoked from the repo root                           | This Hyperloom checkout. Used to locate `.env`, skills, scripts.                                                                                                                     |
| `OOB_SRC`                                 | yes for OOB backends | —                                                                  | Path to the `OOB/` subdirectory inside the Primus-Claw clone.                                                                                                                        |
| `INFERENCEX_PATH`                         | yes for baseline / target analysis | —                                                    | Path to the SemiAnalysisAI/InferenceX repo.                                                                                                                                          |
| `TRACELENS_ROOT`                          | no (installer auto-clones) | `$HYPERLOOM_RUNTIME_DIR/source-mirrors/TraceLens` (auto-clone of `AMD-AGI/TraceLens` pinned to a fixed SHA) | `kernel-agent/scripts/install.sh` clones the public repo here when unset. Export it to opt into a pre-existing checkout you maintain — that is an explicit operator override and skips both the clone and the SHA pin. |
| `USER_DATA_PATH`                          | no                   | `/workspace/hyperloom`                                             | Session directory root (logs, runs, mirrors, breakdown). Replaces the retired `INFERENCE_OPTIMIZER_SESSION_DIR` and `WORKSPACE_PATH`.                                                |
| `HYPERLOOM_ROOT`                          | no                   | derived from `REPO_ROOT`                                           | Writable Hyperloom asset root used by the installer for GEAK / OOB / TraceLens mirrors. Defaults to `$REPO_ROOT`; override if `REPO_ROOT` is on a read-only mount.                  |
| `MAGPIE_DIR`                              | no                   | discovered from sibling layouts                                    | Magpie source root for benchmark wrappers.                                                                                                                                            |
| `SESSION_DIR`                             | no (robustness-agent)| scan known paths                                                   | Path containing `storage/conductor.db`; the robustness FindingSink writes under `{session_dir}/agents/robustness/findings/{session_id}.jsonl`.                                       |
| `ROBUSTNESS_SERVER_URL`                   | no (robustness-agent)| scan known DNS                                                     | M1 primary data source; empty disables the primary path and forces local-only probes.                                                                                                |
| `WORKSPACE_PATH` *(deprecated)*           | no                   | unset                                                              | **Retired** during the all-artefacts-under-`USER_DATA_PATH` migration. Logged with a warning when set; do not rely on it. See [UPGRADING.md](UPGRADING.md).                            |
| `INFERENCE_OPTIMIZER_SESSION_DIR` *(deprecated)* | no            | unset                                                              | **Retired** — replaced by `USER_DATA_PATH`. **No longer read.**                                                                                                                       |

---

## 3. Auth-proxy

| Variable           | Required | Default | Description                                                                                                  |
|--------------------|----------|---------|--------------------------------------------------------------------------------------------------------------|
| `AUTH_PROXY_PORT`  | no       | `4002`  | Bind port for the OOB auth-proxy on `127.0.0.1`. Change only if 4002 is occupied.                            |
| `OOB_API_KEY`      | no       | inherits `SAFE_API_KEY` | Internal — only used inside the auth-proxy subprocess.                                              |

---

## 4. Workload parameters (set by the CLI; can be pre-set for resume)

These are the canonical envs the Coordinator reads. The CLI sets them
from `--model` / `--framework` / `--isl` etc., but agents may also
read them when invoked standalone.

| Variable          | Default     | Description                                                          |
|-------------------|-------------|----------------------------------------------------------------------|
| `MODEL_PATH`      | —           | Path or HF id of the model to optimize.                              |
| `FRAMEWORK`       | `sglang`    | `sglang` or `vllm`. A session cannot mix.                            |
| `GPU_TYPE`        | auto-detect | `mi300x` / `mi325x` / `mi355x`.                                      |
| `TARGET_GPU_TYPE` | mirrors `GPU_TYPE` | Set by the CLI; used by Magpie YAML rendering for script pinning. |
| `MODEL_CLASS`     | unset       | Required since v0.6 (`classify` action removed).                     |
| `TP`              | `1`         | Tensor-parallel size.                                                |
| `CONC`            | `8`         | Benchmark concurrency.                                               |
| `ISL`             | `256`       | Input sequence length.                                               |
| `OSL`             | `256`       | Output sequence length.                                              |
| `MAX_MODEL_LEN`   | `8192`      | Server-side max sequence length.                                     |
| `PRECISION`       | `bf16`      | Model precision (`bf16`, `fp8`, `mxfp4`, ...).                       |
| `RANDOM_RANGE_RATIO` | unset    | Optional Magpie random-range jitter.                                 |
| `ROCR_VISIBLE_DEVICES` | inherited | Standard ROCm visible-device mask.                                  |
| `HIP_VISIBLE_DEVICES` | inherited | Standard HIP visible-device mask.                                   |
| `RUN_EVAL`        | unset       | When set to a non-empty value, runs the accuracy eval step inside the workload runner. |

---

## 5. Kernel-opt backend selection

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `KERNEL_OPT_BACKEND_ORDER`     | unset                         | Comma-separated override for the kernel-opt backend ladder. Values: `geak`, `claude`, `codex`, `cursor`. Honoured before the auto-derived default `geak,claude,codex[,cursor]`.                    |
| `KERNEL_OPT_MAX_PARALLEL`      | `2`                           | Max parallel kernel-opt attempts per request (per-kernel race fan-out).                                                                                                                            |
| `INFERENCE_OPTIMIZER_KERNEL_OPT_MAX_PARTIAL` | unset           | Cap on how many `PARTIAL` kernel-opt verdicts an action can yield before it short-circuits to `NEEDS_REVIEW`. Useful for keeping budget contained when GEAK is consistently timing out.            |

---

## 6. Framework / source-tree discovery

| Variable                                          | Default                                                                | Description                                                                                                                                            |
|---------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS`      | union with `/sgl-workspace/{aiter,sglang,vllm}`                        | Colon-separated list of source roots used by PolicyGate and flag discovery. Populated automatically by `kernel-agent/scripts/install.sh`'s probe step.   |
| `INFERENCE_OPTIMIZER_SGLANG_SERVER_ARGS`          | derived from sglang source                                             | Path override for the file used to enumerate sglang server flags.                                                                                       |
| `INFERENCE_OPTIMIZER_VLLM_ARG_UTILS`              | derived from vllm source                                               | Path override for the file used to enumerate vllm CLI flags.                                                                                            |
| `INFERENCE_OPTIMIZER_RESCUE_PATHS`                | unset                                                                  | Colon-separated list of extra directories the harvest step scans for stray `result.json` files written outside the session dir (InferenceX-native scripts that hardcode `--result-dir`). |
| `INFERENCE_OPTIMIZER_AITER_JIT_DIR`               | aiter default                                                          | Override the aiter JIT cache root for cold-cap sizing.                                                                                                  |
| `INFERENCE_OPTIMIZER_STRICT_PATHS`                | `1` when CLI bootstraps                                                | When `1`, missing path env raises instead of falling back to discovery. Set by the CLI at session start; do not override unless debugging.              |
| `HYPERLOOM_SGLANG_PATCH_EXACT_VERSIONS`           | unset                                                                  | Pin the sglang server-patch step to specific upstream versions; advanced compatibility option.                                                          |
| `HYPERLOOM_ENABLE_PATCH`                          | `1`                                                                    | Set to `0` to skip the in-place server patch step (useful when the upstream is already pre-patched).                                                    |

---

## 8. Critic / Robustness / KB

| Variable                              | Default                | Description                                                                                                                          |
|---------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `INFERENCE_OPTIMIZER_KB_ROOT`         | unset (KB disabled)    | Directory of cross-run optimization KB JSONL files. See [KB_GUIDE.md](KB_GUIDE.md).                                                  |
| `CRITIC_KB_CLIENT_MODE`               | `inmemory`             | `inmemory` (test default), `live` (HTTP KB endpoint).                                                                                |
| `KB_BASE_URL`                         | unset                  | Required when `CRITIC_KB_CLIENT_MODE=live`.                                                                                          |
| `CRITIC_AGENT_ROOT`                   | derived from `REPO_ROOT` | Override location of the critic-agent runtime.                                                                                    |
| `ROBUSTNESS_AGENT_ROOT`               | derived from `REPO_ROOT` | Override location of the robustness-agent runtime.                                                                                |
| `ROBUSTNESS_LLM_RCA_DISABLED`         | unset                  | Set to `1` to forcibly disable the LLM RCA engine even when credentials are present.                                                 |
| `ROBUSTNESS_AGENT_ENABLE_HARD_ACTIONS`| unset                  | M4 milestone gate for scheduling-police hard actions (`prune_branch`, `force_dispatch`, ...). Default keeps them disabled.           |
| `LLM_MODEL`                           | `claude-opus-4-7`      | RCA model name for robustness-agent.                                                                                                 |
| `ROBUST_ANALYZER_URL`                 | scan known DNS         | Optional hybrid-provider endpoint used by robustness-agent local/server data-source discovery.                                      |

---

## 9. Session / observability hand-off

These are read by `manifest.py` and `breakdown/collectors.py` to
populate `session_breakdown.json` for downstream consumers
(`claw-stats-service`).

| Variable          | Description                                                                                |
|-------------------|--------------------------------------------------------------------------------------------|
| `CLAW_SESSION_ID` | Hosted SaFE / Claw session id, written to `session.claw_session_id` in `session_breakdown.json`. Set by the PrimusClaw sandbox; unset for local runs. |
| `SANDBOX_USER_ID` | Hosted SaFE / Claw user id, written to `session.sandbox_user_id`. Set by PrimusClaw; unset for local runs.                                            |
| `HYPERLOOM_LANGFUSE_ENABLE` | Master switch (default **off**) for live Langfuse trace push. NOTE: when this flag is on in the environment / `.env`, `scripts/install.sh` auto-installs the optional `langfuse` SDK on demand (and skips it entirely when off), so no separate `pip install '...[trace]'` is required. When `1/true/yes/on` *and* the three `LANGFUSE_*` credentials are set, every in-process LLM call is mirrored into Langfuse while the run is live, and a session-end flush backfills the out-of-process children (geak / oob / robustness / specialist) and KEEP/REVERT decision Scores. The local `reports/trace/*.jsonl` ledger is always written regardless. Requires the optional `langfuse` dependency (`pip install 'hyperloom-inference_optimizer[trace]'`); a missing SDK degrades to a no-op. **Correlation:** the Langfuse trace id and `session_id` grouping are derived from `claw_session_id` (env `CLAW_SESSION_ID`), falling back to the internal session id for standalone runs, so live push and the offline `backfill_langfuse` CLI collapse onto one trace per PrimusClaw session. **Span layout:** `trace → phase span (PRELUDE/EXPLORE/KERNEL/SWEEP/…) → agent span (component: orchestration/kernel/specialist/critic/geak/oob/…) → Generation`; each KEEP/REVERT/`gain_pct` Score attaches to the agent span that produced the decision (trace-level fallback when no matching span exists). **Receipt:** every session records a `langfuse` section in `session_breakdown.json` (and `reports/trace/langfuse_receipt.json`) noting whether the push was enabled (or the `disabled_reason`), the redacted connection config (host + key-presence booleans, never the keys), the derived `trace_id`/`session_id`, and how many generations/scores/spans were actually sent — so an operator can confirm post-hoc whether a run reached Langfuse. |

#### Langfuse / artifact-package — security & known limitations

* **Sensitive data surface.** When live push is on, `conversations.jsonl`
  (and Langfuse Generations) carry full prompt/response text. `redact_secrets`
  scrubs common token shapes (Bearer, `sk-`/`pk-`, GitHub tokens, some
  `KEY=value`) but is **not** a complete DLP filter — bare keys without a
  recognizable prefix (e.g. raw AWS `AKIA…`) can slip through. The artifact
  packager also copies `reports/trace/*.jsonl` and, with the loose mode on by
  default (`HYPERLOOM_SESSION_PACKAGE_LOOSE`), drops them under `/workspace`
  for the Claw sync. If a session may contain customer code / secrets, define
  an explicit retention + access-control policy for both the Langfuse project
  and the `/workspace` package destination, and consider disabling live push
  or loose packaging for those runs.
* **`live push` + `backfill_langfuse` overlap.** Both derive the same
  `trace_id` from `claw_session_id`, so running the offline backfill *after* a
  live run re-emits the out-of-process children onto the same trace and can
  duplicate observations. Use one path per session, or treat backfill as a
  recovery tool only when live push did not run.
* **`flush_session` is idempotent.** A second flush only re-writes the receipt
  (no re-emit), so a duplicated CLOSE step won't double-push.
* **Package truncation.** The bundle caps at 5000 files / 256 MB. On a very
  long session the cap can stop the bundle short; the `PACKAGE_MANIFEST` then
  sets `truncated: true` and lists `dropped_files`, so consumers must not treat
  a truncated package as complete.
* **Generation duration is ~0.** Both live and backfill stamp a single
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

## 10. Variables intentionally **not** exposed

These are read by `os.environ` somewhere in the codebase but are
internal-only — do not set them by hand:

* `_KERNEL_AGENT_ROOT_ENV` — internal CLI-only handoff to the kernel
  subprocess.
* `WORKSPACE_PATH` — kept for legacy launcher warnings only; never
  read for behaviour.
* `ANTHROPIC_BASE_URL` — set by the auth-proxy at process launch.
* Any `_INFERENCE_OPTIMIZER_*_INTERNAL_*` symbol — internal toggles for
  the test suite.

If you find one of these in a log message, treat it as diagnostic
detail rather than something you should tune.

---

## See also

* [ENV_AND_AUTH.md](ENV_AND_AUTH.md) — credential precedence and the
  auth-proxy in detail.
* [KB_GUIDE.md](KB_GUIDE.md) — KB stores referenced by
  `INFERENCE_OPTIMIZER_KB_ROOT`.
* [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — symptom → variable
  reverse-lookup for common failures.
