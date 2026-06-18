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
| `OPENAI_BASE_URL`      | yes      | —       | LLM gateway URL. Production: `https://core42.primus-safe.amd.com/api/v1/llm-proxy/v1`.                                                                                                                  |
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
| `TRACELENS_ROOT`                          | no (installer auto-clones) | `${HYPERLOOM_OPEN_SOURCE_ROOT:-${TMPDIR:-/tmp}/hyperloom/open-source-repos}/TraceLens` (auto-clone of `AMD-AGI/TraceLens` pinned to a fixed SHA) | `kernel-agent/scripts/install.sh` clones the public repo into the pod-local open-source checkout root when unset. Export it to opt into a pre-existing checkout you maintain — that is an explicit operator override and skips both the clone and the SHA pin. |
| `USER_DATA_PATH`                          | no                   | `/workspace/hyperloom`                                             | Session directory root (logs, runs, mirrors, breakdown). Replaces the retired `INFERENCE_OPTIMIZER_SESSION_DIR` and `WORKSPACE_PATH`.                                                |
| `HYPERLOOM_ROOT`                          | no                   | `$HYPERLOOM_RUNTIME_DIR/source-mirrors`                            | Legacy source-mirror root kept for compatibility. Current open-source dependency checkouts default to the pod-local open-source root (`HYPERLOOM_OPEN_SOURCE_ROOT` / `$TMPDIR`), not this path. |
| `HYPERLOOM_OPEN_SOURCE_ROOT`              | no                   | `${TMPDIR:-/tmp}/hyperloom/open-source-repos`                      | Pod-local root for auto-cloned open-source dependencies such as Magpie, TraceLens, GEAK, and OOB. Decoupled from `USER_DATA_PATH` so shared session storage does not collocate concurrent pods' checkouts. |
| `MAGPIE_DIR`                              | no                   | `$HYPERLOOM_OPEN_SOURCE_ROOT/Magpie`                               | Magpie source root for benchmark wrappers.                                                                                                                                            |
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
| `FRAMEWORK`       | `sglang`    | `sglang`, `vllm`, or single-node-only `atom`. A session cannot mix.   |
| `GPU_TYPE`        | auto-detect | `mi300x` / `mi325x` / `mi355x`.                                      |
| `TARGET_GPU_TYPE` | mirrors `GPU_TYPE` | Set by the CLI; used by Magpie YAML rendering for script pinning. |
| `MODEL_CLASS`     | unset       | Optional launcher hint. When unset, Coordinator boot infers and persists it from model metadata or model-path family keywords; the old live `classify` action is removed. |
| `TP`              | `1`         | Tensor-parallel size.                                                |
| `CONC`            | `8`         | Benchmark concurrency.                                               |
| `ISL`             | `256`       | Input sequence length.                                               |
| `OSL`             | `256`       | Output sequence length.                                              |
| `MAX_MODEL_LEN`   | `8192`      | Server-side max sequence length.                                     |
| `PRECISION`       | `bf16`      | Model precision (`bf16`, `fp8`, `mxfp4`, ...).                       |
| `RANDOM_RANGE_RATIO` | unset    | Optional Magpie random-range jitter.                                 |
| `ROCR_VISIBLE_DEVICES` | inherited | Standard ROCm visible-device mask.                                  |
| `HIP_VISIBLE_DEVICES` | inherited | Standard HIP visible-device mask.                                   |
| `RUN_EVAL`        | `true`      | Runs the accuracy eval step inside the workload runner by default. Set to `false`/`0`/`no`/`off` to disable; disabling emits a warning. |

---

## 5. Kernel-opt backend selection

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `KERNEL_OPT_BACKEND_ORDER`     | unset                         | Comma-separated override for the kernel-opt backend ladder. Values: `forge`, `geak`, `claude`, `codex`, `cursor`. Honoured before the auto-derived default `forge,geak`; OOB backends (`claude`, `codex`, `cursor`) require an explicit override.                    |
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

## 7. Critic / Robustness / KB

| Variable                              | Default                | Description                                                                                                                          |
|---------------------------------------|------------------------|--------------------------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_LOCAL_KB_ROOT`             | `$USER_DATA_PATH/kb`   | Filesystem root for the local recipe-snapshot KB store. Overridden by `--local-kb-root`. See [KB_GUIDE.md](KB_GUIDE.md).             |
| `CORTEX_KB_URL`                       | unset                  | Optional remote Cortex KB service URL. Also set by `--cortex-kb-url`. No remote KB is contacted unless this is configured.            |
| `RECIPE_KB_REMOTE`                    | `auto`                 | Remote KB **read** mode (writes always stay local). `auto` (default, also when unset) composites gbrain + Cortex dual-read only when **both** are configured, else uses the single configured remote, else local-only. `both` always composites around whatever is configured (even one source). `gbrain` reads from gbrain only (paired with `RECIPE_KB_MIRROR_MODE`). Any other value (or unset `CORTEX_KB_URL`) resolves to a single Cortex remote when `CORTEX_KB_URL` is set, else local-only. See [KB_GUIDE.md](KB_GUIDE.md). |
| `RECIPE_KB_MIRROR_MODE`               | `external`             | gbrain mirror mode (only when `RECIPE_KB_REMOTE=gbrain`). `external` (default): an out-of-band CronJob ingests the local store into gbrain. `inline`: best-effort in-process mirror of each local write into gbrain (local write stays authoritative). |
| `HYPERLOOM_SPECIALIST_KB_MCP_URL`     | unset                  | Read-only KB-graph (`cortex_kb`) MCP endpoint advertised to specialist subprocesses; also settable via `--specialist-kb-mcp-url`. When unset, falls back to the gbrain MCP derived from `$GBRAIN_BASE_URL` (+ `/mcp`). Specialist KB writes always stay local. |
| `HYPERLOOM_SPECIALIST_KB_MCP_TOKEN`   | unset                  | Bearer token for `HYPERLOOM_SPECIALIST_KB_MCP_URL` (sent as `Authorization: Bearer …`).                                              |
| `GBRAIN_BASE_URL` / `GBRAIN_TOKEN`    | unset                  | gbrain knowledge-graph base URL + bearer token. Used both for the gbrain remote-read path (`RECIPE_KB_REMOTE=auto`/`both`/`gbrain`) and as the default specialist `cortex_kb` MCP endpoint (`$GBRAIN_BASE_URL/mcp`) when no explicit override is set. |
| `CRITIC_AGENT_ROOT`                   | derived from `REPO_ROOT` | Override location of the critic-agent runtime.                                                                                    |
| `ROBUSTNESS_AGENT_ROOT`               | derived from `REPO_ROOT` | Override location of the robustness-agent runtime.                                                                                |
| `ROBUSTNESS_LLM_RCA_DISABLED`         | unset                  | Set to `1` to forcibly disable the LLM RCA engine even when credentials are present.                                                 |
| `ROBUSTNESS_AGENT_ENABLE_HARD_ACTIONS`| unset                  | M4 milestone gate for scheduling-police hard actions (`prune_branch`, `force_dispatch`, ...). Default keeps them disabled.           |
| `LLM_MODEL`                           | `claude-opus-4-7`      | RCA model name for robustness-agent.                                                                                                 |
| `ROBUST_ANALYZER_URL`                 | scan known DNS         | Optional hybrid-provider endpoint used by robustness-agent local/server data-source discovery.                                      |

---

## 8. Session / observability hand-off

These are read by `manifest.py` and `breakdown/collectors.py` to
populate `session_breakdown.json` for downstream consumers
(`claw-stats-service`).

| Variable          | Description                                                                                |
|-------------------|--------------------------------------------------------------------------------------------|
| `CLAW_SESSION_ID` | Hosted SaFE / Claw session id, written to `session.claw_session_id` in `session_breakdown.json`. Set by the PrimusClaw sandbox; unset for local runs. |
| `SANDBOX_USER_ID` | Hosted SaFE / Claw user id, written to `session.sandbox_user_id`. Set by PrimusClaw; unset for local runs.                                            |
| `HYPERLOOM_LANGFUSE_ENABLE` | Master switch (default **off**) for live Langfuse trace push. NOTE: when this flag is on in the environment / `.env`, `scripts/install.sh` auto-installs the optional `langfuse` SDK on demand (and skips it entirely when off), so no separate `pip install '...[trace]'` is required. When `1/true/yes/on` *and* the three `LANGFUSE_*` credentials are set, every in-process LLM call is mirrored into Langfuse while the run is live, and a session-end flush backfills the out-of-process children (geak / oob / robustness / specialist) and KEEP/REVERT decision Scores. The local `reports/trace/*.jsonl` ledger is always written regardless. If the SDK is unavailable, live push degrades to a no-op. **Correlation:** the Langfuse trace id and `session_id` grouping are derived from `claw_session_id` (env `CLAW_SESSION_ID`), falling back to the internal session id for standalone runs, so live push and the offline `backfill_langfuse` CLI collapse onto one trace per PrimusClaw session. **Span layout:** `trace → phase span (PRELUDE/EXPLORE/KERNEL/SWEEP/…) → agent span (component: orchestration/kernel/specialist/critic/geak/oob/…) → Generation`; each KEEP/REVERT/`gain_pct` Score attaches to the agent span that produced the decision (trace-level fallback when no matching span exists). **Receipt:** every session records a `langfuse` section in `session_breakdown.json` (and `reports/trace/langfuse_receipt.json`) noting whether the push was enabled (or the `disabled_reason`), the redacted connection config (host + key-presence booleans, never the keys), the derived `trace_id`/`session_id`, and how many generations/scores/spans were actually sent — so an operator can confirm post-hoc whether a run reached Langfuse. |

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
(`reports/trace/llm_calls.jsonl`). It is purely derived from
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

## 9. Long-run / roofline convergence tuning

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_OPTIMIZER_SATURATION_WITHIN_PCT` | `95.0` | Roofline ceiling threshold used by `direction_saturation()`. A dominant direction is marked saturated when achieved throughput is at least this percentage of the relevant roofline ceiling. Values outside `(0, 100]` fall back to `95.0`. |
| `INFERENCE_OPTIMIZER_SATURATION_CONVERGENCE` | enabled | Set to `0`/`false`/`no`/`off` to disable the cyclic-phase stop condition that blocks another macro-cycle when every tracked roofline domain is saturated. |

### Orchestration-memory checkpoint guardrail (long-run durability)

These bound the persistent orchestration conversation so multi-day runs do not
overflow the model context window. The guardrail measures *real* token usage
(input + cache-read + cache-creation) against the orchestration model's context
window (`MODEL_CONTEXT_WINDOWS`; 200k default), scaled by the fractions below.

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_OPTIMIZER_CTX_SOFT_FRACTION` | `0.70` | Soft checkpoint threshold as a fraction of the orchestration context window. At/above it a compaction is attempted (a degenerate summary is skipped, preserving the live conversation + prior memory). Values outside `(0, 1]` fall back to the default. |
| `INFERENCE_OPTIMIZER_CTX_HARD_FRACTION` | `0.85` | Hard checkpoint threshold (fraction of the context window). At/above it the run compacts via the deterministic fallback even if the summary looks degenerate. Values outside `(0, 1]` fall back to the default. Keep `> SOFT`. |
| `INFERENCE_OPTIMIZER_ORCH_MEMORY_ROLLBACK` | unset | Recovery escape hatch: set to a positive integer `n` to re-seed the next orchestration push from the `n`-th-from-newest snapshot in the bounded `orchestration_memory_history` ring (capped at 10), instead of the live memory — used to recover when a recent compaction dropped a key thread. Out-of-range / non-integer values warn and fall back to live memory. |

### Specialist exploration (phase interleave & GPU pool)

| Variable | Default | Description |
|----------|---------|-------------|
| `INFERENCE_OPTIMIZER_PHASE_INTERLEAVE` | off | EXPLORE↔KERNEL phase interleave, **off by default** (matches `origin/main`). Set to `1`/`true`/`yes`/`on` to opt in: EXPLORE may then propose kernel-owned actions and KERNEL may propose the explore triple (`explore`/`specialist`/`integrate_patch`). Default-off keeps phase gating strict. |
| `INFERENCE_OPTIMIZER_GPU_SPECIALIST_CAPACITY` | detected GPU count | Max number of GPU-specialist leases (the `gpu_research_lane` pool size). When set it wins (parsed as a non-negative int; `0` disables `needs_gpu=true` dispatch). When unset it defaults to the visible GPU count probed at launch (`detect_gpu_count()`), which honours `ROCR_VISIBLE_DEVICES` → `HIP_VISIBLE_DEVICES` → `CUDA_VISIBLE_DEVICES`, else whole-machine via `rocm-smi`. |
| `INFERENCE_OPTIMIZER_GPU_SPECIALIST_DEVICES` | derived from mask | Explicit comma/semicolon-separated **absolute** GPU id pool for specialists (capped to capacity). When unset, the pool is derived from the visible-device mask (`ROCR_VISIBLE_DEVICES`, then `HIP`/`CUDA`) so specialists stay on the operator's pinned cards; with no mask set it falls back to `0..capacity-1`. The leased ids are written verbatim into each specialist subprocess's `ROCR_VISIBLE_DEVICES`. |

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
* [KB_GUIDE.md](KB_GUIDE.md) — local recipe KB and optional Cortex KB setup.
* [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — symptom → variable
  reverse-lookup for common failures.
