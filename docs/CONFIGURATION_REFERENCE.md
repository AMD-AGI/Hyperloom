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

---

## 2. Path environment

| Variable                                  | Required             | Default                                                            | Description                                                                                                                                                                          |
|-------------------------------------------|----------------------|--------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `REPO_ROOT`                               | yes (local mode)     | `$(pwd)` when invoked from the repo root                           | This Hyperloom checkout. Used to locate `.env`, skills, scripts.                                                                                                                     |
| `OOB_SRC`                                 | yes for OOB backends | —                                                                  | Path to the `OOB/` subdirectory inside the Primus-Claw clone.                                                                                                                        |
| `INFERENCEX_PATH`                         | yes for baseline / target analysis | —                                                    | Path to the SemiAnalysisAI/InferenceX repo.                                                                                                                                          |
| `TRACELENS_ROOT`                          | yes for profile / kernel detection | `/wekafs/hyperloom/TraceLens-internal` when present  | Path to a checkout of `release/hyperloom_integration_v0.3` on TraceLens-internal.                                                                                                    |
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

## 7. PMC roofline (optional)

| Variable                                       | Default | Description                                                                                                                       |
|------------------------------------------------|---------|-----------------------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_ENABLE_PMC_ROOFLINE`                | `0`     | Set to `1` to layer Magpie PMC roofline analysis on top of TraceLens.                                                              |
| `HYPERLOOM_PMC_ROOFLINE_FORCE`                 | `0`     | Force the roofline step even when the bottleneck classifier marks the workload as `memory-bound` (where roofline adds no signal). |
| `HYPERLOOM_PMC_ROOFLINE_PORT`                  | `30001` | Server port the roofline harness binds.                                                                                            |
| `HYPERLOOM_PMC_ROOFLINE_GPU_TYPE`              | inherits `GPU_TYPE` | Override the GPU type used for roofline limits.                                                                        |
| `HYPERLOOM_PMC_ROOFLINE_MODE`                  | `launch`| `launch` or `attach`.                                                                                                              |
| `HYPERLOOM_PMC_ROOFLINE_DURATION_MS`           | `15000` | PMC sampling window in milliseconds.                                                                                               |
| `HYPERLOOM_PMC_ROOFLINE_STARTUP_TIMEOUT_S`     | `600`   | How long to wait for the roofline server to come up before failing the step.                                                        |
| `HYPERLOOM_ALLOW_DIRECT_PMC_ROOFLINE`          | `0`     | Allow `pmc_roofline` to run outside Ray for local debugging only. Never set to `1` in production.                                  |

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
| `ROBUST_ANALYZER_URL`                 | scan known DNS         | Legacy provider URL, kept for the `--mode legacy` robustness loop.                                                                   |

---

## 9. Session / observability hand-off

These are read by `manifest.py` and `breakdown/collectors.py` to
populate `session_breakdown.json` for downstream consumers
(`claw-stats-service`).

| Variable          | Description                                                                                |
|-------------------|--------------------------------------------------------------------------------------------|
| `CLAW_SESSION_ID` | Hosted SaFE / Claw session id, written to `session.claw_session_id` in `session_breakdown.json`. Set by the PrimusClaw sandbox; unset for local runs. |
| `SANDBOX_USER_ID` | Hosted SaFE / Claw user id, written to `session.sandbox_user_id`. Set by PrimusClaw; unset for local runs.                                            |

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
