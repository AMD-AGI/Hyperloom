---
myst:
    html_meta:
        "description": "Diagnose and fix common Hyperloom failures: LLM gateway 401s, Ray GPU issues, VRAM exhaustion, TraceLens CLI errors, KB write failures, and session resume problems."
        "keywords": "Hyperloom, troubleshooting, LLM gateway, Ray, VRAM, OOM, GEAK, TraceLens, IntelliKit, KB, LLM inference, AMD GPU, session resume, debugging"
---
# Troubleshooting Hyperloom

A consolidated symptom → cause → fix index for the most common Hyperloom failures. If a symptom isn't listed here, check the
upstream SKILL file for the component you're touching:
[`inference_optimizer/SKILL.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/SKILL.md),
[`kernel-execution-path.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/reference/kernel-execution-path.md),
[`critic/SKILL.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/agents/critic/SKILL.md).

```{note}
Shell paths on this page follow the recommended `pip install --target .` layout.
In a source checkout, replace the `hyperloom/` prefix with `src/hyperloom/`.
The command blocks assume `REPO_ROOT` points at that workspace. No installer
exports it for you, so run `export REPO_ROOT="$(pwd -P)"` from the workspace
first — it does not survive a new shell.
```

---

## 401 Unauthorized

**Symptom**: A tool exits with one of:

* `HTTP 401 Unauthorized`
* `Claude SDK exit code 1`
* `OpenAI SDK: AuthenticationError`

**Cause**: The LLM gateway credentials are missing, expired, or not
propagated to `~/.claude/config.json`. Hyperloom talks directly to the
configured upstream gateway.

**Fix**:

1. Confirm `OPENAI_API_KEY` is set without printing the secret:
   ```bash
   test -n "${OPENAI_API_KEY:-}"
   ```
2. Re-run preflight (idempotent — rewrites `~/.claude/config.json`
   `customApiUrl` and `primaryApiKey` and re-derives all alias keys):
   ```bash
   bash "$REPO_ROOT/hyperloom/agents/kernel/scripts/install.sh" --check-only
   # If check-only reports issues, re-run without --check-only:
   bash "$REPO_ROOT/hyperloom/agents/kernel/scripts/install.sh"
   ```
3. Inspect `~/.claude/config.json` — `customApiUrl` must point at the
   upstream gateway (for example, `https://<your-gateway-host>/api/v1/llm-proxy/v1`).

See [Hyperloom authentication and credentials](authentication.md) for credential setup and gateway configuration.

---

## A rotated API key does not take effect

**Symptom**: A new key is written into `.env`, but the run keeps failing on
credentials. `tr '\0' '\n' < /proc/<pid>/environ | grep ANTHROPIC_API_KEY` on the
running optimizer shows the previous key.

**Cause**: `install.sh` snapshots the resolved credentials into
`$USER_DATA_PATH/runtime/kernel-agent.env.sh`, and every launch sources `.env`
first and that file second. The snapshot is a fallback, so the rotated value
wins — but only if it is in the environment when the file is sourced. Sourcing
the file in a shell that never loaded the new `.env` still yields the old key.

**Fix**:

1. Source `.env` before `kernel-agent.env.sh`, which is the documented launch
   order:
   ```bash
   set -a; . "$REPO_ROOT/.env"; set +a
   . "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
   ```
   The file prints `ANTHROPIC_API_KEY differs from the install-time snapshot` on
   a mismatch; that line means the rotated value is the one in effect.
2. To refresh the snapshot itself, re-run the installer:
   ```bash
   bash "$REPO_ROOT/hyperloom/inference_optimizer/assets/install.sh"
   ```

---

## TLS or certificate errors against the LLM gateway

**Symptom**: Preflight or an LLM client fails before authentication with one of:

* `certificate verify failed`
* `SSL: CERTIFICATE_VERIFY_FAILED`
* `gateway catalog unreachable after retries`, while `curl -k "$OPENAI_BASE_URL/models"` succeeds

**Cause**: The container or host does not trust the certificate authority used by
the configured LLM gateway. This is common on AMD-internal networks that use an
internal CA, or on self-hosted gateways with private certificates.

**Fix**:

1. For the AMD Primus-SaFE gateway, install the AMD certificate bundle inside the
   container or host:
   ```bash
   curl -fsSL https://raw.githubusercontent.com/AMD-AGI/Primus-SaFE/main/Scripts/setup-certs/setup.sh | bash
   ```
2. For a self-hosted gateway with a private CA, configure the standard Python /
   requests certificate variables before launching:
   ```bash
   export REQUESTS_CA_BUNDLE=/path/to/ca-bundle.pem
   export SSL_CERT_FILE=/path/to/ca-bundle.pem
   ```
3. Re-run preflight or the installer after updating certificates.

---

## Ray `--num-gpus` rejected

**Symptom**: `ray start --head ... --num-gpus=N` fails with
`Error: no such option: --num-gpus` or similar Click errors.

**Cause**: Click ≥ 8.3 is incompatible with the Ray 2.44 CLI shipped
by Hyperloom's installer.

**Fix**:

```bash
pip install --quiet 'click<8.3.0' 'ray[default]==2.44.1'
ray --version
```

The same fix applies when `ray --version` itself fails after a pod or
venv rebuild.

On an interpreter with no 2.44.1 wheel (cp314 postdates that release), install
the lowest published version above it instead and leave the `click` ceiling off
— it guards 2.44.1's CLI alone, and forcing it onto a newer Ray downgrades a
`click` that works:

```bash
pip install --quiet 'ray[default]>=2.44.1'
ray --version
```

---

## Ray tasks stuck pending forever

**Symptom**: `ray status` shows pending tasks; GPU usage is 0% even
though the node has free GPUs.

**Cause**: Ray was started with `--num-gpus=0` (or omitted, which
defaults to 0 on some images). GEAK submits tasks with
`num_gpus>=1` and will wait indefinitely.

**Fix**:

```bash
RAY_NUM_GPUS="${RAY_NUM_GPUS:-$(python3 -c 'import torch; print(torch.cuda.device_count() or 1)')}"
ray stop --force || true
# issue #433: raise the soft open-files limit before `ray start` so the
# raylet stays up (see "Ray raylet unstable / zombie" below).
ulimit -Sn "${RAY_MIN_NOFILE:-65536}" 2>/dev/null || true
ray start --head --disable-usage-stats --num-gpus="$RAY_NUM_GPUS" --include-dashboard=false
ray status
```

> **Note**: current Hyperloom startup paths can auto-start or reuse a local Ray
> head. If `ray_current_cluster` points at a stale or incompatible cluster, stop
> Ray first so Hyperloom can create a fresh head with the required GPU and
> custom-resource configuration.

---

## Ray raylet unstable or zombie (fd-limit too low)

**Symptom**: During GEAK dispatch the raylet aborts on startup
(`SIGABRT`), or `ray stop` reports it could not be stopped and leaves
it behind, for example:

```
WARN scripts.py:1287 -- Stopped only 0 out of 391 Ray processes within
the grace period 16 seconds. Remaining processes [... name='raylet',
status='zombie' ...]
```

**Cause**: At the container default `ulimit -n` (1024) the open-files
limit is far too low for the raylet, which opens many fds (sockets,
plasma store, per-worker pipes) — on a 384-CPU node with hundreds of
workers it needs `nofile >= 65536` (issue #433).

**Fix**: Launch the kernel-agent container with a high open-files limit
(this is a **hard requirement** — only the container launch can lift the
*hard* cap):

```bash
docker run --ulimit nofile=1048576 ...   # minimum: --ulimit nofile=65536
```

The runtime also runs an fd-limit preflight that raises this process's
*soft* limit (up to the hard cap) before every `ray start`
(`hyperloom/agents/kernel/scripts/install.sh` `ensure_fd_limit_for_ray` and
`hyperloom/agents/kernel/tools/backends/ray_runtime.py` `ensure_fd_limit`), so a
high hard cap is enough; you do not need to set the soft limit yourself.
Override the target with `RAY_MIN_NOFILE` if needed. If the preflight
warns that the **hard** cap is below the target, the container was not
launched with `--ulimit nofile=...` — fix the launch command.

---

## VRAM exhaustion or IR-1 error

**Symptom**: Inference server exits with `HSA: out of memory`,
`std::runtime_error: ROCm IR-1`, `OOM` during the prefill step, or
the baseline benchmark fails with VRAM allocation errors.

**Cause**: One of:

* `TP` is too small for the model's weights.
* `MAX_MODEL_LEN` is set higher than the KV cache budget allows at
  the current `CONC`.
* A previous server process leaked memory and didn't release it.

**Fix**:

1. Confirm no zombie inference server is holding VRAM:
   `rocm-smi --showmemuse` then kill stragglers with
   `pkill -f sglang.launch_server` (or `vllm`).
2. Bump `TP` (for example, 4 → 8) so weights and KV cache fit.
3. Lower `MAX_MODEL_LEN` to the smallest length your workload actually
   needs. When unset it is auto-derived as ISL + OSL + headroom, clamped
   to the model's native context; pin a smaller value with
   `--max-model-len`.
4. Lower `CONC` to reduce simultaneous KV cache pressure.

Inspect the failed action's server log and result artifacts under
`$SESSION_DIR/runs/<action>/<task_id>/` for the first allocation failure and
its workload context. There is no runtime recovery agent that diagnoses or
restarts the session for you.

---

## GEAK fails fast with "profiler_mcp not installed"

**Symptom**: GEAK attempts abort within 4 minutes with zero-byte
baseline files; logs mention a missing `profiler_mcp` or one of the
other GEAK MCP packages.

**Cause.** `install.sh` did not finish installing GEAK and its
dependencies. Common trigger: pip install failed on a transient registry
hiccup and the installer continued.

**Fix**:

```bash
bash "$REPO_ROOT/hyperloom/agents/kernel/scripts/install.sh" --check-only
# If --check-only reports missing packages, re-run without --check-only:
bash "$REPO_ROOT/hyperloom/agents/kernel/scripts/install.sh"
```

The installer is idempotent and re-installs only what's missing.

---

## Codex SDK turns stall or TraceLens roofline times out

**Symptom**: On code paths that use the shared Codex SDK session helper
(TraceLens roofline via `run_codex_turn`, orchestrator Codex turns), one or
more of:

* A Codex stage hits its phase budget with no declared output (for example,
  no `analysis.md`).
* Individual turns take minutes even when the upstream LLM responds in seconds.
* Codex internal logs report `pool timed out while waiting for an open
  connection` or `state db update_thread_metadata failed`.

**Cause**: Hyperloom picks exactly one parent directory for each SDK turn via
`_codex_home_parent` in `codex_session.py`:

1. `HYPERLOOM_RUNTIME_DIR` when set (must be outside a source checkout and
   creatable, or the turn raises `CodexSessionUnavailableError`).
2. Otherwise the first safe declared **writable root** for that turn.
3. Otherwise the run working directory.

Each turn then creates a mode-`0700` `.hyperloom-codex-home-*` directory under
that parent; Codex stores SQLite (WAL) state there. When the SDK client closes,
`_cleanup_codex_home` removes that directory — it is not left behind for a
post-stage listing.

Installers often **export** `HYPERLOOM_RUNTIME_DIR=$USER_DATA_PATH/runtime`, but
that is launch configuration, not a separate placement rule inside
`_codex_home_parent`. If that path (or an unset-runtime writable root) sits on
storage that does not support the file locking SQLite needs, concurrent writers
from the main agent and spawned sub-agents can block for minutes on the critical
path.

**Scope**: This entry covers SDK turns that go through `_codex_home_parent` only.
It does **not** cover Forge-fusion / GEAK Codex (KernelForge uses its own home
under `~/.cache/kernelforge/codex_home`) or subprocess **specialist** Codex
tasks (`CODEX_HOME=<task-workspace>/.codex`, which also ignores
`HYPERLOOM_RUNTIME_DIR`). Treat stalls on those paths as separate placement
issues.

**Fix**: Before launch, set `HYPERLOOM_RUNTIME_DIR` to a private directory
outside any source checkout, on storage suitable for SQLite (typically
node-local fast disk). Session artifacts can stay under `USER_DATA_PATH` on a
shared mount:

```bash
export USER_DATA_PATH=/path/on/shared/storage/hyperloom-sessions
export HYPERLOOM_RUNTIME_DIR=/var/lib/hyperloom/runtime
mkdir -p "$HYPERLOOM_RUNTIME_DIR"
chmod 700 "$HYPERLOOM_RUNTIME_DIR"
```

Use a per-job subdirectory when several runs can share one node (for example,
`/var/lib/hyperloom/runtime-$SLURM_JOB_ID`).

**Verify**:

1. Before and during the run, confirm the configured parent is on suitable
   storage:
   ```bash
   df -T "$HYPERLOOM_RUNTIME_DIR"
   ```
2. While a turn is still open (before the SDK client closes), pool-timeout
   warnings in Codex `logs_*.sqlite` under the active `.hyperloom-codex-home-*`
   directory indicate the parent is still unsuitable. After close, rely on the
   stage finishing within budget rather than inspecting removed directories.

See [Environment variables](environment-variables.md) for
`HYPERLOOM_RUNTIME_DIR` and Codex `CODEX_HOME` lifecycle.

---

## TraceLens root or CLI check fails

**Symptom.** `trace_analyze` returns `TraceLens root not found`,
`incomplete (not a git checkout)`, or
`TraceLens_generate_perf_report_pytorch_inference: command not found`.

**Cause**: `TraceLens-internal` isn't installed, or the legacy
training-mode CLI is being looked for (no longer accepted as of v0.4).

**Fix**:

1. Re-run `install.sh` (it clones AMD-AGI/TraceLens into the pod-local
   open-source checkout root, pins it to a fixed SHA, runs `pip install -e`,
   and smokes the CLI):
   ```bash
   bash "$REPO_ROOT/hyperloom/agents/kernel/scripts/install.sh"
   ```
2. If `install.sh` succeeds but the CLI still isn't on PATH, install
   manually. By default use the installer-managed clone; only point
   `TRACELENS_ROOT` at a pre-existing checkout you maintain as an
   explicit operator override — that skips both the clone and the SHA
   pin:
   ```bash
   export TRACELENS_ROOT="${TRACELENS_ROOT:-${HYPERLOOM_CACHE_DIR:-$REPO_ROOT/.cache}/TraceLens}"
   cd "$TRACELENS_ROOT"
   pip install -e .
   TraceLens_generate_perf_report_pytorch_inference --help
   ```
3. The optional internal extension is enabled only when
   `TRACELENS_INTERNAL_ROOT` is set to your own existing checkout
   (no default path; leave unset for the open-source-only report).

---

## TraceLens root dangling after a deps-cache default change

**Symptom**: `trace_analyze` fails with `TraceLens root not found` or
`incomplete (not a git checkout)` even after re-running `install.sh`,
and the runtime never self-heals the checkout.

**Cause**: The open-source deps now default to the writable, repo-local
`${HYPERLOOM_CACHE_DIR:-$REPO_ROOT/.cache}`, cloned per revision as
`TraceLens@<sha>`. A stale `kernel-agent.env.sh` or `.env` that still
pins `TRACELENS_ROOT` to an old path (for example, `/tmp/...` or
`/opt/hyperloom/open-source-repos/...`) is treated as an explicit
operator override — self-heal is scoped to the installer-managed default
only, so a non-default path is never auto-re-cloned and fails fast when
that path is missing or reaped.

**Fix**: Pick one:

1. **Re-run the installer (preferred).** Remove the stale pin so the
   default is re-resolved to the cache root, then reinstall:
   ```bash
   unset TRACELENS_ROOT   # remove any hard-coded old path from env or .env first
   bash "$REPO_ROOT/hyperloom/agents/kernel/scripts/install.sh"
   ```
   The installer rewrites `kernel-agent.env.sh` with the
   `${HYPERLOOM_CACHE_DIR:-$REPO_ROOT/.cache}/TraceLens@<sha>` default and
   clones and pins TraceLens there.
2. **Point the deps cache at a different writable directory** (for example, a
   shared or larger volume):
   ```bash
   export HYPERLOOM_CACHE_DIR="$USER_DATA_PATH/.hyperloom-cache"
   unset TRACELENS_ROOT
   bash "$REPO_ROOT/hyperloom/agents/kernel/scripts/install.sh"
   ```
3. **Keep an operator checkout** only if you deliberately maintain one —
   set `TRACELENS_ROOT` to that path. It is adopted as-is (no clone, no
   SHA pin, no self-heal), so you own keeping it present and on the right
   ref.

---

## Resume fails: "manifest.json not found"

**Symptom.** `python -m hyperloom.inference_optimizer.cli optimize --resume-from` exits with
`manifest.json not found under <dir>` or `state.json missing`.

**Cause**: `USER_DATA_PATH` points at a different directory than the
original session, or the session never reached the point of writing
`manifest.json` (failed before the session manifest was written).

**Fix**:

1. Verify env and locate the real session dir (the default
   `per_model_ts` layout nests it at
   `$USER_DATA_PATH/<model>/<UTC_ts>/`, not the workspace root):
   ```bash
   echo "$USER_DATA_PATH"
   find "$USER_DATA_PATH" -name manifest.json
   ```
2. Pass the actual session directory:
   ```bash
   python3 -m hyperloom.inference_optimizer.cli optimize --resume-from "$SESSION_DIR"
   ```
3. If `manifest.json` truly never existed, resume is not possible —
   restart with a fresh `--model …` launch.

---

## KB writes silently failing

**Symptom**: Recipe KB warm-start is empty or new local KB records do not
appear under the selected local KB root.

**Cause**: The selected local root is read-only/full, permission is denied, a
one-time legacy Recipe migration failed, or the explicitly selected remote
KB Store backend is unreachable.

**Fix**:

1. Resolve and test the local KB root:
   ```bash
   KB_ROOT="${KNOWLEDGE_LOCAL_ROOT:-${HYPERLOOM_LOCAL_KB_ROOT:-${USER_DATA_PATH:-$HOME/.cache/hyperloom}/knowledge}}"
   echo "$KB_ROOT"
   mkdir -p "$KB_ROOT"
   touch "$KB_ROOT/.write-test" && rm "$KB_ROOT/.write-test"
   ```
2. If `KNOWLEDGE_STORE_MODE=remote`, verify `KB_STORE_URL` reachability and
   `KB_STORE_TOKEN` from the same pod. Remote mode does not fall back to local
   and currently writes only at CLOSE.
3. After an upgrade, inspect the startup error before moving data manually.
   Stop old writers, back up `$USER_DATA_PATH/kb` (or
   `/workspace/hyperloom/kb`), correct permissions, and retry the one-time
   migration.
4. To skip KB hooks deliberately for a diagnosis run, pass `--degraded-kb`.

Missing remote credentials and failed legacy migration are startup errors by
design; they prevent silent cold starts. See
[Integrate Recipe knowledge base in Hyperloom](integrate-kb.md) for the
detailed resolver and mode contracts.

---

## InferenceX target comparison missing

**Symptom**: Target-analysis step writes a `no_target_gpu_configured`
marker and the run proceeds without an external reference (no "vs
B200" number in the report).

**Cause**: `--compare-against-gpu` was not supplied. The removed `classify`
action no longer derives this automatically.

**Fix**: Add the flag at launch:

```bash
python3 -m hyperloom.inference_optimizer.cli optimize ... --compare-against-gpu B200
```

The marker is informational, not a failure — the optimisation still
runs against your local baseline.

---

## `result.json` written outside the session dir

**Symptom**: A benchmark "succeeds" but the breakdown shows
`baseline.throughput_tok_s_per_gpu = null`; logs reference a
`result.json` written to `--result-dir /tmp/...` outside
`$USER_DATA_PATH`.

**Cause**: A model-specific InferenceX-native benchmark script
hardcodes `--result-dir`, bypassing Hyperloom's session-dir pinning.

**Fix**: Set `$INFERENCE_OPTIMIZER_RESCUE_PATHS` to the directories
the script writes to:

```bash
export INFERENCE_OPTIMIZER_RESCUE_PATHS="/tmp/inferencex_results:/var/tmp/bench"
```

The harvest step scans these on each tick and copies any orphaned
`result.json` into the session dir. See
[Environment variables](environment-variables.md) — "Framework / source-tree discovery".

---

## How do I tell what's actually happening right now?

Three commands give you a fast situation report:

```bash
# Resolve the active session dir first (from launch-info or a running shell).
# Defaults to INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR when set.
SD="${INFERENCE_OPTIMIZER_CURRENT_SESSION_DIR:-$SESSION_DIR}"

# 1. Are events landing?
python -m hyperloom.inference_optimizer.tools.event_counts "$SD"

# 2. What was the last action's outcome?
jq '.optimization_stack | last' "$SD/state.json"

# 3. What phase, lifecycle events, and stop reason are persisted?
python -m hyperloom.inference_optimizer.tools.read_optimizer_state "$SD"
```

See [Hyperloom operator scripts](operator-scripts.md) for the full set of
inspection tools.

---

## Getting more help

If the steps above did not resolve your issue, use the following options to get help.

* Open an issue at
  [https://github.com/AMD-AGI/Hyperloom/issues](https://github.com/AMD-AGI/Hyperloom/issues)
  with: Hyperloom git SHA, the launch command, the
  `session_breakdown.json` (or partial state.json) for the failed
  session, and the relevant log excerpt.
* For security-relevant issues, follow the disclosure process in
  [`SECURITY.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/SECURITY.md) instead.
