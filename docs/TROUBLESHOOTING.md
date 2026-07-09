# Troubleshooting

A consolidated symptom → cause → fix index for the failures Hyperloom
users hit most often. If a symptom isn't listed here, check the
upstream SKILL file for the component you're touching:
[`src/hyperloom/inference_optimizer/SKILL.md`](../src/hyperloom/inference_optimizer/SKILL.md),
[`src/hyperloom/agents/kernel/SKILL.md`](../src/hyperloom/agents/kernel/SKILL.md),
[`src/hyperloom/agents/critic/SKILL.md`](../src/hyperloom/agents/critic/SKILL.md),
[`src/hyperloom/agents/robustness/SKILL.md`](../src/hyperloom/agents/robustness/SKILL.md).

---

## Gateway 401

**Symptom.** A tool exits with one of:

* `HTTP 401 Unauthorized`
* `Primus.00009 token not present`
* `Claude SDK exit code 1`
* `OpenAI SDK: AuthenticationError`

**Cause.** The `claude` / `codex` CLIs and sub-agents call the upstream
gateway directly. A 401 means the resolved gateway credentials are
missing or expired.

**Fix.**

1. Verify `SAFE_API_KEY` is set: `echo "$SAFE_API_KEY"`.
2. Re-run preflight or the installer to refresh resolved gateway
   aliases:
   ```bash
   bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh"
   ```

See [`ENV_AND_AUTH.md`](ENV_AND_AUTH.md) §5 for gateway wiring and 401
recovery.

---

## Ray `--num-gpus` rejected

**Symptom.** `ray start --head ... --num-gpus=N` fails with
`Error: no such option: --num-gpus` or similar Click errors.

**Cause.** Click ≥ 8.3 is incompatible with the Ray 2.44 CLI shipped
by Hyperloom's installer.

**Fix.**

```bash
pip install --quiet 'click<8.3.0' 'ray[default]==2.44.1'
ray --version
```

The same fix applies when `ray --version` itself fails after a pod or
venv rebuild.

---

## Ray tasks stuck pending forever

**Symptom.** `ray status` shows pending tasks; GPU usage is 0% even
though the node has free GPUs.

**Cause.** Ray was started with `--num-gpus=0` (or omitted, which
defaults to 0 on some images). GEAK and OOB submit tasks with
`num_gpus>=1` and will wait indefinitely.

**Fix.**

```bash
RAY_NUM_GPUS="${RAY_NUM_GPUS:-$(python3 -c 'import torch; print(torch.cuda.device_count() or 1)')}"
ray stop --force || true
# issue #433: raise the soft open-files limit before `ray start` so the
# raylet stays up (see "Ray raylet unstable / zombie" below).
ulimit -Sn "${RAY_MIN_NOFILE:-65536}" 2>/dev/null || true
ray start --head --disable-usage-stats --num-gpus="$RAY_NUM_GPUS" --include-dashboard=false
ray status
```

> **Note.** `hyperloom.inference_optimizer.cli` does **not** auto-start Ray.
> Always start it before launching `inference_optimizer optimize`.

---

## Ray raylet unstable / zombie (fd-limit too low)

**Symptom.** During GEAK dispatch the raylet aborts on startup
(`SIGABRT`), or `ray stop` reports it could not be stopped and leaves
it behind, e.g.:

```
WARN scripts.py:1287 -- Stopped only 0 out of 391 Ray processes within
the grace period 16 seconds. Remaining processes [... name='raylet',
status='zombie' ...]
```

**Cause.** At the container default `ulimit -n` (1024) the open-files
limit is far too low for the raylet, which opens many fds (sockets,
plasma store, per-worker pipes) — on a 384-CPU node with hundreds of
workers it needs `nofile >= 65536` (issue #433).

**Fix.** Launch the kernel-agent container with a high open-files limit
(this is a **hard requirement** — only the container launch can lift the
*hard* cap):

```bash
docker run --ulimit nofile=1048576 ...   # minimum: --ulimit nofile=65536
```

The runtime also runs an fd-limit preflight that raises this process's
*soft* limit (up to the hard cap) before every `ray start`
(`src/hyperloom/agents/kernel/scripts/install.sh` `ensure_fd_limit_for_ray` and
`src/hyperloom/agents/kernel/tools/backends/ray_runtime.py` `ensure_fd_limit`), so a
high hard cap is enough; you do not need to set the soft limit yourself.
Override the target with `RAY_MIN_NOFILE` if needed. If the preflight
warns that the **hard** cap is below the target, the container was not
launched with `--ulimit nofile=...` — fix the launch command.

---

## VRAM exhaustion / IR-1 error

**Symptom.** Inference server exits with `HSA: out of memory`,
`std::runtime_error: ROCm IR-1`, `OOM` during the prefill step, or
the baseline benchmark fails with VRAM allocation errors.

**Cause.** One of:

* `TP` is too small for the model's weights.
* `MAX_MODEL_LEN` is set higher than the KV cache budget allows at
  the current `CONC`.
* A previous server process leaked memory and didn't release it.

**Fix.**

1. Confirm no zombie inference server is holding VRAM:
   `rocm-smi --showmemuse` then kill stragglers with
   `pkill -f sglang.launch_server` (or `vllm`).
2. Bump `TP` (e.g. 4 → 8) so weights and KV cache fit.
3. Lower `MAX_MODEL_LEN` to the smallest length your workload actually
   needs (default 8192 is often too generous).
4. Lower `CONC` to reduce simultaneous KV cache pressure.

The Robustness agent classifies repeated OOMs as a `log_error_pattern`
high-severity symptom and emits an `escalate_strategy_change` intent;
check the latest finding in
`$SESSION_DIR/agents/robustness/findings/<session_id>.jsonl` for
context.

---

## GEAK fails fast with "profiler_mcp not installed"

**Symptom.** GEAK attempts abort within 4 minutes with zero-byte
baseline files; logs mention a missing `profiler_mcp` or one of the
other GEAK MCP packages.

**Cause.** `install.sh` did not finish installing the GEAK MCP packages
(`rag-mcp`, `profiler-mcp`, `cross-session-memory-mcp`,
`automated-test-discovery`). `metrix-mcp` is no longer installed separately;
its functionality is provided through `profiler-mcp`. Common trigger: pip
install failed on a transient registry hiccup and the installer continued.

**Fix.**

```bash
bash "$REPO_ROOT/src/hyperloom/agents/kernel/scripts/install.sh" --check-only
# If --check-only reports missing packages, re-run without --check-only:
bash "$REPO_ROOT/src/hyperloom/agents/kernel/scripts/install.sh"
```

The installer is idempotent and re-installs only what's missing.

---

## TraceLens root or CLI check fails

**Symptom.** `trace_analyze` returns `TraceLens root not found`,
`incomplete (not a git checkout)`, or
`TraceLens_generate_perf_report_pytorch_inference: command not found`.

**Cause.** TraceLens-internal isn't installed, or the legacy
training-mode CLI is being looked for (no longer accepted as of v0.4).

**Fix.**

1. Re-run `install.sh` (it clones AMD-AGI/TraceLens into the pod-local
   open-source checkout root, pins it to a fixed SHA, runs `pip install -e`,
   and smokes the CLI):
   ```bash
   bash "$REPO_ROOT/src/hyperloom/agents/kernel/scripts/install.sh"
   ```
2. If `install.sh` succeeds but the CLI still isn't on PATH, install
   manually. By default use the installer-managed clone; only point
   `TRACELENS_ROOT` at a pre-existing checkout you maintain as an
   explicit operator override — that skips both the clone and the SHA
   pin:
   ```bash
   export TRACELENS_ROOT="${TRACELENS_ROOT:-${HYPERLOOM_OPEN_SOURCE_ROOT:-/opt/hyperloom/open-source-repos}/TraceLens}"
   cd "$TRACELENS_ROOT"
   pip install -e .
   TraceLens_generate_perf_report_pytorch_inference --help
   ```
3. The optional internal extension is enabled only when
   `TRACELENS_INTERNAL_ROOT` is set to your own existing checkout
   (no default path; leave unset for the open-source-only report).

---

## TraceLens root dangling after the /tmp → /opt default move (#722)

**Symptom.** On a pod provisioned before #722, `trace_analyze` still fails
with `TraceLens root not found` (or `incomplete (not a git checkout)`) even
after a tmp-reaper wipe, and the runtime never self-heals it.

**Cause.** The open-source deps default moved from the ephemeral
`${TMPDIR:-/tmp}/hyperloom/open-source-repos` to the pod-internal
`/opt/hyperloom/open-source-repos`. A stale `kernel-agent.env.sh` (or `.env`)
that still pins `TRACELENS_ROOT` to the old `/tmp/...` path is treated as an
**explicit operator override**: self-heal is scoped to the installer-managed
default only, so a non-default `/tmp` path is never auto-re-cloned and fails
fast once `/tmp` is reaped.

**Fix.** Pick one:

1. **Re-run the installer (preferred).** Drop the stale pin so the default is
   re-resolved to `/opt`, then reinstall:
   ```bash
   # remove any hard-coded old /tmp TRACELENS_ROOT from env/.env first
   unset TRACELENS_ROOT
   bash "$REPO_ROOT/src/hyperloom/agents/kernel/scripts/install.sh"
   ```
   The installer rewrites `kernel-agent.env.sh` with the `/opt` default and
   clones+pins TraceLens there.
2. **Point the deps root at a writable dir** if the pod runs as non-root or
   `/opt` is read-only (the installer `mkdir -p`s the root, so it must be
   writable):
   ```bash
   export HYPERLOOM_OPEN_SOURCE_ROOT="$USER_DATA_PATH/open-source-repos"
   unset TRACELENS_ROOT
   bash "$REPO_ROOT/src/hyperloom/agents/kernel/scripts/install.sh"
   ```
3. **Keep an operator checkout** only if you deliberately maintain one — set
   `TRACELENS_ROOT` to that path. It is adopted as-is (no clone, no SHA pin,
   no self-heal), so you own keeping it present and on the right ref.

---

## Cursor backend gets HTTP 401 (separate from the AMD gateway creds)

**Symptom.** The OOB `cursor` backend specifically returns 401 even
though `claude` / `codex` work fine.

**Cause.** `CURSOR_API_KEY` is missing or invalid. The Cursor backend
talks to Cursor's own gateway, **not** the AMD primus-safe gateway,
and requires a separate `crsr_...` key.

**Fix.**

* If you don't have a Cursor account: nothing to do. `cursor` is the tail of the
  default `forge,geak,claude,codex,cursor` ladder but is auto-dropped when
  `CURSOR_API_KEY` is unset, so the default `forge,geak,claude,codex` run is
  fully functional without it. If you explicitly requested `--backends cursor`
  and don't have a key, remove the flag.
* If you do have a Cursor account:
  ```bash
  export CURSOR_API_KEY=crsr_...
  bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh"   # picks up the new key
  ```

See [`ENV_AND_AUTH.md`](ENV_AND_AUTH.md) §3 for the Cursor key
specifics.

---

## Resume fails: "manifest.json not found"

**Symptom.** `inference_optimizer optimize --resume` exits with
`manifest.json not found under <dir>` or `state.json missing`.

**Cause.** `USER_DATA_PATH` points at a different directory than the
original session, or the session never reached the point of writing
`manifest.json` (failed before the session manifest was written).

**Fix.**

1. Verify env and locate the real session dir (the default
   `per_model_ts` layout nests it at
   `$USER_DATA_PATH/<model>/<UTC_ts>/`, not the workspace root):
   ```bash
   echo "$USER_DATA_PATH"
   find "$USER_DATA_PATH" -name manifest.json
   ```
2. If you used a custom path the first time, pass the actual session
   directory explicitly:
   ```bash
   inference_optimizer optimize --resume --resume-from "$SESSION_DIR"
   ```
3. If `manifest.json` truly never existed, resume is not possible —
   restart with a fresh `--model …` launch.

---

## KB writes silently failing

**Symptom.** Recipe KB warm-start is empty, gbrain read-side enrichment is
missing, or new local KB records do not appear under the selected local KB root.

**Cause.** The local recipe KB root is on a read-only/full mount, permission is
denied, or the optional remote gbrain KB is unreachable.

**Fix.**

1. Resolve and test the local KB root:
   ```bash
   KB_ROOT="${HYPERLOOM_LOCAL_KB_ROOT:-${USER_DATA_PATH:-/workspace/hyperloom}/kb}"
   echo "$KB_ROOT"
   mkdir -p "$KB_ROOT"
   touch "$KB_ROOT/.write-test" && rm "$KB_ROOT/.write-test"
   ```
2. If you configured `GBRAIN_BASE_URL` / `GBRAIN_TOKEN`, verify gbrain
   reachability from inside the same pod. If it is intentionally unavailable,
   unset them and run local-only.
3. To skip KB hooks deliberately for a diagnosis run, pass `--degraded-kb`.

KB unreachability is **never** fatal. Hyperloom continues with local-only or
degraded KB behaviour. See [`KB_GUIDE.md`](KB_GUIDE.md) for the detailed
resolver order.

---

## InferenceX target comparison missing

**Symptom.** Target-analysis step writes a `no_target_gpu_configured`
marker and the run proceeds without an external reference (no "vs
B200" number in the report).

**Cause.** `--compare-against-gpu` was not supplied. Since v0.6, the
`classify` action no longer derives this automatically.

**Fix.** Add the flag at launch:

```bash
inference_optimizer optimize ... --compare-against-gpu B200
```

The marker is informational, not a failure — the optimisation still
runs against your local baseline.

---

## `result.json` written outside the session dir

**Symptom.** A benchmark "succeeds" but the breakdown shows
`baseline.throughput_tok_s_per_gpu = null`; logs reference a
`result.json` written to `--result-dir /tmp/...` outside
`$USER_DATA_PATH`.

**Cause.** A model-specific InferenceX-native benchmark script
hardcodes `--result-dir`, bypassing Hyperloom's session-dir pinning.

**Fix.** Set `$INFERENCE_OPTIMIZER_RESCUE_PATHS` to the directories
the script writes to:

```bash
export INFERENCE_OPTIMIZER_RESCUE_PATHS="/tmp/inferencex_results:/var/tmp/bench"
```

The harvest step scans these on each tick and copies any orphaned
`result.json` into the session dir. See
[`CONFIGURATION_REFERENCE.md`](CONFIGURATION_REFERENCE.md) §6.

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

# 3. Any Robustness findings since the last tick?
tail -n 5 "$SD"/agents/robustness/findings/*.jsonl 2>/dev/null
```

See [`OPERATOR_SCRIPTS.md`](OPERATOR_SCRIPTS.md) for the full set of
inspection tools.

---

## Still stuck

* Open an issue at
  [https://github.com/AMD-AGI/Hyperloom/issues](https://github.com/AMD-AGI/Hyperloom/issues)
  with: Hyperloom git SHA, the launch command, the
  `session_breakdown.json` (or partial state.json) for the failed
  session, and the relevant log excerpt.
* For security-relevant issues, follow the disclosure process in
  [`SECURITY.md`](../SECURITY.md) instead.
