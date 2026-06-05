# Troubleshooting

A consolidated symptom → cause → fix index for the failures Hyperloom
users hit most often. If a symptom isn't listed here, check the
upstream SKILL file for the component you're touching:
[`inference_optimizer/SKILL.md`](../inference_optimizer/SKILL.md),
[`kernel-agent/SKILL.md`](../kernel-agent/SKILL.md),
[`critic-agent/SKILL.md`](../critic-agent/SKILL.md),
[`robustness-agent/SKILL.md`](../robustness-agent/SKILL.md).

---

## Auth-proxy 401

**Symptom.** A tool exits with one of:

* `HTTP 401 Unauthorized`
* `Primus.00009 token not present`
* `Claude SDK exit code 1`
* `OpenAI SDK: AuthenticationError`

**Cause.** The OOB auth-proxy on `127.0.0.1:4002` is down or stuck.
The proxy is what rewrites the upstream `x-api-key` header to
`Authorization: Bearer <SAFE_API_KEY>` for the AMD primus-safe
gateway. Without it, every `claude` / `codex` CLI request 401s.

**Fix.** Re-run the supervisor (idempotent — noop if healthy):

```bash
bash "$REPO_ROOT/kernel-agent/scripts/ensure_auth_proxy.sh"
```

It TCP-probes `:4002`, then HTTP-probes via `curl`. If the port is
open but the probe times out (stuck proxy), it kills the existing
`auth_proxy.py` process and relaunches.

**Still failing?**

* Verify `SAFE_API_KEY` is set: `echo "$SAFE_API_KEY"`.
* Verify nothing else is on port 4002:
  `ss -ltnp | grep :4002`.
* Override the port if 4002 is occupied:
  `AUTH_PROXY_PORT=4012 bash "$REPO_ROOT/kernel-agent/scripts/install.sh"`.

See [`ENV_AND_AUTH.md`](ENV_AND_AUTH.md) §5 for proxy internals.

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
ray start --head --disable-usage-stats --num-gpus="$RAY_NUM_GPUS" --include-dashboard=false
ray status
```

> **Note.** `inference_optimizer.cli` does **not** auto-start Ray.
> Always start it before launching `inference_optimizer optimize`.

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
`$USER_DATA_PATH/agents/robustness/findings/<session_id>.jsonl` for
context.

---

## GEAK fails fast with "profiler_mcp not installed"

**Symptom.** GEAK attempts abort within 4 minutes with zero-byte
baseline files; logs mention a missing `profiler_mcp` or one of the
other GEAK MCP packages.

**Cause.** `install.sh` did not finish installing all five GEAK MCP
packages (`rag-mcp`, `profiler-mcp`, `metrix-mcp`,
`cross-session-memory-mcp`, `automated-test-discovery`). Common
trigger: pip install failed on a transient registry hiccup and the
installer continued.

**Fix.**

```bash
bash "$REPO_ROOT/kernel-agent/scripts/install.sh" --check-only
# If --check-only reports missing packages, re-run without --check-only:
bash "$REPO_ROOT/kernel-agent/scripts/install.sh"
```

The installer is idempotent and re-installs only what's missing.

---

## TraceLens CLI not found

**Symptom.** `tracelens_analysis` returns `CLI not found` or
`TraceLens_generate_perf_report_pytorch_inference: command not found`.

**Cause.** TraceLens-internal isn't installed, or the legacy
training-mode CLI is being looked for (no longer accepted as of v0.4).

**Fix.**

1. Re-run `install.sh` (it clones AMD-AGI/TraceLens to
   `$HYPERLOOM_RUNTIME_DIR/source-mirrors/TraceLens`, pins it to a fixed
   SHA, runs `pip install -e`, and smokes the CLI):
   ```bash
   bash "$REPO_ROOT/kernel-agent/scripts/install.sh"
   ```
2. If `install.sh` succeeds but the CLI still isn't on PATH, install
   manually. By default use the installer-managed clone; only point
   `TRACELENS_ROOT` at a different checkout (e.g. legacy
   `/wekafs/hyperloom/TraceLens-internal`) as an explicit operator
   override — that skips both the clone and the SHA pin:
   ```bash
   export TRACELENS_ROOT="${TRACELENS_ROOT:-${HYPERLOOM_RUNTIME_DIR:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime}/source-mirrors/TraceLens}"
   cd "$TRACELENS_ROOT"
   pip install -e .
   TraceLens_generate_perf_report_pytorch_inference --help
   ```
3. The optional internal extension is enabled only when
   `TRACELENS_INTERNAL_ROOT` is set to your own existing checkout
   (no default path; leave unset for the open-source-only report).

---

## Cursor backend gets HTTP 401 (separate from auth-proxy)

**Symptom.** The OOB `cursor` backend specifically returns 401 even
though `claude` / `codex` work fine.

**Cause.** `CURSOR_API_KEY` is missing or invalid. The Cursor backend
talks to Cursor's own gateway, **not** the AMD primus-safe gateway,
and requires a separate `crsr_...` key.

**Fix.**

* If you don't have a Cursor account: stop trying — Hyperloom auto-skips
  `cursor` when `CURSOR_API_KEY` is unset. The default ladder of
  `geak,claude,codex` is fully functional without it. If you explicitly
  requested `--backends cursor` and don't have a key, remove the flag.
* If you do have a Cursor account:
  ```bash
  export CURSOR_API_KEY=crsr_...
  bash "$REPO_ROOT/kernel-agent/scripts/install.sh"   # picks up the new key
  ```

See [`ENV_AND_AUTH.md`](ENV_AND_AUTH.md) §3 for the Cursor key
specifics.

---

## Resume fails: "manifest.json not found"

**Symptom.** `inference_optimizer optimize --resume` exits with
`manifest.json missing` or `state.json missing`.

**Cause.** `USER_DATA_PATH` points at a different directory than the
original session, or the session never reached the point of writing
`manifest.json` (failed before the session manifest was written).

**Fix.**

1. Verify env:
   ```bash
   echo "$USER_DATA_PATH"
   ls "$USER_DATA_PATH"/{manifest,state}.json
   ```
2. If you used a custom path the first time, re-export it before
   resuming:
   ```bash
   export USER_DATA_PATH=/path/to/your/session
   inference_optimizer optimize --resume
   ```
3. If `manifest.json` truly never existed, resume is not possible —
   restart with a fresh `--model …` launch.

---

## KB writes silently failing

**Symptom.** `judge_bundle.kb_read_skipped_reason` is `kb_unreachable`
in Critic emit JSONs; new lessons don't appear in
`$INFERENCE_OPTIMIZER_KB_ROOT/*.jsonl`.

**Cause.** The KB root is unset, on a read-only mount, full, or
permission-denied.

**Fix.**

1. Verify the env:
   ```bash
   echo "$INFERENCE_OPTIMIZER_KB_ROOT"
   touch "$INFERENCE_OPTIMIZER_KB_ROOT/.write-test" && rm "$INFERENCE_OPTIMIZER_KB_ROOT/.write-test"
   ```
2. If unset, decide: ship with a seed KB
   ([`KB_GUIDE.md`](KB_GUIDE.md) §2 option A) or set `=skip` to
   suppress the warning.
3. If read-only, the agent will degrade gracefully (priors are read,
   writes log warnings, optimisation continues).

KB unreachability is **never** fatal. Critic falls back to
packet-only evidence. See [`KB_GUIDE.md`](KB_GUIDE.md) §4 for the
detailed behaviour.

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
# 1. Are events landing?
python -m inference_optimizer.scripts.event_counts

# 2. What was the last action's outcome?
jq '.optimization_stack | last' "$USER_DATA_PATH/state.json"

# 3. Any Robustness findings since the last tick?
tail -n 5 "$USER_DATA_PATH"/agents/robustness/findings/*.jsonl 2>/dev/null
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
