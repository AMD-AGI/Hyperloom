# Upgrading Hyperloom

Per-version migration steps. This page is a companion to
[`CHANGELOG.md`](../CHANGELOG.md): the changelog answers *what
changed*, this page answers *what you have to do about it*.

If you are starting fresh, skip this page and follow the root
[`README.md`](../README.md) quickstart.

---

## Conventions

* **Required** — your run will fail or behave incorrectly until you do
  this.
* **Recommended** — your run will still work, but you'll get a
  deprecation warning or sub-optimal behaviour.
* **Optional** — strictly improves UX or unlocks new features.

Hyperloom does **not** mutate your `.env` on upgrade; all migrations
below are explicit.

---

## Upgrading from 0.5.x → 0.6.0

### Required: rename `INFERENCE_OPTIMIZER_SESSION_DIR` → `USER_DATA_PATH`

The session directory env was renamed and the legacy variable is **no
longer read**. Any launcher that exports the old name will silently
fall back to the default `/workspace/hyperloom` instead of honouring
your override.

```diff
# .env, run launchers, k8s ConfigMaps
- INFERENCE_OPTIMIZER_SESSION_DIR=/wekafs/hyperloom/sessions/me
+ USER_DATA_PATH=/wekafs/hyperloom/sessions/me
```

`WORKSPACE_PATH` is retired for optimizer state — rename to
`USER_DATA_PATH`. A few legacy tools still read it as a fallback (critic
static assets, TraceLens workspace discovery).

### Recommended: review `--model-class` if you relied on live classification

The live `classify` action was removed. Current Coordinator boot still infers
and persists `model_class` from model metadata or model-path family keywords
when possible, but launchers that know the class should pass it explicitly to
avoid a generic fallback:

```diff
inference_optimizer optimize \
    --model /path/to/GLM-5-FP8 \
    --framework sglang \
    --gpu-type mi355x \
+   --model-class moe_mla_nsa \
    --isl 1024 --osl 1024 \
    --max-hours 2.0
```

Supported `--model-class` values (see
`src/hyperloom/inference_optimizer/SKILL.md` §"Model classes"): `dense`,
`moe_mla`, `moe_swa`, `moe_mla_nsa`.

If `--model-class` is omitted and inference cannot determine the family, the
Coordinator falls back to a generic dense prior — likely sub-optimal for MoE /
MLA / NSA models.

### Required: pass `--compare-against-gpu` to opt into InferenceX reference fetching

The `classify` action used to derive this implicitly. Without
`--compare-against-gpu`, `target_analysis` writes a
`no_target_gpu_configured` marker and the run proceeds without an
external reference (the optimisation still works; you just don't get
the "vs B200" comparison number).

```diff
+ --compare-against-gpu B200
```

### Required: setup is no longer an in-loop action

Earlier launchers may have waited for the Coordinator to emit a
`setup` action. Move all setup work to **before** the
`inference_optimizer optimize` call:

```diff
# launcher.sh
- inference_optimizer optimize ... # expects setup as first action
+ . "${USER_DATA_PATH:-/workspace/hyperloom}/runtime/local-setup.env.sh"
+ bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh"
+ . "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"
+ ray stop --force; ulimit -Sn "${RAY_MIN_NOFILE:-65536}" 2>/dev/null || true; ray start --head --num-gpus="$RAY_NUM_GPUS" --include-dashboard=false
+ inference_optimizer optimize ...
```

### Recommended: review the `KERNEL_OPT_BACKEND_ORDER` default

The default kernel-opt ladder is now `forge,geak,claude,codex,cursor`
(Forge-GEAK-OOB). `cursor` is auto-dropped from the default when
`$CURSOR_API_KEY` is unset. If you had a custom order hard-coded (e.g.
`claude,geak`) or relied on the previous `forge,geak`-only default, confirm it
is still intentional.

### Recommended: set `INFERENCE_OPTIMIZER_RESCUE_PATHS` if you use model-specific benchmark scripts

When `--gpu-type` is set, Hyperloom pins
`benchmark_script=<framework>_<gpu_type>.sh` to stop InferenceX-native
scripts from leaking `result.json` outside the session dir. If you
intentionally use a model-specific script (e.g. `dsr1_fp8_mi300x.sh`),
keep passing `benchmark_script=` explicitly — operator overrides
still win against the generic-script pin. You may additionally want
to set `$INFERENCE_OPTIMIZER_RESCUE_PATHS` so the harvest step can
recover any leaked `result.json` files written to hardcoded
`--result-dir` locations.

### Recommended: stop expecting `standalone_analysis.md` / `tracelens_report.md`

The kernel-agent no longer aliases the TraceLens v0.3 report. The
canonical analysis path is now surfaced by the `trace_analyze` request
handler as `trace_report_path` and forwarded to the lifecycle as
`analysis_report_path` (pointing at
`$SESSION_DIR/kernel-agent/runs/<session_id>/tracelens/analysis.md`).
The `--compat-report-path` argument was removed.

### Optional: roofline analysis

Roofline analysis (roofline → trace_analyze → TraceLens) is controlled
by the `--enable-roofline` CLI flag (on by default). Useful for
compute-bound workloads; adds time per profile call. Diffusion runs
emit a `diffusion_roofline.json` sidecar under the session's
`kernel-agent/runs/...` tree.

### Optional: opt into the Cursor backend

If you have a Cursor account, export `CURSOR_API_KEY=crsr_...` to add
`cursor` to the kernel-opt ladder. Without the key, behaviour is
unchanged (silently dropped).

### Schema compatibility

`session_breakdown.json` currently emits either
`hyperloom.session_breakdown.v2` or `hyperloom.session_breakdown.v3.0`
depending on the aggregation path. The `v3.0` file is additive and
wire-compatible for `v2` consumers that tolerate unknown fields. Consumers must
not gate on exact string equality; accept the v2/v3 family as described in
[`INTEGRATION_SESSION_BREAKDOWN.md`](INTEGRATION_SESSION_BREAKDOWN.md).

---

## Upgrading from pre-0.5 (training / MLPerf-training)

Hyperloom is **inference-only** since v0.4. If you are still on a
training-mode build:

* The training-mode TraceLens CLI
  (`TraceLens_generate_perf_report_pytorch_training`) is no longer
  accepted by `install.sh`. Use the `_inference` variant.
* Training and MLPerf-training skills have been removed from this
  repo. There is no in-place migration; switch to the inference flow
  documented in the root [`README.md`](../README.md).

---

## Generic upgrade procedure

For any minor / patch upgrade:

1. Pull the new Hyperloom revision into `$REPO_ROOT`.
2. Re-run `bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh"`. The
   installer is idempotent: it picks up new GEAK / TraceLens versions,
   regenerates gateway aliases, and rewrites `kernel-agent.env.sh`.
3. Re-source the env file:
   ```bash
   . "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"
   ```
4. If you have ongoing sessions you want to resume across the upgrade,
   verify `manifest.json` and `state.json` are intact, then run
   `inference_optimizer optimize --resume`.

Upgrades **do not** touch `HYPERLOOM_LOCAL_KB_ROOT` or `$USER_DATA_PATH`.
Your local KB and historical sessions are preserved.

---

## See also

* [`CHANGELOG.md`](../CHANGELOG.md) — full per-release notes.
* [`ENV_AND_AUTH.md`](ENV_AND_AUTH.md) — credential & path env
  reference.
* [`CONFIGURATION_REFERENCE.md`](CONFIGURATION_REFERENCE.md) — every
  environment variable read by the runtime.
* [`OPERATIONS.md`](OPERATIONS.md) — self-host runbook.
