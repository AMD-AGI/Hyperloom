---
myst:
    html_meta:
        "description": "Step-by-step migration guide for upgrading Hyperloom. Covers required and recommended changes from 0.5.x to 0.6.0 and the generic upgrade procedure."
        "keywords": "Hyperloom, upgrade, migration, version, changelog, 0.6.0, USER_DATA_PATH, GEAK, TraceLens, Ray, AMD GPU, ROCm, session, inference optimizer"
---
# Upgrade Hyperloom version

Per-version migration steps. This page is a companion to
[`CHANGELOG.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/CHANGELOG.md): the changelog answers *what
changed*, this page answers *what you have to do about it*.

If you are starting fresh, skip this page and follow the
[installation instructions](../install/install.md).

---

## Conventions

These labels are used throughout this page to indicate urgency:

* **Required**: Your run will fail or behave incorrectly until you do
  this.
* **Recommended**: Your run will still work, but you'll get a
  deprecation warning or sub-optimal behavior.
* **Optional**: Strictly improves UX or unlocks new features.

Hyperloom doesn't mutate your `.env` on upgrade; all migrations
below are explicit.

```{note}
Shell paths on this page follow the recommended `pip install --target .` layout.
In a source checkout, replace the `hyperloom/` prefix with `src/hyperloom/`.
The command blocks assume `REPO_ROOT` points at that workspace. No installer
exports it for you, so run `export REPO_ROOT="$(pwd -P)"` from the workspace
first — it does not survive a new shell.

The `${USER_DATA_PATH:-/workspace/hyperloom}` fallback below is the last resort
in the chain and only correct where `/workspace` is writable; on a bare-metal
host the CLI defaults to `session/` under the current directory instead. Prefer
`KERNEL_AGENT_ENV`, which the installer writes for you.
```

---

## Upgrading from 0.5.x → 0.6.0

Apply the following changes in order. Required steps must be completed before running; recommended and optional steps improve behavior or unlock new features.

### Required: rename multi-node image / GPU flags and envs

The `optimize` CLI multi-node image and per-node GPU flags were renamed
and the legacy names are no longer accepted (no alias). Any launcher that
passes the old flags will fail with `unrecognized arguments`. The image
and per-node GPU count are now set purely through flags; the previous
`INFERENCE_OPTIMIZER_RAYJOB_IMAGE` env is no longer read, so move its value
onto `--mn-image`.

```diff
# run launchers, k8s ConfigMaps, .env
- --rayjob-image harbor/...
+ --mn-image harbor/...
- --rayjob-gpus-per-node 8
+ --gpus-per-node 8
- INFERENCE_OPTIMIZER_RAYJOB_IMAGE=harbor/...
+ --mn-image harbor/...
```

The new flags cover both multi-node backends (`rayjob` head+workers and
`infera` worker/prefill/decode pods), which is why they dropped the
`rayjob`-specific prefix.

### Recommended: review `--model-class` if you relied on live classification

The live `classify` action was removed. Current Coordinator boot still infers
and persists `model_class` from model metadata or model-path family keywords
when possible, but launchers that know the class should pass it explicitly to
avoid a generic fallback:

```diff
python3 -m hyperloom.inference_optimizer.cli optimize \
    --model /path/to/GLM-5-FP8 \
    --framework sglang \
    --gpu-type mi355x \
+   --model-class moe_mla_nsa \
    --isl 1024 --osl 1024 \
    --max-hours 2.0
```

Recognised `--model-class` values (case-insensitive, with `-`/`+`/space
tolerated; see [Inference Optimizer Skill](https://github.com/AMD-AGI/Hyperloom/blob/main/src/hyperloom/inference_optimizer/SKILL.md)):
`dense`, `moe_mla`, `moe_swa`, `moe_mla_nsa`.

If `--model-class` is omitted and inference cannot determine the family, the
Coordinator falls back to a generic dense prior — likely sub-optimal for mixture of experts (MoE) /
multi-head latent attention (MLA) / native sparse attention (NSA) models.

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

Earlier launchers might have waited for the Coordinator to emit a
`setup` action. Move all setup work to **before** the
`python -m hyperloom.inference_optimizer.cli optimize` call:

```diff
# launcher.sh
- python3 -m hyperloom.inference_optimizer.cli optimize ... # expects setup as first action
+ bash "$REPO_ROOT/hyperloom/inference_optimizer/assets/install.sh"
+ . "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"
+ ray stop --force; ulimit -Sn "${RAY_MIN_NOFILE:-65536}" 2>/dev/null || true; ray start --head --num-gpus="$RAY_NUM_GPUS" --include-dashboard=false
+ python3 -m hyperloom.inference_optimizer.cli optimize ...
```

### Recommended: review the `KERNEL_OPT_BACKEND_ORDER` default

Bare-metal installs now default `KERNEL_OPT_BACKEND_ORDER` to `geak`. If you
had a custom order hard-coded, confirm it is still intentional.

### Recommended: set `INFERENCE_OPTIMIZER_RESCUE_PATHS` if you use model-specific benchmark scripts

When `--gpu-type` is set, Hyperloom pins
`benchmark_script=<framework>_<gpu_type>.sh` to stop InferenceX-native
scripts from leaking `result.json` outside the session dir. If you
intentionally use a model-specific script (for example, `dsr1_fp8_mi300x.sh`),
keep passing `benchmark_script=` explicitly — operator overrides
still win against the generic-script pin. You might additionally want
to set `$INFERENCE_OPTIMIZER_RESCUE_PATHS` so the harvest step can
recover any leaked `result.json` files written to hardcoded
`--result-dir` locations.

### Recommended: stop expecting `standalone_analysis.md` or `tracelens_report.md`

The kernel-agent no longer aliases the TraceLens v0.3 report. The
canonical analysis path is now surfaced by the `trace_analyze` request
handler as `trace_report_path` and forwarded to the lifecycle as
`analysis_report_path` (pointing at
`$SESSION_DIR/kernel-agent/runs/<session_id>/tracelens/analysis.md`).
The `--compat-report-path` argument was removed.

### Roofline is now on by default (no PMC env toggle)

There is no `HYPERLOOM_ENABLE_PMC_ROOFLINE` environment variable. Roofline
analysis (the composite `profile` + `trace_analyze` + `analysis.md` path) is
controlled by the CLI flag `--enable-roofline`, which defaults **on**. Pass
`--no-enable-roofline` for a profile-only run. See
[Environment variables](environment-variables.md).

### Schema compatibility

`session_breakdown.json` emits `hyperloom.session_breakdown.v6.0`. V6 is a
breaking cutover for the timeline: each action records its own event as it
runs, so an event's `start_time` is when the work started rather than when its
artefacts were written, and the KERNEL and BASELINE projections are no longer
emitted. Ordering that relied on the old collapsed windows changes as a result.

V5 was the preceding cutover, for optimization results: adopted optimizations
are reported only through `optimizations`, and the `optimization_stack`,
`attribution`, GEAK invocation, Forge invocation, and GEMM-tuning projections
are no longer emitted. Consumers reading archived v2 / v3 / v4 / v5 documents
need a downstream migration, as described in
[`session_breakdown.json` integration in Hyperloom](session-breakdown.md).

---

## Upgrading from pre-0.5 (training and MLPerf-training)

Hyperloom is **inference-only** since v0.4. If you are still on a
training-mode build:

* The training-mode TraceLens CLI
  (`TraceLens_generate_perf_report_pytorch_training`) is no longer
  accepted by `install.sh`. Use the `_inference` variant.
* Training and MLPerf-training skills have been removed from this
  repo. There is no in-place migration; switch to the inference flow
  documented in the [setup and examples guide](../../examples/README.md).

---

## Required: migrate to exclusive KnowledgePlane modes

KnowledgePlane now selects exactly one Recipe backend. This is a breaking
change for deployments that previously got remote-first reads merely by
exporting GBrain credentials:

```diff
  GBRAIN_BASE_URL=https://gbrain.example
  GBRAIN_TOKEN=...
+ KNOWLEDGE_STORE_MODE=remote
```

Without the explicit mode, local/default ignores both credentials. Set
`KNOWLEDGE_STORE_MODE=local` (or leave it unset) for local-only operation.

The implicit local root changed from `$USER_DATA_PATH/kb` (or
`/workspace/hyperloom/kb`) to `$USER_DATA_PATH/knowledge` (or
`~/.cache/hyperloom/knowledge`). Upgrade local deployments in this order:

1. Stop every old Hyperloom process that can write the legacy Recipe KB.
2. Back up the legacy root, including `recipe.json`, `history/`, and
   `attempts.ndjson`.
3. Install the new revision while leaving `KNOWLEDGE_LOCAL_ROOT`,
   `--local-kb-root`, and `HYPERLOOM_LOCAL_KB_ROOT` unset for the first start.
4. Start one new local-mode process. It migrates the Recipe corpus once,
   excludes live lock/temp files, preserves unrelated data already under the
   new knowledge root, and writes a durable marker only after success.
5. Verify warm-start data, then roll out the remaining new processes.

If the destination already contains Recipes or the completion marker,
migration is intentionally skipped. A migration error while legacy Recipes
exist fails startup; restore from the backup or correct permissions and retry.
Deployments that explicitly keep `--local-kb-root` or
`HYPERLOOM_LOCAL_KB_ROOT` continue using that path and are not migrated.

---

## Generic upgrade procedure

For any minor or patch upgrade:

1. Pull the new Hyperloom revision into `$REPO_ROOT`.
2. Re-run `bash "$REPO_ROOT/hyperloom/inference_optimizer/assets/install.sh"`. The
   installer is idempotent: it picks up new GEAK / TraceLens versions,
   refreshes generated LLM gateway aliases, and regenerates `kernel-agent.env.sh`.
3. Re-source the env file:
   ```bash
   . "${KERNEL_AGENT_ENV:-${USER_DATA_PATH:-/workspace/hyperloom}/runtime/kernel-agent.env.sh}"
   ```
4. If you have ongoing sessions you want to resume across the upgrade,
   verify `manifest.json` and `state.json` are intact, then run
   `python -m hyperloom.inference_optimizer.cli optimize --resume-from "$SESSION_DIR"`.

Upgrades do not rewrite explicit `HYPERLOOM_LOCAL_KB_ROOT` paths or historical
sessions. The one-time implicit Recipe-root migration described above is the
only automatic storage move.

---

## Related guides

Use these resources for related reference information:

* [`CHANGELOG.md`](https://github.com/AMD-AGI/Hyperloom/blob/main/CHANGELOG.md): Full per-release notes.
* [Hyperloom authentication and credentials](authentication.md): Credential and path env reference.
* [Environment variables](environment-variables.md): Every
  environment variable read by the runtime.
* [Hyperloom self-hosting and operations guide](operations.md): Self-host runbook.
