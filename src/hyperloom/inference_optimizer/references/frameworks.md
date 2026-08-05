# Framework Selection & GPU Runner Type

## Framework Selection

A session is single-framework. Pick `sglang` (default), `vllm`, `atom`,
`xdit`, `hunyuan_image3`, or `worldplay` via `--framework` or `$FRAMEWORK`:

```bash
python3 -m hyperloom.inference_optimizer.cli optimize --framework vllm --model "$MODEL_PATH" --max-hours 2
FRAMEWORK=vllm python3 -m hyperloom.inference_optimizer.cli optimize --model "$MODEL_PATH" --max-hours 2
python3 -m hyperloom.inference_optimizer.cli optimize --framework atom --model "$MODEL_PATH" --max-hours 2  # IR-8 single-node only
python3 -m hyperloom.inference_optimizer.cli optimize --framework xdit --model "$MODEL_PATH" --max-hours 2  # scriptable diffusion
python3 -m hyperloom.inference_optimizer.cli optimize --framework worldplay --model "$MODEL_PATH" --max-hours 12  # scriptable AR video (fps)
```

Resolution order: `--framework` > `$FRAMEWORK` > `sglang` (default).

What this controls:
- Which Magpie YAML the executors default to — `baseline_{sglang,vllm,atom,xdit}.yaml`
  and `profile_{sglang,vllm,atom,xdit}.yaml`. The per-framework resolver
  `_default_profile_config()` in `src/hyperloom/orchestrator/actions/executors/profile.py` picks the right
  file from `$FRAMEWORK`.
- Which framework-specific seed grid the `explore` action falls back to when no
  `params.grid` is supplied. atom is the only framework with a programmatic seed
  today (`_default_grid_for_framework("atom", ...)` in
  `src/hyperloom/orchestrator/actions/executors/explore.py`, populated by `_atom_default_grid()`); sglang
  and vllm continue to rely on the orchestration LLM emitting
  `provenance='default_grid'` variants and will fail with
  `error_class="empty_grid"` on a cold-start with no LLM input.
- Which extra-args env name `_grid_runner` writes (`EXTRA_VLLM_ARGS` /
  `EXTRA_SGLANG_ARGS` / `EXTRA_ATOM_ARGS` / `EXTRA_XDIT_ARGS` /
  `EXTRA_WORLDPLAY_ARGS`).
- Which KB partition orchestration reads for hints.

Mixing frameworks in a single session is not supported; the CLI locks
`$FRAMEWORK` for the run. Resume re-reads `$FRAMEWORK` from the shell — set it
when you resume a non-default session.

### `--framework atom` specifics (IR-8)

Single-node only (`--nodes>=2` fails fast). Shipped configs `baseline_atom.yaml`
/ `profile_atom.yaml`; the Magpie atom wrapper bridges `PROFILE=1` to atom's
`--torch-profiler-dir`, and TraceLens consumes the resulting
`*.pt.trace.json.gz` unchanged. atom source roots (`/app/ATOM/atom/`) are in
PolicyGate's allowlist + `_REUSABLE_SOURCE_ROOTS`, and the repo URL
`https://github.com/ROCm/ATOM.git` is in `hyperloom.agents.framework.repo_map`. Unlike
sglang/vllm, atom is the only framework with a programmatic cold-start seed grid
(`_atom_default_grid`: `atom_level_{2,3}`, `atom_prefix_cache`, `atom_kv_fp8` on
FP8, model-class-gated `atom_ep` / `atom_dp_attn` / `atom_mtp_{1,3}`,
`atom_cudagraph_bracket`) — sglang/vllm fail `error_class="empty_grid"` on a
cold start with no LLM variants.

### `--framework worldplay` specifics (HY-WorldPlay / HunyuanVideo-1.5)

Scriptable (server-less) autoregressive **video** diffusion; the metric is
steady-state **generated frames per second** (`throughput_unit="fps"`, higher =
better), surfaced as `output_throughput`. Requires the bypass benchmark backend:

```bash
export HYPERLOOM_BENCHMARK_BACKEND=bypass
export WORLDPLAY_REPO_PATH=/path/to/HY-WorldPlay        # code checkout
python3 -m hyperloom.inference_optimizer.cli -v optimize \
  --framework worldplay --gpu-type mi355x \
  --model /path/to/models/HunyuanVideo-1.5 \
  --tp 8 --max-hours 12
```

Shipped configs `baseline_worldplay.yaml` / `profile_worldplay.yaml`. The
entrypoint `worldplay_{runner_type}.sh`, `worldplay_bench_common.sh` and
`bench_fps.py` all ship in `assets/benchmark_scripts/` and are pinned by
absolute path at materialization, so `HYPERLOOM_BYPASS_SCRIPTS_DIR` is only an
operator override, not a requirement. The entrypoint supplies on-disk
`--model_path`/`--action_ckpt` because an HF hub cache may hold empty snapshot
stubs; `--action_ckpt` defaults to
`<model_parent>/HY-WorldPlay/<ar|bidirectional>_model/diffusion_pytorch_model.safetensors`.

`WORLDPLAY_REPO_PATH` is the HY-WorldPlay **code** checkout (the `hyvideo`
package), separate from the weights under `--model`. It registers the checkout as
a framework source root, which PolicyGate requires before any specialist or
framework-agent patch against `hyvideo/` can land — the probe cannot find a git
checkout on its own, only pip-installed packages. Materialization now publishes
the resolved path into the orchestrator's own environment, so a session that did
not export it still gets the source root (an operator-set value always wins).
Leave it unset and the entrypoint clones into `$HYPERLOOM_CACHE_DIR`.

### Naming the source tree without the framework prefix

`FRAMEWORK_REPO_PATH` is the framework-agnostic spelling of the same thing, and it
works for **any** framework rather than only the scriptable ones. Resolution order
is `<FRAMEWORK>_REPO_PATH` > `<FRAMEWORK>_DIR` > `FRAMEWORK_REPO_PATH`, so a
prefixed value keeps precedence and nothing existing changes behaviour.

Prefer the generic form. A session is single-framework by construction (the CLI
locks `$FRAMEWORK` for the run), so the prefix resolves no possible collision — it
only requires the operator to know the framework's name before setting the right
variable, and to change variable names when switching frameworks.

```bash
export FRAMEWORK_REPO_PATH=/path/to/checkout    # any framework
```

The prefixed and generic forms cover different situations by default rather than by
kind:

- **A pip-installed framework** (`sglang`, `vllm`, `atom`) needs neither. Its source
  root is discovered from the import machinery and the site-packages scan, which is
  why patching those has never required a path.
- **A scriptable framework** (`worldplay`, `worldmirror`, `xdit`) runs from a git
  checkout that neither mechanism can see, so one of the two variables is required.
- **An editable checkout of a normally-installed framework** — a `vllm` source tree
  you build yourself rather than the wheel — is equally invisible, and the generic
  variable is the supported way to point at it. Previously that case had to be
  handled by hand through `INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS`.

The FRAMEWORK phase works here (the registry carries the HY-WorldPlay
`repo_url`) and is the only phase that can restructure the pipeline itself —
sequence-parallel all-to-all, per-step host-to-device copies, repeated work
hoisted out of the AR rollout. `explore` cannot reach any of that: a variant is
only CLI flags plus env, and the accepted flag surface is four knobs. Pass
`--no-framework-agent` to skip it, but do not assume it is the default.

#### Framework-level source rewrites (the high-ceiling path)

Because worldplay is scriptable, the FRAMEWORK phase's authoring arm dispatches
`framework_rewrite_specialist` rather than `serving_specialist`. The two share no
optimization surface: an AR video rollout has no scheduler, no continuous
batching and no KV-cache admission policy, and its wins are the redundant work
its loop structure creates. See `references/framework_rewrite_patterns.md` for
the pattern catalogue the specialist works from.

Evidence. A `profile` leg arms a host-side probe
(`assets/host_probe/hl_host_probe.py`, injected via a `PYTHONPATH` prefix so the
customer's entrypoint is untouched) and writes
`framework_rewrite_evidence.json` next to the workspace. It measures what the GPU
kernel breakdown structurally cannot: object collectives round-tripping through
the host, device-to-host syncs, repeated host-to-device copies, and — with
`HYPERLOOM_FRAMEWORK_REWRITE_EVIDENCE_DEEP=1` — per-function call counts with
argument-repeat rates that separate a memoization candidate from a loop-hoist
enabler. Tier 1 is on by default and cheap;
`HYPERLOOM_FRAMEWORK_REWRITE_EVIDENCE=0` disables the probe entirely. The deep
tier inflates host time enough to skew a co-collected torch trace, so give it its
own leg.

Deliverable. Every rewrite must sit behind its own environment switch that
defaults OFF, declared in a `framework_switches` manifest with `category`,
`target`, `depends_on` and `enables`. That discipline buys three things:

- a **switch-off parity leg** runs with every switch unset and must reproduce the
  base within ±2%, so a patch that is not genuinely inert when disabled is
  reverted rather than silently poisoning every later measurement;
- a bundle that passes correctness but misses the throughput threshold is
  **kept inert** instead of reverted — default-off code costs nothing, and
  reverting would discard the rewrites that do pay along with the one that does
  not;
- accepted switches are registered as **search levers**, so `explore` measures
  each rewrite's own contribution (additive when the levers are dormant,
  leave-one-out when they are already on) and searches combinations, following
  the declared dependency closure so an enabler is never judged alone.

That last point is not a nicety. A hoist whose only value is making a downstream
cache hit measures flat on its own; a greedy accept/reject loop rejects it and
then measures every dependent rewrite against a permanently cold cache, losing
the bundle rather than the lever.

Locked / workload-spec (blacklisted in `_WORLDPLAY_ENV_BLACKLIST`, never explored):
precision is BF16 (`WORLDPLAY_USE_FP8_GEMMS`/`FP4_GEMMS`/`SAGEATTN` forced off),
and resolution / frame-count / step-count (`WORLDPLAY_HEIGHT`/`WIDTH`/
`NUM_FRAMES`/`NUM_STEPS`) + `WORLDPLAY_FEW_STEP` are part of the workload spec, not
tunables. The operating point is the customer's headline: 50 steps, 125 frames,
832×480, `model_type=ar`.

Correctness is a **self-calibrating** SSIM/MSE/LPIPS band (not a fixed
threshold): the baseline (establish) leg measures the pipeline's own drift under
an eps latent perturbation and stores the accept band in the reference `.npz`;
compare legs read it back. This matters because the best configs land near
SSIM≈0.79 — a fixed 0.85 threshold would false-fail them. Enabled via
`WORLDPLAY_QUALITY_CALIBRATE=1` in the baseline yaml. The scriptable quality-ref
choke point injects `XDIT_QUALITY_REF_WRITE` (establish) / `XDIT_QUALITY_REF`
(compare), which the customer body reads.

Seed grid `_worldplay_default_grid` (`worldplay_resident_ar`,
`worldplay_torch_compile`, `worldplay_group_offload_block`,
`worldplay_buffer_ops`, `worldplay_scratch_reclaim_off`). `TP` is the
sequence-parallel degree and must match the GPU count you compare against — the
customer's headline is 8-GPU `sp=8`, so fps from a different `TP` is **not
directly comparable**.

Measured on 8×MI355X at the headline operating point: baseline lands at
0.352 fps with run-to-run std 0.26–0.45%, and one generation takes ~345s, so a
leg costs `(1 warmup + WORLDPLAY_REPEATS) × 5.75 min` plus a ~9.5 min model load
(the probe puts the start of the hot loop at 569s), and the baseline leg adds
`WORLDPLAY_QUALITY_CALIB_SAMPLES` cheap 8-frame calibration generations. Both
sampling counts are 1: three repeats measured 0.348/0.349/0.349 fps, 0.3% apart
against a 1–2% keep threshold, and the calibration band takes the worst sample,
which on a real run was the first. A `roofline` leg costs far more than its
generation time — with the torch profiler on, exporting a 2.6 GB trace took 31.6
of its 55.3 minutes. Of the seed grid, `torch.compile` is a
reproducible **regression** (−13%: attention is `@torch.compiler.disable`, so
compile covers none of the hot path and still pays its overhead) and the
offloading / resident / ROCm-env knobs all measured inside the noise band. The
+51.8% win on that node came from the kernel path rewriting `attn_fwd`
(37.9% of GPU time) in Triton, not from `explore`.

**Knob surface for proposers (LLM explore + specialists) — read this before
proposing variants.** The customer's byte-identical `bench_fps.py` is a thin
wrapper that exposes ONLY a small CLI; it is **not** the model's full
HunyuanVideo CLI. Do NOT propose the model's native diffusion knobs
(`--use_cache teacache`/`fbcache`/`magcache`, `--attention_backend`,
`--enable_step_distill`, `--cfg_distilled`, `--enable_tiling`/`--enable_slicing`,
`--infer_steps`): the wrapper rejects them with an argparse error and the leg
dies as `no_measurement` (they are auto-dropped pre-dispatch by
`worldplay_server_args_reason`). Instead search these two productive surfaces:

- **Accepted workload tunables (CLI or `WORLDPLAY_*` env):**
  `--enable_torch_compile` (`WORLDPLAY_USE_TORCH_COMPILE=1`),
  `--group_offloading <block_level|leaf_level>` (`WORLDPLAY_GROUP_OFFLOADING`),
  `--offloading 0|1` (`WORLDPLAY_OFFLOADING`, baseline already 0),
  `--transformer_resident_ar_rollout` (`WORLDPLAY_TRANSFORMER_RESIDENT=1`).
- **Runtime / system env (does NOT touch the customer scripts — injected around
  them; this is where gains beyond the customer's own knobs live):** rocBLAS/
  hipBLASLt autotune (`PYTORCH_TUNABLEOP_ENABLED`, but NOT
  `PYTORCH_TUNABLEOP_TUNING` with torch.compile — GPU fault), allocator
  (`PYTORCH_HIP_ALLOC_CONF`/`PYTORCH_CUDA_ALLOC_CONF`), MIOpen find-mode,
  `GPU_MAX_HW_QUEUES`, attention-backend env toggles, and the seed-grid ROCm
  knobs above. Any numerics-altering env (fp8/fp4/sageattn) is blacklisted.

**Step count is LOCKED at 50.** Step-distillation / few-step (4-step etc.) hits
~24× fps but is a *different operating point*, not an optimization — it is out of
scope, cannot pass the self-calibrating gate (no quality-ref emitted → gate
skipped → not a valid KEEP), and must not be recommended in findings. The
customer explicitly moved away from the distilled path.

## GPU Runner Type

Pick the GPU explicitly with `--gpu-type` or `$GPU_TYPE`; without either, the
optimizer auto-detects via `rocm-smi --showproductname` (falling back to
`torch.cuda.get_device_properties(0).gcnArchName`).

```bash
python3 -m hyperloom.inference_optimizer.cli optimize --gpu-type mi355x --model "$MODEL_PATH" --max-hours 2
GPU_TYPE=mi300x python3 -m hyperloom.inference_optimizer.cli optimize --model "$MODEL_PATH" --max-hours 2
```

Accepted values: `mi300x`, `mi308x`, `mi325x`, `mi355x`. **`mi308x` and
`mi325x` map to `runner_type=mi300x`** with a warning, since the GPUs share the
same runner family and Magpie has not shipped MI308X/MI325X-specific SGLang/vLLM
scripts yet. If you need a true MI308X/MI325X-specific script, uncomment the `benchmark_script:` template in the
relevant YAML and point it at your script under `InferenceX/benchmarks/...`.

Do not set `HIP_VISIBLE_DEVICES` on the known ROCm stack unless the user asks;
it can make `torch.cuda.is_available()` return false. Use
`ROCR_VISIBLE_DEVICES` for GPU pinning.
