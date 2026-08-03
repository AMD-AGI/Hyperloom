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
better), surfaced as `output_throughput`. Requires the bypass benchmark backend
and the vendored customer bench-kit:

```bash
export HYPERLOOM_BENCHMARK_BACKEND=bypass
export HYPERLOOM_BYPASS_SCRIPTS_DIR=/primus/xiaofei/HY-WorldPlay/hyperloom_bench
python3 -m hyperloom.inference_optimizer.cli -v optimize \
  --framework worldplay --gpu-type mi355x \
  --model /primus/xiaofei/HY-WorldPlay/models/HunyuanVideo-1.5 \
  --max-hours 12 --no-framework-agent
```

Shipped configs `baseline_worldplay.yaml` / `profile_worldplay.yaml`. The
Hyperloom entrypoint `worldplay_{runner_type}.sh` is the ONLY Hyperloom-authored
file in `HYPERLOOM_BYPASS_SCRIPTS_DIR`; it wires the customer's byte-identical
`worldplay_bench_common.sh` + `bench_fps.py` and hands off. It supplies on-disk
`--model_path`/`--action_ckpt` because this node's HF hub cache holds empty
snapshot stubs.

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
`worldplay_buffer_ops`, `worldplay_scratch_reclaim_off`). Note this node has a
single MI355X → `TP=1`; the customer's headline is 8-GPU `sp=8`, so single-GPU
fps is **not directly comparable**.

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
