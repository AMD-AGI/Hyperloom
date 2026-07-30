# Framework Selection & GPU Runner Type

## Framework Selection

A session is single-framework. Pick `sglang` (default), `vllm`, `atom`,
`xdit`, or `worldplay` via `--framework` or `$FRAMEWORK`:

```bash
python3 -m hyperloom.inference_optimizer.cli optimize --framework vllm --model "$MODEL_PATH" --max-hours 2
FRAMEWORK=vllm python3 -m hyperloom.inference_optimizer.cli optimize --model "$MODEL_PATH" --max-hours 2
python3 -m hyperloom.inference_optimizer.cli optimize --framework atom --model "$MODEL_PATH" --max-hours 2  # IR-8 single-node only
python3 -m hyperloom.inference_optimizer.cli optimize --framework xdit --model "$MODEL_PATH" --max-hours 2  # scriptable diffusion
python3 -m hyperloom.inference_optimizer.cli optimize --framework worldplay --model "$MODEL_PATH" --max-hours 2  # scriptable video (HY-WorldPlay)
```

Resolution order: `--framework` > `$FRAMEWORK` > `sglang` (default).

What this controls:
- Which Magpie YAML the executors default to — `baseline_{sglang,vllm,atom,xdit,worldplay}.yaml`
  and `profile_{sglang,vllm,atom,xdit,worldplay}.yaml`. The per-framework resolver
  `_default_profile_config()` in `src/hyperloom/orchestrator/actions/executors/profile.py` picks the right
  file from `$FRAMEWORK`.
- Which framework-specific seed grid the `explore` action falls back to when no
  `params.grid` is supplied. atom, xdit and worldplay ship programmatic seeds
  (`_default_grid_for_framework(...)` in
  `src/hyperloom/orchestrator/actions/executors/explore.py`, populated by `_atom_default_grid()` /
  `_xdit_default_grid()` / `_worldplay_default_grid()`); sglang
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

### `--framework worldplay` specifics

HY-WorldPlay is an 8B HunyuanVideo-1.5 interactive world model. SCRIPTABLE and
single-node; the unit of work is a whole **video** (`video/s`), produced by an
autoregressive rollout over latent chunks rather than a single denoise pass.

Run it on the **bypass** benchmark backend — bypass routes scriptable
frameworks off `framework_registry.is_scriptable()` alone
(`bypass_runner._run_scriptable_benchmark`) and never imports Magpie, whose
`BenchmarkFramework` enum does not know this framework:

```bash
export HYPERLOOM_BENCHMARK_BACKEND=bypass
export HYPERLOOM_BYPASS_SCRIPTS_DIR=/primus/xiaofei/HY-WorldPlay/magpie_scripts
python3 -m hyperloom.inference_optimizer.cli optimize \
    --framework worldplay --gpu-type mi355x \
    --model /primus/xiaofei/HY-WorldPlay/models/HunyuanVideo-1.5 --max-hours 4
```

`--gpu-type` matters: bypass defaults `runner_type` to `mi300x` and would look
for `worldplay_mi300x.sh`.

**Dual gate.** A variant only wins when it is faster *and* not degraded:

- performance — median of the timed full-length generations, carried in
  `mean_e2el_ms` (`e2el_stat: median_of_timed_runs`). The median, not the mean:
  a single slow run otherwise dominates a small sample.
- quality — every output frame is scored (SSIM/MSE, LPIPS on a subsample)
  against a lossless BF16 reference frame stack, and the gate keys on both the
  mean and the **worst** frame. AR drift accumulates towards the end of the
  rollout, so a mean-only gate lets a variant that wrecks the last chunk pass.
  `is_valid_measurement()` drops any scriptable variant whose gate failed
  regardless of throughput, and the wrapper additionally exits non-zero.

The reference is a `.npz` frame stack, not an mp4 — the scriptable choke point
in `_workload_envs` hands the wrapper a `.png` path and the wrapper swaps the
extension, so codec noise can never be read as a regression.

**Locked knobs.** SageAttention and FP8 GEMMs are CUDA-only, and the distilled
4-step schedule is pinned: dropping steps is a different model, not a speedup.
All three are hard-locked in the wrapper and rejected by
`worldplay_blacklist_reason` in `_grid_variant_filter.py`.

**Workload shape.** The AR model requires `[(frames-1)//4+1] % 4 == 0` and the
pose must supply exactly that many latents — 61 frames pairs with `w-15`, 125
with `w-31`. The wrapper preflights both before the ~51s model load.

**Per-chunk timings are diagnostic only.** `chunk_latencies_ms` falls
monotonically across a rollout because the KV cache grows with it, so there is
no steady-state window *inside* one generation to average; windowing happens at
the run level, where a full-length generation is the comparable unit. The
`stage_breakdown_ms` block is where the optimization targets show up — on the
61-frame MI355X baseline the AR rollout is ~55% of the wall clock and the 3D VAE
decode is ~25%, so the denoise steps are not the whole story.

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
