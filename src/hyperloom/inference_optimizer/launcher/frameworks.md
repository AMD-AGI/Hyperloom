# Framework Selection & GPU Runner Type

## Framework Selection

A session is single-framework. Pick `sglang` (default), `vllm`, or `atom` via
`--framework` or `$FRAMEWORK`:

```bash
inference_optimizer optimize --framework vllm --model "$MODEL_PATH" --max-hours 2
FRAMEWORK=vllm inference_optimizer optimize --model "$MODEL_PATH" --max-hours 2
inference_optimizer optimize --framework atom --model "$MODEL_PATH" --max-hours 2  # IR-8 single-node only
```

Resolution order: `--framework` > `$FRAMEWORK` > `sglang` (default).

What this controls:
- Which Magpie YAML the executors default to — `baseline_{sglang,vllm,atom}.yaml`
  and `profile_{sglang,vllm,atom}.yaml`. The per-framework resolver
  `_default_profile_config()` in `action_executors/profile.py` picks the right
  file from `$FRAMEWORK`.
- Which framework-specific seed grid the `explore` action falls back to when no
  `params.grid` is supplied. atom is the only framework with a programmatic seed
  today (`_default_grid_for_framework("atom", ...)` in
  `action_executors/explore.py`, populated by `_atom_default_grid()`); sglang
  and vllm continue to rely on the orchestration LLM emitting
  `provenance='default_grid'` variants and will fail with
  `error_class="empty_grid"` on a cold-start with no LLM input.
- Which extra-args env name `_grid_runner` writes (`EXTRA_VLLM_ARGS` /
  `EXTRA_SGLANG_ARGS` / `EXTRA_ATOM_ARGS`).
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
`https://github.com/ROCm/ATOM.git` is in `framework_agent.repo_map`. Unlike
sglang/vllm, atom is the only framework with a programmatic cold-start seed grid
(`_atom_default_grid`: `atom_level_{2,3}`, `atom_prefix_cache`, `atom_kv_fp8` on
FP8, model-class-gated `atom_ep` / `atom_dp_attn` / `atom_mtp_{1,3}`,
`atom_cudagraph_bracket`) — sglang/vllm fail `error_class="empty_grid"` on a
cold start with no LLM variants.

## GPU Runner Type

Pick the GPU explicitly with `--gpu-type` or `$GPU_TYPE`; without either, the
optimizer auto-detects via `rocm-smi --showproductname` (falling back to
`torch.cuda.get_device_properties(0).gcnArchName`).

```bash
inference_optimizer optimize --gpu-type mi355x --model "$MODEL_PATH" --max-hours 2
GPU_TYPE=mi300x inference_optimizer optimize --model "$MODEL_PATH" --max-hours 2
```

Accepted values: `mi300x`, `mi325x`, `mi355x`. **`mi325x` is mapped to
`mi300x`** with a warning, since the two GPUs share the same arch and Magpie has
not shipped `sglang_mi325x.sh` / `vllm_mi325x.sh` yet. If you need a true
MI325X-specific script, uncomment the `benchmark_script:` template in the
relevant YAML and point it at your script under `InferenceX/benchmarks/...`.

Do not set `HIP_VISIBLE_DEVICES` on the known ROCm stack unless the user asks;
it can make `torch.cuda.is_available()` return false. Use
`ROCR_VISIBLE_DEVICES` for GPU pinning.
