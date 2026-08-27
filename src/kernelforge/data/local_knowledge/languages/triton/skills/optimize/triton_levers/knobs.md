---
title: Triton on AMD — the full knob set & autotune config space
kind: language
gens: [gfx942, gfx950]
dtypes: [bf16, fp16, fp8_e4m3_fnuz, int8]
regimes: [both]
status: competitive
updated: 2026-07-08
sources:
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://github.com/pytorch/pytorch/pull/143286
---

# Triton on AMD — knobs & autotune

All knob names verified against `third_party/amd/backend/compiler.py` `HIPOptions`. **AMD-only knobs
take effect only inside the `triton.Config({...})` kwargs dict** (they map to `HIPOptions` fields).
Setting them as Python variables does nothing.

## 0. The landscape
| Knob | Group | Range | Default | Matters most |
|---|---|---|---|---|
| `BLOCK_M/N/K` | constexpr | pow2 16…256 | — | every GEMM/attn — primary lever |
| `GROUP_SIZE_M` | constexpr | 1,4,**8**,16 | — | GEMM L2 reuse (×XCD=8) |
| `SPLIT_K` | constexpr | 1,2,4,8,16 | 1 | skinny/decode GEMM |
| `num_warps` | standard | 1,2,**4**,8 | 4 | occupancy vs VGPR spill (wave64!) |
| `num_stages` | standard | **1**,2,(3) | 2 | stream-pipeliner depth |
| `matrix_instr_nonkdim` | **AMD** | 0,**16**,32 | 0 (auto) | MFMA tile size |
| `kpack` | **AMD** | 1,**2** | 1 | LDS read width (gfx942 only) |
| `waves_per_eu` | **AMD** | 0–8 | 0 | force occupancy by trimming VGPRs |
| `schedule_hint` | **AMD** | none/attention/memory-bound-attention | none | attention sched |
| `OPTIMIZE_EPILOGUE` | env | 0/1 | 0 | drop epilogue convert (GEMM → 1) |
| `maxnreg` | standard | int | None | hard VGPR cap (rarely needed) |

## 1. `matrix_instr_nonkdim` — MFMA size
`16` → `v_mfma_f32_16x16x16` (**recommended on MI300X**); `32` → `v_mfma_f32_32x32x8` (bigger acc →
more AGPR/VGPR pressure, coarser scheduling). `nonkdim=32` requires `BLOCK_M,BLOCK_N` divisible by 32.
Prefer 16 unless 32 measurably wins.

## 2. `waves_per_eu` — occupancy via register trimming
Emits `amdgpu-waves-per-eu`. Hardware: **512 VGPR/EU**, allocated in **16-granules**. Achievable iff
`round_up_16(vgpr_used) · waves_per_eu ≤ 512`.

| vgpr_used | rounds to | max waves/EU |
|---|---|---|
| ≤64 | 64 | 8 |
| 128 | 128 | 4 |
| 170 | 176 | 2 (176×3 = 528 > 512) |
| 256 | 256 | 2 |

Use when you're **just over an occupancy boundary** (VGPR=176 → set `waves_per_eu=3`; LLVM may shave
under 170 to fit 3 waves). Too aggressive → spills (counterproductive). Typical tuned: **2–3** GEMM,
**3–4** memory-bound. Verify: `AMDGCN_ENABLE_DUMP=1 | grep .vgpr_count`; `occ.sh` (ROCm/triton).

## 3. `kpack` — K-packing for LDS reads
`2` packs 2 K-slices → emits 128-bit `ds_read_b128` (vs two `b64`), halving LDS instruction count.
Win for fp16/bf16 GEMM with `BLOCK_K≥64` on **gfx942**. Costs VGPRs (holds 2 slices) → tune with
`waves_per_eu`. **Deprecated/forced to 1 on gfx950** (backend warns).

## 4. `num_warps` — wave64 & spill avoidance
Warp = 64 lanes; `num_warps=N` → `N·64` threads. **The #1 AMD perf bug** is carrying `num_warps=8`
from NVIDIA: 8 warps → 2 waves share a SIMD → ~256 VGPR each → spill to scratch (HBM) → **3–5×
slower**. Start GEMM at `4`; go to `8` only if VGPR-light and occupancy-bound. Memory-bound: `2`/`4`.

## 5. `num_stages` — stream-pipeliner depth
Single GEMM **2**; fused two-GEMM (FA) **1**; no-GEMM **1**. Higher stages buffer more in-flight loads
in LDS → crush occupancy on 64 KB LDS. `>1` enables block ping-pong (`knobs.amd.use_block_pingpong`).

## 6. `GROUP_SIZE_M` / `SPLIT_K` — grid shaping
- `GROUP_SIZE_M` reorders block scheduling for L2 reuse; use multiples of **XCD=8**. `8` is a strong
  default; bigger → more reuse, worse balance for small grids.
- `SPLIT_K` splits the K reduction (atomic accumulate) for skinny/decode shapes to reach ≥1024
  programs. Costs a C zero-init + atomics. Not needed when M·N already gives ≥1024 tiles.

## 7. `schedule_hint` (instruction scheduling)
HIPOptions field (default `none`): `attention` / `memory-bound-attention` tune the sched pipeline for
FA-style chained dots (built on LLVM `sched_group_barrier`/IGLP). **Experimental** — older
ROCm/triton forks used `instruction_sched_variant` (`default`/`iglp0`/`iglp1`). Always
`grep schedule_hint third_party/amd/backend/compiler.py`. Leave `none` unless tuning attention; gain
for GEMM is small. Raw control: `llvm_fn_attrs="amdgpu-sched-strategy=iterative-ilp"`.

## 8. Env / `knobs.amd.*` (process-wide)
| Variable | Effect | Recommendation |
|---|---|---|
| `OPTIMIZE_EPILOGUE=1` | drop epilogue convert_layout | **ON for GEMM** |
| `TRITON_PRINT_AUTOTUNING=1` | print winner + timing | ON while tuning |
| `AMDGCN_ENABLE_DUMP=1` / `knobs.amd.dump_amdgcn` | dump ISA | check `_dwordx4`, `ds_*_b128` |
| `MLIR_ENABLE_DUMP=1` | dump TTGIR / TritonAMDGPU IR | check MFMA layout, LDS bytes |
| `knobs.amd.use_buffer_ops` | `buffer_load/store` (bounds-checked) | **ON for masked loads** (not default!) |
| `knobs.amd.use_async_copy` | `global_load_lds` async copy | gfx950 default; experimental gfx942 |
| `knobs.amd.use_block_pingpong` | ping-pong two warp groups (needs stages>1) | try for GEMM |
| `TRITON_ALWAYS_COMPILE=1` | bypass kernel cache | force re-tune |

`supported_fp8_dtypes` (AMD) = `("fp8e4nv","fp8e5","fp8e5b16","fp8e4b8")`; the **fnuz** MFMA types are
`fp8e4b8`/`fp8e5b16`.

## 9. Complete MI300X GEMM config space + LDS prune
```python
def _space():
    s = []
    for (BM,BN) in [(128,128),(128,256),(256,128),(256,256),(128,64),(64,128)]:
        for BK in (32,64,128):
            for nkd in (16,32):
                if nkd==32 and (BM%32 or BN%32): continue
                for kp in (1,2):
                    for nw in (4,8):
                        for we in (0,2,3):
                            s.append(triton.Config(
                                {"BLOCK_M":BM,"BLOCK_N":BN,"BLOCK_K":BK,"GROUP_SIZE_M":8,"SPLIT_K":1,
                                 "matrix_instr_nonkdim":nkd,"kpack":kp,"waves_per_eu":we},
                                num_warps=nw, num_stages=2))
    return s

def _prune(configs, named_args, **kw):
    M,N,K = named_args["M"],named_args["N"],named_args["K"]; out=[]
    for c in configs:
        k=c.kwargs
        lds=(k["BLOCK_M"]*k["BLOCK_K"]+k["BLOCK_K"]*k["BLOCK_N"])*2*c.num_stages
        if lds > 64*1024: continue                       # 64 KB CDNA3 (160 KB on gfx950)
        if k["BLOCK_M"]>2*M or k["BLOCK_N"]>2*N: continue
        out.append(c)
    return out or configs[:1]

@triton.autotune(_space(), key=["M","N","K"],
                 prune_configs_by={"early_config_prune":_prune}, warmup=25, rep=100)
@triton.jit
def gemm(...): ...    # body = patterns.md §1
```
`TRITON_PRINT_AUTOTUNING=1` prints e.g.: `BLOCK_M:128, BLOCK_N:256, BLOCK_K:64, GROUP_SIZE_M:8,
matrix_instr_nonkdim:16, kpack:2, waves_per_eu:2, num_warps:4, num_stages:2`.

## 10. Baking the winner (drop autotune from the hot path)
- **A:** single hard-coded `triton.Config` under `@triton.autotune([WINNER], key=...)`.
- **B (what vLLM/SGLang ship):** per-shape JSON dispatch table (e.g. `E=…,N=…,device_name=MI300X.json`
  for fused MoE), generated by a `tuning_*.py` sweep, loaded at startup — no runtime autotune.
- **C:** `triton.compile` AOT for the exact specialization (ships an HSACO).

> Tuned tables are **ROCm/Triton-build-specific** — note the build; never ship a hand-copied table as
> portable (sourcing rule #2).

## 11. TorchInductor / max-autotune
Inductor emits Triton for `mm`/`addmm`/attention; `max-autotune` searches a template space. The AMD
GEMM knobs (`waves_per_eu`, `kpack`, `matrix_instr_nonkdim`) were wired into the Inductor ROCm GEMM
template in pytorch/pytorch #143286 — set via `torch._inductor.config` / `max_autotune_gemm_backends`.
Inductor is the practical path to "Triton GEMM without hand-writing one."

## Sources
- `HIPOptions` (all AMD knobs, supported_fp8_dtypes, knobs.amd.*): https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- matrix_instr_nonkdim / waves_per_eu / num_stages / split-K / ≥1024 grid (MI300X workload opt): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- ROCm GEMM tuning params in TorchInductor (waves_per_eu, kpack, matrix_instr_nonkdim): https://github.com/pytorch/pytorch/pull/143286
- matmul perf vs matrix_instr_nonkdim & kpack on MI300X: https://github.com/triton-lang/triton/issues/4959
- per-shape tuned configs / num_warps spill: https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/
