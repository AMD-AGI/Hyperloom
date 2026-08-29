---
title: Triton on AMD — the knob space and autotune
kind: language
lever: triton_knob_space
gens: [gfx950]
updated: 2026-08-28
sources:
  - https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
  - https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
  - https://github.com/pytorch/pytorch/pull/143286
  - https://github.com/triton-lang/triton/issues/4959
---

# The knob space

> **AMD-only knobs take effect only inside the `triton.Config({...})` kwargs dict** — they map to
> `HIPOptions` fields. Setting them as Python variables does nothing, silently. This is the single most
> common "I tuned it and nothing changed" cause.

All names below verified against `third_party/amd/backend/compiler.py`. **Re-grep on your build** —
they drift between upstream and `ROCm/triton`.

## The landscape

| Knob | Group | Range | Default | Matters most for |
|---|---|---|---|---|
| `BLOCK_M/N/K` | constexpr | pow2 16…256 | — | every GEMM/attention — **the primary lever** |
| `GROUP_SIZE_M` | constexpr | 1, 4, **8**, 16 | — | GEMM L2 reuse (× XCD = 8) |
| `SPLIT_K` | constexpr | 1, 2, 4, 8, 16 | 1 | skinny / decode GEMM |
| `num_warps` | standard | 1, 2, **4**, 8 | 4 | occupancy vs VGPR spill (**wave64**) |
| `num_stages` | standard | **1**, 2, (3) | 2 | stream-pipeliner depth |
| `matrix_instr_nonkdim` | **AMD** | 0, **16**, 32 | 0 (auto) | MFMA tile size |
| `waves_per_eu` | **AMD** | 0–8 | 0 | force occupancy by trimming VGPRs |
| `kpack` | **AMD** | 1, 2 | 1 | LDS read width — **gfx942 only** |
| `schedule_hint` | **AMD** | none / attention / memory-bound-attention | none | attention scheduling |
| `OPTIMIZE_EPILOGUE` | env | 0/1 | 0 | **set 1 for GEMM** |
| `maxnreg` | standard | int | None | hard VGPR cap (rarely needed) |

## `matrix_instr_nonkdim` — MFMA size

`16` → the 16×16 MFMA family (**recommended**). `32` → 32×32, which needs `BLOCK_M`, `BLOCK_N`
divisible by 32.

**Prefer 16 for two independent reasons**: 16×16 carries **4 C-registers/lane** vs 32×32's **16** (that
4× comes out of the 512-register budget), *and* the 32×32 op draws more power so the part clocks lower.
Only switch if 32 measurably wins on your shape.

## `waves_per_eu` — occupancy via register trimming

Emits `amdgpu-waves-per-eu`. Hardware: **512 registers/SIMD**, allocated in **16-granules**. Achievable
iff `round_up_16(vgpr_used) × waves_per_eu ≤ 512`.

| vgpr_used | rounds to | max waves/SIMD |
|---:|---:|---:|
| ≤ 64 | 64 | 8 |
| 128 | 128 | 4 |
| 170 | **176** | 2 (176×3 = 528 > 512) |
| 256 | 256 | 2 |

**Use it when you are just over a boundary.** VGPR = 176 → set `waves_per_eu=3` and LLVM may shave
under 170 to fit three waves. Push too far and you get spills, which cost more than the occupancy buys.
Typical tuned values: **2–3** for GEMM, **3–4** for memory-bound.

Verify with `AMDGCN_ENABLE_DUMP=1 | grep .vgpr_count`, or `occ.sh` from `ROCm/triton`.

## `num_warps` — wave64 and spill avoidance

A warp is **64 lanes**; `num_warps=N` → N·64 threads.

**The #1 AMD perf bug is carrying `num_warps=8` from NVIDIA.** Eight warps → two waves share a SIMD →
~256 VGPR each → spill to scratch (HBM) → **3–5× slower**.

Start GEMM at **4**. Go to 8 only if the kernel is VGPR-light *and* occupancy-bound. Memory-bound: 2 or 4.

## `num_stages` — stream-pipeliner depth

| Pattern | `num_stages` |
|---|---|
| single GEMM | **2** |
| fused two-GEMM (Flash-Attention) | **1** |
| no GEMM (elementwise, reduction) | **1** |

Higher stages buffer more in-flight loads in LDS. gfx950's **160 KiB** LDS makes a third stage more
affordable than it was on a 64 KiB part — worth testing, but it is not free.

`num_stages > 1` is the prerequisite for **block ping-pong**
(`knobs.amd.use_block_pingpong`), where two warp groups alternate so one issues MFMA while the other
issues VMEM/DS.

> **A flat `num_stages` sweep is a diagnostic, not a result.** It means the loop is not being pipelined
> at all — see `triton_traps.md` and `common_methodology/optimization/lever_loop_form.md`.

## `kpack` — gfx942 only

`kpack=2` packs 2 K-slices → emits 128-bit `ds_read_b128` instead of two `b64`, halving LDS instruction
count. A near-universal win for fp16/bf16 GEMM with `BLOCK_K ≥ 64` **on gfx942**.

**Deprecated and forced to 1 on gfx950** (the backend warns). On gfx950 you should see `ds_read_b128`
without it. Do not carry `kpack=2` into a gfx950 config space.

## `GROUP_SIZE_M` / `SPLIT_K` — grid shaping

- **`GROUP_SIZE_M`** reorders block scheduling for L2 reuse. Use multiples of **8** (the XCD count);
  `8` is a strong default. Bigger gives more reuse but worse balance on small grids.
- **`SPLIT_K`** splits the K reduction (atomic accumulate) for skinny/decode shapes so the grid reaches
  **≥1024 programs** across 256 CUs. Costs a C zero-init plus atomics. Skip it when M·N already yields
  ≥1024 tiles.

## `schedule_hint`

`HIPOptions` field, default `none`. `attention` / `memory-bound-attention` tune the scheduling pipeline
for FA-style chained dots (built on LLVM `sched_group_barrier` / IGLP).

**Experimental** — older `ROCm/triton` forks used `instruction_sched_variant`
(`default`/`iglp0`/`iglp1`). Always grep for it. Leave at `none` unless tuning attention; the GEMM gain
is small. Raw control: `llvm_fn_attrs="amdgpu-sched-strategy=iterative-ilp"`.

## Env and `knobs.amd.*`

| Variable | Effect | Recommendation |
|---|---|---|
| `OPTIMIZE_EPILOGUE=1` | drops the epilogue `convert_layout` | **ON for GEMM** |
| `TRITON_PRINT_AUTOTUNING=1` | prints the winner + timing | ON while tuning |
| `AMDGCN_ENABLE_DUMP=1` | dump ISA | check `_dwordx4`, `ds_*_b128` |
| `MLIR_ENABLE_DUMP=1` | dump TTGIR / TritonAMDGPU IR | check MFMA layout, LDS bytes |
| `knobs.amd.use_buffer_ops` | `buffer_load/store` (HW bounds-checked) | **ON for masked loads — not default!** |
| `knobs.amd.use_async_copy` | `global_load_lds` direct-to-LDS | **default on gfx950**; experimental gfx942 |
| `knobs.amd.use_block_pingpong` | ping-pong two warp groups (needs stages > 1) | try for GEMM |
| `TRITON_ALWAYS_COMPILE=1` | bypass the kernel cache | force a re-tune |

## A config space with an LDS prune

```python
def _space():
    s = []
    for (BM, BN) in [(128,128),(128,256),(256,128),(256,256),(128,64),(64,128)]:
        for BK in (32, 64, 128):
            for nkd in (16, 32):
                if nkd == 32 and (BM % 32 or BN % 32): continue
                for nw in (4, 8):
                    for we in (0, 2, 3):
                        s.append(triton.Config(
                            {"BLOCK_M":BM, "BLOCK_N":BN, "BLOCK_K":BK,
                             "GROUP_SIZE_M":8, "SPLIT_K":1,
                             "matrix_instr_nonkdim":nkd, "waves_per_eu":we},
                            num_warps=nw, num_stages=2))
    return s

def _prune(configs, named_args, **kw):
    M, N, K = named_args["M"], named_args["N"], named_args["K"]
    out = []
    for c in configs:
        k = c.kwargs
        lds = (k["BLOCK_M"]*k["BLOCK_K"] + k["BLOCK_K"]*k["BLOCK_N"]) * 2 * c.num_stages
        if lds > 160*1024: continue                      # gfx950: 160 KiB (was 64 KiB on gfx942)
        if k["BLOCK_M"] > 2*M or k["BLOCK_N"] > 2*N: continue
        out.append(c)
    return out or configs[:1]

@triton.autotune(_space(), key=["M","N","K"],
                 prune_configs_by={"early_config_prune": _prune}, warmup=25, rep=100)
@triton.jit
def gemm(...): ...      # body from triton_templates.md
```

Note the LDS bound is **160 KiB** here. A space pruned against 64 KiB throws away configs that are
legal on gfx950.

## Baking the winner

Autotune in a serving hot path adds first-call latency and is non-deterministic. Three ways to remove it:

| | Approach |
|---|---|
| **A** | a single hard-coded `triton.Config` under `@triton.autotune([WINNER], key=...)` |
| **B** | **what vLLM/SGLang ship** — a per-shape JSON dispatch table (e.g. `E=…,N=…,device_name=….json`), generated by a `tuning_*.py` sweep and loaded at startup |
| **C** | `triton.compile` AOT for the exact specialization (ships an HSACO) |

> A tuned table is **ROCm/Triton-build-specific.** Record the build. Never ship a hand-copied table as
> portable.

## TorchInductor

Inductor emits Triton for `mm`/`addmm`/attention; `max-autotune` searches a template space. The AMD
GEMM knobs (`waves_per_eu`, `kpack`, `matrix_instr_nonkdim`) were wired into the Inductor ROCm GEMM
template in pytorch/pytorch #143286, settable via `torch._inductor.config` /
`max_autotune_gemm_backends`. **This is the practical path to "a Triton GEMM without hand-writing
one."**

## Sources
- `HIPOptions` (all AMD knobs, `supported_fp8_dtypes`, `knobs.amd.*`): https://github.com/triton-lang/triton/blob/main/third_party/amd/backend/compiler.py
- MI300X workload optimization (`matrix_instr_nonkdim`, `waves_per_eu`, split-K, grid sizing): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
- ROCm GEMM tuning params in TorchInductor: https://github.com/pytorch/pytorch/pull/143286
- matmul perf vs `matrix_instr_nonkdim` and `kpack`: https://github.com/triton-lang/triton/issues/4959
- Per-shape tuned configs / `num_warps` spill: https://pytorch.org/blog/enabling-vllm-v1-on-amd-gpus-with-triton/
