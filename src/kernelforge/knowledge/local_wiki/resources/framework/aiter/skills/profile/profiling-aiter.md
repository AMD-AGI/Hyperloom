---
name: profiling-aiter
description: >
  Profile aiter on a real workload: prove kernel engagement (AITER_LOG_TUNED_CONFIG,
  rocprofv3 kernel names) before trusting any delta, bench in-context not isolated,
  classify the dominant GPU-time op for Amdahl targeting, and decide DB-tune vs
  author-a-kernel from the profile. Use when deciding what aiter change is worth
  making from measured evidence. Usage: /profiling-aiter
allowed-tools: Read Bash Grep Glob
---

# Profiling aiter

aiter optimization is **e2e / in-context**, not isolated microbench — the lever (DB tune or rebind) only
matters if it engages the live path and moves the Amdahl-dominant op. Hardware peaks live in
`local_knowledge/hardware/`.

## 1. Find the Amdahl-dominant op first
```bash
rocprofv3 --kernel-trace --stats -f csv -- <run the real workload>
```
Rank kernels by total GPU time. On dense LLMs the mass is usually the GEMM family (e.g. ~79% on
Qwen3.5-27B) → that's where DB tuning pays; a 2× win on a 1%-of-time op is noise. Optimize the top rows.

## 2. Prove engagement BEFORE believing any delta
The aiter failure mode is a change that looks deployed but never runs:
- `AITER_LOG_TUNED_CONFIG=1` → `grep -c 'is tuned on cu_num'` must be **> 0** (misses log
  `not found tuned config in … will use default config!`).
- rocprofv3 kernel names: confirm the intended `*ck_*` / asm / triton kernel ran, **not a fallback**
  (missing gfx942 shape → generic Triton, several× slower).
- On sglang confirm `SGLANG_USE_AITER=1`; on vLLM confirm the `VLLM_ROCM_USE_AITER*` gate + that the op
  stayed an opaque custom op through `torch.compile`.

## 3. Bench in-context, not isolated
Isolated aiter benchmarks mislead — they miss allocation patterns, cache effects, and occupancy
interactions with neighbouring kernels. An authored kernel measured 0.99–1.47× isolated still **lost e2e**
to the aiter env path. Gate with a same-session A/B: accept iff `delta > 0.5% AND cand_min > ref_max AND
parity holds`.

## 4. PMC → decision (tune vs author)
| profile signal | reading | action |
|---|---|---|
| dominant GEMM, no tuned rows hit | un-tuned dispatch | **DB tune** ([../../overall/tuning_db.md](../../overall/tuning_db.md)) |
| tuned + engaged, still below roofline | library ceiling for this shape | consider authoring in CK/HIP/Triton/FlyDSL ([../../overall/authoring_delegation.md](../../overall/authoring_delegation.md)) |
| Triton fallback in trace | coverage gap | tune/generate the shape, or `AITER_ONLINE_TUNE=1` |
| memory-bound decode (M=1..8) | skinny regime | ensure `skinny`/`wvSpltK` variant selected |

## 5. Parity gate (aiter swaps math variants)
DB tuning is same-math (gradlib gates `err_ratio<0.05`, dominant rows `0.0`) → parity-safe. But fp8/fp4
scaled variants and MLA can regress accuracy — add a downstream task-accuracy gate (greedy temp=0 parity,
small eval) when enabling those.

## Sources
- rocprofv3 / rocprof-compute: https://rocm.docs.amd.com/projects/omniperf/en/amd-staging/what-is-rocprof-compute.html
- MI300X workload optimization (roofline, Amdahl): https://rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html
