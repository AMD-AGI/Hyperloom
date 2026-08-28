---
name: debug-aiter
description: >
  Diagnose aiter library issues: JIT/build failures and stale-cache no-ops
  (AITER_REBUILD), ENABLE_CK exclusions, ABI/version mismatch, the tuned-CSV
  0-engagement trap (bias/key mismatch), picking the wrong CK/ASM/Triton variant,
  preshuffle/moe-align/quant constraints, and fp8 KV / causal-backward correctness
  traps. Use when an aiter call is wrong, won't build, or a "deployed" tune does
  nothing. Usage: /debug-aiter
allowed-tools: Read Bash Grep Glob
---

# Debug aiter

Diagnostic workflow for the aiter library (MI300X gfx942 / MI350 gfx950). aiter is a library +
dispatcher, so most "bugs" are build/dispatch/engagement issues, not kernel-source bugs — for the latter
see the language folder ([../../overall/authoring_delegation.md](../../overall/authoring_delegation.md)).

## Step 1: classify the symptom
| Symptom | Likely cause | Go to |
|---|---|---|
| A "deployed" tuned CSV changes nothing | DB key mismatch (usually `bias`) → 0 engagement | §2 |
| `TypeError: 'NoneType' object is not callable` | `@compile_ops` swallowed a compile error | §3 |
| Missing symbol / silent wrong results | aiter-amd ↔ ROCm/container version mismatch (ABI) | §3 |
| Edited CK/kernel source, no change | stale JIT cache | §3 |
| `ModuleNotFoundError`/`AttributeError` on a CK op | `ENABLE_CK=0` | §3 |
| Wrong results, fp8 / preshuffle / MoE | variant constraint violated (silent) | §4 |
| NaN backward, correct forward | causal + fused_backward/dropout (CK FA limit) | §4 |

## 2. The 0-engagement trap (the aiter #1 gotcha)
A tuned CSV only helps if the live call's **10-tuple** key matches
(`gfx, cu_num, padded_M, N, K, bias, dtype, otype, scaleAB, bpreshuffle`).
- **`bias` is the classic miss**: sglang dense GEMMs are `bias=False`; a CSV synthesized with `bias=True`
  → every lookup misses → silent no-op (looked deployed, did nothing).
- **`gfx`/`cu_num` mismatch also misses**: a CSV tuned on a different arch (gfx942 vs gfx950 vs gfx1250) or
  CU count won't hit. Legacy CSVs missing the `gfx` column are backfilled from `cu_num` only when merged
  via `AITER_CONFIGS` — regenerate on the target box rather than rely on it.
- **Always capture live** (`AITER_TUNE_GEMM=1`), never hand-author/guess the CSV.
- **Prove it**: `AITER_LOG_TUNED_CONFIG=1` then `grep -c 'is tuned on cu_num' server.log` must be > 0.
  `not found tuned config in … will use default config!` lines are misses.
- On sglang, also need `SGLANG_USE_AITER=1` to route to `tgemm.mm` at all. Full recipe:
  [../../overall/tuning_db.md](../../overall/tuning_db.md).

## 3. Build / JIT / version traps
- **Stale JIT cache masks source edits**: `AITER_REBUILD=1` (rebuild) / `2` (rebuild + delete .so), or
  clear `~/.aiter/jit/`.
- **JIT hang on first use**: import triggers a 30s+ compile — pre-warm the op before benching;
  `PREBUILD_KERNELS=1` at install; persistent `AITER_JIT_DIR`.
- **`ENABLE_CK=0` silently excludes** CK-backed ops → `AttributeError` at call. Default is 1.
- **ABI/version mismatch**: `aiter-amd` package must match the container ROCm → missing symbols / silent
  wrong results. `pip show aiter-amd`; rebuild from source if mismatched.
- **`@compile_ops` hides compile errors** → decorated fn returns None → `NoneType not callable`. Debug
  with `AITER_LOG_LEVEL=DEBUG AITER_LOG_MORE=1`; read `~/.aiter/jit/<mod>/build.log`.
Details: [../../overall/jit_and_build.md](../../overall/jit_and_build.md).

## 4. Variant / correctness constraints (silent wrong results)
- **Multiple backends per op** (`gemm_a8w8_ck` vs `gemm_a8w8_asm`; `pa_fwd_asm` vs `paged_attention_v1`) —
  different optimal shapes AND KV formats. Compare the same variant consistently.
- **Preshuffle** (`bpreshuffle=True`) requires `N%16==0`, `K%32==0` — else silent wrong results.
- **MoE**: `moe_align_block_size()` is mandatory before any fused MoE op; MXFP4 `block_size_M=32` is
  hardcoded; quant algo names are strings (`"fp8smoothquant"`…), not the enum.
- **fp8 KV cache** needs `head_size%16==0` and `high_precision=2`; fp8 FA forbids `dropout_p>0` /
  `return_softmax_lse`.
- **Causal + fused_backward / dropout → NaN grads** (CK FA limitation).
- **32-bit stride overflow** on 128+ heads (LLaMA-3-405B) → `AITER_INT64_STRIDES=1`.
- **fp8 tolerance is wide** (atol=0.3): a "passing" fp8 test ≠ parity — verify cosine ≥ 0.96.
This list is not exhaustive and is deliberately not maintained per-operator — the authoritative list is
the `assert`s and shape guards in the aiter source plus `op_tests/test_<op>.py`. Read those for the
operator you are on.

## 5. Confirm which kernel actually ran
`AITER_LOG_TUNED_CONFIG=1` (config selection) + rocprofv3 kernel-trace → confirm the intended
`*ck_*` / asm / triton kernel ran, not a fallback (a missing gfx942 shape falls back to generic Triton,
several× slower).
