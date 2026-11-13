# Qwen3-14B-FP8 attention seam: one monokernel launch per layer and rank (TP4, gfx942)

## Objective

Lower the launch time of `qwen3_seam/seam.py` at TP4 for 1, 4, 8 and 16 rows (a decode step's
tokens). The score is the mean of the four cases' speed-ups over the pristine launch, timed by the
driver's `--bench-mode` under CUDA graph replay, the slowest rank of each replay counting.

## What the launch computes

Per rank, once per decoder layer, from the attention output to the input of the MLP:

| | tensor (per rank) |
|---|---|
| in | `attn` [S, 1280] bf16, this rank's 10 heads; `residual` [S, 5120] bf16, the same on every rank |
| out | `res_out` [S, 5120] bf16, the residual after attention; `x` [S, 5120] bf16, the post-attention RMSNorm |
| weights | o_proj [5120, 1280] e4m3fnuz with 128x128 block scales [40, 10] fp32; norm weight [5120] bf16 |

SGLang runs this in four launches on this container: `aiter_per1x128_quant`, aiter's
`gemm_a8w8_blockscale`, the custom all-reduce, and `aiter.rmsnorm2d_fwd_with_add`. The contract
follows the Triton GEMM SGLang runs with `SGLANG_AITER_GFX942_BLOCKSCALE_USE_CK=0` (its default
there is aiter's CK GEMM, whose bits differ on a few elements). The task's launch
runs all of it in one, which is only possible with in-kernel waits: the all-reduce crosses GPUs and
the norm needs a whole row, whose columns other CTAs computed.

## Contract

Every candidate is held to all of the following; a breach is refused before it is timed.

1. **A monokernel.**
   - One launch per call: the driver captures a call and requires exactly one graph node, a kernel.
   - The whole grid resident: the driver requires the HIP occupancy of the captured kernel times
     the CUs to cover its grid.
   - Every wait on another CTA or GPU is a guarded `Mailbox.poll` of `sgl_amd_mono.common`: no
     `while`, `s_sleep`, `_spin` or `spin_until` in `qwen3_seam/` (checked statically), and the
     acceptance suite stalls each wait and removes a rank, and requires every rank to stop.
   - A fresh tag per launch: parity over two back-to-back calls, the output of the last timed replay,
     and the replay tests (interleaved widths, back-to-back launches, the epoch's wrap).
2. **Numerics** (`benchmark/forge_mono/judge.py` against `reference.py`): `res_out` bit for bit;
   `x` within 1 bf16 step on at most 0.1% of its elements; the guard's error words clear. `res_out` is
   exact because the launch does each operation SGLang does, in the same order:
   - quantization per 1x128 as aiter does it: amax from `1e-10`, scale `amax * (float)(1. / 240)`,
     the inverse `1.0f / scale` correctly rounded, `x * inverse` clamped to +-240, then
     `v_cvt_pk_fp8_f32` (round to nearest even, e4m3fnuz);
   - the GEMM as aiter's Triton kernel compiles at its default gfx942 config (kpack 2): each 128-wide
     K block is four `mfma_f32_16x16x32_fp8_fp8` from zero, step `j` taking, for lane group `g`
     (lane / 16), K offsets `64 (j / 2) + 16 g + 8 (j % 2) + [0, 8)`; then
     `acc = fma(block * a_scale, w_scale, acc)`; the partial rounded to bf16;
   - the all-reduce: the four partials summed in rank order 0..3 in f32, rounded once to bf16;
   - the add: `s = f32(reduced) + f32(residual)` in f32, `res_out = bf16(s)`.

   `x` is `bf16((s * rstd) * w)` with `rstd = rsqrt(sum(s^2) / 5120 + 1e-6)` over the unrounded `s`;
   the sum of squares may be reduced in any order.
3. **Interface.** The runner imports these names from `seam.py`; their values may change, their
   names and signatures may not: `HIDDEN`, `MAX_TOKENS`, `BLOCKS`, `THREADS`, `EPOCH_WORDS`,
   `SeamBuild(tokens, tp, fault_site, fault_after)`, `build_seam(key)`, which returns
   `launch(attn, res_in, res_out, x_out, wo, wos, norm_w, scratch, sym, peers, rank, epoch, mode,
   stream)` (`mode` 0 returns at once: the runner compiles with it), `kernel_name(key)`,
   `peer_bytes(tp)`, `scratch_bytes(s)`.
4. **Runtime.** `sgl_amd_mono.common` from the installed sgl-amd-mono at `b10f3b7`, used as a library.
5. **Scope.** Only `qwen3_seam/` is editable: `seam.py` and modules added beside it.
   `benchmark/` (the driver, the runner, the reference, the judge and the acceptance suite) and
   `config.yaml` are protected.

## gfx942

- 304 CUs in 8 XCDs of 38, each XCD with its own 4 MB L2; 256 MB Infinity Cache; 64 KB LDS per CU.
- FP8 is e4m3fnuz: largest finite 240, NaN `0x80`, no infinities.
- MFMA: `mfma_f32_16x16x32_fp8_fp8`, `mfma_f32_32x32x16_fp8_fp8`, bf16 `mfma_f32_16x16x16bf16_1k`.
- Not on gfx942: `permlane16_swap` / `permlane32_swap` (so `ops.xred` and `ops.butterfly` at offsets
  16 and 32 do not compile; `ops.xshfl` does), MX-scaled MFMA, the `cvt_scalef32` family, 16x16x32
  bf16 / f16 MFMA (`ops.mfma_bf16`), `ds_read_b64_tr`, and 128-bit global-to-LDS loads
  (`global_load_lds_dwordx4`; gfx942's move one dword a lane).
- Cache policy bits: `CM_DEV` (SC1) is device scope, `CM_SYS` (SC0 | SC1) system scope.

## The pristine launch

256 CTAs of 512 threads. CTAs 0-159 each push and then reduce 32 columns; CTAs 240-255 each
normalize one row; CTAs 160-239 have no work. Known room:

- a push CTA runs the GEMM on 2 of its 8 waves, after the whole quantization;
- the weight loads are issued only after the quantization;
- each hand-off pair carries two bf16 and a 32-bit tag: 8 bytes per 2 values;
- CTAs 160-239 are idle.

## Tools

- `python3 benchmark/forge_mono/driver.py`: correctness at 1, 4, 8, 13 and 16 rows; `judge:` lines
  per call and an `allclose:` verdict.
- `python3 benchmark/forge_mono/driver.py --bench-mode`: `case_ms:` at 1, 4, 8 and 16 rows.
- `python3 benchmark/forge_mono/driver.py --profile-run`: three calls of the first case, for a profiler.
- `python3 -m pytest -q benchmark/acceptance/test_seam_compile.py`: every width compiles and its grid
  is resident.
  `config.yaml` lists the whole acceptance suite.

## Report

Name the stage you changed and why, the `case_ms` of every width, and the driver's judge lines.
Keep one change per KEEP, each explainable on its own.
