# Rewrite SGLang MXFP8 grouped GEMM to FlyDSL

## Operator contract

Port `_mxfp8_grouped_gemm_kernel` and `_grouped_gemm_mxfp8` from
`mxfp8_grouped_gemm.py` to a standalone FlyDSL implementation in `kernel.py`.
The workload is the pair of grouped GEMMs in one MiniMax-M3 MoE forward:

- GEMM1 gathers one shared activation row per route with
  `a_row = sorted_token_id // top_k` and writes BF16 gate/up results.
- GEMM2 reads one activation row per route, applies its top-k weight in the
  epilogue, and writes FP32 down-projection results.

Operands are MXFP8 E4M3 with uint8 E8M0 scales per contiguous 1x32 K block.
Accumulate in FP32. Routing metadata follows SGLang `moe_align_block_size`.

## Required interface

Implement the factory and launch signatures documented at the top of
`driver.py`. The launch must write the provided output tensor in place and use
the supplied FlyDSL stream so HIP graph capture records the work.

## Correctness and workload

`driver.py` compares both GEMM outputs directly against the protected Triton
source for every case in `session_cases.json`. Correctness caps token count at
64; performance covers decode T=1 and T=64 plus prefill T=16384. The benchmark
times both launches together under HIP graph replay.

## Rules

- Implement the result in FlyDSL only.
- Edit only `kernel.py`.
- Do not change the factory or launch ABI.
- Do not bypass grouped routing, MXFP8 scaling, output dtype conversion, or the
  weighted GEMM2 epilogue.
- Keep all shape-dependent compilation in the factory, not in timed launches.
