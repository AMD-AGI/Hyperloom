# R9700 BF16 activation × MXFP4 weight `matmul_ogs`

For GPT-OSS-20B MoE, keep BF16 activations and packed FP4 weights with E8M0
scales shared by blocks of 32 weights. The ROCm Triton serving path executes
`triton_kernels.matmul_ogs`. Benchmark its routed up and down projections and
the installed padding and routing behavior. A BF16 unquantized GEMM does not
measure this path. Keep BF16 activations when evaluating weight-only MXFP4.

Treat the stock flag recipe and the upstream RDNA recipe as separate baselines.
The latter is Triton PR #10810, commit 4cd5c49: routed M ≤ 512 uses
16/64/512; larger M uses 64/128/128, four warps, one stage, and split-K 1.
Verify the XCD field against the installed source. Preserve correctness checks
and measure the selected candidate independently in serving.

The MXFP4 storage format alone does not establish a native scaled-FP4 matrix
ceiling. Report that ceiling as unknown without measured hardware evidence.
