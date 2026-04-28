# compiler_tuning — torch.compile / inductor pass tuning

**Family**: `long` · **Cost**: ~45‑75 min · **Risk**: 5% accuracy

Marathon‑only. Tunes inductor pass options:
`fx_graph_remote_cache`, `coordinate_descent_tuning`,
`epilogue_fusion_first`, `triton.unique_kernel_names`. Always uses
`patch_inductor.py --target-file ... --best-config ...` (IR‑6).
