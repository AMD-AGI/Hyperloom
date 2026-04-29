# deep_kernel_analysis — marathon-only kernel dispatch study

**Family**: `deep_kernel` · **Cost**: ~30‑60 min · **Risk**: zero (read‑only)

Marathon Sage / Watchdog joint task. Reads recent profiler traces, looks
for kernel‑dispatch patterns indicating that aiter / vendor kernels are
dominating wall time, then emits findings to `findings/kernel_analysis.md`.

Output is read by `operator_tuning` and `vendor_kernel_config` next.
