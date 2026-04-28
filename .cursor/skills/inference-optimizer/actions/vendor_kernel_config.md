# vendor_kernel_config — adjust vendor (e.g. aiter) configs

**Family**: `deep_kernel` · **Cost**: ~15‑30 min · **Risk**: 10% accuracy

Edit vendor‑provided config files (without modifying GEAK config — IR-7)
to better match the model's typical shapes. Side effect is purely a
workspace patch; the rebuild + benchmark cycle is delegated to a
follow‑up `integrate` invocation.
