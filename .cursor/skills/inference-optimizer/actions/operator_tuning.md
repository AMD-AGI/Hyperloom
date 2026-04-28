# operator_tuning — autotune top-N hot operators

**Family**: `deep_kernel` · **Cost**: ~25‑50 min · **Risk**: 10% accuracy

Marathon‑only. Generates per‑shape configs for the top dispatched
operators (e.g., `aiter::flash_attn_v3`, `aiter::fused_moe`). Writes a
JSON config under `kernel_configs/<op>.json` and rebuilds.
