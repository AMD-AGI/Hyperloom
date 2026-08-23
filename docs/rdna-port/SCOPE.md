# RDNA support port — scoping (superseded by the implementation)

Issue: https://github.com/AMD-AGI/Hyperloom/issues/1196

**This document is kept for history and is partly WRONG.** It was written
against a clone that was 402 commits stale, and upstream has since refactored
the integration points it names. Recorded here because the correction is the
useful part:

| Scoped (stale) | Actual |
|---|---|
| Edit `_AMD_GPU_TYPES`, `_GFX_TO_RUNNER` and `_AMD_GPU_DISPATCH_IDENTITIES` separately in `gpu_types.py` | Identities moved to `hyperloom/common/gpu_identity.py`; `_AMD_GPU_TYPES` and the rocm-smi product tags are **derived** from it. Adding a board is one row + one `_GFX_TO_RUNNER` entry |
| Add R9700/RADEON product tags to `_autodetect_gpu_type` | Not needed — tags derive from the identities table, so autodetect works the moment the row exists |
| Deliver `custom_r9700.sh` as a core change | The runner is operator-supplied via `$HYPERLOOM_BYPASS_SCRIPTS_DIR`; no core change. Shipped as `examples/rdna/custom_r9700.sh` |
| Add an RDNA4 quant table to `quantization_schemes.py` | The existing MI355X gate already excludes mxfp4 everywhere else. What was missing was the *reason* (gfx950-gated silicon, not a missing kernel) and a test locking it |
| Flip `test_profile_and_kernel_handlers.py`'s gfx1100 expectation | Left alone; coverage added as `test_rdna4_r9700_support.py` instead |

Net: the port is far smaller than scoped, because upstream had already done the
single-source-of-truth refactor this needed. See `README.md` in this directory
for the delivered state.
