# fw-boundary-003 FP8 quantization: kernel op vs Python config

Framework:
Tags: boundary, fp8, quantization

FP8 weight quantisation involves three layers; ownership splits:

1. **Kernel op** (GEMM, attention): kernel-agent owns. Never patch
   from framework agent.
2. **Quant-aware loading + scaling-factor application**: framework
   owns -- this is Python orchestration of the kernel call.
3. **CLI flag exposure** (`--quantization fp8`, `--kv-cache-dtype
   fp8_e4m3`): orchestration `params` arm grids; framework agent
   discovers the flag names but does not flip them.

Cross-checks (`fw-keep` lessons must respect this split): a framework
patch that changes both kernel-csrc and Python config triggers
critic review with `mixed_layer_violation`.
