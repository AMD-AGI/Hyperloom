# fw-boundary-002 KV quantisation kernel vs framework config

Framework:
Tags: boundary, kv_cache, quantization

`--kv-cache-dtype fp8_e4m3` is a framework-config knob: orchestrator
proposes it via the `params` arm. The fused fp8 attention KERNEL
itself (aiter / triton variants) lives in kernel-agent territory.
Framework patches must NOT modify `kernels/fused_attention_*.cu`;
framework agent should propose the CLI flag flip, never the kernel.
