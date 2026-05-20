# fw-boundary-001 sampler kernel vs Python config split

Framework:
Tags: boundary, sampler

Sampling-strategy CHOICE (top-p / top-k / temperature) lives in
framework Python config (sglang `sampling_defaults`, vllm
`SamplingParams`). Sampling kernel REWRITE (e.g. fused softmax+argmax)
lives in kernel-agent. The framework agent must NOT patch sampling
kernels under `csrc/` / `cuda/`; the kernel agent must NOT patch
Python sampling defaults.
