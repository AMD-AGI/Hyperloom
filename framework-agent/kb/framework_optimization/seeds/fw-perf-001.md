# fw-perf-001 vllm chunked prefill split granularity

Framework: vllm
Tags: prefill, chunked, scheduling

Smaller `--chunked-prefill-size` reduces TTFT-degrading prefill-decode
interleaving. Marathon-validated sweet spot on Llama / Qwen serving:
8k-16k tokens. Larger (32k+) starves decode steps; smaller (<4k)
re-introduces TTFT spikes when batch is full.

Reference: marathon `params` round on Qwen3-32B + sglang yielded +0.3-0.6%
on its own, +1.2% combined with `--mem-fraction-static 0.92`.
