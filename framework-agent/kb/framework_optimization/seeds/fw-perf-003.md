# fw-perf-003 PagedAttention block size sweep

Framework: vllm
Tags: pagedattention, kvcache, block_size

`--block-size 256` cuts KV-cache fragmentation vs default 16 for long
context (ISL>=4096). Trade-off: smaller batches see negligible diff;
beyond 512 the per-block bookkeeping overhead eats the win.

Reference: marathon validated +4.3% on gpt-oss-120b at ISL=8192; +1.1%
on Llama-70B at ISL=4096; flat at ISL<=2048.
