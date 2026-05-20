# fw-perf-002 sglang radix tree prefix cache eviction

Framework: sglang
Tags: cache, scheduling, eviction

`--radix-eviction-policy lru` outperforms `random` when prompts have
overlapping prefixes (chat templates, system prompts). On large-conc
sweeps (CONC>=128) lru saves ~3-5% tput; on synthetic-random workloads
the two are within noise.

Reference: sglang 0.4+ default already lru; `random` was tried in
marathon and tied below baseline.
