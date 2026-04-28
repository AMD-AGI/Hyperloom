# comm_optimization — TP / DP communication tuning

**Family**: `long` · **Cost**: ~30‑60 min · **Risk**: 5% accuracy

Marathon‑only. Tunes NCCL / RCCL env vars (e.g.,
`NCCL_NCHANNELS=4`, `NCCL_MIN_NRINGS`, `NCCL_ALGO`) and TP/DP topology.
Each candidate rebuilds at most one symbol; otherwise launches and
benchmarks under the new env.
