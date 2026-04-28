# baseline — first-light tok/s/GPU measurement

**Family**: `prep` · **Cost**: ~5‑10 min · **Risk**: low

Run `scripts/run_baseline.sh` (IR-3) to obtain the canonical
`tok/s/GPU` for the model‑class default backend and parameter set. Result
is written to `<session_dir>/results/baseline/metrics.json`.

Preflight (IR-4):

1. `kill_server sglang` then `kill_server vllm`
2. `check_gpu_memory` — fail if any GPU is using > MIN_GPU_PCT (=3%)
3. unset `PROFILE` / `SGLANG_TORCH_PROFILER_DIR`
4. launch via `scripts/run_baseline.sh`

Outputs:

- `update_state` `baseline_tput=...`, `current_tput=baseline_tput`
- `send_message` topic=`decision` "baseline locked in"
- file artifact `results/baseline/metrics.json`
