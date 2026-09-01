# Kimi-K3 AgentX Launch Guide (for an executing agent)

Follow these steps in order. Do not skip the concurrency check in Step 4 — it
is the difference between a run that completes and one that silently produces
a failed/invalid result.

## Step 0 — Evidence this guide is based on

Every parameter below is copied from a real run, not invented:
- `/shared_nfs/lzeng/k3-three-group/run-20260817T045500Z/B-C4/vllm_command.txt`
  and `benchmark_command.txt` — the only Kimi-K3 run in that evidence set with
  **zero** `InvalidInferenceResultError` (334 requests, 290 profiled, 44
  warmup-dropped, 0 error-dropped).
- `/shared_nfs/lzeng/k3-three-group/run-20260817T045500Z/B-C64/` — same vLLM
  config, only `--concurrency` changed from 4 to 64: 1003 requests, only 296
  profiled, 575 `InvalidInferenceResultError`. **Do not use conc=64.**

## Step 1 — Prerequisites

- 8×MI355X (or equivalent) GPU host, model weights available locally (path
  used in evidence: `/shared_nfs/hyperloom/models/Kimi-K3`).
- Anthropic credentials (Hyperloom's optimization loop itself is Claude-driven,
  required regardless of AgentX):
  ```bash
  export ANTHROPIC_API_KEY=<value>
  export ANTHROPIC_BASE_URL=<value>
  ```
- AgentX mode flag:
  ```bash
  export HYPERLOOM_AGENTX=1
  ```

## Step 2 — Install

```bash
export REPO_ROOT="$(pwd -P)"
export USER_DATA_PATH="/path/to/hyperloom-run"
bash "$REPO_ROOT/src/hyperloom/inference_optimizer/assets/install.sh"
source "$USER_DATA_PATH/runtime/kernel-agent.env.sh"
```

`HYPERLOOM_AGENTX=1` being set before this step matters: `install.sh` only
installs the `aiperf` binary when `HYPERLOOM_AGENTX` (or `INSTALL_AIPERF`) is
truthy. If you skip this and set the flag later, the run will fail preflight
with a missing-`aiperf` error.

## Step 3 — Verify the aiperf capability gate before launching

```bash
python3 -c "
from hyperloom.inference_optimizer.agentx.preflight import check_aiperf_capability
check_aiperf_capability()
"
```

If this raises or warns, stop and resolve it first — a stale `aiperf` build
produces a run that looks identical to a valid one but is not leaderboard
comparable (missing `require_streaming`, different trace-idle-gap cap).

## Step 4 — Choose concurrency (do not default to 64)

| `--conc` | Evidence | Verdict |
|---|---|---|
| 4 | B-C4: 0 errors | **Use this to start.** The only value with a clean, evidenced run. |
| 8 | Repeatedly clean on 8×MI355X: `submission_valid=true`, <1% error, warmup ~3000s (87 requests) | Safe. This is the value most of the evidence below was gathered at. |
| 16 | Runs, but with a ~30% error floor recorded on this host | Usable only if you accept that raising `AGENTX_FAILED_REQUEST_THRESHOLD` past the canonical 0.10 marks the round `submission_valid=false` — which makes `benchmark_result.py` refuse the measurement and the coordinator relaunch PRELUDE forever. Do not raise it to "fix" the error rate. |
| 32 | Warmup ~5000s at conc=16 extrapolates linearly; not yet closed end-to-end | Step up one value at a time and re-check `submission_valid` + error count after each. |
| 64 | B-C64: 575 `InvalidInferenceResultError`, only 296/1003 requests profiled | **Do not use.** Known to fail: warmup itself completes (707/707, ~12075s), but the subsequent profiling phase's fixed benchmark-duration window is too short for Kimi-K3's long-tail request latency (median ~19min, max ~36min), so in-flight requests get cancelled and counted as errors. |

## Step 5 — Launch command

```bash
python3 -m hyperloom.inference_optimizer.cli optimize \
    --model /shared_nfs/hyperloom/models/Kimi-K3 \
    --framework vllm \
    --gpu-type mi355x \
    --tp 8 \
    --conc 4 \
    --max-hours 24
```

Notes:
- `--framework vllm`: the only framework with an evidenced clean Kimi-K3 run.
  The underlying `vllm serve` invocation Hyperloom will produce should be
  equivalent to (verify server args match if you customize anything):
  ```
  vllm serve <model> --served-model-name moonshotai/Kimi-K3 \
    --tensor-parallel-size 8 --gpu-memory-utilization 0.9 \
    --max-model-len 1048576 --trust-remote-code --enable-prefix-caching \
    --moe-backend auto --load-format auto --max-num-seqs 512 \
    --max-cudagraph-capture-size 256 --max-num-batched-tokens 16384 \
    --enable-auto-tool-choice --tool-call-parser kimi_k3 \
    --reasoning-parser kimi_k3
  ```
- `--isl`/`--osl` are not meaningful under AgentX (real corpus replay ignores
  them) — omit them.
- `--max-hours 24`: warmup alone can take multiple hours at low concurrency
  too (44 warmup requests at conc=4 still drains against Kimi-K3's real
  prefill speed); do not set this aggressively low.

## Step 5b — CPU KV offload: size it from the cgroup, not from `free`

Kimi-K3's on-GPU KV pool holds a small fraction of an agentic working set, so
without a CPU tier the prefix cache thrashes: MEASURED at conc=8, GPU-side
`prefix_cache_hit` was **4.0%** against a `theoretical_prefix_cache_hit` of 94%.
Adding the official `SimpleCPUOffloadConnector` took the same run to
`prefix_cache_hit=68.5%` with a new `ext_cache_hit=80.7%` (the CPU tier), and
the closed session reported 56.3 tok/s/GPU against 18.86 without it — the
throughput comparison is a strong signal but has not been isolated to the
connector alone, so treat the hit rates as the established result.

The official configuration is three things and nothing more:

```bash
export PYTHONHASHSEED=42        # identical prefixes must hash to identical block keys across ranks
--kv-transfer-config '{"kv_connector":"SimpleCPUOffloadConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"cpu_bytes_to_use_per_rank":N,"lazy_offload":false}}'
```

`eviction_policy` / `store_threshold` / `max_tracker_size` / `kv_offload_backend`
stay at vLLM defaults. Tuning them is beyond the official recipe, and is
second-order next to capacity anyway.

**Deriving `N`.** Do not hardcode it, and do not size it from `free`. Inside a
container `free` reports the HOST while the cgroup is what actually kills you.
MEASURED on an 8×MI355X pod: `free` showed 2751 GiB, the cgroup capped the
container at 2048 GiB, and a pool sized off the host number put worker RSS at
222.9 GiB/rank with cgroup usage pinned at 2047.97/2048 GiB — 99.998%, no
page-cache headroom left.

```bash
TP=8
KV_POOL_PCT=66                  # leave a third for framework + activations
_cg=$(cat /sys/fs/cgroup/memory.max 2>/dev/null)
case "${_cg:-}" in "" | max | *[!0-9]*) _cg=0 ;; esac
_host=$(( $(awk '/^MemTotal:/{print $2}' /proc/meminfo) * 1024 ))
[ "$_cg" -gt 0 ] && [ "$_cg" -lt "$_host" ] && BUDGET=$_cg || BUDGET=$_host
CPU_BYTES_PER_RANK=$(( BUDGET * KV_POOL_PCT / 100 / TP ))
```

On the box above that yields 181 GB/rank (1451 GB total) instead of the
hand-written 229 GB/rank (1834 GB), which is close to the largest
previously-measured-stable sizing on that host.

**Two things that will bite:**

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is a HARD incompatibility
  with any KV connector — `vllm/config/vllm.py` raises `ValueError` unless
  `enable_cumem_allocator` is on, and that only auto-enables under sleep mode.
  The official recipe never sets this variable; if you added it, remove it.
- More capacity keeps paying: `ext_cache_hit` was still climbing (84.7%) as the
  working set grew, with no plateau. But on a 2048 GiB cgroup there is no room
  left to buy, so closing the remaining gap to the ~96% ceiling needs a
  larger-memory node, not a different eviction policy.

## Step 6 — What a healthy run looks like

- Warmup phase log line: `Phase warmup (warmup) complete | completed=<N>,
  cancelled=0, errors=0`. Any `cancelled` or `errors` > 0 here is a red flag.
- Profiling phase log line: `Phase profiling (profiling) complete |
  completed=<N>, cancelled=0, errors=0` with `grace_period_timeout` absent or
  `False`. `grace_period_timeout=True` with nonzero `cancelled` means the
  profiling window closed on in-flight requests — this is the exact B-C64
  failure mode; if you see it, the run should be treated as failed even if it
  produced a JSON result file.
- Result JSON: check `num_requests_successful` against
  `request_accounting.records_total` — a low ratio (as in B-C64: 296/1003)
  means the run does not meet the failed-request-threshold (0.10) and should
  not be reported as a successful benchmark.

## Step 7 — If it fails mid-warmup with a Hyperloom timeout (not an aiperf error)

If the round is killed by Hyperloom's own subprocess timeout before warmup
even finishes (distinct from the aiperf-level failure above), the cap is
derived — reach for the input, not the answer:

```bash
export AGENTX_WARMUP_GRACE_PERIOD=3600      # the warmup share, anchored at CONC=8
```

`agentx_baseline_timeout_sec()` in `orchestrator/actions/executors/baseline.py`
builds the cap as `AGENTX_DURATION + non-warmup overhead + warmup share`, and
scales the warmup share by `CONC / 8` — warmup is per-lane requests × CONC
lanes, so it is linear in concurrency by construction. At CONC=32 a 3600s
anchor becomes a 14400s warmup share and a 23400s cap; at or below CONC=8 the
derivation is unchanged.

Anchor the grace at the **CONC=8** measurement (~3000s of warmup), not at a
number you tuned for a higher concurrency — the floor multiplies whatever you
give it, so handing it an already-conc-16-sized value double-counts.

`AGENTX_BASELINE_TIMEOUT_SEC` pins the cap outright and short-circuits all of
the above. Use it only when you want a fixed number, and be aware it disables
the CONC scaling.

## Step 8 — Resume after an interruption

```bash
python3 -m hyperloom.inference_optimizer.cli optimize --resume-from "$SESSION_DIR"
```
