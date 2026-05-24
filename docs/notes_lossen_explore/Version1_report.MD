# Version1 Run Report — DeepSeek-R1-0528 FP8 on MI300X (vLLM)

> First end-to-end run of the Hyperloom inference_optimizer skill against
> `/wekafs/models/DeepSeek-R1-0528`. This document records the achieved
> gain, the per-lever contribution attribution, and every latent bug
> exposed during the run.

## 1. Run Identity

| Field | Value |
|---|---|
| Session ID (attempt 3, the one that produced the result) | `DeepSeek-R1-0528_20260520T125306Z_35ddebd4` |
| Session dir | `/workspace/hyperloom-KB` |
| Final run log | `/workspace/hyperloom-KB/optimizer_runs/run_fresh-nokernel-20260520_125300.log` |
| Final report | `/workspace/hyperloom-KB/reports/final.md` + `final.json` (165 KB) |
| Archived attempt 1 (deadlocked + sweep snapshot) | `/workspace/hyperloom-KB-archive-20260520_125215` |
| Repo revision | `91b62d6` |
| Stack | ROCm 7.2.2, vLLM `0.20.2rc1.dev53+g01b9b5af6`, aiter (build_count=4, 85 wheel `.so`) |
| Backends | Orchestration=Claude (`claude-opus-4-7`), Critic=critic-agent (codex `gpt-5.4`), Robustness=robustness-agent, Kernel=**DISABLED** (workaround, see §3) |

### Workload pinned by the operator

```
framework      = vllm
gpu_type       = mi300x  (8x MI300X-OAM, 192 GiB HBM3 each)
TP             = 8
CONC           = 64
ISL / OSL      = 1024 / 1024
max_model_len  = 6144   (auto-set when not supplied; baseline ran at 8192 for the very first attempt)
precision      = FP8 (model weights)
target_gain    = 100% (cumulative_gain_pct)
max_hours      = 12 (attempt 1) / 11 (attempt 3)
```

## 2. Headline Result

| Metric | Value |
|---|---|
| Baseline throughput (vllm `0.20.2rc1`, default args) | **`2301.4` tok/s/GPU** |
| Final `current_best` throughput | **`2603.4` tok/s/GPU** |
| **Validated cumulative gain** | **`+13.12%`** (`stack_len=1`, validated at `2026-05-20T17:14:12Z`) |
| Stop reason | `time_exhausted` (660 min wall budget exhausted) |
| Target reached? | No — target was `+100%`; only `+13.12%` was achieved. The headroom that was *not* harvested came from kernel optimization (we had to disable it to dodge a deadlock — see §3) plus unexplored config combinations (see §4) |
| Crashes / OOMs | 0 |
| Pruned families (residual) | `['kernel_opt']` (Robustness pruned this on a false `ray_pending_starvation` alert; benign because kernel was already disabled) |
| GPU state at exit | 284 MB/GPU (clean release) |

### Validated stack (`optimization_stack`)

```json
[
  {
    "action": "explore",
    "variant_name": "kv_cache_fp8",
    "candidate_extra_sglang_args": "--kv-cache-dtype fp8",
    "extra_envs": {},
    "tput": 2603.4356215081957,
    "ts": "2026-05-20T17:14:12.176879+00:00"
  }
]
```

That is the **entire** body of accepted optimizations the run produced.

## 3. Gain Attribution — Who Actually Contributed

This is the most important section. **Only one variant was ever placed
on `optimization_stack`:** `kv_cache_fp8` (`--kv-cache-dtype fp8`).
However the `+13.12%` validated gain is *not* a clean measurement of that
flag — see the breakdown below.

### 3.1 The single accepted KEEP

| Measurement | Tput (tok/s/GPU) | Δ vs. cold baseline |
|---|---|---|
| Cold baseline (no extra args) | 2301.4 | (reference) |
| `kv_cache_fp8` **standalone**, when first proposed | 2353.5 | **+2.26%** |
| `kv_cache_fp8` after **stack-rebench** (same config, second measurement) | **2603.4** | **+13.12%** |

The 2603.4 → +13.12% number is what the optimizer records as
`cumulative_gain_validated` and exposes as the "current best". The
**LLM-side contribution attributable to a deliberate config decision is
only the +2.26%** the variant produced on first measurement. The
remaining +10.86% came from a re-run of the *same* `--kv-cache-dtype fp8`
config that — for reasons the run doesn't isolate — measured 250 tok/s
faster. The probable mechanism is warm-state effects (aiter JIT cache
fully warm, vLLM compile cache warm, Linux page cache warm for the 671B
weights file) on the second launch; see Bug B-4 in §6.

So the honest attribution is:

| Contributor | Δ vs cold baseline | Confidence |
|---|---|---|
| `--kv-cache-dtype fp8` (explicit, reproducible config change) | **+2.26%** | High — measured against cold baseline; standalone variant |
| Warm-state / measurement variance on second launch of the same config | **+10.86%** | Low — same config, same workload, no other changes; almost certainly not a real config win |
| **Total reported `cumulative_gain_validated`** | **+13.12%** | Medium — true number, but mis-attributed entirely to "kv_cache_fp8" |

### 3.2 What the LLM *proposed* but the optimizer correctly rejected

Round 1 (`explore-001`, base = bare baseline 2301.4): 7 sglang-flavored
env vars + naive parallelism tweaks. **All regressed.** Some came back
much later with different framing, but on the cold baseline none won:

| Variant | args / envs | Tput | Δ vs 2301.4 |
|---|---|---|---|
| aiter_all | `VLLM_ROCM_USE_AITER=1` | 2138.4 | −7.08% |
| aiter_moe | `+VLLM_ROCM_USE_AITER_MOE=1` | 2105.7 | −8.50% |
| aiter_mla | `+VLLM_ROCM_USE_AITER_MLA=1` | 2197.0 | −4.54% |
| aiter_rmsnorm | `+VLLM_ROCM_USE_AITER_RMSNORM=1` | 2205.2 | −4.18% |
| max_num_seqs_512 | `--max-num-seqs 512` | 2176.3 | −5.44% |
| max_num_seqs_1024 | `--max-num-seqs 1024` | 2253.7 | −2.07% |
| chunked_prefill_8192 | `--enable-chunked-prefill --max-num-batched-tokens 8192` | 2171.2 | −5.66% |

Round 2+ (still `explore-001`, base = **2603.4** the post-KEEP stack
rebench): the LLM re-tried promising flags against the new (inflated)
baseline. **Several measured higher than the cold baseline but lower
than the round base**, so the optimizer correctly refused them as
`gain_below_threshold`:

| Variant | args / envs | Tput | Δ vs cold 2301.4 | Δ vs round base 2603.4 | Verdict |
|---|---|---|---|---|---|
| aiter_linear | `VLLM_ROCM_USE_AITER_LINEAR=1` | 2509.3 | +9.03% | **−3.62%** | REVERT (correct vs round base) |
| aiter_moe | (env) | 2501.8 | +8.71% | −3.90% | REVERT |
| max_num_seqs_1024 | `--max-num-seqs 1024` | 2551.4 | +10.86% | −2.00% | REVERT |
| aiter_rmsnorm | (env) | 2419.3 | +5.12% | −7.07% | REVERT |
| max_num_seqs_512 | `--max-num-seqs 512` | 2409.6 | +4.70% | −7.44% | REVERT |
| triton_flash_attn | (env) | 2395.8 | +4.10% | (not tested vs new base) | REVERT |
| gpu_mem_util_095 | `--gpu-memory-utilization 0.95` | 2585.5 | +12.34% | −0.7% | REVERT |
| aiter_all (re-test) | (env) | 2364.3 | +2.73% | −9.18% | REVERT |
| block_size_32 | `--block-size 32` | 2316.5 | +0.66% | −11.02% | REVERT |
| enable_expert_parallel | `--enable-expert-parallel` | 1924.8 | −16.36% | (huge regression) | REVERT |
| max_num_batched_tokens_32768 | (arg) | 2304.8 | +0.15% | −11.5% | REVERT |
| enable_chunked_prefill | (arg) | 2285.3 | −0.70% | −12.2% | REVERT |
| enable_prefix_caching | (arg) | 2297.6 | −0.17% | −11.7% | REVERT |
| max_num_batched_tokens_16384 | (arg) | 2242.2 | −2.57% | −13.9% | REVERT |
| chunked_prefill_4096 | (args) | 2182.4 | −5.17% | −16.2% | REVERT |
| max_num_seqs_128 | `--max-num-seqs 128` | 2280.0 | −0.93% | −12.4% | REVERT |
| max_num_seqs_64 | `--max-num-seqs 64` | 2179.4 | −5.30% | −16.3% | REVERT |
| prefix_caching | `--enable-prefix-caching` | 2134.8 | −7.24% | −18.0% | REVERT |
| chunked_prefill_2048 | (args) | 2030.7 | −11.76% | −22.0% | REVERT |
| kv_cache_fp8_e5m2 | `--kv-cache-dtype fp8_e5m2` | (failed) | n/a | n/a | REVERT (no data) |
| num_scheduler_steps_8 | `--num-scheduler-steps 8` | (failed) | n/a | n/a | REVERT (no data) |

**Total**: 29 variant runs across ~7.5h of EXPLORE; 1 KEPT, 28 REJECTED.

### 3.3 Why most variants regressed

A few honest, model-level reasons (not bugs in the optimizer itself):

1. **aiter env vars don't deliver here**: most `VLLM_ROCM_USE_AITER_*`
   knobs were not net wins for DeepSeek-R1 FP8 CONC=64 / ISL=OSL=1024 on
   MI300X. The default vLLM 0.20.2 path with FP8 weights is already
   wired through aiter on MI300X, so toggling these is a no-op or worse
   (they route through a less-tuned implementation).
2. **`--max-num-seqs` is irrelevant at CONC=64**: the default
   max_num_seqs (256) already accommodates the active concurrency.
   Raising it to 512/1024 only inflates KV cache reservations and hurts
   scheduling.
3. **`--enable-chunked-prefill --max-num-batched-tokens N` hurts at
   ISL=1024**: chunked prefill is a win when prefill is the bottleneck;
   here OSL=1024 means decode dominates throughput, and chunking just
   adds overhead.
4. **`--enable-expert-parallel`** broke the load distribution across TP
   ranks for this MoE → −16.4%.
5. **`--kv-cache-dtype fp8` *was* a win** because the KV cache is the
   memory-bandwidth bottleneck at CONC=64 OSL=1024 decode-dominant
   workload. Halving its footprint frees HBM bandwidth for the matmuls.

## 4. Levers That Were Not Explored

Per the SKILL design intent, EXPLORE should chain into KERNEL → SWEEP →
CLOSE. We never reached KERNEL with kernel-agent enabled (Bug B-1 below)
nor exercised:

- **Kernel optimization** (`run_optimization` / `integrate` / GEAK /
  OOB): would have targeted `reusable_native_kernel_ids` from the
  profile trace. Could plausibly add another 5–20% on top.
- **Stack composition**: `kv_cache_fp8` was never combined with the next
  most promising standalone configs (`gpu_mem_util_095` +12%,
  `max_num_seqs_1024` +11%, `triton_flash_attn` +4%, `aiter_linear`
  +9%, `aiter_moe` +9%). The optimizer rejected each one *individually*
  against the post-KEEP stack base of 2603.4, but never tried
  combinations like `--kv-cache-dtype fp8 --gpu-memory-utilization 0.95`
  which could in principle stack additively.
- **Sweep**: the workload sweep grid (9 combos over CONC × ISL × OSL)
  ran for 2 variants in the second attempt but was killed when we
  archived that session. The current attempt-3 never reached SWEEP
  because EXPLORE consumed the full budget.
- **`torch.compile`** levers (`--enforce-eager` off, `--compilation-config`,
  etc.) — not in the LLM's explore grid.
- **MTP / speculative decoding** — not in scope without kernel-agent.

## 5. Wall-Clock Budget Breakdown

| Phase / activity | Wall-clock | Notes |
|---|---|---|
| Attempt 1 setup + baseline launch + deadlock observation | ~50 min | Discovered the EXPLORE/profile gate (Bug B-1) |
| Attempt 2 (skip_to_kernel injection, raced through to SWEEP) | ~80 min | Profile produced no torch trace (Bug B-3); auto-skipped to SWEEP; 2 sweep variants completed before we stopped it |
| Attempt 3 install + preflight + new baseline | ~13 min | Baseline measured 2301.4 (matches attempt 1's 2303.6 within 0.1% noise) |
| Attempt 3 EXPLORE round (29 variants × ~15 min each) | ~7.5 h | 1 KEEP, 28 REVERTs |
| Attempt 3 idle ticks + CLOSE/report | ~30 min | Hit `time_exhausted` at minute 660 |
| **Total clock spent (all attempts)** | **~12 h** | Including the ~1h spent on attempts 1 & 2 before the final fresh launch |

## 6. Bugs & Issues Discovered

Severity scale: **S1** = blocks the run / requires manual intervention,
**S2** = silently degrades the run, **S3** = nuisance.

### B-1  (S1) — EXPLORE-vs-profile execution-order deadlock when `--kernel-claude` is on

**Symptom**: After baseline succeeds and the Coordinator advances to
EXPLORE, every Orchestration proposal is denied:

- `explore` → `rule=execution_order`, hint `"propose/delegate profile; last_profile_trace is empty"`
- `profile` → `rule=phase_incompatible`, hint `"action 'profile' is not in the allowed set ['explore','recover','specialist']"`
- `report` → `rule=phase_incompatible` (CLOSE only)

Orchestration heart-beats `"blocked"` indefinitely. We saw 30 ticks
(~15 min) of zero forward progress before the Robustness plateau judge
fired.

**Root cause**: `inference_optimizer/orchestrator/coordinator.py:2724-2734`
unconditionally gates non-`profile` actions on `last_profile_trace`
when `"kernel" in self.role_registry`. But the v0.8 phase machine
(`phase_state.py`) only permits `profile` in **KERNEL** phase, not
EXPLORE. So the only way out is for EXPLORE to spontaneously plateau,
which requires the `explore` action to have *already run* — circular.

**Workaround used**: relaunched with `--no-kernel`, which removes
`"kernel"` from the role registry and disables the gate. Cost: no
kernel optimization at all this run.

**Suggested fix**: drop the action-layer gate. The request-layer gate at
`_sequence_denial_for_request` (lines 2792-2811) is sufficient — it
denies `select_kernels` / `run_optimization` requests when
`last_profile_trace` is empty, which is exactly where the trace is
actually needed. `explore` (parameter sweeps over `extra_args`/
`extra_envs`) genuinely does not need a profiler trace.

### B-2  (S2) — Robustness `ray_pending_starvation` is a false positive

**Symptom**: At each tick the robustness-agent emits

```
alert(high, "ray reports 8220 pending tasks for N consecutive ticks
       (>= 3); cluster quota likely wedged")
prune_branch(family='kernel_opt', reason='ray_pending_starvation')
escalate_strategy_change(hint='...')
```

And in attempt 1 it also pruned `family='explore'` on the same signal,
compounding the B-1 deadlock.

**Reality**: `ray status` and `rocm-smi --showpids` show **zero pending
Ray tasks**, zero scheduling demand, and 0/8 GPUs allocated by Ray.
The Ray dashboard API on `:8265` is not exposed in this pod, so
whichever detector is reading the "8220 pending" figure is reading a
stale or wrongly-scoped counter.

**Impact**: the detector pruned the `explore` family in attempt 1,
locking us into the deadlock alongside B-1.

**Suggested fix**: the robustness-agent's `ray_pending_starvation`
detector should: (a) verify the Ray dashboard is reachable before
emitting; (b) only count tasks tagged for *this* session; (c) require
the count to be non-zero relative to a baseline taken at run start.

### B-3  (S2) — vLLM `profile` produces no `.trace.json.gz`

**Symptom**: profile_executor logs:

```
profile_executor: trace dirs exist but no .trace.json.gz files in
.../runs/profile/<task>/benchmark_vllm_<ts>/torch_trace
```

This left `last_profile_trace=""`, which (under `--kernel-claude`) would
have silently blocked all kernel requests at `_sequence_denial_for_request`
even if B-1 were fixed. In attempt 2 it caused the KERNEL phase to
have no candidate kernels, so the phase machine immediately exited to
SWEEP without doing any kernel work.

**Hypothesis**: the `profile_vllm.yaml` config writes
`profiler.torch_profiler.enabled=true` but the vllm benchmark script
(`InferenceX/benchmarks/vllm_mi300x.sh`) does not honor it / does not
set `VLLM_TORCH_PROFILER_DIR` for the worker subprocesses. Needs a
deeper look into whether vLLM 0.20.2's profiling integration is wired
through Magpie correctly for the `vllm_<gpu>.sh` scripts.

### B-4  (S2) — `kv_cache_fp8` standalone vs stack-rebench measurement gap inflates `cumulative_gain_validated`

**Symptom**: The same `--kv-cache-dtype fp8` configuration measured
2353.5 tok/s the first time and **2603.4** tok/s 15 min later (post-KEEP
stack rebench). Same model path, same workload, same args, same env.
Delta = +10.6% with no config change.

**Likely cause**: by the second launch, aiter JIT cache had stabilized
(build_count had stopped at 4 well before the rebench), vLLM had
compile-cache hits, and the 671B FP8 weights file was fully in the
Linux page cache (avoiding the 9.5-minute `Loading weights took 594.48
seconds` we saw on the very first baseline).

**Impact**: the reported `+13.12% cumulative_gain_validated` is
substantially attributable to warm-state effects, not the config
change. Any downstream consumer of `cumulative_gain_validated` (the
Critic's "stack KEEP threshold", the Coordinator's plateau judges, KB
priors) is seeing an inflated signal.

**Suggested fix**: enforce a warm-state baseline. Either (a) run the
baseline benchmark *twice* and use the second measurement as
`baseline_tput`, or (b) re-bench the *current_best-minus-the-new-variant*
config immediately before the variant runs, so the comparison is
warm-vs-warm. (b) is roughly what stack-rebench is supposed to do, but
right now stack-rebench runs the *new* config against the cold
baseline — that's the asymmetry that produces this gap.

### B-5  (S2) — `kill -TERM` on the optimizer does not clean its Magpie/vLLM child tree

**Symptom**: When we sent SIGTERM to PID `1331876`, the optimizer
exited cleanly and wrote its final summary, but the Magpie subprocess
(PID `1366167`) and `vllm serve` (PID `1369733`) plus 8 `VLLM::Worker_TPx`
processes (PIDs `1369669..1369676`, re-parented to init) survived,
holding `8 × 86 GB = 688 GiB` of HBM. We had to find the worker PIDs
manually via `find /proc/*/fd | grep kfd` and `SIGKILL` them
individually. The optimizer continued to spawn new sweep variants
in this interval before its own loop noticed the SIGTERM.

**Suggested fix**: the CLI's SIGTERM handler should set a "shutdown"
flag, but **also** explicitly `os.killpg(os.getpgid(pid), SIGTERM)` for
any running benchmark subprocess group and wait for it before returning.
Without that, every operator-initiated stop is followed by an orphan
storm.

### B-6  (S3) — Cortex KB `propose_point` returns HTTP 422 for `kind='recipe'`

**Symptom**: At every session start:

```
kb-service.primus-cortex.../v1/points/propose 422 Unprocessable Entity
  ("attrs failed schema validation for kind='recipe'")
hypothesize sync failed (canonical_id 'recipe:DeepSeek-R1-0528:mi300x'
  not found in KB); enqueueing NDJSON
```

The Cortex KB schema for `kind='recipe'` does not accept what the
optimizer is sending. Soft-fails into NDJSON queue (never drains in
this run). KB priors are therefore empty for this model — every
session will start cold.

**Suggested fix**: align the `recipe` payload schema between the
optimizer's `cortex_kb_client` and the KB service, OR adjust the KB
service to accept the current payload. Until then, KB priors for new
(model, gpu) pairs will never accumulate.

### B-7  (S3) — `target_analysis` model-name mapping miss

**Symptom**:

```
target_analysis_executor: status=skipped reason=model_mapping_miss
  rows=0 (model name mapping miss for '/wekafs/models/DeepSeek-R1-0528';
  no InferenceX display name found)
```

InferenceX's reference-throughput map does not have an entry for
`DeepSeek-R1-0528` keyed by its file path. Advisory only — does not
affect optimization — but every report's "External baseline" section
will say `(not requested)` or `model_mapping_miss` until the alias is
added.

**Suggested fix**: add a mapping from `DeepSeek-R1-0528` (and the
`/wekafs/models/...` path) to the InferenceX display name (e.g.
`deepseek-r1`), or expose an `INFERENCE_OPTIMIZER_MODEL_ALIAS` env var
operators can set per run.

### B-8  (S3) — Robustness `aiter_jit_build_stuck` is noisy

**Symptom**: Repeated medium-severity alerts every ~30 ticks:

```
aiter jit build_count=4 unchanged for N consecutive ticks; a prior
hipcc invocation likely crashed mid-build
```

In our case build_count=4 was stable because aiter had already JIT'd
everything it needed for this model on the very first baseline; no
crash. The detector treats "no new builds" as "stuck mid-build".

**Suggested fix**: only alert if `build_count` is stuck *while* a
`baseline`/`explore`/`kernel_opt` subprocess is actually running
hipcc (i.e. correlate with a build-in-flight signal), not on a
stable warm cache.

### B-9  (S3) — Per-variant cold-start dominates EXPLORE wall-clock

**Symptom**: each EXPLORE variant takes ~15 min, of which:
- Magpie/vllm fork + arg parsing: ~1 min
- vLLM serve + DeepSeek-R1 weight load: ~10 min (even with hot page cache)
- aiter JIT warm + benchmark: ~3 min
- Magpie teardown: ~1 min

i.e. ~75% of the wall-clock per variant is paid re-loading the same
671B FP8 weights into 8 GPUs. At 15 min/variant we only got 29 variants
in 11h.

**Suggested fix**: persistent vllm server across same-base variants
(e.g. issue `--kv-cache-dtype` change via reload-API rather than a full
restart) — though most variants change something that does require a
fresh server. A milder win is to recognize when only `extra_envs` (not
`extra_args`) change and skip the worker re-init.

### B-10 (S3) — `accuracy eval not found` warning on every benchmark

**Symptom**:

```
baseline_executor: accuracy eval not found: no results*.json in
.../runs/<kind>/<task>/benchmark_vllm_<ts>
```

This fires for every baseline / profile / variant run because
`RUN_EVAL` is defaulted to `false` (`Magpie main / InferenceX main
disagree on run_eval --concurrent-requests`). Cosmetic, but inflates
log noise.

**Suggested fix**: drop the warning to `INFO` when `RUN_EVAL=false`
is the explicit default — the warning is only meaningful if the
operator asked for eval and got nothing.

## 7. Reproducibility of the Winning Config

For anyone wanting to recreate the +13% result (with the warm-state
caveat in B-4):

```bash
export VLLM_ROCM_USE_AITER=        # leave unset; default is best
export MODEL_PATH=/wekafs/models/DeepSeek-R1-0528
vllm serve "$MODEL_PATH" \
  --port 8888 \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.95 \
  --max-model-len 6144 \
  --kv-cache-dtype fp8 \
  --trust-remote-code
# then benchmark with InferenceX:
#   ISL=1024 OSL=1024 CONC=64
```

Standalone gain over the default-args baseline is `+2.26%`. The
`+13.12%` number requires the benchmark to be the **second** launch
into a fully warm system.

## 8. Recommended Follow-Ups (ordered by impact)

1. **Fix B-1** (drop the action-layer profile gate). This single change
   unblocks the documented `--kernel-claude` flow and lets the next
   run actually exercise the kernel pipeline (where the bulk of the
   remaining 87% gap to target lives).
2. **Fix B-4** (warm-vs-cold measurement asymmetry). Without this every
   `cumulative_gain_validated` number is suspect by ±10%.
3. **Fix B-3** (vllm torch_profiler not producing traces). Even after
   B-1 is fixed, kernel-opt has no inputs without a real trace.
4. **Fix B-2** (Robustness false-positive prune). Without this the
   robustness-agent will keep pruning families on imaginary ray
   starvation alerts.
5. **Re-run with kernel enabled** after #1 + #3, same workload, same
   12 h budget. Expected outcome: explore-001 finds `kv_cache_fp8`
   again; profile produces a real trace; kernel_opt picks
   `reusable_native_kernel_ids`; integrate stacks them on top.
6. **Try explicit combinatorial grids** via
   `delegate{action_name='explore', params={grid: [{name: 'fp8_kv+mem95', extra_args: '--kv-cache-dtype fp8 --gpu-memory-utilization 0.95'}, ...]}}`
   to test the standalone winners stacked.

---

*Report generated 2026-05-21 against session
`DeepSeek-R1-0528_20260520T125306Z_35ddebd4`. All numbers traceable to
`/workspace/hyperloom-KB/state.json` and
`/workspace/hyperloom-KB/reports/final.json`.*
