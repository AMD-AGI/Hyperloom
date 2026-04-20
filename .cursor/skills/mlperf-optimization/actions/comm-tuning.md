# Action: Communication Tuning

## Overview

After [`config-selection.md`](config-selection.md) fixes EP/TP/DP, this action
first tunes NCCL/RCCL and DeepEP for that topology, then uses TraceLens CLI overlap
metrics to drive buffer/channel and verification steps when communication is not
fully hidden behind compute. **Prerequisite:** config selection is complete and
the winning parallelism config is applied.

## Inputs

- Winning parallelism config (EP, TP, DP) and baseline ms/iter
- Profile / TraceLens: communication fraction and `state["comm_compute_overlap"]`
- Trace path under `$RESULT_DIR` for overlap verification

Part A runs for all topologies. Pure DP (EP=1, TP=1) still benefits from AllReduce
algorithm/buffer/channel tuning. Part B always runs to verify overlap opportunities.

## KB Query

```bash
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B NCCL DeepEP communication" --top-k 5 --compact
```

## Tuning Reference

### NCCL/RCCL (multi-GPU collectives)

| Variable | Options | Impact |
|----------|---------|--------|
| `NCCL_ALGO` | `Ring`, `Tree` | AllReduce algorithm |
| `RCCL_MSCCL_ENABLE` | `0`, `1` | MSCCL acceleration for ROCm |
| `NCCL_MIN_NCHANNELS` | `4`, `8`, `16` | Minimum communication channels |
| `NCCL_MAX_NCHANNELS` | `8`, `16`, `32` | Maximum communication channels |
| `NCCL_NTHREADS` | `256`, `512` | Threads per NCCL kernel |

### DeepEP (when EP > 1)

| Setting | Values | Notes |
|---------|--------|-------|
| `turbo_deepep_num_cu` | 32, 48, 64, 96, 128 | CUs for DeepEP comm overlap |
| `moe_deepep_num_sms` | 16, 24, 32 | SMs for DeepEP dispatch |

More DeepEP CUs trade compute for overlap—balance with profiling; skip redundant
CU trials in Part B when Part A already yields high `comm_compute_overlap`.

## Procedure

### Part A: NCCL/RCCL parameter tuning

Part A runs for all topologies.

#### Step A1: NCCL algorithm test

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "nccl_ring" 1 "" "NCCL_ALGO=Ring"
run_mlperf_trial "nccl_tree" 1 "" "NCCL_ALGO=Tree"
```

Keep the faster algorithm for later steps.

#### Step A2: MSCCL test

```bash
run_mlperf_trial "msccl_on" 1 "" "RCCL_MSCCL_ENABLE=1"
run_mlperf_trial "msccl_off" 1 "" "RCCL_MSCCL_ENABLE=0"
```

#### Step A3: DeepEP CU allocation (EP > 1 only)

**Only DeepEP CU sweep in this action**—Part B uses these results, no second loop.

```bash
for cu_count in 32 48 64 96 128; do
    # YAML: turbo_deepep_num_cu: $cu_count
    run_mlperf_trial "deepep_cu${cu_count}" 1
done
```

Pick best `turbo_deepep_num_cu` (and `moe_deepep_num_sms` if varied).

#### Step A4: Validate combined comm tuning

```bash
run_mlperf_trial "comm_tuned" 2 500 "NCCL_ALGO=<best> RCCL_MSCCL_ENABLE=<best>"
```

Check loss matches baseline noise (comm-only changes).

### Part B: TraceLens-guided overlap optimization

Part B always runs. When `comm_compute_overlap > 0.7`, expected gains are smaller
but the agent still tests buffer/channel settings to verify. Log measured overlap
as advisory context.

#### Step B1: Overlap gap (when Part B runs)

```python
overlap = state["comm_compute_overlap"]
comm_pct = categories.get("communication", 0)
non_overlapped_ms = (1.0 - overlap) * comm_pct / 100 * ms_per_iter
```

#### Step B2: Gradient AllReduce overlap tuning

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "overlap_nccl_buf16m" 1 "" "NCCL_BUFFSIZE=16777216"
run_mlperf_trial "overlap_nccl_buf64m" 1 "" "NCCL_BUFFSIZE=67108864"
run_mlperf_trial "overlap_nccl_ch16" 1 "" "NCCL_MIN_NCHANNELS=16 NCCL_MAX_NCHANNELS=32"
```

#### Step B3: DeepEP (EP > 1 only)

Do not re-sweep CUs. Keep Part A’s winning DeepEP settings unless TraceLens and
B2 show a clear mismatch (large residual non-overlapped comm).

#### Step B4: Pipeline overlap verification

```bash
run_mlperf_trial "overlap_verify" 1 10
```

```bash
gzip -kf "$RESULT_DIR/overlap_verify_filtered.json" 2>/dev/null || true

mkdir -p "$RESULT_DIR/tracelens_output/overlap_verify/perf_report_csvs"
TraceLens_generate_perf_report_pytorch \
  --profile_json_path "$RESULT_DIR/overlap_verify_filtered.json.gz" \
  --output_csvs_dir "$RESULT_DIR/tracelens_output/overlap_verify/perf_report_csvs" \
  --gpu_arch_json_path /hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/utils/arch/MI355X.json \
  --enable_pseudo_ops

PYTHONPATH="/hyperloom/TraceLens-internal:$PYTHONPATH" \
python3 /hyperloom/TraceLens-internal/TraceLens/AgenticMode/Standalone/orchestrator_prepare.py \
  --trace-path "$RESULT_DIR/overlap_verify_filtered.json.gz" \
  --platform MI355X \
  --output-dir "$RESULT_DIR/tracelens_output/overlap_verify"
```

Keep settings if overlap improves; else revert and log to KB.

#### Step B5: Tier 2 validation

```bash
run_mlperf_trial "overlap_combined" 2 500 "<all_winning_settings>"
```

## Outputs

- Winning NCCL/RCCL / overlap env vars and DeepEP YAML (EP > 1)
- ms/iter delta; comm fraction and overlap before/after; updated
  `state.comm_compute_overlap` when TraceLens re-runs; `state.kept_env_vars`

## Heuristic Update

- NCCL/RCCL gain > 1%: boost remaining comm-tuning candidates
- TraceLens overlap improved: update state.comm_compute_overlap
- No setting improves ms/iter: keep defaults, reduce comm-tuning score by 0.7x (floor 0.5)
- comm-tuning score has a floor of 0.5 — even with no measured gains, it is not removed from the stack

## Failure Handling

- NCCL/RCCL: crash, loss regression, or hang (~5 min) → revert env defaults.
- DeepEP: OOM / instability → reduce CU or SMS; avoid untested CU stacks with
  other big memory changes.
- No gain on ms/iter or overlap → defaults, KB log, lower comm-tuning score.
- If TraceLens CLI not installed or fails: run Part B without overlap measurement; try default buffer/channel grid.
