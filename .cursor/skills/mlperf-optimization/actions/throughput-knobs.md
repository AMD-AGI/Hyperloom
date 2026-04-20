# Action: Throughput Knobs (Fusion Flags / Training Params / Runtime Tunables)

## Overview

Three throughput-only DFS actions share the same KEEP/DISCARD logic and Tier 1 entry
point. They are consolidated here with anchored subsections:

- [Fusion Flags](#fusion-flags) — swap generic kernels for fused, hardware-optimized
  alternatives. Highest-impact action for MoE models.
- [Training Params](#training-params) — per-iteration tuning (MBS, recompute, turbo
  params) that keeps GBS fixed.
- [Runtime Tunables](#runtime-tunables) — OS / ROCm / NCCL knobs that do not touch the
  training config.

All three stay in the DFS as distinct scoring keys (`fusion-flags`, `params`,
`runtime-tunables`). Only the markdown surface is shared; scoring, classification
(`throughput-only`), and heuristic updates remain action-specific.

## Shared Procedure Template

Every knob follows the same four-step flow. The subsections below only supply the
matrix of knobs and any action-specific notes.

### Step 1 — Apply the knob

YAML edit (fusion flags, training params) **or** env override passed to
`run_mlperf_trial` as the fourth argument (runtime tunables, env-flag fusion).

### Step 2 — Tier 1 trial + TRIAL_RESULT parse

```bash
source "$SKILL_ROOT/scripts/common.sh"

run_mlperf_trial "<label>" 1 "" "<extra_env>"   # extra_env empty for YAML knobs

eval "$(parse_trial_result "$(grep TRIAL_RESULT $RESULT_DIR/attempt_<label>.log)")"
gain=$(compute_gain_pct "$baseline_ms_per_iter" "$TRIAL_MS_PER_ITER")
```

### Step 3 — Gain > 1% triggers Tier 2 validation

```bash
run_mlperf_trial "<label>_validate" 2 500
```

Confirm `TRIAL_STATUS != "nan"` and the loss trajectory is stable before committing.

### Step 4 — KEEP / DISCARD

- **Improved (gain > 1%, Tier 2 confirms):** KEEP — update YAML (or export env) permanently.
- **Marginal (0–1%):** KEEP tentatively, verify again in the combined test at the end
  of the sub-action.
- **Same or worse:** DISCARD — revert.
- **Crash / NaN:** log the error, mark as `crash` / `nan`, revert, record in KB.

### Shared Failure Handling

- Flag/param crashes → mark `crash`, revert, move to next candidate.
- Loss divergence on validation → mark `convergence_fail`, revert.
- All candidates in a sub-action fail → reduce that sub-action's score by 0.7×
  (Score Update Rule #2) and move to comm-tuning or kernel-opt.

---

## Fusion Flags {#fusion-flags}

Fusion flags replace generic PyTorch kernels with fused, hardware-optimized
alternatives. Each flag is a single YAML override — zero code changes, zero crash
risk. This is the **highest-impact action for MoE models**.

### Inputs

- Baseline ms/iter and kept_overrides from prior attempts
- Model class from classify step
- Profile data (which kernels dominate)
- Current YAML config state

### KB Query

```
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B fusion flags MoE" --top-k 5 --compact
```

### Pre-Check: Already-Enabled Flags

GPT-OSS-20B config already enables several fusion flags by default. Check before testing:

```python
already_enabled = {
    "moe_permute_fusion": True,         # Already in config
    "gradient_accumulation_fusion": True, # Already in config
    "apply_rope_fusion": True,           # Already in config
    "moe_grouped_gemm": True,            # Already in config
    "moe_router_fusion": True,           # Already in config
    "cross_entropy_loss_fusion": True,    # Already in config
}
```

### Fusion Flag Matrix

Test flags NOT already enabled:

#### Tier 1: High-confidence flags (try first)

| Flag | Expected Impact | Notes |
|------|----------------|-------|
| `moe_use_fused_router_with_aux_score=true` | ~0–0.5% | Currently `false` in config. Fused TopK router |
| `use_turbo_grouped_mlp=true` | -1% to +1% | Currently `false`. Fused SwiGLU for MoE |
| `use_turbo_attention=true` | +0–1% | Currently `false`. PrimusTurbo attention |
| `moe_shared_expert_overlap=true` | +0–1% | Overlap shared expert with routing |

#### Tier 2: DeepEP and sync-free MoE

| Flag | Expected Impact | Notes |
|------|----------------|-------|
| `moe_enable_deepep=true` + `use_turbo_deepep=true` | +2–5% | Enable DeepEP for expert comm |
| `turbo_sync_free_moe_stage=2` | +1–3% | Sync-free MoE pipeline (requires legacy grouped gemm) |
| `turbo_sync_free_moe_stage=3` | +2–4% | Full sync-free (more memory) |

#### Tier 3: TE/FP8 specific

| Flag | Expected Impact | Notes |
|------|----------------|-------|
| `NVTE_USE_CAST_TRANSPOSE_TRITON=1` | **+0.2% (confirmed)** | Triton-based FP8 cast+transpose. Known prior result — apply early. |
| `NVTE_ROCM_ENABLE_MXFP8=1` | Variable | MX-FP8 (newer format, not all kernels support) |

#### Flags to AVOID

| Flag | Why |
|------|-----|
| Changing `window_size` or `window_attn_skip_freq` | Affects model quality |
| `moe_use_legacy_grouped_gemm=false` | Required for current EP=1 config |

### Fusion Flags — combined test

After testing all flags individually, test all KEPT flags together with Tier 2:

```bash
# Apply all winning flags to YAML
run_mlperf_trial "fusion_combined" 2 500
```

If the combined result is better than the individual best, the combination becomes
the new baseline.

### Fusion Flags — heuristic update

- Individual flag gain > 1%: boost remaining untested flags by 1.5×.
- All flags neutral/negative: reduce `fusion-flags` score by 0.7×.
- After 2+ wins: push combined fusion test (Score Update Rule #3).
- After all flags tested: push re-profile — see [profile.md § Re-Profile Trigger](profile.md#re-profile-trigger) (Score Update Rule #4).

### Fusion Flags — outputs

- `winning_flags`: list of flags that improved ms/iter
- Combined gain percentage
- Updated YAML config with winning flags
- KB entries for each tested flag

---

## Training Params {#training-params}

Training configuration parameters that affect per-iteration performance without
changing hyperparameters or parallelism. FP8 knobs live in
[fp8-recipe-tuning.md](fp8-recipe-tuning.md); gradient clipping lives in
[convergence-speed.md](convergence-speed.md).

### Inputs

- Current config and kept_overrides
- Profile data

### KB Query

```
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B training parameters MBS recompute" --top-k 5 --compact
```

### Parameter Matrix

#### Memory / Batch Parameters

| Parameter | Range | Impact | Notes |
|-----------|-------|--------|-------|
| `micro_batch_size` | 1–8 | Medium | Larger MBS = fewer GA steps. Must maintain GBS |
| `recompute_granularity` | none/selective/full | Medium | Trades compute for memory |

#### Communication Overlap

| Parameter | Current | Try | Notes |
|-----------|---------|-----|-------|
| `overlap_grad_reduce` | true | — | Already enabled |
| `overlap_param_gather` | true | — | Already enabled |
| `use_distributed_optimizer` | true | — | Already enabled |

#### PrimusTurbo Parameters

| Parameter | Current | Range | Notes |
|-----------|---------|-------|-------|
| `enable_primus_turbo` | true | — | Master switch, already on |
| `turbo_sync_free_moe_stage` | 0 | 0–3 | Sync-free MoE pipeline |
| `turbo_deepep_num_cu` | 64 | 32–128 | CUs for DeepEP |

### Training Params — prioritization

```python
param_priority = []

# MBS tuning (if GA > 2)
if ga_steps > 2 and mbs < 4:
    param_priority.append(("micro_batch_size", 4, "Reduce GA overhead"))

# Sync-free MoE
if turbo_sync_free_moe_stage == 0:
    param_priority.append(("turbo_sync_free_moe_stage", 2, "Enable sync-free MoE"))
```

### Training Params — GBS verification for MBS changes

When changing MBS, verify GBS is maintained:

```
new_ga = GBS / (new_mbs × dp)
assert new_ga * new_mbs * dp == baseline_gbs
```

### Training Params — heuristic update

- MBS improvement: boost remaining MBS candidates.
- Sync-free MoE gain > 1%: boost `turbo_sync_free_moe_stage=3` candidate.
- All params neutral: reduce `params` score by 0.7×.

### Training Params — failure-handling delta

- OOM on larger MBS: revert, try intermediate value.
- Loss divergence: revert to prior config.

### Training Params — outputs

- `winning_params`: list of parameters that improved ms/iter
- Per-parameter gain percentages
- Updated config

---

## Runtime Tunables {#runtime-tunables}

System-level tuning that doesn't modify the training config. These are safe to
apply and usually provide a consistent baseline improvement.

### Inputs

- System access level (sudo available?)
- Current system configuration

### KB Query

```
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B runtime tunables NCCL" --top-k 5 --compact
```

### Tunable Matrix

#### CPU/Memory (requires sudo)

| Tunable | Command | Impact | Notes |
|---------|---------|--------|-------|
| Drop caches | `echo 3 > /proc/sys/vm/drop_caches` | Low | Clear page cache before run |
| CPU governor | `cpupower frequency-set -g performance` | Low–Med | Max CPU frequency |
| CPU idle | `cpupower idle-set -d 2` | Low | Disable deep idle states |
| NMI watchdog | `echo 0 > /proc/sys/kernel/nmi_watchdog` | Low | Reduce interrupts |
| NUMA balancing | `echo 0 > /proc/sys/kernel/numa_balancing` | Low–Med | Disable auto NUMA migration |
| ASLR | `echo 0 > /proc/sys/kernel/randomize_va_space` | Low | Reduce TLB misses |
| Transparent hugepages | `echo 'always' > .../transparent_hugepage/enabled` | Low–Med | Better memory allocation |
| THP defrag | `echo 'always' > .../transparent_hugepage/defrag` | Low | Defragment for hugepages |

#### ROCm/GPU Environment

| Variable | Current | Options | Impact | Notes |
|----------|---------|---------|--------|-------|
| `HIP_FORCE_DEV_KERNARG` | 1 | 0/1 | Low | Force device kernarg allocation |
| `HSA_FORCE_FINE_GRAIN_PCIE` | 1 | 0/1 | Low | Fine-grain PCIe access |
| `HSA_KERNARG_POOL_SIZE` | 12582912 | 12M–64M | Low | Kernel argument pool size |
| `TORCH_NCCL_HIGH_PRIORITY` | 1 | 0/1 | Low | High-priority NCCL streams |
| `ENABLE_NUMA_BINDING` | 1 | 0/1 | Low–Med | Bind GPUs to NUMA nodes |

#### NCCL/RCCL Tuning

| Variable | Values | Impact | Notes |
|----------|--------|--------|-------|
| `NCCL_BUFFSIZE` | 4M–128M | Low | NCCL buffer size |
| `NCCL_MIN_NCHANNELS` | 4–16 | Low | Minimum comm channels |
| `RCCL_MSCCL_ENABLE` | 0/1 | Low–Med | MSCCL acceleration |
| `NCCL_NET_GDR_LEVEL` | 0–5 | Low | GPUDirect RDMA level |

### Runtime Tunables — system-level setup

```bash
if [ -w /proc/sys/vm/drop_caches ]; then
    bash runtime_tunables.sh
fi
```

### Runtime Tunables — example trials

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "nccl_tuned" 1 "" "NCCL_BUFFSIZE=16777216 RCCL_MSCCL_ENABLE=1"

run_mlperf_trial "hsa_pool_64m" 1 "" "HSA_KERNARG_POOL_SIZE=67108864"
```

### Runtime Tunables — heuristic update

- Large gains (>5%): boost similar system-level tuning scores.
- All tunables < 1%: proceed to next action category.

### Runtime Tunables — failure-handling delta

- If sudo not available: skip CPU/memory tunables.
- If NCCL tuning causes hangs: revert to defaults.

### Runtime Tunables — outputs

- Applied system tunables
- NCCL/RCCL optimal settings
- Environment variable overrides
