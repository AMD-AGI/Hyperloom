# Action: Kernel Integration

## Overview

Integrates a GEAK-optimized, OOB-Claude-optimized, OOB-Codex-optimized, or manually-optimized
kernel into the MLPerf training stack and benchmarks it as a normal optimization attempt.

## Inputs

- Optimized kernel source (from GEAK, OOB-Claude, OOB-Codex, or manual optimization)
- Original kernel location (path in training stack)
- Current kept_overrides and kept_patches

## KB Query

```
python3 $SKILL_ROOT/kb/kb_query.py "GPT-OSS-20B kernel integration" --top-k 5 --compact
```

## Procedure

### Step 1: Backup original

```bash
cp "$ORIGINAL_KERNEL_PATH" "${ORIGINAL_KERNEL_PATH}.bak"
```

### Step 2: Choose integration path

**Path A: Source file patch** (primary for distributed training)
Replace the kernel function in the Primus/Megatron source file.

**Path B: Monkey-patch at import** (least invasive)
Add a monkey-patch module that overrides the kernel at import time.

### Step 3: Benchmark with Tier 1 trial

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "kernel_NAME" 1
eval "$(parse_trial_result "$(grep TRIAL_RESULT $RESULT_DIR/attempt_kernel_NAME.log)")"
gain=$(compute_gain_pct "$baseline_ms_per_iter" "$TRIAL_MS_PER_ITER")
```

If gain > 1%, validate with Tier 2:

```bash
run_mlperf_trial "kernel_NAME_validate" 2 500
```

### Step 4: KEEP or REVERT

- **If gain > 1% confirmed by Tier 2:** KEEP. Add to `kept_patches`.
- **If `TRIAL_STATUS` is `nan` or `no_data`:** REVERT immediately.
- **If same or worse:** REVERT: `cp "${ORIGINAL_KERNEL_PATH}.bak" "$ORIGINAL_KERNEL_PATH"`
- **If crashes:** REVERT and log crash.

### Step 5: Post-integration re-profile (if kept)

Push a re-profile trigger ([`profile.md § Re-Profile Trigger`](profile.md#re-profile-trigger))
to discover if the optimization exposed new bottlenecks.

## Outputs
- `actual_e2e_pct`: actual end-to-end speedup percentage
- KEEP/REVERT decision
- Backup files for revert

## Heuristic Update

- Kept kernel (gain > 1%): push re-profile, boost kernel-opt for similar types
- Reverted kernel: reduce this kernel's score by 0.7x (floor at 0.5)

## Failure Handling

- If patch breaks import: revert from backup
- If training hangs: kill after timeout, revert
- If numerical differences in loss: revert (kernel not functionally correct)
