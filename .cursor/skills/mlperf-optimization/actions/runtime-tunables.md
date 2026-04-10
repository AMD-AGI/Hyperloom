# Action: Runtime Tunables

## Overview

System-level tuning that doesn't modify the training config. These are safe to apply
and usually provide a consistent baseline improvement.

## Inputs
- System access level (sudo available?)
- Current system configuration

## Tunable Matrix

### CPU/Memory (requires sudo)

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

### ROCm/GPU Environment

| Variable | Current | Options | Impact | Notes |
|----------|---------|---------|--------|-------|
| `HIP_FORCE_DEV_KERNARG` | 1 | 0/1 | Low | Force device kernarg allocation |
| `HSA_FORCE_FINE_GRAIN_PCIE` | 1 | 0/1 | Low | Fine-grain PCIe access |
| `HSA_KERNARG_POOL_SIZE` | 12582912 | 12M–64M | Low | Kernel argument pool size |
| `TORCH_NCCL_HIGH_PRIORITY` | 1 | 0/1 | Low | High-priority NCCL streams |
| `ENABLE_NUMA_BINDING` | 1 | 0/1 | Low–Med | Bind GPUs to NUMA nodes |

### NCCL/RCCL Tuning

| Variable | Values | Impact | Notes |
|----------|--------|--------|-------|
| `NCCL_BUFFSIZE` | 4M–128M | Low | NCCL buffer size |
| `NCCL_MIN_NCHANNELS` | 4–16 | Low | Minimum comm channels |
| `RCCL_MSCCL_ENABLE` | 0/1 | Low–Med | MSCCL acceleration |
| `NCCL_NET_GDR_LEVEL` | 0–5 | Low | GPUDirect RDMA level |

## Procedure

### Step 1: Apply system tunables

```bash
if [ -w /proc/sys/vm/drop_caches ]; then
    bash runtime_tunables.sh
fi
```

### Step 2: Test NCCL tuning (Tier 1)

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "nccl_tuned" 1 15 "NCCL_BUFFSIZE=16777216 RCCL_MSCCL_ENABLE=1"

eval "$(parse_trial_result "$(grep TRIAL_RESULT $RESULT_DIR/attempt_nccl_tuned.log)")"
gain=$(compute_gain_pct "$baseline_ms_per_iter" "$TRIAL_MS_PER_ITER")
```

### Step 3: Test HSA pool size increase (Tier 1)

```bash
run_mlperf_trial "hsa_pool_64m" 1 15 "HSA_KERNARG_POOL_SIZE=67108864"
```

## Outputs
- Applied system tunables
- NCCL/RCCL optimal settings
- Environment variable overrides

## Failure Handling
- If sudo not available: skip CPU/memory tunables
- If NCCL tuning causes hangs: revert to defaults
