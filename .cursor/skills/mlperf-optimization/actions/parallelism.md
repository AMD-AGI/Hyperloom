# Action: Parallelism Configuration

## Overview

Explores expert parallelism (EP), tensor parallelism (TP), pipeline parallelism (PP),
and DeepEP configurations. The current config uses EP=1 (all experts on each GPU),
which may not be optimal for 32-expert MoE.

**Caution:** Parallelism changes have **higher crash risk** than fusion flags.
Always verify convergence after each change.

## Inputs
- Current parallelism: `tp=1`, `pp=1`, `ep=1` (from config)
- `GPUS_PER_NODE=8`
- Profile data (communication fraction)

## Parallelism Search Space for 8 GPUs

### Expert Parallelism (Primary)

| EP | Experts/GPU | Communication | Notes |
|----|-------------|---------------|-------|
| 1 | 32 | None | Current: all experts on all GPUs (replicated) |
| 8 | 4 | AlltoAll | Each GPU handles 4 experts. Reduces memory, adds comm |
| 4 | 8 | AlltoAll | Balance between memory and communication |

### Combined Configurations

| Config | TP | PP | EP | DP | Notes |
|--------|----|----|----|----|-------|
| Current | 1 | 1 | 1 | 8 | EP replicated, pure DP |
| EP8 | 1 | 1 | 8 | 8 | Expert parallel, maintains DP=8 |
| TP2+EP4 | 2 | 1 | 4 | 4 | Tensor parallel + expert parallel |

### DeepEP Configuration (when EP>1)

| Setting | Values | Notes |
|---------|--------|-------|
| `moe_enable_deepep` | true/false | Enable DeepEP communication |
| `use_turbo_deepep` | true/false | PrimusTurbo DeepEP kernels |
| `turbo_deepep_num_cu` | 32–128 | CUs for DeepEP (64 default for EP=8) |
| `moe_deepep_num_sms` | 16–32 | SMs for DeepEP dispatch |

## Procedure

### Step 1: Test EP=8 configuration

First modify the YAML config at `$EXP`:
- `expert_model_parallel_size: 8`
- `moe_enable_deepep: true`
- `use_turbo_deepep: true`
- `turbo_deepep_num_cu: 64`
- `moe_token_dispatcher_type: alltoall`

Then run a Tier 1 trial to check for crashes:

```bash
source "$SKILL_ROOT/scripts/common.sh"
run_mlperf_trial "ep8" 1
```

If `TRIAL_STATUS == "ok"`, escalate to Tier 2 for convergence check:

```bash
run_mlperf_trial "ep8_validate" 2 500
```

**If the trial crashes:** Revert the YAML changes immediately and log to KB.

### Step 2: Verify GBS and measure

**CRITICAL:** When changing EP/TP, DP changes too. Verify GBS from MLLOG:

```
GBS = micro_batch_size × data_parallel_size × gradient_accumulation_steps
```

If DP changes, adjust MBS or GA to maintain desired GBS.

### Step 3: Communication tuning (NCCL env vars)

If EP=8 is kept, tune NCCL for AlltoAll:

| Variable | Options | Impact |
|----------|---------|--------|
| `NCCL_ALGO` | `Ring`, `Tree` | AllReduce algorithm |
| `RCCL_MSCCL_ENABLE` | `0`, `1` | MSCCL acceleration |
| `NCCL_MIN_NCHANNELS` | `4`, `8` | Minimum channels |

## Outputs
- Best parallelism config (TP, PP, EP, DP)
- Communication overhead change
- Updated YAML config with parallelism flags

## Failure Handling
- If NCCL init fails: invalid TP/PP/EP combo, skip
- If GBS changes unexpectedly: recalculate MBS/GA
- If OOM: reduce MBS (maintaining GBS via more GA steps)
