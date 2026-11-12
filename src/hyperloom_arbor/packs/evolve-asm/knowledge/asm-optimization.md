# ASM Kernel Optimization — evolve-asm pack

## Goal
Push any kernel to 85% of MAF roofline via pure hand-tuned AMDGCN ASM.
Languages (FlyDSL, Triton, CK, HIP) are scaffolding to generate a better
starting `.amdgcn`. Hand-tuned ASM is always the end state.

## MAF Roofline — gfx950 / MI355X

| Metric | Peak | MAF (achievable) |
|--------|------|-----------------|
| FP4 (MXFP4) GEMM | 10 PFLOPS | **5 PFLOPS** |
| FP8 GEMM | 5 PFLOPS | **2.5 PFLOPS** |
| BF16 GEMM | 2.5 PFLOPS | **1.25 PFLOPS** |
| HBM bandwidth | 8 TB/s | **6.4 TB/s** |

**Always use MAF (50% of FLOP peak, 80% of BW peak), never vendor peak.**
A kernel at half of MAF has real headroom. A kernel at 85%+ MAF is done.

## Tier ladder (enforced by orchestrator, not suggestions)

```
TIER 1 — native lang:    exhaust compile params (FlyDSL k_wave/tile, Triton BK/BV/stages)
TIER 2 — peer lang:      compile better starting .amdgcn (FlyDSL for GEMM/MoE, Triton for attention)
TIER 3 — hand-tuned ASM: edit .s directly with rocprofv3 PMC guidance
TIER 4 — cross-kernel:   apply techniques from campaign_archive/ that worked elsewhere
TIER 5 — algorithmic:    fundamentally different algorithm → new .amdgcn baseline → back to TIER 1
```

Null-streak thresholds: 3 (native), 5 (all others). On threshold → orchestrator escalates.

## ASM round-trip workflow

```bash
LLVM=$(find /opt/rocm* -name llvm-mc 2>/dev/null | head -1 | xargs dirname)

# 1. Disassemble ref .co → .s
$LLVM/../lib/llvm/bin/llvm-objdump -d --mcpu=gfx950 ref.co | \
  grep -A999999 '<.text>:' | tail -n+2 > working.s

# 2. Validate round-trip FIRST (cos_sim must be 1.000000 before ANY edit)
python3 /arbor/packs/evolve-asm/validator/patch_co.py \
  ref.co working.s /tmp/cand.co --llvm-mc $LLVM/llvm-mc --mcpu gfx950

# 3. Edit working.s (one change)

# 4. Assemble + patch
python3 /arbor/packs/evolve-asm/validator/patch_co.py \
  ref.co working.s /tmp/cand.co --llvm-mc $LLVM/llvm-mc --mcpu gfx950

# 5. Correctness check via validator/correctness.sh (SNR >= threshold)

# 6. Bench via validator/bench.sh — harness reads gpu_us: lines
```

## PMC counters (rocprofv3)

```bash
rocprofv3 --pmc SQ_INSTS_VMEM SQ_WAIT_INST_ANY SQ_INSTS_LDS \
              SQ_ACTIVE_INST_ANY TCP_GATE_EN1 -- python3 bench_driver.py
```

Key ratios:
- `WAIT/VMEM > 10x` → VMEM latency stall → loosen `s_waitcnt`, software prefetch
- `LDS/VMEM > 5x`  → LDS bank conflict → padding, access pattern change
- `ACTIVE/WAIT < 2x` → compute-bound → MFMA scheduling, accumulator feeding

## Required output after every bench

```
gpu_us: XX.X           ← one line per run (harness gates on weakest)
sig_vgpr: <n>          ← for MAP-Elites structural diversity tracking
sig_barriers: <n>
sig_spills: <n>
sig_mfma: <n>
```

## Integration — aiter CSV deployment

Winning tile configs are written to the CSV the bench driver's `--deploy-csv` names (its default is
in `validator/bench_moe_fp8fp4.py`).

Use `--deploy --config-name <config> --tile-config <tile>` with the bench driver.
If your setup bind-mounts the CSV from a git repository on the host, its changes are tracked there.
