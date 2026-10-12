# TODO: Hyperloom RX 9000 Series & GEAK Adaptation

## Overview

Adapt Hyperloom to support AMD Radeon RX 9000 series cards, beginning with the RX 9070 XT (RDNA4, GFX1201). AMD lists the RX 9070 XT for Windows 11 in ROCm 7.2.0. Runtime/compiler support does not by itself establish Hyperloom workflow support or prove that CDNA tuning assumptions transfer; validate serving, profiling, and kernel optimization on the card.

**Status (2026-10-04)**: RX 9070 XT identity, product-name detection, and GFX1201 routing are implemented. Native Windows Hyperloom operation, RX-specific managed benchmarking, calibrated roofline data, and end-to-end hardware validation remain incomplete. See [the compatibility notes](docs/compatibility.rst#rx-9070-xt).

**Effort estimate**: 7-10 weeks  
**Risk level**: Low-to-medium (high code reuse, architecture-unified)  
**Primary goal**: Enable agentic LLM optimization for 16GB consumer cards (Qwen3-8B, Llama3.1-8B, etc.)

---

## Phase A: GPU Detection & Hardware Profiling (1 week)

### A1: Extend GPU Architecture Detection
**File**: `src/hyperloom/common/gpu_identity.py`

- [x] Add the RX 9070 XT SKU and GFX1201 identity to `AMD_GPU_DISPATCH_IDENTITIES`
  ```python
  AMD_GPU_DISPATCH_IDENTITIES = {
      # gpu_type: (gfx_arch, compute_units)
      "rx9070xt": ("gfx1201", 64),
  }
  ```

- [x] Normalize local and remote product names and resolve the RX SKU before falling back to architecture-only detection
- [ ] Verify product/architecture detection on RX 9070 XT hardware under Windows 11 ROCm 7.2.0
- [ ] Confirm session provenance records `gfx_arch=gfx1201` on a hardware run

### A2: Create Hardware Profile for RX 9070 XT
**File**: `src/hyperloom/orchestrator/kernel/hardware_targets.py` (new or extend)

- [ ] Define `HARDWARE_PROFILES` with parameterized specs for both CDNA and RDNA4
  ```python
  HARDWARE_PROFILES = {
      "gfx942": {  # MI300X
          "name": "mi300x",
          "wmma_capable": True,
          "ai_accelerators": 560,  # Approximate per MI300X
          "compute_units": 228,
          "peak_fp32_tflops": 192,
          "bandwidth_gbs": 5200,
          "cache_l0_kb": 64,  # Per WGP
          "cache_l1_kb": 128,
          "cache_l2_mb": 8,
          "cache_l3_mb": 32,
          "register_file_vgpr_per_wf": 256 * 1024,  # bytes
          "lds_kb": 128,
          "wavefront_size": 64,
          "optimization_strategy": "maximize_memory_throughput",
      },
      "gfx1201": {  # RX 9070 XT
          "name": "rx9070xt",
          "wmma_capable": True,
          "ai_accelerators": 128,  # 2 per CU, 64 CUs
          "compute_units": 64,
          "peak_fp32_tflops": 48.7,
          "bandwidth_gbs": 640,
          "cache_l0_kb": 64,  # Per WGP (same)
          "cache_l1_kb": 128,  # Same
          "cache_l2_mb": 8,   # Same
          "cache_l3_mb": 64,  # Larger Infinity Cache
          "register_file_vgpr_per_wf": 256 * 1024,  # Same per wavefront
          "lds_kb": 128,  # Same
          "wavefront_size": 32,  # RDNA4 supports 32-bit wavefronts
          "optimization_strategy": "maximize_memory_throughput",  # Same strategy
      }
  }
  ```

- [ ] Add validation: reject incompatible configurations for RX 9070 XT (e.g., TP=8 on 16GB)

---

## Phase B: Roofline Model Recalibration (1 week)

### B1: Parameterize Roofline Analysis
**File**: `src/hyperloom/orchestrator/kernel/roofline_ceiling.py` (extend/refactor)

- [ ] Update roofline model to use `HARDWARE_PROFILES` constants
  ```python
  def compute_roofline_ceiling(gfx_arch: str, arithmetic_intensity: float) -> float:
      """
      Compute performance ceiling: min(peak_compute, bandwidth * AI)
      
      Args:
          gfx_arch: GFX ISA identifier (e.g., "gfx942", "gfx1201")
          arithmetic_intensity: Operations per byte (FLOPs / bytes loaded)
      
      Returns:
          Ceiling in GFLOPS
      """
      specs = HARDWARE_PROFILES[gfx_arch]
      peak_compute = specs["peak_fp32_tflops"] * 1e12  # FLOPS
      bandwidth = specs["bandwidth_gbs"] * 1e9  # bytes/sec
      
      # Roofline: min(compute, bandwidth * arithmetic_intensity)
      compute_bound = peak_compute
      bandwidth_bound = bandwidth * arithmetic_intensity
      
      return min(compute_bound, bandwidth_bound)
  ```

- [ ] Verify roofline classification for RX 9070 XT kernels
  - At low arithmetic intensity (<10), RX 9070 XT is memory-bound (same as MI300X)
  - Optimization strategies transfer directly

### B2: Update Profiling Backend
**File**: `src/hyperloom/orchestrator/actions/executors/baseline.py`

- [ ] Ensure baseline measurement logic works for GFX1201
- [ ] Test throughput measurement on RX 9070 XT (Magpie/SGLang/vLLM)
- [ ] Validate roofline analysis generates reasonable bottleneck classifications

---

## Phase C: Kernel Optimization & GEAK Integration (3-4 weeks)

### C1: WMMA Instruction Support (Verify Existing)
**File**: `src/hyperloom/orchestrator/kernel/backends/` (triton, hip, etc.)

- [ ] **Triton backend**: Verify `@triton.jit` kernels compile to GFX1201
  - WMMA instructions (`v_wmma_f32_16x16x16_i8`, etc.) should emit identically
  - Test compile-and-run cycle on RX 9070 XT

- [ ] **HIP backend**: Verify HIP kernels with `__mma_` intrinsics work on GFX1201
  - hipcc should accept `-march=gfx1201` flag
  - Test kernel compilation and execution

- [ ] **FlyDSL backend**: Investigate GFX1201 support (internal AMD tool, may need updates)
  - Determine if FlyDSL already supports GFX1201
  - If not, plan investigation/workaround

- [ ] **Status check**: Confirm GEAK backends already emit WMMA instructions correctly for GFX1201
  - If true, no backend changes needed; only profiling/scaling

### C2: Optimization Pass Compatibility
**File**: `src/hyperloom/orchestrator/kernel/tools/` (GEAK optimization passes)

- [ ] Validate vectorization, matrix instructions, LDS usage, cache tiling, register pressure, and wave scheduling independently on GFX1201
- [ ] Accept a pass only after it compiles, passes correctness checks, and shows a measured benefit; record unsupported techniques and tuning differences

### C3: Conditional Phase Skip for KERNEL_AGENT (Optional Simplification)
**File**: `src/hyperloom/orchestrator/coordinator.py` or phase control logic

- [ ] **Option A (Recommended for MVP)**: Auto-skip KERNEL_AGENT for RX 9070 XT
  ```python
  if gpu_arch == "gfx1201":
      args.no_kernel = True  # Skip kernel optimization phase
      print("KERNEL_AGENT auto-disabled for RX 9070 XT (GEAK support TBD)")
  ```

- [ ] **Option B (Full support)**: Allow KERNEL_AGENT on RX 9070 XT
  - Requires Phase B roofline validation + empirical GEAK tuning
  - Deferred to Phase B2 if needed

---

## Phase D: Memory & Model Constraints (1 week)

### D1: VRAM Guard Rails
**File**: `src/hyperloom/inference_optimizer/cli/preflight.py` or model gate logic

- [ ] Add memory validation for RX 9070 XT (16 GB GDDR6)
  ```python
  MAX_VRAM_GFX1201 = 16 * 1024**3  # 16 GB
  
  # Estimate model size: model_params_billions * bytes_per_param
  # FP8: 1 byte/param, INT8: 1 byte/param
  # BF16: 2 bytes/param, FP16: 2 bytes/param
  
  def validate_model_fits_vram(model_size_gb, gfx_arch, tp_factor=1):
      max_vram = HARDWARE_PROFILES[gfx_arch]["max_vram_gb"]
      required = model_size_gb * tp_factor
      
      if required > max_vram * 0.9:  # Leave 10% headroom
          raise ValueError(
              f"Model requires {required}GB (TP={tp_factor}), "
              f"but {gfx_arch} has {max_vram}GB. "
              f"Try reducing TP or using quantization (INT8/FP8)."
          )
  ```

- [ ] Reject invalid configurations early (preflight, before costly baseline)
- [ ] Recommend TP=1 (no tensor parallelism) for RX 9070 XT by default
- [ ] Recommend precision: INT8 or FP8 (for 7B-13B models on 16GB)

### D2: Update CLI Defaults
**File**: `src/hyperloom/inference_optimizer/cli/parser.py`

- [ ] Add `--gpu-type` choice: `rx9070xt` (in addition to `mi300x`, `mi325x`, `mi355x`)
- [ ] Auto-detect GFX1201 if `--gpu-type` not specified (via `rocminfo`)
- [ ] Set sensible defaults for RX 9070 XT:
  - `--tp 1` (no parallelism)
  - `--precision int8` or `--precision fp8` (memory efficient)
  - `--max-model-len 2048` (conservative for 16GB)
  - `--conc 16` (lower concurrency than MI300X defaults)

---

## Phase E: Knowledge Base & Empirical Validation (2-3 weeks)

### E1: Seed GEAK Knowledge Base for RX 9070 XT
**File**: `src/hyperloom/agents/framework/kb/` (recipe database)

- [ ] Populate initial recipes for RX 9070 XT targeting small models:
  ```python
  RECIPES_RX9070XT = [
      {
          "hardware": "gfx1201",
          "model": "qwen3-8b",
          "precision": "int8",
          "tp": 1,
          "conc": 16,
          "isl": 512,
          "osl": 512,
          "framework": "sglang",
          "throughput_tokens_per_sec": 150,  # Estimated baseline
          "optimization_gains": {
              "wmma_fusion": 1.08,  # 8% speedup from WMMA optimizations
              "cache_tuning": 1.05,  # 5% from cache optimization
          }
      },
      # Add similar entries for llama3.1-8b, etc.
  ]
  ```

- [ ] Run empirical validation campaigns on real RX 9070 XT hardware
  - Baseline: unoptimized Qwen3-8B INT8
  - Apply GEAK optimizations, measure delta
  - Confirm optimization strategies from MI300X transfer to RX 9070 XT
  - Record measured baseline and candidate performance; do not assume MI300X gains transfer

### E2: Empirical Tuning Checklist
- [ ] **Qwen3-8B INT8** on RX 9070 XT
  - Baseline throughput: ____ tokens/sec
  - After GEAK: ____ tokens/sec
  - Gain: ____%

- [ ] **Llama3.1-8B INT8** on RX 9070 XT
  - Baseline throughput: ____ tokens/sec
  - After GEAK: ____ tokens/sec
  - Gain: ____%

- [ ] **Qwen3-14B FP8** on RX 9070 XT (if fits in 16GB)
  - Baseline throughput: ____ tokens/sec
  - After GEAK: ____ tokens/sec
  - Gain: ____%

### E3: Update Documentation
**File**: `docs/compatibility.rst`, `README.md`, new `docs/rx9000-guide.md`

- [ ] Add RX 9070 XT to supported platforms table
- [ ] Document memory constraints and model size recommendations
- [ ] Provide quick-start example for RX 9070 XT
- [ ] Publish expected performance baselines (throughput, power efficiency)

---

## Phase F: Integration Testing & CI (1-2 weeks)

### F1: Unit Test Coverage
**File**: `src/hyperloom/tests/` (new test files for GFX1201)

- [ ] `test_gpu_identity_gfx1201.py`
  - Verify GFX1201 detection from rocminfo
  - Verify hardware profile loading

- [ ] `test_roofline_gfx1201.py`
  - Verify roofline model with GFX1201 constants
  - Confirm arithmetic intensity classification

- [ ] `test_optimization_compatibility_gfx1201.py`
  - Verify optimization passes don't regress on GFX1201
  - Test kernel compilation (Triton, HIP)

### F2: End-to-End Test on Real Hardware
**Prerequisite**: Access to RX 9070 XT

- [ ] Run full optimization pipeline (PRELUDE → OPTIMIZE → SWEEP → CLOSE)
  - Model: Qwen3-8B, precision: INT8
  - Framework: SGLang, TP=1, CONC=16
  - Budget: 3 hours (smoke test)
  - Expected outcome: Baseline + at least one optimized variant

- [ ] Validate session output
  - `state.json` records gfx_arch correctly
  - Performance improvement metrics are reasonable
  - No regressions vs. MI300X pipeline

### F3: CI/CD Updates
**File**: `.github/workflows/tests.yml` (if applicable)

- [ ] Add GFX1201 to matrix of tested targets (if CI has access to RX 9070 XT)
- [ ] If no CI hardware: Document manual testing checklist for maintainers

---

## Phase G: Skills & Demo Updates (1 week)

### G1: Create RX 9000 Series Demo Skills
**File**: `examples/hyperloom-qwen3-8b-rx9070xt/SKILL.md` (new)

- [ ] Copy structure from existing `hyperloom-qwen3-8b-3h/SKILL.md`
- [ ] Adapt for RX 9070 XT constraints:
  - `--tp 1` (always)
  - `--precision int8` or `--precision fp8`
  - `--conc 16` (reasonable for 16GB)
  - `--max-model-len 2048` (conservative window)
  - Budget: 3-4 hours (shorter for consumer hardware)
  - Expect 10-15% optimization gain (vs. 30%+ on MI300X)

### G2: Update Setup & Quickstart
**File**: `examples/README.md`, `docs/how-to/optimize.md`

- [ ] Add RX 9000 series as supported platform in compatibility matrix
- [ ] Include memory/budget guidance for consumer cards
- [ ] Document expected performance (tokens/sec) for reference models

---

## Architecture: Summary of Changes

### Files to Modify:
1. **GPU Detection**: `src/hyperloom/common/gpu_identity.py`
2. **Hardware Profiling**: `src/hyperloom/orchestrator/kernel/hardware_targets.py` (new or extend)
3. **Roofline Analysis**: `src/hyperloom/orchestrator/kernel/roofline_ceiling.py`
4. **CLI & Defaults**: `src/hyperloom/inference_optimizer/cli/parser.py`, `preflight.py`
5. **Knowledge Base**: `src/hyperloom/agents/framework/kb/` (recipes)
6. **Testing**: `src/hyperloom/tests/test_gfx1201_*.py` (new)
7. **Examples**: `examples/hyperloom-qwen3-8b-rx9070xt/SKILL.md` (new)
8. **Documentation**: `docs/`, `README.md`, new `docs/rx9000-guide.md`

### Potentially Reusable After Testing
- Generic optimization orchestration and critic/validation logic
- Kernel backend and tuning behavior require GFX1201 hardware validation

---

## Key Assumptions & Unknowns

AMD's ROCm 7.2.0 Windows documentation lists the RX 9070 XT/GFX1201, and the
PyTorch/HIP and Triton paths have been reported working on this card. Hyperloom
SKU and compiler-target routing is now covered by unit tests. Still unverified
are native Windows operation of Hyperloom's orchestration, the managed serving
runner, GEAK compile-and-run on hardware, calibrated roofline limits, and the
transferability of CDNA optimization recipes. Do not publish predicted gains
until measured on the RX 9070 XT.

---

## Success Criteria

- [ ] **Phase A**: GFX1201 detected correctly from `rocminfo`; hardware profile created
- [ ] **Phase B**: Roofline model parameterized; no hardcoded MI300X assumptions
- [ ] **Phase C**: WMMA kernels compile and run on GFX1201; optimization passes verified
- [ ] **Phase D**: CLI rejects invalid RX 9070 XT configurations; sensible defaults set
- [ ] **Phase E**: Record empirical RX 9070 XT baseline and candidate results
- [ ] **Phase F**: E2E test passes; no regressions on MI300X
- [ ] **Phase G**: RX 9070 XT demo skill works; documentation complete

---

## Future Work (Post-MVP)

- [ ] Multi-GPU RX 9070 XT support (if GPU linking becomes feasible)
- [ ] Training workload support for RX 9000 series (currently inference-only)
- [ ] Larger model support via aggressive quantization (INT4, FP4)
- [ ] Profile-guided optimization for low-power inference (power-perf tradeoffs)
- [ ] Community feedback loop: gather user results, refine recipes

---

## References

- **WMMA on RDNA4**: https://gpuopen.com/learn/using_matrix_core_amd_rdna4/
- **WMMA Guide (RDNA4)**: https://gpuopen.com/learn/wmma-guide-amd-rdna-4-gpus-part-2/
- **RX 9070 XT Specs**: https://www.amd.com/en/products/graphics/desktops/radeon/9000-series/amd-radeon-rx-9070xt.html
- **Hyperloom Docs**: https://rocm.docs.amd.com/projects/hyperloom/en/latest/
- **GEAK Docs**: https://rocm.docs.amd.com/projects/geak/en/latest/

---

## Notes

Runtime support is not equivalent to Hyperloom support. Keep the RX 9070 XT
identified as its own SKU, retain GFX1201 as the compile target, and derive
roofline and optimization guidance from measured RX data rather than CDNA
assumptions.

**Last updated**: 2026-10-04
