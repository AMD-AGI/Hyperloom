# RDNA support port — private scoping (pre-PR)

Issue: https://github.com/AMD-AGI/Hyperloom/issues/1196

## Integration points found (all Hyperloom-side; Magpie untouched)

1. `src/hyperloom/inference_optimizer/gpu_types.py`
   - `_AMD_GPU_TYPES` += "r9700" (RDNA4 wave-1 target)
   - `_GFX_TO_RUNNER` += {"gfx1201": "r9700", "gfx1200": "r9700"}
   - `_AMD_GPU_DISPATCH_IDENTITIES` += {"r9700": ("gfx1201", 64)}
   - `_autodetect_gpu_type`: add R9700/RADEON product tags + gfx fallthrough
2. Runner: `custom_{runner_type}.sh` convention in
   `orchestrator/actions/executors/_workload_envs.py` — deliver
   `custom_r9700.sh` wrapping llama-bench/llama-server (tg/pp/PPL gates,
   interleaved A/B, anchor+config emission). No vLLM/SGLang dependency.
3. Quant schemes: `orchestrator/phases/quantization_schemes.py` — add RDNA4
   table (dp4a-universal set; WMMA int8/fp8; NO mxfp4 [gfx950-only],
   NO MFMA; sparse24 available).
4. Profiling guards (TraceLens/Magpie consumers): document + guard
   gfx1201 realities: host_trap-only PC sampling (min interval 512),
   FETCH_SIZE/PMC-derived counters read zero, GL2C ratios-only under
   profile_standard, setperflevel high = -10% throughput trap.
5. Tests: flip `test_profile_and_kernel_handlers.py` gfx1100 expectation
   pattern into positive gfx1201 coverage.

## Evidence base (ours, measured on gfx1201)
- instrument map + measurement protocol: roc10 ledger 2026-08-11..16
- scheme/datapath grid: tools/bench-card isa-grid (17 arches x 32 instrs)
- llama.cpp harness: bench-card provenance + PPL gates + interleaved A/B
