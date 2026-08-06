---
name: origami-gemm-fallback
description: Detect AITER a8w8 blockscale shapes that would use the hardcoded CK default and generate an Origami-selected CSV overlay. Use during Hyperloom GEMM tuning after runtime shapes have been captured.
---

# Origami GEMM fallback

Replace only proven `gemm_a8w8_blockscale` config misses with an
Origami-selected CK template. Do not replace an existing CSV decision, including
a row that explicitly selects the same template as AITER's default.

This integration is opt-in. Run it only when
`HYPERLOOM_ORIGAMI_GEMM_FALLBACK=1`. When the variable is unset or false, do
not resolve shapes, create an Origami workspace, launch the selector, inject an
AITER config, or add Origami telemetry.

## Inputs

- A JSON shape source containing `(M, N, K)` records from TraceLens or profile
  capture.
- The active `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` CSV.
- A writable output directory.
- An importable `origami` package and AITER checkout/package containing
  `csrc/ck_gemm_a8w8_blockscale/gemm_a8w8_blockscale_instance.py`.

## Workflow

1. Run the deterministic selector:

   ```bash
   python "$REPO_ROOT/src/hyperloom/agents/kernel/tools/origami_gemm_select.py" \
     --input-json "$INPUT_JSON"
   ```

2. The tool starts as a fresh process so AITER's `lru_cache` and CSV caches
   cannot carry stale dispatch decisions.
3. For each observed shape, call AITER's real `get_CKGEMM_config()`:
   - no row: `dispatch_source=fallback`;
   - non-empty row selecting kernelId 7: `dispatch_source=csv_default_template`;
   - any other non-empty row: `dispatch_source=csv`.
4. Rank only `fallback` shapes over AITER's real CK candidate table. Use:
   - macro-tile and MFMA geometry from the AITER template;
   - occupancy 2 and splitK 0;
   - Origami's default wave and tile mapping;
   - base cache policy, then non-temporal B, then non-temporal A if needed.
5. Benchmark the selected template against AITER's actual default for that
   `(M,N,K)`:
   - selected: `gemm_a8w8_blockscale_tune(..., kernelId, splitK=0)`;
   - default: `gemm_a8w8_blockscale_ck(..., splitK=0, kernelName="")`;
   - warm both kernels before timing;
   - alternate timing order across rounds and use GPU events;
   - compare median per-launch latency;
   - validate the selected output against the default output with the existing
     blockscale mismatch threshold.
6. Add a CSV row only when the selected template is correct and its measured
   median latency is lower than the default's. If it ties, regresses, fails, or
   selects kernelId 7, preserve the default.
7. Emit:
   - `origami_a8w8_blockscale.csv`: partial candidate rows;
   - `origami_a8w8_blockscale_merged.csv`: complete active config plus those
     fallback rows;
   - `origami_a8w8_blockscale_report.json`: provenance and ranking evidence;
   - one JSON result on stdout for the caller.
8. Export the complete merged CSV as
   `AITER_CONFIG_GEMM_A8W8_BLOCKSCALE` before invoking GEAK or Forge.
9. Always continue into the configured GEAK/Forge tuner. Origami is the
   pre-tuning fallback baseline; measured tuner output remains authoritative
   and may replace any Origami-selected row.

Default timing is 3 warmups, 5 alternating rounds, and 10 event-timed launches
per round. Override with `HYPERLOOM_ORIGAMI_BENCHMARK_WARMUP`,
`HYPERLOOM_ORIGAMI_BENCHMARK_ROUNDS`, and
`HYPERLOOM_ORIGAMI_BENCHMARK_ITERATIONS`.

## Safety and fallback

- Do not modify AITER source or its base tuned CSV.
- Do not infer fallback from kernelId 7 or a GPU kernel symbol; both can also be
  selected explicitly by CSV.
- Do not use a learned size curve or prediction threshold as proof of a win;
  make the decision from the direct paired microbenchmark for that exact shape.
- Do not run for blockscale-bpreshuffle, per-token A8W8, FP4, or non-CK paths.
- Do not return Origami as the GEMM tuning backend and do not skip GEAK/Forge
  when an Origami overlay was produced.
- Set `HYPERLOOM_ORIGAMI_GEMM_FALLBACK=1` to enable this pre-tuner. Unset,
  `0`, `false`, `no`, and `off` all disable it.
- If Origami, AITER metadata, hardware discovery, or every feasible candidate
  is unavailable, or the paired benchmark cannot complete, return
  `status=skipped` with a reason and leave dispatch unchanged.
- If `AITER_BYPASS_TUNE_CONFIG` is enabled, skip rather than overriding the
  operator's explicit choice.
- The first profile used to discover shapes may still execute AITER's default.
  This skill changes subsequent launches; intercepting unseen shapes live
  requires an AITER runtime change.
