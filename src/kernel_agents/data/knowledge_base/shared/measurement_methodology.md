# Measurement Methodology

## SNR (Signal-to-Noise Ratio) Correctness Gate
```
SNR = 20 * log10(||ref|| / ||out - ref||)
```
- fp32 reference: SNR ≥ 30 dB required
- fp16/bf16: SNR ≥ 25 dB acceptable
- SNR < 0: output is wrong (anti-correlated with reference)
- Always test with small shapes first (e.g., [1,1,128]) before scaling up

## Cosine Similarity Gate (alternative for attention)
```python
cos = F.cosine_similarity(ref.float().flatten().unsqueeze(0),
                          out.float().flatten().unsqueeze(0))
```
- cos ≥ 0.9999: pass (used for VSA sparse attention fwd)
- For large shapes where fp32 reference OOMs, check NaN/Inf absence instead
- Multi-seed stability: run 10+ seeds, all must pass the gate

## Wall-Clock Benchmarking
```python
torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(30):
    result = kernel(...)  # IN-CONTEXT, not isolated
torch.cuda.synchronize()
wall_ms = (t0 - time.perf_counter()) / 30 * 1000
```

### Critical Rules
1. Use IN-CONTEXT measurement (full pipeline), not isolated kernel calls
   - Isolated calls inflate inter-stage overhead by 3-10×
2. Minimum 30-iteration median to reduce noise
3. 10+ warmup iterations to fill caches
4. torch.cuda.synchronize() before AND after timing loop
5. Use median, not mean (outlier-resistant)

## Decomposition by Subtraction
To measure component X in a chain A→B→C:
1. Measure A+B+C total
2. Measure A+B (without C)
3. X_time = total - (A+B)

NEVER trust isolated component benchmarks — cache/occupancy effects change results.

## PMC Profiling (rocprofv3)
```bash
rocprofv3 --pmc SQ_INSTS_VALU_MFMA_BF16 SQ_INSTS_VMEM \
          SQ_WAIT_INST_LDS SQ_WAIT_INST_ANY \
     -d <outdir> --output-format csv -- python driver.py
```

### Counter Interpretation
- `SQ_WAIT_INST_ANY / SQ_INSTS_VALU_MFMA_*` ratio:
  - < 5: compute-bound (MFMA well utilized)
  - 5-10: balanced
  - > 10: memory-bound (optimize data movement)
- ALWAYS collect PMC BEFORE proposing structural changes
- ALWAYS predict expected counter impact BEFORE rebuilding

## Anti-Patterns
- Speculating about speedups without measurement
- Rebuilding without predicting PMC impact
- Trusting cross-backend chain measurements (instruction cache eviction at backend boundaries)
- Chasing isolated bench improvements that don't transfer in-context
