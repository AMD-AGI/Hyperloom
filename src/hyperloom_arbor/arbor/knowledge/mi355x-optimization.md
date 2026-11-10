# MI355X Optimization Patterns

## How to find config-tier knobs

Search the framework source code inside the container for env vars that control kernel dispatch, precision, and communication:
```bash
docker exec $CONTAINER grep -r "os.environ\|os.getenv" /sgl-workspace/ --include="*.py" | grep -i "aiter\|cast\|rccl\|nccl\|reduce\|fuse\|quantiz" | head -30
```

Look for env-gated code paths in: AITER dispatch (fused_moe, attention), RCCL/NCCL collective tuning, precision casting, and kernel selection. Reason about each before applying it: `.claude/skills/serving-playbook/knobs.json` says whether a known knob changes numerics, depends on the operating point or belongs to a bundle; then measure, and confirm in the server log that it took effect (unknown env vars are silently ignored).

## Hardware characteristics (gfx950: MI355X at 1400 W, MI350X at 1000 W, the same chip)

- 256 CUs (8 XCDs × 32), wave64; L2 is per XCD, not device-wide; 256 MiB Infinity Cache; LDS 160 KiB/CU, 64 banks
- 288 GB HBM3E at 8 TB/s spec. Achievable is lower; measure it on the node and cite the measurement
- Matrix peaks (spec): BF16 2.5, FP8 5, FP4/FP6 10 PF; no TF32. Use the fp4 roofline for MXFP4 work
- OCP float8_e4m3fn (NOT fnuz like MI300) — dtype mismatch silently corrupts results
- XGMI, 8 GPUs fully connected; measure collective bus bandwidth on the node (RCCL tests)
- Sources, the peak table, ISA and memory detail: `.claude/skills/hardware/` (read its SKILL.md first). The serving rules and the knob catalog: `.claude/skills/serving-playbook/`

## Server and bench lifecycle

The harness owns server start/stop and benchmark execution. The agent should NOT manage the server or run benchmarks — just edit config and report DONE.

## Measurement

- GPU noise can be large, partly host-side. Pin measurement node and audit host tuning.
- Always measure device time (rocprofv3), not wall time — host-dispatch gaps inflate wall time.
- A microbench that doesn't reproduce the e2e delta is not evidence.
