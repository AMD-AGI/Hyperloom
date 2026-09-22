# Device profiles

A device profile is the set of hardware roofs a roofline ceiling is divided by:
peak throughput per instruction path, bandwidth per memory level, and the cost
of one unavoidable kernel dispatch. Profiles are measured once per machine
configuration, reviewed, and committed to
`src/kernelforge/data/roofline_ceiling/device_profiles/`. Nothing measures them
at run time.

That is a deliberate trade. Two campaigns a month apart divide by the same
numbers, no campaign needs a profiler installed, and every figure a ceiling
rests on has been through review rather than being whatever the box reported
that morning. The cost is that a wrong figure stays wrong until someone
notices, which is why the procedure below is worth following exactly and why
`validate_profile` refuses a profile whose figures contradict the datasheet.

## Which profile applies

A profile declares a `match` block and is used only on a machine that satisfies
every field it states:

| Field | Why it is part of the identity |
|:--|:--|
| `arch` | `gfx950`, `gfx942` — different chips, different roofs |
| `device_name` | Marketing name from `rocminfo`, e.g. `AMD Instinct MI355X` |
| `compute_partition` | `SPX` / `CPX` — slicing a card changes what one slice reaches |
| `memory_partition` | `NPS1` / `NPS4` — changes the bandwidth a slice sees |

A field the profile omits is not checked, so a profile may deliberately claim a
whole architecture. A field this machine could not determine never matches a
profile that states one: guessing that an undetected partition is the profile's
is the substitution the identity exists to prevent.

When no profile matches, the ceiling falls back to the vendor datasheet in
`specs.py` and says so. Datasheet peaks are roughly twice what a chip sustains,
so attainment measured against them reads far below what the kernel deserves
and an attainment target will never be reached. That is the intended direction
of failure: a campaign that runs too long is recoverable, one that stops early
on a roof nobody checked is not.

## Adding a machine

### 1. Measure the roofs

```bash
rocprof-compute profile --roof-only --name ceiling --path ./roofs --device 0 -- <a short GPU workload>
```

Do not trust the exit code. The tool exits non-zero on some installations
because its PDF export needs Kaleido, after having written a complete
`roofline.csv`. Find the file instead — it is nested under
`<path>/<name>/<SoC>/` and that layout has moved between releases:

```bash
find ./roofs -name roofline.csv
```

### 2. Read the columns

Four things about `roofline.csv` are easy to get wrong:

- **The first column is the device id.** Drop it before pairing the header with
  a row, or every column name shifts by one.
- **Units are `G`-prefixed.** GFLOP/s, GIOP/s, GB/s. Multiply by `1e9`.
- **Pick the row for the device you profiled**, not the first row.
- **Low-precision matrix column names move between releases.** Some builds
  report a merged `MFMA_FLOPs_F6F4`, others separate `MFMAF4Flops` and
  `MFMAF6Flops`. Check which spelling your CSV uses.

Map columns to instruction paths:

| Instruction path | Column |
|:--|:--|
| `bf16_mfma` | `MFMABF16Flops` — but see the correction below |
| `fp16_mfma` | `MFMAF16Flops` |
| `fp8_mfma`, `mxfp8_scaled_mfma` | `MFMAF8Flops` |
| `fp6_mfma`, `mxfp6_scaled_mfma` | `MFMAF6Flops`, else `MFMA_FLOPs_F6F4` |
| `fp4_mfma`, `mxfp4_scaled_mfma` | `MFMAF4Flops`, else `MFMA_FLOPs_F6F4` |
| `int8_mfma` | `MFMAI8Ops` |
| `fp32_matrix`, `fp64_matrix` | `MFMAF32Flops`, `MFMAF64Flops` |
| `fp16_valu`, `bf16_valu`, `fp32_valu`, `fp64_valu` | `FP16Flops`, `BF16Flops`, `FP32Flops`, `FP64Flops` |

The scaled MXFP paths share their unscaled twin's column because they share the
pipeline: the scaled variants run on the same matrix cores, at the rate set by
the widest operand.

Map bandwidth columns to memory levels: `HBMBw` → `hbm`, `MALLBw` → `mall`,
`L2Bw` → `l2`, `L1Bw` → `l1`, `LDSBw` → `lds`. Record every level the tool
measured, not only HBM — a working set resident in Infinity Cache rides a roof
well above HBM, and one bounded by LDS throughput rides one below it.

### 3. Correct the bf16 halving artifact

rocprofiler-compute reports `MFMABF16Flops` at **exactly half** `MFMAF16Flops`
on architectures that run both MFMA paths at one rate. The knowledge base lists
them as a single entry (`FP16/BF16 2.5 PF` on gfx950, `FP16/BF16 1307 TF` on
gfx942), so a ratio near two is a tool artifact and not a property of the chip.

Use the fp16 measurement for `bf16_mfma` and record the substitution in
`source_by_figure`. Left uncorrected, every bf16 ceiling comes out twice as
loose as it should, which reads as the kernel being twice as close to done as
it is — and under an attainment target, stops the campaign early.

`validate_profile` refuses a profile that still has the two apart, so this is
not a step you can silently skip.

Only substitute where the vendor documents a single rate. Measured throughput
that merely shares a datasheet peak may legitimately differ: gfx950 rates fp4
and fp6 together at 10 PF and a real card measures them 17% apart.

### 4. Measure the dispatch floor

Time a **captured graph**, not eager dispatch. Eager timing also pays the
framework's per-op host cost: on an MI355X that was 4.27 us against 1.55 us for
the same kernel replayed from a graph, so nearly two thirds of the eager figure
is host submission that a graph-timed driver never pays. Charging a stage's
serial dispatches at the eager rate overstates the latency term threefold,
exactly on the small shapes where that term decides the ceiling.

```python
import time, torch

buffer = torch.zeros(1, device="cuda")
for _ in range(500):
    buffer.add_(1.0)
torch.cuda.synchronize()

graph = torch.cuda.CUDAGraph()
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for _ in range(3):
        buffer.add_(1.0)
torch.cuda.current_stream().wait_stream(side)
torch.cuda.synchronize()
with torch.cuda.graph(graph):
    for _ in range(200):
        buffer.add_(1.0)
torch.cuda.synchronize()
for _ in range(10):
    graph.replay()
torch.cuda.synchronize()

samples = []
for _ in range(25):
    torch.cuda.synchronize()
    started = time.perf_counter()
    graph.replay()
    torch.cuda.synchronize()
    samples.append((time.perf_counter() - started) / 200)
print(min(samples))
```

### 5. Write the profile

Name the file after the configuration, e.g.
`gfx950-mi355x-spx-nps1.json`:

```json
{
  "schema_version": 1,
  "match": {
    "arch": "gfx950",
    "device_name": "AMD Instinct MI355X",
    "compute_partition": "SPX",
    "memory_partition": "NPS1"
  },
  "peak_flops": { "bf16_mfma": 1235891500000000.0 },
  "bandwidth": { "hbm": 6281015100000.0 },
  "dispatch_floor_s": 1.5518348664045334e-06,
  "source_by_figure": {
    "bf16_mfma": "rocprof-compute --roof-only, column MFMAF16Flops (substituted for MFMABF16Flops)"
  },
  "measurement": {
    "measured_at": "2026-09-17",
    "rocm_version": "7.2.0",
    "rocprofiler_compute_version": "3.4.0",
    "power_cap_w": 1400.0
  },
  "notes": []
}
```

`source_by_figure` and `measurement` are not decoration. They are how a reader
a year from now decides whether a figure still applies, and the only way a
substitution like the bf16 one stays visible.

Put anything you do not trust in `notes`, and leave the raw figure in place
rather than guessing at a correction. The shipped MI355X profile does this for
`int8_mfma`, which measures at exactly half `fp8_mfma` — the same halving
signature as bf16, but with no vendor statement to justify substituting it.

### 6. Re-measure when the machine changes

Nothing detects a stale profile. Re-measure and commit a new one after a ROCm
upgrade, a partition change, a power-cap change, or a firmware update. The
`measurement` block is what tells you which of those has happened since.
