---
title: profiling — rocprof-compute workflow + the self-contained profiling script
kind: technique
gens: [gfx942, gfx950]
updated: 2026-07-16
---

# rocprof-compute workflow (how to actually get profiling data here)

## TL;DR
[ROCm Compute Profiler](https://github.com/ROCm/rocm-systems) (`rocprof-compute`, formerly Omniperf) is
the kernel-level profiler this methodology uses. It collects **all** relevant hardware counters via
application replay and derives a per-kernel **System Speed-of-Light** (every engine's % of peak), a
**memory chart**, and an empirical **roofline** ([`measure_roofline.md`](measure_roofline.md)). It is
**language-agnostic** — it profiles GPU *dispatches*, so it works the same for Triton, FlyDSL, HIP, ASM.

**Don't invoke `rocprof-compute` by hand — use the self-contained script in this folder:**

```bash
python3 <this_folder>/rocpc_profile.py --driver <driver.py> [--roofline] [--kernel <index>]
```

It runs `profile` + `analyze` for you and prints rocprof-compute's own **Top-Stats + Speed-of-Light**
tables (and, with `--roofline`, the **Roofline** section). Then classify the bottleneck by reading
[`measure_triage.md`](measure_triage.md) and
[`measure_roofline.md`](measure_roofline.md). The script bakes in no kernel name or bottleneck rule — it
just surfaces the profiler's tables; **you** interpret them.
The script invokes the driver only as `<driver.py> --profile-run`; the driver
owns representative-case selection.

To isolate YOUR kernel: run once, find your kernel's row + index in the printed "Top Stats" table, then
re-run with `--kernel <index>` for that kernel's isolated Speed-of-Light.

## The dependency gate — why "unavailable" happens, and how to enable it
A bare `rocprof-compute profile`/`analyze` aborts if its Python deps are missing:

```
[ERROR] The 'dash>=3.0.0' package was not found ...
[ERROR] The 'textual' package was not found ...
Please verify all of the python dependencies ...
```

Cause: the launcher runs an all-or-nothing `verify_deps()` preflight that walks *every* line in its
`requirements.txt` and aborts (`sys.exit(1)`) on the first package that is missing or whose version pin
is unmet. So rocprof-compute needs its full dependency set present to run — shipping only the
`rocprof-compute` binary (as some ROCm images do) is not enough.

How forge handles it — **no per-user configuration**:
1. **Auto-detects a usable interpreter**: the script (and the forge-loop backend) probe the current
   interpreter, the system `/usr/bin/python3`, then `python3` on PATH, and run `rocprof-compute` under
   the first one whose `verify_deps` passes (checked with a fast `rocprof-compute --help`). If **none**
   can, the script prints "unavailable — skipping" and exits 3, and the forge-loop profiler **degrades
   to the PMC path**. No env var, no hand-built venv.
2. **Runs the supported CLI directly** — in a subprocess, so rocprof-compute's `sys.exit`/global state
   stays isolated from the loop. It never patches the shared `/opt/rocm` install.

### Enabling it — install the `forge-profiling` extra (recommended)
Hyperloom, which ships forge, carries rocprof-compute's dependency set as an optional extra, so
installing forge with it drops those deps into forge's OWN interpreter — the first one the profiler
auto-detects. Nothing else to configure:

```bash
pip install -e ".[forge-profiling]"    # docker run line: pip install -e "/path/to/Hyperloom[forge-profiling]"
```

Without the extra, forge stays lean and profiling degrades to the PMC path.

Alternative (without reinstalling forge): install rocprof-compute's requirements into any interpreter
the script probes — e.g. the system python, kept separate from your kernel/torch env:

```bash
/usr/bin/python3 -m pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt
```

Do **not** patch `/opt/rocm`.

## What the two phases produce
- **profile** — replays the driver ~13× to collect all counters into a workload dir
  (`<out>/workloads/run/<gpu>/`, incl. the raw `pmc_perf.csv`). `--roofline` adds a one-time ~70s
  microbench that measures the box's *empirical* peaks (a machine constant) into `roofline.csv`.
- **analyze** — derives metrics; the script requests these blocks and prints them:
  - **block 0 — Top Stats**: the per-kernel time breakdown (find your kernel + its index here).
  - **block 2 — System Speed-of-Light**: `Metric / Value / Peak / Pct of Peak` per engine — VALU/MFMA/
    VMEM utilization, occupancy, IPC, cache hit rates, L2-fabric BW, LDS bank conflicts.
  - **block 4 — Roofline** (only with `--roofline`): arithmetic intensity (AI, flop/byte) + achieved
    vs **empirical peak** per engine (→ how close to the HBM / compute roof).

## Reading the output (short version)
- **Pct of Peak (SoL)** = distance to each hardware ceiling; the highest one is your closest roof.
- **Roofline (AI)** = which roof *fundamentally* binds (AI vs ridge) and how far below it you are.
- If **no** engine is near its ceiling → latency/occupancy-bound, not a throughput wall.
- Full decision flow: [`measure_triage.md`](measure_triage.md); roofline
  detail + per-dtype roofs: [`measure_roofline.md`](measure_roofline.md).
- Pick the compute roof for the kernel's real dtype; FLOP-less kernels (copy/gather) → judge by the BW
  roof, not FLOP/s.

## Pitfalls
- Calling `rocprof-compute` directly and hitting the dependency gate — use `rocpc_profile.py`.
- Profiling a driver that also runs a torch/library reference, then reading the aggregate — isolate
  YOUR kernel with `--kernel <index>` (find it in the Top Stats table) so the numbers are your kernel's.
- Trusting a profiled run's *timing* — profiling perturbs time; measure speed separately
  ([`measure_protocol.md`](measure_protocol.md)).

## Verify
The script exits 0, the Top Stats table lists your kernel (not only torch/`rocblas`/runtime
dispatches), and the raw workload exists under the printed dir.
