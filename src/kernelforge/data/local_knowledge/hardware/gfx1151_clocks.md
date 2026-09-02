---
title: Radeon 8060S / gfx1151 — clocks, thermals, power and measurement hygiene
kind: hardware
topic: clocks
gens: [gfx1151]
updated: 2026-09-02
---

# Clocks, thermals, power and measurement hygiene

Do not encode one fixed Radeon 8060S clock as a performance constant. An integrated GPU shares a
package power/thermal envelope and memory system with the CPU. DVFS, CPU load, skin/cooling state,
memory pressure and display/co-tenant activity can move results.

## What can be observed on the qualified node

Relevant read-only sysfs paths include:

```text
/sys/class/drm/card1/device/gpu_busy_percent
/sys/class/drm/card1/device/mem_busy_percent            # when exposed
/sys/class/drm/card1/device/hwmon/hwmon*/temp1_input
/sys/class/drm/card1/device/hwmon/hwmon*/freq1_input
/sys/class/drm/card1/device/hwmon/hwmon*/power1_average
/sys/class/drm/card1/device/hwmon/hwmon*/power1_cap      # when exposed
```

An **illustrative idle snapshot** at `2026-09-02T02:52:24-05:00` showed:

| Sensor | Value |
|---|---:|
| GPU busy | 0% |
| temperature | 40.0 °C |
| reported frequency | 600 MHz |
| reported average power | ~11.0 W |

These are not specification values or benchmark baselines. They only demonstrate that the sensors
are live and that idle state differs materially from sustained compute state.

## Why clocks move

- GPU and CPU share the package power/thermal budget.
- DVFS raises frequency after work begins and lowers it after demand drops.
- Short kernels may complete before frequency stabilizes.
- Sustained matrix/VALU work can encounter package or thermal limits.
- Memory-bound kernels may not benefit proportionally from higher engine clock.
- CPU load can consume package power and shared-memory bandwidth simultaneously.
- Fan/cooling and ambient state affect sustained behavior.

The PCI-reported link speed is not the model-memory bandwidth clock for this integrated GPU and must
not be used in roofline calculations.

## Measurement protocol

### Before each A/B pair

Record:

- exact source/object/runtime hashes;
- GPU busy and KFD owner state;
- temperature, frequency and power sensors;
- CPU load/governor and competing memory traffic;
- `MemAvailable`, swap and PSI;
- model/request geometry and warm/cold state.

### Warmup

Warm up until:

- JIT/autotune is complete;
- model pages are resident as intended;
- frequency/temperature behavior is representative;
- server route/caches are stable.

Discard cold/JIT/first-request results unless cold-start is the explicit metric.

### Pairing

Use paired/interleaved orders such as ABBA or balanced A→B/B→A blocks. A candidate run hours after a
baseline is not a controlled clock/thermal comparison.

### Report

For every result include:

```text
median/geomean as appropriate
all raw repetitions and spread
sensor range or sampled trace
host wall and device timing
route and dispatch count
thermal/power anomaly notes
```

A throughput change within the sensor/order noise band is not an optimization claim.

## Clock versus bottleneck

| Bottleneck | Expected clock sensitivity |
|---|---|
| WMMA/VALU compute-bound | engine clock can matter strongly |
| LDS/dependency-bound | clock may help, but conflicts/waits remain primary |
| shared-memory bandwidth-bound | memory traffic/channel state dominates |
| launch/host-bound | GPU frequency may barely matter |
| mixed small-model decode | fixed launch + per-byte memory effects both matter |

Classify with measurement rather than assuming every higher-clock result is compute-bound.

## Thermal and co-tenant discipline

- Do not run a second iGPU/GTT workload during a hard A/B.
- A CPU-only memory-intensive process is still a co-tenant on UMA.
- Account for display/compositor traffic when it changes during the run.
- Do not reset the GPU, kill GUI processes or change system-wide power controls without explicit approval.
- If clocks cannot be locked safely, monitor and reject mismatched pairs rather than forcing control.

## What it means for kernels

1. Optimize the real bottleneck; engine clock cannot fix extra bytes or launches.
2. Warm long enough to leave idle DVFS state.
3. Compare candidates in one controlled session with balanced order.
4. Include power/thermal state in reproducibility records.
5. Separate cold start, steady state and sustained long-run behavior.
6. Re-run marginal gains across a second thermal/order block.

## Pitfalls

- Publishing the 600 MHz idle sample as a GPU specification.
- Using a marketing boost clock to compute achieved utilization.
- Comparing a cold baseline to a warm candidate.
- Ignoring CPU package power and memory traffic.
- Interpreting memory-bound throughput as proportional to GPU engine clock.
- “Fixing” variance by mutating global power settings without authorization.
- Hiding raw spread behind one average.

## Verify

A lightweight read-only sampler can poll sysfs around each repetition. Verify sensor units from the
kernel interface (`temp*_input` is typically millidegrees C, `freq*_input` Hz, `power*_average`
microwatts) rather than assuming a userspace presentation format.

For a decisive gate, require comparable sensor ranges and order controls in addition to benchmark
correctness and route proof.

## Sources

- Live amdgpu sysfs/hwmon interfaces on the qualified EVO-X2.
- AMD RDNA3.5 ISA for execution behavior; it does not specify this board's sustained clock.
- Retained paired gfx1151 benchmark methodology/results.

## Related

`gfx1151_overview.md` · `gfx1151_topology.md` · `gfx1151_memory.md` ·
`common_methodology/profiling/measure_protocol.md`
