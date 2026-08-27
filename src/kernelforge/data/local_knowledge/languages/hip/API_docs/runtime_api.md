---
title: HIP runtime API — memory, streams, events, graphs, error handling
kind: api_reference
gens: [gfx942, gfx950]
regimes: [both]
status: sota
updated: 2026-07-09
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/index.html
  - https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api.html
---

# HIP runtime API reference

Host-side API to allocate memory, move data, launch/sequence work, and check errors. This is the driver
surface a test-driver / harness uses; the kernel-language surface is in
[kernel_language.md](kernel_language.md).

## Memory management
```cpp
hipMalloc(&d, bytes);                         // device memory
hipMallocManaged(&d, bytes);                  // unified (migratable) memory
hipHostMalloc(&h, bytes, hipHostMallocDefault);   // pinned host → true async DMA
hipMemcpy(dst, src, bytes, kind);             // sync; kind = hipMemcpyHostToDevice / DeviceToHost / DeviceToDevice
hipMemcpyAsync(dst, src, bytes, kind, stream);
hipMemcpyPeerAsync(dst, dstDev, src, srcDev, bytes, stream);  // P2P
hipMemset(d, value, bytes);  hipMemsetAsync(d, value, bytes, stream);
hipFree(d);  hipHostFree(h);
```
Pinned host memory (`hipHostMalloc`) is required for real copy/compute overlap. Prefer 128-bit-aligned
allocations for vectorized loads (see hip_levers).

## Streams (ordering + concurrency)
```cpp
hipStream_t s;
hipStreamCreate(&s);
hipStreamCreateWithFlags(&s, hipStreamNonBlocking);   // don't serialize with the default stream
hipStreamSynchronize(s);       // wait for all work in s
hipStreamWaitEvent(s, ev, 0);  // s waits until ev completes (cross-stream dependency)
hipStreamDestroy(s);
```
Overlap copy/compute by issuing them on **separate** streams; sequence with events. Multi-GPU: prefer one
process per GPU; `GPU_MAX_HW_QUEUES=2`.

## Events (timing + dependencies)
```cpp
hipEvent_t e0, e1; hipEventCreate(&e0); hipEventCreate(&e1);
hipEventRecord(e0, s); /* ... */ hipEventRecord(e1, s);
hipEventSynchronize(e1);
float ms; hipEventElapsedTime(&ms, e0, e1);   // device-side timing
```
For benchmarking discipline (warmup, median, in-context) see
`local_knowledge/common_methodology/` and the profiling skill.

## HIP graphs (kill per-launch overhead in decode loops)
```cpp
hipStreamBeginCapture(s, hipStreamCaptureModeGlobal);
/* issue the kernel/copy sequence on s */
hipGraph_t graph; hipStreamEndCapture(s, &graph);
hipGraphExec_t exec; hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
hipGraphLaunch(exec, s);       // replay with ~zero launch overhead
```
Graphs need **fully static shapes and no host syncs** in the captured region (no `.item()` / GPU→CPU in
the loop) — a common capture failure on dynamic decode paths.

## Occupancy & device query
```cpp
int blocks; hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (void*)kernel, blockThreads, dynSmem);
hipDeviceProp_t p; hipGetDeviceProperties(&p, dev);   // gcnArchName ("gfx942"/"gfx950"), CU count, LDS, warpSize
```

## Error handling (never skip)
```cpp
hipError_t err = hipGetLastError();
if (err != hipSuccess) fprintf(stderr, "%s\n", hipGetErrorString(err));
// After every async launch: check hipGetLastError(); after sync points: check the returned hipError_t.
#define HIP_CHECK(x) do{ hipError_t e=(x); if(e){ /* log hipGetErrorString(e) */ } }while(0)
```
A kernel launch failure is reported asynchronously — check `hipGetLastError()` after launch AND a
`hipDeviceSynchronize()`/`hipStreamSynchronize()` to catch device-side faults.

## Sources
- HIP runtime API (memory, stream, event, graph, error): https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/index.html
- HIP runtime how-to (streams, graphs, cooperative launch): https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api.html
