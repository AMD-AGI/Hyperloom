# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""HIP sources for the on-node hardware probes.

Embedded as strings rather than shipped as data files so the probes travel with
the wheel and need no ``package_data`` wiring.

Two probes, answering the two terms of the roofline:

``MFMA_PROBE_SRC``
    Matrix-core issue rate in FLOPs/clock/CU, per precision. Back-to-back MFMA
    with enough independent accumulator chains to cover instruction latency puts
    the matrix core at its issue limit, so the achieved rate *is* the
    architectural rate. No library kernel is involved, so kernel quality cannot
    leak into the number -- which is what keeps the resulting compute roof a
    true upper bound rather than something a tuned kernel can beat.

``BANDWIDTH_PROBE_SRC``
    Absolute achievable streaming-read bandwidth in GB/s. Reported as an
    absolute figure rather than a fraction of a theoretical peak, because the
    theoretical peak is not reliably derivable at runtime: ``hipDeviceProp_t``
    reports MI355X as 8192-bit at 2000 MHz, which yields 4096 GB/s under the
    usual double-data-rate formula against an actual 8000 GB/s, since HBM3E
    clocks its pins at four times the reported rate and that multiplier moves
    with the HBM generation.

Both are self-describing on stdout as JSON lines so the caller never has to
guess which variants a given architecture supports.
"""

from __future__ import annotations

#: Independent accumulator chains per thread. MFMA has multi-cycle latency, so
#: a single dependent chain measures latency rather than issue rate; eight
#: chains is comfortably past the point where the pipeline stays full.
MFMA_ACCUMULATORS = 8

#: Matrix-core issue-rate probe.
#:
#: Every candidate instruction is wrapped in ``#if __has_builtin``, which is only
#: meaningful in the *device* compilation pass -- the AMDGCN builtins are
#: invisible to the host pass, where the same guard always reports false. So the
#: kernel *signatures* are unconditional (the host pass needs the symbols in
#: order to launch them) while only their *bodies* are guarded, and a small
#: availability kernel reports back at runtime which bodies are real. One source
#: therefore compiles unchanged on any architecture and self-selects the
#: variants that target actually has, so a new part needs no table entry -- only
#: an added guarded block if it introduces a new opcode.
#:
#: On gfx950 the fp8 and fp4 rates do NOT come from separate opcodes: both use
#: ``mfma_scale_f32_16x16x128_f8f6f4`` and differ only in the cbsz/blgp format
#: selector (0 = e4m3 fp8, 4 = e2m1 fp4). Probing fp8 through the 16x16x32
#: opcode instead reports the bf16 rate, because that variant carries identical
#: FLOPs per instruction.
MFMA_PROBE_SRC = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>

typedef __bf16 bf16x4 __attribute__((ext_vector_type(4)));
typedef __bf16 bf16x8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16x4 __attribute__((ext_vector_type(4)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

#define NACC 8

// Bit index per candidate, shared between the device-side availability report
// and the host-side registration.
#define BIT_BF16_16X16X32 0
#define BIT_BF16_16X16X16 1
#define BIT_F16_16X16X16 2
#define BIT_FP8_16X16X32 3
#define BIT_F8F6F4 4
#define BIT_F16_16X16X32 5

// Body shared by every candidate: fill NACC independent chains, hammer the
// matrix core, then consume the result through a branch that never fires so
// the optimizer cannot delete the loop.
#define MFMA_BODY(SETUP, EXPR)                                            \
  SETUP;                                                                  \
  f32x4 acc[NACC];                                                        \
  _Pragma("unroll") for (int i = 0; i < NACC; ++i) acc[i] = f32x4{};      \
  for (int it = 0; it < iters; ++it) {                                    \
    _Pragma("unroll") for (int i = 0; i < NACC; ++i) acc[i] = (EXPR);     \
  }                                                                       \
  float s = 0;                                                            \
  _Pragma("unroll") for (int i = 0; i < NACC; ++i) s += acc[i][0];        \
  if (s == -1.0f) out[0] = s;

#define BF16_SETUP                                                        \
  bf16x8 a, b;                                                            \
  for (int i = 0; i < 8; ++i) { a[i] = (__bf16)1.0f; b[i] = (__bf16)1.0f; }
#define BF16X4_SETUP                                                      \
  bf16x4 a, b;                                                            \
  for (int i = 0; i < 4; ++i) { a[i] = (__bf16)1.0f; b[i] = (__bf16)1.0f; }
#define F16X4_SETUP                                                       \
  f16x4 a, b;                                                             \
  for (int i = 0; i < 4; ++i) { a[i] = (_Float16)1.0f; b[i] = (_Float16)1.0f; }
#define F16_SETUP                                                         \
  f16x8 a, b;                                                             \
  for (int i = 0; i < 8; ++i) { a[i] = (_Float16)1.0f; b[i] = (_Float16)1.0f; }
#define I64_SETUP long a = 0x0101010101010101L, b = 0x0101010101010101L;

// Reports which candidate bodies the device pass actually compiled. The host
// pass cannot answer this itself, so it asks the device at runtime.
__global__ void availability(unsigned* out) {
  unsigned m = 0;
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x32_bf16)
  m |= 1u << BIT_BF16_16X16X32;
#endif
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
  m |= 1u << BIT_BF16_16X16X16;
#endif
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16f16)
  m |= 1u << BIT_F16_16X16X16;
#endif
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8)
  m |= 1u << BIT_FP8_16X16X32;
#endif
#if __has_builtin(__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4)
  m |= 1u << BIT_F8F6F4;
#endif
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x32_f16)
  m |= 1u << BIT_F16_16X16X32;
#endif
  out[0] = m;
}

// Signatures are unconditional so the host pass can take their addresses; only
// the bodies are guarded. An unavailable variant compiles to an empty kernel
// that is never registered, because its availability bit stays clear.
__global__ void mfma_bf16_16x16x32(float* out, int iters) {
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x32_bf16)
  MFMA_BODY(BF16_SETUP, __builtin_amdgcn_mfma_f32_16x16x32_bf16(a, b, acc[i], 0, 0, 0))
#else
  (void)out;
  (void)iters;
#endif
}

__global__ void mfma_bf16_16x16x16(float* out, int iters) {
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16bf16_1k)
  MFMA_BODY(BF16X4_SETUP, __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, acc[i], 0, 0, 0))
#else
  (void)out;
  (void)iters;
#endif
}

__global__ void mfma_f16_16x16x16(float* out, int iters) {
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x16f16)
  MFMA_BODY(F16X4_SETUP, __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, acc[i], 0, 0, 0))
#else
  (void)out;
  (void)iters;
#endif
}

__global__ void mfma_f16_16x16x32(float* out, int iters) {
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x32_f16)
  MFMA_BODY(F16_SETUP, __builtin_amdgcn_mfma_f32_16x16x32_f16(a, b, acc[i], 0, 0, 0))
#else
  (void)out;
  (void)iters;
#endif
}

__global__ void mfma_fp8_16x16x32(float* out, int iters) {
#if __has_builtin(__builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8)
  MFMA_BODY(I64_SETUP, __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(a, b, acc[i], 0, 0, 0))
#else
  (void)out;
  (void)iters;
#endif
}

// The gfx950 double/quad-rate path. cbsz/blgp must be compile-time immediates,
// so the format selector is a template parameter rather than an argument.
template <int FMT>
__global__ void mfma_f8f6f4(float* out, int iters) {
#if __has_builtin(__builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4)
  MFMA_BODY(i32x8 a{}; i32x8 b{},
            __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, acc[i], FMT, FMT, 0, 0, 0, 0))
#else
  (void)out;
  (void)iters;
#endif
}

struct Candidate {
  const char* precision;
  const char* variant;
  void (*kernel)(float*, int);
  double flops_per_inst;  // 2*M*N*K for one wavefront instruction
  int bit;
};

int main(int argc, char** argv) {
  int device = (argc > 1) ? atoi(argv[1]) : 0;
  int iters = (argc > 2) ? atoi(argv[2]) : 20000;
  if (hipSetDevice(device) != hipSuccess) {
    fprintf(stderr, "hipSetDevice(%d) failed\n", device);
    return 1;
  }
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, device) != hipSuccess) {
    fprintf(stderr, "hipGetDeviceProperties failed\n");
    return 1;
  }
  int cus = prop.multiProcessorCount;

  unsigned* dmask = nullptr;
  if (hipMalloc(&dmask, sizeof(unsigned)) != hipSuccess) {
    fprintf(stderr, "hipMalloc failed\n");
    return 1;
  }
  hipLaunchKernelGGL(availability, dim3(1), dim3(1), 0, 0, dmask);
  unsigned mask = 0;
  if (hipMemcpy(&mask, dmask, sizeof(unsigned), hipMemcpyDeviceToHost) != hipSuccess) {
    fprintf(stderr, "availability probe failed\n");
    return 1;
  }
  (void)hipFree(dmask);

  const Candidate all[] = {
      {"bf16", "mfma_f32_16x16x32_bf16", mfma_bf16_16x16x32, 2.0 * 16 * 16 * 32,
       BIT_BF16_16X16X32},
      {"bf16", "mfma_f32_16x16x16bf16_1k", mfma_bf16_16x16x16, 2.0 * 16 * 16 * 16,
       BIT_BF16_16X16X16},
      {"fp16", "mfma_f32_16x16x16f16", mfma_f16_16x16x16, 2.0 * 16 * 16 * 16, BIT_F16_16X16X16},
      {"fp16", "mfma_f32_16x16x32_f16", mfma_f16_16x16x32, 2.0 * 16 * 16 * 32, BIT_F16_16X16X32},
      {"fp8", "mfma_f32_16x16x32_fp8_fp8", mfma_fp8_16x16x32, 2.0 * 16 * 16 * 32,
       BIT_FP8_16X16X32},
      {"fp8", "mfma_scale_f32_16x16x128_f8f6f4[e4m3]", mfma_f8f6f4<0>, 2.0 * 16 * 16 * 128,
       BIT_F8F6F4},
      {"fp4", "mfma_scale_f32_16x16x128_f8f6f4[e2m1]", mfma_f8f6f4<4>, 2.0 * 16 * 16 * 128,
       BIT_F8F6F4},
  };
  std::vector<Candidate> cands;
  for (const auto& c : all) {
    if (mask & (1u << c.bit)) cands.push_back(c);
  }

  float* out = nullptr;
  if (hipMalloc(&out, sizeof(float)) != hipSuccess) {
    fprintf(stderr, "hipMalloc failed\n");
    return 1;
  }

  const int threads = 256;
  const int waves = threads / 64;
  const int blocks = cus * 2;

  printf("{\"kind\":\"device\",\"arch\":\"%s\",\"cus\":%d,\"boost_mhz\":%.0f}\n", prop.gcnArchName,
         cus, prop.clockRate / 1000.0);
  for (const auto& c : cands) {
    hipLaunchKernelGGL(c.kernel, dim3(blocks), dim3(threads), 0, 0, out, 100);
    if (hipDeviceSynchronize() != hipSuccess) continue;

    std::vector<double> runs;
    for (int rep = 0; rep < 5; ++rep) {
      hipEvent_t t0, t1;
      if (hipEventCreate(&t0) != hipSuccess) break;
      if (hipEventCreate(&t1) != hipSuccess) break;
      (void)hipEventRecord(t0);
      hipLaunchKernelGGL(c.kernel, dim3(blocks), dim3(threads), 0, 0, out, iters);
      (void)hipEventRecord(t1);
      if (hipEventSynchronize(t1) != hipSuccess) break;
      float ms = 0;
      if (hipEventElapsedTime(&ms, t0, t1) != hipSuccess || ms <= 0) break;
      double insts = (double)blocks * waves * iters * NACC;
      runs.push_back(insts * c.flops_per_inst / (ms * 1e-3));
      (void)hipEventDestroy(t0);
      (void)hipEventDestroy(t1);
    }
    if (runs.empty()) continue;
    std::sort(runs.begin(), runs.end());
    printf("{\"kind\":\"mfma\",\"precision\":\"%s\",\"variant\":\"%s\",\"flops_per_sec\":%.6e}\n",
           c.precision, c.variant, runs.back());
    fflush(stdout);
  }
  (void)hipFree(out);
  return 0;
}
"""

#: Streaming-read bandwidth probe.
#:
#: Non-temporal (cache-bypassing) fully coalesced loads, so the figure is the
#: hardware streaming limit rather than any particular kernel's efficiency. The
#: decode roofline counts read traffic (weights plus KV cache), so a read probe
#: is the right shape; copy and triad land far lower (~61% of peak on MI355X)
#: and would model write-heavy traffic this roofline does not have.
#:
#: ``float4`` is spelled as a native ``ext_vector_type`` rather than HIP's
#: ``float4``, which is a class type the non-temporal builtins reject.
BANDWIDTH_PROBE_SRC = r"""
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>

typedef float vec_t __attribute__((ext_vector_type(4)));

static double now_epoch() {
  struct timespec ts;
  clock_gettime(CLOCK_REALTIME, &ts);
  return ts.tv_sec + ts.tv_nsec * 1e-9;
}

__global__ void stream_read(const vec_t* __restrict__ src, size_t n, float* out) {
  size_t stride = (size_t)gridDim.x * blockDim.x;
  vec_t acc = {0.f, 0.f, 0.f, 0.f};
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
    vec_t v = __builtin_nontemporal_load(&src[i]);
    acc += v;
  }
  float s = acc[0] + acc[1] + acc[2] + acc[3];
  if (s == -1.0f) out[0] = s;
}

int main(int argc, char** argv) {
  int device = (argc > 1) ? atoi(argv[1]) : 0;
  double target_gib = (argc > 2) ? atof(argv[2]) : 16.0;
  // Wall-clock instant at which to begin the timed loop. Concurrent instances
  // are given a common value so their measured windows genuinely overlap:
  // without it each process finishes allocation and warm-up at a different
  // moment, and an "all GPUs loaded" run reports the same figure as a solo one.
  double start_at = (argc > 3) ? atof(argv[3]) : 0.0;
  double measure_sec = (argc > 4) ? atof(argv[4]) : 2.0;
  if (hipSetDevice(device) != hipSuccess) {
    fprintf(stderr, "hipSetDevice(%d) failed\n", device);
    return 1;
  }
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, device) != hipSuccess) return 1;

  size_t free_b = 0, total_b = 0;
  if (hipMemGetInfo(&free_b, &total_b) != hipSuccess) return 1;
  // Stay well clear of whatever else is resident; the probe must never be the
  // reason a serving process hits an allocation failure.
  size_t want = (size_t)(target_gib * (1ull << 30));
  size_t cap = (size_t)(free_b * 0.5);
  size_t bytes = want < cap ? want : cap;
  bytes &= ~(size_t)(sizeof(vec_t) - 1);
  if (bytes < (1ull << 28)) {
    fprintf(stderr, "insufficient free VRAM for bandwidth probe\n");
    return 1;
  }

  vec_t* buf = nullptr;
  if (hipMalloc(&buf, bytes) != hipSuccess) {
    fprintf(stderr, "hipMalloc(%zu) failed\n", bytes);
    return 1;
  }
  (void)hipMemset(buf, 1, bytes);
  float* out = nullptr;
  if (hipMalloc(&out, sizeof(float)) != hipSuccess) return 1;

  size_t n = bytes / sizeof(vec_t);
  int threads = 256;
  int blocks = prop.multiProcessorCount * 8;

  hipLaunchKernelGGL(stream_read, dim3(blocks), dim3(threads), 0, 0, buf, n, out);
  if (hipDeviceSynchronize() != hipSuccess) return 1;

  // Spin to the shared start instant. Sleeping would risk waking late and
  // missing the window the other instances are measuring in.
  while (start_at > 0 && now_epoch() < start_at) {
  }

  // Duration-based rather than a fixed iteration count: one pass over a 16 GiB
  // buffer takes only ~2.4 ms at these rates, so a handful of iterations would
  // finish well inside the process-start skew between concurrent instances and
  // measure nothing about contention.
  std::vector<double> runs;
  double deadline = now_epoch() + measure_sec;
  while (now_epoch() < deadline) {
    hipEvent_t t0, t1;
    if (hipEventCreate(&t0) != hipSuccess) break;
    if (hipEventCreate(&t1) != hipSuccess) break;
    (void)hipEventRecord(t0);
    hipLaunchKernelGGL(stream_read, dim3(blocks), dim3(threads), 0, 0, buf, n, out);
    (void)hipEventRecord(t1);
    if (hipEventSynchronize(t1) != hipSuccess) break;
    float ms = 0;
    if (hipEventElapsedTime(&ms, t0, t1) != hipSuccess || ms <= 0) break;
    runs.push_back((double)bytes / (ms * 1e-3) / 1e9);
    (void)hipEventDestroy(t0);
    (void)hipEventDestroy(t1);
  }
  if (runs.empty()) {
    fprintf(stderr, "no successful bandwidth iterations\n");
    return 1;
  }
  std::sort(runs.begin(), runs.end());
  printf("{\"kind\":\"bandwidth\",\"arch\":\"%s\",\"buffer_bytes\":%zu,\"gb_per_sec\":%.3f}\n",
         prop.gcnArchName, bytes, runs.back());
  (void)hipFree(buf);
  (void)hipFree(out);
  return 0;
}
"""
