// ruppell step markers: a one-thread kernel enqueued at every step boundary, so a
// step is an interval on the GPU timeline. The step id and kind travel on the
// external correlation id pushed around the launch (see records.h); the
// argument is kept so the dispatch is never optimized into nothing.
#include <hip/hip_runtime.h>

__global__ void
ruppell_step_marker(unsigned long long code)
{
    if(code == 0xFFFFFFFFFFFFFFFFull && threadIdx.x == 1u) __builtin_trap();
}

extern "C" int
ruppell_marker_launch(void* stream, unsigned long long code)
{
    hipLaunchKernelGGL(ruppell_step_marker, dim3(1), dim3(1), 0,
                       reinterpret_cast<hipStream_t>(stream), code);
    return static_cast<int>(hipGetLastError());
}
