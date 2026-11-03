#!/bin/bash
# Build the runtime (tool library, marker library) and the probe program.
#   RUPPELL_BUILD     output directory (default runtime/build)
#   RUPPELL_PLATFORM  rocm or cuda (default: cuda if nvcc is found and hipcc is not)
#   ROCM_PATH         ROCm install (default /opt/rocm)
#   CUDA_PATH         CUDA install (default /usr/local/cuda)
#   RUPPELL_ARCH      GPU arch (default: the first gfx target on this node; sm_100 on CUDA)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ROCM=${ROCM_PATH:-/opt/rocm}
CUDA=${CUDA_PATH:-/usr/local/cuda}
B=${RUPPELL_BUILD:-$ROOT/runtime/build}
if [ -z "${RUPPELL_PLATFORM:-}" ]; then
    if [ -x "$CUDA/bin/nvcc" ] && [ ! -x "$ROCM/bin/hipcc" ]; then RUPPELL_PLATFORM=cuda; else RUPPELL_PLATFORM=rocm; fi
fi
mkdir -p "$B"

if [ "$RUPPELL_PLATFORM" = cuda ]; then
    ARCH=${RUPPELL_ARCH:-sm_100}
    CUPTI_INC=$CUDA/targets/x86_64-linux/include
    CUPTI_LIB=$CUDA/targets/x86_64-linux/lib
    [ -f "$CUPTI_INC/cupti.h" ] || { CUPTI_INC=$CUDA/extras/CUPTI/include; CUPTI_LIB=$CUDA/extras/CUPTI/lib64; }
    echo "ruppell build: cuda=$("$CUDA/bin/nvcc" --version | tail -1) arch=$ARCH out=$B"
    CXXFLAGS="-std=c++17 -O2 -g -fPIC -Wall -Wno-unused-function -I$CUDA/include -I$CUPTI_INC -I$ROOT/runtime/src"
    # gray_parrot_cuda.cpp: hardware counters per step (gray-parrot), inert unless RUPPELL_COUNTERS is set
    g++ $CXXFLAGS -shared "$ROOT/runtime/src/tool_cuda.cpp" "$ROOT/runtime/src/gray_parrot_cuda.cpp" \
        -L"$CUPTI_LIB" -L"$CUDA/lib64" -L"$CUDA/lib64/stubs" -Wl,-rpath,"$CUPTI_LIB" \
        -lcupti -lcuda -lpthread -ldl -o "$B/libruppell_tool.so"
    # gray-parrot's device samples (RUPPELL_DEVICE_COUNTERS), evaluated after the run: CUPTI's Host API alone
    g++ $CXXFLAGS "$ROOT/runtime/src/pm_eval.cpp" -L"$CUPTI_LIB" -Wl,-rpath,"$CUPTI_LIB" -lcupti -lpthread \
        -o "$B/ruppell_pm_eval"
    # NCCL's profiler plugin (roadmap-v2 M8; on by default, RUPPELL_NCCL=0 off): NCCL's own view of its calls
    g++ -std=c++17 -O2 -g -fPIC -Wall -shared "$ROOT/runtime/src/nccl_profiler.cpp" -lpthread -ldl \
        -o "$B/libruppell_nccl_profiler.so"

    "$CUDA/bin/nvcc" -O2 -Xcompiler -fPIC -shared -arch="$ARCH" "$ROOT/runtime/src/marker.cu" \
        -o "$B/libruppell_marker.so"

    "$CUDA/bin/nvcc" -O2 -arch="$ARCH" "$ROOT/tests/probes/graph_probe.cu" \
        -L"$B" -Xlinker -rpath,"$B" -lruppell_marker -ldl -o "$B/graph_probe"
else
    ARCH=${RUPPELL_ARCH:-$("$ROCM/bin/rocm_agent_enumerator" 2>/dev/null | grep -m1 '^gfx' || echo gfx950)}
    echo "ruppell build: rocm=$(cat "$ROCM/.info/version" 2>/dev/null) arch=$ARCH out=$B"
    CXXFLAGS="-std=c++17 -O2 -g -fPIC -Wall -Wno-unused-function -D__HIP_PLATFORM_AMD__ -I$ROCM/include -I$ROOT/runtime/src"
    # gray_parrot.cpp: hardware counters (gray-parrot), inert unless RUPPELL_COUNTERS or
    # RUPPELL_DEVICE_COUNTERS is set
    g++ $CXXFLAGS -shared "$ROOT/runtime/src/tool.cpp" "$ROOT/runtime/src/gray_parrot.cpp" \
        -L"$ROCM/lib" -Wl,-rpath,"$ROCM/lib" -lrocprofiler-sdk -lpthread \
        -o "$B/libruppell_tool.so"
    # RCCL's profiler plugin interface v4 (roadmap-v2 M8; RUPPELL_NCCL=0 off): the same plugin as CUDA's
    g++ -std=c++17 -O2 -g -fPIC -Wall -shared "$ROOT/runtime/src/nccl_profiler.cpp" -lpthread -ldl \
        -o "$B/libruppell_nccl_profiler.so"

    "$ROCM/bin/hipcc" -O2 -fPIC -shared --offload-arch="$ARCH" "$ROOT/runtime/src/marker.hip" \
        -o "$B/libruppell_marker.so"

    "$ROCM/bin/hipcc" -O2 --offload-arch="$ARCH" "$ROOT/tests/probes/graph_probe.hip" \
        -L"$B" -Wl,-rpath,"$B" -lruppell_marker -ldl -o "$B/graph_probe"
fi
echo "$RUPPELL_PLATFORM" > "$B/PLATFORM"

# the commit: from git, or from .commit where the tree was copied without .git
git -C "$ROOT" rev-parse HEAD > "$B/COMMIT" 2>/dev/null || cp "$ROOT/.commit" "$B/COMMIT"
sha256sum "$B/libruppell_tool.so" "$B/libruppell_marker.so" "$B/graph_probe" | tee "$B/DIGESTS"
