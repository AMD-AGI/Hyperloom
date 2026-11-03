#!/bin/bash
# Build the CPU op library (src/cpu_ops.cpp) against the libtorch of the
# Python that will be traced; run it with that Python (inside the serving
# container, say). The same source on CUDA and ROCm.
#   RUPPELL_BUILD  output directory (default runtime/build)
#   PYTHON         the traced program's python (default python3)
# Writes libruppell_cpu_ops.so and CPU_OPS_TORCH (the torch version it was built
# for; ruppell refuses to load it into another).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
B=${RUPPELL_BUILD:-$ROOT/runtime/build}
PY=${PYTHON:-python3}
mkdir -p "$B"
read -r TORCH_DIR TORCH_VER ABI < <("$PY" -c 'import os, torch; print(os.path.dirname(torch.__file__), torch.__version__, int(torch._C._GLIBCXX_USE_CXX11_ABI))')
echo "ruppell cpu ops build: torch $TORCH_VER ($TORCH_DIR) abi=$ABI out=$B"
g++ -std=c++20 -O2 -g -DNDEBUG -fPIC -shared -Wall -Wno-unused-function -D_GLIBCXX_USE_CXX11_ABI="$ABI" \
    -I"$TORCH_DIR/include" -I"$TORCH_DIR/include/torch/csrc/api/include" -I"$ROOT/runtime/src" \
    "$ROOT/runtime/src/cpu_ops.cpp" -L"$TORCH_DIR/lib" -Wl,-rpath,"$TORCH_DIR/lib" -ltorch_cpu -lc10 \
    -o "$B/libruppell_cpu_ops.so"
echo "$TORCH_VER" > "$B/CPU_OPS_TORCH"
sha256sum "$B/libruppell_cpu_ops.so" | tee -a "$B/DIGESTS"
