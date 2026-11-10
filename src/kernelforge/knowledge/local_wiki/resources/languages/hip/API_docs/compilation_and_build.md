---
title: HIP compilation & build reference — hipcc/amdclang++ flags
kind: api_reference
gens: [gfx942, gfx950]
regimes: [both]
status: sota
updated: 2026-07-09
sources:
  - https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html
  - https://rocm.docs.amd.com/en/latest/reference/rocmcc.html
---

# HIP Kernel Compilation & Build Reference

> API/build reference (how to compile). For *perf-oriented* flag choices and ISA verification see
> [../skills/optimize/hip_levers/hip_authoring_model.md](../skills/optimize/hip_levers/hip_authoring_model.md) and
> [../skills/optimize/hip_levers/hip_traps.md](../skills/optimize/hip_levers/hip_traps.md).

## hipcc Compiler Flags

### Production Build (Raw HIP, gfx950)
```bash
hipcc -x hip --offload-arch=gfx950 \
    -O3 \
    -std=c++20 \
    -fgpu-rdc \
    -fvisibility=hidden \
    -mllvm -amdgpu-early-inline-all=true \
    -mllvm --lsr-drop-solution=1 \
    -mllvm -enable-post-misched=0 \
    -mllvm -amdgpu-coerce-illegal-types=1 \
    -DHIP_ENABLE_WARP_SYNC_BUILTINS=1 \
    kernel.cpp -o kernel
```

### Flag Explanations
| Flag | Purpose |
|------|---------|
| `-fgpu-rdc` | Relocatable device code (needed for separate compilation) |
| `-fvisibility=hidden` | Hide symbols for smaller binary |
| `-amdgpu-early-inline-all=true` | Inline all functions early — critical for register control |
| `--lsr-drop-solution=1` | Loop strength reduction hint — helps some register patterns |
| `-enable-post-misched=0` | Disable post-RA machine scheduling — preserves hand-scheduled order |
| `-amdgpu-coerce-illegal-types=1` | Allow non-standard types (gfx950+ only) |
| `-DHIP_ENABLE_WARP_SYNC_BUILTINS=1` | Enable warp-sync built-in functions |

### HipKittens Build
```bash
hipcc -x hip --offload-arch=gfx950 \
    -O3 \
    -std=c++20 \
    -I/path/to/HipKittens/include \
    -DKITTENS_CDNA4 \
    kernel.cpp -o kernel
```

### Multi-Architecture Build
```bash
hipcc --offload-arch=gfx942 --offload-arch=gfx950 \
    -O3 -std=c++20 kernel.cpp -o kernel
```

## Preprocessor Guards

### Architecture-Specific Code
```cpp
#if defined(__gfx950__)
    // CDNA4-specific code (MI350X/MI355X)
    // Scaled MFMA, FP4/FP6, 16-byte buffer_load_lds
#elif defined(__gfx942__)
    // CDNA3-specific code (MI300X/MI325X)
    // Standard MFMA, 4-byte buffer_load_lds only
#else
    static_assert(false, "Unsupported architecture");
#endif
```

### Compile-Time Architecture Detection
```cpp
// In device code
#if defined(__gfx950__)
    constexpr int BUFFER_LOAD_BYTES = 16;
    constexpr int NUM_XCDS = 32;
#elif defined(__gfx942__)
    constexpr int BUFFER_LOAD_BYTES = 4;
    constexpr int NUM_XCDS = 8;
#endif
```

### Runtime Architecture Detection
```cpp
hipDeviceProp_t props;
hipGetDeviceProperties(&props, 0);
// props.gcnArchName: "gfx950" or "gfx942"
// props.warpSize: 64 (CDNA)
// props.sharedMemPerBlock: LDS size
// props.regsPerBlock: VGPR count
```

## PyTorch Extension Build (setup.py)

```python
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            name="my_kernel",
            sources=["kernel.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++20"],
                "nvcc": [
                    "-O3",
                    "-std=c++20",
                    "--offload-arch=gfx950",
                    "-fgpu-rdc",
                    "-DHIP_ENABLE_WARP_SYNC_BUILTINS=1",
                    "-mllvm", "-amdgpu-early-inline-all=true",
                    "-mllvm", "--lsr-drop-solution=1",
                    "-mllvm", "-enable-post-misched=0",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

Note: Despite the name "CUDAExtension", this works for HIP on AMD via ROCm's
CUDA compatibility layer. File extensions can be `.cu` or `.hip`.

## CMake Build

```cmake
cmake_minimum_required(VERSION 3.21)
project(my_kernel LANGUAGES CXX HIP)

set(CMAKE_HIP_ARCHITECTURES "gfx950")
set(CMAKE_HIP_STANDARD 20)

add_library(my_kernel SHARED kernel.hip)
target_compile_options(my_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:HIP>:
        -O3
        -fgpu-rdc
        -mllvm -amdgpu-early-inline-all=true
    >
)
target_link_libraries(my_kernel PRIVATE hip::host hip::device)
```

## Dynamic Shared Memory

```cpp
// Must declare attribute before launch if exceeding default (48KB)
hipFuncSetAttribute(
    my_kernel,
    hipFuncAttributeMaxDynamicSharedMemorySize,
    shared_mem_bytes  // Up to 256KB on CDNA4
);

// In kernel
extern __shared__ char smem[];
// Or: extern __shared__ int __shm[];

// Launch with shared memory size
my_kernel<<<grid, block, shared_mem_bytes, stream>>>(...);
```

## Build System Gotchas

### JIT Cache Invalidation
```bash
# hipcc caches compiled kernels; stale after header changes
rm -rf /tmp/comgr_*
```

### Dependency Tracking
```bash
# hipcc doesn't track transitive header deps by default
# Force clean rebuild when headers change:
rm -f *.o && hipcc ...

# Or generate dependency files:
hipcc --write-dependencies -MD kernel.cpp -o kernel
```

### Separate Compilation + Linking
```bash
# More reliable than single-step for complex projects
hipcc -c -x hip --offload-arch=gfx950 -O3 kernel.cpp -o kernel.o
hipcc --offload-arch=gfx950 kernel.o -o kernel
```

### Stale .so Artifact (Python modules)
After rebuilding, Python may still load the old .so from a cached path.
```bash
# Force copy to expected location
cp build/module/build/module.so ../../module.so

# Or reinstall the Python package
pip install -e . --no-build-isolation
```

## Profiling Integration

### rocprof v3
```bash
# PMC counters
rocprofv3 --pmc SQ_INSTS_VALU_MFMA_BF16 SQ_INSTS_VMEM SQ_WAIT_INST_LDS SQ_WAIT_INST_ANY \
    -- ./my_kernel

# ISA dump (verify register allocation)
rocprofv3 --isa -- ./my_kernel
```

### Register Count Verification
```bash
# Check VGPR/SGPR/AGPR usage
hipcc -x hip --offload-arch=gfx950 -O3 -Rpass-analysis=regalloc kernel.cpp 2>&1 | grep "vgpr\|sgpr\|agpr"
```

## Container Builds

### MI350X/MI355X Docker
```bash
# Base image with ROCm
docker run -it --device=/dev/kfd --device=/dev/dri \
    --group-add video \
    -v /path/to/code:/workspace \
    rocm/dev-ubuntu-22.04:latest

# Inside container
hipcc --version  # Verify ROCm version
rocminfo | grep gfx  # Verify GPU target
```

### HipKittens Docker Setup
See: `${KA_WORKSPACE}/HipKittens/docs/docker/launch_docker_mi350x.md`
