# kaolin — a kernel-optimization adapter

The minimap2 adapter sweeps *configuration*. This one exposes a **kernel** to the loop: kaolin's
spherical-gaussian inner product, JIT-built from source so the KERNEL phase can edit the `.cu` and
this adapter re-benchmarks the result. The whole measure → optimize → validate cycle closes through
one script.

Built from public source only: [`AMD-Ecosystem/kaolin`](https://github.com/AMD-Ecosystem/kaolin),
branch `rocm-enablement`.

## Run it

```bash
git clone --branch rocm-enablement https://github.com/AMD-Ecosystem/kaolin.git

export KAOLIN_SRC=$PWD/kaolin
export KAOLIN_ROCM_INC=/path/to/rocm/include     # see "Build environment" below
RESULT_DIR=/tmp/kaolin-run ./custom_mi355x.sh
cat /tmp/kaolin-run/inferencex_result.json
```

The objective is `kops/s = 1000 / geomean_fwd_bwd_ms` — the geometric mean of forward and backward
latency across three shapes (64×128, 128×512, 256×1024). Higher is better, as the loop requires.

## Build environment — two fixes that are load-bearing

Both bite on the slim `rocm/pytorch` images and neither error names the real cause:

1. **Missing math-library headers.** `torch/headeronly/util/complex.h` includes `thrust/complex.h`,
   and `ATen/hip/HIPContextLight.h` includes `hipsparse/hipsparse.h`; the slim image ships neither.
   Point `KAOLIN_ROCM_INC` at an include tree that has them (a `pip install "rocm[devel]"` tree
   works) and the adapter adds it to `HIPCC_COMPILE_FLAGS_APPEND` and `CPLUS_INCLUDE_PATH`.
2. **No unversioned `libamdhip64.so`.** The image ships only `libamdhip64.so.N`, so the final link
   fails with `cannot find -lamdhip64`. The adapter creates the symlink itself.

Two more that will catch you if you build this kernel by hand:

- The host `.cpp` must be compiled with **`-DWITH_CUDA`**. Without it the op compiles and links
  fine, then refuses at runtime with *"Kaolin built without CUDA, cannot run with GPU tensors"* —
  the ROCm build still uses the `WITH_CUDA` guard.
- kaolin's pybind entry point lives in its own `bindings.cpp` and exports the whole library. A
  single-op build needs a minimal `PYBIND11_MODULE` of its own, which `benchmark.py` generates;
  without it the import fails with `does not define module export function`.

## The correctness gate

The first run records the kernel's output as a reference; later runs compare against it and must
match within `1e-3`. Each run also calls the kernel twice and requires bit-identical results.

This is deliberately **not** an analytic re-derivation of the SG math. What a kernel-optimization
loop has to prove is that the *rewritten* kernel still agrees with the *pre-rewrite* one on the same
inputs — an independent reimplementation would only test whether the reimplementation is right.
Delete `$KAOLIN_REF_STORE` to re-baseline after an intentional change.

## What the levers do here — honestly, very little

Measured on one MI350X (gfx950), ROCm 10, torch 2.13:

| Config | kops/s | vs baseline |
|---|---|---|
| *(baseline)* | 15,297.77 | 1.00x |
| `hw_queues_4` | 15,310.89 | 1.00x |
| `kernarg_plus_queues` | 15,177.50 | 0.99x |

All within ~1%, which is measurement noise. **That is the useful finding**: this kernel is not
configuration-tunable, and `levers.json` is shipped mainly to show the shape of a declaration. The
gain on this workload comes from the KERNEL phase rewriting the `.cu` — a different mechanism from
minimap2's configuration sweep, and the reason both adapters are worth reading side by side.

If your workload behaves like this, a near-flat sweep is a signal to let the kernel path do the
work rather than evidence that the loop failed.

## Files

| File | Purpose |
|---|---|
| `custom_mi355x.sh` | Hyperloom entrypoint |
| `_kaolin_bench.sh` | Build-env fixes, runs the benchmark, writes `inferencex_result.json` |
| `benchmark.py` | JIT-builds the kernel, times fwd+bwd, applies the gate |
| `levers.json` | Six HIP runtime variants (near-no-ops here, by measurement) |

`levers.json` is read by the declared-lever grid in
[#1546](https://github.com/AMD-AGI/Hyperloom/pull/1546); until that lands the adapter still runs and
still reports through the contract, but the declared variants are not seeded.
