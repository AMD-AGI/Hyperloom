# Phase 0 — Gate A: native multi-file HIP `.pyd` on Windows gfx1100

**Status: PASSED (2026-06-29)** on AMD Radeon RX 7900 XT (gfx1100), Windows 11.

This is the single most important Phase-0 de-risking experiment for the whole project, and
it had **no public precedent**: building a multi-file native HIP extension and importing it
into Python against AMD's native-Windows ROCm PyTorch.

## Result
```
[1/3] hipcc.exe ... --offload-arch=gfx1100/1101/1102/1103 ... kernel.cu  -> kernel.cuda.o
[2/3] hipcc.exe ... bindings.cu -> bindings.cuda.o
[3/3] MSVC link.exe ... c10_hip.lib torch_hip.lib amdhip64.lib -> phase0_hip_ext.pyd
BUILD_OK in 101.8 s
HIP_EXT_RESULT_OK True   max_err 0.0
```
→ HIP SDK 7.2 (device compile, clang 22) + torch 2.10+rocm7.13 (host link) coexist with no
fatal ABI mismatch for a simple custom kernel.

## How to run
From a shell with the MSVC toolchain active (see header of `build_and_test.py`):
```
cmd /c "\"E:\BuildTools\VC\Auxiliary\Build\vcvars64.bat\" && set ROCM_HOME=C:\HIP-SDK && set HIP_PATH=C:\HIP-SDK && set ROCM_PATH=C:\HIP-SDK && python build_and_test.py"
```

## Two gotchas this PoC solves (carry forward)
1. **torch cpp_extension hipify None bug** (torch 2.10+rocm7.13): `_jit_compile` doesn't guard
   `hipified_path` against `None` (hipify returns None for unchanged files) → crash. We monkeypatch
   `torch.utils.hipify.hipify_python.hipify` to replace any `None` with the original source path.
   *(The real vLLM build uses CMake, not cpp_extension, so this specific bug won't apply there.)*
2. **ROCm device library not found**: HIP SDK 7.2 keeps device bitcode under
   `C:\HIP-SDK\lib\llvm\amdgcn\bitcode` (not `<root>/amdgcn/bitcode`), so clang errors. We pass
   `--rocm-device-lib-path=C:\HIP-SDK\lib\llvm\amdgcn\bitcode` via `extra_cuda_cflags`.

## Next escalations
- Compile a *real* kernel (an RMSNorm / a paged-attention tile) the same way.
- Try compiling vLLM's `csrc/rocm` `__GFX11__` paged-attention via CMake on this toolchain.
- Verify the W4A16 native HIP kernel (vLLM PR #41394) builds here (primary quant path).

See `../../research/` for the full foundation analysis.
