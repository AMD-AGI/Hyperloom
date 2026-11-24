---
myst:
    html_meta:
        "description": "Targeted build (Rung 5) variables for Hyperloom: auto-escalation disable switch, aiter JIT directory, GPU target architecture, compile parallelism cap, and framework Python interpreter."
        "keywords": "Hyperloom, environment variables, targeted builds, Rung 5, HYPERLOOM_ENABLEMENT_DISABLE_TARGETED_BUILD, PYTORCH_ROCM_ARCH, MAX_JOBS, HYPERLOOM_FRAMEWORK_PYTHON, aiter"
---
# Targeted builds (Rung 5)

These variables control the Rung-5 off-loop compiled-component acquisition
step (AITER FP4/MLA/NSA kernels, sgl-kernel, and vLLM from source).  All are
optional; defaults are safe for standard single-node deployments.

| Variable | Default | Description |
|---|---|---|
| `HYPERLOOM_ENABLEMENT_DISABLE_TARGETED_BUILD` | Unset (`0`) | Set to `1` to completely disable Rung-5 auto-escalation.  When set, compiled-gap failures proceed to the stall gate without attempting a build.  Useful when the compile toolchain is unavailable or the session budget is too tight. |
| `INFERENCE_`<br>`OPTIMIZER_`<br>`AITER_JIT_DIR` | Aiter default | Per-attempt override set automatically to `<attempt_root>/aiter_jit` by each targeted build.  Override manually only when you need the global JIT cache to point at a pre-built location; leaving it unset lets each build use its own isolated directory. |
| `PYTORCH_ROCM_ARCH` | Detected | Explicit GPU target architecture (e.g. `gfx942`, `gfx950`) injected into each compile.  Set automatically from the session `--gpu-type`; operator-override applies to bare-metal installs outside the session.  **Compile target only — it does not participate in architecture detection.**  It names the archs a wheel is *built* for, not the installed device, so provenance ignores it entirely and resolves `gfx_arch` from `HYPERLOOM_GFX_ARCH`, then `--gpu-type`, then `rocminfo`. |
| `MAX_JOBS` | `8` | Parallelism cap for cmake/hipcc compile steps inside a targeted build.  Reduce on memory-constrained nodes (`MAX_JOBS=4` for a 64 GB compile node).  The default `8` is conservative enough for MI300X/MI355X nodes with 512 GB+. |
| `HYPERLOOM_`<br>`FRAMEWORK_PYTHON` | Unset | Explicit interpreter that launches the server for a from-source build (the venv Python the artifact was compiled against).  Set automatically from `FrameworkRuntime.runtime_python_exe` through `apply_runtime_override` into the per-variant YAML `benchmark.envs`.  Both backends export that mapping to the server env; the bypass backend additionally uses this value as the `python -m` interpreter.  Operators normally do not set this by hand. |
