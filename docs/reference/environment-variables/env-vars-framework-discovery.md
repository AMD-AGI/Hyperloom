---
myst:
    html_meta:
        "description": "Framework and source-tree discovery variables for Hyperloom: source search roots, rescue paths, aiter JIT directory, sglang patch version pins, and framework preflight gate bypass."
        "keywords": "Hyperloom, environment variables, framework discovery, INFERENCE_OPTIMIZER_FRAMEWORK_SOURCE_ROOTS, HYPERLOOM_ENABLE_PATCH, HYPERLOOM_SKIP_FRAMEWORK_CHECK, AITER_REF, ATOM_REF"
---
# Hyperloom framework discovery

The following variables configure framework source discovery and path overrides.

| Variable                                          | Default                                                                | Description                                                                                                                                            |
|---------------------------------------------------|------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `INFERENCE_`<br>`OPTIMIZER_`<br>`FRAMEWORK_`<br>`SOURCE_ROOTS`      | Union with `/sgl-workspace`<br>`/{aiter,sglang`<br>`,vllm}`                        | Colon-separated supplement to the search roots returned by `hyperloom.inference_optimizer.framework_paths.resolve_kernel_search_roots`, used for source search and flag discovery. Populated automatically by `src/hyperloom/inference_optimizer/assets/install.sh`'s `_probe_framework_source_roots` step (using `hyperloom.inference_optimizer.framework_paths.probe_framework_source_roots_for_env`).   |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_RESCUE_PATHS`                | Unset                                                                  | Colon-separated list of extra directories the harvest step scans for stray `result.json` files written outside the session dir (InferenceX-native scripts that hardcode `--result-dir`). |
| `INFERENCE_`<br>`OPTIMIZER`<br>`_AITER_JIT_DIR`               | Aiter default                                                          | Override the aiter just-in-time (JIT) cache root. See [Targeted builds (Rung 5)](env-vars-targeted-builds.md).  |
| `HYPERLOOM_`<br>`SGLANG_PA`<br>`TCH_EXACT`<br>`_VERSIONS`           | Unset                                                                  | Pin the sglang server-patch step to specific upstream versions; advanced compatibility option.                                                          |
| `HYPERLOOM_`<br>`ENABLE`<br>`_PATCH`                          | `1`                                                                    | Set to `0` to skip the in-place server patch step (useful when the upstream is already pre-patched).                                                    |
| `HYPERLOOM_`<br>`SKIP_FRAME`<br>`WORK_CHECK`                  | Unset (check enabled)                                                  | Truthy skips the `optimize` preflight gate that requires the selected serving framework to be importable and a ROCm build. Last resort: when the server runs elsewhere, set `BENCHMARK_BASE_URL` instead, which exempts the check and configures the supported path. The gate already stays out of the way for `xdit`/`custom` (server-less) and external multi-node. |
| `AITER_REF` | Unset | Optional bare-metal AITER install pin. When unset, the installer selects the newest tag compatible with the installed torch/triton stack. |
| `ATOM_REF` | `fe1099b15ddc52e6873e1932935f722864cdeee0` | Full commit SHA that `--install-framework atom` checks out from `ATOM_REPO` (default `https://github.com/ROCm/ATOM.git`) into `ATOM_ROOT` (default `<deps root>/atom`). The default is the commit `rocm/atom-dev:v0.1.7-rc0` was built from. |
