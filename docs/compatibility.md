---
myst:
  html_meta:
    "description": "Compatibility matrix for Hyperloom: supported AMD Instinct GPUs, inference frameworks (SGLang, vLLM), container images, and component dependencies."
    "keywords": "Hyperloom, compatibility, AMD Instinct, MI300X, MI355X, SGLang, vLLM, ROCm, container images, GPU support"
---

# Hyperloom compatibility matrix

This topic lists the hardware, inference frameworks, and container images that
Hyperloom is validated against.

```{note}
This matrix tracks the currently validated combinations. Other ROCm versions
or framework builds might work but are not regularly tested.
```

## GPU support

The following AMD Instinct GPUs are validated with Hyperloom:

| GPU | Architecture | Status |
|-----|--------------|--------|
| AMD Instinct™ MI300X GPU | gfx942 | Supported |
| AMD Instinct™ MI325X GPU | gfx942 | Supported |
| AMD Instinct™ MI355X GPU | gfx950 | Supported |

```{note}
MI325X shares the gfx942/CDNA3 runner family with MI300X. Hyperloom
keeps the resolved GPU type distinct (`mi325x`), but Magpie benchmark
rendering reuses the MI300X runner scripts and image family unless a dedicated
image is supplied.
```

## Inference frameworks

The following inference frameworks are supported:

| Framework | ROCm version | Status | Notes |
|-----------|--------------|--------|-------|
| SGLang (ROCm) | 7.2.0 | Supported | Default framework |
| vLLM (ROCm) | 7.2.0 | Supported | Do not mix frameworks within one session |
| Atom (ROCm) | 7.2.0 | Supported | Single-node only (multi-node rejected by the IR-8 guard) |
| xDiT (diffusion) | 7.2.0 | Supported | Scriptable diffusion pipeline (no serving server). Internal throughput is tracked in img/s, but the primary session-facing metric is end-to-end latency `e2el_mean_ms` (ms). |

## Container images

Pick the image that matches your environment. Public Docker Hub refs
(`primussafe/sglang:<tag>`) are used on your own GPU machine; the
`harbor.crusoe.primus-safe.amd.com/proxy/` prefix is the internal mirror used
inside Primus-SaFE.

| Image | GPU |
|-------|-----|
| `primussafe/sglang:v0.5.12-rocm720-mi30x-profilerfix` | MI300X / MI325X |
| `primussafe/sglang:v0.5.12-rocm720-mi35x-profilerfix` | MI355X |
| `primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix` | MI300X / MI325X / MI355X |

Browse all available SGLang tags at
[hub.docker.com/r/primussafe/sglang/tags](https://hub.docker.com/r/primussafe/sglang/tags).

## Bare-metal recommended environment

For `baremetal` setup, align the host to this combination before running setup.
Hyperloom does not install ROCm or torch itself.

| Item | Recommended | Notes |
|------|-------------|-------|
| ROCm | 7.2.x | Matches the validated framework stacks above. |
| Python | 3.12 | Required by the vLLM ROCm wheel. |
| ROCm torch | ROCm build matching the host ROCm | Preinstalled by the operator; not managed by Hyperloom. |
| SGLang | v0.5.12 | Installed in `shared` mode (reuses the host torch). |
| vLLM | isolated venv | vLLM's ROCm wheel pins its own torch, so it installs into a dedicated venv (`--framework-env isolated`, the default for vLLM) and never touches the host torch. |

For a fully validated, pre-aligned vLLM stack, prefer `docker` mode with
`primussafe/vllm-openai-rocm:v0.21.0-rocm720-profilerfix` — the bare-metal vLLM
wheel index only publishes rolling versions, so exact `v0.21.0` parity is
available through the container image, not pip.

Framework versions are overridable via env (`SGLANG_REF`, `VLLM_VERSION`,
`VLLM_ROCM_VARIANT`) for hosts that need a different pinned stack.

## Component dependencies

Hyperloom is dependent on the following components:

| Component | Source |
|-----------|--------|
| TraceLens | <https://github.com/AMD-AGI/TraceLens> |
| Magpie | <https://github.com/AMD-AGI/Magpie> |
| IntelliKit | <https://github.com/AMDResearch/intellikit> |
| GEAK | <https://github.com/AMD-AGI/GEAK> |
| AgentKernelArena (optional; not in the default install / optimization loop) |<https://github.com/AMD-AGI/AgentKernelArena> |
| AMD Quark (optional, quantization) | <https://quark.docs.amd.com/> |
