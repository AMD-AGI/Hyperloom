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

These AMD Instinct GPUs are validated with Hyperloom:

| GPU | Architecture | Status |
|-----|--------------|--------|
| AMD Instinct™ MI300X GPU | gfx942 | Supported |
| AMD Instinct™ MI308X GPU | gfx942 | Supported |
| AMD Instinct™ MI325X GPU | gfx942 | Supported |
| AMD Instinct™ MI355X GPU | gfx950 | Supported |

```{note}
MI308X and MI325X share the gfx942/CDNA3 runner family with MI300X. Hyperloom
keeps the resolved GPU type distinct (`mi308x` / `mi325x`), but Magpie benchmark
rendering reuses the MI300X runner scripts and image family unless a dedicated
image is supplied.
```

## Inference frameworks

These inference frameworks are supported:

| Framework | Status | Notes |
|-----------|--------|-------|
| SGLang (ROCm) | Supported | Default framework |
| vLLM (ROCm) | Supported | Do not mix frameworks within one session |

## Container images

Pick the image that matches your environment. Public Docker Hub refs
(`primussafe/sglang:<tag>`) are used on your own GPU machine; the
`harbor.crusoe.primus-safe.amd.com/proxy/` prefix is the internal mirror used
inside Primus-SaFE.

| Image | GPU |
|-------|-----|
| `primussafe/sglang:v0.5.11-rocm720-mi30x-profilerfix` | MI300X |
| `primussafe/sglang:v0.5.11-rocm720-mi35x-profilerfix` | MI355X |
| `vllm/vllm-openai-rocm:v0.19.0` | MI300X and MI355X |

Browse all available SGLang tags at
[hub.docker.com/r/primussafe/sglang/tags](https://hub.docker.com/r/primussafe/sglang/tags).

## Component dependencies

These are the Hyperloom components:

| Component | Source |
|-----------|--------|
| TraceLens | <https://github.com/AMD-AGI/TraceLens> |
| Magpie | <https://github.com/AMD-AGI/Magpie> |
| IntelliKit | <https://github.com/AMDResearch/intellikit> |
| GEAK | <https://github.com/AMD-AGI/GEAK> |
| AMD Quark (optional, quantization) | <https://quark.docs.amd.com/> |
