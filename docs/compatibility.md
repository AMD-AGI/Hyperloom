# Compatibility Matrix

This page lists the hardware, inference frameworks, and container images that
Hyperloom is validated against.

```{note}
This matrix tracks the currently validated combinations. Other ROCm versions
or framework builds may work but are not regularly tested.
```

## GPU support

| GPU | Architecture | Status |
|-----|--------------|--------|
| AMD Instinct MI300X | gfx942 | Supported |
| AMD Instinct MI308X | gfx942 | Supported |
| AMD Instinct MI325X | gfx942 | Supported |
| AMD Instinct MI355X | gfx950 | Supported |

```{note}
MI308X and MI325X share the gfx942/CDNA3 runner family with MI300X. Hyperloom
keeps the resolved GPU type distinct (`mi308x` / `mi325x`), but Magpie benchmark
rendering reuses the MI300X runner scripts and image family unless a dedicated
image is supplied.
```

## Inference frameworks

| Framework | Status | Notes |
|-----------|--------|-------|
| SGLang (ROCm) | Supported | Default framework |
| vLLM (ROCm) | Supported | Do not mix frameworks within one session |

## Container images

Pick the image that matches your environment. Public Docker Hub refs
(`primussafe/sglang:<tag>`) are used on your own GPU machine; the
`harbor.core42.example-internal-host.invalid/proxy/` prefix is the internal mirror used
inside Primus-SaFE.

| Image | GPU |
|-------|-----|
| `primussafe/sglang:v0.5.11-rocm720-mi30x-profilerfix` | MI300X |
| `primussafe/sglang:v0.5.11-rocm720-mi35x-profilerfix` | MI355X |
| `vllm/vllm-openai-rocm:v0.19.0` | MI300X / MI355X |

Browse all available SGLang tags at
[hub.docker.com/r/primussafe/sglang/tags](https://hub.docker.com/r/primussafe/sglang/tags).

## Component dependencies

| Component | Source |
|-----------|--------|
| TraceLens | <https://github.com/AMD-AGI/TraceLens> |
| Magpie | <https://github.com/AMD-AGI/Magpie> |
| IntelliKit | <https://github.com/AMDResearch/intellikit> |
| GEAK | <https://github.com/AMD-AGI/GEAK> |
| AMD Quark (optional, quantization) | <https://quark.docs.amd.com/> |
