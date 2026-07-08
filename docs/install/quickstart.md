---
myst:
    html_meta:
        "description": "Get started with Hyperloom quickly using the hosted Primus-Claw UI. No local GPU setup required — launch an optimization run directly from your browser."
        "keywords": "Hyperloom, quickstart, Primus-Claw, hosted UI, AMD GPU, LLM inference, ROCm, GEAK, TraceLens, optimization"
---
# Quickstart — hosted UI (Primus-Claw)

The fastest way to start is through the hosted Hyperloom web interface,
powered by Primus-Claw. Any team member can launch an optimization run from a
browser — no local GPU setup or environment configuration required.

- **Easy to scale** — each job runs in isolated sandboxed containers. Single-node
  optimizations run in-sandbox; multi-node workloads fan out using RayJob for
  distributed benchmarking.
- **Data flywheel** — every run feeds results back through Minio storage and
  Langfuse observability, creating a closed feedback loop that continuously
  improves the agent's knowledge base and scoring heuristics.
- **Full Skills support** — sandboxes load optimization Skills on demand, giving
  the agent the same profiling, kernel-rewrite, and domain-specific capabilities
  at cloud scale.

```{note}
Primus-Claw is currently available to AMD-internal users and approved AMD
partners only. If you are an external user and need access, use the
[Hyperloom support form](https://crusoe.primus-safe.amd.com/hyperloom/) to
request access or open an issue in the
[Hyperloom repository](https://github.com/AMD-AGI/Hyperloom/issues).
```

## Prerequisites

Bind your [LLM Gateway](https://llm.amd.com/) key to
[Hyperloom](https://crusoe.primus-safe.amd.com/hyperloom/) to obtain your API
key. This key provides access to TraceLens, GEAK, and OOB services.

## Start a run

1. Go to [crusoe.primus-safe.amd.com/hyperloom](https://crusoe.primus-safe.amd.com/hyperloom/).

  ![primus-landing](../images/hyperloom_landing.png)

2. Select **Claw Agent** or **Get Started** to enter Primus-Claw.
3. Select the tab that matches your task:
   - **Hyperloom** — end-to-end model performance optimization.

    ![hyperloom-tab](../images/hyperloom_claw_v2.png)
  
   - **TraceLens-only** — performance and gap analysis and bridge planning.

    ![trace-tab](../images/tracelens_quickstart.png)
  
   - **GEAK-only** — kernel optimization.

    ![geak-tab](../images/geak_quickstart.png)

4. Provide your workload and launch.


## Hosted tier limits

| Resource | Hosted default |
|----------|---------------|
| GPUs per session | 1–8 × MI300X / MI325X / MI355X (single-node); 16+ via RayJob (multi-node) |
| Concurrent sessions per account | 2 |
| Session wall-clock | 24 hours (extensible on request) |
| `USER_DATA_PATH` quota | 200 GB per session, with daily snapshots |
| LLM-gateway request rate | Bound to your `SAFE_API_KEY` quota |
| Outbound network | Allowlisted (model registries, HuggingFace, GitHub) |

The hosted tier is currently free for AMD-internal users and approved AMD
partners via Primus-SaFE. For higher limits, dedicated capacity, or air-gapped
deployment, see [Hyperloom self-hosting and operations guide](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/reference/operations.md).

## Optional — quantization (AMD Quark)

Use `--quantize` when you want to optimize a model that is not already available
in a quantized format (FP8 or MXFP4). Quantizing reduces VRAM consumption and
typically increases throughput on AMD Instinct hardware.

The optional `--quantize` prelude requires an [AMD Quark](https://quark.docs.amd.com/)
checkout at runtime. Set `QUARK_ROOT` to point at it. See the Hyperloom README's
[quantization section](https://github.com/AMD-AGI/Hyperloom/blob/main/README.md#quantization-optional-amd-quark-dependency)
for more information.

## Next steps

To run Hyperloom on your own AMD GPU hardware instead, see
[Install Hyperloom](hyperloom-installation.md).
