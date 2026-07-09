---
myst:
    html_meta:
        "description": "Get started with Hyperloom quickly using the hosted Primus-Claw UI. No local GPU setup required — launch an optimization run directly from your browser."
        "keywords": "Hyperloom, quickstart, Primus-Claw, hosted UI, AMD GPU, LLM inference, ROCm, GEAK, TraceLens, optimization"
---
# Hyperloom quickstart — hosted UI (AMD-internal users)

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
[Hyperloom support form](https://crusoe.example-internal-host.invalid/hyperloom/) to
request access or open an issue in the
[Hyperloom repository](https://github.com/AMD-AGI/Hyperloom/issues).
```

## Prerequisites

Bind your [LLM Gateway](https://llm.amd.com/) key to
[Hyperloom](https://crusoe.example-internal-host.invalid/hyperloom/) to obtain your API
key. This key provides access to TraceLens, GEAK, and OOB services.

## Start a run

1. Go to [crusoe.example-internal-host.invalid/hyperloom](https://crusoe.example-internal-host.invalid/hyperloom/).
2. Enter your email or NTID in the **Request Access** prompt if this is your first time using the site. Click **Join**. You'll be granted access within two hours. Once you have access, you can sign in with Okta SSO.
3. Select **Claw Agent** or **Get Started** to enter Primus-Claw.

      ![primus-landing](../images/hyperloom_landing.png)

4. Select the tab that matches your task:
   - **Hyperloom** — end-to-end model performance optimization.

      ![hyperloom-tab](../images/hyperloom_claw_v2.png)

   - **TraceLens-only** — performance and gap analysis and bridge planning.

      ![trace-tab](../images/tracelens_quickstart.png)

   - **GEAK-only** — kernel optimization.

      ![geak-tab](../images/geak_quickstart.png)

5. Provide your workload and launch.


## Hosted tier limits

| Resource | Hosted default |
|----------|---------------|
| GPUs per session | 1–8 × MI300X / MI308X / MI325X / MI355X (single-node); 16+ via RayJob (multi-node) |
| Concurrent sessions per account | 2 |
| Session wall-clock | 24 hours (extensible on request) |
| `USER_DATA_PATH` quota | 200 GB per session, with daily snapshots |
| LLM-gateway request rate | Bound to your `SAFE_API_KEY` quota |
| Outbound network | Allowlisted (model registries, HuggingFace, GitHub) |

The hosted tier is currently free for AMD-internal users and approved AMD
partners via Primus-SaFE. For higher limits, dedicated capacity, or air-gapped
deployment, see [Hyperloom self-hosting and operations guide](https://github.com/AMD-AGI/Hyperloom/blob/main/docs/reference/operations.md).

## Next steps

- To run Hyperloom on your own AMD GPU hardware instead, see
  [Local Mode quickstart](local-mode.md) or [Bare-metal quickstart](bare-metal.md).
- [Run a Hyperloom optimization](../how-to/optimize.md).
