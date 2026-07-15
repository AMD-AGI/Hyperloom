# ROCm Hyperloom

Hyperloom is an agentic system for optimizing LLM inference on AMD GPUs. Given
a workload, it profiles the runtime, explores candidate changes, benchmarks each
change against the real workload, and promotes only validated improvements. Its
search strategy is based on **[Arbor](https://arxiv.org/abs/2606.12563)** \[1\].

<p align="center"><img width="600" alt="Hyperloom Architecture" src="slides/hyperloom_loop.png" /></p>

Hyperloom combines:

- Trace analysis through [TraceLens](https://github.com/AMD-AGI/TraceLens) and
  [Magpie](https://github.com/AMD-AGI/Magpie).
- Kernel optimization through KernelForge, GEAK, Claude, Codex, and Cursor
  backends.
- A validated optimization loop that writes reproducible artifacts and
  `session_breakdown.json` for downstream consumers.

## Get Started

| Goal | Guide |
|------|-------|
| Start through the hosted UI | [Hosted UI quickstart](docs/install/quickstart.md) |
| Run directly on a ROCm host | [Bare-metal quickstart](docs/install/bare-metal.md) |
| Launch and monitor an optimization | [Run an optimization](docs/how-to/optimize.md) |
| Understand the algorithm | [Optimization loop](docs/conceptual/optimization-loop.md) |

## Documentation

| Topic | Link |
|-------|------|
| Authentication and credentials | [Authentication & credentials](docs/reference/authentication.md) |
| Environment variables | [Environment variables](docs/reference/environment-variables.md) |
| Components | [Components](docs/components/index.md) |
| Compatibility | [Compatibility matrix](docs/compatibility.md) |
| Troubleshooting | [Troubleshooting](docs/reference/troubleshooting.md) |
| Operations | [Operations & self-hosting](docs/reference/operations.md) |
| Session output schema | [`session_breakdown.json`](docs/reference/session-breakdown.md) |

---

## Hosted Tier — Limits & Pricing

The hosted [Hyperloom UI / PrimusClaw](https://crusoe.primus-safe.amd.com/hyperloom/)
is operated by AMD on shared infrastructure. Defaults for the public
PrimusClaw tier:

| Resource                          | Hosted default                                                                 |
|-----------------------------------|---------------------------------------------------------------------------------|
| Per-session GPU budget            | 1–8 × MI300X / MI308X / MI325X / MI355X for single-node runs (matches TP); 16+ GPUs via RayJob for multi-node (nodes ≥ 2) |
| Concurrent sessions per account   | 2                                                                               |
| Session wall-clock                | 24 hours (extensible on request)                                                |
| `USER_DATA_PATH` quota            | 200 GB per session, with daily snapshots                                        |
| LLM-gateway request rate          | Bound to your `SAFE_API_KEY` quota (see [LLM Gateway](https://llm.amd.com/))     |
| Outbound network                  | Allowlisted (model registries, HuggingFace, GitHub)                             |

Pricing for the hosted tier is currently **free for AMD-internal users
and approved AMD partners** via Primus-SaFE. Public / enterprise
pricing is under active definition by the BRAIN team; reach out via
the [Hyperloom support form](https://crusoe.primus-safe.amd.com/hyperloom/)
or open an issue if your organization needs a quote.

For higher limits, dedicated capacity, or air-gapped deployment, self-host
Hyperloom in your own cluster following [`docs/reference/operations.md`](docs/reference/operations.md).
Self-hosted Hyperloom is MIT licensed (see
[Licensing](#licensing)); there is no per-seat or per-session fee.

---

## Developer Entry Points

- Runtime package: `src/hyperloom/`
- Main agent instructions: [`src/hyperloom/inference_optimizer/SKILL.md`](src/hyperloom/inference_optimizer/SKILL.md)
- CLI entry point: `inference_optimizer optimize`
- Operator tools: `python -m hyperloom.inference_optimizer.tools.*`
- Documentation source: `docs/`

For contribution workflow, testing, and linting, see
[`CONTRIBUTING.md`](CONTRIBUTING.md).

---

## References

\[1\] Prakriya, N., Hou, C., Gong, Z., Zhao, H., Zhao, X., Li, M., Gu, Z., & Barsoum, E. (2026). **Arbor: Tree Search as a Cognition Layer for Autonomous Agents**. arXiv:2606.12563. https://arxiv.org/abs/2606.12563

---

## Licensing

Hyperloom is released under the **MIT License**. The full license text
is in [`LICENSE`](LICENSE).

You may use Hyperloom commercially, modify it, and distribute it under
the terms of the MIT license, provided the copyright notice and the
permission notice are retained in all copies or substantial portions of
the software.

Third-party tools and agents (Cursor, Visual Studio, and Claude Code)
that Hyperloom invokes are governed by their own separate license terms
and are NOT covered by the MIT license above — see the "Third-Party
Tools and Agents" section in [`LICENSE`](LICENSE). You are responsible
for reviewing and complying with each tool's individual license.

For security-relevant issues, see [`SECURITY.md`](SECURITY.md). For
contribution conventions, see [`CONTRIBUTING.md`](CONTRIBUTING.md).

