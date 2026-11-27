---
myst:
    html_meta:
        "description": "Fusion lane variables for Hyperloom: skip switch, wrapper timeout, agent turn cap, and minimum throughput gain required for a queued fusion patch to be KEPT on integration."
        "keywords": "Hyperloom, environment variables, fusion lane, HYPERLOOM_SKIP_FUSION, FORGE_FUSION_TIMEOUT, FORGE_FUSION_MAX_TURNS, HYPERLOOM_FUSION_KEEP_PCT"
---
# Hyperloom Fusion lane

The fusion lane is Coordinator-owned and forge-only: it runs at KERNEL entry on
the forge branch, never as an agent request, and the default `geak` backend
returns before reaching it. Its gate needs a fusion-eligible framework
(`sglang`, `vllm` or `vllm-aiter`), a decode trace to discover from, and no
fusion that already succeeded this session.

| Variable                       | Default                       | Description                                                                                                                                                                                       |
|--------------------------------|-------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `HYPERLOOM_SKIP_FUSION`        | Unset (lane enabled)          | Truthy (`1` / `true` / `yes` / `on`) disables the fusion lane outright, before any other gate is evaluated.                                                                                        |
| `FORGE_FUSION_TIMEOUT`         | `7200` (2h)                   | Wrapper timeout in seconds for one forge-fusion run. Unset or unparseable, the fusion lane's share of the KERNEL phase is used, else the default.                                                  |
| `FORGE_FUSION_MAX_TURNS`       | `100`                         | Agent turn cap handed to forge-fusion for one run.                                                                                                                                                 |
| `HYPERLOOM_FUSION_KEEP_PCT`    | `1.0`                         | Minimum throughput gain, in percent, that a queued fusion patch must reach to be KEPT when it is integrated. Every nominated patch is queued; the value is recorded on each queued record. A malformed value raises `EnvValueError`. |
