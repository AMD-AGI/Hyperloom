"""Hyperloom Robustness Server.

Bridges Primus-Claw business events (NATS JetStream + KV) and SaFE
Workload sandbox lifecycle with Primus-Robust pod-dimension metrics, so
Hyperloom callers can query session / sub-agent views without baking
session_id into the time series.

Public surface lives under ``robustness_server.api``; runtime services
(NATS consumer, reconciler, store) under ``robustness_server.services``;
persistence under ``robustness_server.store``.
"""

__all__: list[str] = []
