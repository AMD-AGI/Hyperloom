# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Robustness runtime — subprocess-friendly entry point.

Dependency-light package exposing a JSON-IO CLI hosts shell out to: construct
``request.json``, run ``python -m robustness_agent.runtime.cli tick --request
request.json --out emit.json``, and read ``emit.json`` whose ``intent_envelope``
follows the same schema as critic-agent's ``commit-review`` (validated by
``inference_optimizer.protocol.intent.validate_envelope``). Transport only;
the reactor lives in :mod:`robustness_agent.role` / :mod:`robustness_agent.signals`.
"""
