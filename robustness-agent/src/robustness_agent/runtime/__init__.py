# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Robustness runtime — subprocess-friendly entry point.

The :mod:`robustness_agent.runtime` namespace mirrors the layout of
``critic-agent/runtime/``: a small, dependency-light package whose only
job is to expose a JSON-IO CLI that hosts (Coordinator, smoke harness,
operator tooling) can shell out to.

Hosts construct ``request.json``, invoke
``python -m robustness_agent.runtime.cli tick --request request.json
--out emit.json``, and read the resulting ``emit.json`` whose
``intent_envelope`` field follows the same schema as critic-agent's
``commit-review`` output (and is validated by upstream
``inference_optimizer.protocol.intent.validate_envelope``).

Keeping the agent on the far side of a subprocess boundary is a
deliberate architectural choice — see and the project
handover. The reactor still lives in :mod:`robustness_agent.role` and
:mod:`robustness_agent.signals`; this package is *transport only*.
"""
