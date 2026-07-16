# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Robustness runtime — subprocess-friendly entry point.

Dependency-light package exposing a JSON-IO CLI hosts shell out to: construct
``request.json``, run ``python -m robustness_agent.runtime.cli tick --request
request.json --out emit.json``, and read ``emit.json`` whose ``intent_envelope``
follows the same schema as critic-agent's ``commit-review`` (validated by
``hyperloom.inference_optimizer.protocol.intent.validate_envelope``). Transport only;
the reactor lives in :mod:`robustness_agent.role` / :mod:`robustness_agent.signals`.
"""
