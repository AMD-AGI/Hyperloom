# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Critic runtime adapter package.

This package provides the deterministic Python layer that backs the
LLM-driven Critic SKILL. Responsibilities:

* Parse Coordinator-style inbox prompts and dialogue-style decision
  requests into a uniform :class:`CriticRequest` (``request_models``).
* Maintain per-session memory across calls (``session_memory``).
* Translate Critic JSON outputs into Coordinator-compatible intent
  envelopes (``intent_envelope``).
* Mediate KB read/write side effects with retries, dead-lettering and
  metrics (``kb_client``, ``kb_writer``, ``dead_letter``, ``metrics``).
* Compose the prepare-review / commit-review / init-session /
  close-session flows expected by the Critic actions (``decision_reviewer``).

The runtime is invoked from the Critic SKILL via the CLI in
``runtime.cli``; the LLM never calls these modules directly.
"""

from __future__ import annotations

__all__: list[str] = []
