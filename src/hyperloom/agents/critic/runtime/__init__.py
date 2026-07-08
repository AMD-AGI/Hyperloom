# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Deterministic Python layer backing the LLM-driven Critic SKILL.

Parses requests, maintains per-session memory, emits Coordinator-compatible
intent envelopes, and mediates KB side effects. Invoked via ``runtime.cli``;
the LLM never calls these modules directly.
"""

from __future__ import annotations

__all__: list[str] = []
