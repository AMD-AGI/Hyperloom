# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Critic skill package (tree-reform.MD P2.5, promoted from the sibling
``critic-agent/`` checkout).

The deterministic Python layer lives under :mod:`hyperloom.agents.critic.runtime`
and is invoked via ``python -m hyperloom.agents.critic.runtime.cli``; the LLM
skill assets (``SKILL.md``, ``actions/*.md``, ``references/*.md``) live
alongside it in this package.
"""

from __future__ import annotations
