# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Thin leaf sub-agent definition for single-layer specialist fan-out.

A specialist may ``Task(subagent_type="hyperloom-leaf")`` to parallelize a
focused, single-shot sub-task. The leaf has bash + read tools but no ``Task``,
so fan-out depth is fixed at one. Leaves run inside the specialist subprocess
and inherit its ``VISIBLE_DEVICES``, sharing the parent's GPU lease.
"""

from __future__ import annotations

import json

from .system_prompts.specialist_prompt_builder import BASH_KILL_SAFETY_PREAMBLE

LEAF_AGENT_NAME = "hyperloom-leaf"

LEAF_AGENT_TOOLS: tuple[str, ...] = ("Bash", "Read", "Grep", "Glob")

_LEAF_AGENT_PROMPT = (
    "You are a leaf sub-agent dispatched by a specialist for one focused, "
    "single-shot task. Complete it in a single pass: gather what you need "
    "with Read/Grep/Glob/Bash and return a concise, self-contained result. "
    "You cannot dispatch further sub-agents. "
    "Default to CPU-only work; if GPUs were allocated to your parent they are "
    "already exposed via the inherited VISIBLE_DEVICES — use only those and "
    "never the production serving GPU. "
    "As your last action, also write your result to a `leaf_result.json` file "
    "in your working directory so it is captured as a distinct artifact. "
    + BASH_KILL_SAFETY_PREAMBLE
)

_LEAF_AGENT_DESCRIPTION = (
    "Single-turn worker for a specialist: runs one focused sub-task "
    "(e.g. benchmark one candidate, read one subsystem) and returns the result."
)


def build_leaf_agents_json() -> str:
    """Return the ``--agents`` JSON declaring the leaf sub-agent type."""
    return json.dumps(
        {
            LEAF_AGENT_NAME: {
                "description": _LEAF_AGENT_DESCRIPTION,
                "prompt": _LEAF_AGENT_PROMPT,
                "tools": list(LEAF_AGENT_TOOLS),
            }
        }
    )


__all__ = [
    "LEAF_AGENT_NAME",
    "LEAF_AGENT_TOOLS",
    "build_leaf_agents_json",
]
