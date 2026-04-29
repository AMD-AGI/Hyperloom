"""Per-agent runtime assets — one subdirectory per CLI agent.

Layout::

    src/inference_optimizer/agents/
        executor/
            agent_card.yaml
            system_prompt.md
            scripts/             (optional, agent-private)
        critic/
            agent_card.yaml
            system_prompt.md
        watchdog/
            agent_card.yaml
            system_prompt.md
        sage/
            agent_card.yaml
            system_prompt.md

This directory replaces the implicit "everyone shares one Backend +
one ``orchestrator/system_prompts/<role>.md``" model with a per-agent
package the launcher can discover at startup. Existing prompts under
``orchestrator/system_prompts/`` remain authoritative until each
``agents/<role>/system_prompt.md`` wraps them; the latter just adds an
``inbox/outbox`` workflow header.
"""

from __future__ import annotations

from pathlib import Path

AGENTS_ROOT = Path(__file__).resolve().parent


def agents_root() -> Path:
    """Return the absolute path to ``src/inference_optimizer/agents/``.

    Override hook: the multi-CLI router consumes the env var
    ``INFERENCE_OPTIMIZER_AGENTS_ROOT`` first when set (tests inject a
    tmp tree this way) before falling back to the package path.
    """
    import os

    override = os.environ.get("INFERENCE_OPTIMIZER_AGENTS_ROOT")
    if override:
        return Path(override)
    return AGENTS_ROOT


__all__ = ["AGENTS_ROOT", "agents_root"]
