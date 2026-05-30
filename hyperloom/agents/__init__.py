"""Hyperloom agent subpackages.

Each agent is a self-contained subpackage with:
  - SKILL.md (or AGENTS.md): skill definition for Claude sessions
  - runtime/ or tools/: Python runtime code
  - actions/: prompt templates for agent actions
  - tests/: agent-specific test suite

Agents are invoked by the orchestrator via subprocess dispatch:
  python -m hyperloom.agents.critic.runtime.cli <command>
  python -m hyperloom.agents.kernel <command>
  python -m hyperloom.agents.framework.runtime.cli <command>
  python -m hyperloom.agents.robustness <command>

Or by mounting their SKILL.md into a Claude CLI session.
"""

from pathlib import Path

AGENTS_ROOT = Path(__file__).parent


def get_skill_path(agent_name: str) -> Path | None:
    """Get the SKILL.md path for a named agent."""
    agent_dir = AGENTS_ROOT / agent_name
    for candidate in ("SKILL.md", "AGENTS.md"):
        path = agent_dir / candidate
        if path.exists():
            return path
    return None


def get_agent_dir(agent_name: str) -> Path:
    """Get the root directory for a named agent."""
    return AGENTS_ROOT / agent_name


AGENT_NAMES = ["critic", "kernel", "framework", "robustness"]
