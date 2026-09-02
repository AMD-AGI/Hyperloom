"""Internal driver package for quantization_agent.

Drives the selected Claude/Codex/Hermes session per ``SKILL.md``: single-attempt session
runner, multi-attempt retry orchestrator, artifact collection, eval gap
gating, and outcome classification. Public symbols are re-exported by
``quantization_agent.__init__``; do not import from ``driver`` directly
from outside this package.
"""
