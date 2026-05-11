"""Decision layer.

* :mod:`policy_aware` — local mirror of upstream PolicyGate payload
  schema, used to validate intents before they leave the reactor.
* :mod:`action_ladder` (M1) — symptom -> intent translation in the
  observe / diagnose / recommend tiers.
* :mod:`rca_engine` (M1 stub) — LLM-backed RCA, disabled by default in
  M1 and exercised in M2.
"""

from .action_ladder import ActionLadder, ActionLadderConfig, Finding
from .policy_aware import PolicyAware, PolicyViolation

__all__ = [
    "ActionLadder",
    "ActionLadderConfig",
    "Finding",
    "PolicyAware",
    "PolicyViolation",
]
