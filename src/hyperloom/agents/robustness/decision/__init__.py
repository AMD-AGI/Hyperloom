# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Decision layer.

Import concrete modules directly:

* :mod:`policy_aware` — local mirror of upstream PolicyGate payload schema.
* :mod:`action_ladder` — symptom -> intent (observe/diagnose/recommend).
* :mod:`rca_engine` — LLM-backed RCA, disabled by default in M1.

Only the ``PolicyAware`` / ``PolicyViolation`` pair is re-exported here (the
validation contract callers assert against); ``ActionLadder`` / ``Finding`` /
RCA engines are imported from their concrete modules.
"""

from ..role.envelope import PolicyViolation
from .policy_aware import PolicyAware

__all__ = ["PolicyAware", "PolicyViolation"]
