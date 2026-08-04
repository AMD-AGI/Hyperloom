# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Decision layer.

Import concrete modules directly:

* :mod:`policy_aware` — local mirror of upstream PolicyGate payload schema.
* :mod:`action_ladder` — symptom -> intent (observe/diagnose/recommend).
* :mod:`rca_engine` — LLM-backed RCA; auto-enabled once LLM credentials are
  discovered, off via ``llm_rca_enabled=False`` or
  ``ROBUSTNESS_LLM_RCA_DISABLED``, otherwise a no-op engine.

Only the ``PolicyAware`` / ``PolicyViolation`` pair is re-exported here (the
validation contract callers assert against); ``ActionLadder`` / ``Finding`` /
RCA engines are imported from their concrete modules.
"""

from ..role.envelope import PolicyViolation
from .policy_aware import PolicyAware

__all__ = ["PolicyAware", "PolicyViolation"]
