# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scheduling side's contracts: what it gives the other modules and what it calls on them.

* :mod:`.forks`: fork admission, which the rule gate awaits;
* :mod:`.peers`: provisional protocols of the specialist lifecycle, the rule gate and the experiment repository, with
  only what the scheduler, the integrator and the final selection call, until their owners' land.
"""

from hyperloom_multibranch.contracts.scheduling.forks import AdmissionStatus, ForkAdmission, ForkAdmissionService
from hyperloom_multibranch.contracts.scheduling.peers import (
    ExperimentRepo,
    ForkRequest,
    PatchConflict,
    RepoMerge,
    RuleGate,
    SpecialistLifecycle,
)

__all__ = [
    "AdmissionStatus",
    "ExperimentRepo",
    "ForkAdmission",
    "ForkAdmissionService",
    "ForkRequest",
    "PatchConflict",
    "RepoMerge",
    "RuleGate",
    "SpecialistLifecycle",
]
