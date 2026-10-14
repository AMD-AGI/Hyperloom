# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The scheduling side's contracts: what it gives the other modules and what it calls on them.

* :mod:`.quota`: the quota and priority policy the toolbox queues by;
* :mod:`.forks`: fork admission, which the rule gate awaits;
* :mod:`.peers`: provisional protocols of the specialist lifecycle, the toolbox, the rule gate and the experiment
  repository, with only what the scheduler, the integrator and the final selection call, until their owners' land.
"""

from hyperloom_multibranch.contracts.scheduling.forks import AdmissionStatus, ForkAdmission, ForkAdmissionService
from hyperloom_multibranch.contracts.scheduling.peers import (
    ExperimentRepo,
    ForkRequest,
    OperatingPoint,
    PatchConflict,
    QueueStatus,
    RepoMerge,
    RequestWithdrawn,
    RuleGate,
    SpecialistLifecycle,
    SweepResult,
    Toolbox,
)
from hyperloom_multibranch.contracts.scheduling.quota import (
    OUTSIDE_BRANCH_K,
    PRIORITY_ORDER,
    QUOTA_POLICY_VERSION,
    BranchQuota,
    PriorityClass,
    QuotaPolicy,
)

__all__ = [
    "OUTSIDE_BRANCH_K",
    "PRIORITY_ORDER",
    "QUOTA_POLICY_VERSION",
    "AdmissionStatus",
    "BranchQuota",
    "ExperimentRepo",
    "ForkAdmission",
    "ForkAdmissionService",
    "ForkRequest",
    "OperatingPoint",
    "PatchConflict",
    "PriorityClass",
    "QueueStatus",
    "QuotaPolicy",
    "RepoMerge",
    "RequestWithdrawn",
    "RuleGate",
    "SpecialistLifecycle",
    "SweepResult",
    "Toolbox",
]
