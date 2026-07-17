# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Inference Optimizer — single-mode 4-agent runtime.

Roles:

* Orchestration — proposes actions, delegates sub-agents, REQUESTs Kernel
* Kernel        — owns 5 deep-kernel actions, responder-only via REQUEST/RESPONSE
* Critic        — reviews proposals (approve/reject/redirect/advise), owns KB
* Robustness    — always-on health monitoring, RCA, recovery, scheduling police
"""

__version__ = "0.6.0"
