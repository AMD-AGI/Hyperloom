# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Inference Optimizer v0.6 — single-mode 4-agent runtime.

Roles:

* Orchestration — proposes actions, delegates sub-agents, REQUESTs Kernel
* Kernel        — owns 5 deep-kernel actions, responder-only via REQUEST/RESPONSE
* Critic        — reviews proposals (approve/reject/redirect/advise), owns KB
* Robustness    — always-on health monitoring, RCA, recovery, scheduling police

See ``inference_optimizer-DESIGN-v2.md`` for the canonical specification.
"""

__version__ = "0.6.0"
