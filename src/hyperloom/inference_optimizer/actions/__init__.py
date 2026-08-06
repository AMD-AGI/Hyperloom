# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optimization action catalogue.

Layout:

* ``_meta/<name>.yaml`` — machine-readable action metadata loaded by
  :class:`hyperloom.orchestrator.actions.registry.ActionRegistry`
* ``<name>.md`` — human/agent-facing playbook for a subset of actions (4 of
  the current set); shipped as package data and read ad hoc by agents, but
  never loaded by Python runtime code — PolicyGate and ActionRegistry read
  only ``_meta/*.yaml``

The 6 "kernel_agent-owned" actions (kernel_opt / integrate /
deep_kernel_analysis / operator_tuning / vendor_kernel_config / gemm_tuning)
are reachable only via REQUEST(target_agent="kernel_agent") — PolicyGate
rejects a direct delegate or propose_action of any name in
:data:`hyperloom.inference_optimizer.protocol.action_surfaces.KERNEL_AGENT_OWNED_ACTIONS`.
"""
