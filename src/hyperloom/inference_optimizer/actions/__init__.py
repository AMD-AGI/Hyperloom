# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optimization action catalogue.

Layout:

* ``_meta/<name>.yaml`` — machine-readable action metadata loaded by
  :class:`hyperloom.orchestrator.actions.registry.ActionRegistry`
* ``<name>.md`` — agent-facing playbook (loaded lazily by SubAgentRunner
  when composing a sub-agent prompt; not required for PolicyGate)

The 5 "kernel_agent-owned" actions (kernel_opt / integrate /
deep_kernel_analysis / operator_tuning / vendor_kernel_config) are reachable
only via REQUEST(target_agent="kernel_agent") — PolicyGate rejects direct
delegate of any name in
:data:`hyperloom.orchestrator.policy.gate.KERNEL_AGENT_OWNED_ACTIONS`.
"""
