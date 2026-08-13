# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Optimization action playbooks.

``<name>.md`` holds the human/agent-facing playbook for a subset of actions
(4 of the current set); shipped as package data and read ad hoc by agents,
but never loaded by Python runtime code.

Action metadata lives in
:data:`hyperloom.inference_optimizer.protocol.action_surfaces.ACTION_CATALOGUE`.

The 3 "kernel_agent-owned" actions (kernel_opt / integrate / gemm_tuning)
are reachable only via REQUEST(target_agent="kernel_agent") — PolicyGate
rejects a direct delegate or propose_action of any name in
:data:`hyperloom.inference_optimizer.protocol.action_surfaces.KERNEL_AGENT_OWNED_ACTIONS`.
"""
