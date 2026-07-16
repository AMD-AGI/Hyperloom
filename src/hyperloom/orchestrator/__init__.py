# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator + protocol layer.

The orchestrator package owns:

* MessageBus + IntentEnvelope (A2A transport)
* ResourceLockManager (4 mutex lanes)
* TaskRegistry (DelegatedTask state machine)
* CursorStore (per-agent idempotent replay)
* AgentRole + PolicyGate
* Coordinator main loop
"""
