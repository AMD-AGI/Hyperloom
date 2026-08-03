# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Coordinator + protocol layer.

The orchestrator package owns:

* MessageBus + IntentEnvelope (A2A transport)
* ResourceLockManager (7 scheduling lanes; serving-side lanes are mutually
  exclusive, research/build lanes are serialization-only)
* TaskRegistry (DelegatedTask state machine)
* CursorStore (per-agent idempotent replay)
* AgentRole + PolicyGate
* Coordinator main loop
"""
