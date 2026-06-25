# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Coordinator + protocol layer.

The orchestrator package owns:

* MessageBus + IntentEnvelope (A2A transport)
* ResourceLockManager (4 mutex lanes)
* TaskRegistry (DelegatedTask state machine)
* CursorStore (per-agent idempotent replay)
* AgentRole + PolicyGate
* Coordinator main loop
"""
