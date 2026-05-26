"""Coordinator + protocol layer ().

The orchestrator package owns:

* MessageBus + IntentEnvelope (A2A transport)
* ResourceLockManager (4 mutex lanes)
* TaskRegistry (DelegatedTask state machine)
* CursorStore (per-agent idempotent replay)
* AgentRole + PolicyGate (added in P0-2)
* Coordinator main loop (added in P0-3)
"""
