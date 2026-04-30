"""Conductor + protocol layer (DESIGN v0.6 §7 / §13 / §14 / §15).

The orchestrator package owns:

* MessageBus + IntentEnvelope (A2A transport)
* ResourceLockManager (4 mutex lanes)
* TaskRegistry (DelegatedTask state machine)
* CursorStore (per-agent idempotent replay)
* AgentRole + PolicyGate (added in P0-2)
* Conductor main loop (added in P0-3)
"""
