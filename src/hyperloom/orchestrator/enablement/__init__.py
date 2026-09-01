# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Enablement: make a ``(model, backend)`` combination runnable at all.

Phase-orthogonal by construction. The Coordinator pumps this subsystem on
every tick regardless of ``state.phase`` (``coordinator.py`` calls
``_pump_enablement_safely`` from both the tick and run loops), because the
failure it repairs -- a combination that cannot boot, or boots and misses its
accuracy floor -- traps the run in PRELUDE, before any phase that could host
the repair has been entered. It lived inside the FRAMEWORK_AGENT phase handler
for historical reasons only; a comment on the pump records that housing it in
the perf pump made it unreachable for exactly the case it exists to fix.

The four collaborators split the lifecycle: :mod:`params` builds the authoring
specialist's request, :mod:`lane` owns round admission and re-arm,
:mod:`build` owns the off-loop compiled-build escalation and its outcome
routing, and :mod:`revalidation` owns the genuine-baseline re-measurement that
finalises an eval-origin KEEP.
"""
