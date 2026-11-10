# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Audit-trail vocabulary for the ``<action>_attempts`` ledgers on :class:`..shared_state.SharedState`."""

from __future__ import annotations

# audit-action name -> (result-dict key, key_metric_kind); kernel_agent-owned actions excluded (dedicated structures).
_KEY_METRIC_MAP: dict[str, tuple[str, str]] = {
    "baseline": ("output_throughput", "output_throughput"),
    "profile": ("output_throughput", "output_throughput"),
    "explore": ("best_gain_pct", "gain_pct"),
    # ``roofline`` runs profile + trace_analyze atomically.
    "roofline": ("snapshot_id", "snapshot_id"),
}

# Per-action audit trail kinds.
_AUDIT_ACTIONS: frozenset[str] = frozenset(_KEY_METRIC_MAP)
