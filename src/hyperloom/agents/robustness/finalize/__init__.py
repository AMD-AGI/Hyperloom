# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Session-end finalize layer (L1 + L2).

:class:`PostmortemFinalizer` runs once per session (when ``stop_reason`` is
set) to aggregate the raw findings JSONL + per-task ``result.json`` files into:

* L1 — the **flashpoint** (first HIGH-severity finding) and the intents emitted;
* L2 — ``decision_trace.json`` with per-task ``{action, decision, gain_pct,
  error_class, ts, workspace, ...}``.

Both go to ``<session_dir>/reports/``; idempotent via the
``.robustness_finalized`` marker so it is safe to re-run.
"""

from __future__ import annotations

from .postmortem import (
    PostmortemFinalizer,
    PostmortemFinalizerConfig,
    finalize_session,
)


__all__ = [
    "PostmortemFinalizer",
    "PostmortemFinalizerConfig",
    "finalize_session",
]
