# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Session-end finalize layer (L1 + L2).

When the inference_optimizer Coordinator sets ``state.json::stop_reason``
the run is winding down. Robustness has been writing findings to
``<session_dir>/agents/robustness/findings/<session_id>.jsonl`` the
whole time, and the Coordinator has been writing one
``<session_dir>/runs/<action>/<task_id>/result.json`` per executor
invocation. None of these are useful to a post-run reviewer in their
raw form — too many rows, too much repetition.

:class:`PostmortemFinalizer` is the once-per-session aggregator that:

* picks the **flashpoint** finding (first HIGH-severity row) and the
  intents Robustness emitted in response (L1);
* walks every ``runs/*/<task_id>/result.json`` to assemble a uniform
  ``decision_trace.json`` with per-task ``{action, decision, gain_pct,
  error_class, ts, workspace, ...}`` (L2);
* writes both to ``<session_dir>/reports/``; idempotent via the
  ``.robustness_finalized`` marker so it can be safely re-run.
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
