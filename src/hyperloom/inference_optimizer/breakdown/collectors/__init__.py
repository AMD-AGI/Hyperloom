# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Deterministic collectors for ``session_breakdown.json``.

Each ``collect_<section>`` is a pure function over ``session_dir`` / ``state`` /
``manifest`` returning its schema section (see :mod:`..schema`). Collectors never
mutate state, fabricate values, or raise — failures are recorded in ``warnings``
and the section returns a best-effort partial.

What is left here is what the export cannot be told at author time: the session
and workload configuration it was launched with, the langfuse receipt another
process wrote, and the projection of the recorder's own fragments and durable
events into the shape the report reads.
"""

from __future__ import annotations

from ._common import (
    _load_json_safe as _load_json_safe,
    _load_jsonl_safe as _load_jsonl_safe,
    _to_float as _to_float,
    _to_int as _to_int,
    _safe_get as _safe_get,
    _parse_iso_unix as _parse_iso_unix,
)
from .sessions import (
    log as log,
    _detect_image_for_session as _detect_image_for_session,
    _close_phase_stop_reason as _close_phase_stop_reason,
    _should_use_close_stop_reason as _should_use_close_stop_reason,
    _collect_recovery as _collect_recovery,
    collect_session as collect_session,
    session_elapsed_minutes as session_elapsed_minutes,
    collect_workload as collect_workload,
    collect_model_info as collect_model_info,
)
from .langfuse import collect_langfuse as collect_langfuse
from .v6 import (
    collect_v6_metadata as collect_v6_metadata,
    langfuse_block as langfuse_block,
    collect_v6_outcome as collect_v6_outcome,
    collect_v6_timeline as collect_v6_timeline,
)
from .v6_close import collect_v6_close as collect_v6_close
from .v6_critic import collect_v6_critic as collect_v6_critic
from .v6_robustness import collect_v6_robustness as collect_v6_robustness

__all__ = [
    "collect_langfuse",
    "collect_model_info",
    "collect_session",
    "collect_v6_close",
    "collect_v6_critic",
    "collect_v6_metadata",
    "collect_v6_outcome",
    "collect_v6_robustness",
    "collect_v6_timeline",
    "collect_workload",
    "langfuse_block",
    "session_elapsed_minutes",
]
