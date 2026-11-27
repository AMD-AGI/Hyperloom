# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real-time Inspector: a read-only web view of a live or finished session directory.

It opens the coordinator database with ``mode=ro``, tails the
session's JSON/JSONL artefacts by cursor, redacts secrets before anything
leaves the process, and serves a single static page plus a Server-Sent Events
stream. It never writes into the session directory.
"""

from __future__ import annotations

__all__ = ["serve"]


def serve(*args, **kwargs):  # pragma: no cover - thin re-export
    from hyperloom.inspector.server import serve as _serve

    return _serve(*args, **kwargs)
