# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Framework-arm identity read off a candidate id.

The arm has no runtime type; it is recognised by string markers, and these helpers are the one reading of them.
"""

from __future__ import annotations

LOCAL_EXPLORE_CANDIDATE_PREFIX: str = "local_explore:"


def is_local_explore_candidate(candidate_id: str) -> bool:
    """True when the candidate id was minted by the candidate-free local arm."""
    return str(candidate_id).startswith(LOCAL_EXPLORE_CANDIDATE_PREFIX)


__all__ = [
    "LOCAL_EXPLORE_CANDIDATE_PREFIX",
    "is_local_explore_candidate",
]
