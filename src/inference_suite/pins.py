# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The commits of the tools a base is measured with, the one place each is pinned.

A pin moves only by a pull request that changes it here, never with the head of a tool's ``main``: two sessions
that resolve their bases a day apart still measure them with the same client.
"""

from __future__ import annotations

#: The InferenceX checkout whose harness drives load against every base.
INFERENCEX_CLIENT_COMMIT = "41cc713af82bc69204f572e03c2dd8f60485e6ac"

__all__ = ["INFERENCEX_CLIENT_COMMIT"]
