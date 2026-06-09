# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Persistent findings sink — M1 writes JSONL under the session dir; M5
augments with an HTTP POST to robustness-server."""

from .sink import FindingSink, FindingSinkConfig

__all__ = ["FindingSink", "FindingSinkConfig"]
