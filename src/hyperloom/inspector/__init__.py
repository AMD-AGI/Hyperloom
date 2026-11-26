# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real-time Inspector: a read-only view of a live or finished session directory.

It opens the coordinator database with ``mode=ro`` and tails the
session's JSON/JSONL artefacts by cursor. It never writes into the session directory.
"""
