# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Real-time Inspector: a read-only view of a live or finished session directory.

It opens the coordinator database with ``mode=ro``, tails the
session's JSON/JSONL artefacts by cursor, and redacts secrets before anything
leaves the process. It never writes into the session directory.
"""
