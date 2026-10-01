# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Resumable driver for a Meta RSI round: data, findings, changes, A/B validation and report.

Judgment steps (findings, implementing a change, diagnosing an A/B difference, writing the
report) run a Claude Code agent; every other step is a deterministic script. Run it with
``PYTHONPATH=<repo>/src:<repo>/scripts python -m meta_rsi.rsi --help``.
"""
