# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The short-prompt consistency check: a frozen set of short prompts, sent greedily to several fresh server processes.

Some defects show only on short sequences and only in some launches of a server (a prefill kernel that is wrong for a
few tokens in about one process in three), so a long, batched, averaged evaluation does not see them. The check boots
the candidate's own launch several times, sends every launch the same prompts, and records each output; comparing the
outputs across the launches, and with the parent's, is the toolbox's job (``compare``). Nothing here says pass or fail.

Modules: ``plan`` (what the sheet asks for), ``prompts`` and ``staging`` (the frozen set and its offline entry),
``build`` (the command that makes an entry), ``client`` (greedy completions over HTTP), ``compare`` (the outputs of
several processes against each other and against a reference), ``rounds`` (the passes and the file they leave).
"""
