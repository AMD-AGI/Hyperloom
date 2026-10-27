# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The long-context check: a frozen set of long synthetic prompts, answered greedily by the candidate's own server.

An approximation in attention (a KV-cache dtype, a window, a top-k set) can leave short and averaged evaluations
untouched and still lose a needle that sits far back in a long prompt. The check sends RULER-style prompts of a fixed
token length (needle retrieval, multi-key retrieval, variable tracking, common-word extraction) and grades each answer by
substring match; the toolbox states the scores and never says pass or fail.

Modules: ``plan`` (what the sheet asks for), ``tasks`` (the generators), ``staging`` and ``resolve`` (the frozen set and
its offline entry, on ``measure.short_prompts.staging``), ``build`` (the command that makes an entry), ``client`` (greedy
completions of text prompts), ``rounds`` (the pass and the file it leaves), ``outputs`` (that file), ``grade`` (the
match), ``held`` (both accuracy checks on the test's own server, before its teardown).
"""
