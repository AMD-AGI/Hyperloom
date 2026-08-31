# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Learning module — self-evolving knowledge base.

Two mechanisms that make agents stronger with every experiment:

  1. TuningDatabase: config→performance lookup that grows with every benchmark
  2. PostMortem: extract lessons from completed experiments

Tied together by AutoEvolver, which hooks into the iteration lifecycle:
  - AFTER benchmark → log to tuning DB
  - AFTER experiment → run postmortem, discover transfer rules
"""
