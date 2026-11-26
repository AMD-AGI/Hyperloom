# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which of an EXPLORE round's variants are benchmarked.

The session's own verdicts (``measured_variants``), the opt-in pre-filters run once
before the loop (``_prefilter``: measurement store, InferaSim projection), and
the rule for launches that must always be measured again (``_remeasure``), and the speculative decoding a
synthetic benchmark refuses before launch (``spec_decode_golden``).
"""
