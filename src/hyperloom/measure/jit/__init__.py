# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The shared compiled-code cache: keys, store, build log, per-test session, notes and host publishing (design 4.21).

The kinds of cache are ``hyperloom.measure.jit_kinds``, outside this package: the signed result manifest names their
states, and the store reaches the manifest through ``hyperloom.measure.archive``, so the kinds must not sit above it.
"""
