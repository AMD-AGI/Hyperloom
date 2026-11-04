# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The native build service: turns a component's patch stack into a signed entry of the build store
(:mod:`hyperloom.measure.jit.native`).

``service`` coalesces requests of one key under its lock and signs what a build made, ``builder`` applies the stack to a
copy of the pinned upstream tree and compiles it, ``harvest`` takes the declared outputs out, ``recipes`` hold each
component's compile facts and ``parity`` compares an unpatched build with the image's own binary.
"""
