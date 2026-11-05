# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The native build store: compiled outputs of a component's patched C++ tree, built once per key and signed.

``key`` names a build by exactly its inputs, ``record`` is the signed manifest of one, ``trust`` signs and checks it,
``store`` keeps entries and blobs and coalesces concurrent builds of one key, ``builder`` and ``harvest`` turn a patch
stack into an entry, ``recipes`` hold the per-component compile facts and ``parity`` compares an unpatched build with the
image's own binary.
"""
