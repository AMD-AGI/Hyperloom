# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Phase state machine + per-phase handler collaborators.

``machine_state.py`` holds the pure phase transition functions and
``PHASE_ALLOWED_ACTIONS``. ``base.py`` provides the ``PhaseHandler`` base class;
the sibling modules (``machine``, ``prelude``, ``sweep``, ``close``,
``internal``, ``kernel_stack``, ``kernel``, ``explore``, ``framework``) each own
one phase's Coordinator method cluster.
"""
