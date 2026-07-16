# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Kernel-agent skill package.

The tools under :mod:`hyperloom.agents.kernel.tools` are consumed as standalone
CLI scripts (``python3 <root>/tools/<tool>.py --args``) via
``HYPERLOOM_KERNEL_AGENT_ROOT``, not as a dotted-import library. This package's
``__init__.py`` files exist so the ``hyperloom.agents.kernel.tools.*`` dotted
path is also importable, without disturbing the ``sys.path``-based script contract.
"""

from __future__ import annotations
