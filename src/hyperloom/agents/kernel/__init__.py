# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Kernel-agent skill package (tree-reform.MD P2.5, promoted from the sibling
``kernel-agent/`` checkout).

Unlike the other agent packages, the tools under
:mod:`hyperloom.agents.kernel.tools` are consumed as standalone CLI scripts
(``python3 <root>/tools/<tool>.py --args``) via ``HYPERLOOM_KERNEL_AGENT_ROOT``
(see ``src/hyperloom/orchestrator/kernel/request_handlers.py``), not as a
dotted-import library. This package's ``__init__.py`` files exist so the
``hyperloom.agents.kernel.tools.*`` dotted path is also importable when
needed, without disturbing the existing ``sys.path``-based script contract.
"""

from __future__ import annotations
