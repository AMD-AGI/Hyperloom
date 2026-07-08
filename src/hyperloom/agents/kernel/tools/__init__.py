# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Kernel-agent tools (TraceLens analysis, kernel optimization, patch apply).

These modules are primarily consumed as standalone CLI scripts invoked by
:mod:`hyperloom.orchestrator.kernel.request_handlers` via
``HYPERLOOM_KERNEL_AGENT_ROOT`` (``python3 <root>/tools/<tool>.py --args``),
and rely on ``sys.path``-based sibling imports (e.g.
``sys.path.insert(0, str(Path(__file__).resolve().parent))``) rather than
package-relative imports so they keep working both as a package member and
as a directly-executed script.
"""

from __future__ import annotations
