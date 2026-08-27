# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compatibility alias for the agent backends, which now live in ``forge_llm``.

They moved so the fusion pipeline could reach them while it still shipped as a
standalone wheel that must never import ``kernelforge``; it has since been
absorbed, but ``forge_llm`` remains their home and the one every caller should
use. This alias stays because the provider entry-point group is a published
extension point, still named
``kernelforge.agent_providers``, and a third-party plugin registered against
it almost certainly imports its base classes from here. Without the alias those
plugins would fail to import, and ``discover_agent_providers`` records a plugin
failure as one log line rather than raising -- so the loss would be silent.

The submodules are aliased into ``sys.modules`` as well, because callers import
``kernelforge.agent_backends.registry`` and friends by path, not only the
names re-exported here. Binding the same module objects keeps the registry a
singleton: a provider registered through either path is visible from both.

New code should import from ``forge_llm.agent_backends`` directly.
"""

from __future__ import annotations

import importlib
import sys

from forge_llm.agent_backends import *  # noqa: F401,F403
from forge_llm.agent_backends import __all__ as _FORGE_LLM_ALL

_SUBMODULES = ("base", "claude", "codex", "registry", "session_resume")

for _name in _SUBMODULES:
    # Importing the provider modules here is safe: each one defers its agent SDK
    # to the call that needs it, so this costs nothing on an install that has
    # neither.
    sys.modules[f"{__name__}.{_name}"] = importlib.import_module(f"forge_llm.agent_backends.{_name}")

__all__ = [*_FORGE_LLM_ALL, *_SUBMODULES]
