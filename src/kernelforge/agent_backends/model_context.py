# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Name the context window a Claude session runs on, where one is offered.

Anthropic selects the window with a suffix on the model id
(``claude-opus-5[1m]``) rather than with a request field, so "which context
window" is not a knob a caller can set -- it is part of the name. This module
owns that spelling, and :meth:`AgentRunSpec.resolved` applies it to every
session, so no call site decides it and no operator has to remember to type it.

The window is a deployment fact, not a tuning parameter, which is why it is
resolved once into :class:`AgentRuntimeConfig` and never named at a call site.
It is nonetheless *configured* rather than assumed, because whether a window
exists at all is a property of the gateway in front of the model, not of the
model. Upstream KernelForge appends ``[1m]`` unconditionally on the strength of
a gateway that serves it; the gateway Hyperloom deploys against does not, and
rejects every bracketed id in its catalog:

.. code-block:: text

    claude-opus-5         -> 200
    claude-opus-5[1m]     -> 400 Invalid model name
    claude-opus-5[200k]   -> 400

Since the suffix is validated rather than ignored, an unconditional one is not
a degradation but a hard failure of every session, which is why the window is
named by :envvar:`CLAUDE_CONTEXT_WINDOW` and is absent by default. An operator on a gateway that publishes a windowed id
sets it once, for the whole campaign, and gets upstream's behaviour.

The Codex line has no such spelling, so a Codex id is returned untouched:
inventing a suffix there would fail on the first call with an unrecognized
model.
"""

from __future__ import annotations

import re

#: The window to ask for on a gateway that publishes one, spelled as Anthropic
#: spells it. This is the value to give the environment variable, not a default.
EXTENDED_CONTEXT = "1m"

# ``[`` is a terminator: an id that already names a window is one of ours.
_CLAUDE_MODEL_RE = re.compile(r"(^|[/.:_-])claude(?:[/.:_\-\[]|$)", re.IGNORECASE)
_CONTEXT_SUFFIX_RE = re.compile(r"\[[^\]]*\]\s*$")


def with_context_window(model: str, window: str = "") -> str:
    """Return ``model`` naming ``window``, where the provider spells one.

    An empty ``window`` -- the default, and what an unconfigured deployment
    resolves to -- returns the id unchanged, because a bracketed suffix a
    gateway does not publish is a 400 on every call rather than a smaller
    window. An id that already carries a bracketed suffix is also returned
    unchanged, so an operator who spelled the window into the model themselves
    is not double-suffixed.

    Returns a stripped id on every path. The suffix branch has to strip in order
    to append, so returning the raw string on the others made one function
    sometimes normalize and sometimes not -- and the result goes onto a CLI
    argument, where a trailing space is a different model id.
    """
    stripped = model.strip()
    requested = window.strip()
    if not stripped or not requested:
        return stripped
    if not _CLAUDE_MODEL_RE.search(stripped):
        return stripped
    if _CONTEXT_SUFFIX_RE.search(stripped):
        return stripped
    return f"{stripped}[{requested}]"


__all__ = ["EXTENDED_CONTEXT", "with_context_window"]
