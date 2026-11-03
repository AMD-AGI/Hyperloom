"""Serving frameworks behind one interface (roadmap CORE-06).

``get(config)`` is the adapter for the framework a campaign config names in
``"framework"``; the harness launches, finds, stops and probes its server
through it (base.py lists what an adapter holds). ``union`` is what a site
that does not know the framework needs from all of them at once: armcheck's
patterns, stop_server's sweep.

The defaults are the ones the harness always had, the inconsistency included:

- A config that names no framework (missing, empty or None) is SGLang, to the
  launch, the server agent's help hint and armcheck alike.
- An unknown name is SGLang to ``get``, as it was to the launch and the help
  hint, which tested for "vllm" and took SGLang otherwise. armcheck reads the
  same name as vLLM (its lookup falls back to vLLM's pattern), so an unknown
  framework is launched as SGLang and looked for as vLLM. Both moved as they
  were: settling on one default, or refusing an unknown name, changes
  behaviour, and this move does not.
"""

from __future__ import annotations

from arbor.frameworks.base import FrameworkAdapter
from arbor.frameworks.sglang import SGLang
from arbor.frameworks.vllm import VLLM

DEFAULT = "sglang"

# SGLang first, vLLM second: the order union() keeps, so stop_server signals
# SGLang's server before vLLM's, as it always did.
_ADAPTERS: dict[str, FrameworkAdapter] = {fw.name: fw for fw in (SGLang(), VLLM())}


def name_of(name_or_config) -> str:
    """The framework a config (or a bare name) names, lower-case; DEFAULT when none."""
    if isinstance(name_or_config, dict):
        name_or_config = name_or_config.get("framework")
    return (name_or_config or DEFAULT).lower()


def get(name_or_config) -> FrameworkAdapter:
    """The adapter for a config or a name. An unknown name is SGLang (see above)."""
    return _ADAPTERS.get(name_of(name_or_config), _ADAPTERS[DEFAULT])


def names() -> tuple[str, ...]:
    """The registered frameworks, in registry order."""
    return tuple(_ADAPTERS)


def union(attr: str) -> tuple:
    """Every adapter's ``attr`` (a tuple), in registry order, each item once."""
    out: list = []
    for fw in _ADAPTERS.values():
        for item in getattr(fw, attr):
            if item not in out:
                out.append(item)
    return tuple(out)
