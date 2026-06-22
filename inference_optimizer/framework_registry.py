# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Framework capability registry — single source of truth for what a given
inference framework *is*, so the optimizer loop can gate actions on capabilities

A capability descriptor answers:
  - ``serving``        "online" (server + client load) vs "offline" (CLI batch)
  - ``has_server``     does the framework expose an inference server we launch?
  - ``launch``         how Hyperloom invokes a run ("server" vs verbatim cmd)
  - ``modality``       "text" (LLM decode loop) vs "diffusion" (image/video gen)
  - ``kpi``            the accept/reject metric for a change
  - ``search_knobs``   parameters worth sweeping (empty => no parameter search)
  - ``accuracy_gate``  how correctness is checked after a change

These axes are orthogonal, and call sites must gate on the one that names their
decision rather than using any single field as a stand-in for a whole class of
framework. 
The mapping is:
  - ``modality``      "is there a token decode loop?" (trace splitting, the
                      trace-only profiling pass)
  - ``accuracy_gate`` "lm-eval score vs media diff?"
  - ``search_knobs``  "is there anything to sweep?" (and "conc" specifically
                      for the concurrency sweep)
  - ``launch``        "server boot vs verbatim run command?"

``modality`` is coarse ("no decode loop; generates media"); the artifact-
specific correctness check lives on the finer ``accuracy_gate`` axis, so a new
output type (e.g. video) is a new gate value, not a new modality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final


@dataclass(frozen=True)
class FrameworkCapabilities:
    """Declarative capabilities for one inference framework."""

    name: str
    serving: str               # "online" | "offline"
    has_server: bool
    launch: str                # "server" | "run_cmd"
    modality: str              # "text" | "diffusion"
    kpi: str                   # "output_throughput" | "latency_per_image" | ...
    accuracy_gate: str         # "lm_eval" | "image_diff" | "none"
    search_knobs: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_offline(self) -> bool:
        return self.serving == "offline"

    @property
    def is_online(self) -> bool:
        return self.serving == "online"

    @property
    def is_diffusion(self) -> bool:
        """True for media-generation engines (no token decode loop).

        Gates decisions that assume a steady-state decode loop: the trace
        splitter has no execution-step markers to find, and the profiling pass
        yields a trace but no throughput number.
        """
        return self.modality == "diffusion"

    @property
    def gates_on_image_diff(self) -> bool:
        """True when post-change correctness is checked by image diff."""
        return self.accuracy_gate == "image_diff"

    @property
    def supports_param_search(self) -> bool:
        """True when there is anything worth sweeping (non-empty knobs)."""
        return bool(self.search_knobs)

    @property
    def supports_conc_sweep(self) -> bool:
        """True when a post-sweep concurrency sweep is meaningful."""
        return "conc" in self.search_knobs


# Slug returned for an unset/unknown framework.
_DEFAULT_FRAMEWORK: Final[str] = "sglang"


# Canonical registry. Keys are lowercased framework slugs.
_REGISTRY: Final[dict[str, FrameworkCapabilities]] = {
    "sglang": FrameworkCapabilities(
        name="sglang",
        serving="online",
        has_server=True,
        launch="server",
        modality="text",
        kpi="output_throughput",
        accuracy_gate="lm_eval",
        search_knobs=("server_params", "conc"),
    ),
    "vllm": FrameworkCapabilities(
        name="vllm",
        serving="online",
        has_server=True,
        launch="server",
        modality="text",
        kpi="output_throughput",
        accuracy_gate="lm_eval",
        search_knobs=("server_params", "conc"),
    ),
    "atom": FrameworkCapabilities(
        name="atom",
        serving="online",
        has_server=True,
        launch="server",
        modality="text",
        kpi="output_throughput",
        accuracy_gate="lm_eval",
        search_knobs=("server_params", "conc"),
    ),
    "xdit": FrameworkCapabilities(
        name="xdit",
        serving="offline",
        has_server=False,
        launch="run_cmd",
        modality="diffusion",
        kpi="latency_per_image",
        accuracy_gate="image_diff",
        search_knobs=(),  # explicitly empty — no parameter sweeping
    ),
}


def supported_frameworks() -> tuple[str, ...]:
    """Sorted tuple of registered framework slugs."""
    return tuple(sorted(_REGISTRY))


def is_supported(framework: str) -> bool:
    return _slug(framework) in _REGISTRY


def get_capabilities(framework: str) -> FrameworkCapabilities:
    """Return capabilities for ``framework``.  """
    return _REGISTRY.get(_slug(framework), _REGISTRY[_DEFAULT_FRAMEWORK])


def _slug(framework: str) -> str:
    return (framework or "").strip().lower()
