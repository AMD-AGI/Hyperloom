"""Engine-agnostic GPU profile capture and attribution.

The unit of work here is a **profile record**: per-kernel GPU time for one
isolated phase of one engine step, plus enough bookkeeping to tell an honest
component share from a silently-partial one.

Three properties the pack's evaluators depend on, and which the first-generation
``sikl_profile_decode.sh`` / ``profile_decode_vllm.py`` scripts did not provide:

1. **The denominator is the whole step.** A record carries every op, not a
   top-N slice, so ``share = matched / total`` means what it says.
2. **Coverage is reported.** Attribution keeps an explicit ``<rest>`` bucket.
   A component that matched nothing is distinguishable from a component that
   is genuinely cheap — the distinction the Amdahl projection turns on.
3. **Phase isolation is verified, not assumed.** A decode profile says how many
   of its captured steps actually contained no prefill work, and says
   ``unverified`` when it could not tell.

Nothing in :mod:`arbor.profiling.trace`, :mod:`~arbor.profiling.attribution` or
:mod:`~arbor.profiling.record` imports torch or vLLM, so the analysis path is
testable on a laptop. Only :mod:`~arbor.profiling.capability` and
:mod:`~arbor.profiling.vllm_driver` touch an engine.
"""

from __future__ import annotations

from arbor.profiling.attribution import (
    Component,
    DEFAULT_COMPONENTS,
    attribute,
)
from arbor.profiling.record import (
    ProfileRecord,
    component_share,
    load_record,
)
from arbor.profiling.trace import (
    DECODE_KERNELS,
    PREFILL_KERNELS,
    KernelTotals,
    aggregate_kernels,
    classify_phase,
    load_events,
    phase_report,
    segment_steps,
)

__all__ = [
    "Component",
    "DEFAULT_COMPONENTS",
    "DECODE_KERNELS",
    "PREFILL_KERNELS",
    "KernelTotals",
    "ProfileRecord",
    "aggregate_kernels",
    "attribute",
    "classify_phase",
    "component_share",
    "load_events",
    "load_record",
    "phase_report",
    "segment_steps",
]
