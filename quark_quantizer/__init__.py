# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""quark_quantizer — a thin, parameter-gated shell over ``quantization_agent``.

The original ``quantization_agent`` (the Hyperloom driver that turns a natural
-language prompt into Quark CLI calls and executes them) is kept as-is. This
shell adds one thing for stability: the decision to quantize is a **parameter**
(``enabled``), supplied by the caller (a backend, driven by a frontend toggle)
— never inferred from free-form natural language by an outer orchestration LLM.

Two inputs go in conceptually: the ``enabled`` switch and the NL ``prompt``.
When enabled, the shell forwards the prompt to
``quantization_agent.quantize_via_prompt`` (which does prompt -> CLI -> execute);
when disabled it is a no-op.

Public entry: :func:`quantize` (and :func:`quantize_sync`).
"""

from __future__ import annotations

from .runner import QuarkRunResult, quantize, quantize_sync

__all__ = [
    "QuarkRunResult",
    "quantize",
    "quantize_sync",
]
