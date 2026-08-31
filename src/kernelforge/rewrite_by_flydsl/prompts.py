# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Program text for the PORT phase — instructs the flydsl to translate the
source kernel into FlyDSL, correctness first. Injected as the agent system prompt
(stable across attempts, so the SDK prompt cache reuses it).

The interface the port must satisfy is NOT hard-coded here (it varies by
operator). Instead the task's measurement driver is embedded read-only: it is the
single source of truth for how ``build_<op>_module`` and its launch callable are
invoked, so the agent matches the real call signatures rather than a fixed rowwise
shape.
"""

from __future__ import annotations

from pathlib import Path

from kernelforge.rewrite_by_flydsl.spec import RewriteSpec

_MAX_SOURCE_CHARS = 16000  # keep the embedded source bounded for the prompt
_MAX_DRIVER_CHARS = 8000  # the driver is small; cap defensively

# Per source language: the markdown fence to embed it under, and the name to call
# it by. A HIP kernel fenced as ``python`` misleads the agent in the block it
# reads most closely.
_SOURCE_PRESENTATION: dict[str, tuple[str, str]] = {
    "triton": ("python", "Triton"),
    "hip": ("cpp", "HIP"),
    "cuda": ("cpp", "CUDA"),
    "cpp": ("cpp", "C++"),
}


def _read_bounded(path: str, cap: int, missing: str) -> str:
    try:
        text = Path(path).read_text()
    except OSError:
        return missing
    if len(text) > cap:
        text = text[:cap] + "\n# ... (truncated) ...\n"
    return text


def build_port_program_md(spec: RewriteSpec, driver_path: str) -> str:
    """Assemble the port program.md (objective + driver contract + source + rules)."""
    source = _read_bounded(spec.source_kernel, _MAX_SOURCE_CHARS, "(source unavailable)")
    driver = _read_bounded(driver_path, _MAX_DRIVER_CHARS, "(driver unavailable)")

    builder = spec.builder_symbol
    candidate = spec.flydsl_kernel_relpath
    targets = ", ".join(spec.target_functions) or "the target kernel"
    entry_hint = (
        f"The source host entry `{spec.source_entry}` runs the kernel end-to-end.\n" if spec.source_entry else ""
    )
    fence, language = _SOURCE_PRESENTATION.get(spec.source_language, ("", ""))
    source_heading = (
        f"## Source kernel to port ({language}, READ-ONLY reference)"
        if language
        else "## Source kernel to port (READ-ONLY reference)"
    )
    banned = f"{language}, torch or any other GPU library" if language else "torch or any other GPU library"
    return f"""\
# Program: rewrite `{spec.op_name}` to FlyDSL (correctness first)

## Objective
Port the source kernel(s) `{targets}` (in `{spec.source_kernel_name}`) into an
equivalent **FlyDSL** kernel written in `{candidate}`. This phase is
about CORRECTNESS: the FlyDSL output must match the original kernel within the SNR
gate (>= {spec.snr_threshold:g} dB). A later phase optimizes it; here, just make it correct.

## Interface contract (MUST match exactly)
`{candidate}` MUST define a factory `{builder}(...)` that returns a
launch callable. The EXACT argument signatures (order, count, meaning) are defined
by the measurement driver below: it imports `{builder}`, calls it to build the
kernel, then calls the returned launch callable each run. Match those calls
exactly. {entry_hint}
## Measurement driver (READ-ONLY — defines how your kernel is called + checked)
```python
{driver}
```

{source_heading}
```{fence}
{source}
```

## Rules
- Implement in FlyDSL ONLY (import flydsl...). Do NOT call {banned} to
  compute the result — that defeats the rewrite.
- Edit ONLY `{candidate}`. `{spec.source_kernel_name}`, the driver,
  and any harness define the reference/measurement; editing them is blocked.
- Expose the `{builder}` factory and match the launch signature the driver calls.
- Consult the FlyDSL knowledge (operator cards, API docs, examples) before
  writing — work from the docs, not from memory.
"""
