# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Generate the FlyDSL skeleton the port agent fills in.

The skeleton is deterministic scaffolding, NOT a real implementation: it fixes
only the factory SYMBOL the measurement driver imports (``build_<op>_module``) so
the file always imports, and raises from the returned launch callable so the
correctness gate fails until a genuine FlyDSL kernel is written. It deliberately
makes NO assumption about the factory / launch argument signature — that is
defined by the task's driver (shown to the agent in the port prompt) and varies
by operator (softmax: ``launch(x, out, M)``; rmsnorm: ``launch(x, w, out, M)``;
gemm: ``launch(a, b, c)`` …).
"""

from __future__ import annotations

from pathlib import Path

from kernelforge.rewrite_by_flydsl.spec import RewriteSpec


def generate_seed(spec: RewriteSpec, dest: str | Path) -> str:
    """Write the FlyDSL skeleton ``kernel.py`` for ``spec`` and return its path."""
    dest = Path(dest)
    # The candidate may live in a subdir (e.g. flydsl/kernel.py) declared via the
    # task's target_file_path; ensure the parent exists before writing.
    dest.parent.mkdir(parents=True, exist_ok=True)
    op = spec.op_name
    src = spec.source_kernel_name
    builder = spec.builder_symbol
    body = f'''"""FlyDSL port of `{op}` (rewritten from {src}).

TODO: implement this in FlyDSL. This is a skeleton that fixes only the factory
symbol the measurement driver imports; it is NOT a working implementation yet.

Contract:
    {builder}(...) -> launch_fn
The exact `{builder}` and `launch_fn` argument signatures are the ones the task's
measurement driver calls (see the driver shown in your task instructions) — match
them exactly. Implement with FlyDSL only (import flydsl...) and match the source
kernel's numerics (the loop gates correctness on an SNR threshold against the
original kernel).
"""


def {builder}(*args, **kwargs):
    # TODO: build and return a FlyDSL launch callable. Replace the stub below with
    # a real @flyc.kernel implementation + @flyc.jit launcher whose signatures
    # match how the measurement driver calls them. Until then correctness fails
    # on purpose.
    def launch_fn(*a, **k):
        raise NotImplementedError(
            "FlyDSL {op} kernel not implemented yet — port {src} to FlyDSL here."
        )

    return launch_fn
'''
    dest.write_text(body)
    return str(dest)
