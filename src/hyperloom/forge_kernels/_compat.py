# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""FlyDSL API-drift shim for KernelForge-generated kernels.

KernelForge authors ``serving_patches/kernels/*/kernel.py`` against whichever FlyDSL
the generating fellow had; the serving image pins its own. The names below were
public in the authoring FlyDSL and moved (or were dropped) in later releases, so
a kernel that is perfectly valid at KernelForge fails at import/trace time here
with a bare ``AttributeError``. This restores them on the installed module.

Same idea as the ``fly_values -> extract_to_ir_values`` shim the kernel-agent
already applies before an aiter forge loop
(``agents/kernel/tools/backends/forge_submit.py``), kept separate because this
one runs inside the *serving* process rather than the optimization loop.

Every alias is applied only when the installed FlyDSL is missing the name, so a
FlyDSL that still exports it is left untouched. :func:`install` is idempotent
and returns the names it had to add, which the preflight report records as
provenance.
"""

from __future__ import annotations

import threading

_LOCK = threading.Lock()
_APPLIED: list[str] | None = None


def install() -> list[str]:
    """Patch the installed ``flydsl`` in place; return the names that were added.

    Returns:
        The aliases this process had to add, in application order. Empty when
        the installed FlyDSL already exports everything the generated kernels
        expect. Cached: repeat calls return the first result without re-patching.

    Raises:
        ImportError: when ``flydsl`` is not importable at all.
    """
    global _APPLIED
    with _LOCK:
        if _APPLIED is not None:
            return list(_APPLIED)
        _APPLIED = _install_locked()
        return list(_APPLIED)


def _install_locked() -> list[str]:
    import flydsl.expr as fx
    import flydsl.expr.math as fmath
    import flydsl.expr.rocdl as frocdl
    from flydsl._mlir import ir
    from flydsl._mlir.dialects import arith as _arith
    from flydsl.expr.numeric import _unwrap_value

    added: list[str] = []

    # Dropped public alias. ``_unwrap_value`` is the same "FlyDSL wrapper ->
    # raw MLIR value" coercion the old ``as_ir_value`` performed.
    if not hasattr(fx, "as_ir_value"):
        fx.as_ir_value = _unwrap_value
        added.append("as_ir_value")

    # Relocated from the ``flydsl.expr`` root into ``flydsl.expr.math``.
    for name in ("exp2", "fma"):
        if not hasattr(fx, name):
            setattr(fx, name, getattr(fmath, name))
            added.append(name)

    # ``readlane`` lost the implicit int -> arith.constant coercion on its lane
    # operand, so a generated ``readlane(ty, v, 63)`` now raises "Operand 1 ...
    # must be a Value".
    if not getattr(frocdl, "_hyperloom_readlane_coerced", False):
        _raw_readlane = frocdl.readlane

        def readlane(res, src0, src1, **kwargs):
            src0 = _unwrap_value(src0)
            if not isinstance(src1, ir.Value):
                src1 = _arith.constant(ir.IntegerType.get_signless(32), int(src1))
            return _raw_readlane(res, src0, src1, **kwargs)

        frocdl.readlane = readlane
        frocdl._hyperloom_readlane_coerced = True
        added.append("rocdl.readlane")

    return added
