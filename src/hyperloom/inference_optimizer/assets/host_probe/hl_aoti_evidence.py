# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Record whether a run really exported and ran an AOTInductor package.

The export gate needs to know that a KEEP's module exports. Reading the source
cannot answer that: the markers appear in a helper nobody calls, in a branch under
``if False``, and in the ``except`` arm of a fallback to eager -- all of which look
identical to a working export. This counts the calls instead.

Three counters, in increasing order of what they prove:

``compiled``  ``aoti_compile_and_package`` returned -- an export compiled.
``loaded``    ``aoti_load_package`` returned -- a package was opened.
``invoked``   the loaded model was called -- the exported path actually ran.

``invoked`` is the one the gate wants. A module that compiles a package and then
returns the eager model scores ``compiled>0, invoked==0``.

Armed by ``HYPERLOOM_AOTI_EVIDENCE_DIR``, into which each process writes
``<pid>.json``: a benchmark may fork, and one shared path would leave last-writer-wins.

``instrumented`` says the shim ran and wrapped torch. Without it the reader cannot
tell "nothing exported" from "the shim never loaded" -- not a hypothetical, since
``python -I`` silently suppresses ``sitecustomize`` and has cost us a campaign
before. No file at all means the shim never ran, and the reader refuses rather than
reporting zero.

Unarmed, this costs an environment lookup. Nothing here may raise: a run that cannot
be measured is worse than one that is not instrumented.
"""

from __future__ import annotations

import importlib.abc
import importlib.util
import json
import os
import sys
import threading

SCHEMA = "hyperloom.aoti_evidence/1"

_LOCK = threading.Lock()
_COUNTS = {"compiled": 0, "loaded": 0, "invoked": 0}
_INSTALLED = False
_INSTRUMENTED = False
_OUT_DIR = ""


def _bump(key: str) -> None:
    """Count one call, publishing only the first of each kind.

    Publishing every call cost 4.4x on the inference hot path (34.8 -> 154.5 us per
    call, measured): a JSON write per invocation. The gate reads these as booleans,
    so the first of each is the whole signal and the rest is bookkeeping flushed at
    exit.
    """
    with _LOCK:
        _COUNTS[key] += 1
        first = _COUNTS[key] == 1
        counts = dict(_COUNTS)
    if first:
        _write(counts)


def _write(counts: dict) -> None:
    """Publish this process's counters. One file per pid; the reader aggregates."""
    if not _OUT_DIR:
        return
    pid = os.getpid()
    payload = {"schema": SCHEMA, "pid": pid, "instrumented": _INSTRUMENTED, **counts}
    try:
        os.makedirs(_OUT_DIR, exist_ok=True)
        final = os.path.join(_OUT_DIR, f"{pid}.json")
        tmp = f"{final}.tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle)
        os.replace(tmp, final)
    except OSError:
        pass


def _wrap(owner, name: str, key: str, *, once: bool = False) -> bool:
    """Count calls to ``owner.name``. Returns whether it was wrapped.

    ``once`` restores the original after the first call. Use it on anything the
    benchmark calls in its timed loop: the gate only needs to know the path ran, and
    an un-restored wrapper is a Python frame on every inference.
    """
    target = getattr(owner, name, None)
    if target is None or getattr(target, "_hl_aoti_wrapped", False):
        return False

    def wrapper(*args, **kwargs):
        result = target(*args, **kwargs)
        # Counted after the call: a compile that raised is not a compile that happened,
        # which is exactly the fallback-to-eager case the gate must not be fooled by.
        _bump(key)
        if once:
            try:
                setattr(owner, name, target)
            except (AttributeError, TypeError):
                pass
        return result

    wrapper._hl_aoti_wrapped = True  # type: ignore[attr-defined]
    try:
        setattr(owner, name, wrapper)
    except (AttributeError, TypeError):
        return False
    return True


def _instrument(torch_module) -> None:
    """Wrap the AOTI entry points on an imported ``torch``."""
    global _INSTRUMENTED
    wrapped = []
    try:
        inductor = getattr(torch_module, "_inductor", None)
        if inductor is None:
            import torch._inductor as inductor  # type: ignore[no-redef]
        for name, key in (("aoti_compile_and_package", "compiled"), ("aoti_load_package", "loaded")):
            if _wrap(inductor, name, key):
                wrapped.append(name)
        # The call counter has to sit on the class: the loaded object is constructed
        # inside torch, so there is no instance to wrap at load time.
        from torch.export.pt2_archive._package import AOTICompiledModel

        if _wrap(AOTICompiledModel, "__call__", "invoked", once=True):
            wrapped.append("AOTICompiledModel.__call__")
    except Exception as exc:  # noqa: BLE001 - instrumentation must never break a run
        sys.stderr.write(f"[hl_aoti_evidence] instrumentation failed: {exc!r}\n")
    # Only the call counter settles whether an export ran, so it alone decides whether
    # this process can report. With the others wrapped but not it, zero counts would be
    # indistinguishable from an uninstrumented run.
    _INSTRUMENTED = "AOTICompiledModel.__call__" in wrapped
    # Publish immediately, so "torch imported, nothing exported" is distinguishable
    # from "the shim never ran" -- the failure mode that cost us a campaign before.
    _write(dict(_COUNTS))


class _TorchImportHook(importlib.abc.MetaPathFinder):
    """Instrument ``torch`` right after the process first imports it.

    Stays registered until the module actually executes: a bare ``find_spec`` lookup
    must not consume the hook.
    """

    def __init__(self) -> None:
        self._resolving = False

    def find_spec(self, fullname, path, target=None):
        if fullname != "torch" or self._resolving:
            return None
        self._resolving = True
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            self._resolving = False
        if spec is None or spec.loader is None:
            return spec
        exec_module = spec.loader.exec_module
        hook = self

        def exec_then_instrument(module) -> None:
            exec_module(module)
            if hook in sys.meta_path:
                sys.meta_path.remove(hook)
                _instrument(module)

        spec.loader.exec_module = exec_then_instrument
        return spec


def install_from_env() -> bool:
    """Arm the counters when ``HYPERLOOM_AOTI_EVIDENCE_DIR`` names an output directory."""
    global _INSTALLED, _OUT_DIR
    if _INSTALLED:
        return True
    out_dir = str(os.environ.get("HYPERLOOM_AOTI_EVIDENCE_DIR", "") or "").strip()
    if not out_dir:
        return False
    _OUT_DIR = out_dir
    _INSTALLED = True
    if "torch" in sys.modules:
        _instrument(sys.modules["torch"])
    else:
        sys.meta_path.insert(0, _TorchImportHook())
    return True


def counts() -> dict:
    """The counters as they stand, for in-process assertions and tests."""
    with _LOCK:
        return dict(_COUNTS)


__all__ = ["SCHEMA", "counts", "install_from_env"]
