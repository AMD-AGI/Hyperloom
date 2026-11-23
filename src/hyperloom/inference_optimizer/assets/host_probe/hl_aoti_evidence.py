# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Count a run's AOTInductor calls, so the export gate need not read source.

Three counters, in increasing order of what they prove: ``compiled`` (an export
compiled), ``loaded`` (a package was opened), ``invoked`` (the exported path ran).
Only ``invoked`` settles it -- a module that compiles a package and then returns the
eager model scores ``compiled>0, invoked==0``.

``invoked>0`` on its own only proves the exported path ran once. A module that calls
the package to validate it and then serves every inference from the eager model scores
``invoked=1`` and is the fallback case this exists to catch, so the timed region is
counted separately: the harness opens a window and reports its own calls at the module
boundary into ``timed_boundary``, against which ``timed_invoked`` must hold up.

Armed by ``HYPERLOOM_AOTI_EVIDENCE_DIR``, into which each process writes
``<pid>.json``: a benchmark may fork, and one shared path is last-writer-wins.

``instrumented`` says the shim wrapped torch, so the reader can tell "nothing
exported" from "the shim never loaded" -- ``python -I`` suppresses ``sitecustomize``
and has cost us a campaign before.

Nothing here may raise: a run that cannot be measured beats one that is broken.
"""

from __future__ import annotations

import atexit
import importlib.abc
import importlib.util
import json
import os
import sys
import threading

SCHEMA = "hyperloom.aoti_evidence/3"

_LOCK = threading.Lock()
_COUNTS = {
    "compiled": 0,
    "loaded": 0,
    "invoked": 0,
    "boundary": 0,
    "timed_invoked": 0,
    "timed_boundary": 0,
    "timed_eager_boundary": 0,
}
_INSTRUMENTED = False
_OUT_DIR = ""
_TIMED = False
#: Package calls since the last boundary mark, and whether a marked interval is open.
_SINCE_BOUNDARY = 0
_BOUNDARY_OPEN = False


def _bump(key: str, *, timed_key: str = "", boundary_work: bool = False) -> None:
    """Count one call, publishing only the first of each kind.

    Publishing every call cost 4.4x on the inference hot path (34.8 -> 154.5 us,
    measured), so the write stays off it: the first of each kind lands immediately,
    the rest when the timed window closes and again at exit. Counting alone is 0.15 us
    per call against that same 34.8, which is inside the run-to-run noise.
    """
    global _SINCE_BOUNDARY
    with _LOCK:
        _COUNTS[key] += 1
        if boundary_work:
            _SINCE_BOUNDARY += 1
        if timed_key and _TIMED:
            _COUNTS[timed_key] += 1
        # Copied only on the first of each kind: the snapshot is for ``_write`` alone, and
        # taking one per call puts a dict copy on the inference path for nothing.
        counts = dict(_COUNTS) if _COUNTS[key] == 1 else None
    if counts is not None:
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
        # Nothing here may raise: a run that cannot publish its counters beats one that is broken.
        pass


def _wrap(owner, name: str, key: str, *, timed_key: str = "", boundary_work: bool = False) -> bool:
    """Count calls to ``owner.name``; returns whether it was wrapped.

    The wrapper stays installed for the life of the process. It used to restore the
    original after the first call, which left ``invoked`` saturated at 1 and let a
    module that calls the package once and then runs eager pass the gate.
    """
    target = getattr(owner, name, None)
    if target is None or getattr(target, "_hl_aoti_wrapped", False):
        return False

    def wrapper(*args, **kwargs):
        result = target(*args, **kwargs)
        # After the call: a compile that raised is not a compile that happened, which is
        # the fallback-to-eager case exactly.
        _bump(key, timed_key=timed_key, boundary_work=boundary_work)
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

        if _wrap(AOTICompiledModel, "__call__", "invoked", timed_key="timed_invoked", boundary_work=True):
            wrapped.append("AOTICompiledModel.__call__")
    except Exception as exc:  # noqa: BLE001 - instrumentation must never break a run
        print(f"[hl_aoti_evidence] instrumentation failed: {exc!r}", file=sys.stderr)
    # Only the call counter settles whether an export ran, so it alone decides whether
    # this process can report. With the others wrapped but not it, zero counts would be
    # indistinguishable from an uninstrumented run.
    _INSTRUMENTED = "AOTICompiledModel.__call__" in wrapped
    # Publish now, so "nothing exported" is distinguishable from "the shim never ran".
    _write(dict(_COUNTS))


class _TorchImportHook(importlib.abc.MetaPathFinder):
    """Instrument ``torch`` right after the process first imports it.

    Stays registered until the module executes: a bare ``find_spec`` must not consume it.
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


def _charge_open_interval() -> None:
    """Charge the interval since the last mark if nothing exported ran in it.

    Reads the lifecycle rather than owning it: whoever closes an interval clears
    ``_BOUNDARY_OPEN`` itself, so closing twice cannot charge twice. Call under ``_LOCK``.
    """
    if _BOUNDARY_OPEN and _SINCE_BOUNDARY == 0:
        _COUNTS["timed_eager_boundary"] += 1


def timed_window(open_: bool) -> None:
    """Mark the region whose calls the gate judges, so a warm-up cannot stand in for it.

    The harness owns both ends. Closing one publishes, because the counts matter most
    where the process is most likely to be killed -- a timed loop that overran its
    deadline still has to be judged on what it ran. It also charges the last interval,
    which has no following mark to close it.
    """
    global _TIMED, _SINCE_BOUNDARY, _BOUNDARY_OPEN
    if open_:
        with _LOCK:
            _SINCE_BOUNDARY = 0
            _BOUNDARY_OPEN = False
        _TIMED = True
        return
    with _LOCK:
        _charge_open_interval()
        _BOUNDARY_OPEN = False
    _TIMED = False
    _write(counts())


def note_boundary_call() -> None:
    """Count one call the harness made at the module boundary, and close the one before it.

    Totals are not enough on their own. A module calling the package k>1 times per
    inference can serve (k-1)/k of the timed calls eagerly and still report
    ``timed_invoked >= timed_boundary`` -- a flow-matching policy whose denoising step
    runs ten times per forward is exactly that shape. So each interval between marks is
    judged on its own, and one that ran no exported call is counted into
    ``timed_eager_boundary``, which the gate requires to be zero.
    """
    global _SINCE_BOUNDARY, _BOUNDARY_OPEN
    with _LOCK:
        if _TIMED:
            _charge_open_interval()
            _BOUNDARY_OPEN = True
        _SINCE_BOUNDARY = 0
    _bump("boundary", timed_key="timed_boundary")


def install_from_env() -> bool:
    """Arm the counters when ``HYPERLOOM_AOTI_EVIDENCE_DIR`` names an output directory."""
    global _OUT_DIR
    if _OUT_DIR:
        return True
    out_dir = str(os.environ.get("HYPERLOOM_AOTI_EVIDENCE_DIR", "") or "").strip()
    if not out_dir:
        return False
    _OUT_DIR = out_dir
    atexit.register(lambda: _write(counts()))
    if "torch" in sys.modules:
        _instrument(sys.modules["torch"])
    else:
        sys.meta_path.insert(0, _TorchImportHook())
    return True


def counts() -> dict:
    """The counters as they stand, for in-process assertions and tests."""
    with _LOCK:
        return dict(_COUNTS)


__all__ = ["SCHEMA", "counts", "install_from_env", "note_boundary_call", "timed_window"]
