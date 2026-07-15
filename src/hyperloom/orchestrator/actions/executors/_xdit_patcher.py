# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Verifier for xDiT's baked diffusion-profiling adaptations.

The two adaptations live as source in the ``hyperloom-xdit-adaptation`` overlay
repo and are baked into the sandbox image at build time; this module only
VERIFIES that the running xfuser carries them (it no longer mutates any file).

The two adaptations (their sentinels are asserted here):

* ``repeat=1`` in ``torch.profiler.schedule`` — retains the ACTIVE profiler window
  (upstream default ``repeat=0`` discards it, exporting an empty trace). Sentinel:
  ``# hyperloom: retain active window``.
* per-denoise-step ``record_function("denoise_step_<i>")`` markers around the
  diffusers ``scheduler.step`` — deterministic per-step roofline split anchors.
  Sentinel: ``# hyperloom: per-denoise-step annotation``.

Verification is fail-soft: a missing/stale bake logs a remediation warning and
returns ``False`` (callers proceed; the roofline just degrades).
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

from ._patch_sentinel import file_contains_sentinel

log = logging.getLogger(__name__)

# xfuser package-relative path to the profiled ``base_model.py``.
_XFUSER_REL = ("model_executor", "models", "runner_models", "base_model.py")

# Sentinels the overlay bakes into base_model.py.
_PROFILER_SENTINEL = "# hyperloom: retain active window"      # repeat=1
_ANNOT_SENTINEL = "# hyperloom: per-denoise-step annotation"  # per-step

# First image tag that bakes the adaptations (for the remediation message).
_REQUIRED_IMAGE = "pytorch-xdit:v26.6-hyperloom15"


def _discover_xfuser_base_models() -> list[Path]:
    """Return every existing xfuser ``base_model.py`` to verify.

    Discovery order (deduped by resolved path):

    * ``$XDIT_PATH/xfuser/<rel>`` when ``XDIT_PATH`` points at an xDiT checkout.
    * The importable ``xfuser`` package location (``importlib`` find-spec, which
      does not import the module), i.e. the copy the ``xdit`` subprocess runs.

    Returns:
        A deduped list of existing ``base_model.py`` files, or ``[]`` when none
        resolve (callers fail-soft; fine for tests / non-xDiT runs).
    """
    out: list[Path] = []
    seen: set[Path] = set()

    def _add(candidate: Path | str | None) -> None:
        if not candidate:
            return  # pragma: no cover - defensive; current callers pass real paths
        try:
            resolved = Path(candidate).expanduser().resolve()
        except OSError:  # pragma: no cover - resolve() rarely raises for these paths
            return
        if not resolved.is_file() or resolved in seen:
            return
        seen.add(resolved)
        out.append(resolved)

    xdit_root = (os.environ.get("XDIT_PATH") or "").strip()
    if xdit_root:
        _add(Path(xdit_root).joinpath("xfuser", *_XFUSER_REL))

    try:
        spec = importlib.util.find_spec("xfuser")
    except (ImportError, ValueError, ModuleNotFoundError):
        spec = None
    if spec is not None:
        for loc in list(spec.submodule_search_locations or []):
            _add(Path(loc).joinpath(*_XFUSER_REL))

    return out


def _is_baked(src: Path) -> bool:
    """Return whether ``base_model.py`` carries BOTH baked adaptation sentinels."""
    return file_contains_sentinel(
        src, _PROFILER_SENTINEL, log, "_xdit_patcher"
    ) and file_contains_sentinel(src, _ANNOT_SENTINEL, log, "_xdit_patcher")


def verify_xdit_profiler_baked() -> bool:
    """Verify the running xfuser carries the baked diffusion-profiling adaptations.

    Returns ``True`` when at least one discovered ``base_model.py`` carries both
    sentinels. When none do (or none are discovered), logs a fail-soft remediation
    warning and returns ``False`` — the run proceeds but the diffusion roofline
    trace will be empty / lack per-step split boundaries.
    """
    files = _discover_xfuser_base_models()
    if not files:
        log.warning(
            "_xdit_patcher: no xfuser base_model.py discovered (checked $XDIT_PATH "
            "and the importable xfuser package) — cannot verify the baked diffusion "
            "profiler adaptations (this is fine for tests and non-xDiT runs)"
        )
        return False

    for src in files:
        if _is_baked(src):
            log.debug(
                "_xdit_patcher: verified baked diffusion-profiling adaptations in %s",
                src,
            )
            return True

    log.warning(
        "_xdit_patcher: xfuser base_model.py is MISSING the baked Hyperloom "
        "diffusion-profiling adaptations (repeat=1 + per-denoise-step markers) in "
        "%s. The diffusion roofline trace will be empty or lack per-step split "
        "boundaries. This image predates the overlay bake — rebuild/run with the "
        "%s image (or newer) from hyperloom-xdit-adaptation.",
        ", ".join(str(f) for f in files),
        _REQUIRED_IMAGE,
    )
    return False


# Backward-compatible alias for callers importing ``ensure_xdit_profiler_patched``.
ensure_xdit_profiler_patched = verify_xdit_profiler_baked

__all__ = ["ensure_xdit_profiler_patched", "verify_xdit_profiler_baked"]
