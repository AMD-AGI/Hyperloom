# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Idempotent, backward-compatible patcher for xDiT's torch-profiler schedule.

xDiT (``xfuser``) profiles diffusion runs with

    schedule = torch.profiler.schedule(
        wait=self.config.profile_wait,
        warmup=self.config.profile_warmup,
        active=self.config.profile_active,
    )
    num_repetitions = profile_wait + profile_warmup + profile_active
    for _ in range(num_repetitions):
        run_one_image()
        profile_object.step()

Because ``num_repetitions`` equals the schedule cycle length, the ACTIVE step is
always the last iteration; the loop then calls ``profile_object.step()`` once
more. With the ``torch.profiler.schedule`` default ``repeat=0`` (repeat forever)
and no ``on_trace_ready`` callback, that trailing ``step()`` rolls the profiler
into a fresh collection cycle and DISCARDS the just-recorded active window. The
exported ``profile_trace_rank_*.json.gz`` then contains only metadata + a lone
``hipDeviceSynchronize`` (``Op count == 0``), so TraceLens/roofline get an empty
trace regardless of eager vs torch.compile. Verified on MI355X SD3.5-large:
``repeat=1`` turns a 1.3 KB empty trace into a 6 MB trace with ~30k kernel and
~130k cpu_op events.

This patch inserts ``repeat=1`` into that schedule so one cycle is retained and
exported. Applied in place, once, never reverted: idempotent via a sentinel
substring, serialized across processes via ``fcntl.flock``, written atomically
(temp file + ``os.replace``). Returns ``False`` (non-fatal) when the legacy
block is missing so callers can fail-soft.
"""

from __future__ import annotations

import importlib.util
import logging
import os
from pathlib import Path

from ._inferencex_patcher import _ensure_patched
from ._magpie_patcher import atomic_write_text
from ._patch_sentinel import file_contains_sentinel

log = logging.getLogger(__name__)

# xfuser package-relative path to the profiler schedule.
_XFUSER_REL = ("model_executor", "models", "runner_models", "base_model.py")

# Exact upstream schedule tail (whitespace-anchored to the single
# ``torch.profiler.schedule(...)`` call in base_model.py).
_LEGACY_BLOCK = (
    "            active=self.config.profile_active,\n"
    "        )"
)
_PATCHED_BLOCK = (
    "            active=self.config.profile_active,\n"
    "            repeat=1,  # hyperloom: retain active window "
    "(upstream default repeat=0 discards it -> empty trace)\n"
    "        )"
)
# "Already patched?" sentinel.
_PATCH_SENTINEL = "# hyperloom: retain active window"

# System-wide lock (``/tmp`` is writable; cross-reboot persistence not needed).
_LOCK_PATH = "/tmp/hyperloom_xdit_profiler_patcher.lock"


def _discover_xfuser_base_models() -> list[Path]:
    """Return every existing xfuser ``base_model.py`` Hyperloom should patch.

    Discovery order (deduped by resolved path):

    * ``$XDIT_PATH/xfuser/<rel>`` when ``XDIT_PATH`` points at an xDiT checkout.
    * The importable ``xfuser`` package location (``importlib`` find-spec, which
      does not import the module), i.e. the copy the ``xdit`` subprocess runs.

    Returns:
        A deduped list of existing ``base_model.py`` files, or ``[]`` when none
        resolve (callers fail-soft; this is fine for tests / non-xDiT runs).
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


def _is_patched(src: Path) -> bool:
    """Return whether ``base_model.py`` already carries the ``repeat=1`` patch.

    Args:
        src: The xfuser ``base_model.py`` file to inspect.

    Returns:
        ``True`` if the patch sentinel is present; ``False`` on a miss or read
        error.
    """
    return file_contains_sentinel(src, _PATCH_SENTINEL, log, "_xdit_patcher")


def _apply_patch_atomic(src: Path) -> bool:
    """Insert ``repeat=1`` into the profiler schedule via temp-file + atomic
    rename so a crash mid-write cannot corrupt ``base_model.py``.

    Args:
        src: The xfuser ``base_model.py`` file to patch in place.

    Returns:
        ``True`` when the patched bytes were written; ``False`` when the legacy
        block is missing or any IO step fails.
    """
    try:
        original = src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_xdit_patcher: cannot read %s: %s", src, e)
        return False

    if _LEGACY_BLOCK not in original:
        log.warning(
            "_xdit_patcher: expected torch.profiler.schedule(...) block not "
            "found in %s; xDiT layout may have changed and Hyperloom needs an "
            "updated patch. The profiler active window will be discarded "
            "(empty diffusion trace -> roofline REVERT). Manual review needed.",
            src,
        )
        return False

    patched = original.replace(_LEGACY_BLOCK, _PATCHED_BLOCK, 1)
    if patched == original:
        return False

    if not atomic_write_text(
        src,
        patched,
        tmp_prefix=".base_model.py.hyperloom_",
        log_prefix="_xdit_patcher",
    ):
        return False

    log.info(
        "_xdit_patcher: added repeat=1 to the torch.profiler.schedule in %s "
        "so the diffusion profiler retains its active window (else the trace "
        "is empty: Op count == 0)",
        src,
    )
    return True


def ensure_xdit_profiler_patched() -> bool:
    """Ensure xDiT's ``base_model.py`` profiler schedule keeps ``repeat=1``.

    Returns ``True`` when patched at exit, ``False`` (non-fatal) when the file
    is missing or the legacy block is absent. Concurrency-safe (flock + atomic
    rename; already-patched fast-path skips the lock).

    Returns:
        ``True`` when at least one discovered ``base_model.py`` is patched (or
        already patched), ``False`` when none could be patched.
    """
    return _ensure_patched(
        _discover_xfuser_base_models(),
        _is_patched,
        _apply_patch_atomic,
        _LOCK_PATH,
        empty_msg=(
            "_xdit_patcher: no xfuser base_model.py discovered (checked "
            "$XDIT_PATH and the importable xfuser package) — skipping "
            "profiler repeat=1 patch (this is fine for tests and non-xDiT runs)"
        ),
        failure_msg=(
            "_xdit_patcher: failed to patch %s; other discovered roots will "
            "still be attempted"
        ),
    )


__all__ = ["ensure_xdit_profiler_patched"]
