# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Idempotent, backward-compatible patcher for xDiT's diffusion profiling.

This module applies two in-place, fail-soft patches to xDiT's
``base_model.py`` so the diffusion roofline pipeline gets a usable trace:

Patch 1 — ``repeat=1`` (retain the profiler active window)
----------------------------------------------------------
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
~130k cpu_op events. This patch inserts ``repeat=1`` into that schedule so one
cycle is retained and exported.

Patch 2 — per-denoise-step annotations (deterministic roofline split)
---------------------------------------------------------------------
The diffusion roofline needs per-denoise-step boundaries to attribute kernel
time to steps. ``profile()`` only wraps a whole image in a single
``record_function("model_inference")``; the denoise loop lives inside the
diffusers pipeline ``__call__`` (per model subclass), which base_model.py
cannot reach directly. Instead this patch injects, at the top of the profiler
``with`` block, a best-effort wrapper around ``self.pipe.scheduler.step`` — which
diffusers calls exactly once per denoise step — that emits a
``record_function("denoise_step_<i>")`` marker per step. Those ``user_annotation``
events give TraceLens deterministic per-step split anchors instead of relying
solely on steady-state kernel-pattern heuristics. Model-agnostic (all xDiT DiT
pipelines expose ``.scheduler.step``) and fully fail-soft: any missing attribute
or error leaves the run unchanged.

Both patches are applied in place, once, never reverted: idempotent via sentinel
substrings, serialized across processes via ``fcntl.flock``, written atomically
(temp file + ``os.replace``). A file already carrying only Patch 1 (from an older
Hyperloom) is upgraded to also carry Patch 2. Each patch fails soft independently
when its anchor is missing so callers can proceed on a partially-patched file.
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
# Patch-1 ("repeat=1 present?") sentinel.
_PATCH_SENTINEL = "# hyperloom: retain active window"

# ---- Patch 2: per-denoise-step annotation ---------------------------------
# Anchor A: opening line of the profiler ``with`` block. Present in both the
# single-line (``with profile(schedule=schedule) as profile_object:``) and the
# multi-line upstream form; the leading indentation stays because it precedes
# the matched substring.
_ANNOT_INSTALL_ANCHOR = ") as profile_object:\n"
# Injected right after the ``with profile(...) as profile_object:`` line, at the
# 12-space body indent. Wraps ``self.pipe.scheduler.step`` (one call per denoise
# step) in a ``record_function("denoise_step_<i>")`` marker. Best-effort: any
# missing attribute / error leaves the run unchanged. ``_hl_reset_denoise_counter``
# is always defined so the per-image reset call (Anchor B) is safe even if the
# wrap could not be installed.
_ANNOT_INSTALL_BLOCK = (
    ") as profile_object:\n"
    "            # hyperloom: per-denoise-step annotation -- wrap the diffusers\n"
    "            # scheduler.step (one call per denoise step) in a record_function\n"
    "            # so TraceLens gets deterministic per-step split boundaries.\n"
    "            _hl_denoise_step = {\"i\": 0, \"active\": False}\n"
    "            def _hl_reset_denoise_counter():\n"
    "                _hl_denoise_step[\"i\"] = 0\n"
    "            try:\n"
    "                _hl_sched = getattr(getattr(self, \"pipe\", None), \"scheduler\", None)\n"
    "                # xfuser wraps the real diffusers scheduler; its __setattr__\n"
    "                # redirects attribute writes to ``.module`` while its ``step``\n"
    "                # delegates to ``self.module.step``. Capturing and patching the\n"
    "                # same object (the module when present) is required -- otherwise\n"
    "                # the captured original is the wrapper's ``step`` and calling it\n"
    "                # re-enters our wrapper forever (RecursionError).\n"
    "                _hl_target = getattr(_hl_sched, \"module\", None)\n"
    "                if _hl_target is None:\n"
    "                    _hl_target = _hl_sched\n"
    "                _hl_orig_step = getattr(_hl_target, \"step\", None)\n"
    "                if _hl_orig_step is not None and not getattr(\n"
    "                    _hl_orig_step, \"_hyperloom_wrapped\", False\n"
    "                ):\n"
    "                    def _hl_wrapped_step(*_hl_a, **_hl_k):\n"
    "                        # Re-entrancy guard: only the outermost step call emits\n"
    "                        # the marker and advances the counter; any nested\n"
    "                        # re-dispatch delegates straight to the original.\n"
    "                        if _hl_denoise_step[\"active\"]:\n"
    "                            return _hl_orig_step(*_hl_a, **_hl_k)\n"
    "                        _hl_denoise_step[\"active\"] = True\n"
    "                        try:\n"
    "                            with record_function(\n"
    "                                f\"denoise_step_{_hl_denoise_step['i']}\"\n"
    "                            ):\n"
    "                                _hl_ret = _hl_orig_step(*_hl_a, **_hl_k)\n"
    "                        finally:\n"
    "                            _hl_denoise_step[\"active\"] = False\n"
    "                        _hl_denoise_step[\"i\"] += 1\n"
    "                        return _hl_ret\n"
    "                    _hl_wrapped_step._hyperloom_wrapped = True\n"
    "                    _hl_target.step = _hl_wrapped_step\n"
    "            except Exception:  # noqa: BLE001 - annotation is best-effort\n"
    "                pass\n"
)
# Anchor B: the per-image inference marker. A ``_hl_reset_denoise_counter()``
# call is inserted before it so each profiled image restarts step indexing at 0.
_ANNOT_LOOP_ANCHOR = '                with record_function("model_inference"):'
_ANNOT_LOOP_PATCHED = (
    "                _hl_reset_denoise_counter()  "
    "# hyperloom: per-denoise-step annotation\n"
    '                with record_function("model_inference"):'
)
# Patch-2 ("per-step annotation present?") sentinel.
_ANNOT_SENTINEL = "# hyperloom: per-denoise-step annotation"

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
    """Return whether ``base_model.py`` carries BOTH Hyperloom patches.

    A file with only Patch 1 (``repeat=1``, from an older Hyperloom) reports
    ``False`` so ``_apply_patch_atomic`` upgrades it to also add Patch 2.

    Args:
        src: The xfuser ``base_model.py`` file to inspect.

    Returns:
        ``True`` only if both the ``repeat=1`` and the per-denoise-step sentinels
        are present; ``False`` on a miss or read error.
    """
    return file_contains_sentinel(
        src, _PATCH_SENTINEL, log, "_xdit_patcher"
    ) and file_contains_sentinel(src, _ANNOT_SENTINEL, log, "_xdit_patcher")


def _apply_repeat_patch(text: str, src: Path) -> "tuple[str, bool]":
    """Apply Patch 1 (``repeat=1``) to ``text`` if not already present.

    Returns the (possibly unchanged) text and whether it was modified. Logs a
    warning and leaves ``text`` untouched when the legacy schedule block is
    missing (xDiT layout drift), so the caller can still attempt Patch 2.
    """
    if _PATCH_SENTINEL in text:
        return text, False
    if _LEGACY_BLOCK not in text:
        log.warning(
            "_xdit_patcher: expected torch.profiler.schedule(...) block not "
            "found in %s; xDiT layout may have changed and Hyperloom needs an "
            "updated patch. The profiler active window will be discarded "
            "(empty diffusion trace -> roofline REVERT). Manual review needed.",
            src,
        )
        return text, False
    patched = text.replace(_LEGACY_BLOCK, _PATCHED_BLOCK, 1)
    return patched, patched != text


def _apply_annotation_patch(text: str, src: Path) -> "tuple[str, bool]":
    """Apply Patch 2 (per-denoise-step annotation) to ``text`` if not present.

    Requires both anchors (the profiler ``with`` line and the per-image
    ``model_inference`` marker). Logs a warning and leaves ``text`` untouched
    when either anchor is missing, so the caller can still keep Patch 1.
    """
    if _ANNOT_SENTINEL in text:
        return text, False
    if _ANNOT_INSTALL_ANCHOR not in text or _ANNOT_LOOP_ANCHOR not in text:
        log.warning(
            "_xdit_patcher: per-denoise-step annotation anchors not found in "
            "%s (profiler-with line and/or model_inference marker); xDiT layout "
            "may have changed. The diffusion roofline will fall back to "
            "steady-state kernel-pattern splitting (no explicit per-step "
            "annotations). Manual review needed.",
            src,
        )
        return text, False
    patched = text.replace(_ANNOT_INSTALL_ANCHOR, _ANNOT_INSTALL_BLOCK, 1)
    patched = patched.replace(_ANNOT_LOOP_ANCHOR, _ANNOT_LOOP_PATCHED, 1)
    return patched, patched != text


def _apply_patch_atomic(src: Path) -> bool:
    """Apply both Hyperloom patches to ``base_model.py`` via temp-file + atomic
    rename so a crash mid-write cannot corrupt the file.

    Patch 1 (``repeat=1``) and Patch 2 (per-denoise-step annotation) each apply
    independently and fail soft when their anchor is missing. The two are
    written together in a single atomic replace.

    Args:
        src: The xfuser ``base_model.py`` file to patch in place.

    Returns:
        ``True`` when new patched bytes were written; ``False`` when nothing
        changed (both already present, both anchors missing) or any IO step
        fails.
    """
    try:
        original = src.read_text(encoding="utf-8")
    except OSError as e:
        log.warning("_xdit_patcher: cannot read %s: %s", src, e)
        return False

    text = original
    text, repeat_applied = _apply_repeat_patch(text, src)
    text, annot_applied = _apply_annotation_patch(text, src)

    if text == original:
        return False

    if not atomic_write_text(
        src,
        text,
        tmp_prefix=".base_model.py.hyperloom_",
        log_prefix="_xdit_patcher",
    ):
        return False

    if repeat_applied:
        log.info(
            "_xdit_patcher: added repeat=1 to the torch.profiler.schedule in "
            "%s so the diffusion profiler retains its active window (else the "
            "trace is empty: Op count == 0)",
            src,
        )
    if annot_applied:
        log.info(
            "_xdit_patcher: added per-denoise-step record_function markers "
            "(denoise_step_<i>) around scheduler.step in %s so TraceLens gets "
            "deterministic per-step roofline split boundaries",
            src,
        )
    return True


def ensure_xdit_profiler_patched() -> bool:
    """Ensure xDiT's ``base_model.py`` carries both Hyperloom diffusion patches.

    Applies Patch 1 (``repeat=1``, retain the profiler active window) and Patch 2
    (per-denoise-step ``record_function`` markers for roofline splitting). Returns
    ``True`` when patched at exit, ``False`` (non-fatal) when the file is missing
    or every patch anchor is absent. Concurrency-safe (flock + atomic rename;
    already-patched fast-path skips the lock).

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
            "diffusion profiler patches (this is fine for tests and non-xDiT runs)"
        ),
        failure_msg=(
            "_xdit_patcher: failed to patch %s; other discovered roots will "
            "still be attempted"
        ),
    )


__all__ = ["ensure_xdit_profiler_patched"]
