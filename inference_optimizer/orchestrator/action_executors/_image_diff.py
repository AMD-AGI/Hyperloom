# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Image-diff accuracy gate for diffusion frameworks.

LLM runs gate correctness with lm-eval (GSM8K). Diffusion runs have no such
eval: a kernel rewrite is "correct" if it still produces (essentially) the same
image for a fixed seed. We compare the post-change generated PNG against the
pre-change baseline PNG and accept when the difference is within tolerance.

Metric: PSNR (peak signal-to-noise ratio, in dB). Higher = more similar;
identical images => +inf. The default threshold (``DEFAULT_PSNR_THRESHOLD_DB``)
is intentionally permissive — kernel rewrites (different reduction order, fp8
GEMMs) perturb pixels slightly without changing the image meaningfully. Tighten
via ``$HYPERLOOM_IMAGE_PSNR_THRESHOLD_DB``.

numpy/Pillow are optional. If neither is importable (e.g. the Hyperloom control
process lacks them), the gate degrades to "skip" (returns None) rather than
failing the run — matching how the lm-eval gate treats a missing result.
"""

from __future__ import annotations

import logging
import math
import os
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# Permissive default: ~35 dB is visually near-identical for natural images;
# kernel-rewrite pixel perturbation typically lands well above this.
DEFAULT_PSNR_THRESHOLD_DB = 30.0

# Image file extensions we consider, in preference order.
_IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def psnr_threshold_db() -> float:
    """Resolve the PSNR pass threshold (env override > default)."""
    raw = os.environ.get("HYPERLOOM_IMAGE_PSNR_THRESHOLD_DB", "").strip()
    if raw:
        try:
            return float(raw)
        except ValueError:
            log.warning(
                "invalid HYPERLOOM_IMAGE_PSNR_THRESHOLD_DB=%r; using default %.1f",
                raw, DEFAULT_PSNR_THRESHOLD_DB,
            )
    return DEFAULT_PSNR_THRESHOLD_DB


def find_latest_image(directory: Path | str) -> Optional[Path]:
    """Return the most-recently-modified image under ``directory`` (recursive).

    Returns None if the directory is missing or contains no image.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    candidates: list[Path] = []
    for ext in _IMAGE_EXTS:
        candidates.extend(directory.rglob(f"*{ext}"))
    if not candidates:
        return None
    return max(candidates, key=_safe_mtime)


def compute_psnr(baseline_img: Path | str, candidate_img: Path | str) -> Optional[float]:
    """Compute PSNR (dB) between two images.

    Returns:
        float dB (``math.inf`` for identical images), or None when the images
        cannot be loaded/compared (missing deps, decode error, shape mismatch).
    """
    try:
        import numpy as np  # type: ignore
        from PIL import Image  # type: ignore
    except ImportError:
        log.warning(
            "image-diff gate: numpy/Pillow unavailable; skipping (gate=None)"
        )
        return None

    try:
        a = np.asarray(Image.open(str(baseline_img)).convert("RGB"), dtype=np.float64)
        b = np.asarray(Image.open(str(candidate_img)).convert("RGB"), dtype=np.float64)
    except Exception as exc:  # noqa: BLE001 - decode/IO errors degrade to skip
        log.warning("image-diff gate: failed to load images: %r", exc)
        return None

    if a.shape != b.shape:
        log.warning(
            "image-diff gate: shape mismatch baseline=%s candidate=%s; skipping",
            a.shape, b.shape,
        )
        return None

    mse = float(((a - b) ** 2).mean())
    if mse == 0.0:
        return math.inf
    return 10.0 * math.log10((255.0 ** 2) / mse)


def image_diff_passed(
    baseline_dir: Path | str,
    candidate_dir: Path | str,
    threshold_db: Optional[float] = None,
) -> Optional[bool]:
    """Compare the latest image in each dir; return pass/fail or None to skip.

    Returns:
        True  — images within tolerance (PSNR >= threshold), accept the change.
        False — images diverged beyond tolerance, reject the change.
        None  — gate could not run (missing image/deps); caller treats as skip.
    """
    base_img = find_latest_image(baseline_dir)
    cand_img = find_latest_image(candidate_dir)
    if base_img is None or cand_img is None:
        log.warning(
            "image-diff gate: missing image (baseline=%s candidate=%s); skipping",
            base_img, cand_img,
        )
        return None

    psnr = compute_psnr(base_img, cand_img)
    if psnr is None:
        return None

    thr = threshold_db if threshold_db is not None else psnr_threshold_db()
    passed = psnr >= thr
    log.info(
        "image-diff gate: PSNR=%.2f dB (threshold=%.2f) -> %s "
        "(baseline=%s candidate=%s)",
        psnr, thr, "PASS" if passed else "FAIL", base_img, cand_img,
    )
    return passed


def _safe_mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


__all__ = [
    "DEFAULT_PSNR_THRESHOLD_DB",
    "compute_psnr",
    "find_latest_image",
    "image_diff_passed",
    "psnr_threshold_db",
]
