# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Unit tests for the offline diffusion image-diff accuracy gate."""

from __future__ import annotations

import math

import pytest

from inference_optimizer.orchestrator.action_executors import _image_diff as idf


# numpy/Pillow are optional in the control process; PSNR-math tests are skipped
# when unavailable, but the degrade-to-skip behaviour is always tested.
_HAS_IMAGING = True
try:  # pragma: no cover - availability probe
    import numpy as _np  # noqa: F401
    from PIL import Image as _Image  # noqa: F401
except ImportError:  # pragma: no cover
    _HAS_IMAGING = False

requires_imaging = pytest.mark.skipif(
    not _HAS_IMAGING, reason="numpy/Pillow not installed"
)


class TestThreshold:
    def test_default_threshold(self, monkeypatch):
        monkeypatch.delenv("HYPERLOOM_IMAGE_PSNR_THRESHOLD_DB", raising=False)
        assert idf.psnr_threshold_db() == idf.DEFAULT_PSNR_THRESHOLD_DB

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_IMAGE_PSNR_THRESHOLD_DB", "42.5")
        assert idf.psnr_threshold_db() == pytest.approx(42.5)

    def test_invalid_env_falls_back_to_default(self, monkeypatch):
        monkeypatch.setenv("HYPERLOOM_IMAGE_PSNR_THRESHOLD_DB", "notafloat")
        assert idf.psnr_threshold_db() == idf.DEFAULT_PSNR_THRESHOLD_DB


class TestFindLatestImage:
    def test_returns_none_for_missing_dir(self, tmp_path):
        assert idf.find_latest_image(tmp_path / "nope") is None

    def test_returns_none_when_no_images(self, tmp_path):
        (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
        assert idf.find_latest_image(tmp_path) is None

    def test_finds_png(self, tmp_path):
        img = tmp_path / "generated_0.png"
        img.write_bytes(b"\x89PNG\r\n")
        assert idf.find_latest_image(tmp_path) == img

    def test_picks_most_recent(self, tmp_path):
        import os
        import time

        old = tmp_path / "old.png"
        new = tmp_path / "new.png"
        old.write_bytes(b"\x89PNG")
        new.write_bytes(b"\x89PNG")
        # Force a clear mtime ordering.
        os.utime(old, (time.time() - 100, time.time() - 100))
        assert idf.find_latest_image(tmp_path) == new

    def test_finds_nested_image(self, tmp_path):
        sub = tmp_path / "sub"
        sub.mkdir()
        img = sub / "out_0.png"
        img.write_bytes(b"\x89PNG")
        assert idf.find_latest_image(tmp_path) == img


class TestImageDiffPassedDegradesToSkip:
    def test_missing_baseline_image_returns_none(self, tmp_path):
        cand = tmp_path / "cand"
        cand.mkdir()
        (cand / "c.png").write_bytes(b"\x89PNG")
        assert idf.image_diff_passed(tmp_path / "missing", cand) is None

    def test_missing_candidate_image_returns_none(self, tmp_path):
        base = tmp_path / "base"
        base.mkdir()
        (base / "b.png").write_bytes(b"\x89PNG")
        assert idf.image_diff_passed(base, tmp_path / "missing") is None


@requires_imaging
class TestPsnrMath:
    def _save(self, path, arr):
        from PIL import Image

        Image.fromarray(arr).save(str(path))

    def test_identical_images_return_inf(self, tmp_path):
        import numpy as np

        arr = (np.ones((8, 8, 3)) * 128).astype("uint8")
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        self._save(a, arr)
        self._save(b, arr)
        assert idf.compute_psnr(a, b) == math.inf

    def test_different_images_finite_psnr(self, tmp_path):
        import numpy as np

        a_arr = (np.ones((8, 8, 3)) * 100).astype("uint8")
        b_arr = (np.ones((8, 8, 3)) * 110).astype("uint8")
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        self._save(a, a_arr)
        self._save(b, b_arr)
        psnr = idf.compute_psnr(a, b)
        # MSE = 100 -> PSNR = 10*log10(65025/100) ~= 28.13 dB
        assert psnr == pytest.approx(10 * math.log10(255.0 ** 2 / 100.0), rel=1e-3)

    def test_shape_mismatch_returns_none(self, tmp_path):
        import numpy as np

        self._save(tmp_path / "a.png", (np.ones((8, 8, 3)) * 100).astype("uint8"))
        self._save(tmp_path / "b.png", (np.ones((4, 4, 3)) * 100).astype("uint8"))
        assert idf.compute_psnr(tmp_path / "a.png", tmp_path / "b.png") is None

    def test_identical_images_pass_gate(self, tmp_path):
        import numpy as np

        arr = (np.ones((8, 8, 3)) * 128).astype("uint8")
        base = tmp_path / "base"
        cand = tmp_path / "cand"
        base.mkdir()
        cand.mkdir()
        self._save(base / "b.png", arr)
        self._save(cand / "c.png", arr)
        assert idf.image_diff_passed(base, cand) is True

    def test_divergent_images_fail_gate(self, tmp_path):
        import numpy as np

        base = tmp_path / "base"
        cand = tmp_path / "cand"
        base.mkdir()
        cand.mkdir()
        self._save(base / "b.png", (np.zeros((8, 8, 3))).astype("uint8"))
        self._save(cand / "c.png", (np.ones((8, 8, 3)) * 255).astype("uint8"))
        # Maximal difference -> very low PSNR -> fail against any sane threshold.
        assert idf.image_diff_passed(base, cand) is False

    def test_threshold_boundary_respected(self, tmp_path, monkeypatch):
        import numpy as np

        base = tmp_path / "base"
        cand = tmp_path / "cand"
        base.mkdir()
        cand.mkdir()
        self._save(base / "b.png", (np.ones((8, 8, 3)) * 100).astype("uint8"))
        self._save(cand / "c.png", (np.ones((8, 8, 3)) * 110).astype("uint8"))
        # Actual PSNR ~28.13 dB. Threshold just below => pass; just above => fail.
        monkeypatch.setenv("HYPERLOOM_IMAGE_PSNR_THRESHOLD_DB", "28.0")
        assert idf.image_diff_passed(base, cand) is True
        monkeypatch.setenv("HYPERLOOM_IMAGE_PSNR_THRESHOLD_DB", "29.0")
        assert idf.image_diff_passed(base, cand) is False
