#!/usr/bin/env python3
"""Unit tests for FlyDSL kernel classification in tracelens_analysis.

Issue #211: ``source_type_for()`` must recognize FlyDSL kernels by
content-sniffing the ``.py`` source (looking for ``@flyc.kernel`` /
``flydsl.compiler`` / ``flydsl.expr`` markers). Without this, FlyDSL
kernels fall through to ``source_type='python'`` and get mis-routed by
the downstream GEAK kernel_type mapping.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from tracelens_analysis import (  # noqa: E402
    _looks_like_flydsl_source,
    source_type_for,
)


HELIOS_FLYDSL_KERNEL = Path(
    "/wekafs/yunkai/helios-demo/app/helios_demo/flydsl_mla_decode/"
    "flydsl_indexed_pv_kernel.py"
)


class TestFlyDSLClassification(unittest.TestCase):
    def _write(self, body: str) -> str:
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".py", delete=False, encoding="utf-8",
        )
        tmp.write(body)
        tmp.flush()
        tmp.close()
        return tmp.name

    def test_detects_flyc_kernel_decorator(self) -> None:
        path = self._write(
            "import flydsl.compiler as flyc\n"
            "import flydsl.expr as fx\n"
            "\n"
            "@flyc.kernel\n"
            "def my_kernel(In: fx.Tensor, Out: fx.Tensor):\n"
            "    pass\n"
        )
        self.assertTrue(_looks_like_flydsl_source(path))
        self.assertEqual(source_type_for("my_kernel", path), "flydsl")

    def test_detects_flydsl_compiler_import(self) -> None:
        path = self._write(
            "from flydsl.compiler import kernel\n"
            "\n"
            "@kernel\n"
            "def f(): pass\n"
        )
        self.assertTrue(_looks_like_flydsl_source(path))
        self.assertEqual(source_type_for("f", path), "flydsl")

    def test_detects_helios_demo_kernel(self) -> None:
        """End-to-end fixture: real helios-demo FlyDSL kernel.

        Skipped when the helios-demo checkout is not on disk; the
        local Python-snippet fixtures above cover the same code path.
        """
        if not HELIOS_FLYDSL_KERNEL.exists():
            self.skipTest(f"fixture missing: {HELIOS_FLYDSL_KERNEL}")
        self.assertTrue(_looks_like_flydsl_source(str(HELIOS_FLYDSL_KERNEL)))
        self.assertEqual(
            source_type_for(
                "flydsl_indexed_pv_kernel", str(HELIOS_FLYDSL_KERNEL),
            ),
            "flydsl",
        )

    def test_plain_python_is_not_flydsl(self) -> None:
        path = self._write(
            "import torch\n"
            "def add(a, b): return a + b\n"
        )
        self.assertFalse(_looks_like_flydsl_source(path))
        self.assertEqual(source_type_for("add", path), "python")

    def test_triton_still_wins_over_flydsl(self) -> None:
        """Triton classification has priority when both signals exist.

        A file named with the ``triton`` substring and a ``.py`` suffix
        should keep its ``triton`` source_type even if (hypothetically)
        it also imported flydsl — the kernel-name signal is more stable
        than first-4KiB content sniffing.
        """
        path = self._write(
            "import triton\n"
            "import triton.language as tl\n"
            "@triton.jit\n"
            "def k(): pass\n"
        )
        self.assertEqual(source_type_for("triton_kernel", path), "triton")

    def test_hip_extension_wins_over_flydsl(self) -> None:
        """``.cu`` / ``.cuh`` files keep ``hip_cpp`` regardless of name."""
        tmp = tempfile.NamedTemporaryFile(
            "w", suffix=".cu", delete=False, encoding="utf-8",
        )
        tmp.write("// flydsl mention in a HIP comment\n")
        tmp.flush()
        tmp.close()
        self.assertEqual(source_type_for("k", tmp.name), "hip_cpp")

    def test_empty_or_missing_source(self) -> None:
        self.assertFalse(_looks_like_flydsl_source(""))
        self.assertFalse(_looks_like_flydsl_source("/nonexistent/path.py"))
        self.assertEqual(source_type_for("k", ""), "unknown")


if __name__ == "__main__":
    unittest.main()
