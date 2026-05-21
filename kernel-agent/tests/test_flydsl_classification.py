#!/usr/bin/env python3
"""Unit tests for FlyDSL kernel classification in tracelens_analysis.

Issue #211: ``source_type_for()`` must recognize FlyDSL kernels by
content-sniffing the ``.py`` source (looking for ``@flyc.kernel`` /
``flydsl.compiler`` / ``flydsl.expr`` markers). Without this, FlyDSL
kernels fall through to ``source_type='python'`` and get mis-routed by
the downstream GEAK kernel_type mapping.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from tracelens_analysis import (  # noqa: E402
    _extra_reusable_roots_from_env,
    _looks_like_flydsl_source,
    _reusable_source_roots,
    classify_patchability,
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


class TestReusableSourceRoots(unittest.TestCase):
    """Issue #211: FlyDSL / mori install paths must pass the patchability gate.

    Uses ``source_type="flydsl"`` to exercise the full post-issue-#211
    routing path: content-sniff (item 1) → reusable-root allowlist
    (item 2) → ``source_type`` allowlist (item 3, the entry asserted
    below). All three gates must pass for a FlyDSL kernel to reach
    backend dispatch.
    """

    def _flydsl_candidate(self, source_file: str) -> dict:
        return {
            "name": "flydsl_indexed_pv_kernel",
            "source_file": source_file,
            "source_type": "flydsl",
        }

    def test_flydsl_sgl_workspace_root_is_reusable(self) -> None:
        roots = _reusable_source_roots()
        self.assertIn("/sgl-workspace/flydsl/", roots)
        cand = self._flydsl_candidate(
            "/sgl-workspace/flydsl/python/flydsl/ops/some_kernel.py",
        )
        reusable, skip = classify_patchability(cand)
        self.assertTrue(reusable, msg=skip)
        self.assertEqual(skip, "")

    def test_flydsl_site_packages_root_is_reusable(self) -> None:
        cand = self._flydsl_candidate(
            "/usr/local/lib/python3.12/dist-packages/flydsl/kernels/k.py",
        )
        reusable, skip = classify_patchability(cand)
        self.assertTrue(reusable, msg=skip)

    def test_mori_root_is_reusable(self) -> None:
        cand = {
            "name": "mori_kernel",
            "source_file": "/sgl-workspace/mori/ops/k.py",
            "source_type": "python",
        }
        reusable, skip = classify_patchability(cand)
        self.assertTrue(reusable, msg=skip)

    def test_random_path_still_rejected(self) -> None:
        cand = self._flydsl_candidate("/wekafs/random/user/checkout/k.py")
        reusable, skip = classify_patchability(cand)
        self.assertFalse(reusable)
        self.assertIn("reusable framework root", skip)

    def test_env_extension_admits_extra_root(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"HYPERLOOM_EXTRA_REUSABLE_ROOTS": "/wekafs/yunkai/helios-demo/app/"},
        ):
            self.assertIn(
                "/wekafs/yunkai/helios-demo/app/",
                _extra_reusable_roots_from_env(),
            )
            self.assertIn(
                "/wekafs/yunkai/helios-demo/app/", _reusable_source_roots(),
            )
            cand = self._flydsl_candidate(
                "/wekafs/yunkai/helios-demo/app/helios_demo/flydsl_mla_decode/"
                "flydsl_indexed_pv_kernel.py",
            )
            reusable, skip = classify_patchability(cand)
            self.assertTrue(reusable, msg=skip)

    def test_env_extension_normalizes_trailing_slash(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"HYPERLOOM_EXTRA_REUSABLE_ROOTS": "/no/trailing/slash"},
        ):
            roots = _extra_reusable_roots_from_env()
            self.assertEqual(roots, ("/no/trailing/slash/",))

    def test_env_extension_multiple_colon_separated(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"HYPERLOOM_EXTRA_REUSABLE_ROOTS": "/a/:/b/:/c"},
        ):
            roots = _extra_reusable_roots_from_env()
            self.assertEqual(roots, ("/a/", "/b/", "/c/"))

    def test_env_extension_absent_returns_empty(self) -> None:
        env = dict(os.environ)
        env.pop("HYPERLOOM_EXTRA_REUSABLE_ROOTS", None)
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertEqual(_extra_reusable_roots_from_env(), ())


class TestSourceTypeAdmission(unittest.TestCase):
    """Issue #211: ``source_type='flydsl'`` must pass the patchability gate."""

    _BASE = "/sgl-workspace/flydsl/python/flydsl/ops/k.py"

    def test_flydsl_source_type_admitted(self) -> None:
        cand = {"name": "k", "source_file": self._BASE, "source_type": "flydsl"}
        reusable, skip = classify_patchability(cand)
        self.assertTrue(reusable, msg=skip)
        self.assertEqual(skip, "")

    def test_triton_source_type_still_admitted(self) -> None:
        cand = {
            "name": "k",
            "source_file": "/sgl-workspace/aiter/aiter/ops/triton/k.py",
            "source_type": "triton",
        }
        self.assertTrue(classify_patchability(cand)[0])

    def test_python_source_type_still_admitted(self) -> None:
        cand = {
            "name": "k",
            "source_file": "/sgl-workspace/aiter/aiter/ops/k.py",
            "source_type": "python",
        }
        self.assertTrue(classify_patchability(cand)[0])

    def test_hip_cpp_source_type_still_admitted(self) -> None:
        cand = {
            "name": "k",
            "source_file": "/sgl-workspace/aiter/csrc/k.cu",
            "source_type": "hip_cpp",
        }
        self.assertTrue(classify_patchability(cand)[0])

    def test_unknown_source_type_still_rejected(self) -> None:
        cand = {
            "name": "k",
            "source_file": "/sgl-workspace/aiter/k.bin",
            "source_type": "unknown",
        }
        reusable, skip = classify_patchability(cand)
        self.assertFalse(reusable)
        self.assertIn("source_type=", skip)
        # Reason string is updated to include "flydsl" in the admitted set.
        self.assertIn("flydsl", skip)


class TestOrchestratorReusableRootsInSync(unittest.TestCase):
    """The reusable-root allowlist is duplicated in two files on purpose
    (kernel-agent side vs orchestrator-side guard). Both lists must
    advertise FlyDSL / mori roots and honour ``HYPERLOOM_EXTRA_REUSABLE_ROOTS``
    so callers see identical routing behaviour from either entry point.
    """

    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root))
        from inference_optimizer.orchestrator import kernel_request_handlers
        self.handlers = kernel_request_handlers

    def test_orchestrator_allowlist_has_flydsl(self) -> None:
        self.assertIn(
            "/sgl-workspace/flydsl/", self.handlers._REUSABLE_SOURCE_ROOTS_STATIC,
        )
        self.assertIn(
            "/usr/local/lib/python3.12/dist-packages/flydsl/",
            self.handlers._REUSABLE_SOURCE_ROOTS_STATIC,
        )

    def test_orchestrator_allowlist_has_mori(self) -> None:
        self.assertIn(
            "/sgl-workspace/mori/", self.handlers._REUSABLE_SOURCE_ROOTS_STATIC,
        )

    def test_orchestrator_honours_env_override(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"HYPERLOOM_EXTRA_REUSABLE_ROOTS": "/wekafs/yunkai/helios-demo/app/"},
        ):
            roots = self.handlers._reusable_source_roots()
            self.assertIn("/wekafs/yunkai/helios-demo/app/", roots)


if __name__ == "__main__":
    unittest.main()
