#!/usr/bin/env python3
"""Unit tests for FlyDSL kernel classification in tracelens_analysis."""

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
    derive_kernel_category,
    source_type_for,
)
from tracelens_skill_runner import (  # noqa: E402
    UPSTREAM_CATEGORY_TO_GEAK,
    normalize_upstream_category,
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
        """Real helios-demo FlyDSL kernel; skipped when checkout is absent."""
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
        """Triton classification has priority when both signals exist."""
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
    """FlyDSL / mori install paths must pass the patchability gate."""

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
    """``source_type='flydsl'`` must pass the patchability gate."""

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
        self.assertIn("flydsl", skip)


class TestKernelCategoryDerivation(unittest.TestCase):
    """FlyDSL must surface as ``kernel_category="FlyDSL"``."""

    def test_upstream_tracelens_flydsl_mapped(self) -> None:
        self.assertEqual(UPSTREAM_CATEGORY_TO_GEAK["flydsl"], "FlyDSL")
        self.assertEqual(normalize_upstream_category("flydsl"), "FlyDSL")
        self.assertEqual(normalize_upstream_category("FlyDSL"), "FlyDSL")

    def test_derive_uses_tracelens_category_when_present(self) -> None:
        cand = {
            "name": "some_op",
            "source_type": "flydsl",
            "tracelens_category": "flydsl",
        }
        self.assertEqual(derive_kernel_category(cand), "FlyDSL")

    def test_derive_falls_back_to_source_type(self) -> None:
        cand = {"name": "some_op", "source_type": "flydsl"}
        self.assertEqual(derive_kernel_category(cand), "FlyDSL")

    def test_derive_existing_categories_unchanged(self) -> None:
        self.assertEqual(
            derive_kernel_category({"name": "fused_moe_kernel"}), "MoE",
        )
        self.assertEqual(
            derive_kernel_category({"name": "gemm_a16w16"}), "GEMM",
        )


class TestGEAKKernelTypeMapping(unittest.TestCase):
    """``source_type=flydsl`` must map to GEAK's ``kernel_type="flydsl"``."""

    def setUp(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        import kernel_optimization
        self.mod = kernel_optimization

    def test_flydsl_source_type_maps_to_flydsl(self) -> None:
        self.assertEqual(self.mod._GEAK_KERNEL_TYPE["flydsl"], "flydsl")

    def test_existing_mappings_preserved(self) -> None:
        self.assertEqual(self.mod._GEAK_KERNEL_TYPE["triton"], "triton")
        self.assertEqual(self.mod._GEAK_KERNEL_TYPE["hip_cpp"], "hip")
        self.assertEqual(self.mod._GEAK_KERNEL_TYPE["python"], "other")
        self.assertEqual(self.mod._GEAK_KERNEL_TYPE["vendor_binary"], "other")
        self.assertEqual(self.mod._GEAK_KERNEL_TYPE["unknown"], "other")


class TestCandidateEnvForwarding(unittest.TestCase):
    """``FLYDSL_*`` env vars must be forwarded to GEAK candidate metadata."""

    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root))
        from inference_optimizer.orchestrator import kernel_request_handlers
        self.h = kernel_request_handlers

    def test_flydsl_prefix_allowed(self) -> None:
        self.assertIn("FLYDSL_", self.h._CANDIDATE_ENV_PREFIXES)
        self.assertTrue(self.h._candidate_env_allowed("FLYDSL_AUTOTUNE_CACHE_DIR"))
        self.assertTrue(self.h._candidate_env_allowed("FLYDSL_RUNTIME_ENABLE_CACHE"))

    def test_existing_prefixes_preserved(self) -> None:
        self.assertTrue(self.h._candidate_env_allowed("SGLANG_FOO"))
        self.assertTrue(self.h._candidate_env_allowed("TRITON_BAR"))

    def test_sensitive_keys_still_blocked(self) -> None:
        self.assertFalse(self.h._candidate_env_allowed("FLYDSL_API_KEY"))
        self.assertFalse(self.h._candidate_env_allowed("FLYDSL_SECRET_TOKEN"))

    def test_unrelated_envs_still_rejected(self) -> None:
        self.assertFalse(self.h._candidate_env_allowed("HOME"))
        self.assertFalse(self.h._candidate_env_allowed("PATH"))


class TestOrchestratorReusableRootsInSync(unittest.TestCase):
    """Orchestrator-side allowlist must stay in sync with the classifier."""

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
