#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for FlyDSL kernel classification in tracelens_analysis."""

from __future__ import annotations

import os
import tempfile
import unittest
import unittest.mock as mock

from hyperloom.inference_optimizer.framework_paths import resolve_known_source_prefixes
from hyperloom.orchestrator.trace_analysis._candidate_assembly import derive_kernel_category
from hyperloom.orchestrator.trace_analysis._kernel_classification import (
    _defines_traced_triton_kernel,
    _flydsl_reusable_roots,
    _looks_like_flydsl_source,
    _reusable_roots,
    classify_patchability,
    source_type_for,
)


class TestFlyDSLClassification(unittest.TestCase):
    def _write(self, body: str) -> str:
        tmp = tempfile.NamedTemporaryFile(
            "w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        )
        self.addCleanup(os.unlink, tmp.name)
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
        path = self._write("from flydsl.compiler import kernel\n\n@kernel\ndef f(): pass\n")
        self.assertTrue(_looks_like_flydsl_source(path))
        self.assertEqual(source_type_for("f", path), "flydsl")

    def test_plain_python_is_not_flydsl(self) -> None:
        path = self._write("import torch\ndef add(a, b): return a + b\n")
        self.assertFalse(_looks_like_flydsl_source(path))
        self.assertEqual(source_type_for("add", path), "python")

    def test_triton_still_wins_over_flydsl(self) -> None:
        """Triton classification has priority when both signals exist."""
        path = self._write("import triton\nimport triton.language as tl\n@triton.jit\ndef k(): pass\n")
        self.assertEqual(source_type_for("triton_kernel", path), "triton")

    def test_shim_imported_triton_kernel_classified_by_definition(self) -> None:
        """A device-symbol name and a shim import still classify as Triton."""
        path = self._write(
            "from vllm.triton_utils import tl, triton\n"
            "\n"
            "@triton.jit\n"
            "def _gqa_sparse_decode_kernel(q_ptr, o_ptr, BLOCK: tl.constexpr):\n"
            "    pass\n"
        )
        name = "hipModuleLaunchKernel->_gqa_sparse_decode_kernel (Synthetic Op)"
        self.assertTrue(_defines_traced_triton_kernel(name, path))
        self.assertEqual(source_type_for(name, path), "triton")

    def test_flydsl_source_keeps_identity_when_it_also_uses_triton(self) -> None:
        """FlyDSL stays FlyDSL even when the file carries a Triton reference path."""
        path = self._write(
            "import flydsl.compiler as flyc\n"
            "import triton\n"
            "\n"
            "@triton.jit\n"
            "def _ref_kernel(x): pass\n"
            "\n"
            "@flyc.kernel\n"
            "def k(x): pass\n"
        )
        self.assertEqual(source_type_for("_ref_kernel", path), "flydsl")

    def test_python_without_a_triton_def_is_not_claimed(self) -> None:
        """A plain def matching the traced name proves nothing about Triton."""
        path = self._write("import torch\ndef _my_kernel(a): return a\n")
        self.assertFalse(_defines_traced_triton_kernel("_my_kernel", path))
        self.assertEqual(source_type_for("_my_kernel", path), "python")

    def test_unmatched_symbol_among_several_triton_defs_is_not_claimed(self) -> None:
        """Several Triton defs and no name match leaves the kernel unproven."""
        path = self._write(
            "import triton\n@triton.jit\ndef _first_kernel(x): pass\n@triton.jit\ndef _second_kernel(x): pass\n"
        )
        self.assertFalse(_defines_traced_triton_kernel("_absent_kernel", path))
        self.assertEqual(source_type_for("_absent_kernel", path), "python")

    def test_missing_source_is_not_claimed(self) -> None:
        self.assertFalse(_defines_traced_triton_kernel("_k", ""))
        self.assertFalse(_defines_traced_triton_kernel("_k", "/nonexistent/path.py"))

    def test_hip_extension_wins_over_flydsl(self) -> None:
        """``.cu`` / ``.cuh`` files keep ``hip_cpp`` regardless of name."""
        tmp = tempfile.NamedTemporaryFile(
            "w",
            suffix=".cu",
            delete=False,
            encoding="utf-8",
        )
        self.addCleanup(os.unlink, tmp.name)
        tmp.write("// flydsl mention in a HIP comment\n")
        tmp.flush()
        tmp.close()
        self.assertEqual(source_type_for("k", tmp.name), "hip_cpp")

    def test_empty_or_missing_source(self) -> None:
        self.assertFalse(_looks_like_flydsl_source(""))
        self.assertFalse(_looks_like_flydsl_source("/nonexistent/path.py"))
        self.assertEqual(source_type_for("k", ""), "unknown")


class TestReusableSourceRoots(unittest.TestCase):
    """FlyDSL install paths must pass the patchability gate."""

    def _flydsl_candidate(self, source_file: str) -> dict:
        return {
            "name": "flydsl_kernel",
            "source_file": source_file,
            "source_type": "flydsl",
        }

    def test_flydsl_sgl_workspace_root_is_reusable(self) -> None:
        roots = _reusable_roots()
        self.assertIn("/sgl-workspace/flydsl/", roots)
        cand = self._flydsl_candidate(
            "/sgl-workspace/flydsl/python/flydsl/ops/some_kernel.py",
        )
        reusable, skip = classify_patchability(cand)
        self.assertTrue(reusable, msg=skip)
        self.assertEqual(skip, "")

    def test_flydsl_env_configured_root_is_reusable(self) -> None:
        # A DSL2_ROOT/FLYDSL_ROOT-configured checkout is reusable; no personal or internal storage path is assumed as
        # a built-in default.
        with mock.patch.dict(os.environ, {"FLYDSL_ROOT": "/opt/flydsl"}):
            cand = self._flydsl_candidate(
                "/opt/flydsl/kernels/moe_gemm_2stage.py",
            )
            reusable, skip = classify_patchability(cand)
            self.assertTrue(reusable, msg=skip)

    def test_random_path_still_rejected(self) -> None:
        cand = self._flydsl_candidate("/path/random/user/checkout/k.py")
        reusable, skip = classify_patchability(cand)
        self.assertFalse(reusable)
        self.assertIn("reusable framework root", skip)

    def test_dsl2_root_env_injects_flydsl_root(self) -> None:
        # $DSL2_ROOT checkout is surfaced lower-cased with a trailing slash.
        extra = "/path/user-local/FlyDSL"
        with mock.patch.dict(os.environ, {"DSL2_ROOT": extra}):
            roots = _flydsl_reusable_roots()
            self.assertIn("/path/user-local/flydsl/", roots)
            cand = self._flydsl_candidate(
                "/path/user-local/flydsl/kernels/k.py",
            )
            reusable, skip = classify_patchability(cand)
            self.assertTrue(reusable, msg=skip)


class TestSourceTypeAdmission(unittest.TestCase):
    """Admitting ``source_type='flydsl'`` leaves the other source types' admission unchanged."""

    def test_existing_source_types_still_admitted(self) -> None:
        for source_type, source_file in (
            ("triton", "/sgl-workspace/aiter/aiter/ops/triton/k.py"),
            ("python", "/sgl-workspace/aiter/aiter/ops/k.py"),
            ("hip_cpp", "/sgl-workspace/aiter/csrc/k.cu"),
        ):
            cand = {"name": "k", "source_file": source_file, "source_type": source_type}
            self.assertTrue(classify_patchability(cand)[0], source_type)

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

    def test_derive_uses_tracelens_category_when_present(self) -> None:
        for raw in ("flydsl", "FlyDSL"):
            cand = {
                "name": "some_op",
                "source_type": "flydsl",
                "tracelens_category": raw,
            }
            self.assertEqual(derive_kernel_category(cand), "FlyDSL")

    def test_derive_falls_back_to_source_type(self) -> None:
        cand = {"name": "some_op", "source_type": "flydsl"}
        self.assertEqual(derive_kernel_category(cand), "FlyDSL")

    def test_derive_existing_categories_unchanged(self) -> None:
        self.assertEqual(
            derive_kernel_category({"name": "fused_moe_kernel"}),
            "MoE",
        )
        self.assertEqual(
            derive_kernel_category({"name": "gemm_a16w16"}),
            "GEMM",
        )


class TestOrchestratorReusableRootsInSync(unittest.TestCase):
    """Orchestrator-side allowlist must stay in sync with the classifier."""

    def test_orchestrator_allowlist_has_flydsl(self) -> None:
        roots = resolve_known_source_prefixes()
        self.assertIn("/sgl-workspace/flydsl/", roots)

    def test_orchestrator_honours_dsl2_root_env(self) -> None:
        extra = "/path/user-local/FlyDSL"
        with mock.patch.dict(os.environ, {"DSL2_ROOT": extra}):
            self.assertIn(
                "/path/user-local/flydsl/",
                resolve_known_source_prefixes(),
            )


class TestFlyDSLPseudoOpIdentification(unittest.TestCase):
    """pseudo_op names must be classified as FlyDSL by name prefix without on-disk source."""

    def test_flydsl_pseudo_ops_recognised(self) -> None:
        for name in ("pseudo_op::moe_flydsl_stage1", "pseudo_op::moe_flydsl_stage2", "pseudo_op::flydsl_custom_kernel"):
            self.assertEqual(source_type_for(name, ""), "flydsl", name)

    def test_unrelated_pseudo_op_falls_back_to_unknown(self) -> None:
        """Only flydsl-flavoured pseudo ops should be claimed."""
        self.assertEqual(
            source_type_for("pseudo_op::moe_fused_aiter", ""),
            "unknown",
        )


if __name__ == "__main__":
    unittest.main()
