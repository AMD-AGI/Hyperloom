#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Unit tests for FlyDSL kernel classification in tracelens_analysis."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unittest
from pathlib import Path
import unittest.mock as mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from tracelens_analysis import (  # noqa: E402
    _defines_traced_triton_kernel,
    _flydsl_kernel_params,
    _flydsl_reusable_roots,
    _looks_like_flydsl_source,
    _reusable_roots,
    classify_patchability,
    derive_kernel_category,
    enrich_candidates_with_runtime_metadata,
    source_type_for,
)
from tracelens_skill_runner import (  # noqa: E402
    UPSTREAM_CATEGORY_TO_GEAK,
    normalize_upstream_category,
)


class TestFlyDSLClassification(unittest.TestCase):
    def _write(self, body: str) -> str:
        tmp = tempfile.NamedTemporaryFile(
            "w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
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
        # A DSL2_ROOT/FLYDSL_ROOT-configured checkout is reusable; no personal
        # or internal storage path is assumed as a built-in default.
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

    def test_flydsl_roots_always_include_known_defaults(self) -> None:
        # Known FlyDSL checkouts present even with no env override.
        env = dict(os.environ)
        env.pop("DSL2_ROOT", None)
        env.pop("FLYDSL_ROOT", None)
        with mock.patch.dict(os.environ, env, clear=True):
            roots = _flydsl_reusable_roots()
            self.assertIn("/sgl-workspace/flydsl/", roots)


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
            derive_kernel_category({"name": "fused_moe_kernel"}),
            "MoE",
        )
        self.assertEqual(
            derive_kernel_category({"name": "gemm_a16w16"}),
            "GEMM",
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


class TestFlyDSLKernelParams(unittest.TestCase):
    """FlyDSL-specific metadata enrichment for GEAK prompt construction."""

    def _write_source(self, body: str) -> str:
        tmp = tempfile.NamedTemporaryFile(
            "w",
            suffix=".py",
            delete=False,
            encoding="utf-8",
        )
        tmp.write(body)
        tmp.flush()
        tmp.close()
        return tmp.name

    def test_target_arch_mapping(self) -> None:
        params = _flydsl_kernel_params("", "mi300x")
        self.assertEqual(params["FLYDSL_TARGET_ARCH"], "gfx942")
        params = _flydsl_kernel_params("", "MI355X")
        self.assertEqual(params["FLYDSL_TARGET_ARCH"], "gfx950")
        params = _flydsl_kernel_params("", "")
        self.assertNotIn("FLYDSL_TARGET_ARCH", params)

    def test_env_passthrough(self) -> None:
        env = {
            "FLYDSL_AUTOTUNE_CACHE_DIR": "/tmp/flydsl-cache",
            "FLYDSL_RUNTIME_ENABLE_CACHE": "1",
        }
        with mock.patch.dict(os.environ, env):
            params = _flydsl_kernel_params("", "mi355x")
            self.assertEqual(params["FLYDSL_AUTOTUNE_CACHE_DIR"], "/tmp/flydsl-cache")
            self.assertEqual(params["FLYDSL_RUNTIME_ENABLE_CACHE"], "1")

    def test_source_smem_marker_detected(self) -> None:
        path = self._write_source(
            "import flydsl.compiler as flyc\n"
            "from flydsl.utils.smem_allocator import SmemAllocator\n"
            "@flyc.kernel\n"
            "def k(): pass\n"
        )
        params = _flydsl_kernel_params(path, "mi355x")
        self.assertTrue(params.get("FLYDSL_USES_SMEM"))

    def test_source_buffer_load_marker_detected(self) -> None:
        path = self._write_source(
            "import flydsl.expr as fx\nfrom flydsl.expr import rocdl\nx = fx.rocdl.make_buffer_tensor(In)\n"
        )
        params = _flydsl_kernel_params(path, "mi355x")
        self.assertTrue(params.get("FLYDSL_USES_BUFFER_LOAD"))

    def test_source_without_markers_omits_fields(self) -> None:
        path = self._write_source("import flydsl.compiler as flyc\n@flyc.kernel\ndef k(): pass\n")
        params = _flydsl_kernel_params(path, "mi355x")
        self.assertNotIn("FLYDSL_USES_SMEM", params)
        self.assertNotIn("FLYDSL_USES_BUFFER_LOAD", params)

    def test_missing_source_does_not_raise(self) -> None:
        params = _flydsl_kernel_params("/no/such/path.py", "mi355x")
        self.assertEqual(params.get("FLYDSL_TARGET_ARCH"), "gfx950")
        self.assertNotIn("FLYDSL_USES_SMEM", params)

    def test_enrich_attaches_flydsl_params_only_for_flydsl(self) -> None:
        path = self._write_source(
            "import flydsl.compiler as flyc\n"
            "from flydsl.utils.smem_allocator import SmemAllocator\n"
            "@flyc.kernel\n"
            "def k(): pass\n"
        )
        args = argparse.Namespace(
            framework="sglang",
            model_name="",
            analysis_mode="inference",
            runtime_env="local",
            target_platform="mi355x",
        )
        flydsl_cand = {
            "name": "k",
            "source_file": path,
            "source_type": "flydsl",
        }
        triton_cand = {
            "name": "t",
            "source_file": path,
            "source_type": "triton",
        }
        enrich_candidates_with_runtime_metadata([flydsl_cand, triton_cand], args)
        self.assertEqual(
            flydsl_cand["kernel_params"]["FLYDSL_TARGET_ARCH"],
            "gfx950",
        )
        self.assertTrue(flydsl_cand["kernel_params"]["FLYDSL_USES_SMEM"])
        self.assertFalse(
            any(k.startswith("FLYDSL_") for k in triton_cand["kernel_params"]),
        )


class TestCandidateEnvForwarding(unittest.TestCase):
    """``FLYDSL_*`` env vars must be forwarded to GEAK candidate metadata."""

    def setUp(self) -> None:
        repo_root = Path(__file__).resolve().parents[5]
        sys.path.insert(0, str(repo_root))
        from hyperloom.orchestrator.kernel import request_handlers as kernel_request_handlers

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
        repo_root = Path(__file__).resolve().parents[5]
        sys.path.insert(0, str(repo_root))
        from hyperloom.orchestrator.kernel import request_handlers as kernel_request_handlers

        self.handlers = kernel_request_handlers

    def test_orchestrator_allowlist_has_flydsl(self) -> None:
        roots = self.handlers._reusable_source_roots()
        self.assertIn("/sgl-workspace/flydsl/", roots)

    def test_orchestrator_honours_dsl2_root_env(self) -> None:
        extra = "/path/user-local/FlyDSL"
        with mock.patch.dict(os.environ, {"DSL2_ROOT": extra}):
            self.assertIn(
                "/path/user-local/flydsl/",
                self.handlers._reusable_source_roots(),
            )


class TestFlyDSLPseudoOpIdentification(unittest.TestCase):
    """pseudo_op names must be classified as FlyDSL by name prefix without on-disk source."""

    def test_pseudo_op_moe_flydsl_stage1_recognised(self) -> None:
        self.assertEqual(
            source_type_for("pseudo_op::moe_flydsl_stage1", ""),
            "flydsl",
        )

    def test_pseudo_op_moe_flydsl_stage2_recognised(self) -> None:
        self.assertEqual(
            source_type_for("pseudo_op::moe_flydsl_stage2", ""),
            "flydsl",
        )

    def test_generic_pseudo_op_flydsl_recognised(self) -> None:
        self.assertEqual(
            source_type_for("pseudo_op::flydsl_custom_kernel", ""),
            "flydsl",
        )

    def test_unrelated_pseudo_op_falls_back_to_unknown(self) -> None:
        """Only flydsl-flavoured pseudo ops should be claimed."""
        self.assertEqual(
            source_type_for("pseudo_op::moe_fused_aiter", ""),
            "unknown",
        )


class TestPseudoOpSourceFallback(unittest.TestCase):
    """GEAK source resolution must fall back off pseudo-op frame labels (prefer real readable source)."""

    def setUp(self) -> None:
        sys.path.insert(0, str(ROOT / "tools"))
        import kernel_optimization

        self.mod = kernel_optimization

    def test_frame_label_candidate_falls_back_to_real_explicit(self) -> None:
        real = str(Path(__file__).resolve())  # any guaranteed-real file
        cand = {"source_file": "aiter/fused_moe.py(986): fused_moe_2stages"}
        out = self.mod._resolve_source_file(real, cand, "k001")
        self.assertEqual(out, real)

    def test_real_candidate_still_wins_over_llm(self) -> None:
        real = str(Path(__file__).resolve())
        cand = {"source_file": real}
        out = self.mod._resolve_source_file("/tmp/some_other.py", cand, "k002")
        self.assertEqual(out, real)

    def test_empty_candidate_uses_llm(self) -> None:
        out = self.mod._resolve_source_file("/tmp/x.py", {"source_file": ""}, "k003")
        self.assertEqual(out, "/tmp/x.py")

    def test_both_unresolvable_returns_candidate(self) -> None:
        """No real file anywhere -> keep candidate (existing behaviour)."""
        cand = {"source_file": "aiter/fused_moe.py(986): fused_moe_2stages"}
        out = self.mod._resolve_source_file("/no/such/file.py", cand, "k004")
        self.assertEqual(out, "aiter/fused_moe.py(986): fused_moe_2stages")


if __name__ == "__main__":
    unittest.main()
