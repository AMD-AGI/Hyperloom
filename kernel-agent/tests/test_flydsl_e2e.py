#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""End-to-end regression test for the FlyDSL kernel path (issue #211 §5).

Drives the ``flydsl_naive_gemm.py`` fixture through the full classification + enrichment pipeline.
"""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from tracelens_analysis import (  # noqa: E402
    _flydsl_kernel_params,
    _looks_like_flydsl_source,
    classify_patchability,
    derive_kernel_category,
    enrich_candidates_with_runtime_metadata,
    source_type_for,
)
import kernel_optimization  # noqa: E402

FIXTURE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "flydsl_naive_gemm.py"
)
FIXTURE_DIR = str(FIXTURE_PATH.parent) + "/"


class TestFlyDSLNaiveGemmEndToEnd(unittest.TestCase):
    """Drive the bundled FlyDSL naive GEMM sample through every stage."""

    def setUp(self) -> None:
        self.assertTrue(
            FIXTURE_PATH.is_file(),
            f"FlyDSL fixture missing at {FIXTURE_PATH}; CI cannot detect "
            "FlyDSL-path regressions without it.",
        )

    def _candidate(self) -> dict:
        return {
            "name": "naive_gemm",
            "source_file": str(FIXTURE_PATH),
            "source_type": "flydsl",
        }

    def test_source_sniff_recognises_flydsl(self) -> None:
        self.assertTrue(_looks_like_flydsl_source(str(FIXTURE_PATH)))
        self.assertEqual(
            source_type_for("naive_gemm", str(FIXTURE_PATH)), "flydsl",
        )

    def test_kernel_category_is_flydsl(self) -> None:
        cand = self._candidate()
        self.assertEqual(derive_kernel_category(cand), "FlyDSL")

    def test_patchability_admits_fixture_via_env_override(self) -> None:
        """$FLYDSL_ROOT/$DSL2_ROOT override admits a fixture outside standard roots."""
        with mock.patch.dict(
            os.environ, {"FLYDSL_ROOT": FIXTURE_DIR},
        ):
            reusable, skip = classify_patchability(self._candidate())
        self.assertTrue(reusable, msg=skip)
        self.assertEqual(skip, "")

    def test_kernel_params_extract_arch_and_intrinsics(self) -> None:
        params = _flydsl_kernel_params(str(FIXTURE_PATH), "mi355x")
        self.assertEqual(params["FLYDSL_TARGET_ARCH"], "gfx950")
        self.assertTrue(params.get("FLYDSL_USES_SMEM"))
        self.assertTrue(params.get("FLYDSL_USES_BUFFER_LOAD"))

    def test_enrich_attaches_flydsl_kernel_params(self) -> None:
        cand = self._candidate()
        args = argparse.Namespace(
            framework="sglang",
            model_name="",
            analysis_mode="inference",
            runtime_env="local",
            target_platform="mi355x",
        )
        enrich_candidates_with_runtime_metadata([cand], args)
        params = cand.get("kernel_params") or {}
        self.assertEqual(params["FLYDSL_TARGET_ARCH"], "gfx950")
        self.assertTrue(params["FLYDSL_USES_SMEM"])
        self.assertTrue(params["FLYDSL_USES_BUFFER_LOAD"])

    def test_geak_kernel_type_mapping(self) -> None:
        self.assertEqual(
            kernel_optimization._GEAK_KERNEL_TYPE["flydsl"], "flydsl",
        )


if __name__ == "__main__":
    unittest.main()
