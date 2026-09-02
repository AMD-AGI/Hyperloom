#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""End-to-end regression test for the FlyDSL kernel path.

Drives the ``flydsl_naive_gemm.py`` fixture through classification + enrichment.
"""

from __future__ import annotations

import argparse
import os
import sys
import unittest
from pathlib import Path
import unittest.mock as mock


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from tracelens_analysis import (  # noqa: E402
    _flydsl_kernel_params,
    _flydsl_reusable_roots,
    _looks_like_flydsl_source,
    classify_patchability,
    derive_kernel_category,
    enrich_candidates_with_runtime_metadata,
    source_type_for,
)
import apply_kernel_patch  # noqa: E402

FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "flydsl_naive_gemm.py"
FIXTURE_DIR = str(FIXTURE_PATH.parent) + "/"


class TestFlyDSLNaiveGemmEndToEnd(unittest.TestCase):
    """Drive the bundled FlyDSL naive GEMM sample through every stage."""

    def setUp(self) -> None:
        self.assertTrue(
            FIXTURE_PATH.is_file(),
            f"FlyDSL fixture missing at {FIXTURE_PATH}; CI cannot detect FlyDSL-path regressions without it.",
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
            source_type_for("naive_gemm", str(FIXTURE_PATH)),
            "flydsl",
        )

    def test_kernel_category_is_flydsl(self) -> None:
        cand = self._candidate()
        self.assertEqual(derive_kernel_category(cand), "FlyDSL")

    def test_patchability_admits_fixture_via_env_override(self) -> None:
        """$FLYDSL_ROOT/$DSL2_ROOT override admits a fixture outside standard roots."""
        with mock.patch.dict(
            os.environ,
            {"FLYDSL_ROOT": FIXTURE_DIR},
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


class TestFlyDSLReachesTheApplyGate(unittest.TestCase):
    """A source the discovery gate admits must also survive the apply gate."""

    def setUp(self) -> None:
        apply_kernel_patch._CACHED_KNOWN_TARGET_ROOTS = None
        self.addCleanup(
            setattr,
            apply_kernel_patch,
            "_CACHED_KNOWN_TARGET_ROOTS",
            None,
        )

    def test_apply_roots_include_flydsl(self) -> None:
        with mock.patch.dict(os.environ, {"FLYDSL_ROOT": FIXTURE_DIR}):
            roots = apply_kernel_patch.known_target_roots()
        self.assertTrue(any("flydsl" in r.lower() for r in roots), roots)

    def test_detect_strategy_admits_a_flydsl_target(self) -> None:
        with mock.patch.dict(os.environ, {"FLYDSL_ROOT": FIXTURE_DIR}):
            strategy = apply_kernel_patch._detect_strategy(
                FIXTURE_PATH,
                allow_unknown_target=False,
            )
        self.assertEqual(strategy["root"], FIXTURE_DIR.rstrip("/"))
        # FlyDSL compiles at import time, so nothing is rebuilt.
        self.assertEqual(strategy["rebuild_command"], [])
        self.assertEqual(strategy["artifact_roots"], [])
        self.assertFalse(strategy["compiled"])

    def test_root_keeps_the_on_disk_spelling(self) -> None:
        # The root list carries a lower-cased variant of every env root, and
        # snapshot mode resolves the patch destination under the returned root,
        # so a spelling that does not exist on disk would fabricate a tree.
        with mock.patch.dict(os.environ, {"FLYDSL_ROOT": FIXTURE_DIR.lower()}):
            root = apply_kernel_patch._flydsl_root_for(FIXTURE_PATH)
        self.assertIsNotNone(root)
        self.assertTrue(root.is_dir(), root)
        self.assertTrue(str(FIXTURE_PATH).startswith(f"{root}/"), root)

    def test_discovery_and_apply_agree_on_the_flydsl_roots(self) -> None:
        with mock.patch.dict(os.environ, {"DSL2_ROOT": FIXTURE_DIR}):
            discovery = set(_flydsl_reusable_roots())
            apply_roots = {r.lower() for r in apply_kernel_patch.known_target_roots()}
        self.assertTrue(discovery <= apply_roots, discovery - apply_roots)


if __name__ == "__main__":
    unittest.main()
