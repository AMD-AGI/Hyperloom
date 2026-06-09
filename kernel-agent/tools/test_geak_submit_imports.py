# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression: geak_submit must import ``ensure_ray_cluster`` (else run_geak NameErrors at runtime)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backends"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geak_submit  # noqa: E402


def test_ensure_ray_cluster_is_in_scope():
    assert hasattr(geak_submit, "ensure_ray_cluster"), (
        "geak_submit calls ensure_ray_cluster() but never imported it"
    )
    assert callable(geak_submit.ensure_ray_cluster)
