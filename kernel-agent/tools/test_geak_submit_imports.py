# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression: geak_submit must import ``ensure_ray_cluster``.

``geak_submit.run_geak`` calls ``ensure_ray_cluster(...)`` (around line 266)
to boot/attach a Ray cluster before submitting the GEAK job, mirroring
``oob_submit``. The import block only pulled ``quiet_ray_init``, so the call
raised ``NameError: name 'ensure_ray_cluster' is not defined`` at runtime,
failing every GEAK attempt that reached the backend.

This test pins the module-level symbol so the regression cannot reappear.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "backends"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import geak_submit  # noqa: E402


def test_ensure_ray_cluster_is_in_scope():
    # The name must be resolvable in the module globals; otherwise the
    # call site at run_geak() would NameError at runtime.
    assert hasattr(geak_submit, "ensure_ray_cluster"), (
        "geak_submit calls ensure_ray_cluster() but never imported it"
    )
    assert callable(geak_submit.ensure_ray_cluster)
