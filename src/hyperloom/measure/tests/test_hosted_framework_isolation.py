# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Two tests one after the other on the local sandbox provider: the second sees the framework the first one damaged.

The fake runner is the stand-in for a test that writes into the image's framework install (aiter's tuned-config
cleanup was one); its root is a directory of the test's own, never the machine's ``/sgl-workspace``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.measure.tests.hosted_fixtures import make_call, make_rig, point_yaml, runner_template
from hyperloom.sandbox.tests.framework_overlay import make_framework, overlay_unavailable_reason

_PRISTINE = b"M,N,K\n64,5120,5120\n"


def _test(rig, tmp: Path, task: str, target: Path):
    call = make_call(tmp, task, point=point_yaml(tmp, f"{task}.yaml", fake_framework_file=target))
    return call, rig.backend.run_test(call)


def test_a_test_starts_from_the_framework_as_it_was_whatever_the_tests_before_it_wrote(tmp_path: Path) -> None:
    reason = overlay_unavailable_reason(tmp_path)
    if reason is not None:
        pytest.skip(f"this machine cannot give a process a private mount namespace: {reason}")
    root = make_framework(tmp_path)
    target = root / "aiter/configs/tuned.csv"
    rig = make_rig(tmp_path, templates=[runner_template(framework_roots=(str(root),))])
    try:
        for task in ("first", "second", "third"):
            call, outcome = _test(rig, tmp_path, task, target)
            assert outcome.returncode == 0
            assert (call.output_dir / "framework_seen.txt").read_bytes() == _PRISTINE, f"{task} saw a damaged table"
    finally:
        rig.backend.close()
        rig.provider.close()
    assert target.read_bytes() == _PRISTINE
    assert (root / "aiter/configs/framework_victim").exists()
    assert not (root / "aiter/configs/framework_leftover").exists()
