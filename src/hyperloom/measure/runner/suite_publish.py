# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish the inference_suite staging a runner of one commit runs, into ``<runner cache>/inference-suite/``.

It runs where the runner cache is built, on a host with network, after the commit's entry is in place
(:mod:`hyperloom.measure.runner.cache_build`): never at import time, and never in a sandbox. The staging is the one of
the commit's own pins, so it is published by the commit's own ``inference_suite``, run from the entry's source
(``python -m inference_suite publish``): it downloads the pinned CPython build and the client lock's wheels, verifies
both by sha256, fetches InferenceX at its pin, and publishes the staging by rename under its staging key
(:func:`inference_suite.staging_key`). A staging already published is verified and used as it is.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from hyperloom.measure.runner.tools import INFERENCE_SUITE_DIR

__all__ = ["SuitePublishError", "publish_inference_suite"]

_TAIL = 2000
#: Below an entry's source: the packages the commit's ``inference_suite`` imports from.
_SOURCE_PACKAGES = "src"


class SuitePublishError(Exception):
    """The commit's inference_suite staging could not be published; the message says why."""


def publish_inference_suite(source: Path, cache_root: Path) -> Path:
    """Publish the staging of the inference_suite in the commit source tree ``source`` below ``cache_root``; returns
    the published directory.

    Raises:
        SuitePublishError: The publish command failed (its ``staging_missing`` line is in the message).
    """
    argv = [sys.executable, "-m", "inference_suite", "publish", "--stagings", str(cache_root / INFERENCE_SUITE_DIR)]
    env = {**os.environ, "PYTHONPATH": str(source / _SOURCE_PACKAGES)}
    try:
        done = subprocess.run(argv, capture_output=True, text=True, check=False, env=env)
    except OSError as exc:
        raise SuitePublishError(f"publishing the inference_suite staging failed: {exc}") from exc
    if done.returncode != 0:
        raise SuitePublishError(
            f"publishing the inference_suite staging exited {done.returncode}: {done.stderr.strip()[-_TAIL:]}"
        )
    return Path(done.stdout.strip())
