# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Publish the inference_suite staging a runner of one commit runs, into ``<runner cache>/inference-suite/``.

It runs where the runner cache is built, on a host with network, after the commit's entry is in place
(:mod:`hyperloom.measure.runner.cache_build`): never at import time, and never in a sandbox. The staging is the one of
the commit's own pins, so it is published by the commit's own ``inference_suite``, run from the entry's source
(``python -m inference_suite publish``): it downloads the pinned CPython build and the client lock's wheels, verifies
both by sha256, fetches InferenceX at its pin, and publishes the staging by rename under its staging key
(:func:`inference_suite.staging_key`). A staging already published is verified and used as it is. Each publish then
fills the runner cache's InferenceX store with the InferenceX commit the published staging records
(:func:`~hyperloom.measure.runner.tools.fill_inferencex_store`), so that a test that checks that commit out fetches it
from the store, reading only, with no network. A test checks out the commit its sheet names, which is not compared
with the staging's: one whose commit the store lacks fetches it from ``INFERENCEX_REPO`` or upstream, with a warning. A
fill that fails is logged as a warning and the publish goes on, since the tests then fetch as they would without the
store; the staging holds InferenceX's files but no repository, so the commit cannot be taken from it. A session whose
runners start from this installation publishes the staging of the ``inference_suite`` this interpreter imports
(:func:`publish_own_inference_suite`).
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

import inference_suite
from hyperloom.measure.runner.tools import INFERENCE_SUITE_DIR, ToolError, fill_inferencex_store

__all__ = ["SuitePublishError", "publish_inference_suite", "publish_own_inference_suite"]

log = logging.getLogger(__name__)

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
    return _publish(source / _SOURCE_PACKAGES, cache_root)


def publish_own_inference_suite(cache_root: Path) -> Path:
    """Publish the staging of the inference_suite this interpreter imports below ``cache_root``; returns the published
    directory.

    Raises:
        SuitePublishError: The publish command failed (its ``staging_missing`` line is in the message).
    """
    return _publish(Path(inference_suite.__file__).resolve().parents[1], cache_root)


def _publish(packages: Path, cache_root: Path) -> Path:
    argv = [sys.executable, "-m", "inference_suite", "publish", "--stagings", str(cache_root / INFERENCE_SUITE_DIR)]
    # The entry is checked file by file on reuse, so running its source must not write bytecode into it.
    env = {**os.environ, "PYTHONPATH": str(packages), "PYTHONDONTWRITEBYTECODE": "1"}
    try:
        done = subprocess.run(argv, capture_output=True, text=True, check=False, env=env)
    except OSError as exc:
        raise SuitePublishError(f"publishing the inference_suite staging failed: {exc}") from exc
    if done.returncode != 0:
        raise SuitePublishError(
            f"publishing the inference_suite staging exited {done.returncode}: {done.stderr.strip()[-_TAIL:]}"
        )
    published = Path(done.stdout.strip())
    try:
        commit = inference_suite.staged_inferencex_commit(published)
    except inference_suite.StagingError as exc:
        raise SuitePublishError(f"the staging published at {published} cannot be read: {exc}") from exc
    if commit is not None:
        try:
            fill_inferencex_store(cache_root, commit)
        except ToolError as exc:
            log.warning(
                "the runner cache's InferenceX store was not filled, so each test fetches InferenceX from "
                "INFERENCEX_REPO or upstream: %s",
                exc,
            )
    return published
