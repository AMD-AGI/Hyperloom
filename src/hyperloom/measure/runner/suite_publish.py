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
(:func:`~hyperloom.measure.runner.tools.fill_inferencex_store`): a runner refuses a sheet whose InferenceX commit is not
its staging's, so that is the commit every test of the staging checks out, and it fetches it from the store, reading
only, with no network. A session whose runners start from this installation publishes the staging of the ``inference_suite`` this interpreter imports
(:func:`publish_own_inference_suite`).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import inference_suite
from hyperloom.measure.runner.tools import INFERENCE_SUITE_DIR, ToolError, fill_inferencex_store
from inference_suite.pins import INFERENCEX_TOOL
from inference_suite.staging import MANIFEST_FILE, StagingError, StagingManifest

__all__ = ["SuitePublishError", "publish_inference_suite", "publish_own_inference_suite"]

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
    commit = _staged_inferencex(published)
    if commit is not None:
        try:
            fill_inferencex_store(cache_root, commit)
        except ToolError as exc:
            raise SuitePublishError(f"the runner cache's InferenceX store was not filled: {exc}") from exc
    return published


def _staged_inferencex(published: Path) -> str | None:
    """The InferenceX commit the staging published at ``published`` records; ``None`` for a staging without a manifest
    or without InferenceX, which no runner runs (it verifies the manifest first)."""
    try:
        manifest = StagingManifest.from_document(json.loads((published / MANIFEST_FILE).read_bytes()))
    except FileNotFoundError:
        return None
    except (OSError, ValueError, StagingError) as exc:
        raise SuitePublishError(f"the staging published at {published} cannot be read: {exc}") from exc
    tool = manifest.tools.get(INFERENCEX_TOOL)
    return tool.commit if tool is not None else None
