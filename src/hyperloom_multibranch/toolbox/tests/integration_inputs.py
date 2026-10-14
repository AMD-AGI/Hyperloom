# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""What both variants of the integration test start from: the baseline recipe, the changes the specialists try, and the
process environment the canonical measurement point is rendered from."""

from __future__ import annotations

from pathlib import Path

import pytest

from hyperloom.contracts.recipe import ComponentChange, Recipe, RecipeDelta
from hyperloom.measure.compile import COMPONENT_LAYOUTS, LocalBlobStore
from hyperloom.measure.tests._recipes import IMAGE, git_patch, put_patch, recipe_with
from hyperloom_multibranch.toolbox.tests.integration_round import Probe, RoundProbes, config_delta

PATCHED_FILE = "launch_server.py"
_WORKLOAD = {"FRAMEWORK": "sglang", "ISL": "256", "OSL": "64", "INFERENCE_OPTIMIZER_DISABLE_TP_CLAMP": "1"}
_CLEARED = ("CONC", "RUN_EVAL", "TP", "EP", "PORT", "MAX_MODEL_LEN", "PRECISION", "RANDOM_RANGE_RATIO")
__all__ = ["IMAGE", "PATCHED_FILE", "baseline_recipe", "probes", "set_workload_environment"]


def baseline_recipe() -> Recipe:
    """The baseline: the active image and an unpatched SGLang."""
    return recipe_with()


def set_workload_environment(monkeypatch: pytest.MonkeyPatch, data_dir: Path, **extra: str) -> None:
    """The process environment the canonical point and the request points are rendered from."""
    for name in _CLEARED:
        monkeypatch.delenv(name, raising=False)
    for name, value in {**_WORKLOAD, "USER_DATA_PATH": str(data_dir), **extra}.items():
        monkeypatch.setenv(name, value)


def comment_patch_delta(workdir: Path, blobs: LocalBlobStore, original: str) -> RecipeDelta:
    """A small patch to the framework: one comment line on top of ``PATCHED_FILE``; it changes nothing it runs."""
    key = f"{COMPONENT_LAYOUTS['sglang'].upstream_prefix}/{PATCHED_FILE}"
    diff = git_patch(workdir, {key: original}, {key: "# hyperloom integration test\n" + original})
    return RecipeDelta(
        components=(ComponentChange("sglang", add_patches=(put_patch(blobs, "w15-comment.patch", diff),)),)
    )


def probes(patch: RecipeDelta) -> RoundProbes:
    """s1: a framework patch and a second change; s2: a config change and one queued behind it; s3: two it never runs."""
    return RoundProbes(
        patch=Probe("launch", "comment-only source patch", "source", patch),
        config=Probe("scheduler", "cap running requests at 256", "config", config_delta(256)),
        explore2=Probe("scheduler", "cap running requests at 128", "config", config_delta(128)),
        queued=Probe("scheduler", "cap running requests at 192", "config", config_delta(192)),
        never=(
            Probe("scheduler", "cap running requests at 96", "config", config_delta(96)),
            Probe("scheduler", "cap running requests at 64", "config", config_delta(64)),
        ),
    )
