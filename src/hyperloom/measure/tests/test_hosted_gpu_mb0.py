# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""MB0 on a GPU machine: one hosted test in a fresh local sandbox, with a real framework patch, from the recipe.

Needs an image with SGLang installed, a model directory and a git checkout of InferenceX; on any other machine the
module is skipped, so a pass in a log means a real server ran. Set ``HL_MB0_MODEL`` (the model directory) and
``HL_MB0_INFERENCEX`` (a git repository of InferenceX whose HEAD is the commit to measure with); the test uses card 0.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
import yaml

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    EnvironmentCompiler,
    LocalBlobStore,
    locate_installed_package,
)
from hyperloom.measure.hosted_log import STATE_FINISHED
from hyperloom.measure.local_templates import local_runner_template
from hyperloom.measure.tests._recipes import IMAGE, git_patch, put_patch, recipe_with
from hyperloom.measure.tests.hosted_fixtures import make_call, make_rig

MODEL = os.environ.get("HL_MB0_MODEL", "")
INFERENCEX = os.environ.get("HL_MB0_INFERENCEX", "")
pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("sglang") is None or not Path(MODEL).is_dir() or not Path(INFERENCEX, ".git").exists(),
    reason="needs SGLang, HL_MB0_MODEL and HL_MB0_INFERENCEX (a git checkout)",
)

TARGET = "launch_server.py"
MARKER_ENV = "HL_MB0_MARKER"
MARKER_CODE = (
    "import os as _hl_os\n"
    f"_hl_marker = _hl_os.environ.get({MARKER_ENV!r})\n"
    "if _hl_marker:\n"
    "    with open(_hl_marker, 'a', encoding='utf-8') as _hl_f:\n"
    "        _hl_f.write(__file__ + '\\n')\n"
    "del _hl_os, _hl_marker\n"
)
SERVER_READY_SEC = "900"


def _point(path: Path, marker: Path) -> Path:
    envs = {
        "TP": 1,
        "CONC": 8,
        "ISL": 256,
        "OSL": 64,
        "NUM_PROMPTS": 32,
        "NUM_WARMUPS": 4,
        "RUN_EVAL": "false",
        "EXTRA_SGLANG_ARGS": "--mem-fraction-static 0.7",
        MARKER_ENV: str(marker),
    }
    benchmark = {"framework": "sglang", "model": MODEL, "precision": "bf16", "run_mode": "local", "envs": envs}
    path.write_text(yaml.safe_dump({"benchmark": benchmark}), encoding="utf-8")
    return path


def _compile_with_marker(blobs: LocalBlobStore, workdir: Path):
    """A recipe whose patch puts the marker code at the top of the image's ``launch_server.py``."""
    layout = COMPONENT_LAYOUTS["sglang"]
    original = (locate_installed_package(layout).path / TARGET).read_text()
    key = f"{layout.upstream_prefix}/{TARGET}"
    patch = git_patch(workdir, {key: original}, {key: MARKER_CODE + original})
    recipe = recipe_with((put_patch(blobs, "mb0-marker.patch", patch),))
    return EnvironmentCompiler(blobs, IMAGE).compile(recipe, RecipeDelta())


def test_a_patched_recipe_is_rebuilt_measured_signed_verified_and_archived(tmp_path: Path) -> None:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=INFERENCEX, capture_output=True, text=True, check=True
    ).stdout.strip()
    template = local_runner_template(
        gpus=1,
        envs={"INFERENCEX_REPO": INFERENCEX, "INFERENCE_OPTIMIZER_BASELINE_SERVER_READY_SEC": SERVER_READY_SEC},
    )
    rig = make_rig(
        tmp_path,
        templates=[template],
        gpus=1,
        spec={"inferencex_commit": commit, "benchmark_backend": "bypass", "image_digest": IMAGE, "gpu_type": "MI355X"},
        poll_interval_sec=1.0,
        runner_start_timeout_sec=120,
    )
    try:
        environment = _compile_with_marker(rig.config.spec.blobs, tmp_path / "patch-repo")
        (root,) = environment.roots
        expected = {e.path: hashlib.sha256(rig.config.spec.blobs.get(e.blob)).hexdigest() for e in root.entries}
        marker = tmp_path / "marker.txt"
        call = make_call(
            tmp_path, "mb0-1", point=_point(tmp_path / "point.yaml", marker), timeout_sec=1800, environment=environment
        )
        outcome = rig.backend.run_test(call)
        assert outcome.returncode == 0, outcome.stderr
        record = rig.backend.journal.latest("mb0-1")
        assert record.state == STATE_FINISHED and record.extra["verified"] is True
        assert record.extra["verdict"] == "accepted"
        archived = rig.archive.bundle_dir("mb0-1", record.extra["manifest_sha256"])
        assert record.extra["archive"] == str(archived)
        assert (archived / "manifest.json").is_file() and (archived / "manifest.sig").is_file()
        # The server imported the rebuilt file, not the image's.
        imported = set(marker.read_text().split())
        assert len(imported) == 1
        (farm_file,) = imported
        assert "/farm/site/sglang/" + TARGET in farm_file
        assert record.sandbox_id in farm_file, "the file the server imported is in the test's own sandbox"
        inventory = json.loads(next(call.output_dir.rglob("farm_inventory.json")).read_text())
        assert inventory["roots"][root.id]["upserts"] == expected
    finally:
        rig.backend.close()
        rig.provider.close()
