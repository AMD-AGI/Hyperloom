# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Compile a real patch against the SGLang in the image and check that the overlay applies and imports.

Needs an image with SGLang installed (editable from its checkout, or from site-packages); on any other machine the
module is skipped, so a pass in a log means it ran against a real installation.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom.contracts.recipe import RecipeDelta
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    EnvironmentCompiler,
    LocalBlobStore,
    locate_installed_package,
)
from hyperloom.measure.tests._recipes import IMAGE, git_patch, put_patch, recipe_with

pytestmark = pytest.mark.skipif(importlib.util.find_spec("sglang") is None, reason="SGLang is not installed here")

MARKER = "HL_W5_OVERLAY_MARKER"
TARGET = "version.py"


def lay_over(base: Path, entries, blobs: LocalBlobStore, overlay: Path) -> None:
    """What a runner does: the base package as it is in the image, with the root's entries laid over it."""

    shutil.copytree(base, overlay, ignore=shutil.ignore_patterns("__pycache__"))
    for entry in entries:
        if entry.op == "delete":
            (overlay / entry.path).unlink()
        else:
            (overlay / entry.path).write_bytes(blobs.get(entry.blob))


def run_python(code: str, cwd: Path, pythonpath: Path | None = None) -> subprocess.CompletedProcess[str]:
    env = {"PATH": "/usr/bin:/bin", "HOME": str(cwd)}
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(
        [sys.executable, "-c", code], cwd=cwd, env=env, capture_output=True, text=True, timeout=600, check=False
    )


def test_a_small_patch_against_the_installed_sglang_applies_and_imports(tmp_path):
    layout = COMPONENT_LAYOUTS["sglang"]
    base = locate_installed_package(layout)
    original = (base.path / TARGET).read_text()
    patched = original + f'\n{MARKER} = "compiled-overlay"\n'
    prefix = layout.upstream_prefix
    blobs = LocalBlobStore(tmp_path / "blobs")
    patch = git_patch(tmp_path / "repo", {f"{prefix}/{TARGET}": original}, {f"{prefix}/{TARGET}": patched})
    recipe = recipe_with((put_patch(blobs, "marker.patch", patch),))

    environment = EnvironmentCompiler(blobs, IMAGE).compile(recipe, RecipeDelta())

    (root,) = environment.roots
    assert [(e.path, e.op) for e in root.entries] == [(TARGET, "upsert")]
    assert (base.path / TARGET).read_text() == original, "compiling must not touch the image"
    overlay = tmp_path / "overlay"
    lay_over(base.path, root.entries, blobs, overlay / "sglang")

    without = run_python(f"import sglang.version as v; print(hasattr(v, {MARKER!r}))", tmp_path)
    with_overlay = run_python(
        f"import sglang, sglang.version as v, sglang.srt.server_args; print(sglang.__file__); print(v.{MARKER})",
        tmp_path,
        pythonpath=overlay,
    )
    assert without.stdout.strip() == "False", without.stderr
    assert with_overlay.returncode == 0, with_overlay.stderr
    assert with_overlay.stdout.split() == [str(overlay / "sglang" / "__init__.py"), "compiled-overlay"]
