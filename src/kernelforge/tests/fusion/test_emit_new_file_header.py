# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The non-git export must mark an authored module as a file creation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from kernelforge.fusion.emit import _export_nongit, _unified_file_diff


_APPLY_PATH = (
    Path(__file__).resolve().parents[3] / "hyperloom" / "agents" / "kernel" / "tools" / "apply_kernel_patch.py"
)
_SPEC = importlib.util.spec_from_file_location("apply_kernel_patch_tool", _APPLY_PATH)
assert _SPEC and _SPEC.loader
apply_kernel_patch = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(apply_kernel_patch)


def test_unified_file_diff_marks_an_absent_base_as_a_creation():
    out = _unified_file_diff("m/foo_fused.py", "", "x = 1\n", created=True)

    assert out == (
        "diff --git a/m/foo_fused.py b/m/foo_fused.py\n"
        "new file mode 100644\n"
        "--- /dev/null\n"
        "+++ b/m/foo_fused.py\n"
        "@@ -0,0 +1 @@\n"
        "+x = 1\n"
    )


def test_an_existing_empty_base_is_not_a_creation():
    """``git apply`` refuses a creation whose path already exists on disk."""
    out = _unified_file_diff("m/empty_init.py", "", "x = 1\n", created=False)

    assert out == (
        "diff --git a/m/empty_init.py b/m/empty_init.py\n"
        "--- a/m/empty_init.py\n"
        "+++ b/m/empty_init.py\n"
        "@@ -0,0 +1 @@\n"
        "+x = 1\n"
    )


def test_unified_file_diff_leaves_a_modification_unchanged():
    out = _unified_file_diff("m/model.py", "a = 1\n", "a = 2\n", created=False)

    assert out == (
        "diff --git a/m/model.py b/m/model.py\n--- a/m/model.py\n+++ b/m/model.py\n@@ -1 +1 @@\n-a = 1\n+a = 2\n"
    )


def test_nongit_export_is_read_back_as_one_create_plus_one_edit(tmp_path):
    """The apply side keys creation off the headers, so the export must set them."""
    repo = tmp_path / "fw"
    repo.mkdir()
    (repo / "model.py").write_text("import torch\nx = 1\n", encoding="utf-8")
    (repo / "model_fused.py").write_text("def fused():\n    return 2\n", encoding="utf-8")
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "model.py").write_text("x = 1\n", encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()

    arts = _export_nongit(
        str(repo),
        str(repo / "model.py"),
        out,
        snap,
        fused_module=str(repo / "model_fused.py"),
    )

    descriptors = apply_kernel_patch.parse_patch_manifest(Path(arts.patch).read_text(encoding="utf-8"))
    assert descriptors == [
        {"op": "write", "path": "model.py", "mode": "", "binary": False, "is_new": False},
        {"op": "write", "path": "model_fused.py", "mode": "0644", "binary": False, "is_new": True},
    ]
