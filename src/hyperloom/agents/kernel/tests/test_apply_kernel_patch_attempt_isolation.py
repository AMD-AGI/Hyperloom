# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Backup isolation between two applies that share a kernel id and a target.

The fusion lane passes a constant ``kernel_id`` and an integrate request may
carry none at all, so neither the id nor the target path separates one attempt
from the next. Without a per-attempt directory the second apply overwrites the
first's pristine copy and its manifest, and a later revert restores the patch it
was supposed to undo while reporting success.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

apk = importlib.import_module("hyperloom.agents.kernel.tools.apply_kernel_patch")

_BODY = "import torch\n\n\ndef fused_moe(x):\n    return x\n"


def _apply(tmp_path: Path, target: Path, marker: str, kernel_id: str) -> dict:
    patch = tmp_path / f"{marker}.py"
    patch.write_text(f"# {marker}\n{_BODY}", encoding="utf-8")
    return apk.apply_kernel_patch(
        patch_path=patch,
        target_file=target,
        backup_root=tmp_path / "backup",
        kernel_id=kernel_id,
        skip_rebuild=True,
        allow_unknown_target=True,
    )


def _first_line(path: Path) -> str:
    return path.read_text(encoding="utf-8").splitlines()[0]


def test_two_applies_on_one_target_keep_separate_backups(tmp_path: Path) -> None:
    target = tmp_path / "fused_moe.py"
    target.write_text(f"# PRISTINE\n{_BODY}", encoding="utf-8")

    first = _apply(tmp_path, target, "PATCH_A", "forge_fusion")
    second = _apply(tmp_path, target, "PATCH_B", "forge_fusion")

    assert first["manifest_path"] != second["manifest_path"]
    kept = json.loads(Path(first["manifest_path"]).read_text(encoding="utf-8"))
    assert _first_line(Path(kept["source_backup"]["backup_path"])) == "# PRISTINE"


def test_reverting_both_applies_restores_the_original(tmp_path: Path) -> None:
    target = tmp_path / "fused_moe.py"
    target.write_text(f"# PRISTINE\n{_BODY}", encoding="utf-8")

    first = _apply(tmp_path, target, "PATCH_A", "forge_fusion")
    second = _apply(tmp_path, target, "PATCH_B", "forge_fusion")

    apk.revert_kernel_patch(second["manifest_path"])
    apk.revert_kernel_patch(first["manifest_path"])
    assert _first_line(target) == "# PRISTINE"


def test_a_blank_kernel_id_still_separates_attempts(tmp_path: Path) -> None:
    target = tmp_path / "fused_moe.py"
    target.write_text(f"# PRISTINE\n{_BODY}", encoding="utf-8")

    first = _apply(tmp_path, target, "PATCH_A", "")
    second = _apply(tmp_path, target, "PATCH_B", "")

    assert first["manifest_path"] != second["manifest_path"]
    apk.revert_kernel_patch(second["manifest_path"])
    apk.revert_kernel_patch(first["manifest_path"])
    assert _first_line(target) == "# PRISTINE"


def test_a_clean_revert_drops_the_payload_and_keeps_the_manifest(tmp_path: Path) -> None:
    target = tmp_path / "fused_moe.py"
    target.write_text(f"# PRISTINE\n{_BODY}", encoding="utf-8")

    applied = _apply(tmp_path, target, "PATCH_A", "k001")
    backup_dir = Path(applied["manifest_path"]).parent
    assert any(c.name != "manifest.json" for c in backup_dir.iterdir())

    apk.revert_kernel_patch(applied["manifest_path"])
    assert [c.name for c in backup_dir.iterdir()] == ["manifest.json"]
    status = json.loads((backup_dir / "manifest.json").read_text(encoding="utf-8"))["status"]
    assert status == "reverted"
