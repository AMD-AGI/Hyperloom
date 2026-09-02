# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.orchestrator.kernel.controller_publication import (
    ControllerPublicationError,
    discover_controller_patch_dirs,
    load_controller_publication,
)


def _publication(root: Path, repo: Path, kernel_name: str = "kernel") -> Path:
    operator_id = f"kernel:forge-loop:{kernel_name}:standalone:unknown:triton:mi355x"
    patch_dir = root / operator_id
    patch_dir.mkdir(parents=True)
    (patch_dir / "change.patch").write_text("diff\n", encoding="utf-8")
    (patch_dir / "report.md").write_text("# Report\n", encoding="utf-8")
    (patch_dir / "publication.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "operator_id": operator_id,
                "identity": {
                    "producer": "forge-loop",
                    "kernel_name": kernel_name,
                    "framework": "standalone",
                    "framework_version": "unknown",
                    "backend": "triton",
                    "gpu": "mi355x",
                },
                "base_commit": "a" * 40,
                "best_commit": "b" * 40,
                "repo_root": str(repo),
                "kernel_path": "kernel.py",
                "operator_name": kernel_name,
                "micro_validated": True,
                "manifest": {},
            }
        ),
        encoding="utf-8",
    )
    return patch_dir


def test_discovery_and_parser_accept_complete_v2_publications(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    root = tmp_path / "patches"
    second = _publication(root, repo, "second")
    first = _publication(root, repo, "first")

    discovered = discover_controller_patch_dirs(root)
    parsed = [load_controller_publication(path) for path in discovered]

    assert discovered == tuple(sorted((first, second), key=lambda path: path.name))
    assert [item.identity["kernel_name"] for item in parsed] == ["first", "second"]
    assert all(item.repo_root == repo.resolve() for item in parsed)


def test_parser_rejects_old_or_unvalidated_publication(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    patch_dir = _publication(tmp_path / "patches", repo)
    metadata_path = patch_dir / "publication.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["schema_version"] = 1
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ControllerPublicationError, match="unsupported publication schema"):
        load_controller_publication(patch_dir)


def test_discovery_ignores_incomplete_and_hidden_versions(tmp_path: Path) -> None:
    root = tmp_path / "patches"
    root.mkdir()
    hidden = root / ".versions"
    hidden.mkdir()
    incomplete = root / "kernel:forge-loop:broken:standalone:unknown:triton:mi355x"
    incomplete.mkdir()
    (incomplete / "change.patch").write_text("diff\n", encoding="utf-8")

    assert discover_controller_patch_dirs(root) == ()
