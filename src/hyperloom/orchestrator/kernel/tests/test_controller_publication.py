# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import json
from pathlib import Path

import pytest

from kernelforge.kernel_rewrite_controller.paths import operator_directory_name
from hyperloom.orchestrator.kernel.controller_publication import (
    ControllerPublicationError,
    discover_controller_patch_dirs,
    load_controller_publication,
)


def _publication(root: Path, repo: Path, kernel_name: str = "kernel") -> Path:
    operator_id = f"kernel:forge-loop:{kernel_name}:standalone:unknown:triton:mi355x"
    patch_dir = root / operator_directory_name(operator_id)
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


def test_parser_carries_the_declared_patch_scope(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    patch_dir = _publication(tmp_path / "patches", repo)
    metadata_path = patch_dir / "publication.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["changed_files"] = ["kernel.py", "helper/util.py"]
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    assert load_controller_publication(patch_dir).changed_files == ("kernel.py", "helper/util.py")


def test_parser_rejects_a_patch_scope_that_escapes_the_repository(tmp_path: Path) -> None:
    """The scope decides what integration stages, so it cannot name an outside path."""
    repo = tmp_path / "repo"
    repo.mkdir()
    patch_dir = _publication(tmp_path / "patches", repo)
    metadata_path = patch_dir / "publication.json"
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    payload["changed_files"] = ["../../etc/profile"]
    metadata_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ControllerPublicationError, match="changed_files entry"):
        load_controller_publication(patch_dir)


def test_a_publication_without_a_declared_scope_is_still_accepted(tmp_path: Path) -> None:
    """The field is additive within schema 2; an older bundle simply has none."""
    repo = tmp_path / "repo"
    repo.mkdir()
    patch_dir = _publication(tmp_path / "patches", repo)

    assert load_controller_publication(patch_dir).changed_files == ()


def test_discovery_ignores_incomplete_and_hidden_versions(tmp_path: Path) -> None:
    root = tmp_path / "patches"
    root.mkdir()
    hidden = root / ".versions"
    hidden.mkdir()
    incomplete = root / "kernel:forge-loop:broken:standalone:unknown:triton:mi355x"
    incomplete.mkdir()
    (incomplete / "change.patch").write_text("diff\n", encoding="utf-8")

    assert discover_controller_patch_dirs(root) == ()


def _mutate(patch_dir: Path, **changes: object) -> None:
    """Rewrite the publication payload with ``changes`` applied.

    A key mapped to ``_DROP`` is removed rather than overwritten.
    """
    path = patch_dir / "publication.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key, value in changes.items():
        if value is _DROP:
            payload.pop(key, None)
        else:
            payload[key] = value
    path.write_text(json.dumps(payload), encoding="utf-8")


_DROP = object()


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"micro_validated": False}, "not micro-validated"),
        ({"identity": {"producer": "forge-loop"}}, "six canonical dimensions"),
        ({"base_commit": "abc"}, "full hexadecimal object ids"),
        ({"best_commit": ""}, "full hexadecimal object ids"),
        ({"repo_root": "relative/path"}, "repo_root must be an absolute path"),
        ({"operator_name": "   "}, "operator_name must be a non-empty string"),
        ({"manifest": []}, "manifest must be a JSON object"),
        ({"changed_files": "kernel.py"}, "changed_files must be a JSON list"),
        ({"operator_id": "kernel:forge-loop:other"}, "does not match identity"),
    ],
)
def test_every_publication_rule_refuses_its_own_violation(
    tmp_path: Path,
    changes: dict[str, object],
    expected: str,
) -> None:
    """Each rule is load-bearing: integration stages a patch on the operator's repo."""
    patch_dir = _publication(tmp_path / "patches", tmp_path)
    _mutate(patch_dir, **changes)
    with pytest.raises(ControllerPublicationError) as error:
        load_controller_publication(patch_dir)
    assert expected in str(error.value)


def test_a_blank_identity_dimension_is_refused(tmp_path: Path) -> None:
    """An empty string passes the key-set check and would still name nothing."""
    patch_dir = _publication(tmp_path / "patches", tmp_path)
    payload = json.loads((patch_dir / "publication.json").read_text(encoding="utf-8"))
    payload["identity"]["backend"] = ""
    (patch_dir / "publication.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ControllerPublicationError, match="identity.backend"):
        load_controller_publication(patch_dir)


def test_a_missing_repo_root_directory_is_refused(tmp_path: Path) -> None:
    """An absolute path that no longer exists cannot be staged into."""
    patch_dir = _publication(tmp_path / "patches", tmp_path)
    _mutate(patch_dir, repo_root=str(tmp_path / "gone"))
    with pytest.raises(ControllerPublicationError, match="repo_root does not exist"):
        load_controller_publication(patch_dir)


def test_unreadable_publication_metadata_is_refused(tmp_path: Path) -> None:
    patch_dir = _publication(tmp_path / "patches", tmp_path)
    (patch_dir / "publication.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ControllerPublicationError, match="could not read publication metadata"):
        load_controller_publication(patch_dir)


def test_a_non_object_publication_is_refused(tmp_path: Path) -> None:
    patch_dir = _publication(tmp_path / "patches", tmp_path)
    (patch_dir / "publication.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ControllerPublicationError, match="must be a JSON object"):
        load_controller_publication(patch_dir)


def test_a_missing_patch_directory_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ControllerPublicationError, match="patch directory does not exist"):
        load_controller_publication(tmp_path / "absent")


def test_a_symlinked_artifact_is_refused(tmp_path: Path) -> None:
    """A symlink can point outside the published version directory."""
    patch_dir = _publication(tmp_path / "patches", tmp_path)
    real = tmp_path / "elsewhere.patch"
    real.write_text("diff\n", encoding="utf-8")
    (patch_dir / "change.patch").unlink()
    (patch_dir / "change.patch").symlink_to(real)
    with pytest.raises(ControllerPublicationError, match="must be a regular file"):
        load_controller_publication(patch_dir)
