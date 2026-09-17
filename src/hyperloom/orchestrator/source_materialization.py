# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Export verified accepted Python source trees for a GEAK baseline."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from hyperloom.common.git_safety import safe_directory_args

SCHEMA_VERSION = 1
_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?\Z")
_NATIVE_SUFFIXES = {".so", ".pyd", ".dll", ".dylib", ".a", ".o", ".co", ".hsaco"}


class SourceMaterializationError(ValueError):
    """The required accepted source cannot be reconstructed from its evidence."""

    error_class = "unresolved_baseline_source"

    def __init__(self, reason: str, detail: str) -> None:
        self.reason = reason
        super().__init__(f"{reason}: {detail}")


def _relative(value: Any, *, allow_empty: bool = False) -> str:
    text = str(value)
    if allow_empty and text == "":
        return text
    if not text or any(c in text for c in "\\:\0\n\r") or any(part in ("", ".", "..") for part in text.split("/")):
        raise SourceMaterializationError("invalid_path", repr(text))
    return text


def _regular(path: Path, root: Path) -> bytes:
    current = path
    while current != root:
        if current.is_symlink():
            raise SourceMaterializationError("unsupported_symlink", str(current))
        current = current.parent
    if root.is_symlink() or not path.is_file():
        raise SourceMaterializationError("missing_payload", str(path))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise SourceMaterializationError("missing_payload", str(path))
        return stream.read()


def _git(root: Path, *args: str, input_data: bytes | None = None) -> bytes:
    result = subprocess.run(
        ["git", *safe_directory_args(["-C", str(root), *args])],
        capture_output=True,
        input=input_data,
        timeout=120,
        check=False,
    )
    if result.returncode:
        raise SourceMaterializationError("git_evidence_unavailable", f"{root}: {args[0]}")
    return result.stdout


def _tree(root: Path, commit: str) -> dict[str, tuple[str, str]]:
    entries = {}
    for record in _git(root, "ls-tree", "-rz", "--full-tree", commit).split(b"\0"):
        if not record:
            continue
        header, raw_path = record.split(b"\t", 1)
        mode, kind, object_id = header.decode().split()
        path = _relative(raw_path.decode())
        if kind != "blob" or mode not in ("100644", "100755"):
            raise SourceMaterializationError("unsupported_tree_entry", path)
        entries[path] = (mode, object_id)
    return entries


def _module(path: str, prefix: str, tree: Mapping[str, Any], *, deleted: bool = False) -> str:
    try:
        relative = PurePosixPath(path).relative_to(prefix or ".")
    except ValueError as exc:
        raise SourceMaterializationError("path_outside_import_root", path) from exc
    if relative.suffix != ".py":
        raise SourceMaterializationError("unsupported_runtime_change", path)
    parts = list(relative.with_suffix("").parts)
    if not all(part.isidentifier() for part in parts):
        raise SourceMaterializationError("unsupported_module_name", path)
    if len(parts) == 1:
        raise SourceMaterializationError("unsupported_top_level_module", path)
    if deleted and (len(parts) == 1 or parts[-1] == "__init__"):
        raise SourceMaterializationError("unsupported_package_deletion", path)
    for count in range(1, len(parts)):
        init = PurePosixPath(prefix, *parts[:count], "__init__.py").as_posix()
        if init not in tree:
            raise SourceMaterializationError("unsupported_namespace_package", path)
    if parts[-1] == "__init__":
        parts.pop()
    if not parts:
        raise SourceMaterializationError("unresolved_import_root", path)
    return ".".join(parts)


def _read_layer(entry: Mapping[str, Any]) -> dict[str, Any]:
    layer_id = str(entry.get("variant_name") or entry.get("name") or "")
    if not layer_id.strip() or any(ord(char) < 32 for char in layer_id):
        raise SourceMaterializationError("missing_layer_id", "source layers require stable IDs")
    snapshot = Path(str(entry.get("source_snapshot") or ""))
    if not entry.get("source_snapshot"):
        raise SourceMaterializationError("missing_snapshot", layer_id)
    manifest = json.loads(_regular(snapshot / "manifest.json", snapshot))
    if (
        not isinstance(manifest, dict)
        or type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] not in (1, 2)
    ):
        raise SourceMaterializationError("unsupported_snapshot_schema", layer_id)
    extra = manifest.get("extra") or {}
    if not isinstance(extra, dict):
        raise SourceMaterializationError("invalid_snapshot", layer_id)
    outside = extra.get("artifacts_outside_root", entry.get("source_artifacts_outside_root"))
    if type(outside) is not int or outside != 0:
        raise SourceMaterializationError("unverified_artifact_coverage", layer_id)
    if "source_artifacts_outside_root" in entry and (
        type(entry["source_artifacts_outside_root"]) is not int or entry["source_artifacts_outside_root"] != outside
    ):
        raise SourceMaterializationError("unverified_artifact_coverage", layer_id)
    if manifest.get("provenance") != "integrate_patch":
        raise SourceMaterializationError("unverified_commit_semantics", layer_id)
    if extra.get("commit_semantics", "accepted") != "accepted":
        raise SourceMaterializationError("unverified_commit_semantics", layer_id)
    if (
        ("source_snapshot_complete" in entry and entry["source_snapshot_complete"] is not True)
        or ("complete" in manifest and manifest["complete"] is not True)
        or (manifest["schema_version"] == 2 and "complete" not in manifest)
    ):
        raise SourceMaterializationError("incomplete_snapshot", layer_id)
    root_text = str(entry.get("framework_root") or "")
    commit = str(entry.get("base_sha") or "")
    prefix = _relative(entry.get("source_import_root") or "", allow_empty=True)
    if not root_text or manifest.get("framework_root") != root_text or manifest.get("base_sha") != commit:
        raise SourceMaterializationError("snapshot_identity_mismatch", layer_id)
    if manifest.get("import_root", "") != prefix or not _COMMIT.fullmatch(commit):
        raise SourceMaterializationError("snapshot_identity_mismatch", layer_id)
    root = Path(root_text).resolve()
    if _git(root, "rev-parse", "--show-toplevel").decode().strip() != str(root):
        raise SourceMaterializationError("unsupported_non_git_tree", root_text)
    if _git(root, "rev-parse", f"{commit}^{{commit}}").decode().strip() != commit:
        raise SourceMaterializationError("commit_identity_mismatch", layer_id)
    tree = _tree(root, commit)
    rows = manifest.get("files")
    if not isinstance(rows, list) or not rows:
        raise SourceMaterializationError("empty_snapshot", layer_id)
    operations = {}
    for row in rows:
        if not isinstance(row, dict):
            raise SourceMaterializationError("invalid_snapshot", layer_id)
        path = _relative(row.get("rel") or "")
        op = row.get("op")
        if path in operations or op not in ("upsert", "delete"):
            raise SourceMaterializationError("invalid_snapshot_operation", path)
        _module(path, prefix, tree, deleted=op == "delete")
        if op == "delete":
            if path in tree:
                raise SourceMaterializationError("deletion_commit_mismatch", path)
        else:
            content = _regular(snapshot / "files" / path, snapshot)
            if path not in tree or content != _git(root, "cat-file", "blob", tree[path][1]):
                raise SourceMaterializationError("payload_commit_mismatch", path)
        operations[path] = op
    targets = entry.get("target_files")
    if targets is not None and (
        not isinstance(targets, list) or len(targets) != len(operations) or set(targets) != set(operations)
    ):
        raise SourceMaterializationError("layer_coverage_mismatch", layer_id)
    return {"id": layer_id, "root": root, "commit": commit, "prefix": prefix, "tree": tree, "ops": operations}


def _export(root: Path, tree: Mapping[str, tuple[str, str]], destination: Path) -> None:
    # git archive honors export-ignore/export-subst; blobs preserve the commit's bytes.
    queries = "".join(f"{object_id}\n" for _, object_id in tree.values()).encode()
    blobs = io.BytesIO(_git(root, "cat-file", "--batch", input_data=queries))
    for relative, (mode, object_id) in tree.items():
        header = blobs.readline().decode().split()
        if len(header) != 3 or header[:2] != [object_id, "blob"]:
            raise SourceMaterializationError("missing_git_payload", relative)
        size = int(header[2])
        content = blobs.read(size)
        if len(content) != size or blobs.read(1) != b"\n":
            raise SourceMaterializationError("missing_git_payload", relative)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        target.chmod(int(mode[-3:], 8))


def _native(path: str) -> bool:
    return bool({suffix.lower() for suffix in PurePosixPath(path).suffixes} & _NATIVE_SUFFIXES)


def _check_untracked_runtime(root: Path, group: list[dict[str, Any]]) -> None:
    prefixes = sorted({layer["prefix"] or "." for layer in group})
    for flags in (("--others",), ("--others", "--ignored")):
        for raw in _git(root, "ls-files", *flags, "--exclude-standard", "-z", "--", *prefixes).split(b"\0"):
            if raw:
                path = PurePosixPath(raw.decode())
                if path.parent.name != "__pycache__" or path.suffix != ".pyc":
                    raise SourceMaterializationError("unsupported_untracked_runtime", str(path))


def _assemble(layers: list[dict[str, Any]], staging: Path) -> dict[str, Any]:
    groups: dict[Path, list[dict[str, Any]]] = {}
    for layer in layers:
        groups.setdefault(layer["root"], []).append(layer)
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "required_layer_ids": [layer["id"] for layer in layers],
        "pythonpath_prefixes": [],
        "trees": [],
        "files": [],
        "deleted_paths": [],
        "modules": [],
        "deleted_modules": [],
    }
    for index, (root, group) in enumerate(groups.items()):
        latest = group[-1]
        for earlier, later in zip(group, group[1:]):
            _git(root, "merge-base", "--is-ancestor", earlier["commit"], later["commit"])
            changed = {
                path
                for path in earlier["tree"].keys() | later["tree"].keys()
                if earlier["tree"].get(path) != later["tree"].get(path)
            }
            if not changed.issubset(later["ops"]):
                raise SourceMaterializationError("unrecorded_source_changes", later["id"])
        _check_untracked_runtime(root, group)
        root_rel = f"trees/{index:03d}"
        _export(root, latest["tree"], staging / root_rel)
        manifest["trees"].append(
            {
                "tree_id": f"tree-{index:03d}",
                "root": root_rel,
                "accepted_commit": latest["commit"],
                "layer_ids": [layer["id"] for layer in group],
            }
        )
        prefixes = list(dict.fromkeys(layer["prefix"] for layer in group))
        for prefix in prefixes:
            output_prefix = str(PurePosixPath(root_rel, prefix))
            if not (staging / output_prefix).is_dir():
                raise SourceMaterializationError("missing_import_root", prefix)
            manifest["pythonpath_prefixes"].append(output_prefix)
        effective: dict[str, tuple[str, str]] = {}
        for layer in group:
            effective.update({path: (op, layer["prefix"]) for path, op in layer["ops"].items()})
        for path, (op, prefix) in sorted(effective.items()):
            relative = f"{root_rel}/{path}"
            name = _module(path, prefix, latest["tree"], deleted=op == "delete")
            if op == "delete":
                if (staging / relative).exists():
                    raise SourceMaterializationError("deletion_commit_mismatch", path)
                manifest["deleted_paths"].append(relative)
                manifest["deleted_modules"].append(name)
            else:
                if path not in latest["tree"]:
                    raise SourceMaterializationError("layer_coverage_mismatch", path)
                manifest["modules"].append({"name": name, "path": relative})
    for exported in sorted((staging / "trees").rglob("*")):
        if exported.is_file():
            manifest["files"].append(
                {
                    "path": exported.relative_to(staging).as_posix(),
                    "sha256": hashlib.sha256(exported.read_bytes()).hexdigest(),
                    "mode": exported.stat().st_mode & 0o777,
                }
            )
    _validate_import_ownership(manifest)
    return manifest


def _validate_import_ownership(manifest: Mapping[str, Any]) -> None:
    prefixes = manifest["pythonpath_prefixes"]
    if any(a != b and a.startswith(b + "/") for a in prefixes for b in prefixes):
        raise SourceMaterializationError("overlapping_import_roots", repr(prefixes))
    files = {row["path"]: row for row in manifest["files"]}
    candidates: dict[str, str] = {}
    for prefix in prefixes:
        for path in files:
            if not path.startswith(prefix + "/"):
                continue
            parts = path[len(prefix) + 1 :].split("/")
            if parts[0].split(".")[0] in ("sitecustomize", "usercustomize"):
                raise SourceMaterializationError("unsupported_startup_hook", prefix)
            name = parts[0] if len(parts) > 1 else PurePosixPath(parts[0]).stem if path.endswith(".py") else ""
            if name.isidentifier() and candidates.setdefault(name, prefix) != prefix:
                raise SourceMaterializationError("ambiguous_import_roots", name)
    owned: dict[str, str] = {}
    touched = [row["path"] for row in manifest["modules"]] + manifest["deleted_paths"]
    for path in touched:
        prefix = next(prefix for prefix in prefixes if path.startswith(prefix + "/"))
        owner = path[len(prefix) + 1 :].split("/")[0]
        if owner in owned and owned[owner] != prefix:
            raise SourceMaterializationError("ambiguous_import_roots", owner)
        owned[owner] = prefix
    for owner, prefix in owned.items():
        module_paths: dict[str, str] = {}
        for path in files:
            if not path.startswith(f"{prefix}/{owner}/"):
                continue
            if _native(path) or PurePosixPath(path).suffix.lower() in (".pyc", ".pyo"):
                raise SourceMaterializationError("unsupported_native_runtime", path)
            if path.endswith(".py"):
                name = _module(path, prefix, files)
                if name in module_paths or name in manifest["deleted_modules"]:
                    raise SourceMaterializationError("ambiguous_module_ownership", name)
                module_paths[name] = path


def materialize_source_stack(current_best: Mapping[str, Any], output_dir: Path) -> dict[str, Any] | None:
    """Export the complete accepted git trees, or refuse unresolved source layers.

    Version 1 accepts regular Python changes with explicit artifact coverage.
    It verifies snapshot bytes against each post-KEEP commit and exports the
    final accepted commit, independent of the live worktree's current HEAD.
    Unversioned files under any import prefix, except ordinary Python caches,
    leave runtime closure unresolved, including incidental repo-root output.
    """
    entries = [
        entry
        for entry in (current_best.get("optimization_stack") or [])
        if isinstance(entry, Mapping) and entry.get("scope") == "source_patch"
    ]
    if not entries:
        return None
    staging: Path | None = None
    try:
        layers = [_read_layer(entry) for entry in entries]
        ids = [layer["id"] for layer in layers]
        if len(ids) != len(set(ids)):
            raise SourceMaterializationError("duplicate_layer_id", repr(ids))
        output_dir.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=".assembling-", dir=output_dir))
        manifest = _assemble(layers, staging)
        content = (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()
        digest = hashlib.sha256(content).hexdigest()
        (staging / "manifest.json").write_bytes(content)
        destination = output_dir / digest
        if destination.exists():
            nodes = list(destination.rglob("*"))
            if destination.is_symlink() or any(path.is_symlink() for path in nodes):
                raise SourceMaterializationError("corrupt_materialization", str(destination))
            existing = {p.relative_to(destination).as_posix(): p for p in nodes if not p.is_dir()}
            expected = {row["path"] for row in manifest["files"]} | {"manifest.json"}
            directories = {
                parent.as_posix()
                for name in expected
                for parent in PurePosixPath(name).parents
                if parent.as_posix() != "."
            }
            if {p.relative_to(destination).as_posix() for p in nodes if p.is_dir()} != directories:
                raise SourceMaterializationError("corrupt_materialization", str(destination))
            if set(existing) != expected or _regular(destination / "manifest.json", destination) != content:
                raise SourceMaterializationError("corrupt_materialization", str(destination))
            for row in manifest["files"]:
                path = existing[row["path"]]
                if (
                    hashlib.sha256(_regular(path, destination)).hexdigest() != row["sha256"]
                    or path.stat().st_mode & 0o777 != row["mode"]
                ):
                    raise SourceMaterializationError("corrupt_materialization", row["path"])
        else:
            staging.rename(destination)
            staging = None
        return {
            "schema_version": SCHEMA_VERSION,
            "status": "ready",
            "bundle_root": str(destination.resolve()),
            "manifest_path": "manifest.json",
            "manifest_sha256": digest,
            "required_layer_ids": ids,
            "pythonpath_prefixes": manifest["pythonpath_prefixes"],
        }
    except SourceMaterializationError:
        raise
    except (OSError, ValueError, TypeError, subprocess.TimeoutExpired) as exc:
        raise SourceMaterializationError("source_evidence_unavailable", str(exc)) from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging)
