# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Transactional staging for task-preparer artifacts outside the kernel workspace."""

from __future__ import annotations

import fcntl
import hashlib
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kernelforge.loop.path_ownership import (
    COPY_FILTER_DIRECTORY_NAMES,
    RUNTIME_DIRECTORY_NAMES,
    RUNTIME_FILE_SUFFIXES,
)

# A driver bundle is a handful of sources beside a kernel cache the driver
# rewrites on every compile -- observed: 3 payload files against 693 cache files
# / 382 MB. Staging those costs two copies and three hashes per attempt, and a
# compile mid-attempt makes publish() reject a driver that was fine.
_IGNORED_DIRECTORY_NAMES = RUNTIME_DIRECTORY_NAMES | COPY_FILTER_DIRECTORY_NAMES | {".git"}
_IGNORED_FILE_SUFFIXES = RUNTIME_FILE_SUFFIXES


def _extra_ignored_directory_names() -> set[str]:
    """Deployment-specific cache dirs, since kernel toolchains rename theirs."""
    raw = os.environ.get("FORGE_EXTERNAL_IGNORE_DIRS", "")
    return {name.strip() for name in raw.split(",") if name.strip()}


class ExternalArtifactError(RuntimeError):
    """Raised when an external artifact transaction cannot be completed safely."""


@dataclass(frozen=True)
class ExternalArtifactChanges:
    """Published external artifact paths."""

    wrote_files: tuple[str, ...]
    created_files: tuple[str, ...]


@dataclass(frozen=True)
class _FileEntry:
    kind: str
    digest: str
    mode: int


class ExternalArtifactTransaction:
    """Stage an external driver tree and publish it only after validation.

    The task-preparer agent writes to ``stage_root``, never directly to the
    caller-owned artifact directory. The original tree is mirrored separately so
    a partial publish can restore only this transaction's paths. Out-of-band
    changes are treated as conflicts and are never overwritten.
    Kernel workspaces and audit directories can be excluded from the artifact
    transaction; a nested kernel workspace is exposed in staging through a
    passthrough symlink so existing relative driver imports continue to work.
    """

    def __init__(
        self,
        *,
        driver_path: Path,
        excluded_paths: list[Path] | None = None,
        passthrough_paths: list[Path] | None = None,
        read_only_paths: list[Path] | None = None,
    ) -> None:
        self._ignored_directory_names = _IGNORED_DIRECTORY_NAMES | _extra_ignored_directory_names()
        lexical_driver = Path(os.path.abspath(os.path.expanduser(str(driver_path))))
        if lexical_driver.is_symlink():
            raise ExternalArtifactError(f"external driver cannot be a symlink: {lexical_driver}")
        self.root = lexical_driver.parent.resolve()
        self.driver_path = self.root / lexical_driver.name
        if not self.root.is_dir():
            raise ExternalArtifactError(f"external driver directory does not exist: {self.root}")
        if self.root == Path(self.root.anchor):
            raise ExternalArtifactError(f"refusing to stage a filesystem root as an artifact directory: {self.root}")

        self._lock_fd = os.open(self.root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            os.close(self._lock_fd)
            raise ExternalArtifactError(f"another external artifact transaction is active: {self.root}") from exc

        self._baseline_temporary: tempfile.TemporaryDirectory | None = None
        self._stage_temporary: tempfile.TemporaryDirectory | None = None
        try:
            self._baseline_temporary = tempfile.TemporaryDirectory(prefix="forge_external_baseline_")
            self._stage_temporary = tempfile.TemporaryDirectory(prefix="forge_external_staging_")
            baseline_temporary_root = Path(self._baseline_temporary.name)
            stage_temporary_root = Path(self._stage_temporary.name)
            self._baseline_root = baseline_temporary_root / "artifacts"
            self.stage_root = stage_temporary_root / "artifacts"
            self._published = False
            self._ignored_boundary_rels: set[Path] = set()

            excluded = [p.expanduser().resolve(strict=False) for p in excluded_paths or []]
            passthrough = [p.expanduser().resolve(strict=False) for p in passthrough_paths or []]
            # Avoid recursively copying the transaction itself when the external
            # root is a broad temporary directory such as /tmp.
            excluded.extend(
                [
                    baseline_temporary_root.resolve(),
                    stage_temporary_root.resolve(),
                ]
            )

            self._excluded_rels = self._relative_descendants(excluded)
            self._passthrough: dict[Path, Path] = {
                rel: path for path in passthrough if (rel := self._relative_descendant(path)) is not None
            }
            self._excluded_rels.update(self._passthrough)
            self._read_only_rels: set[Path] = set()
            for path in read_only_paths or []:
                lexical = Path(os.path.abspath(os.path.expanduser(str(path))))
                for candidate in (lexical, lexical.resolve(strict=False)):
                    rel = self._relative_descendant(candidate)
                    if rel is not None:
                        self._read_only_rels.add(rel)

            driver_rel = self.driver_path.relative_to(self.root)
            if self._is_excluded(driver_rel):
                raise ExternalArtifactError(f"external driver is inside an excluded path: {self.driver_path}")

            self._baseline_root.mkdir(parents=True)
            self.stage_root.mkdir(parents=True)
            self._copy_tree(self.root, self._baseline_root)
            self._copy_tree(self._baseline_root, self.stage_root, apply_exclusions=False)
            self._create_passthrough_links()
            self._baseline_manifest = self._manifest(self._baseline_root)
        except Exception:
            if self._stage_temporary is not None:
                self._stage_temporary.cleanup()
            if self._baseline_temporary is not None:
                self._baseline_temporary.cleanup()
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)
            raise

    @property
    def published(self) -> bool:
        return self._published

    @property
    def staged_driver_path(self) -> Path:
        return self.stage_root / self.driver_path.relative_to(self.root)

    def publish(self) -> ExternalArtifactChanges:
        """Publish all staged driver/helper changes to the original directory."""
        if self._published:
            raise ExternalArtifactError("external artifacts were already published")
        self._assert_baseline_intact()

        current = self._manifest(self.root, apply_exclusions=True)
        if current != self._baseline_manifest:
            raise ExternalArtifactError(
                "external artifact directory changed outside the staging transaction; "
                "the concurrent changes were left untouched"
            )

        staged = self._manifest(
            self.stage_root,
            ignored_rels=self._excluded_rels,
        )
        self._validate_staged_symlinks(staged)
        changed = {
            rel
            for rel in set(self._baseline_manifest) | set(staged)
            if self._baseline_manifest.get(rel) != staged.get(rel)
        }
        protected_changes = sorted(rel.as_posix() for rel in changed if self._touches_read_only_path(rel))
        if protected_changes:
            raise ExternalArtifactError(
                "preparer modified read-only external input(s): " + ", ".join(protected_changes)
            )
        excluded_boundary_changes = sorted(
            rel.as_posix() for rel in changed if self._is_ancestor_of_protected_boundary(rel)
        )
        if excluded_boundary_changes:
            raise ExternalArtifactError(
                "preparer changed an ancestor of excluded external state: " + ", ".join(excluded_boundary_changes)
            )

        try:
            self._sync_tree(
                self.stage_root,
                staged,
                scope=changed,
                expected_current=self._baseline_manifest,
            )
        except Exception as exc:
            try:
                self._sync_tree(
                    self._baseline_root,
                    self._baseline_manifest,
                    scope=changed,
                )
            except Exception as rollback_exc:
                raise ExternalArtifactError(
                    f"external artifact publish failed ({exc}); rollback also failed ({rollback_exc})"
                ) from exc
            raise ExternalArtifactError(f"external artifact publish failed and was rolled back: {exc}") from exc

        written = tuple(str(self.root / rel) for rel in sorted(changed))
        created = tuple(
            str(self.root / rel) for rel in sorted(changed) if rel not in self._baseline_manifest and rel in staged
        )
        self._published = True
        return ExternalArtifactChanges(
            wrote_files=written,
            created_files=created,
        )

    def rollback(self) -> None:
        """Confirm that a discarded staging transaction left originals unchanged."""
        if self._published:
            raise ExternalArtifactError("cannot roll back published external artifacts")
        self._assert_baseline_intact()
        if self._manifest(self.root, apply_exclusions=True) != self._baseline_manifest:
            raise ExternalArtifactError(
                "external artifact directory changed outside the staging transaction; "
                "the concurrent changes were left untouched"
            )

    def restore_passthroughs(self) -> None:
        """Reassert read-through links before validating the staged driver."""
        self._create_passthrough_links()

    def close(self) -> None:
        try:
            if self._stage_temporary is not None:
                self._stage_temporary.cleanup()
            if self._baseline_temporary is not None:
                self._baseline_temporary.cleanup()
        finally:
            fcntl.flock(self._lock_fd, fcntl.LOCK_UN)
            os.close(self._lock_fd)

    def _assert_baseline_intact(self) -> None:
        if self._manifest(self._baseline_root) != self._baseline_manifest:
            raise ExternalArtifactError("external artifact rollback snapshot was modified")

    def _relative_descendant(self, path: Path) -> Path | None:
        try:
            return path.relative_to(self.root)
        except ValueError:
            return None

    def _relative_descendants(self, paths: list[Path]) -> set[Path]:
        return {rel for path in paths if (rel := self._relative_descendant(path)) is not None}

    def _is_excluded(self, rel: Path) -> bool:
        return any(rel == excluded or rel.is_relative_to(excluded) for excluded in self._excluded_rels)

    def _is_ignored_name(self, path: Path) -> bool:
        return path.name in self._ignored_directory_names or path.suffix.lower() in _IGNORED_FILE_SUFFIXES

    def _touches_read_only_path(self, rel: Path) -> bool:
        return any(
            rel == protected or rel.is_relative_to(protected) or protected.is_relative_to(rel)
            for protected in self._read_only_rels
        )

    def _is_ancestor_of_protected_boundary(self, rel: Path) -> bool:
        boundaries = self._excluded_rels | self._ignored_boundary_rels
        return any(boundary != rel and boundary.is_relative_to(rel) for boundary in boundaries)

    def _copy_tree(
        self,
        source_root: Path,
        destination_root: Path,
        *,
        apply_exclusions: bool = True,
    ) -> None:
        def copy_directory(source: Path, destination: Path, rel: Path) -> None:
            destination.mkdir(parents=True, exist_ok=True)
            for child in source.iterdir():
                child_rel = rel / child.name
                if self._is_ignored_name(child):
                    if apply_exclusions:
                        self._ignored_boundary_rels.add(child_rel)
                    continue
                if apply_exclusions and self._is_excluded(child_rel):
                    continue

                target = destination / child.name
                child_stat = child.lstat()
                if stat.S_ISLNK(child_stat.st_mode):
                    link_target = os.readlink(child)
                    if os.path.isabs(link_target):
                        raise ExternalArtifactError(f"absolute artifact symlink is not supported: {child}")
                    try:
                        resolved_target = (child.parent / link_target).resolve(strict=False)
                        target_rel = resolved_target.relative_to(source_root.resolve())
                    except (OSError, RuntimeError, ValueError) as exc:
                        raise ExternalArtifactError(f"artifact symlink escapes its staging root: {child}") from exc
                    if apply_exclusions and (self._is_excluded(target_rel) or self._touches_read_only_path(target_rel)):
                        raise ExternalArtifactError(f"artifact symlink targets protected state: {child}")
                    os.symlink(link_target, target)
                elif stat.S_ISDIR(child_stat.st_mode):
                    copy_directory(child, target, child_rel)
                    shutil.copystat(child, target, follow_symlinks=False)
                elif stat.S_ISREG(child_stat.st_mode):
                    shutil.copy2(child, target, follow_symlinks=False)
                else:
                    raise ExternalArtifactError(f"unsupported artifact file type: {child}")

        copy_directory(source_root, destination_root, Path())

    def _validate_staged_symlinks(
        self,
        manifest: dict[Path, _FileEntry],
    ) -> None:
        stage_root = self.stage_root.resolve()
        for rel, entry in manifest.items():
            if entry.kind != "symlink":
                continue
            if os.path.isabs(entry.digest):
                raise ExternalArtifactError(f"staged artifact contains an absolute symlink: {rel}")
            try:
                target = (self.stage_root / rel).parent.joinpath(entry.digest).resolve(strict=False)
                target_rel = target.relative_to(stage_root)
            except (OSError, RuntimeError, ValueError) as exc:
                raise ExternalArtifactError(f"staged artifact symlink escapes the transaction: {rel}") from exc
            if self._is_excluded(target_rel) or self._touches_read_only_path(target_rel):
                raise ExternalArtifactError(f"staged artifact symlink targets protected state: {rel}")

    def _create_passthrough_links(self) -> None:
        for rel, source in self._passthrough.items():
            destination = self.stage_root / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            if os.path.lexists(destination):
                self._remove_path(destination)
            os.symlink(source, destination, target_is_directory=source.is_dir())

    def _manifest(
        self,
        root: Path,
        *,
        apply_exclusions: bool = False,
        ignored_rels: set[Path] | None = None,
    ) -> dict[Path, _FileEntry]:
        entries: dict[Path, _FileEntry] = {}
        ignored_rels = ignored_rels or set()

        def visit(directory: Path, rel: Path) -> None:
            for child in directory.iterdir():
                child_rel = rel / child.name
                if self._is_ignored_name(child):
                    continue
                if apply_exclusions and self._is_excluded(child_rel):
                    continue
                if any(child_rel == ignored or child_rel.is_relative_to(ignored) for ignored in ignored_rels):
                    continue

                child_stat = child.lstat()
                if stat.S_ISLNK(child_stat.st_mode):
                    entries[child_rel] = _FileEntry(
                        kind="symlink",
                        digest=os.readlink(child),
                        mode=0,
                    )
                elif stat.S_ISDIR(child_stat.st_mode):
                    visit(child, child_rel)
                elif stat.S_ISREG(child_stat.st_mode):
                    entries[child_rel] = _FileEntry(
                        kind="file",
                        digest=self._sha256(child),
                        mode=stat.S_IMODE(child_stat.st_mode),
                    )
                else:
                    raise ExternalArtifactError(f"unsupported artifact file type: {child}")

        visit(root, Path())
        return entries

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _sync_tree(
        self,
        source_root: Path,
        desired: dict[Path, _FileEntry],
        *,
        scope: set[Path] | None = None,
        expected_current: dict[Path, _FileEntry] | None = None,
    ) -> None:
        current = self._manifest(self.root, apply_exclusions=True)
        if expected_current is not None:
            checked_paths = scope if scope is not None else set(expected_current) | set(current)
            conflicts = [rel for rel in checked_paths if current.get(rel) != expected_current.get(rel)]
            if conflicts:
                raise ExternalArtifactError(
                    "external artifact path(s) changed concurrently: "
                    + ", ".join(sorted(rel.as_posix() for rel in conflicts))
                )

        removals = [
            rel
            for rel, entry in current.items()
            if (scope is None or rel in scope)
            if rel not in desired or desired[rel].kind != entry.kind
        ]
        for rel in sorted(removals, key=lambda path: len(path.parts), reverse=True):
            self._remove_path(self.root / rel)

        for rel, entry in sorted(desired.items()):
            if scope is not None and rel not in scope:
                continue
            if current.get(rel) == entry:
                continue
            source = source_root / rel
            destination = self.root / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            if entry.kind == "file":
                self._replace_with_file(source, destination)
            elif entry.kind == "symlink":
                self._replace_with_symlink(source, destination)
            else:
                raise ExternalArtifactError(f"unsupported staged artifact kind: {entry.kind}")

        actual = self._manifest(self.root, apply_exclusions=True)
        matches = actual == desired if scope is None else all(actual.get(rel) == desired.get(rel) for rel in scope)
        if not matches:
            raise ExternalArtifactError("external artifact tree does not match the requested state after sync")

    @staticmethod
    def _remove_path(path: Path) -> None:
        if not os.path.lexists(path):
            return
        if path.is_symlink() or path.is_file():
            path.unlink()
            return
        if path.is_dir():
            shutil.rmtree(path)
            return
        path.unlink()

    def _replace_with_file(self, source: Path, destination: Path) -> None:
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.forge-",
            dir=destination.parent,
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        try:
            shutil.copy2(source, temporary_path, follow_symlinks=False)
            os.replace(temporary_path, destination)
        finally:
            if os.path.lexists(temporary_path):
                temporary_path.unlink()

    def _replace_with_symlink(self, source: Path, destination: Path) -> None:
        if destination.is_dir() and not destination.is_symlink():
            shutil.rmtree(destination)
        fd, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.name}.forge-",
            dir=destination.parent,
        )
        os.close(fd)
        temporary_path = Path(temporary_name)
        temporary_path.unlink()
        try:
            os.symlink(os.readlink(source), temporary_path)
            os.replace(temporary_path, destination)
        finally:
            if os.path.lexists(temporary_path):
                temporary_path.unlink()
