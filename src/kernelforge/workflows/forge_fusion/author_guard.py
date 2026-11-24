# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Workspace guard that snapshots a Git worktree before authoring and restores every out-of-scope change."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from kernelforge.utils.git import git_bytes

from .emit import _is_fused_module_name


class AuthorSafetyError(RuntimeError):
    """Report an author workspace state that cannot be safely accepted."""

    def __init__(
        self,
        message: str,
        *,
        paths: Optional[list[str]] = None,
        transient: bool = False,
    ) -> None:
        super().__init__(message)
        self.paths = list(paths or [])
        self.transient = bool(transient)


@dataclass(frozen=True)
class _WorkspacePathState:
    """Exact restorable state for one Git-visible worktree path."""

    kind: str
    data: bytes = b""
    mode: int = 0


@dataclass(frozen=True)
class _AuthorWorkspaceOutcome:
    """What one guarded author run left behind once restoration finished."""

    violations: tuple[str, ...] = ()
    created: tuple[str, ...] = ()


def _git_bytes(
    root: Path,
    *args: str,
    input_data: bytes | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run one bounded Git plumbing command without decoding path bytes."""
    try:
        result = git_bytes("-C", str(root), *args, input=input_data, check=False, timeout=60)
    except (OSError, subprocess.SubprocessError) as exc:
        # Pure I/O, and ``SubprocessError`` covers ``TimeoutExpired``: a 60s ``git ls-files`` timeout against an NFS
        # worktree under a concurrent serving campaign is weather, not a verdict on what the author did.
        raise AuthorSafetyError(
            f"author workspace Git command failed: git {' '.join(args[:3])}: {type(exc).__name__}",
            transient=True,
        ) from exc
    if check and result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()[-400:]
        raise AuthorSafetyError(
            f"author workspace Git command failed: git {' '.join(args[:3])}: {detail or f'exit {result.returncode}'}"
        )
    return result


def _decode_git_path(value: bytes) -> str:
    """Decode a NUL-delimited Git path without losing arbitrary bytes."""
    return value.decode(errors="surrogateescape")


def _nul_git_paths(output: bytes) -> set[str]:
    return {_decode_git_path(value) for value in output.split(b"\0") if value}


def _capture_path_state(path: Path) -> _WorkspacePathState:
    """Capture a regular file, symlink, or absence without following symlinks."""
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return _WorkspacePathState("absent")
    except OSError as exc:
        # Reading the worktree is the guard's own bookkeeping, not a verdict on what the author did, and it recovers
        # on its own -- same reason the Git command and index-lock paths are marked retryable.
        raise AuthorSafetyError(
            f"cannot inspect author workspace path: {path}",
            transient=True,
        ) from exc
    mode = stat.S_IMODE(metadata.st_mode)
    if stat.S_ISLNK(metadata.st_mode):
        try:
            return _WorkspacePathState(
                "symlink",
                os.fsencode(os.readlink(path)),
                mode,
            )
        except OSError as exc:
            raise AuthorSafetyError(
                f"cannot read author workspace symlink: {path}",
                transient=True,
            ) from exc
    if stat.S_ISREG(metadata.st_mode):
        try:
            return _WorkspacePathState("file", path.read_bytes(), mode)
        except OSError as exc:
            raise AuthorSafetyError(
                f"cannot snapshot author workspace file: {path}",
                transient=True,
            ) from exc
    if stat.S_ISDIR(metadata.st_mode):
        return _WorkspacePathState("directory", mode=mode)
    return _WorkspacePathState("unsupported", mode=mode)


FUSION_SCRATCH_DIRNAME = ".forge_fusion"


def _is_fusion_scratch_relpath(rel: str) -> bool:
    """Whether a repo-relative path is forge-fusion's own staging directory."""
    parts = Path(rel).parts
    return len(parts) > 1 and parts[0] == FUSION_SCRATCH_DIRNAME


class AuthorWorkspaceGuard:
    """Restore every Git-visible mutation outside the author's writable scope."""

    def __init__(self, workdir: str, target_files: list[str], new_module_dirs: list[str]) -> None:
        cwd = Path(workdir).expanduser().resolve()
        root_result = _git_bytes(cwd, "rev-parse", "--show-toplevel")
        root_text = root_result.stdout.decode(errors="surrogateescape").strip()
        if not root_text:
            raise AuthorSafetyError("registered authoring requires a Git worktree")
        self.root = Path(root_text).resolve()
        self.cwd = cwd
        self.target_files = [str(self._contained_path(value, "author target")) for value in target_files]
        self.target_relpaths = {Path(path).relative_to(self.root).as_posix() for path in self.target_files}

        # Name inventory per nominated directory.
        self.new_module_baselines: dict[str, frozenset[str]] = {}
        for value in new_module_dirs:
            directory = self._contained_path(value, "author module directory")
            try:
                entries = frozenset(os.listdir(directory))
            except FileNotFoundError as exc:
                # An empty inventory is the most permissive scope there is -- every name in it counts as absent -- and
                # the prompt would then advertise a directory the author cannot write into anyway.
                raise AuthorSafetyError(
                    f"author module directory does not exist: {directory}",
                    paths=[str(directory)],
                ) from exc
            except OSError as exc:
                # A missing directory above is a verdict: it is absent on every attempt.
                raise AuthorSafetyError(
                    f"cannot inventory author module directory: {directory}",
                    paths=[str(directory)],
                    transient=True,
                ) from exc
            self.new_module_baselines[directory.relative_to(self.root).as_posix()] = entries
        self.new_module_dirs = [str(self.root / rel) for rel in self.new_module_baselines]

        self.baseline_head = self._head()
        self.baseline_branch = self._branch()
        self.baseline_index_entries = self._index_entries()
        self.baseline_index_flags = self._index_flags()
        self.index_path = self._index_path()
        self.index_lock_path = Path(f"{self.index_path}.lock")
        if self.index_lock_path.exists():
            # ``index.lock`` exists for milliseconds whenever anything else runs a Git command in this worktree, so
            # the next attempt very likely finds it gone.
            raise AuthorSafetyError(
                "Git index is locked before authoring; workspace snapshot is unsafe",
                paths=[str(self.index_lock_path)],
                transient=True,
            )
        try:
            self.baseline_index_bytes = self.index_path.read_bytes()
            self.baseline_index_mode = stat.S_IMODE(self.index_path.stat().st_mode)
        except OSError as exc:
            raise AuthorSafetyError(f"cannot snapshot Git index: {self.index_path}") from exc
        hidden_index_paths = {rel for rel, flag in self.baseline_index_flags.items() if flag != "H"}
        self.baseline_states: dict[str, _WorkspacePathState] = {}
        for rel in self._status_paths() | hidden_index_paths:
            state = _capture_path_state(self._path(rel))
            if state.kind in {"directory", "unsupported"}:
                raise AuthorSafetyError(
                    f"cannot safely snapshot Git-visible path type: {rel}",
                    paths=[rel],
                )
            self.baseline_states[rel] = state

    def _contained_path(self, value: str, what: str) -> Path:
        """Resolve one caller-named path, refusing one outside the worktree or reached through a symlink."""
        raw = Path(value).expanduser()
        lexical = Path(os.path.abspath(str(raw if raw.is_absolute() else self.cwd / raw)))
        resolved = lexical.resolve(strict=False) if lexical.is_relative_to(self.root) else lexical
        if not resolved.is_relative_to(self.root):
            raise AuthorSafetyError(f"{what} is outside the Git worktree: {lexical}", paths=[str(lexical)])
        if lexical.is_symlink() or resolved != lexical:
            raise AuthorSafetyError(f"{what} symlink/path escape is not allowed: {lexical}", paths=[str(lexical)])
        return resolved

    def _head(self) -> bytes:
        return _git_bytes(
            self.root,
            "rev-parse",
            "--verify",
            "HEAD",
        ).stdout.strip()

    def _branch(self) -> bytes:
        result = _git_bytes(
            self.root,
            "symbolic-ref",
            "-q",
            "HEAD",
            check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else b""

    def _index_path(self) -> Path:
        raw = _git_bytes(self.root, "rev-parse", "--git-path", "index").stdout.decode(errors="surrogateescape").strip()
        path = Path(raw)
        return path if path.is_absolute() else self.root / path

    def _index_entries(self) -> dict[str, tuple[tuple[str, str, int], ...]]:
        entries: dict[str, list[tuple[str, str, int]]] = {}
        output = _git_bytes(
            self.root,
            "ls-files",
            "--stage",
            "-z",
        ).stdout
        for record in output.split(b"\0"):
            if not record:
                continue
            try:
                metadata, raw_path = record.split(b"\t", 1)
                raw_mode, raw_oid, raw_stage = metadata.split()
                entry = (
                    raw_mode.decode("ascii"),
                    raw_oid.decode("ascii"),
                    int(raw_stage),
                )
            except (ValueError, UnicodeError) as exc:
                raise AuthorSafetyError("cannot parse Git index metadata for author workspace") from exc
            entries.setdefault(_decode_git_path(raw_path), []).append(entry)
        return {path: tuple(sorted(values, key=lambda item: item[2])) for path, values in entries.items()}

    def _index_flags(self) -> dict[str, str]:
        flags: dict[str, str] = {}
        output = _git_bytes(self.root, "ls-files", "-v", "-z").stdout
        for record in output.split(b"\0"):
            if not record:
                continue
            if len(record) < 3 or record[1:2] != b" ":
                raise AuthorSafetyError("cannot parse Git index flags for author workspace")
            flags[_decode_git_path(record[2:])] = record[:1].decode(
                "ascii",
                errors="replace",
            )
        return flags

    def _status_paths(self) -> set[str]:
        paths: set[str] = set()
        paths.update(
            _nul_git_paths(
                _git_bytes(
                    self.root,
                    "diff",
                    "--name-only",
                    "--no-renames",
                    "-z",
                    "--",
                    ".",
                ).stdout
            )
        )
        paths.update(
            _nul_git_paths(
                _git_bytes(
                    self.root,
                    "diff",
                    "--cached",
                    "--name-only",
                    "--no-renames",
                    "-z",
                    "--",
                    ".",
                ).stdout
            )
        )
        paths.update(
            _nul_git_paths(
                _git_bytes(
                    self.root,
                    "ls-files",
                    "--others",
                    "--exclude-standard",
                    "-z",
                ).stdout
            )
        )
        return paths

    def _path(self, rel: str) -> Path:
        raw = Path(rel)
        if raw.is_absolute() or ".." in raw.parts:
            raise AuthorSafetyError(
                f"unsafe Git path returned by workspace query: {rel}",
                paths=[rel],
            )
        return self.root / raw

    def _remove_path(self, rel: str) -> None:
        path = self._path(rel)
        self._ensure_safe_parent(path)
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        except OSError as exc:
            raise AuthorSafetyError(
                f"cannot inspect path during author rollback: {rel}",
                paths=[rel],
            ) from exc
        try:
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                shutil.rmtree(path)
            else:
                path.unlink()
        except OSError as exc:
            raise AuthorSafetyError(
                f"cannot remove path during author rollback: {rel}",
                paths=[rel],
            ) from exc

    def _ensure_safe_parent(self, path: Path) -> None:
        relative = path.parent.relative_to(self.root)
        current = self.root
        for part in relative.parts:
            current /= part
            try:
                metadata = current.lstat()
            except FileNotFoundError:
                current.mkdir()
                continue
            if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
                continue
            rel = current.relative_to(self.root).as_posix()
            if rel in self.preserved_relpaths:
                raise AuthorSafetyError(
                    f"cannot restore non-target below a preserved path: {rel}",
                    paths=[rel],
                )
            try:
                current.unlink()
                current.mkdir()
            except OSError as exc:
                raise AuthorSafetyError(
                    f"cannot repair unsafe parent during author rollback: {rel}",
                    paths=[rel],
                ) from exc

    def _restore_state(self, rel: str, state: _WorkspacePathState) -> None:
        path = self._path(rel)
        self._remove_path(rel)
        if state.kind == "absent":
            return
        self._ensure_safe_parent(path)
        try:
            if state.kind == "file":
                path.write_bytes(state.data)
                path.chmod(state.mode)
            elif state.kind == "symlink":
                os.symlink(os.fsdecode(state.data), path)
            else:
                raise AuthorSafetyError(
                    f"unsupported baseline path type during rollback: {rel}",
                    paths=[rel],
                )
        except OSError as exc:
            raise AuthorSafetyError(
                f"cannot restore path after rejected author run: {rel}",
                paths=[rel],
            ) from exc

    def _restore_index(
        self,
        post_entries: dict[str, tuple[tuple[str, str, int], ...]],
        preserved_targets: set[str],
    ) -> None:
        if self.index_lock_path.exists():
            # Held by another Git command, not by the author: retryable for the same reason as the pre-run check
            # above.
            raise AuthorSafetyError(
                "Git index became locked during authoring; restoration is unsafe",
                paths=[str(self.index_lock_path)],
                transient=True,
            )
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".forge-index-",
            dir=str(self.index_path.parent),
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(self.baseline_index_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, self.baseline_index_mode)
            os.replace(temporary, self.index_path)
        except OSError as exc:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise AuthorSafetyError("cannot restore the Git index exactly") from exc

        for rel in sorted(preserved_targets):
            desired = post_entries.get(rel, ())
            baseline = self.baseline_index_entries.get(rel, ())
            if desired == baseline:
                continue
            if any(stage != 0 for _mode, _oid, stage in desired):
                raise AuthorSafetyError(
                    f"cannot preserve conflicted target index state: {rel}",
                    paths=[rel],
                )
            _git_bytes(
                self.root,
                "update-index",
                "--force-remove",
                "--",
                rel,
            )
            if not desired:
                continue
            if len(desired) != 1:
                raise AuthorSafetyError(
                    f"cannot preserve target index state: {rel}",
                    paths=[rel],
                )
            mode, oid, _stage = desired[0]
            _git_bytes(
                self.root,
                "update-index",
                "--add",
                "--cacheinfo",
                mode,
                oid,
                rel,
            )

    def _restore_clean_tracked(self, rel: str) -> None:
        entries = self.baseline_index_entries.get(rel, ())
        if len(entries) != 1 or entries[0][2] != 0:
            raise AuthorSafetyError(
                f"cannot reconstruct tracked baseline path: {rel}",
                paths=[rel],
            )
        mode = entries[0][0]
        if mode == "160000":
            raise AuthorSafetyError(
                f"cannot safely restore changed submodule path: {rel}",
                paths=[rel],
            )
        path = self._path(rel)
        self._remove_path(rel)
        self._ensure_safe_parent(path)
        _git_bytes(
            self.root,
            "checkout-index",
            "--force",
            "--",
            rel,
        )

    def _allowed_path_is_unsafe(self, rel: str) -> bool:
        path = self._path(rel)
        # Replacing an allowlisted path or one of its parent directories with a symlink changes what that path
        # resolves to.
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(self.root)
        except (OSError, ValueError):
            return True
        state = _capture_path_state(path)
        return path.is_symlink() or resolved != path or state.kind not in {"absent", "file"}

    def _permits_new_relpath(self, rel: str) -> bool:
        """Whether one repo-relative path is a fused module the author may add."""
        path = Path(rel)
        entries = self.new_module_baselines.get(path.parent.as_posix())
        if entries is None or path.name in entries:
            return False
        return path.suffix == ".py" and _is_fused_module_name(path.name)

    def permits_new_path(self, value: str) -> bool:
        """Whether one filesystem path lies in the permitted new-module scope."""
        raw = Path(str(value)).expanduser()
        lexical = Path(os.path.abspath(str(raw if raw.is_absolute() else self.cwd / raw)))
        try:
            relative = lexical.relative_to(self.root)
        except ValueError:
            return False
        rel = relative.as_posix()
        # Checked before the nomination gate: staging is the pipeline's own and exists whether or not this run
        # nominates a module directory.
        if _is_fusion_scratch_relpath(rel):
            return True
        if not self.new_module_baselines:
            return False
        return self._permits_new_relpath(rel)

    def _baseline_has_path_object(self, rel: str) -> bool:
        state = self.baseline_states.get(rel)
        if state is not None:
            return state.kind != "absent"
        return rel in self.baseline_index_entries

    def _minimal_restore_paths(self, violations: set[str]) -> set[str]:
        """Remove redundant parent/child paths according to baseline structure."""
        selected = set(violations)
        ordered = sorted(violations, key=lambda value: len(Path(value).parts))
        for index, parent in enumerate(ordered):
            parent_parts = Path(parent).parts
            for child in ordered[index + 1 :]:
                child_parts = Path(child).parts
                if child_parts[: len(parent_parts)] != parent_parts:
                    continue
                if self._baseline_has_path_object(child) and not (self._baseline_has_path_object(parent)):
                    selected.discard(parent)
                else:
                    selected.discard(child)
        return selected

    def _changed_paths(
        self,
        entries: dict[str, tuple[tuple[str, str, int], ...]],
        flags: dict[str, str],
        status: set[str],
    ) -> set[str]:
        """Git-visible paths whose index entry, index flag or worktree state differs from the baseline snapshot."""
        baseline_entries, baseline_flags = self.baseline_index_entries, self.baseline_index_flags
        changed = {
            rel
            for rel in baseline_entries.keys() | entries.keys()
            if baseline_entries.get(rel, ()) != entries.get(rel, ())
        }
        changed.update(rel for rel in baseline_flags.keys() | flags.keys() if baseline_flags.get(rel) != flags.get(rel))
        changed.update(status - self.baseline_states.keys())
        changed.update(
            rel for rel, state in self.baseline_states.items() if _capture_path_state(self._path(rel)) != state
        )
        return changed

    def enforce(self) -> _AuthorWorkspaceOutcome:
        """Restore out-of-scope deltas and report rejections plus new modules."""
        if self._head() != self.baseline_head or self._branch() != self.baseline_branch:
            raise AuthorSafetyError(
                "author changed Git HEAD or branch; automatic restoration is unsafe",
                paths=["<git-head>"],
            )

        post_entries = self._index_entries()
        post_flags = self._index_flags()
        changed = self._changed_paths(post_entries, post_flags, self._status_paths())
        # A new fused helper module inside the nominated scope is part of the authored fusion, so it is allowed to
        # survive.
        scratch = {rel for rel in changed if _is_fusion_scratch_relpath(rel)}
        created = {
            rel
            for rel in changed - self.target_relpaths - scratch
            if rel not in post_entries and self._permits_new_relpath(rel)
        }
        allowed = self.target_relpaths | created | scratch
        unsafe_allowed = {rel for rel in changed & allowed if self._allowed_path_is_unsafe(rel)}
        violations = (changed - allowed) | unsafe_allowed
        if not violations:
            return _AuthorWorkspaceOutcome(created=tuple(sorted(created)))

        preserved = allowed - unsafe_allowed
        unsafe_flags = {rel for rel in preserved if self.baseline_index_flags.get(rel) != post_flags.get(rel)}
        violations |= unsafe_flags
        preserved -= unsafe_flags
        # Paths the current transaction must not clobber while restoring others.
        self.preserved_relpaths = preserved

        self._restore_index(post_entries, preserved)
        restore_paths = self._minimal_restore_paths(violations)
        for rel in sorted(restore_paths, key=lambda value: len(Path(value).parts)):
            baseline = self.baseline_states.get(rel)
            if baseline is not None:
                self._restore_state(rel, baseline)
            elif rel in self.baseline_index_entries:
                self._restore_clean_tracked(rel)
            else:
                self._remove_path(rel)

        failed = self._changed_paths(self._index_entries(), self._index_flags(), self._status_paths()) - preserved
        if failed:
            raise AuthorSafetyError(
                "could not prove exact restoration of non-target paths",
                paths=sorted(failed),
            )
        return _AuthorWorkspaceOutcome(
            violations=tuple(sorted(violations)),
            created=tuple(sorted(created - violations)),
        )
