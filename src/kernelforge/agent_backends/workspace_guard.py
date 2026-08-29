# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Workspace integrity around one agent session.

An implementer session is allowed to edit the files it was pointed at and
nothing else. This snapshots what the session must not disturb -- the target
files, HEAD and the active branch, the protected measurement set, and on a
dirty baseline the index and refs -- then reports what deviated and puts back
what it can.

The distinction the whole thing turns on is a verdict about the session versus
the guard failing at its own bookkeeping; see :class:`WorkspaceSafetyError`.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
from fnmatch import fnmatch
import os
import shutil
import stat
from pathlib import Path
from typing import Any

from kernelforge.agent_backends.base import AgentProviderError, AgentRunSpec
from kernelforge.llm.git import git
from kernelforge.llm.workspace_policy import (
    is_protected_path,
    protected_path_inventory,
)

log = logging.getLogger(__name__)


def _nul_paths(output: str) -> list[str]:
    """Split a NUL-delimited git path list."""
    return [path for path in output.split("\0") if path]


def _summarize_paths(entries: list[str], limit: int = 10) -> str:
    """Name the first few blocking paths and count the rest.

    A workspace can inherit hundreds of them from the loop's own bookkeeping,
    and a refusal nobody can read through is worth little more than one that
    names nothing at all.
    """
    if len(entries) <= limit:
        return ", ".join(entries)
    return ", ".join(entries[:limit]) + f", and {len(entries) - limit} more"


class WorkspaceSafetyError(AgentProviderError):
    """Report a workspace-integrity violation by an agent session.

    ``agent_safety_rejection`` says whether this instance is a VERDICT about what
    the session did -- a violation, a moved HEAD, an unsupported path type -- and
    is therefore identical on every retry. The same class also carries the guard's
    own bookkeeping failures (a snapshot it could not read, a Git query that timed
    out, a restore that did not finish), which say nothing about the session and
    do recover on their own; those pass ``rejection=False``. Callers read the
    attribute rather than the class name, so a stalled ``git ls-files`` on NFS no
    longer abandons the work a real rejection is meant to abandon.
    """

    def __init__(self, *args: Any, rejection: bool = True) -> None:
        super().__init__(*args)
        self.agent_safety_rejection = bool(rejection)


def _git_output(cwd: Path, *args: str) -> str:
    """Run a read-only git query and return decoded stdout."""
    result = git(*args, cwd=cwd, check=False, text=False)
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise WorkspaceSafetyError(f"git {' '.join(args)} failed: {detail}", rejection=False)
    return result.stdout.decode(errors="surrogateescape")


class WorkspaceGuard:
    """Protect Forge git state and benchmark files around an agent session."""

    def __init__(
        self,
        spec: AgentRunSpec,
        *,
        dirty_baseline_default: bool = False,
    ) -> None:
        """Initialize guard state from one run specification."""
        self.spec = spec
        self.allow_dirty_baseline = (
            dirty_baseline_default if spec.allow_dirty_baseline is None else bool(spec.allow_dirty_baseline)
        )
        self.root = Path(spec.cwd).resolve()
        self.head = ""
        self.branch = ""
        self.target_paths: set[Path] = set()
        self.driver_path: Path | None = None
        self.snapshots: dict[Path, tuple[str, bytes, int]] = {}
        self.target_snapshots: dict[Path, tuple[bool, bytes, int]] = {}
        self.baseline_protected_ignored: set[Path] = set()
        self.read_only_state: tuple | None = None
        self.baseline_path_snapshots: dict[str, tuple[str, bytes, int]] = {}
        self.baseline_tracked_paths: set[str] = set()
        self.baseline_dirty_paths: set[str] = set()
        self.baseline_index_entries: dict[str, tuple[str, ...]] = {}
        self.baseline_index_path: Path | None = None
        self.baseline_index_snapshot: tuple[str, bytes, int] | None = None
        self.baseline_refs: dict[str, str] = {}
        self.prepared = False
        self.skipped = False

    @staticmethod
    def is_read_only_session(spec: AgentRunSpec) -> bool:
        """Whether this session cannot write, so the guard has nothing to protect.

        Most of what follows exists to roll a run back: it demands a git
        worktree, refuses a dirty one, snapshots the files an implementer may touch,
        and pins HEAD so a bad turn can be reset away. A session that cannot
        write has nothing to roll back, and the clean-worktree rule would
        additionally refuse to run for a caller holding unrelated uncommitted
        work -- which is the normal state once a loop is under way.

        Skipping also gives up the after-the-fact checks in :meth:`verify`, so
        this stays deliberately narrow. Any route to the filesystem -- a
        writable session, declared target files, a driver script, or a tool
        policy still granting write or shell -- keeps the full guard.

        ``read_only_resume`` is excluded even though it is read-only: its whole
        purpose is the :meth:`verify` check that the caller's dirty state came
        back untouched, which is exactly what skipping would drop.
        """
        policy = spec.tool_policy
        return (
            not spec.writable
            and not spec.read_only_resume
            and not spec.target_files
            and not spec.driver_script
            and policy is not None
            and not policy.write
            and not policy.shell
        )

    def _guards_dirty_baseline(self) -> bool:
        """Whether this turn inherits, instead of rejecting, a dirty worktree."""
        return self.spec.read_only_resume or self.allow_dirty_baseline

    def _resolve_path(self, value: str) -> Path:
        """Resolve a spec path against the session working directory."""
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = Path(self.spec.cwd) / path
        return path.resolve()

    def _target_exempt(self) -> set[Path]:
        """Declared targets that the default name globs must not reclaim.

        ``target_files`` is the caller's own per-turn allowlist, so a path on it
        is by definition permitted to change -- yet ``PROTECTED_GLOBS`` still
        match it by name. That is right for an implementer turn, whose targets
        are framework sources and whose harness is off-limits; it is wrong for
        the turn whose sole deliverable *is* the harness, where the caller lists
        one ``.forge_fusion/kernel_harness_*.py`` target and the guard then
        rejects the very file the agent was told to write.

        Explicit protection still wins: ``protected_paths`` and the driver are
        never exempted, so a caller cannot launder a protected path by also
        naming it a target. Rollback is unaffected -- every path dropped here is
        covered by ``_restore_target_snapshots``.
        """
        explicit = {self._resolve_path(path) for path in self.spec.protected_paths if path}
        if self.driver_path is not None:
            explicit.add(Path(self.driver_path).resolve())
        return self.target_paths - explicit

    def _is_protected(self, relative: str) -> bool:
        """Return whether a repository path belongs to the measurement surface."""
        if (self.root / relative).resolve() in self._target_exempt():
            return False
        exact_paths = list(self.spec.protected_paths)
        if self.driver_path is not None:
            exact_paths.append(str(self.driver_path))
        return is_protected_path(
            relative,
            workspace=self.root,
            exact_paths=exact_paths,
            extra_globs=self.spec.protected_globs,
        )

    def _ignored_protected_paths(self) -> set[Path]:
        """List the complete protected inventory using the shared policy."""

        exact_paths = list(self.spec.protected_paths)
        if self.driver_path is not None:
            exact_paths.append(str(self.driver_path))
        try:
            return (
                set(
                    protected_path_inventory(
                        self.root,
                        exact_paths=exact_paths,
                        extra_globs=self.spec.protected_globs,
                    )
                )
                - self._target_exempt()
            )
        except OSError as error:
            raise WorkspaceSafetyError(
                f"Could not inventory protected workspace paths: {error}",
                rejection=False,
            ) from error

    def prepare(self) -> None:
        """Validate the baseline and snapshot files needed for safe rollback."""
        if self.is_read_only_session(self.spec):
            self.skipped = True
            self.prepared = True
            log.info(
                "workspace guard skipped for a read-only session in %s: "
                "no rollback to protect, and no clean-worktree requirement",
                self.spec.cwd,
            )
            return
        root = _git_output(Path(self.spec.cwd).resolve(), "rev-parse", "--show-toplevel").strip()
        if not root:
            raise WorkspaceSafetyError("the workspace guard requires a git worktree")
        self.root = Path(root).resolve()
        self.target_paths = {self._resolve_path(path) for path in self.spec.target_files if path}
        if self.spec.driver_script:
            self.driver_path = self._resolve_path(self.spec.driver_script)

        unstaged, staged, untracked = self._current_changes()
        if self.spec.read_only_resume:
            policy = self.spec.tool_policy
            if self.spec.writable or policy is None or policy.write or policy.shell:
                raise WorkspaceSafetyError(
                    "a read-only resume requires writable=False and a tool policy with write=False and shell=False"
                )
        elif self._guards_dirty_baseline():
            # Nothing to validate up front: this state was inherited, not produced
            # by the turn, and refusing it here would make the phase unrunnable in
            # the only worktrees it ever runs in. verify() judges the deviations.
            pass
        elif self.spec.allow_dirty_targets:
            unexpected = [
                relative for relative in unstaged if (self.root / relative).resolve() not in self.target_paths
            ]
            violations: list[str] = []
            if staged:
                violations.append(f"staged changes: {', '.join(staged)}")
            if untracked and not self.spec.allow_untracked:
                violations.append(f"untracked files: {', '.join(untracked)}")
            if unexpected:
                violations.append(f"non-target changes: {', '.join(unexpected)}")
            if violations:
                raise WorkspaceSafetyError(
                    "a resumed session requires only unstaged target changes; " + "; ".join(violations)
                )
        else:
            blocking = [
                *(f"staged: {relative}" for relative in staged),
                *(f"modified: {relative}" for relative in unstaged),
            ]
            # The caller owns whether untracked state is expected here, exactly
            # as it does in the resume branch above: the loop writes its own
            # experiment ledger into the workspace it hands the implementer, so
            # every iteration would otherwise be refused for the caller's files.
            if not self.spec.allow_untracked:
                blocking.extend(f"untracked: {relative}" for relative in untracked)
            if blocking:
                raise WorkspaceSafetyError(
                    "the workspace guard requires a clean tracked and non-ignored "
                    "worktree; " + _summarize_paths(blocking)
                )

        for path in self.target_paths:
            exists = path.is_file()
            content = path.read_bytes() if exists else b""
            mode = path.stat().st_mode & 0o777 if exists else 0
            self.target_snapshots[path] = (exists, content, mode)

        self.head = _git_output(self.root, "rev-parse", "HEAD").strip()
        self.branch = _git_output(self.root, "rev-parse", "--abbrev-ref", "HEAD").strip()
        self.baseline_protected_ignored = self._ignored_protected_paths()
        for path in self.baseline_protected_ignored:
            self.snapshots[path] = self._filesystem_snapshot(path)
        if self.spec.read_only_resume:
            self.read_only_state = self._read_only_state()
        if self._guards_dirty_baseline():
            self._snapshot_baseline(unstaged, staged, untracked)
        self.prepared = True

    def _drop_ignored_untracked(self, untracked: list[str]) -> list[str]:
        """Drop untracked paths the caller declared as a tool's own droppings.

        Applied once, in :meth:`_current_changes` -- the single place the
        untracked set is produced -- so all seven readers of it (:meth:`prepare`,
        :meth:`_baseline_deviations`, :meth:`_restore_baseline` twice,
        :meth:`_read_only_state`, :meth:`rollback` and :meth:`verify`) see one
        list rather than seven chances to disagree. Filtering anywhere else is
        redundant; keep this the only call site so that stays true.

        Deliberately narrower than ``allow_untracked``. Profilers write into the
        working directory because the working directory is what they are handed:
        ``rocprofv3`` drops ``.rocprofv3/<pid>-<pid>-counter_values.dat`` and a
        ``<pid>_results.db`` next to it, and a session was failed for those
        rather than for anything it did. Naming them keeps the guard's answer to
        every path nobody declared unchanged.
        """
        patterns = list(self.spec.ignored_untracked_globs)
        if not patterns:
            return untracked
        return [relative for relative in untracked if not any(fnmatch(relative, pattern) for pattern in patterns)]

    def _current_changes(self) -> tuple[list[str], list[str], list[str]]:
        """Return unstaged, staged, and new non-ignored repository paths."""
        unstaged = _nul_paths(_git_output(self.root, "diff", "--name-only", "-z"))
        staged = _nul_paths(_git_output(self.root, "diff", "--cached", "--name-only", "-z"))
        untracked = _nul_paths(
            _git_output(
                self.root,
                "ls-files",
                "--others",
                "--exclude-standard",
                "-z",
            )
        )
        return unstaged, staged, self._drop_ignored_untracked(untracked)

    @staticmethod
    def _content_digest(path: Path) -> str:
        """Hash one untracked path without following symbolic links."""
        digest = hashlib.sha256()
        if path.is_symlink():
            digest.update(os.readlink(path).encode(errors="surrogateescape"))
            return digest.hexdigest()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _filesystem_snapshot(path: Path) -> tuple[str, bytes, int]:
        """Capture one path without following a symbolic link."""
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return ("missing", b"", 0)
        except OSError as exc:
            raise WorkspaceSafetyError(f"Could not snapshot {path}: {exc}", rejection=False) from exc

        mode = stat.S_IMODE(metadata.st_mode)
        try:
            if stat.S_ISLNK(metadata.st_mode):
                target = os.readlink(path).encode(errors="surrogateescape")
                return ("symlink", target, mode)
            if stat.S_ISREG(metadata.st_mode):
                return ("file", path.read_bytes(), mode)
            if stat.S_ISDIR(metadata.st_mode):
                return ("directory", b"", mode)
        except OSError as exc:
            raise WorkspaceSafetyError(f"Could not snapshot {path}: {exc}", rejection=False) from exc
        raise WorkspaceSafetyError(f"the read-only guard does not support path type: {path}")

    @staticmethod
    def _remove_filesystem_path(path: Path) -> None:
        """Remove one path without following a symbolic link."""
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            return
        if stat.S_ISDIR(metadata.st_mode) and not stat.S_ISLNK(metadata.st_mode):
            shutil.rmtree(path)
        else:
            path.unlink()

    @classmethod
    def _restore_filesystem_snapshot(
        cls,
        path: Path,
        snapshot: tuple[str, bytes, int],
    ) -> None:
        """Restore one exact file, link, directory, or missing-path state."""
        kind, content, mode = snapshot
        if kind == "missing":
            cls._remove_filesystem_path(path)
            return

        try:
            metadata = path.lstat()
        except FileNotFoundError:
            metadata = None
        if metadata is not None:
            same_kind = (
                (kind == "file" and stat.S_ISREG(metadata.st_mode))
                or (kind == "symlink" and stat.S_ISLNK(metadata.st_mode))
                or (kind == "directory" and stat.S_ISDIR(metadata.st_mode))
            )
            if not same_kind or kind == "symlink":
                cls._remove_filesystem_path(path)

        path.parent.mkdir(parents=True, exist_ok=True)
        if kind == "file":
            path.write_bytes(content)
            path.chmod(mode)
        elif kind == "symlink":
            os.symlink(content.decode(errors="surrogateescape"), path)
        elif kind == "directory":
            path.mkdir(parents=True, exist_ok=True)
            path.chmod(mode)
        else:
            raise WorkspaceSafetyError(
                f"Unknown workspace snapshot kind: {kind}",
                rejection=False,
            )

    def _current_refs(self) -> dict[str, str]:
        """Return every repository ref visible to the guarded worktree."""
        refs: dict[str, str] = {}
        output = _git_output(
            self.root,
            "for-each-ref",
            "--format=%(refname) %(objectname)",
        )
        for line in output.splitlines():
            ref, separator, object_id = line.partition(" ")
            if separator and ref and object_id:
                refs[ref] = object_id
        return refs

    def _index_entries(self) -> dict[str, tuple[str, ...]]:
        """Return the ``mode oid stage`` records the index holds per path."""
        entries: dict[str, list[str]] = {}
        for record in _git_output(self.root, "ls-files", "--stage", "-z").split("\0"):
            if not record:
                continue
            metadata, separator, relative = record.partition("\t")
            if not separator:
                raise WorkspaceSafetyError(
                    f"Could not parse Git index metadata: {record!r}",
                    rejection=False,
                )
            entries.setdefault(relative, []).append(metadata)
        return {relative: tuple(sorted(values)) for relative, values in entries.items()}

    def _snapshot_baseline(
        self,
        unstaged: list[str],
        staged: list[str],
        untracked: list[str],
    ) -> None:
        """Save enough exact state to reconstruct an arbitrary dirty baseline."""
        self.baseline_tracked_paths = set(_nul_paths(_git_output(self.root, "ls-files", "-z")))
        self.baseline_dirty_paths = set([*unstaged, *staged, *untracked])
        for relative in self.baseline_dirty_paths:
            self.baseline_path_snapshots[relative] = self._filesystem_snapshot(self.root / relative)
        self.baseline_index_entries = self._index_entries()
        self.baseline_refs = self._current_refs()

        index_value = _git_output(
            self.root,
            "rev-parse",
            "--git-path",
            "index",
        ).strip()
        if not index_value:
            raise WorkspaceSafetyError(
                "Could not locate the Git index for the workspace guard",
                rejection=False,
            )
        index_path = Path(index_value)
        if not index_path.is_absolute():
            index_path = self.root / index_path
        self.baseline_index_path = index_path.resolve()
        self.baseline_index_snapshot = self._filesystem_snapshot(self.baseline_index_path)

    def _deviates_from_baseline(
        self,
        relative: str,
        post_entries: dict[str, tuple[str, ...]],
    ) -> bool:
        """Whether this turn, rather than the caller, is responsible for one path."""
        if self.baseline_index_entries.get(relative) != post_entries.get(relative):
            return True
        snapshot = self.baseline_path_snapshots.get(relative)
        if snapshot is None:
            # Clean when the turn started, dirty now: the turn wrote it.
            return True
        return self._filesystem_snapshot(self.root / relative) != snapshot

    def _baseline_deviations(
        self,
    ) -> tuple[list[str], list[str], list[str], list[str]]:
        """Reduce the current dirty sets to the paths this turn itself changed.

        The fourth element is reported separately on purpose. Every other element
        names the bucket a path occupies now, and each bucket has its own rule --
        ``allow_untracked`` forgives untracked paths, for one. An index record the
        turn changed has to be judged before that: unstaging a file the caller had
        staged moves it into the untracked bucket, where the forgiving rule would
        accept the caller's work being undone.
        """
        unstaged, staged, untracked = self._current_changes()
        post_entries = self._index_entries()

        def deviated(relative: str) -> bool:
            return self._deviates_from_baseline(relative, post_entries)

        # Undoing an inherited change leaves the path clean, so it disappears from
        # every current dirty list. Silently accepting that would let a turn revert
        # the caller's own work — including a protected measurement file — unseen.
        # ``untracked`` arrives dropping-filtered from _current_changes, so a
        # declared dropping never subtracts from this difference. Move that filter
        # to the verdict sites and this set silently changes meaning.
        reverted = sorted(
            relative
            for relative in self.baseline_dirty_paths.difference(unstaged, staged, untracked)
            if deviated(relative)
        )
        index_changed = sorted(
            relative
            for relative in set(self.baseline_index_entries) | set(post_entries)
            if self.baseline_index_entries.get(relative) != post_entries.get(relative)
        )
        return (
            [*(relative for relative in unstaged if deviated(relative)), *reverted],
            [relative for relative in staged if deviated(relative)],
            [relative for relative in untracked if deviated(relative)],
            index_changed,
        )

    def _run_git_restore(self, *args: str) -> None:
        """Run one repository mutation used only for exact safety recovery."""
        result = git(*args, cwd=self.root, check=False, text=False)
        if result.returncode != 0:
            detail = result.stderr.decode(errors="replace").strip()
            raise WorkspaceSafetyError(
                f"git {' '.join(args)} failed during safety restore: {detail}",
                rejection=False,
            )

    def _remove_empty_parents(self, path: Path) -> None:
        """Remove empty directories created around a rejected untracked file."""
        parent = path.parent
        while parent != self.root and parent.is_relative_to(self.root):
            try:
                parent.rmdir()
            except OSError:
                break
            parent = parent.parent

    def _checkout_index_path(self, relative: str) -> None:
        """Restore one clean tracked path from the already-restored index."""
        path = self.root / relative
        try:
            metadata = path.lstat()
        except FileNotFoundError:
            metadata = None
        if metadata is not None and stat.S_ISDIR(metadata.st_mode):
            shutil.rmtree(path)
        self._run_git_restore("checkout-index", "--force", "--", relative)

    def _restore_baseline(self) -> None:
        """Reconstruct the exact pre-turn refs, index, and Git-visible files."""
        before_unstaged, before_staged, before_untracked = self._current_changes()
        before_protected = self._ignored_protected_paths()
        current_refs = self._current_refs()

        for ref in sorted(current_refs.keys() - self.baseline_refs.keys()):
            self._run_git_restore("update-ref", "-d", ref)
        for ref, object_id in sorted(self.baseline_refs.items()):
            if current_refs.get(ref) != object_id:
                self._run_git_restore("update-ref", ref, object_id)
        if self.branch == "HEAD":
            self._run_git_restore(
                "update-ref",
                "--no-deref",
                "HEAD",
                self.head,
            )
        else:
            self._run_git_restore(
                "symbolic-ref",
                "HEAD",
                f"refs/heads/{self.branch}",
            )

        if self.baseline_index_path is None or self.baseline_index_snapshot is None:
            raise WorkspaceSafetyError(
                "Workspace safety restore has no Git index snapshot",
                rejection=False,
            )
        self._restore_filesystem_snapshot(
            self.baseline_index_path,
            self.baseline_index_snapshot,
        )

        after_unstaged, after_staged, after_untracked = self._current_changes()
        changed_paths = set(
            [
                *before_unstaged,
                *before_staged,
                *before_untracked,
                *after_unstaged,
                *after_staged,
                *after_untracked,
                *self.baseline_path_snapshots,
            ]
        )
        for relative in sorted(
            changed_paths,
            key=lambda value: len(Path(value).parts),
            reverse=True,
        ):
            path = self.root / relative
            snapshot = self.baseline_path_snapshots.get(relative)
            if snapshot is not None:
                self._restore_filesystem_snapshot(path, snapshot)
            elif relative in self.baseline_tracked_paths:
                self._checkout_index_path(relative)
            else:
                self._remove_filesystem_path(path)
                self._remove_empty_parents(path)

        for path in before_protected - self.baseline_protected_ignored:
            self._remove_filesystem_path(path)
            self._remove_empty_parents(path)
        for path, snapshot in self.snapshots.items():
            if self._filesystem_snapshot(path) != snapshot:
                self._restore_filesystem_snapshot(path, snapshot)
        # A target may be Git-ignored, in which case none of the Git-visible
        # recovery above ever names it; its own snapshot is the only record. This
        # runs on the path whose caller turns a failure into a raised rejection,
        # so it must not suppress one.
        self._restore_target_snapshots(strict=True)

    def _read_only_violations(self) -> list[str]:
        """Describe any Git-visible state changed since the read-only snapshot."""
        violations: list[str] = []
        current_head = _git_output(self.root, "rev-parse", "HEAD").strip()
        current_branch = _git_output(self.root, "rev-parse", "--abbrev-ref", "HEAD").strip()
        if current_head != self.head or current_branch != self.branch:
            violations.append("HEAD or active branch changed")
        if self._current_refs() != self.baseline_refs:
            violations.append("Git refs changed")
        if self._read_only_state() != self.read_only_state:
            violations.append("tracked, staged, or untracked files changed")

        current_protected = self._ignored_protected_paths()
        changed_snapshots = [
            str(path) for path, snapshot in self.snapshots.items() if self._filesystem_snapshot(path) != snapshot
        ]
        new_protected = [str(path) for path in current_protected - self.baseline_protected_ignored]
        if changed_snapshots or new_protected:
            violations.append("protected ignored files changed: " + ", ".join([*changed_snapshots, *new_protected]))
        return violations

    def _read_only_state(self) -> tuple:
        """Fingerprint all Git-visible state a read-only turn must preserve."""
        unstaged, staged, untracked = self._current_changes()

        def diff_digest(*args: str) -> str:
            output = _git_output(
                self.root,
                "diff",
                "--binary",
                "--no-ext-diff",
                *args,
                "--",
                ".",
            )
            return hashlib.sha256(output.encode(errors="surrogateescape")).hexdigest()

        untracked_state: list[tuple[str, int, str]] = []
        for relative in untracked:
            path = self.root / relative
            try:
                mode = path.lstat().st_mode
                content = self._content_digest(path)
            except OSError as exc:
                raise WorkspaceSafetyError(
                    f"Could not fingerprint untracked path {relative}: {exc}",
                    rejection=False,
                ) from exc
            untracked_state.append((relative, mode, content))

        return (
            tuple(unstaged),
            tuple(staged),
            diff_digest(),
            diff_digest("--cached"),
            tuple(untracked_state),
        )

    def rollback(self) -> None:
        """Restore the clean baseline after a failed or unsafe session."""
        if not self.prepared or self.skipped:
            return
        if self.spec.read_only_resume:
            violations = self._read_only_violations()
            if not violations:
                return
            try:
                self._restore_baseline()
                remaining = self._read_only_violations()
            except Exception as exc:
                raise WorkspaceSafetyError(
                    f"the read-only session changed the workspace and automatic restoration failed: {exc}"
                ) from exc
            if remaining:
                raise WorkspaceSafetyError(
                    "the read-only session changed the workspace and could not "
                    "restore the pre-run state; remaining changes: " + "; ".join(remaining)
                )
            raise WorkspaceSafetyError(
                "the read-only session changed the workspace; restored the pre-run Git-visible state"
            )
        if self.allow_dirty_baseline:
            # Resetting to HEAD here would delete the caller's inherited dirty
            # state, which is exactly the state this mode exists to carry through a
            # rejection, so recover the snapshot instead. A failed recovery is a
            # worse outcome than the rejection that triggered it and must not be
            # swallowed the way the clean-baseline path below can afford to.
            try:
                self._restore_baseline()
            except Exception as exc:
                # Not a verdict: rollback also runs on the way out of a timeout or
                # a transport failure, so marking this one a rejection reported an
                # expired clock as a deterministic safety stop and abandoned work
                # a retry could have finished.
                raise WorkspaceSafetyError(
                    f"the session ended and the inherited workspace state could not be restored: {exc}",
                    rejection=False,
                ) from exc
            return
        current_head = ""
        current_branch = ""
        with contextlib.suppress(WorkspaceSafetyError):
            current_head = _git_output(self.root, "rev-parse", "HEAD").strip()
            current_branch = _git_output(self.root, "rev-parse", "--abbrev-ref", "HEAD").strip()
        if current_branch == self.branch:
            if current_head != self.head:
                git("reset", "--hard", self.head, cwd=self.root, check=False)
            else:
                git("reset", "--quiet", "HEAD", "--", ".", cwd=self.root, check=False)
                git("checkout", "--", ".", cwd=self.root, check=False)

        self._restore_target_snapshots()

        with contextlib.suppress(WorkspaceSafetyError):
            _, _, untracked = self._current_changes()
            for relative in untracked:
                path = (self.root / relative).resolve()
                if path.is_file() or path.is_symlink():
                    with contextlib.suppress(OSError):
                        path.unlink()

        with contextlib.suppress(WorkspaceSafetyError):
            current_protected = self._ignored_protected_paths()
            for path in current_protected - self.baseline_protected_ignored:
                if path.is_file() or path.is_symlink():
                    with contextlib.suppress(OSError):
                        path.unlink()

        for path, snapshot in self.snapshots.items():
            with contextlib.suppress(OSError):
                if self._filesystem_snapshot(path) != snapshot:
                    self._restore_filesystem_snapshot(path, snapshot)

    def _restore_target_snapshots(self, *, strict: bool = False) -> None:
        """Put every allowlisted target back to the state the turn started from.

        Args:
            strict: When ``True``, let an ``OSError`` propagate. The
                ``allow_dirty_baseline`` recovery is the only caller that must
                prove the restoration happened -- a Git-ignored target is
                recorded nowhere but this snapshot, so a suppressed write would
                leave a rejected turn's edit on disk while the rejection reports
                a clean rollback. Everything reachable from the best-effort tail
                of :meth:`rollback` keeps the suppressing default.
        """
        for path, (existed, content, mode) in self.target_snapshots.items():
            with contextlib.nullcontext() if strict else contextlib.suppress(OSError):
                if existed:
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(content)
                    path.chmod(mode)
                elif path.exists() or path.is_symlink():
                    path.unlink()

    def count_target_edits(self) -> int:
        """Count target paths changed since this guarded turn began."""
        total = 0
        for path, (existed, content, mode) in self.target_snapshots.items():
            exists = path.is_file()
            if exists != existed:
                total += 1
                continue
            if exists and (path.read_bytes() != content or (path.stat().st_mode & 0o777) != mode):
                total += 1
        return total

    def verify(self) -> list[str]:
        """Verify post-run integrity and return allowed tracked changes."""
        if self.skipped:
            return []
        if self.spec.read_only_resume:
            if self._read_only_violations():
                self.rollback()
            return []

        current_head = _git_output(self.root, "rev-parse", "HEAD").strip()
        current_branch = _git_output(self.root, "rev-parse", "--abbrev-ref", "HEAD").strip()
        if current_head != self.head or current_branch != self.branch:
            raise WorkspaceSafetyError("the session changed HEAD or the active branch; the run was rejected")

        index_changed: list[str] = []
        if self._guards_dirty_baseline():
            unstaged, staged, untracked, index_changed = self._baseline_deviations()
        else:
            unstaged, staged, untracked = self._current_changes()
        tracked_changes = list(dict.fromkeys([*unstaged, *staged]))
        protected_changes = [path for path in tracked_changes if self._is_protected(path)]
        protected_untracked = [path for path in untracked if self._is_protected(path)]
        current_protected = self._ignored_protected_paths()
        changed_snapshots = [
            str(path) for path, snapshot in self.snapshots.items() if self._filesystem_snapshot(path) != snapshot
        ]
        new_protected = [str(path) for path in current_protected - self.baseline_protected_ignored]

        violations: list[str] = []
        if index_changed:
            # Judged before the buckets below, which each carry their own rule: a
            # path unstaged by the turn lands among the untracked, where
            # allow_untracked would forgive the caller's staging being undone.
            violations.append(f"git index entries changed: {', '.join(index_changed)}")
        if staged:
            violations.append(f"staged git changes: {', '.join(staged)}")
        if protected_changes:
            violations.append(f"protected tracked files changed: {', '.join(protected_changes)}")
        if protected_untracked:
            violations.append(f"protected files created: {', '.join(protected_untracked)}")
        if changed_snapshots or new_protected:
            paths = [*changed_snapshots, *new_protected]
            violations.append(f"protected ignored files changed: {', '.join(paths)}")
        allow_untracked = self.spec.allow_untracked
        if untracked and not allow_untracked:
            violations.append(f"new non-ignored files are unsupported: {', '.join(untracked)}")
        if violations:
            self.rollback()
            raise WorkspaceSafetyError("; ".join(violations))
        return list(
            dict.fromkeys(
                [
                    *tracked_changes,
                    *(untracked if allow_untracked else []),
                ]
            )
        )
