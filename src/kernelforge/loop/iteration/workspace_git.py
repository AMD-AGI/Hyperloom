# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The campaign's git workspace: the sources a round reads, and the commits that keep or discard what it wrote."""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
from collections.abc import Sequence
from pathlib import Path

from hyperloom.common.unified_diff import touched_paths
from kernelforge.evaluation.jit_rebuild import tracked_source_changes
from kernelforge.tools.git import git, git_bytes
from kernelforge.tools.new_path_allowlist import matches_commit_new_paths
from kernelforge.workspace.edit_policy import FORGE_PROTECTED_PATHS

log = logging.getLogger(__name__)


# Where a campaign writes its own output inside the workspace.
LOOP_ARTIFACT_ROOT = "forge_experiments"


class WorkspaceGitMixin:
    """Read and move the workspace: what an attempt changed, which new files may ship, and the commit or discard that settles it."""

    def _git(self, *args: str) -> str:
        """Read the workspace repository, reporting git's answer verbatim."""
        result = git(*args, cwd=self.ic.workspace_dir, check=False)
        return (result.stdout + "\n" + result.stderr).strip()

    def _workspace_path(self, value: str) -> str:
        """Normalize a task path relative to the workspace when possible."""
        path = Path(value)
        if not path.is_absolute():
            path = Path(self.ic.workspace_dir) / path
        resolved = path.resolve()
        try:
            return str(resolved.relative_to(Path(self.ic.workspace_dir).resolve()))
        except ValueError:
            return str(resolved)

    def _tracked_diff_from_head(self) -> str:
        """Return all staged and unstaged tracked changes relative to HEAD."""
        return self._git("diff", "HEAD", "--", ".")

    def _candidate_changes(self, base: str) -> tuple[str, list[str]]:
        """Snapshot tracked and allowed new sources without changing the real index."""
        admitted = self._new_paths()[0] if self.ic.commit_new_paths else []

        def read_changes(env=None):
            patch = git("diff", base, "--", ".", cwd=self.ic.workspace_dir, env=env).stdout
            names = git("diff", "--name-only", base, "--", ".", cwd=self.ic.workspace_dir, env=env).stdout
            return patch, [line for line in names.splitlines() if line]

        if not admitted:
            return read_changes()
        with tempfile.TemporaryDirectory(prefix="forge-candidate-index-") as temporary:
            index = Path(self._git("rev-parse", "--git-path", "index"))
            if not index.is_absolute():
                index = Path(self.ic.workspace_dir) / index
            candidate_index = Path(temporary) / "index"
            # Git uses the index mtime to detect same-size edits with unchanged file timestamps.
            shutil.copy2(index, candidate_index)
            env = {"GIT_INDEX_FILE": str(candidate_index)}
            git("add", "--", *admitted, cwd=self.ic.workspace_dir, env=env)
            return read_changes(env)

    def _list_untracked(self) -> list[str]:
        """Every untracked, non-ignored path in the workspace, as git reports it."""
        listed = git_bytes("ls-files", "--others", "--exclude-standard", "-z", cwd=self.ic.workspace_dir, check=False)
        if listed.returncode != 0:
            detail = (listed.stderr or listed.stdout).decode(
                errors="surrogateescape"
            ).strip() or f"git ls-files exited {listed.returncode}"
            raise RuntimeError(f"could not list new files: {detail}")
        return [item.decode(errors="surrogateescape") for item in listed.stdout.split(b"\0") if item]

    def _untracked_snapshot(self) -> set[str] | None:
        """Snapshot the untracked set an iteration or lane starts from."""
        try:
            return set(self._list_untracked())
        except RuntimeError as error:
            log.warning("could not snapshot untracked files: %s", error)
            return None

    def _new_paths(self) -> tuple[list[str], list[str]]:
        """Split the workspace's new files into the shippable ones and the rest."""
        patterns = list(self.ic.commit_new_paths)
        own_roots = [LOOP_ARTIFACT_ROOT]
        if self.ic.build_dir:
            build_dir = self._workspace_path(self.ic.build_dir)
            if not Path(build_dir).is_absolute():
                own_roots.append(build_dir)
        admitted: list[str] = []
        refused: list[str] = []
        for path in sorted(self._list_untracked()):
            if any(path == root or path.startswith(f"{root}/") for root in own_roots):
                continue
            allowed = matches_commit_new_paths(path, patterns) and not FORGE_PROTECTED_PATHS.is_protected(
                path,
                workspace=self.ic.workspace_dir,
                # The campaign driver carries no protected name of its own.
                exact_paths=(self.ic.driver_script,),
            )
            if allowed:
                admitted.append(path)
            else:
                refused.append(path)
        return admitted, refused

    def _new_paths_best_effort(self) -> tuple[list[str], list[str]] | None:
        """``_new_paths`` for callers that are already recovering from a failure."""
        try:
            listed = self._new_paths()
        except RuntimeError as error:
            log.warning("skipping the new-file clean: %s", error)
            print(f"  [git] could not enumerate new files, skipping the new-file clean: {error}")
            # Both callers return early from here without reporting, so a refusal list left standing would be read as
            # this iteration's.
            self._refused_new_paths = []
            self._retained_new_paths = []
            self._new_paths_unreadable = str(error)
            return None
        return listed

    def _report_refused_new_paths(
        self,
        refused: list[str],
        action: str,
        retained: Sequence[str] = (),
    ) -> None:
        """Record and print the new files this ``action`` could not act on."""
        self._refused_new_paths = list(refused)
        self._retained_new_paths = list(retained)
        self._new_paths_unreadable = ""
        if refused:
            print(f"  [git] {len(refused)} new file(s) outside commit_new_paths, not {action}: " + ", ".join(refused))

    def _new_paths_need_discard(self) -> bool:
        """Whether new files alone make a discard necessary, refusals reported."""
        listed = self._new_paths_best_effort()
        if listed is None:
            return False
        admitted, refused = listed
        self._report_refused_new_paths(refused, "removed")
        return bool(admitted)

    def _render_uncommittable_new_paths(self) -> str:
        """Tell the Implementer what the new-file report says this iteration."""
        allowlist = ", ".join(self.ic.commit_new_paths) or "(empty)"
        blocks: list[str] = []
        if self._new_paths_unreadable:
            blocks.append(
                "\n".join(
                    (
                        "## New files could not be listed",
                        (
                            "This iteration could not enumerate the "
                            "workspace's new files "
                            f"({self._new_paths_unreadable}), so nothing "
                            "below reports on them. A file you created may "
                            "be sitting on the measured tree uncommitted "
                            "and unremoved; treat the absence of a new-file "
                            "report as unknown, not as nothing."
                        ),
                    )
                )
            )
        if self._refused_new_paths:
            blocks.append(
                "\n".join(
                    (
                        "## New files that cannot ship",
                        (
                            "A KEEP commits tracked edits plus new files "
                            f"matching {allowlist}. These new files match "
                            "nothing there, so a KEEP cannot carry them and "
                            "a REVERT cannot remove them, and the measured "
                            "tree is not the committed tree while they "
                            "exist: " + ", ".join(self._refused_new_paths)
                        ),
                        (
                            "Put the change in a tracked file, or state in "
                            "your findings which path the operator has to "
                            "allowlist and why the change cannot live in a "
                            "tracked file."
                        ),
                    )
                )
            )
        if self._retained_new_paths:
            blocks.append(
                "\n".join(
                    (
                        "## Allowlisted new files this loop did not create",
                        (
                            f"These match {allowlist} but were already on "
                            "the workspace before this loop touched it, "
                            "so they are the operator's or an earlier "
                            "round's and a REVERT leaves them in place. They "
                            "are on the measured tree without being part of "
                            "any candidate: " + ", ".join(self._retained_new_paths)
                        ),
                        (
                            "If one of them is a leftover of your own work, "
                            "say so in your findings -- its effect is being "
                            "measured and attributed to nothing."
                        ),
                    )
                )
            )
        return "\n\n".join(blocks)

    def _git_commit(self, message: str) -> str:
        """Stage ALL tracked modifications and commit, raising on failure."""
        before = self._git("rev-parse", "HEAD").strip()
        git("add", "-u", cwd=self.ic.workspace_dir)

        # Fail-fast here on purpose: a KEEP built from a file set that could not be enumerated would ship an unknown
        # tree.
        admitted, refused = self._new_paths()
        if admitted:
            git("add", "--", *admitted, cwd=self.ic.workspace_dir)
        self._report_refused_new_paths(refused, "committed")

        git("commit", "-m", message, cwd=self.ic.workspace_dir)

        after = self._git("rev-parse", "HEAD").strip()
        if not after or after == before:
            raise RuntimeError("git commit did not advance HEAD")
        return after

    def _git_revert_last(self) -> None:
        """Revert the last commit, raising when the candidate stays on the tree."""
        git("revert", "--no-edit", "HEAD", cwd=self.ic.workspace_dir)

    def _git_discard_worktree(self) -> None:
        """Discard staged and unstaged tracked edits in the workspace."""
        self._git_discard_all_tracked_changes()

    def _git_discard_all_tracked_changes(self) -> None:
        """Discard an exact pending candidate from both index and worktree."""
        git(
            "restore",
            "--source=HEAD",
            "--staged",
            "--worktree",
            "--",
            ".",
            cwd=self.ic.workspace_dir,
        )

        listed = self._new_paths_best_effort()
        if listed is None:
            return
        admitted, refused = listed
        if self._pre_untracked is None:
            preexisting = []
            if admitted:
                print(f"  [git] no untracked snapshot; removing every allowlisted new file: {', '.join(admitted)}")
        else:
            preexisting = [path for path in admitted if path in self._pre_untracked]
            if preexisting:
                print("  [git] leaving allowlisted file(s) this loop did not create: " + ", ".join(preexisting))
        admitted = [path for path in admitted if path not in preexisting]
        if admitted:
            clean = git(
                "clean",
                "-f",
                "--",
                *admitted,
                cwd=self.ic.workspace_dir,
                check=False,
            )
            if clean.returncode != 0:
                raise RuntimeError(f"git clean failed: {(clean.stderr or clean.stdout).strip()}")
        self._report_refused_new_paths(refused, "removed", preexisting)

    def _git_apply_patch(self, patch: str) -> bool:
        """Apply one archived diff to the working tree, reporting whether it took."""
        if not patch.strip():
            return False
        handle = tempfile.NamedTemporaryFile("w", suffix=".diff", encoding="utf-8", delete=False)

        def _apply(*extra: str) -> bool:
            return (
                git(
                    "apply",
                    *extra,
                    handle.name,
                    cwd=self.ic.workspace_dir,
                    check=False,
                ).returncode
                == 0
            )

        try:
            handle.write(patch if patch.endswith("\n") else patch + "\n")
            handle.close()
            if _apply("--check") and _apply():
                return True
            if _apply("--3way"):
                return True
            self._git_discard_all_tracked_changes()
            return False
        finally:
            Path(handle.name).unlink(missing_ok=True)

    def _read_source_file(self, path: str) -> str:
        """Read a source file's current content (best-effort)."""
        try:
            p = Path(path)
            if not p.is_absolute():
                p = Path(self.ic.workspace_dir) / p
            return p.read_text()
        except OSError as e:
            log.debug("could not read source file %s: %s", path, e)
            return ""

    def _read_kernel_source(self) -> str:
        """Read the anchor kernel file's current on-disk content (best-effort)."""
        return self._read_source_file(self.ic.kernel_file)

    def _kernel_source_for_scope(self) -> list[str | None] | None:
        """Every declared source file's text, None per file that would not read."""
        texts: list[str | None] = []
        unreadable: list[str] = []
        for declared in self._target_source_files():
            path = Path(declared)
            if not path.is_absolute():
                path = Path(self.ic.workspace_dir) / path
            try:
                texts.append(path.read_text())
            except (OSError, ValueError) as e:  # reported, not swallowed
                texts.append(None)
                unreadable.append(str(path))
                log.warning(
                    "lessons: could not read %s; it will not be checked for held-fixed premises this round: %s", path, e
                )
        listed = ", ".join(unreadable)
        if any(text is not None for text in texts):
            if unreadable:
                # Printed, not only logged: a premise checked against part of the declared source is a weaker check
                # than the note reads as.
                print(f"  [lesson] source unreadable ({listed}): held-fixed premises checked against the rest only")
            return texts
        print(f"  [lesson] kernel source unreadable ({listed}): held-fixed premises not checked this round")
        return None

    def _target_source_files(self) -> list[str]:
        """Declared implementation hints, anchor first, de-duplicated."""
        files: list[str] = []
        for f in [self.ic.kernel_file, *self.ic.source_files]:
            if f and f not in files:
                files.append(f)
        return files

    def _jit_source_files(self) -> list[str]:
        """Declared hints plus actual tracked edits that may require recompilation."""

        workspace = getattr(
            self.ic,
            "workspace_dir",
            str(Path(self.ic.kernel_file).resolve().parent),
        )
        return list(
            dict.fromkeys(
                [
                    *self._target_source_files(),
                    *tracked_source_changes(workspace),
                ]
            )
        )

    def _full_diff(self, commit_hash: str) -> str:
        """Full unified diff of one iteration's commit (all files it touched)."""
        if not commit_hash:
            return ""
        try:
            return git("diff", f"{commit_hash}~1", commit_hash, cwd=self.ic.workspace_dir).stdout
        except (OSError, subprocess.SubprocessError) as e:
            log.debug("could not diff commit %s: %s", commit_hash, e)
            return ""

    def _working_tree_diff(self) -> str:
        """Full diff of the current staged/unstaged candidate relative to HEAD."""
        return git("diff", "HEAD", "--", ".", cwd=self.ic.workspace_dir).stdout

    def _diff_summary_from_diff(self, diff: str, max_lines: int = 8) -> str:
        """Compact summary from an already-captured unified diff."""
        if not diff:
            return ""
        import re as _re

        stat_lines = [f"{name} | changed" for name in touched_paths(diff)[:4]]
        signal = _re.compile(
            r"BLOCK_|VEC_|WARP|WAVE|tile|fastmath|const_expr|num_stage|num_warp|"
            r"occupancy|def |return |Vec\(|\.to\(|=|if ",
            _re.IGNORECASE,
        )
        changed: list[str] = []
        for ln in diff.splitlines():
            if ln[:3] in ("+++", "---"):
                continue
            if ln[:1] in "+-":
                content = ln[1:].strip()
                if not content or content.startswith("#"):
                    continue
                if signal.search(content):
                    changed.append(f"{ln[0]} {content[:100]}")
            if len(changed) >= max_lines:
                break
        parts: list[str] = []
        if stat_lines:
            parts.append("files: " + "; ".join(stat_lines[:4]))
        parts.extend(changed[:max_lines])
        return "\n".join(parts)

    def _diff_summary(self, commit_hash: str, max_lines: int = 8) -> str:
        """Mechanical, loop-authored summary of one iteration's NET change."""
        if not commit_hash:
            return ""
        diff = self._git("diff", f"{commit_hash}~1", commit_hash, "--", ".")
        return self._diff_summary_from_diff(diff, max_lines=max_lines)

    def _commit_changed_files(self, commit_hash: str) -> list[str]:
        """Tracked paths changed by one verified KEEP commit."""
        if not commit_hash:
            return []
        output = self._git(
            "diff",
            "--name-only",
            f"{commit_hash}~1",
            commit_hash,
            "--",
            ".",
        )
        return [line.strip() for line in output.splitlines() if line.strip()]
