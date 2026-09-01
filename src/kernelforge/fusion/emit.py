# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Stage 5: export the authored change as a JSON change-manifest + a git patch.

The Hyperloom handoff: besides the fused-kernel file(s), a source-level fusion also
edits the framework model file (wiring the fused path in behind the env gate). This
captures BOTH: a single ``fusion.patch`` (git diff of the framework repo) and a
per-file change list classifying each path as a new kernel vs a framework-wiring
edit, so the caller can apply/review deterministically.
"""

from __future__ import annotations

import contextlib
import difflib
import logging
import subprocess
from pathlib import Path

from .models import FusionArtifacts
from kernelforge.llm.git import git

log = logging.getLogger("forge_fusion")


def _git(repo: str, *args: str, timeout: int = 60) -> subprocess.CompletedProcess:
    return git("-C", repo, *args, check=False, timeout=timeout)


def _is_git_repo(repo_root: str) -> bool:
    """True when ``repo_root`` is inside a git work tree (so git diff/checkout work)."""
    if not repo_root:
        return False
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        r = _git(repo_root, "rev-parse", "--is-inside-work-tree", timeout=30)
        return r.returncode == 0 and r.stdout.strip() == "true"
    return False


# Word-boundary aware so we do NOT match unrelated framework files such as
# ``diffusion*.py`` / ``confusion*.py`` (they contain the bare substring
# "fusion" mid-word but are not author-created fusion kernels).
_FUSED_MODULE_MARKERS = ("_fused", "_fusion")
_FUSED_MODULE_PREFIXES = ("fused", "fusion")


def _is_fused_module_name(name: str) -> bool:
    """Whether ``name`` marks an author-created fused-kernel module.

    Matches ``*_fused*``/``*_fusion*`` (underscore-bounded) or a stem starting with
    ``fused``/``fusion`` (e.g. ``fusion_helper.py``, ``fused_moe.py``), but NOT
    ``diffusion.py``/``confusion.py`` where "fusion" is only a mid-word substring.
    """
    stem = Path(name).stem
    if any(m in name for m in _FUSED_MODULE_MARKERS):
        return True
    return any(stem == p or stem.startswith(p + "_") for p in _FUSED_MODULE_PREFIXES)


def _git_tracks(repo_root: str, source_file: str) -> bool:
    """True only when ``source_file`` is a git-TRACKED file under ``repo_root``.

    Broader-correct than ``_is_git_repo``: a pip-installed framework can live under
    a git work tree (e.g. a project-local ``.venv``/``site-packages``) yet be
    untracked, so ``git diff`` is empty. In that case the snapshot (non-git) path
    must be taken, not the git path.
    """
    if not repo_root or not source_file:
        return False
    try:
        rel = str(Path(source_file).resolve().relative_to(Path(repo_root).resolve()))
    except ValueError:
        return False
    with contextlib.suppress(OSError, subprocess.SubprocessError):
        r = _git(repo_root, "ls-files", "--error-unmatch", "--", rel, timeout=30)
        return r.returncode == 0
    return False


def _unified_file_diff(rel: str, old_text: str, new_text: str) -> str:
    """git-apply-compatible unified diff for one file (empty when unchanged)."""
    if old_text == new_text:
        return ""
    body = "".join(
        difflib.unified_diff(
            old_text.splitlines(keepends=True),
            new_text.splitlines(keepends=True),
            fromfile=f"a/{rel}",
            tofile=f"b/{rel}",
        )
    )
    if not body:
        return ""
    # `diff --git` header keeps it applyable by both `git apply` and `patch -p1`.
    return f"diff --git a/{rel} b/{rel}\n{body}"


def _export_nongit(repo_root: str, source_file: str, out: Path, pristine_dir: Path) -> FusionArtifacts:
    """Export ``fusion.patch`` without git, using a pre-authoring pristine snapshot.

    Needed when the framework is a plain pip install (not a git checkout), where
    ``git diff`` yields nothing so the KEPT fusion would otherwise ship
    ``patch=null`` and never reach e2e integrate. Diffs the snapshot vs the live
    edited source (unified diff); new ``*_fused*`` / ``*fusion*`` modules beside it
    are emitted as whole-file additions.
    """
    arts = FusionArtifacts()
    root = Path(repo_root).resolve() if repo_root else None
    parts: list[str] = []
    names: list[str] = []

    def _rel(p: Path) -> str:
        # POSIX separators always: this string is interpolated straight into the
        # ``diff --git a/<rel>`` header, and git rejects a backslash path as
        # "invalid path" on every platform, so a Windows-side export would
        # otherwise produce a patch nobody can apply.
        if root:
            with contextlib.suppress(ValueError):
                return p.resolve().relative_to(root).as_posix()
        return p.name

    # 1) edited model source: pristine snapshot vs current.
    if source_file and Path(source_file).is_file():
        rel = _rel(Path(source_file))
        snap = pristine_dir / rel
        old_text = snap.read_text(encoding="utf-8", errors="replace") if snap.is_file() else ""
        new_text = Path(source_file).read_text(encoding="utf-8", errors="replace")
        d = _unified_file_diff(rel, old_text, new_text)
        if d:
            parts.append(d)
            names.append(rel)

    # 2) fused modules beside the source: diff snapshot-vs-current. A pre-existing
    #    framework file (snapshotted, unchanged) yields an empty diff and is NOT
    #    emitted; an author-created module has no snapshot so its whole content is
    #    the "new file" add. This avoids emitting/deleting unrelated framework files
    #    that merely match the *_fused*/*fusion* glob.
    src_resolved = Path(source_file).resolve() if source_file else None
    model_dir = Path(source_file).parent if source_file else None
    if model_dir and model_dir.is_dir():
        for f in sorted(model_dir.glob("*.py")):
            name = f.name
            if not _is_fused_module_name(name):
                continue
            if src_resolved is not None and f.resolve() == src_resolved:
                continue  # the edited source is handled by (1)
            rel = _rel(f)
            snap = pristine_dir / rel
            old_text = snap.read_text(encoding="utf-8", errors="replace") if snap.is_file() else ""
            new_text = f.read_text(encoding="utf-8", errors="replace")
            d = _unified_file_diff(rel, old_text, new_text)
            if d:
                parts.append(d)
                names.append(rel)

    diff = "\n".join(p.rstrip("\n") for p in parts if p)
    if diff:
        patch_path = out / "fusion.patch"
        patch_path.write_text(diff.rstrip("\n") + "\n", encoding="utf-8")
        arts.patch = str(patch_path)
    arts.changes = [{"path": n, "kind": _classify(n, source_file)} for n in names]
    if arts.patch:
        arts.repo_root = str(root) if root else ""
    log.info(
        "exported %d fusion file(s) (non-git); patch=%s repo_root=%s", len(arts.changes), arts.patch, arts.repo_root
    )
    return arts


def _tracked_paths(repo_root: str, rel_paths: list[str]) -> set[str]:
    """Return the subset of ``rel_paths`` already tracked by git."""
    if not rel_paths:
        return set()
    out = _git(repo_root, "ls-files", "--", *rel_paths).stdout.split()
    return set(out)


def _classify(rel_path: str, source_file: str) -> str:
    """Classify a changed file for the handoff manifest."""
    name = Path(rel_path).name
    if _is_fused_module_name(name):
        return "new_kernel"
    if source_file and Path(source_file).name == name:
        return "framework_wiring_edit"
    return "framework_wiring_edit"


def _fusion_scoped_paths(repo_root: str, source_file: str) -> list[str]:
    """Repo-relative paths that belong to THIS fusion (not the whole dirty tree).

    Scopes the exported patch to: the edited model source file, plus any untracked
    new module in the SAME directory whose name marks it a fused kernel
    (``*_fused*`` / ``*fusion*``). This avoids the earlier whole-repo ``git diff``
    that swept in dozens of unrelated pre-existing dirty files.
    """
    root = Path(repo_root).resolve()
    paths: list[str] = []
    if source_file:
        with contextlib.suppress(ValueError):
            # POSIX form to match what git itself reports, so the manifest's
            # changed-file paths are comparable across platforms.
            paths.append(Path(source_file).resolve().relative_to(root).as_posix())
    # Untracked fused-kernel modules next to the source file.
    model_dir = Path(source_file).parent if source_file else root
    others = _git(repo_root, "ls-files", "--others", "--exclude-standard").stdout.split()
    for rel in others:
        name = Path(rel).name
        if _is_fused_module_name(name) and (root / rel).parent == model_dir.resolve():
            paths.append(rel)
    # De-dupe, keep order.
    seen: set[str] = set()
    return [p for p in paths if not (p in seen or seen.add(p))]


def export_artifacts(
    repo_root: str,
    source_file: str,
    out_dir: str | Path,
    pristine_dir: str | Path | None = None,
    snapshot_diff_only: bool = False,
) -> FusionArtifacts:
    """Export ``fusion.patch`` + a classified change list, scoped to the fusion.

    Best-effort: returns an empty ``FusionArtifacts`` when the repo is unavailable
    or there are no fusion-scoped changes. Only the fusion files (the edited model
    source + new fused modules beside it) are diffed, NOT the whole repo.

    When the framework source is NOT a git checkout (e.g. a plain pip install), git
    diff yields nothing, so fall back to diffing a pre-authoring ``pristine_dir``
    snapshot — otherwise a KEPT fusion would ship ``patch=null`` and never reach
    e2e integrate.

    ``snapshot_diff_only`` forces the snapshot route even for a tracked file. The
    git route diffs against HEAD, so on a checkout carrying unrelated uncommitted
    edits it would sweep them into the patch; callers that must ship exactly the
    change THIS run made (the compile-pass flip) require the snapshot baseline.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    arts = FusionArtifacts()

    def _nongit() -> FusionArtifacts | None:
        if pristine_dir and source_file:
            return _export_nongit(repo_root, source_file, out, Path(pristine_dir))
        return None

    # Take the git path ONLY when the source file is actually git-TRACKED. A pip
    # install can sit under a git work tree (project-local venv/site-packages) yet
    # be untracked, so `git diff` would be empty and ship patch=null. In that case
    # fall through to the pristine-snapshot path instead.
    if snapshot_diff_only or not (_is_git_repo(repo_root) and _git_tracks(repo_root, source_file)):
        return _nongit() or arts
    if not repo_root:
        return arts

    try:
        rel_paths = _fusion_scoped_paths(repo_root, source_file)
        if not rel_paths:
            return arts
        tracked = _tracked_paths(repo_root, rel_paths)
        parts: list[str] = []
        names: list[str] = []
        tracked_paths = [p for p in rel_paths if p in tracked]
        if tracked_paths:
            parts.append(_git(repo_root, "diff", "--", *tracked_paths).stdout)
            names.extend(_git(repo_root, "diff", "--name-only", "--", *tracked_paths).stdout.split())
        for rel in rel_paths:
            if rel in tracked or not (Path(repo_root) / rel).is_file():
                continue
            cp = _git(repo_root, "diff", "--no-index", "--", "/dev/null", rel)
            if cp.stdout:
                parts.append(cp.stdout)
                names.append(rel)
        diff = "\n".join(p.rstrip("\n") for p in parts if p)
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("artifact export failed: %s", exc)
        return arts

    if not diff:
        # Tracked-but-empty (edits reverted, or CRLF/whitespace-only churn git
        # ignores): try the pristine snapshot before giving up on the patch.
        return _nongit() or arts

    patch_path = out / "fusion.patch"
    patch_path.write_text(diff.rstrip("\n") + "\n", encoding="utf-8")
    arts.patch = str(patch_path)
    arts.repo_root = str(Path(repo_root).resolve())
    arts.changes = [{"path": n, "kind": _classify(n, source_file)} for n in names]
    log.info("exported %d fusion file(s); patch=%s repo_root=%s", len(arts.changes), arts.patch, arts.repo_root)
    return arts


def restore_exported_changes(
    repo_root: str,
    artifacts: FusionArtifacts,
    pristine_dir: str | Path | None = None,
) -> None:
    """Restore live framework repo changes after a successful export.

    forge-fuse is an author/export tool; Hyperloom is responsible for applying
    the emitted patch during e2e integrate. Leaving authored bytes in the live
    framework repo lets later explore rounds consume them without attribution.

    Non-git framework (pip install): git checkout cannot revert, so restore each
    edited file from the pre-authoring ``pristine_dir`` snapshot (and delete new
    fused modules that have no snapshot).
    """
    if not repo_root or not artifacts.patch:
        return
    is_git = _is_git_repo(repo_root)
    pdir = Path(pristine_dir) if pristine_dir else None

    def _restore_nongit(rel: str) -> None:
        """Restore from pristine snapshot, else unlink (author-created new module)."""
        live = Path(repo_root) / rel
        snap = pdir / rel if pdir else None
        with contextlib.suppress(OSError):
            if snap and snap.is_file():
                live.write_text(snap.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
            elif pdir is not None:
                live.unlink(missing_ok=True)

    for change in artifacts.changes:
        rel = str(change.get("path") or "")
        if not rel:
            continue
        # Per-file: only git-checkout files git actually TRACKS. A pip framework
        # under a git work tree (venv in a git project) is untracked, so restore it
        # from the pristine snapshot instead of deleting it via the git branch.
        if is_git and _git(repo_root, "ls-files", "--error-unmatch", rel).returncode == 0:
            _git(repo_root, "checkout", "--", rel)
            continue
        if pdir is not None:
            _restore_nongit(rel)
            continue
        path = Path(repo_root) / rel
        try:
            path.unlink(missing_ok=True)
            # Best-effort prune empty directories left by fused helper modules.
            parent = path.parent
            root = Path(repo_root).resolve()
            while parent.resolve() != root:
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
        except OSError as exc:
            log.warning("could not remove exported untracked fusion file %s: %s", path, exc)
