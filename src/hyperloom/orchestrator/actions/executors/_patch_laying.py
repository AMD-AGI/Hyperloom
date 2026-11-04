# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""How a TraceLens patch plan is laid: in this host's copy of the installed package, in a test's farm, or in place.

vLLM and SGLang are never patched where they are installed: every test and session on the host imports that tree. The
patch is laid in a copy of the package instead, on this host's local disk, and the copy goes first on the ``PYTHONPATH``
of the server that profiles (:class:`PatchedServer`). The copy mirrors the package the way a test's farm does: a
directory the patch does not reach is a link to the installed one, a file it changes is a patched copy. A copy is built
once per patch set and installed package, in a directory of its own that is complete before it is published and whose
files are never changed after (Python adds its bytecode caches), so profiles running at once share it and none can take
it from under another. A published copy is taken only when this user laid it and no one else can write it: the host's
temporary directory is shared, and a server imports whatever stands under that name. A test's farm that rebuilt the
framework gets the patch laid in its own files the same way.

The set is applied as a transaction either way: ``git apply`` per patch, a fuzzy ``patch`` where the plan allows it, and
the applied patches rolled back when one fails or the sentinels say the set did not land.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import Callable, Sequence

from hyperloom_common.unified_diff import parse_unified_diff, strip_path_components

from ._file_lock import best_effort_file_lock

log = logging.getLogger(__name__)


# System-wide lock file.
_LOCK_PATH = str(Path(tempfile.gettempdir()) / "hyperloom_server_patcher.lock")

#: The directory, under this host's temporary directory, holding one patched copy per patch set and installed package.
_COPIES_DIR = "hyperloom-tracelens"
#: The directory of a copy that goes on the server's ``PYTHONPATH``: it holds the patched package and nothing else.
_SITE_DIR = "site"
_CACHE_DIR = "__pycache__"

# Per-``git`` invocation timeout (defensive against hung NFS).
_GIT_TIMEOUT_SEC = 30
_UTF8 = "utf-8"
#: How a patch or a tool's output that is not valid UTF-8 is read: what it says is kept, the rest replaced.
_LENIENT = "replace"


@dataclass(frozen=True)
class PatchedServer:
    """TraceLens's patch for the server a profile launches.

    ``site`` holds this host's patched copy of the framework package and nothing else; it goes first on the server's
    ``PYTHONPATH``. ``None`` when this host lays no copy: the pods of a multi-node run patch their own installs.
    """

    site: Path | None


def is_patched_copy(entry: str) -> bool:
    """Whether the ``PYTHONPATH`` entry ``entry`` is the site of a :class:`PatchedServer`, laid on any host."""
    path = PurePosixPath(entry)
    return path.name == _SITE_DIR and path.parent.parent.name == _COPIES_DIR


@dataclass(frozen=True)
class _PatchPlan:
    """All information needed to apply (or verify) a patch set."""

    framework: str
    version: str
    package: Path  # the package directory; every path the patch set writes lies in it
    patches: tuple[Path, ...]  # in apply order
    sentinel_file: Path  # file we grep to detect "already patched"
    # Substrings that must ALL be present in ``sentinel_file`` to count as patched; multi-element tuples lower the
    # false-positive risk.
    sentinel_text: tuple[str, ...]
    # Extra file-local markers required for a multi-file set to count as complete (SGLang can partial-patch while
    # leaving the historical sentinel).
    extra_sentinels: tuple[tuple[Path, tuple[str, ...]], ...] = ()
    # The ``-p<N>`` that makes the patch's paths relative to ``apply_root``.
    apply_strip: int = 1
    # Patch names that may fail ``git apply --check`` and be skipped instead of rolling back the whole atomic set
    # (e.g. eagle-draft patches whose context drifted across same-version different-commit sglang builds).
    optional_patches: frozenset[str] = frozenset()
    # Whether a patch that fails ``git apply --check`` may fall back to ``patch --fuzz``. Off for sets whose hunks
    # add unconditional calls into hot paths, where a partially-matching apply can break the server.
    allow_fuzzy: bool = True

    @property
    def apply_root(self) -> Path:
        """Where the patch set is applied: the directory holding the package."""
        return self.package.parent


# Application core


def _ensure_patched(plan: _PatchPlan) -> PatchedServer | None:
    """This host's copy of ``plan.package`` with the patch laid in: the one laid before, else one laid now."""
    try:
        done = Path(tempfile.gettempdir()) / _COPIES_DIR / f"{plan.framework}-{_copy_key(plan)}"
        if _is_published(plan, done):
            return PatchedServer(done / _SITE_DIR)
        with best_effort_file_lock(_LOCK_PATH, label="_server_patcher"):
            if _is_published(plan, done):
                return PatchedServer(done / _SITE_DIR)
            built = _build_copy(plan, done)
    except OSError as e:
        log.warning("_server_patcher: cannot lay the %s %s patch in a copy (%s); skip", plan.framework, plan.version, e)
        return None
    return None if built is None else PatchedServer(built / _SITE_DIR)


def _is_published(plan: _PatchPlan, done: Path) -> bool:
    """Whether ``done`` is a complete copy this user published: its sentinels say so only of what it holds, and anyone
    who can write the host's temporary directory could have put it there."""
    return _owned(done) and _is_patched(_in_copy(plan, done / _SITE_DIR))


def _owned(directory: Path) -> bool:
    """Whether ``directory`` is a directory of this user's, not a link, that its group and others cannot write."""
    try:
        info = directory.lstat()
    except OSError:
        return False
    return (
        stat.S_ISDIR(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & (stat.S_IWGRP | stat.S_IWOTH)
    )


def _copy_key(plan: _PatchPlan) -> str:
    """What a copy is made of: the installed package, the patch set and how it applies, and the installed files and
    directories the copy replaces with its own."""
    digest = hashlib.sha256()

    def add(*parts: bytes) -> None:
        for part in parts:
            digest.update(len(part).to_bytes(8, "big") + part)

    add(str(plan.package).encode(), str(plan.apply_strip).encode(), str(plan.allow_fuzzy).encode())
    add(*(name.encode() for name in sorted(plan.optional_patches)))
    for patch in plan.patches:
        add(patch.name.encode(), patch.read_bytes())
    installed = plan.package.parent
    directories: set[PurePosixPath] = set()
    for rel in _site_targets(plan):
        path = installed / rel
        add(rel.encode(), path.read_bytes() if path.is_file() else b"")
        directories.update(PurePosixPath(rel).parents[:-1])
    for directory in sorted(directories):
        path = installed / directory
        add(str(directory).encode(), *(name.encode() for name in (sorted(os.listdir(path)) if path.is_dir() else ())))
    return digest.hexdigest()[:32]


def _build_copy(plan: _PatchPlan, done: Path) -> Path | None:
    """Lay the copy in a directory of its own beside ``done`` and publish it as ``done``; the directory it is in."""
    done.parent.mkdir(parents=True, exist_ok=True)
    building = Path(tempfile.mkdtemp(prefix=f".{done.name}-", dir=done.parent))
    kept: Path | None = None
    try:
        building.chmod(0o755)
        _mirror(plan.package, building / _SITE_DIR / plan.package.name)
        if _lay_in(plan, building / _SITE_DIR):
            kept = _publish(plan, building, done)
    finally:
        if kept != building:
            shutil.rmtree(building, ignore_errors=True)
    return kept


def _publish(plan: _PatchPlan, building: Path, done: Path) -> Path:
    """``building`` renamed to ``done``; ``done`` when another process of this user published it first; ``building``
    kept under its own private name when ``done`` holds no complete copy of this user's, which a server may be importing
    from, so it is never removed or replaced."""
    try:
        building.rename(done)
    except OSError as e:
        if _is_published(plan, done):
            return done
        log.warning(
            "_server_patcher: %s holds no complete copy laid by this user (%s); this profile uses %s", done, e, building
        )
        return building
    return done


def _lay_in(plan: _PatchPlan, site: Path) -> bool:
    """Apply ``plan`` to the mirror of its package in ``site``, each path it writes made the mirror's own first."""
    copy = _in_copy(plan, site)
    if _is_patched(copy):
        return True
    targets = _site_targets(plan)
    outside = [target for target in targets if PurePosixPath(target).parts[0] != plan.package.name]
    if outside:
        log.warning("_server_patcher: %s patch writes outside its package: %s; skip", plan.framework, outside)
        return False
    try:
        for target in targets:
            _unshare(site, target)
    except OSError as e:
        log.warning(
            "_server_patcher: cannot copy the files the %s patch writes into %s (%s); skip", plan.framework, site, e
        )
        return False
    return _apply_atomic(copy)


def _site_targets(plan: _PatchPlan) -> list[str]:
    """Every path the patch set writes, relative to the directory holding the package."""
    # The parsed paths have already lost their a/ and b/ prefixes, the first component ``-p`` strips.
    level = plan.apply_strip - 1
    return sorted(
        {
            strip_path_components(change.path, level)
            for patch in plan.patches
            for change in parse_unified_diff(patch.read_text(encoding=_UTF8, errors=_LENIENT))
        }
    )


def _in_copy(plan: _PatchPlan, site: Path) -> _PatchPlan:
    """``plan`` for the mirror of its package in ``site``: applied there, and checked by its sentinels there."""
    installed = plan.package.parent

    def moved(path: Path) -> Path:
        return site / path.relative_to(installed)

    return replace(
        plan,
        package=site / plan.package.name,
        sentinel_file=moved(plan.sentinel_file),
        extra_sentinels=tuple((moved(path), markers) for path, markers in plan.extra_sentinels),
    )


def _mirror(source: Path, target: Path) -> None:
    """Make ``target`` a directory whose entries link to those of ``source``, its bytecode cache left out (Python
    writes the copy's own beside it)."""
    target.mkdir(parents=True)
    for entry in source.iterdir():
        if entry.name != _CACHE_DIR:
            (target / entry.name).symlink_to(entry)


def _unshare(site: Path, rel: str) -> None:
    """Make ``site/rel`` the mirror's own: each directory on its way a real one, and the file, when it is a link, a
    copy of what it links to."""
    here = site
    for name in PurePosixPath(rel).parts[:-1]:
        here = here / name
        if here.is_symlink():
            source = here.resolve(strict=True)
            here.unlink()
            _mirror(source, here)
    target = site / rel
    if target.is_symlink():
        source = target.resolve(strict=True)
        target.unlink()
        shutil.copy2(source, target)


def _ensure_patched_in_place(plan: _PatchPlan) -> bool:
    """Drive the installed tree of ``plan`` to patched state: fast check, lock, re-check, apply."""
    if _is_patched(plan):
        return True
    with best_effort_file_lock(_LOCK_PATH, label="_server_patcher"):
        if _is_patched(plan):
            return True
        return _apply_atomic(plan)


def _markers_present(path: Path, markers: Sequence[str]) -> bool:
    """True iff ``path`` exists and contains every marker substring."""
    try:
        if not path.exists():
            return False
        content = path.read_text(encoding=_UTF8, errors=_LENIENT)
    except OSError:
        return False
    return all(marker in content for marker in markers)


def _is_patched(plan: _PatchPlan) -> bool:
    """True iff the sentinel file (and every extra_sentinel) exists with all of its marker substrings present."""
    if not _markers_present(plan.sentinel_file, plan.sentinel_text):
        return False
    # The plan only keeps sentinels this patch set writes, so an absent file is an incomplete apply, not an
    # inapplicable layout.
    return all(_markers_present(path, markers) for path, markers in plan.extra_sentinels)


def _apply_atomic(plan: _PatchPlan) -> bool:
    """Apply every patch in ``plan.patches`` as a transaction."""
    git = shutil.which("git")
    if git is None:
        log.warning(
            "_server_patcher: `git` not on PATH — cannot apply %s patches "
            "(install git in the runtime image to enable TraceLens flags)",
            plan.framework,
        )
        return False
    # ``None`` disables the fuzzy fallback (no ``patch`` binary, or a strict-only plan).
    patch_bin = shutil.which("patch") if plan.allow_fuzzy else None

    # Per-patch precheck: each must pass ``git apply --check`` (strict) OR the fuzzy ``patch --fuzz=2 --dry-run``
    # fallback; if neither accepts a patch the whole set fail-softs.
    strip_arg = f"-p{plan.apply_strip}"

    apply_modes: dict[Path, str] = {}
    for p in plan.patches:
        if _git(git, ("apply", "--check", strip_arg, str(p)), plan.apply_root):
            apply_modes[p] = "git"
            continue
        if patch_bin and _patch_dry_run(patch_bin, p, plan.apply_root, plan.apply_strip):
            log.warning(
                "_server_patcher: %s patch %s did not apply cleanly with "
                "`git apply --check %s`; falling back to `patch %s --fuzz=2` "
                "(TraceLens patch may lag deployed %s by a point release)",
                plan.framework,
                p.name,
                strip_arg,
                strip_arg,
                plan.framework,
            )
            apply_modes[p] = "patch"
            continue
        # Forward apply fails: check whether this patch is ALREADY APPLIED (a clean ``git apply -R --check`` succeeds
        # iff the tree already contains its post-image).
        if _git(git, ("apply", "-R", "--check", strip_arg, str(p)), plan.apply_root):
            log.info(
                "_server_patcher: %s patch %s already applied (reverse "
                "`git apply -R --check %s` is clean); skipping it from the "
                "transaction so the remaining patches still apply",
                plan.framework,
                p.name,
                strip_arg,
            )
            apply_modes[p] = "skip"
            continue
        # Symmetric check for patches previously applied via fuzzy `patch` (git reverse may fail on fuzzy-applied
        # hunks).
        if patch_bin and _patch_reverse_dry_run(
            patch_bin,
            p,
            plan.apply_root,
            plan.apply_strip,
        ):
            log.info(
                "_server_patcher: %s patch %s already applied (fuzzy reverse dry-run clean); skipping it",
                plan.framework,
                p.name,
            )
            apply_modes[p] = "skip"
            continue
        if p.name in plan.optional_patches:
            log.warning(
                "_server_patcher: optional %s patch %s does not apply "
                "(`git apply --check %s` + fuzzy both failed, not already "
                "applied) — skipping it so the critical patches still apply "
                "atomically (version %s; non-critical to kernel shape profiling)",
                plan.framework,
                p.name,
                strip_arg,
                plan.version,
            )
            apply_modes[p] = "skip"
            continue
        log.warning(
            "_server_patcher: `git apply --check %s` AND fuzzy `patch %s "
            "--dry-run` both failed for %s (version %s, patch %s), and it is "
            "not already applied (reverse check failed); fail-soft skip",
            strip_arg,
            strip_arg,
            plan.framework,
            plan.version,
            p.name,
        )
        return False

    applied: list[tuple[Path, str]] = []
    skipped = 0
    for p in plan.patches:
        mode = apply_modes[p]
        if mode == "skip":
            # Already applied; nothing to do and nothing to roll back.
            skipped += 1
            continue
        if mode == "git":
            ok = _git(git, ("apply", strip_arg, str(p)), plan.apply_root)
        else:
            ok = _patch_apply(
                patch_bin,
                p,
                plan.apply_root,
                plan.apply_strip,  # type: ignore[arg-type]
            )
        if ok:
            applied.append((p, mode))
            continue
        log.error(
            "_server_patcher: %s patch %s failed during apply after "
            "passing precheck (mode=%s); rolling back %d previously-applied "
            "patches",
            plan.framework,
            p.name,
            mode,
            len(applied),
        )
        _rollback_applied(applied, plan, git, patch_bin, strip_arg)
        return False

    fuzzy_count = sum(1 for _, mode in applied if mode == "patch")
    log.info(
        "_server_patcher: applied %d patch(es) for %s %s "
        "(strict=%d, fuzzy=%d, already-applied/skipped=%d) (issue #194 §4/§5)",
        len(applied),
        plan.framework,
        plan.version,
        len(applied) - fuzzy_count,
        fuzzy_count,
        skipped,
    )
    # Post-apply sentinel verification: confirm all sentinel markers are present (catches all-skipped,
    # semantically-wrong fuzzy apply, or missing annotation-pipeline markers).
    if not _is_patched(plan):
        log.error(
            "_server_patcher: post-apply sentinel check FAILED for %s %s — "
            "patches were applied/skipped (%d applied, %d skipped) but the "
            "sentinel markers are not all present; rolling back %d "
            "applied patch(es) so the framework tree matches the reported "
            "failure (no half-patched fallback)",
            plan.framework,
            plan.version,
            len(applied),
            skipped,
            len(applied),
        )
        _rollback_applied(applied, plan, git, patch_bin, strip_arg)
        return False
    return True


def _rollback_applied(
    applied: list[tuple[Path, str]],
    plan: _PatchPlan,
    git: str,
    patch_bin: str | None,
    strip_arg: str,
) -> None:
    """Reverse-apply ``applied`` patches (newest first) to restore the tree."""
    for prev, prev_mode in reversed(applied):
        if prev_mode == "git":
            rolled_back = _git(
                git,
                ("apply", "-R", strip_arg, str(prev)),
                plan.apply_root,
            )
        else:
            rolled_back = patch_bin is not None and _patch_apply(
                patch_bin,
                prev,
                plan.apply_root,
                plan.apply_strip,
                reverse=True,
            )
        if not rolled_back:
            log.error(
                "_server_patcher: rollback of %s (mode=%s) also failed — "
                "install may be in inconsistent state; manual review "
                "required",
                prev.name,
                prev_mode,
            )


# fuzz=2 tolerates whitespace / single-line drift but rejects multi-line drift hard (a higher fuzz could apply hunks
# to a semantically wrong location).
_FUZZ = 2


@dataclass(frozen=True)
class _PatchMode:
    """One ``patch`` invocation variant: its extra flags, the label its log lines carry, and how failures are logged.

    ``spawn_log`` reports a failure to spawn; ``log_stderr`` also logs a non-zero exit's stderr.
    """

    flags: tuple[str, ...]
    label: str
    spawn_log: Callable[..., None]
    log_stderr: bool = False


def _run_patch(
    patch_bin: str,
    patch_file: Path,
    cwd: Path,
    mode: _PatchMode,
    *,
    strip: int = 1,
) -> bool:
    """Run ``patch -p<strip> --fuzz=2 <mode.flags>`` with ``patch_file`` on stdin."""
    extra_flags = mode.flags
    spawn_log = mode.spawn_log
    log_stderr = mode.log_stderr
    reverse_label = mode.label
    try:
        with patch_file.open("rb") as fh:
            result = subprocess.run(
                (patch_bin, f"-p{strip}", f"--fuzz={_FUZZ}", *extra_flags),
                cwd=str(cwd),
                stdin=fh,
                capture_output=True,
                timeout=_GIT_TIMEOUT_SEC,
            )
    except (OSError, subprocess.TimeoutExpired) as e:
        spawn_log(
            "_server_patcher: patch%s in %s failed to spawn (%s)",
            reverse_label,
            cwd,
            e,
        )
        return False
    if result.returncode != 0:
        if log_stderr:
            err = result.stderr.decode(_UTF8, errors=_LENIENT)[:500] if result.stderr else ""
            log.debug(
                "_server_patcher: patch%s rc=%d stderr=%r",
                reverse_label,
                result.returncode,
                err,
            )
        return False
    return True


def _patch_dry_run(
    patch_bin: str,
    patch_file: Path,
    cwd: Path,
    strip: int = 1,
) -> bool:
    """Probe ``patch -p<strip> --fuzz=2 --dry-run`` for a single patch."""
    return _run_patch(
        patch_bin,
        patch_file,
        cwd,
        _PatchMode(("--dry-run", "--silent"), " --dry-run", log.warning),
        strip=strip,
    )


def _patch_reverse_dry_run(
    patch_bin: str,
    patch_file: Path,
    cwd: Path,
    strip: int = 1,
) -> bool:
    """Probe ``patch -R -p<strip> --fuzz=2 --dry-run`` for a single patch."""
    return _run_patch(
        patch_bin,
        patch_file,
        cwd,
        _PatchMode(("-R", "--dry-run", "--silent"), " -R --dry-run", log.debug),
        strip=strip,
    )


def _patch_apply(
    patch_bin: str,
    patch_file: Path,
    cwd: Path,
    strip: int = 1,
    *,
    reverse: bool = False,
) -> bool:
    """Real ``patch -p<strip> --fuzz=2`` apply (or reverse)."""
    flags = ("--reverse", "--silent") if reverse else ("--silent",)
    return _run_patch(
        patch_bin,
        patch_file,
        cwd,
        _PatchMode(flags, " --reverse" if reverse else "", log.warning, log_stderr=True),
        strip=strip,
    )


def _git(git: str, args: Sequence[str], cwd: Path) -> bool:
    """Run ``git <args>`` in ``cwd``, never in a repository above it: there ``git apply`` would read the patch's paths
    from that repository's top and skip every one outside ``cwd``."""
    try:
        result = subprocess.run(
            (git, *args),
            cwd=str(cwd),
            capture_output=True,
            timeout=_GIT_TIMEOUT_SEC,
            env={**os.environ, "GIT_CEILING_DIRECTORIES": str(cwd.parent)},
        )
    except (OSError, subprocess.TimeoutExpired) as e:
        log.warning(
            "_server_patcher: git %s in %s failed to spawn (%s)",
            list(args),
            cwd,
            e,
        )
        return False
    if result.returncode != 0:
        err = result.stderr.decode(_UTF8, errors=_LENIENT)[:500] if result.stderr else ""
        log.debug(
            "_server_patcher: git %s rc=%d stderr=%r",
            list(args),
            result.returncode,
            err,
        )
        return False
    return True


__all__ = ["PatchedServer", "is_patched_copy"]
