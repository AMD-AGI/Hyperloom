# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which checkout an integrate attempt mutated, so a resume can tell it from a replacement.

An attempt records an identifier kept in the git metadata of the working tree it
patches (the per-worktree git directory, so linked worktrees each have their own).
A resume that reads a different identifier there is looking at another checkout
(the source was cloned again in a new execution environment), so the attempt's
tree-side obligations -- take the candidate back out, hand the operator's stash
back -- went away with the tree that held them. Only targets outside that tree
still carry an obligation.
"""

from __future__ import annotations

import logging
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

from ...delivery.ledger import file_digest, load_prepared_records, restore_records
from ._git import _run_git_cp
from ._nogit_patch import _P_LEVELS

log = logging.getLogger(__name__)

TREE_INSTANCE_FILE = "hyperloom-source-tree-id"
_ENCODING = "utf-8"
_GIT_TIMEOUT_SEC = 30.0
_DISCHARGED_PHASES = frozenset({"restored", "accepted"})
_LEGACY_INFERRED_PHASES = frozenset({"ready", "applied", "files_restored", "applied_with_restored_stash"})


def _instance_path(root: Path) -> Path:
    """The identifier file inside ``root``'s git metadata; OSError when there is none."""
    cp = _run_git_cp(["-C", str(root), "rev-parse", "--absolute-git-dir"], timeout=_GIT_TIMEOUT_SEC)
    text = (cp.stdout or "").strip() if cp is not None and cp.returncode == 0 else ""
    if not text:
        raise OSError(f"cannot locate the git metadata of {root}")
    return Path(text) / TREE_INSTANCE_FILE


def read_tree_instance(root: Path) -> str:
    """Return the identifier recorded in ``root``'s git metadata, ``""`` when none was ever recorded."""
    try:
        return _instance_path(root).read_text(encoding=_ENCODING).strip()
    except FileNotFoundError:
        return ""


def ensure_tree_instance(root: Path) -> str:
    """Return ``root``'s identifier, creating it first if this checkout has none.

    The file is linked into place complete, so a concurrent creator or a crash
    can never leave an empty or half-written identifier behind.
    """
    path = _instance_path(root)
    with tempfile.NamedTemporaryFile("w", encoding=_ENCODING, dir=path.parent, delete=False) as fh:
        fh.write(uuid.uuid4().hex + "\n")
    try:
        os.link(fh.name, path)
    except FileExistsError:
        pass
    finally:
        os.unlink(fh.name)
    return path.read_text(encoding=_ENCODING).strip()


def new_recovery_record(framework_root: Path | None, *, git_head: str, recovery_root: Path) -> dict[str, Any]:
    """The recovery evidence an attempt persists before it mutates anything.

    A git tree also records its identifier. When that cannot be written the
    attempt goes ahead without one, and a resume treats it like any sentinel
    that predates the identifier.
    """
    record: dict[str, Any] = {
        "version": 1,
        "phase": "before_stash",
        "git_head": git_head,
        "stash_oid": "",
        "root": str(recovery_root),
    }
    if git_head and framework_root is not None:
        try:
            record["tree_instance"] = ensure_tree_instance(framework_root)
        except OSError as exc:
            log.warning("integrate_patch: cannot record the identity of %s: %s", framework_root, exc)
    return record


def _stash_object_absent(root: Path, stash_oid: str) -> bool:
    """True only when ``root`` is readable git metadata that does not hold ``stash_oid`` at all."""
    try:
        _instance_path(root)
    except OSError:
        return False
    cp = _run_git_cp(["-C", str(root), "cat-file", "-e", f"{stash_oid}^{{commit}}"], timeout=_GIT_TIMEOUT_SEC)
    return cp is not None and cp.returncode != 0


def _patches_cleanly_absent(root: Path, patches: list[Any]) -> bool:
    """True when every recorded candidate patch would still apply forward to ``root`` as it stands.

    A forward ``--check`` passes only when every hunk finds its pre-image, so a
    candidate that is applied, partly applied, or edited over fails it, as does
    a patch file that is gone. A patch that also reverse-applies is ambiguous
    and counts as present.
    """
    paths = [str(patch) for patch in patches]
    if any(not Path(path).is_file() or _git_apply_checks(root, [path], reverse=True) for path in paths):
        return False
    if not paths:
        return True
    # In sequence first, as one input, for a later patch built on an earlier one (given as separate arguments,
    # git checks each against the unpatched tree); then each alone, as their strip levels may differ.
    try:
        joined = "".join(_with_final_newline(Path(path).read_text(encoding=_ENCODING)) for path in paths)
    except (OSError, UnicodeDecodeError):
        joined = None
    if joined is not None and _git_apply_checks(root, ["-"], reverse=False, stdin=joined):
        return True
    return all(_git_apply_checks(root, [path], reverse=False) for path in paths)


def _with_final_newline(text: str) -> str:
    return text if text.endswith("\n") else text + "\n"


def _git_apply_checks(root: Path, paths: list[str], *, reverse: bool, stdin: str | None = None) -> bool:
    """True when ``git apply --check`` (``-R`` when ``reverse``) accepts ``paths`` at some strip level."""
    direction = ["-R"] if reverse else []
    for level in _P_LEVELS:
        cp = _run_git_cp(
            ["-C", str(root), "apply", *direction, f"-p{level}", "--check", *paths],
            timeout=_GIT_TIMEOUT_SEC,
            input=stdin,
        )
        if cp is not None and cp.returncode == 0:
            return True
    return False


def _in_tree_artifacts_at_preimage(root: Path, recovery: dict[str, Any]) -> bool:
    """True when every artifact target inside ``root`` has the preimage content the attempt recorded (mode not compared)."""
    if not recovery.get("artifacts_prepared"):
        return True
    try:
        records = load_prepared_records(Path(str(recovery.get("root") or "")) / "artifact_backups")
        first: dict[Path, dict[str, Any]] = {}
        for row in records:
            first.setdefault(Path(row["target"]).resolve(), row)
    except (OSError, ValueError, TypeError, KeyError):
        return False
    for target, row in first.items():
        if not target.is_relative_to(root):
            continue
        if (file_digest(target) != row.get("pre_image_sha256")) if row["existed"] else target.exists():
            return False
    return True


def tree_replaced(pending: dict[str, Any], gone_scope: str | None) -> bool:
    """Whether the tree a pending integrate mutated has been replaced by another checkout.

    A window already restored or accepted owes the tree nothing, so it never
    counts. In every other phase a recorded identifier settles it by
    comparison. A sentinel written before identifiers existed is judged
    replaced only when an operator confirmed the task's execution stopped
    (``gone_scope`` is not ``None``), the recorded stash object is not in this
    tree's object store at all, and nothing of the candidate is in this tree:
    every patch still applies forward cleanly (so an applied candidate, or one
    edited over, rules the inference out) and every artifact target inside the
    tree holds the preimage content the attempt recorded. That is an inference,
    and it can misread the same tree with its stash dropped and pruned in two
    cases. The candidate already taken out of it by hand: calling that tree
    replaced loses nothing, as there is nothing to reverse, the targets outside
    are restored all the same, and the stash is already gone for good. An
    artifact inside the tree whose content matches its preimage but whose file
    mode the candidate changed: that mode is left as the candidate set it. A stash
    merely dropped keeps its object and keeps the refusal. Anything unreadable
    answers no, which leaves the ordinary recovery to refuse.
    """
    recovery = pending.get("recovery")
    root_text = str(pending.get("framework_source_root") or "")
    if not isinstance(recovery, dict) or recovery.get("phase") in _DISCHARGED_PHASES or not root_text:
        return False
    root = Path(root_text)
    recorded = str(recovery.get("tree_instance") or "")
    if recorded:
        try:
            return read_tree_instance(root) != recorded
        except OSError:
            return False
    stash_oid = str(recovery.get("stash_oid") or "")
    if recovery.get("phase") not in _LEGACY_INFERRED_PHASES or gone_scope is None or not stash_oid:
        return False
    if not _stash_object_absent(root, stash_oid):
        return False
    if not _patches_cleanly_absent(root, list(pending.get("patches") or [])):
        return False
    return _in_tree_artifacts_at_preimage(root.resolve(), recovery)


def settle_replaced_tree(pending: dict[str, Any]) -> dict[str, Any]:
    """Discharge a pending integrate whose tree was replaced: restore only what outlived it.

    Artifact targets outside the replaced tree are put back from their committed
    preimages. Patches and artifact targets inside the tree went with it. The
    operator's stash is not handed to the new checkout, which is not the tree it
    was taken from; it is reported (it may survive in a stash list the two share).
    """
    summary: dict[str, Any] = {"reversed": [], "artifacts_reverted": [], "failed": [], "tree_replaced": True}
    recovery = pending["recovery"]
    root = Path(pending["framework_source_root"]).resolve()
    if recovery.get("artifacts_prepared") and recovery.get("phase") != "files_restored":
        try:
            records = load_prepared_records(Path(recovery["root"]) / "artifact_backups")
        except (OSError, ValueError, TypeError, KeyError) as exc:
            summary["failed"].append(str(exc))
            return summary
        expected = {str(Path(row["target"]).resolve()) for row in pending.get("artifacts", [])}
        if not expected <= {str(Path(row["target"]).resolve()) for row in records}:
            summary["failed"].append("artifact recovery ledger does not cover the complete mutation plan")
            return summary
        outside = [row for row in records if not Path(row["target"]).resolve().is_relative_to(root)]
        _, errors = restore_records(outside)
        if errors:
            summary["failed"].extend(errors)
            return summary
        summary["artifacts_reverted"] = [row.get("rel_target") or row["target"] for row in outside]
    if recovery.get("stash_oid"):
        summary["unrestored_stash_oid"] = recovery["stash_oid"]
        log.warning(
            "integrate_patch: %s was replaced; the operator changes stashed as %s were not restored into it",
            root,
            recovery["stash_oid"],
        )
    recovery["phase"] = "restored"
    recovery["tree_replaced"] = True
    return summary
