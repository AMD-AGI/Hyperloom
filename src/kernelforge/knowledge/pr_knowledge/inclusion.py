# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether an upstream pull request is already in a local git tree, answered from local git alone.

The evidence is read in this order:

1. With the PR's merge commit sha (its merge or squash commit upstream) present in the tree,
   ``git merge-base --is-ancestor`` says whether ``commit`` contains it.
2. A commit reachable from ``commit`` whose subject ends with the squash-merge suffix ``(#N)``
   is cherry-pick evidence, the way a fork carries an upstream PR. The suffix may sit in a
   trailing run such as ``(#N) (#M)``, which a fork leaves when it merges the cherry-pick
   through a PR of its own; ``(#N)`` anywhere else in the subject, or only in the body,
   does not count.

``not_included`` needs both a definite not-ancestor answer from a complete (not shallow)
history and no such subject. Every other outcome is ``unknown``, with the reason in the
evidence, so a merge sha that was not supplied or is not fetched, or a shallow clone,
never reads as absence. Nothing is fetched: a sha that a partial clone lacks counts as not
fetched. ``included`` means the PR's commit is in the history; a later revert is not
looked for. A subject carries only the number, so a fork that numbers its own PRs into
the same suffix can collide; the evidence names the matching commit and its subject so a
caller can compare titles.

Command line::

    python -m kernelforge.knowledge.pr_knowledge.inclusion --tree PATH --commit SHA --pr N \\
        [--merge-commit-sha SHA]

prints the verdict as one JSON object with ``inclusion``, ``method`` and ``evidence``, and
exits 0 when the PR is included, 4 when it is not, 3 when that is unknown, and 2 on a
usage error. No verdict exits 1, the status of an uncaught error, so a crash never reads
as a verdict.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from hyperloom.common.git_safety import safe_directory_args
from kernelforge.llm.git import git_bytes

try:  # Python 3.11+
    from enum import StrEnum
except ImportError:  # pragma: no cover - Python 3.10 StrEnum shim
    from enum import Enum

    class StrEnum(str, Enum):
        def __str__(self) -> str:
            """Return the enum's string value (mirrors 3.11 ``StrEnum``)."""
            return str(self.value)


# Each call is a local plumbing command; one that runs this long is stuck on the filesystem, not working.
_GIT_TIMEOUT_SEC = 30.0
_NO_ANSWER = f"git did not run or did not answer within {_GIT_TIMEOUT_SEC:g}s"
_SUFFIX_RUN = re.compile(r"(?:\s*\(#\d+\))+\s*$")
_SUFFIX_NUMBER = re.compile(r"\(#(\d+)\)")


class Inclusion(StrEnum):
    """Whether a pull request is in a tree."""

    INCLUDED = "included"
    NOT_INCLUDED = "not_included"
    UNKNOWN = "unknown"


class InclusionMethod(StrEnum):
    """The git evidence a verdict rests on; ``none`` when nothing decided it."""

    MERGE_ANCESTRY = "merge_ancestry"
    SUBJECT_SUFFIX = "subject_suffix"
    NONE = "none"


@dataclass(frozen=True)
class InclusionVerdict:
    """One answer and the git evidence behind it."""

    inclusion: Inclusion
    method: InclusionMethod
    evidence: str

    @classmethod
    def unknown(cls, evidence: str) -> InclusionVerdict:
        """A verdict nothing decided, carrying the reason."""
        return cls(Inclusion.UNKNOWN, InclusionMethod.NONE, evidence)

    def to_record(self) -> dict[str, str]:
        """JSON-ready fields, the vocabularies as their plain values."""
        return {"inclusion": self.inclusion.value, "method": self.method.value, "evidence": self.evidence}


def probe_inclusion(*, tree: Path, commit: str, number: int, merge_commit_sha: str | None) -> InclusionVerdict:
    """Decide whether pull request ``number`` is in ``commit`` of the git tree at ``tree``.

    Args:
        tree: A directory inside the git work tree to read.
        commit: The revision to check, such as a sha or ``HEAD``.
        number: The upstream pull request number.
        merge_commit_sha: The PR's merge or squash commit upstream, when known.
    """
    head = _resolve_commit(tree, commit)
    if head is None:
        return InclusionVerdict.unknown(f"git cannot resolve {commit} to a commit in {tree}")
    ancestry = _merge_ancestry(tree, head, merge_commit_sha)
    if ancestry.inclusion is Inclusion.INCLUDED:
        return ancestry
    subjects = _subject_suffix(tree, head, number)
    if subjects.inclusion is Inclusion.INCLUDED:
        return subjects
    evidence = f"{ancestry.evidence}; {subjects.evidence}"
    if ancestry.inclusion is Inclusion.NOT_INCLUDED and subjects.inclusion is Inclusion.NOT_INCLUDED:
        return InclusionVerdict(Inclusion.NOT_INCLUDED, InclusionMethod.MERGE_ANCESTRY, evidence)
    return InclusionVerdict.unknown(evidence)


def _git(tree: Path, *args: str) -> subprocess.CompletedProcess | None:
    """Run one read-only git command in ``tree``; None when git cannot start there or does not answer in time."""
    try:
        completed = git_bytes(
            *safe_directory_args(list(args), cwd=tree), cwd=tree, check=False, timeout=_GIT_TIMEOUT_SEC
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return subprocess.CompletedProcess(
        completed.args,
        completed.returncode,
        completed.stdout.decode(errors="replace"),
        completed.stderr.decode(errors="replace"),
    )


def _resolve_commit(tree: Path, revision: str) -> str | None:
    """The full sha ``revision`` names in ``tree``, or None when it names no commit there."""
    # A leading dash would reach git as an option rather than a revision.
    if revision.startswith("-"):
        return None
    # rev-parse would fetch a sha that a partial clone lacks; rev-list with a --missing action reads local objects only.
    completed = _git(tree, "rev-list", "--missing=print", "--no-walk", f"{revision}^{{commit}}", "--")
    if completed is None or completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _merge_ancestry(tree: Path, head: str, merge_commit_sha: str | None) -> InclusionVerdict:
    """What ancestry says: included, a definite not-ancestor, or unknown with the reason."""
    if not merge_commit_sha:
        return InclusionVerdict.unknown("no merge commit sha was supplied")
    merge = _resolve_commit(tree, merge_commit_sha)
    if merge is None:
        return InclusionVerdict.unknown(
            f"merge commit {merge_commit_sha} is not in the tree (not fetched, or cut off by a shallow clone)"
        )
    completed = _git(tree, "merge-base", "--is-ancestor", merge, head)
    if completed is None:
        return InclusionVerdict.unknown(_NO_ANSWER)
    if completed.returncode == 0:
        return InclusionVerdict(
            Inclusion.INCLUDED, InclusionMethod.MERGE_ANCESTRY, f"merge commit {merge} is an ancestor of {head}"
        )
    if completed.returncode != 1:
        return InclusionVerdict.unknown(f"git merge-base failed: {completed.stderr.strip()}")
    if _is_shallow(tree):
        return InclusionVerdict.unknown(f"merge commit {merge} is not an ancestor of {head}, but the clone is shallow")
    return InclusionVerdict(
        Inclusion.NOT_INCLUDED, InclusionMethod.MERGE_ANCESTRY, f"merge commit {merge} is not an ancestor of {head}"
    )


def _is_shallow(tree: Path) -> bool:
    """Whether the history may be cut short; a tree git cannot vouch for counts as shallow."""
    completed = _git(tree, "rev-parse", "--is-shallow-repository")
    return completed is None or completed.stdout.strip() != "false"


def _subject_suffix(tree: Path, head: str, number: int) -> InclusionVerdict:
    """What subjects say: included, searched without a match, or unknown when git could not search."""
    completed = _git(tree, "log", "--fixed-strings", f"--grep=(#{number})", "--format=%H%x00%s", head)
    if completed is None:
        return InclusionVerdict.unknown(_NO_ANSWER)
    if completed.returncode != 0:
        return InclusionVerdict.unknown(f"git log failed: {completed.stderr.strip()}")
    # %s never holds a newline, but may hold other characters that splitlines() would break on.
    for line in completed.stdout.split("\n"):
        sha, _, subject = line.partition("\0")
        if _suffix_names(subject, number):
            return InclusionVerdict(
                Inclusion.INCLUDED,
                InclusionMethod.SUBJECT_SUFFIX,
                f"commit {sha} reachable from {head} has the subject {subject!r}",
            )
    return InclusionVerdict(
        Inclusion.NOT_INCLUDED,
        InclusionMethod.SUBJECT_SUFFIX,
        f"no commit subject reachable from {head} ends with (#{number})",
    )


def _suffix_names(subject: str, number: int) -> bool:
    """Whether the trailing run of ``(#k)`` suffixes on ``subject`` includes ``number``."""
    run = _SUFFIX_RUN.search(subject)
    return run is not None and any(int(found) == number for found in _SUFFIX_NUMBER.findall(run.group()))


_EXIT_STATUS = {Inclusion.INCLUDED: 0, Inclusion.NOT_INCLUDED: 4, Inclusion.UNKNOWN: 3}


def main(argv: list[str] | None = None) -> int:
    """Print one verdict as JSON and return the exit status that encodes it."""
    parser = argparse.ArgumentParser(
        prog="python -m kernelforge.knowledge.pr_knowledge.inclusion",
        description="Check from local git whether an upstream pull request is in a tree.",
        epilog="Exit status: 0 included, 4 not included, 3 unknown, 2 usage error.",
    )
    parser.add_argument("--tree", type=Path, required=True, help="a directory inside the git work tree")
    parser.add_argument("--commit", required=True, help="the revision to check, such as a sha or HEAD")
    parser.add_argument("--pr", type=int, required=True, help="the upstream pull request number")
    parser.add_argument("--merge-commit-sha", help="the PR's merge or squash commit upstream")
    args = parser.parse_args(argv)
    if not args.tree.is_dir():
        parser.error(f"--tree {args.tree} is not a directory")
    verdict = probe_inclusion(
        tree=args.tree, commit=args.commit, number=args.pr, merge_commit_sha=args.merge_commit_sha
    )
    print(json.dumps(verdict.to_record()))
    return _EXIT_STATUS[verdict.inclusion]


if __name__ == "__main__":
    raise SystemExit(main())
