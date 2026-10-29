# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Whether an upstream pull request is already in a local git tree, answered from local git alone.

Local history can show that a PR is in a tree, never that it is absent: a backport that was
edited, had a conflict resolved, or was folded into a larger commit looks the same as no carry
at all. So the answer is ``included``, with the git evidence, or ``unknown``, with the reason
nothing showed it. The evidence is read in this order:

1. A commit reachable from ``commit`` whose subject names the PR: a trailing ``(#N)``, alone or
   in a run such as ``(#N) (#M)``, the way a squash merge and a fork's cherry-pick of it record
   the PR, or a ``Merge pull request #N from ...`` subject, the way a merge commit and a
   ``git cherry-pick -m 1`` of it do. ``(#N)`` anywhere else in the subject, or only in the
   body, does not count. A subject carries only the number, so a fork that numbers its own PRs
   the same way can collide; the evidence names the commit and its subject so a caller can
   compare titles.
2. When the PR's merge commit sha (its merge or squash commit upstream) is given: every
   non-merge commit that merge commit brought in is in ``commit``, by sha or by patch-id. That
   holds when the merge commit is an ancestor, and for a cherry-pick under any subject whose
   diff, whitespace and line numbers aside, is identical. A rebase merge records only the PR's
   last commit, so that commit alone is compared.

Nothing is fetched. The checked commit and the merge commit are read through
``rev-list --missing=print``, which turns lazy fetching off, so a sha that a partial clone lacks
fails the git call, and so does a patch-id that needs a blob a blob-filtered clone lacks. A git
call that fails, cannot start or does not answer in time makes the answer ``unknown``, with the
error, git's own message for a failed call, as the evidence. ``included`` means the PR's change
is in the history; a later revert is not looked for.

The ``pr_check_inclusion`` MCP tool answers from subjects alone, because nothing the agent can
call returns a merge commit sha. The command line also takes the sha; the check-prior-art
contributor skill runs it with the PR's ``mergeCommit``, which it reads with ``gh pr view``::

    python -m kernelforge.knowledge.pr_knowledge.inclusion --tree PATH --commit SHA --pr N \\
        [--merge-commit-sha SHA]

It prints the verdict as one JSON object with ``inclusion``, ``method`` and ``evidence``, and
exits 0 when the PR is included, 3 when that is unknown, and 2 on a usage error. No verdict
exits 1, the status of an uncaught error, so a crash never reads as a verdict.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from hyperloom.common.git_safety import safe_directory_args
from kernelforge.utils.git import GitError, git_bytes

# At most three calls run, so a probe ends inside the 60 s the Forge agent gives one pr_monitor tool call.
_GIT_TIMEOUT_SEC = 15.0
_SUFFIX_RUN = re.compile(r"(?:\s*\(#\d+\))+\s*$")
# Any --missing action but error turns lazy fetching off, so a partial clone reads only the objects it has.
_LOCAL_REV_LIST = ("rev-list", "--missing=print")


class Inclusion(str, Enum):
    """Whether a pull request is in a tree, as far as local history can show."""

    INCLUDED = "included"
    UNKNOWN = "unknown"


class InclusionMethod(str, Enum):
    """The git evidence a verdict rests on; ``none`` when nothing showed the PR."""

    SUBJECT = "subject"
    MERGE_COMMIT = "merge_commit"
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
    try:
        head = _git(tree, *_LOCAL_REV_LIST, "--no-walk", "--end-of-options", f"{commit}^{{commit}}", "--").strip()
        return _subject_naming(tree, head, number) or _merge_commit_carried(tree, head, number, merge_commit_sha)
    except (GitError, subprocess.TimeoutExpired, OSError) as error:
        return InclusionVerdict.unknown(f"git gave no answer: {error}")


def _git(tree: Path, *args: str) -> str:
    """Stdout of one read-only git command in ``tree``, decoded whatever the log encoding."""
    completed = git_bytes(*safe_directory_args(list(args), cwd=tree), cwd=tree, timeout=_GIT_TIMEOUT_SEC)
    return completed.stdout.decode(errors="replace")


def _subject_naming(tree: Path, head: str, number: int) -> InclusionVerdict | None:
    """The verdict of the first commit reachable from ``head`` whose subject names PR ``number``, if any."""
    merge_subject = f"Merge pull request #{number} from "
    found = _git(
        tree, "log", "--fixed-strings", f"--grep=(#{number})", f"--grep={merge_subject}", "--format=%H%x00%s", head
    )
    # %s never holds a newline, but may hold other characters that splitlines() would break on.
    for line in found.split("\n"):
        sha, _, subject = line.partition("\0")
        run = _SUFFIX_RUN.search(subject)
        if subject.startswith(merge_subject) or (run is not None and f"(#{number})" in run.group()):
            return InclusionVerdict(
                Inclusion.INCLUDED,
                InclusionMethod.SUBJECT,
                f"commit {sha} reachable from {head} has the subject {subject!r}",
            )
    return None


def _merge_commit_carried(tree: Path, head: str, number: int, merge: str | None) -> InclusionVerdict:
    """Whether ``head`` has every non-merge commit ``merge`` brought in, by sha or by patch-id."""
    searched = f"no commit subject reachable from {head} names #{number}"
    if not merge:
        return InclusionVerdict.unknown(f"{searched}, and no merge commit sha was given")
    # --cherry-pick drops each commit head has by patch-id; ^merge^1 keeps only what the merge brought in.
    uncarried = _git(
        tree, *_LOCAL_REV_LIST, "--no-merges", "--cherry-pick", "--right-only", f"{head}...{merge}", f"^{merge}^1", "--"
    ).split()
    if uncarried:
        return InclusionVerdict.unknown(
            f"{searched}, and {len(uncarried)} non-merge commit(s) merge commit {merge} brought in, such as "
            f"{uncarried[0]}, are not in it by sha or by patch-id"
        )
    return InclusionVerdict(
        Inclusion.INCLUDED,
        InclusionMethod.MERGE_COMMIT,
        f"every non-merge commit merge commit {merge} brought in is in {head}, by sha or by patch-id",
    )


def main(argv: list[str] | None = None) -> int:
    """Print one verdict as JSON and return the exit status that encodes it."""
    parser = argparse.ArgumentParser(
        prog="python -m kernelforge.knowledge.pr_knowledge.inclusion",
        description="Check from local git whether an upstream pull request is in a tree.",
        epilog="Exit status: 0 included, 3 unknown, 2 usage error.",
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
    return 0 if verdict.inclusion is Inclusion.INCLUDED else 3


if __name__ == "__main__":
    raise SystemExit(main())
