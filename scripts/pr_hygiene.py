#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""PR hygiene checks run by ``.github/workflows/pr-hygiene.yml``.

``size``: the diff budget from AGENTS.md. Counts added + deleted lines of production Python
(``src/`` and ``scripts/``, test files excluded) in ``git diff -M --numstat BASE HEAD``. Above
``--warn`` lines it warns; above ``--limit`` it fails unless the PR carries the ``size-exception``
label AND the most recent person who added that label has write, maintain or admin permission on the
repository (checked through the GitHub API). A label that cannot be verified does not exempt.

``template``: the PR body (``PR_BODY``, passed as data through the environment) answers the required
fields of ``.github/PULL_REQUEST_TEMPLATE.md``, and "PR addresses single concern" starts with yes or
no. Advisory unless ``--enforce`` is given: findings are warnings and the exit code stays 0.

Usage:
    python scripts/pr_hygiene.py size --base HEAD^1 --head HEAD
    python scripts/pr_hygiene.py template [--enforce]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

LIMIT = 1000
WARN = 400
EXEMPT_LABEL = "size-exception"
WRITE_ROLES = frozenset({"admin", "maintain", "write"})
_SIZE_BUCKETS = ((10, "XS"), (30, "S"), (100, "M"), (500, "L"), (1000, "XL"))
_PRODUCTION_ROOTS = ("src/", "scripts/")
_TEST_DIRS = frozenset({"tests", "test"})
_DIFF_BUDGET = "diff-budget"
_ENCODING = "utf-8"

# (field, regex matching the start of the normalised bullet, required, must answer yes/no).
# test_pr_hygiene.py pins every key to a line of the real template, so a template edit that
# renames a field fails there instead of silently making the field optional.
TEMPLATE_FIELDS: tuple[tuple[str, str, bool, bool], ...] = (
    ("Description", r"description\b", False, False),
    ("Linked issue(s)", r"linked issues?\b", False, False),
    ("Tests", r"tests\b", True, False),
    ("Size/complexity", r"size/complexity\b", True, False),
    ("Simplifies or refactors", r"if this simplifies or refactors\b", False, False),
    ("Observable effect", r"observable effect\b", True, False),
    ("Breaking changes", r"breaking changes?\b", True, False),
    ("Single concern", r"(pr addresses )?single concern\b", True, True),
    ("Root cause is upstream", r"root cause is upstream\b", False, False),
)
TEMPLATE_PATH = Path(".github/PULL_REQUEST_TEMPLATE.md")


def _escape_annotation(text: str) -> str:
    """Escape a workflow-command message so data cannot end the command or start another."""
    return text.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")


def _annotate(level: str, title: str, message: str) -> None:
    print(f"::{level} title={title}::{_escape_annotation(message)}")


def _summary(lines: list[str]) -> None:
    path = os.environ.get("GITHUB_STEP_SUMMARY")
    if path:
        with open(path, "a", encoding=_ENCODING) as fh:
            fh.write("\n".join(lines) + "\n\n")


# ----------------------------------------------------------------------------- size


def is_production_python(path: str) -> bool:
    """True for ``src/`` or ``scripts/`` ``.py`` files that are not tests."""
    if not path.endswith(".py") or not path.startswith(_PRODUCTION_ROOTS):
        return False
    *dirs, name = path.split("/")
    if _TEST_DIRS & set(dirs):
        return False
    return not (name.startswith("test_") or name.endswith("_test.py") or name == "conftest.py")


def parse_numstat_z(raw: bytes) -> list[tuple[str | None, str, int]]:
    """Parse ``git diff --numstat -z`` into (old path or None, path, added + deleted); binary counts 0.

    A rename is ``A<TAB>D<TAB><NUL>old<NUL>new<NUL>``: it is attributed to the new path, so a pure
    move counts 0 and a move with edits counts only the edits.
    """
    tokens = raw.decode(_ENCODING, errors="surrogateescape").split("\0")
    out: list[tuple[str | None, str, int]] = []
    i = 0
    while i < len(tokens):
        head = tokens[i]
        i += 1
        if not head:
            continue
        added, deleted, path = head.split("\t", 2)
        old = None
        if not path:  # rename/copy: the two paths follow as separate tokens
            old, path = tokens[i], tokens[i + 1]
            i += 2
        lines = 0 if added == "-" else int(added) + int(deleted)
        out.append((old, path, lines))
    return out


def production_rows(rows: list[tuple[str | None, str, int]], line_count: Callable[[str], int]) -> list[tuple[str, int]]:
    """Production files and their changed lines.

    A file moved into production from a path the budget excludes (a test file) is new production
    code, so it counts in full via ``line_count``; a move within production counts only its edits.
    """
    out = []
    for old, path, lines in rows:
        if not is_production_python(path):
            continue
        if old is not None and not is_production_python(old):
            lines = line_count(path)
        out.append((path, lines))
    return out


def size_bucket(lines: int) -> str:
    return next((name for bound, name in _SIZE_BUCKETS if lines < bound), "XXL")


def _api_get(url: str, token: str) -> tuple[object, str]:
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.load(resp), resp.headers.get("Link", "")


def _next_link(link_header: str) -> str | None:
    m = re.search(r'<([^>]+)>;\s*rel="next"', link_header)
    return m.group(1) if m else None


@dataclass
class Exemption:
    granted: bool
    reason: str


def check_exemption(
    labels: list[str],
    repo: str,
    pr_number: str,
    token: str,
    api: str = "https://api.github.com",
    get: Callable[[str, str], tuple[object, str]] = _api_get,
) -> Exemption:
    """Decide whether ``size-exception`` exempts this PR; any doubt is a refusal."""
    if EXEMPT_LABEL not in labels:
        return Exemption(False, f"no '{EXEMPT_LABEL}' label")
    if not (repo and pr_number and token):
        return Exemption(False, f"'{EXEMPT_LABEL}' is present but GITHUB_REPOSITORY/PR_NUMBER/GH_TOKEN is unset")
    try:
        url: str | None = f"{api}/repos/{repo}/issues/{pr_number}/timeline?per_page=100"
        actor = None
        while url:
            events, link = get(url, token)
            for ev in events if isinstance(events, list) else []:
                if ev.get("event") == "labeled" and (ev.get("label") or {}).get("name") == EXEMPT_LABEL:
                    actor = (ev.get("actor") or {}).get("login")
            url = _next_link(link)
        if not actor:
            return Exemption(False, f"'{EXEMPT_LABEL}' is present but no labeled event names who added it")
        perm, _ = get(f"{api}/repos/{repo}/collaborators/{actor}/permission", token)
        perm = perm if isinstance(perm, dict) else {}
        # ``permission`` is the legacy field (maintain reads as write); ``role_name`` names custom roles.
        role = perm.get("role_name") or perm.get("permission")
        writer = perm.get("role_name") in WRITE_ROLES or perm.get("permission") in WRITE_ROLES
    except (urllib.error.URLError, OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
        return Exemption(False, f"'{EXEMPT_LABEL}' is present but could not be verified: {exc}")
    if writer:
        return Exemption(True, f"'{EXEMPT_LABEL}' added by {actor} ({role})")
    return Exemption(False, f"'{EXEMPT_LABEL}' was added by {actor}, whose role '{role}' lacks write access")


def cmd_size(args: argparse.Namespace) -> int:
    raw = subprocess.run(
        ["git", "diff", "-M", "--numstat", "-z", args.base, args.head],
        check=True,
        capture_output=True,
    ).stdout

    def line_count(path: str) -> int:
        blob = subprocess.run(["git", "show", f"{args.head}:{path}"], check=True, capture_output=True).stdout
        return len(blob.splitlines())

    rows = sorted(production_rows(parse_numstat_z(raw), line_count), key=lambda r: -r[1])
    total = sum(n for _, n in rows)
    bucket = size_bucket(total)
    lines = [
        "### Diff budget",
        f"Production Python lines changed (src/, scripts/, tests excluded): **{total}** (size/{bucket}); "
        f"warn above {args.warn}, fail above {args.limit} without `{EXEMPT_LABEL}`.",
    ]
    if rows:
        lines += ["", "| lines | file |", "|---:|---|"]
        lines += [f"| {n} | `{_escape_annotation(p).replace('|', '%7C').replace('`', '%60')}` |" for p, n in rows[:15]]
    rc = 0
    if total > args.limit:
        labels = json.loads(os.environ.get("PR_LABELS") or "[]")
        ex = check_exemption(
            labels,
            os.environ.get("GITHUB_REPOSITORY", ""),
            os.environ.get("PR_NUMBER", ""),
            os.environ.get("GH_TOKEN", ""),
            os.environ.get("GITHUB_API_URL", "https://api.github.com"),
        )
        if ex.granted:
            msg = f"{total} production Python lines changed exceeds the {args.limit}-line budget; exempt: {ex.reason}."
            _annotate("notice", _DIFF_BUDGET, msg)
        else:
            msg = (
                f"diff budget exceeded: {total} production Python lines changed (limit {args.limit}); "
                f"split the PR, or a maintainer with write access adds the '{EXEMPT_LABEL}' label ({ex.reason})."
            )
            _annotate("error", _DIFF_BUDGET, msg)
            rc = 1
        lines += ["", f"**{msg}**"]
    elif total > args.warn:
        msg = (
            f"{total} production Python lines changed is above the {args.warn}-line review trigger; consider splitting."
        )
        _annotate("warning", _DIFF_BUDGET, msg)
        lines += ["", msg]
    print(f"diff budget: {total} production Python lines changed (size/{bucket})")
    _summary(lines)
    return rc


# ----------------------------------------------------------------------------- template


def _normalise(text: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)  # markdown link -> its text
    text = re.sub(r"[*_`]+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _field_bullet(line: str) -> tuple[int, str] | None:
    """(field index, normalised text) for a top-level ``- <field>...:`` bullet, else None.

    Nested bullets are answer text, and a bullet with no colon (``- Tests pass now`` inside a
    description) is not a field line.
    """
    m = re.match(r"^[-*+]\s+(.*)$", line)
    if not m:
        return None
    text = _normalise(m.group(1))
    if ":" not in text:
        return None
    low = text.lower()
    for i, (_, key, _, _) in enumerate(TEMPLATE_FIELDS):
        if re.match(key, low):
            return i, text
    return None


def template_prompts(template: str) -> dict[int, str]:
    """Map field index -> the template's own prompt line (normalised, lower-case), trailing colon removed."""
    prompts: dict[int, str] = {}
    for line in template.splitlines():
        hit = _field_bullet(line)
        if hit:
            prompts[hit[0]] = hit[1].rstrip(":").strip().lower()
    return prompts


def parse_answers(body: str, prompts: dict[int, str]) -> dict[int, str]:
    """Answer text per field found in ``body``: the bullet minus its prompt, plus continuation lines."""
    answers: dict[int, list[str]] = {}
    current: int | None = None
    for line in body.replace("\r\n", "\n").split("\n"):
        hit = _field_bullet(line)
        if hit and hit[0] not in answers:
            current, text = hit
            prompt = prompts.get(current, "")
            if prompt and text.lower().startswith(prompt):
                rest = text[len(prompt) :]
            else:
                rest = text.partition(":")[2]
            answers[current] = [rest]
        elif current is not None:
            answers[current].append(line)
    return {i: _normalise(" ".join(parts)).lstrip(":- ").strip() for i, parts in answers.items()}


def template_findings(body: str, template: str) -> list[str]:
    prompts = template_prompts(template)
    missing_in_template = [f for i, (f, _, req, _) in enumerate(TEMPLATE_FIELDS) if req and i not in prompts]
    if missing_in_template:
        return [f"template has no line for required field(s): {', '.join(missing_in_template)}"]
    answers = parse_answers(body or "", prompts)
    findings = []
    for i, (field, _, required, yes_no) in enumerate(TEMPLATE_FIELDS):
        if not required:
            continue
        answer = answers.get(i)
        if answer is None:
            findings.append(f"'{field}' field is missing")
        elif not answer:
            findings.append(f"'{field}' field is not answered")
        elif yes_no and (not re.match(r"(yes|no)\b", answer, re.I) or re.match(r"yes\s*/\s*no", answer, re.I)):
            findings.append(f"'{field}' must be answered yes or no")
    return findings


def cmd_template(args: argparse.Namespace) -> int:
    if os.environ.get("PR_AUTHOR_TYPE") == "Bot":
        print("PR template: skipped for bot-authored PR")
        return 0
    template = Path(args.template).read_text(encoding=_ENCODING)
    findings = template_findings(os.environ.get("PR_BODY", ""), template)
    level = "error" if args.enforce else "warning"
    mode = "required" if args.enforce else "advisory"
    for f in findings:
        _annotate(level, "pr-template", f)
    lines = [f"### PR template ({mode})"]
    if findings:
        lines += ["Edit the PR description to answer:", ""] + [f"- {f}" for f in findings]
    else:
        lines.append("All required fields answered.")
    _summary(lines)
    print(f"PR template ({mode}): {len(findings)} finding(s)")
    return 1 if findings and args.enforce else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_size = sub.add_parser("size")
    p_size.add_argument("--base", required=True)
    p_size.add_argument("--head", required=True)
    p_size.add_argument("--limit", type=int, default=LIMIT)
    p_size.add_argument("--warn", type=int, default=WARN)
    p_size.set_defaults(func=cmd_size)
    p_tpl = sub.add_parser("template")
    p_tpl.add_argument("--enforce", action="store_true", help="fail instead of warn")
    p_tpl.add_argument("--template", default=str(TEMPLATE_PATH))
    p_tpl.set_defaults(func=cmd_template)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
