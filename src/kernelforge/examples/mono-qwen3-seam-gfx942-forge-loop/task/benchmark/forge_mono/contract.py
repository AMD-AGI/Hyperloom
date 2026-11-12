# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The checks a candidate meets before it is measured: the mono runtime it was written against, and
the static rules on the editable sources (``qwen3_seam/``)."""

import ast
import json
import sys
from importlib import metadata
from pathlib import Path

RUNTIME = "sgl-amd-mono"
RUNTIME_COMMIT = "b10f3b71c45a70f504c5cad2d622c3e423196e90"
EDITABLE = Path(__file__).resolve().parents[2] / "qwen3_seam"
# a wait is a guarded mailbox poll; these are the ways around the guard
UNGUARDED = frozenset({"s_sleep", "_spin", "spin_until"})


def runtime_commit() -> str | None:
    """The commit the installed runtime was built from, as pip recorded it."""
    try:
        record = metadata.distribution(RUNTIME).read_text("direct_url.json")
    except metadata.PackageNotFoundError:
        return None
    return json.loads(record).get("vcs_info", {}).get("commit_id") if record else None


def _called_name(call: ast.Call) -> str | None:
    func = call.func
    return func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)


def source_violations(root: Path = EDITABLE) -> list[str]:
    """Every ``while`` loop, and every call or import of an unguarded wait, in ``root``'s modules."""
    found = []
    for path in sorted(root.rglob("*.py")):
        hits = []
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"), filename=str(path))):
            if isinstance(node, ast.While):
                hits.append((node.lineno, "a while loop; a wait must be a guarded Mailbox.poll"))
            elif isinstance(node, ast.Call) and _called_name(node) in UNGUARDED:
                hits.append((node.lineno, f"{_called_name(node)}() waits outside the guard"))
            elif isinstance(node, ast.ImportFrom):
                hits += [
                    (node.lineno, f"imports {alias.name}, a wait outside the guard")
                    for alias in node.names
                    if alias.name in UNGUARDED
                ]
        where = path.relative_to(root.parent)
        found += [f"{where}:{line}: {what}" for line, what in sorted(hits)]
    return found


def check_contract() -> None:
    """Exit non-zero, naming each breach, unless the runtime and the editable sources meet the contract."""
    problems = []
    commit = runtime_commit()
    if commit != RUNTIME_COMMIT:
        problems.append(
            f"{RUNTIME} is at {commit or 'an unrecorded commit'}; the task is written against {RUNTIME_COMMIT}"
        )
    problems += source_violations()
    if problems:
        for problem in problems:
            print(f"contract: {problem}", file=sys.stderr)
        sys.exit(3)
