# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The style gate judges only what a change adds, by the delivery rules."""

from __future__ import annotations

import subprocess
from pathlib import Path

from kernelforge.finalize.style_gate import style_findings

BASE = '''# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT


def old():
    """A base docstring that already runs
    over three lines and quotes 1.25x,
    which is the base's debt,
    not this change's."""
    return 1
'''


def _repo(tmp_path: Path, text: str) -> tuple[Path, str]:
    (tmp_path / "kernel.py").write_text(BASE)
    run = ["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run([*run, "init", "-q"], check=True)
    subprocess.run([*run, "add", "-A"], check=True)
    subprocess.run([*run, "commit", "-qm", "base"], check=True)
    base = subprocess.run([*run, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    (tmp_path / "kernel.py").write_text(text)
    return tmp_path, base


def test_only_what_the_change_adds_is_judged(tmp_path: Path) -> None:
    added = (
        "\n\ndef new():\n"
        '    """Split K four ways.\n\n    Measured 1.31x on m=7.\n    """\n'
        "    # a\n    # b\n    # c\n    # d: was 2048 before round 12\n"
        "    tmp = 1\n    print(tmp)\n    # 拆分\n    return tmp\n"
    )
    workspace, base = _repo(tmp_path, BASE + added)
    found = {(finding.rule, finding.detail.split(" ")[0]) for finding in style_findings(workspace, base, ["kernel.py"])}
    rules = {rule for rule, _ in found}
    assert rules == {"block", "number", "density", "debug", "language"}
    findings = [str(finding) for finding in style_findings(workspace, base, ["kernel.py"])]
    assert not any("old" in finding for finding in findings)
    assert any("docstring of new has 4 lines" in finding for finding in findings)
    assert any("'1.31x'" not in finding and "'1.31'" in finding for finding in findings)
    assert any("comment block of 4 lines" in finding for finding in findings)
    assert any("'2048'" in finding for finding in findings)


def test_a_clean_change_and_other_files_pass(tmp_path: Path) -> None:
    workspace, base = _repo(tmp_path, BASE + "\n\nSPLIT_K = 4\n")
    assert style_findings(workspace, base, ["kernel.py", "driver.cpp", "missing.py"]) == []
