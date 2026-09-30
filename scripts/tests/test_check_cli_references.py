# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for scripts/check_cli_references.py."""

from __future__ import annotations

from pathlib import Path

import pytest

from check_cli_references import check, main

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def root(tmp_path: Path) -> Path:
    (tmp_path / "src" / "hyperloom" / "agents" / "critic" / "runtime").mkdir(parents=True)
    (tmp_path / "src" / "hyperloom" / "agents" / "critic" / "runtime" / "cli.py").write_text(
        'if __name__ == "__main__":\n    main()\n', encoding="utf-8"
    )
    (tmp_path / "src" / "hyperloom" / "tool.py").write_text("def main(): ...\n", encoding="utf-8")
    return tmp_path


def _skill(root: Path, body: str) -> None:
    (root / "SKILL.md").write_text(body, encoding="utf-8")


def test_resolvable_references_pass(root: Path) -> None:
    _skill(
        root,
        "python3 -m hyperloom optimize --model m\n"
        'python -m hyperloom session events "$S"\n'
        '"$PYTHON" -m hyperloom check "$MODEL_PATH"\n'
        "python -m hyperloom.agents.critic.runtime.cli prepare-review --request r\n"
        '"$PYTHON" -m pip install x\n'
        'python3 "$REPO_ROOT/src/hyperloom/tool.py" x\n'
        "run `python -m hyperloom <command>` for help\n"
        'git commit -m "msg"\n',
    )
    assert check(root) == []


def test_unresolvable_references_are_reported(root: Path) -> None:
    _skill(
        root,
        "python -m runtime.cli prepare-review\n"
        "python -m hyperloom.agents.critic.cli prepare-review\n"
        "python -m hyperloom recover-session --session-dir s\n"
        "python -m hyperloom session dump\n"
        "python3 src/hyperloom/tools/gone.py\n"
        "python -m hyperloom.tool\n",
    )
    assert check(root) == [
        "SKILL.md:1: module runtime.cli is not a hyperloom/kernelforge module",
        "SKILL.md:2: module hyperloom.agents.critic.cli is not runnable with -m",
        "SKILL.md:3: `hyperloom recover-session` is not a hyperloom command",
        "SKILL.md:4: `hyperloom session dump` is not a session command",
        "SKILL.md:5: script src/hyperloom/tools/gone.py does not exist",
        "SKILL.md:6: module hyperloom.tool is not runnable with -m",
    ]
    assert main([str(root)]) == 1


def test_repository_references_resolve() -> None:
    assert check(REPO_ROOT) == []
