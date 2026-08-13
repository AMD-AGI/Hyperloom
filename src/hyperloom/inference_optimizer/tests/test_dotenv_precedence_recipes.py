# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The documented launch recipes must not let ``.env`` override the caller.

The recipes are markdown, so the snippets are extracted from the docs and
executed: a hardcoded copy here would keep passing while the docs drift.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest


PKG_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[4]

# In-package docs ship in the wheel; examples/ only exists in a source checkout.
RECIPE_DOCS = (
    PKG_ROOT / "SKILL.md",
    PKG_ROOT / "references" / "operations.md",
    REPO_ROOT / "examples" / "hyperloom-custom-advanced" / "SKILL.md",
    REPO_ROOT / "examples" / "hyperloom-qwen3-8b-3h" / "SKILL.md",
    REPO_ROOT / "examples" / "hyperloom-qwen3-14b-fp8-12h" / "SKILL.md",
)

# Loads only credential vars, so the path-variable assertions above do not apply,
# but .env must still not outrank a credential the caller exported.
CREDENTIAL_ONLY_DOC = REPO_ROOT / "docs" / "how-to" / "optimize-custom-workload.md"

_FENCE = re.compile(r"^```(?:bash|sh)\s*$")
_FENCE_END = re.compile(r"^```\s*$")

# Lines belonging to the dotenv-load preamble. Extraction stops at the first
# line outside this set, which is where the recipe starts doing real work.
_LOAD_LINE = (
    re.compile(r"^\s*$"),
    re.compile(r"^\s*#"),
    re.compile(r"^\s*cd\s"),
    re.compile(r"^\s*export\s+REPO_ROOT="),
    re.compile(r"/\.env"),
    re.compile(r"^\s*set\s+[-+]a\s*$"),
    re.compile(r"_dotenv_prev"),
)


def _bash_blocks(text: str) -> list[list[str]]:
    blocks: list[list[str]] = []
    current: list[str] | None = None
    for line in text.splitlines():
        if current is None:
            if _FENCE.match(line):
                current = []
            continue
        if _FENCE_END.match(line):
            blocks.append(current)
            current = None
            continue
        current.append(line)
    return blocks


def _extract_dotenv_load(doc: Path) -> str:
    """Return the leading dotenv-loading fragment of the doc's launch recipe."""
    for block in _bash_blocks(doc.read_text(encoding="utf-8")):
        if not any("/.env" in line for line in block):
            continue
        kept: list[str] = []
        for line in block:
            if not any(pattern.search(line) for pattern in _LOAD_LINE):
                break
            kept.append(line)
        fragment = "\n".join(kept)
        if "/.env" in fragment:
            return fragment
    raise AssertionError(f"no dotenv-loading bash block found in {doc}")


def _run_recipe(fragment: str, tmp_path: Path, exported: dict[str, str]) -> dict[str, str]:
    """Run the extracted fragment against a conflicting .env and report the result."""
    (tmp_path / ".env").write_text(
        "USER_DATA_PATH=/from/dotenv\nOPENAI_API_KEY=key-from-dotenv\nONLY_IN_DOTENV=filled\n",
        encoding="utf-8",
    )
    script = tmp_path / "recipe.sh"
    script.write_text(
        fragment + '\nprintf "%s\\n%s\\n%s\\n" "$USER_DATA_PATH" "$OPENAI_API_KEY" "$ONLY_IN_DOTENV"\n',
        encoding="utf-8",
    )

    # Minimal env: the developer shell leaks real credentials into pytest.
    run_env = {"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "REPO_ROOT": str(tmp_path)}
    run_env.update(exported)
    proc = subprocess.run(
        ["bash", str(script)],
        cwd=tmp_path,
        env=run_env,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    user_data_path, api_key, only_in_dotenv = proc.stdout.splitlines()[:3]
    return {
        "USER_DATA_PATH": user_data_path,
        "OPENAI_API_KEY": api_key,
        "ONLY_IN_DOTENV": only_in_dotenv,
    }


@pytest.mark.parametrize("doc", RECIPE_DOCS, ids=lambda p: p.parent.name + "/" + p.name)
def test_recipe_keeps_caller_user_data_path(doc: Path, tmp_path: Path) -> None:
    """A USER_DATA_PATH in .env must not overwrite the one the caller exported."""
    if not doc.exists():
        pytest.skip(f"{doc} not present in this layout")

    result = _run_recipe(
        _extract_dotenv_load(doc),
        tmp_path,
        {"USER_DATA_PATH": "/from/caller", "OPENAI_API_KEY": "key-from-caller"},
    )

    assert result["USER_DATA_PATH"] == "/from/caller"
    assert result["OPENAI_API_KEY"] == "key-from-caller"


@pytest.mark.parametrize("doc", RECIPE_DOCS, ids=lambda p: p.parent.name + "/" + p.name)
def test_recipe_still_fills_missing_values(doc: Path, tmp_path: Path) -> None:
    """Protecting exported values must not stop .env from filling the gaps."""
    if not doc.exists():
        pytest.skip(f"{doc} not present in this layout")

    result = _run_recipe(_extract_dotenv_load(doc), tmp_path, {})

    assert result["USER_DATA_PATH"] == "/from/dotenv"
    assert result["OPENAI_API_KEY"] == "key-from-dotenv"
    assert result["ONLY_IN_DOTENV"] == "filled"


@pytest.mark.parametrize("doc", RECIPE_DOCS, ids=lambda p: p.parent.name + "/" + p.name)
def test_recipe_lets_dotenv_fill_a_blank_export(doc: Path, tmp_path: Path) -> None:
    """An exported-but-empty value is a gap, matching install.sh's restore rule.

    Restoring the blank would discard .env's value, and the recipes that resolve
    ``${USER_DATA_PATH:-/workspace/hyperloom}`` would then silently write to the
    pod-local default -- the very redirection this protection exists to prevent.
    """
    if not doc.exists():
        pytest.skip(f"{doc} not present in this layout")

    result = _run_recipe(_extract_dotenv_load(doc), tmp_path, {"USER_DATA_PATH": ""})

    assert result["USER_DATA_PATH"] == "/from/dotenv"


def test_credential_only_recipe_keeps_the_callers_key(tmp_path: Path) -> None:
    """install.sh snapshots the same credential vars for this exact reason.

    A stale token in .env must not replace the one the caller exported, even
    though this recipe filters the load down to credentials.
    """
    if not CREDENTIAL_ONLY_DOC.exists():
        pytest.skip(f"{CREDENTIAL_ONLY_DOC} not present in this layout")

    fragment = _extract_dotenv_load(CREDENTIAL_ONLY_DOC)

    kept = _run_recipe(fragment, tmp_path, {"OPENAI_API_KEY": "key-from-caller"})
    assert kept["OPENAI_API_KEY"] == "key-from-caller"

    filled = _run_recipe(fragment, tmp_path, {})
    assert filled["OPENAI_API_KEY"] == "key-from-dotenv"
