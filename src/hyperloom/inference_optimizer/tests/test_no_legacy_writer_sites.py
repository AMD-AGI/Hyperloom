# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Static guard for the retired ``extra_sglang_args`` payload field."""

from __future__ import annotations

import re
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[4]

ALLOWED_FILES: dict[str, str] = {
    "src/hyperloom/agents/kernel/tools/_payload_aliases.py": (
        "standalone kernel-agent shim kept for bare subprocess imports"
    ),
    "src/hyperloom/agents/kernel/tests/test_payload_aliases_shim.py": (
        "tests pin the standalone kernel-agent shim contract"
    ),
    "src/hyperloom/orchestrator/state/shared_state.py": (
        "load-time state.json migration only; canonical writers stay on extra_server_args"
    ),
    "src/hyperloom/inference_optimizer/tests/test_shared_state_evolution.py": (
        "tests pin the load-time state.json migration contract"
    ),
    "src/hyperloom/inference_optimizer/tests/test_no_legacy_writer_sites.py": (
        "this guard names the retired field it scans for"
    ),
    "ci/transform_to_session_summary_v2.py": (
        "load-time session_breakdown.json migration only; reads the retired key "
        "as a fallback and emits canonical extra_server_args"
    ),
    "ci/test_ci_transform_v2.py": (
        "tests pin the session_breakdown.json legacy-arg migration contract"
    ),
}

_LEGACY_PATTERN = re.compile(r"extra_sglang_args")


def _git_tracked_files() -> set[str] | None:
    """Repo-relative POSIX paths tracked by git, or ``None`` when git is unavailable."""
    import subprocess

    try:
        proc = subprocess.run(
            ["git", "-C", str(_REPO_ROOT), "ls-files", "-z"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, ValueError):
        return None
    if proc.returncode != 0:
        return None
    return {p for p in (proc.stdout or "").split("\0") if p}


def _iter_repo_files() -> list[Path]:
    """All non-binary, git-tracked repo files we want to scan."""
    tracked = _git_tracked_files()
    out: list[Path] = []
    for path in _REPO_ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(_REPO_ROOT).as_posix()
        if tracked is not None and rel not in tracked:
            continue
        if any(rel.startswith(skip) for skip in (".git/", "node_modules/", "__pycache__/", "build/", ".venv/")):
            continue
        if path.suffix not in {".py", ".md", ".yaml", ".yml", ".toml", ".sh", ".txt", ".json", ".cfg", ".ini"}:
            continue
        out.append(path)
    return out


def _files_with_legacy_key() -> set[str]:
    """Repo-relative POSIX paths containing the retired key literal."""
    hits: set[str] = set()
    for path in _iter_repo_files():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if _LEGACY_PATTERN.search(text):
            hits.add(path.relative_to(_REPO_ROOT).as_posix())
    return hits


def test_no_legacy_writer_sites_outside_allowlist() -> None:
    actual = _files_with_legacy_key()
    unexpected = sorted(actual - set(ALLOWED_FILES))
    assert not unexpected, (
        "Files mentioning retired 'extra_sglang_args' outside the residual kernel-agent shim allowlist:\n  "
        + "\n  ".join(unexpected)
    )


def test_allowlist_is_minimal() -> None:
    actual = _files_with_legacy_key()
    dead_entries = sorted(set(ALLOWED_FILES) - actual)
    assert not dead_entries, (
        "ALLOWED_FILES entries that no longer contain 'extra_sglang_args':\n  " + "\n  ".join(dead_entries)
    )


def test_allowlist_paths_resolve() -> None:
    missing = [p for p in ALLOWED_FILES if not (_REPO_ROOT / p).exists()]
    assert not missing, "ALLOWED_FILES entries pointing at non-existent paths:\n  " + "\n  ".join(missing)


@pytest.mark.parametrize("path,_reason", sorted(ALLOWED_FILES.items()))
def test_allowlist_entries_have_justification(path: str, _reason: str) -> None:
    reason = ALLOWED_FILES[path]
    assert reason and reason.strip(), f"ALLOWED_FILES[{path!r}] needs a justification"
