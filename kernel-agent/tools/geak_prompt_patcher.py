#!/usr/bin/env python3
# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Patch GEAK's bundled ``mini_kernel_strategy_list.yaml``.

Rewrites the hard-coded ``task_runner.py`` example to abstract placeholders so
the sub-agent LLM doesn't burn its budget hunting for a non-existent script.
Idempotent (sentinel-guarded), fail-soft (UX hardening, not a correctness gate).
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import tempfile
from pathlib import Path

_SENTINEL = "<your_benchmark.py>"  # presence == already patched

_OLD_BLOCK = (
    '    - Good example: `command="python3 scripts/task_runner.py performance", '
    'workdir="/path/to/project"`\n'
    '    - Also ok: `command="python3 /absolute/path/to/scripts/task_runner.py '
    'performance"`\n'
    '    - Bad example: `command="cd /path && python3 scripts/task_runner.py '
    'performance"`\n'
)
_NEW_BLOCK = (
    '    - Good example: `command="python3 <your_benchmark.py> <args>", '
    'workdir="/path/to/repo"`\n'
    '    - Also ok: `command="python3 /absolute/path/to/<your_benchmark.py> '
    '<args>"`\n'
    '    - Bad example: `command="cd /path && python3 <your_benchmark.py> '
    '<args>"`\n'
)

_DEFAULT_REL_PATH = Path("minisweagent/config/mini_kernel_strategy_list.yaml")


def _locate_yaml() -> Path | None:
    """Locate the bundled GEAK strategy YAML.

    Resolves ``$HYPERLOOM_GEAK_PROMPT_YAML`` (test override) first, then the
    ``minisweagent`` package location.

    Returns:
        The YAML path, or ``None`` when the override file is missing or the
        package isn't importable.
    """
    override = os.environ.get("HYPERLOOM_GEAK_PROMPT_YAML", "").strip()
    if override:
        p = Path(override).expanduser()
        return p if p.is_file() else None
    try:
        spec = importlib.util.find_spec("minisweagent")
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    pkg_root = Path(list(spec.submodule_search_locations)[0]).parent
    candidate = pkg_root / _DEFAULT_REL_PATH
    return candidate if candidate.is_file() else None


def _atomic_write(target: Path, content: str) -> None:
    """Write content to a file atomically via a tempfile and replace.

    Args:
        target: Destination file path.
        content: Text to write.
    """
    with tempfile.NamedTemporaryFile(
        "w", dir=str(target.parent), delete=False, encoding="utf-8",
    ) as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    try:
        shutil.copystat(target, tmp_path)
    except OSError:
        pass
    tmp_path.replace(target)


def ensure_geak_prompt_patched() -> tuple[bool, str]:
    """Patch the bundled GEAK strategy YAML in place.

    The operation is idempotent and fail-soft: it skips when already patched,
    when the package is absent, or when upstream wording has drifted.

    Returns:
        A ``(ok, message)`` tuple where ``ok`` indicates success and
        ``message`` is a human-readable status.
    """
    yaml_path = _locate_yaml()
    if yaml_path is None:
        return False, "minisweagent not installed (skip)"
    try:
        text = yaml_path.read_text(encoding="utf-8")
    except OSError as exc:
        return False, f"cannot read {yaml_path}: {exc}"
    if _SENTINEL in text:
        return True, f"already patched: {yaml_path}"
    if _OLD_BLOCK not in text:
        # Upstream wording changed; refuse to guess and garble the YAML.
        return False, (
            f"upstream example block changed; manual review required: {yaml_path}"
        )
    patched = text.replace(_OLD_BLOCK, _NEW_BLOCK, 1)
    if patched == text:
        return False, f"replace produced no change: {yaml_path}"
    try:
        _atomic_write(yaml_path, patched)
    except OSError as exc:
        return False, f"cannot write {yaml_path}: {exc}"
    return True, f"patched: {yaml_path}"


def main() -> int:
    """Run the patcher as a CLI entry point for ``install.sh``.

    Exits 0 unless the patch fails and
    ``$HYPERLOOM_GEAK_PROMPT_PATCH_REQUIRED == 1``.

    Returns:
        Process exit code (0 on success or soft-skip, 1 on required failure).
    """
    ok, msg = ensure_geak_prompt_patched()
    status = "OK" if ok else "WARN"
    print(f"[geak-prompt-patcher] {status}: {msg}")
    required = os.environ.get("HYPERLOOM_GEAK_PROMPT_PATCH_REQUIRED", "0").strip()
    if not ok and required == "1":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
