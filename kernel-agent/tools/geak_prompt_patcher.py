#!/usr/bin/env python3
"""Patch GEAK's bundled ``mini_kernel_strategy_list.yaml`` to remove the
hard-coded ``task_runner.py performance`` example that misleads the
sub-agent LLM into searching the filesystem for a non-existent script.

Background
==========

GEAK / ``minisweagent`` ships a system-prompt YAML at::

    <site-packages>/minisweagent/config/mini_kernel_strategy_list.yaml

The ``profile_kernel`` tool definition in that YAML contains three lines
of *example* command strings whose intent is to teach the LLM that
shell operators (``&&`` / ``cd`` / ``|``) are forbidden::

    - Good example: `command="python3 scripts/task_runner.py performance", workdir="/path/to/project"`
    - Also ok: `command="python3 /absolute/path/to/scripts/task_runner.py performance"`
    - Bad example: `command="cd /path && python3 scripts/task_runner.py performance"`

The example was authored against a fictional repo layout, but the
sub-agent LLM treats ``scripts/task_runner.py performance`` as a real
filename and burns its entire budget running ``find / -name
'task_runner*'`` on WekaFS (30-60 minute wall-clock each). Hyperloom's
own prompt explicitly tells the agent ``DO NOT run find /``, but the
upstream example carries more local weight in the LLM's reasoning and
consistently wins.

This patcher rewrites those three lines to abstract placeholders
(``<your_benchmark.py>``) so the LLM has nothing concrete to chase.
The Forbidden / workdir rules stay intact.

Design contract
===============

* **Idempotent** — running twice is a noop. The sentinel substring
  ``<your_benchmark.py>`` is checked first; when already present the
  function returns ``(True, "already patched")`` without writing.
* **Fail-soft** — every IO / path / parse failure returns ``(False,
  reason)`` so the install script can continue (the patcher is a UX
  hardening, not a correctness gate).
* **Bounded probe** — only the bundled YAML is touched. The function
  refuses to patch arbitrary files even when ``$HYPERLOOM_GEAK_PROMPT_YAML``
  points elsewhere (the override is for tests).
* **Audit trail** — on success the function returns the absolute path
  it patched so the caller can log it; on no-op it returns the same
  path with a different status string.
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
    """Return the absolute path to the bundled GEAK strategy YAML.

    Resolution order:
      1. ``$HYPERLOOM_GEAK_PROMPT_YAML`` env override — primary use is
         in unit tests against a sandbox copy. Production callers do
         NOT set this.
      2. ``importlib.util.find_spec('minisweagent')`` → ``config/...``
         relative to the package root. Works for editable installs and
         wheel installs alike.

    Returns:
        Path | None: Absolute path to the bundled strategy YAML, or
            ``None`` when the override points at a missing file or the
            ``minisweagent`` package is not importable in this
            interpreter (e.g. ``install.sh --check-only`` invoked before
            GEAK's pip install ran).
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
    """Write ``content`` to ``target`` atomically via tempfile + replace.

    Site-packages files are sometimes read concurrently by other
    Python processes (e.g. a long-running GEAK run started before
    install.sh re-runs). Atomic replace prevents a half-written YAML
    from being parsed.

    Args:
        target (Path): Destination file to overwrite in place.
        content (str): Full text to write.
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
    """Patch the bundled YAML in-place; idempotent and fail-soft.

    Returns:
        tuple[bool, str]: ``(ok, message)`` where ``ok`` is True on
            success or already-patched, and False on any fail-soft
            outcome (file missing, upstream block changed, write
            failed, …). ``message`` is a short human-readable status
            suitable for the install-script log.
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
        # Upstream changed the example wording; refuse to guess at the
        # new structure so we never garble the YAML.
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
    """CLI entry for install.sh: prints status and exits 0 on success/noop.

    Exit code 0 always when ``$HYPERLOOM_GEAK_PROMPT_PATCH_REQUIRED != 1``
    (default), so the patcher cannot block install on a fail-soft
    outcome. Set the env to ``1`` to make missing-or-failed patch
    fatal (CI / production guard rails).

    Returns:
        int: Process exit code — 0 on success/noop, or when the patch
            is not marked required; 1 only when the patch failed and
            ``$HYPERLOOM_GEAK_PROMPT_PATCH_REQUIRED == '1'``.
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
