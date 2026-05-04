"""Import-compatible package facade for the legacy ``inference-optimizer`` tree."""

from __future__ import annotations

from pathlib import Path

_LEGACY_PACKAGE_ROOT = Path(__file__).resolve().parent.parent / "inference-optimizer"

if not _LEGACY_PACKAGE_ROOT.is_dir():
    raise ImportError(f"legacy inference-optimizer package root not found: {_LEGACY_PACKAGE_ROOT}")

# Keep submodule imports such as ``inference_optimizer.orchestrator`` resolving
# to the existing source tree without renaming the on-disk directory.
__path__ = [str(_LEGACY_PACKAGE_ROOT)]

_legacy_init = _LEGACY_PACKAGE_ROOT / "__init__.py"
exec(compile(_legacy_init.read_text(encoding="utf-8"), str(_legacy_init), "exec"), globals())
