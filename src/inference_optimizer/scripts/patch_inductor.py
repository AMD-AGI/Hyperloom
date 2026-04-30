"""patch_inductor — Iron Rule IR-6 (soft) entry point.

DESIGN §4.5 IR-6 — Plan A softened: any patch_inductor invocation
*should* carry ``--target-file`` and (if it tunes ``block_size`` or
``num_warps``) ``--best-config``. The deprecated ``--cache-dir`` flag is
*not recommended*. Violations log a warning to stderr instead of raising,
so a kernel-opt loop with one borderline argv survives instead of
breaking. Set ``INFERENCE_OPTIMIZER_IR6_STRICT=1`` to restore the
original BLOCK behaviour (raise + exit 2).

This module is intentionally thin: it parses argv, validates the IR-6
invariants, and prints a JSON manifest of what *would* be patched. The
actual patching of inductor source files lives in the sprint scripts —
on a sandbox without inductor installed this module exits ``0`` after
printing the manifest, which keeps unit tests green and CI fast.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable


__all__ = [
    "main",
    "validate_argv",
    "ir6_strict_enabled",
    "PatchInductorError",
]


class PatchInductorError(RuntimeError):
    """Raised on IR-6 violation only when strict mode is on."""


_STRICT_ENV = "INFERENCE_OPTIMIZER_IR6_STRICT"
_TRUE_VALUES = frozenset({"1", "true", "yes", "on"})


def ir6_strict_enabled() -> bool:
    """Return True when the legacy hard-block behaviour is requested."""
    return os.environ.get(_STRICT_ENV, "").strip().lower() in _TRUE_VALUES


def _emit_violation(message: str, *, errors: list[str]) -> None:
    """Log a WARN to stderr (soft mode) or accumulate for raise (strict)."""
    print(f"WARNING: patch_inductor IR-6: {message}", file=sys.stderr)
    if ir6_strict_enabled():
        errors.append(message)


def validate_argv(argv: Iterable[str]) -> list[str]:
    """Validate ``argv`` against IR-6.

    In Plan A's soft mode each violation prints a WARNING to stderr and
    the function returns the (possibly empty) list of violation strings.
    Set ``INFERENCE_OPTIMIZER_IR6_STRICT=1`` to restore the legacy
    behaviour where the first violation raises :class:`PatchInductorError`
    immediately. Returning the list (instead of always raising) lets
    callers in soft mode see how many violations fired without aborting.
    """
    argv = list(argv)
    joined = " ".join(argv)
    errors: list[str] = []
    if "--target-file" not in argv:
        _emit_violation(
            "patch_inductor invocation should include --target-file",
            errors=errors,
        )
    if "--cache-dir" in argv:
        _emit_violation(
            "--cache-dir is deprecated (DESIGN §4.5); ignored at runtime",
            errors=errors,
        )
    if any(k in joined for k in ("block_size", "num_warps")):
        if "--best-config" not in argv:
            _emit_violation(
                "--best-config recommended when tuning "
                "block_size or num_warps (otherwise tiling drifts)",
                errors=errors,
            )
    if errors and ir6_strict_enabled():
        # Strict mode: raise on the first accumulated violation message
        # so behaviour matches the pre-Plan-A contract.
        raise PatchInductorError("IR-6: " + "; ".join(errors))
    return errors


def _build_parser() -> argparse.ArgumentParser:
    """Argparse parser. Plan A: ``--target-file`` is no longer
    ``required=True`` — validate_argv emits a soft warning instead, and
    a missing target file just yields an empty ``target_file`` in the
    manifest."""
    p = argparse.ArgumentParser(
        prog="patch_inductor",
        description="IR-6-soft inductor patcher (Plan A).",
    )
    p.add_argument("--target-file", type=Path, default=None)
    p.add_argument("--best-config", type=Path, default=None)
    p.add_argument("--tuning-keys", default="")
    p.add_argument("--dry-run", action="store_true",
                   help="emit the manifest but skip the actual patch")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    try:
        validate_argv(argv)
    except PatchInductorError as exc:
        # Strict mode only — soft mode never raises.
        print(f"patch_inductor: {exc}", file=sys.stderr)
        return 2

    parser = _build_parser()
    args, _unknown = parser.parse_known_args(argv)

    manifest = {
        "kind": "patch_inductor_manifest",
        "target_file": str(args.target_file) if args.target_file else None,
        "best_config": str(args.best_config) if args.best_config else None,
        "tuning_keys": [
            k.strip() for k in str(args.tuning_keys).split(",") if k.strip()
        ],
        "dry_run": bool(args.dry_run),
        "ir6_strict": ir6_strict_enabled(),
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
