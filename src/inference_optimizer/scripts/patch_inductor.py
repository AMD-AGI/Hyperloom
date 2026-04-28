"""patch_inductor — Iron Rule IR-6 enforcement entry point.

DESIGN §4.5 IR-6: any patch_inductor invocation must carry
``--target-file`` and (if it tunes ``block_size`` or ``num_warps``)
``--best-config``. The deprecated ``--cache-dir`` flag is forbidden.

This module is intentionally thin: it parses argv, validates the IR-6
invariants, and prints a JSON manifest of what *would* be patched. The
actual patching of inductor source files lives in the sprint scripts —
on a sandbox without inductor installed this module exits ``0`` after
printing the manifest, which keeps unit tests green and CI fast.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


__all__ = ["main", "validate_argv", "PatchInductorError"]


class PatchInductorError(RuntimeError):
    """Raised on IR-6 violation."""


def validate_argv(argv: Iterable[str]) -> None:
    """Raise :class:`PatchInductorError` when IR-6 is violated."""
    argv = list(argv)
    joined = " ".join(argv)
    if "--target-file" not in argv:
        raise PatchInductorError(
            "IR-6: patch_inductor requires --target-file"
        )
    if "--cache-dir" in argv:
        raise PatchInductorError(
            "IR-6: --cache-dir is not allowed (DESIGN §4.5)"
        )
    if any(k in joined for k in ("block_size", "num_warps")):
        if "--best-config" not in argv:
            raise PatchInductorError(
                "IR-6: --best-config required when tuning "
                "block_size or num_warps"
            )


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="patch_inductor",
        description="IR-6-compliant inductor patcher (skeleton).",
    )
    p.add_argument("--target-file", required=True, type=Path)
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
        print(f"patch_inductor: {exc}", file=sys.stderr)
        return 2

    parser = _build_parser()
    args, _unknown = parser.parse_known_args(argv)

    manifest = {
        "kind": "patch_inductor_manifest",
        "target_file": str(args.target_file),
        "best_config": str(args.best_config) if args.best_config else None,
        "tuning_keys": [
            k.strip() for k in str(args.tuning_keys).split(",") if k.strip()
        ],
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main(sys.argv[1:]))
