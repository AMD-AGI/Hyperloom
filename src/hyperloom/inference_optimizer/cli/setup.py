# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Backend for the Hyperloom setup skill.

The skill writes ``.env`` in the user's target directory, then invokes this
module with ``PYTHONPATH=<target> python3 -m ...``. This backend locates the
packaged installers and runs the bare-metal setup flow with the target
directory as the runtime/config root.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent.parent / "assets"
_INSTALL_BAREMETAL_SH = _ASSETS_DIR / "install_baremetal.sh"
_PACKAGE_SKILL = Path(__file__).resolve().parent.parent / "SKILL.md"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hyperloom setup",
        description=(
            "Run Hyperloom setup from a pip --target installation. "
            "The current directory is used for .env and runtime artifacts. "
            "Unknown arguments are forwarded to the packaged installer."
        ),
    )
    parser.add_argument("--check-only", action="store_true", help="Verify only; do not mutate.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned actions only.")
    parser.add_argument(
        "extra_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded verbatim to the installer (after --).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    installer = _INSTALL_BAREMETAL_SH
    if not installer.is_file():
        print(f"[hyperloom setup] ERROR: installer not found at {installer}", file=sys.stderr)
        return 1

    root = Path.cwd().resolve()
    env = os.environ.copy()
    env.setdefault("REPO_ROOT", str(root))
    env.setdefault("HYPERLOOM_ENV_FILE", str(root / ".env"))
    env.setdefault("HYPERLOOM_SKILL_PATH", str(_PACKAGE_SKILL))

    cmd: list[str] = ["bash", str(installer)]
    if args.check_only:
        cmd.append("--check-only")
    if args.dry_run:
        cmd.append("--dry-run")
    forwarded = [a for a in (args.extra_args or []) if a != "--"]
    cmd.extend(forwarded)

    print(f"[hyperloom setup] running: {' '.join(cmd)}")
    return subprocess.run(cmd, env=env).returncode


if __name__ == "__main__":
    raise SystemExit(main())
