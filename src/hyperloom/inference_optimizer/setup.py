# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Backend for the Hyperloom setup skill.

The skill writes ``.env`` in the user's target directory, then invokes this
module with ``PYTHONPATH=<target> python3 -m ...``. This backend locates the
packaged installers and runs the bare-metal setup flow with the target
directory as the runtime/config root.

Kept directly under the lightweight ``hyperloom.inference_optimizer`` package
(not under ``cli/``) so ``python -m hyperloom.inference_optimizer.setup`` starts
with zero third-party imports; runtime deps are installed later by install.sh.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_ASSETS_DIR = Path(__file__).resolve().parent / "assets"
_INSTALL_BAREMETAL_SH = _ASSETS_DIR / "install_baremetal.sh"
_PACKAGE_SKILL = Path(__file__).resolve().parent / "SKILL.md"

_AMBIENT_LLM_ENV_PREFIXES = ("ANTHROPIC_", "OPENAI_", "DEEPSEEK_")
_AMBIENT_LLM_ENV_KEYS = {
    "SAFE_API_KEY",
    "LLM_GATEWAY_KEY",
    "CLAUDE_MODEL",
    "CODEX_MODEL",
    "DEEPSEEK_MODEL",
    "GEAK_CLAUDE_MODEL",
}
_AMBIENT_RUNTIME_ENV_KEYS = {
    "USER_DATA_PATH",
    "HYPERLOOM_RUNTIME_DIR",
    "KERNEL_AGENT_ENV",
    "HYPERLOOM_ROOT",
    "HYPERLOOM_KERNEL_AGENT_ROOT",
    "KERNEL_AGENT_ROOT",
    "FRAMEWORK_AGENT_ROOT",
    "HYPERLOOM_SKILL_PATH",
    "PYTHONPATH",
}


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


def _setup_dotenv_is_authoritative(env_file: Path) -> bool:
    """Whether .env came from the interactive setup skill flow."""
    if not env_file.is_file():
        return False
    try:
        dotenv_text = env_file.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return "HYPERLOOM_RUN_MODE=" in dotenv_text


def _scrub_ambient_setup_env(env: dict[str, str], env_file: Path) -> None:
    """Keep the setup skill's freshly-written .env authoritative.

    Ambient shell variables left behind by local Claude / LiteLLM setup must not
    override the setup-written .env and get persisted back into it by
    install_baremetal.sh. Runtime path variables from an earlier Hyperloom
    workspace must not make a new ``pip --target .`` workspace write artifacts
    into the old session root. Only scrub when the .env already exists so
    standalone non-interactive installer use can still rely on process env.
    """
    if not _setup_dotenv_is_authoritative(env_file):
        return
    for key in list(env):
        if (
            key in _AMBIENT_LLM_ENV_KEYS
            or key in _AMBIENT_RUNTIME_ENV_KEYS
            or key.startswith(_AMBIENT_LLM_ENV_PREFIXES)
        ):
            env.pop(key, None)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    installer = _INSTALL_BAREMETAL_SH
    if not installer.is_file():
        print(f"[hyperloom setup] ERROR: installer not found at {installer}", file=sys.stderr)
        return 1

    root = Path.cwd().resolve()
    env_file = root / ".env"
    env = os.environ.copy()
    _scrub_ambient_setup_env(env, env_file)
    env["REPO_ROOT"] = str(root)
    env["HYPERLOOM_SKILL_PATH"] = str(_PACKAGE_SKILL)
    if _setup_dotenv_is_authoritative(env_file):
        env["HYPERLOOM_SETUP_ENV_AUTHORITATIVE"] = "1"

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
