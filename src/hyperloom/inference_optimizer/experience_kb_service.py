# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The workspace's local Experience KB service: its ``.env`` entries, its data home, and its on-demand start."""

from __future__ import annotations

import argparse
import os
import re
import secrets
import sys
from pathlib import Path

from hyperloom.common.llm_config import DEFAULT_CLAUDE_MODEL
from hyperloom.inference_optimizer.session.paths import workspace_root
from hyperloom_kb import (
    LocalService,
    LocalServiceError,
    RemoteClientError,
    RemoteConfig,
    ensure_local_service,
    is_loopback,
)

LOCAL_URL = "http://127.0.0.1:8787"
SERVICE_DIR = "experience-kb"
_PLACEHOLDER = "<PLEASE_FILL_IN>"
_PLANNER_MODEL_KEYS = ("LOCAL_KB_PLANNER_MODEL", "CLAUDE_MODEL", "ANTHROPIC_MODEL")


def service_home() -> Path:
    return workspace_root() / SERVICE_DIR


def spool_root() -> Path:
    """Writes the service has not accepted yet; beside its data, so a removed container keeps them."""

    return service_home() / "spool"


def init_env(env_file: Path) -> dict[str, str]:
    """Point ``env_file`` at the local service with a generated token, keeping any value it already sets."""

    lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.is_file() else []
    defaults = {"HYPERLOOM_KB_URL": LOCAL_URL, "HYPERLOOM_KB_TOKEN": secrets.token_urlsafe(32)}
    status: dict[str, str] = {}
    for key, default in defaults.items():
        assignment = re.compile(rf"^\s*(?:export\s+)?{key}\s*=(.*)$")
        index = next((i for i in reversed(range(len(lines))) if assignment.match(lines[i])), None)
        match = None if index is None else assignment.match(lines[index])
        value = "" if match is None else match.group(1).strip().strip("\"'")
        if value and value != _PLACEHOLDER:
            status[key] = "kept"
            continue
        if index is None:
            lines.append(f"{key}={default}")
        else:
            lines[index] = f"{key}={default}"
        status[key] = "written"
    if "written" in status.values():
        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return status


def ensure_service() -> LocalService | None:
    """Bring the configured loopback service to serving; any other URL names a service this workspace does not run."""

    config = RemoteConfig.from_env()
    if config is None or not is_loopback(config.base_url):
        return None
    env = dict(os.environ)
    if not any(env.get(key) for key in _PLANNER_MODEL_KEYS):
        env["LOCAL_KB_PLANNER_MODEL"] = DEFAULT_CLAUDE_MODEL
    return ensure_local_service(config, service_home(), env=env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hyperloom.inference_optimizer.experience_kb_service")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-env", help="Write the local service URL and a generated token into .env.")
    init.add_argument("--env-file", type=Path, default=Path(".env"))
    commands.add_parser("ensure", help="Start the local service unless it already serves, then check its health.")
    args = parser.parse_args(argv)

    if args.command == "init-env":
        for key, status in init_env(args.env_file).items():
            print(f"{key}: {status}")
        return 0
    try:
        service = ensure_service()
    except (LocalServiceError, RemoteClientError) as exc:
        print(f"Experience KB service failed: {exc}", file=sys.stderr)
        return 1
    if service is None:
        print("HYPERLOOM_KB_URL does not name a local Experience KB service", file=sys.stderr)
        return 1
    state = "started" if service.process is not None else "already running"
    print(
        f"Experience KB service {state} at {os.environ['HYPERLOOM_KB_URL']}: "
        f"{service.health.get('experience_count', 0)} Experiences under {service_home()}"
    )
    return 0


__all__ = ["LOCAL_URL", "SERVICE_DIR", "ensure_service", "init_env", "main", "service_home", "spool_root"]


if __name__ == "__main__":
    raise SystemExit(main())
