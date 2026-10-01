# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The workspace's local Experience KB service: its ``.env`` entries, data home, on-demand start, and global sync."""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import secrets
import socket
import sys
from pathlib import Path

from hyperloom.common.env import EnvValueError, env_bool
from hyperloom.common.llm_config import DEFAULT_CLAUDE_MODEL
from hyperloom.inference_optimizer.session.paths import workspace_root
from hyperloom_kb import (
    GLOBAL_URL_ENV,
    LocalService,
    LocalServiceError,
    RemoteClient,
    RemoteClientError,
    RemoteConfig,
    SyncUnavailable,
    ensure_local_service,
    global_config_from_env,
    is_loopback,
)
from hyperloom_kb.schema import JsonValue

log = logging.getLogger(__name__)

# A workspace's service port is drawn from here, below the ephemeral range a later client socket could take.
LOCAL_PORTS = range(20_000, 30_000)
# What a run waits on the local service for one read, write, or health check: above the planner's 20 s default, so a
# read fails only when the planner does. Push and pull keep the client default, since the service forwards a whole
# batch to the global KB inside one request.
REQUEST_TIMEOUT_SECONDS = 30.0
SERVICE_DIR = "experience-kb"
AUTO_PUSH_ENV = "HYPERLOOM_KB_AUTO_PUSH"
_PLACEHOLDER = "<PLEASE_FILL_IN>"
_PLANNER_MODEL_KEYS = ("LOCAL_KB_PLANNER_MODEL", "CLAUDE_MODEL", "ANTHROPIC_MODEL")


def service_home() -> Path:
    return workspace_root() / SERVICE_DIR


def spool_root() -> Path:
    """Writes the service has not accepted yet; beside its data, so a removed container keeps them."""

    return service_home() / "spool"


def _port_is_free(port: int) -> bool:
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def local_url(workspace: Path) -> str:
    """The loopback URL of ``workspace``'s own service: a port derived from its path, past any port in use.

    Two workspaces on one host land on different ports even while neither service runs.
    """

    start = int(hashlib.sha256(str(workspace.resolve()).encode()).hexdigest(), 16) % len(LOCAL_PORTS)
    for offset in range(len(LOCAL_PORTS)):
        port = LOCAL_PORTS[(start + offset) % len(LOCAL_PORTS)]
        if _port_is_free(port):
            return f"http://127.0.0.1:{port}"
    raise LocalServiceError(f"no free port in {LOCAL_PORTS.start}-{LOCAL_PORTS.stop - 1} for the Experience service")


def init_env(env_file: Path) -> dict[str, str]:
    """Point ``env_file`` at its workspace's local service with a generated token, keeping any value it already sets."""

    lines = env_file.read_text(encoding="utf-8").splitlines() if env_file.is_file() else []
    defaults = {
        "HYPERLOOM_KB_URL": local_url(env_file.resolve().parent),
        "HYPERLOOM_KB_TOKEN": secrets.token_urlsafe(32),
    }
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


def ensure_service(*, restart: bool = True) -> LocalService | None:
    """Bring the configured loopback service to serving; any other URL names a service this workspace does not run.

    ``restart`` applies this environment's settings to a service started with others, which a launch does. Without it
    such a service keeps serving, and the result says why it is stale.
    """

    config = RemoteConfig.from_env(timeout_seconds=REQUEST_TIMEOUT_SECONDS)
    if config is None or not is_loopback(config.base_url):
        return None
    env = dict(os.environ)
    if not any(env.get(key) for key in _PLANNER_MODEL_KEYS):
        env["LOCAL_KB_PLANNER_MODEL"] = DEFAULT_CLAUDE_MODEL
    return ensure_local_service(config, service_home(), env=env, restart=restart)


def sync_with_global(direction: str) -> dict[str, JsonValue]:
    """``push`` or ``pull`` through the workspace's local service as it runs; a run it may be serving is never stopped."""

    if global_config_from_env(os.environ) is None:
        raise SyncUnavailable(f"{GLOBAL_URL_ENV} is not configured")
    config = RemoteConfig.from_env(spool_root=spool_root())
    if config is None or not is_loopback(config.base_url):
        raise SyncUnavailable("HYPERLOOM_KB_URL does not name a local Experience KB service")
    service = ensure_service(restart=False)
    if service is not None and service.stale:
        log.warning(
            "The Experience KB service runs with other settings than this environment's (%s); %s uses it as it "
            "runs, and the next optimize launch or `ensure` applies them",
            service.stale,
            direction,
        )
    client = RemoteClient(config)
    if direction == "pull":
        from hyperloom.inference_optimizer.experience_collect import mapping_schema_ref

        return client.pull(mapping_schema_ref())
    # Writes spooled while the service was down belong to this workspace too; deliver them before pushing.
    client.flush_spool()
    return client.push()


def check_auto_push() -> bool:
    """Whether a run pushes at its end; a switch or global KB it cannot use is a warning, never a failed run."""

    try:
        enabled = env_bool(AUTO_PUSH_ENV)
        if enabled and global_config_from_env(os.environ) is None:
            raise SyncUnavailable(f"{AUTO_PUSH_ENV} is on but {GLOBAL_URL_ENV} is not configured")
    except (EnvValueError, SyncUnavailable) as exc:
        log.warning("Experience KB auto push is off for this run: %s", exc)
        return False
    return enabled


def auto_push() -> None:
    """Push a run's newly written Experiences when the workspace opted in; a failed push never fails the run."""

    if not check_auto_push():
        return
    try:
        report = sync_with_global("push")
    except (LocalServiceError, RemoteClientError, SyncUnavailable, OSError) as exc:
        log.warning("Experience KB auto push failed; the next push sends these Experiences: %s", exc)
        return
    if report["status"] != "completed" or report["rejected"]:
        log.warning("Experience KB auto push did not finish; the next push resumes it: %s", _summary("push", report))
    else:
        log.info("%s", _summary("push", report))


def _summary(direction: str, report: dict[str, JsonValue]) -> str:
    keys = ("created", "unchanged", "skipped", *(("held_back",) if direction == "push" else ()))
    counts = ", ".join(f"{report[key]} {key}" for key in keys)
    rejected = report["rejected"] if isinstance(report["rejected"], list) else []
    line = f"Experience KB {direction} with {report['global_url']}: {counts}, {len(rejected)} rejected"
    saved = report.get("saved")
    if isinstance(saved, dict):
        line += f"; the state before it is saved as label {saved['label_id']} ({saved['name']})"
    return line if report["status"] == "completed" else f"{line}; stopped: {report.get('error', '')}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m hyperloom.inference_optimizer.experience_kb_service")
    commands = parser.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-env", help="Write the local service URL and a generated token into .env.")
    init.add_argument("--env-file", type=Path, default=Path(".env"))
    commands.add_parser("ensure", help="Start the local service unless it already serves, then check its health.")
    commands.add_parser("push", help="Send the Experiences written here and not yet pushed to the global KB.")
    commands.add_parser(
        "pull",
        help="Bring this workspace's schema to everything the global KB holds of it; an unlabelled state is labelled "
        "first, so the pull can be undone.",
    )
    args = parser.parse_args(argv)

    if args.command == "init-env":
        for key, status in init_env(args.env_file).items():
            print(f"{key}: {status}")
        return 0
    if args.command in ("push", "pull"):
        try:
            report = sync_with_global(args.command)
        except (LocalServiceError, RemoteClientError, SyncUnavailable) as exc:
            print(f"Experience KB {args.command} failed: {exc}", file=sys.stderr)
            return 1
        print(_summary(args.command, report))
        rejected = report["rejected"] if isinstance(report["rejected"], list) else []
        for item in rejected:
            print(f"  rejected: {item}", file=sys.stderr)
        return 0 if report["status"] == "completed" and not rejected else 1
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


__all__ = [
    "AUTO_PUSH_ENV",
    "LOCAL_PORTS",
    "SERVICE_DIR",
    "auto_push",
    "ensure_service",
    "init_env",
    "local_url",
    "main",
    "service_home",
    "spool_root",
    "check_auto_push",
    "sync_with_global",
]


if __name__ == "__main__":
    raise SystemExit(main())
