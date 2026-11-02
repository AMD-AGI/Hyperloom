# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The workspace's local Experience KB service: its ``.env`` entries, data home, on-demand start, and global sync."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from hyperloom_common.env import EnvValueError, env_bool
from hyperloom_kb import workspace as kb_workspace
from hyperloom_kb.workspace import (
    HOME_ENV,
    LOCAL_PORTS,
    SERVICE_DIR,
    init_env,
    local_url,
)
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
from hyperloom_kb.cli import add_commands, run_command
from hyperloom_kb.collect import load_mapping
from hyperloom_kb.schema import JsonValue

log = logging.getLogger(__name__)

# What a run waits on the local service for one read, write, or health check: above the planner's 20 s default, so a
# read fails only when the planner does. Push and pull keep the client default, since the service forwards a whole
# batch to the global KB inside one request.
REQUEST_TIMEOUT_SECONDS = 30.0
# The packaged mapping a session's Experiences are collected through; its declaration is the schema this workspace
# writes, reads, and syncs.
MAPPING = "hyperloom-sbd-v6"
AUTO_PUSH_ENV = "HYPERLOOM_KB_AUTO_PUSH"


def mapping_schema_ref() -> str:
    """Return the declaration the packaged mapping produces, proving the mapping loads."""

    return load_mapping(MAPPING).declaration.schema_ref


def service_home() -> Path:
    return kb_workspace.service_home(root=workspace_root())


def spool_root() -> Path:
    """Writes the service has not accepted yet; beside its data, so a removed container keeps them."""

    return service_home() / "spool"


def ensure_service(*, restart: bool = True) -> LocalService | None:
    """Bring the configured loopback service to serving; any other URL names a service this workspace does not run.

    ``restart`` applies this environment's settings to a service started with others, which a launch does. Without it
    such a service keeps serving, and the result says why it is stale.
    """

    config = RemoteConfig.from_env(timeout_seconds=REQUEST_TIMEOUT_SECONDS)
    if config is None or not is_loopback(config.base_url):
        return None
    return ensure_local_service(config, service_home(), env=kb_workspace.service_env(), restart=restart)


def _workspace_client(command: str) -> RemoteClient:
    """The workspace's local service as it runs, started when nothing serves it; a run it may serve is never stopped."""

    config = RemoteConfig.from_env(spool_root=spool_root())
    if config is None or not is_loopback(config.base_url):
        raise LocalServiceError("HYPERLOOM_KB_URL does not name a local Experience KB service")
    service = ensure_service(restart=False)
    if service is not None and service.stale:
        log.warning(
            "The Experience KB service runs with other settings than this environment's (%s); %s uses it as it "
            "runs, and the next optimize launch or `ensure` applies them",
            service.stale,
            command,
        )
    return RemoteClient(config)


def sync_with_global(direction: str) -> dict[str, JsonValue]:
    """``push`` or ``pull`` through the workspace's local service as it runs; a run it may be serving is never stopped."""

    if global_config_from_env(os.environ) is None:
        raise SyncUnavailable(f"{GLOBAL_URL_ENV} is not configured")
    client = _workspace_client(direction)
    if direction == "pull":
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
    add_commands(commands, schema_ref=mapping_schema_ref())
    args = parser.parse_args(argv)

    if args.command == "init-env":
        for key, status in init_env(args.env_file).items():
            print(f"{key}: {status}")
        return 0
    if hasattr(args, "run"):
        try:
            client = _workspace_client(args.command)
        except (LocalServiceError, RemoteClientError) as exc:
            print(f"Experience KB {args.command} failed: {exc}", file=sys.stderr)
            return 1
        return run_command(client, args)
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
    state = "restarted" if service.restarted else "started" if service.process is not None else "already running"
    print(
        f"Experience KB service {state} at {os.environ['HYPERLOOM_KB_URL']}: "
        f"{service.health.get('experience_count', 0)} Experiences under {service_home()}"
    )
    return 0


__all__ = [
    "AUTO_PUSH_ENV",
    "HOME_ENV",
    "LOCAL_PORTS",
    "MAPPING",
    "SERVICE_DIR",
    "auto_push",
    "ensure_service",
    "init_env",
    "local_url",
    "main",
    "mapping_schema_ref",
    "service_home",
    "spool_root",
    "check_auto_push",
    "sync_with_global",
]


if __name__ == "__main__":
    raise SystemExit(main())
