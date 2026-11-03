"""Arbor fires its lifecycle's events (arbor/lifecycle.py ``EVENTS``): each is journaled, the pack's ``trigger`` skills
for it run, and a host's callbacks for it are called. None of them holds the session up, unless a skill says
``blocking: true``.

- **The journal:** ``<session>/lifecycle.jsonl`` gets ``{at, kind: "event", name, iteration, ...}``, the feed a host
  follows (docs/hosts.md).
- **Trigger skills:** the pack's ``triggers:`` lists them by event.
  - **How each runs:** in an agent turn of its own, in a process of its own (``python3 -m arbor.triggers run ...``),
    dispatched as ``/<skill> <brief>``. Its log is under ``<session>/runs/``, and its end goes to the journal (kind
    ``trigger``).
  - **A blocking skill** (``metadata.arbor.blocking: true``) runs to its end before the session goes on.
- **Callbacks:** a host registers them in the config's component block, ``"callbacks": {"<event>": [<command>, ...]}``.
  - **A command** is a string (split as a shell would, never run by one) or a list of arguments.
  - **How it runs:** started without waiting for it, with ``ARBOR_EVENT``, ``ARBOR_SESSION`` and ``ARBOR_EVENT_JSON``
    (the journal entry) in its environment.
  - **Each call** goes to the journal (kind ``callback``).

**Fired today:** ``session_start``, ``iteration_start``, ``keep``, ``revert``, ``probe``, ``invalid``,
``agent_failed``, ``map_set``, ``round_end`` and ``end``. The other events are named, and fired as their record points
are wired.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

from arbor import lifecycle

#: Events the driver fires today (the rest of ``lifecycle.EVENT_NAMES`` are named, not yet fired).
FIRED = ("session_start", "iteration_start", "keep", "revert", "probe", "invalid", "agent_failed", "map_set",
         "round_end", "end")
#: How long a blocking trigger skill may take before the session goes on without it.
BLOCKING_TIMEOUT_S = 1800
#: How a process is started without waiting for it, and run to its end: read when an event fires, so a test can
#: replace them here.
SPAWN = subprocess.Popen
RUN = subprocess.run


def callback_errors(callbacks: object) -> list[str]:
    """What is wrong with a component block's ``callbacks``."""

    if not isinstance(callbacks, dict):
        return [f"component.callbacks must map events to commands: {callbacks!r}"]
    errors = []
    for event, commands in callbacks.items():
        if event not in lifecycle.EVENT_NAMES:
            errors.append(f"component.callbacks: unknown event {event!r} (events: {', '.join(lifecycle.EVENT_NAMES)})")
        elif not isinstance(commands, list) or not commands or not all(_argv(c) for c in commands):
            errors.append(f"component.callbacks.{event} must be a list of commands (strings or argument lists)")
    return errors


def _argv(command: object) -> list[str]:
    if isinstance(command, str):
        try:
            return shlex.split(command)
        except ValueError:
            return []
    if isinstance(command, list) and command and all(isinstance(a, str) and a for a in command):
        return list(command)
    return []


def callbacks(config: dict, event: str) -> list[list[str]]:
    """The host's commands for ``event``, as argument lists; ``[]`` outside component mode."""

    if (config or {}).get("mode") != "component":
        return []
    block = (config.get("component") or {}).get("callbacks") or {}
    return [argv for argv in (_argv(c) for c in block.get(event) or []) if argv]


def fire(campaign, session_dir, arbor_root, config_path, event: str, iteration: int | None = None,
         spawn=None, run=None, **detail) -> dict:
    """Fire ``event``: journal it, start its trigger skills and the host's callbacks; the journal entry. Nothing
    raised here reaches the session: an event is a notice, never a reason to stop."""

    spawn, run = spawn or SPAWN, run or RUN
    entry = lifecycle.record(session_dir, "event", event, iteration=iteration, **detail)
    try:
        _start_skills(campaign, session_dir, arbor_root, config_path, event, iteration, spawn, run)
        _call_back(campaign, session_dir, event, entry, spawn)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        lifecycle.record(session_dir, "event_error", event, iteration=iteration, error=str(exc)[:300])
    return entry


def _blocking(arbor_root, skill: str) -> bool:
    from arbor import skillcheck

    found = skillcheck.find(Path(arbor_root)).get(skill)
    return bool(found and found.arbor.get("blocking") is True)


def _start_skills(campaign, session_dir, arbor_root, config_path, event, iteration, spawn, run) -> None:
    plan = lifecycle.declared(campaign)
    for skill in plan.triggered(event) if plan is not None else ():
        argv = [sys.executable, "-m", "arbor.triggers", "run", "--config", str(config_path), "--arbor-root",
                str(arbor_root), "--session", str(session_dir), "--skill", skill, "--event", event,
                "--iteration", str(iteration if iteration is not None else -1)]
        log = Path(session_dir) / "runs" / f"trigger-{skill.replace(':', '__')}-{event}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        with log.open("a", encoding="utf-8") as out:
            if _blocking(arbor_root, skill):
                run(argv, stdout=out, stderr=subprocess.STDOUT, timeout=BLOCKING_TIMEOUT_S, check=False)
            else:
                spawn(argv, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True)
        lifecycle.record(session_dir, "trigger", skill, event=event, iteration=iteration,
                         status="ran" if _blocking(arbor_root, skill) else "started")


def _call_back(campaign, session_dir, event, entry, spawn) -> None:
    for argv in callbacks(getattr(campaign, "config", None) or {}, event):
        env = {**os.environ, "ARBOR_EVENT": event, "ARBOR_SESSION": str(session_dir),
               "ARBOR_EVENT_JSON": json.dumps(entry)}
        spawn(argv, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
              start_new_session=True)
        lifecycle.record(session_dir, "callback", event, argv=argv)


def run_skill(config_path, arbor_root, session_dir, skill: str, event: str, iteration: int,
              run_agent=None) -> int:
    """Run the trigger skill ``skill`` for ``event``: a brief, then one agent turn; its end goes to the journal."""

    from arbor.pack import load_campaign

    campaign = load_campaign(config_path, arbor_root)
    brief = Path(session_dir) / "runs" / f"trigger-{skill.replace(':', '__')}-{event}.brief.md"
    brief.parent.mkdir(parents=True, exist_ok=True)
    brief.write_text("\n".join([
        f"# Brief: {skill}, on {event}, iteration {iteration}",
        "",
        "Arbor runs this skill because an event it is registered for fired. Do its work, without changing the served",
        "tree: the session goes on beside you.",
        "",
        f"- **event:** `{event}`; the journal's entries are in `{Path(session_dir) / lifecycle.JOURNAL}`",
        f"- **session:** `{session_dir}`",
        f"- **config:** `{config_path}`",
        f"- **campaign:** {getattr(campaign, 'identity', '')}: {getattr(campaign, 'goal', '')}",
    ]) + "\n", encoding="utf-8")
    plan = lifecycle.declared(campaign)
    try:
        if run_agent is None:
            from arbor.policy.claude_loop import _run_claude
            text = asyncio.run(_run_claude(f"/{skill} {brief}", str(arbor_root), allowed_tools=["Bash", "Read", "Edit"],
                                           label=f"trigger:{skill}", skills=[],
                                           plugins=lifecycle.plugin_paths(campaign, arbor_root)))
        else:
            text = run_agent(f"/{skill} {brief}")
    except Exception as exc:  # noqa: BLE001 - recorded, and the session never waits on it
        lifecycle.record(session_dir, "trigger", skill, event=event, iteration=iteration, status="failed",
                         error=str(exc)[:300])
        return 1
    lifecycle.record(session_dir, "trigger", skill, event=event, iteration=iteration, status="done",
                     uses=list(plan.uses(lifecycle.ITERATE)) if plan else [], chars=len(text or ""))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python3 -m arbor.triggers")
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run", help="Run one trigger skill for an event (Arbor starts this itself).")
    for name in ("--config", "--arbor-root", "--session", "--skill", "--event"):
        one.add_argument(name, required=True)
    one.add_argument("--iteration", type=int, default=-1)
    args = parser.parse_args(argv)
    return run_skill(args.config, args.arbor_root, args.session, args.skill, args.event, args.iteration)


if __name__ == "__main__":
    sys.exit(main())
