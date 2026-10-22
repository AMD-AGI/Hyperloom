# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Making an autonomous branch's agent ready to run: its preflight.

The agent's source is fetched once into the branch's directory. Each attempt then asks the onboarding model, which
reads that checkout, for the manifest (:mod:`agent_manifest`), writes the agent's config from it, runs its install
commands in the branch's sandbox and starts it. An attempt fails with one line that says which step failed and what
the step printed last, and the next attempt's prompt carries every earlier line, so the model corrects the manifest
instead of repeating it. After :data:`ATTEMPTS` failures the preflight raises
:class:`~hyperloom_multibranch.specialist.runtime.PreflightFailed` with every line.

The agent runs commands, not containers: a ``docker`` on its ``PATH`` (:data:`DOCKER_SHIM`) runs ``docker exec`` and
``docker cp`` in place, for an agent written to drive a container, and refuses everything else.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from hyperloom_multibranch.specialist.agent_manifest import AgentManifest, ManifestError
from hyperloom_multibranch.specialist.runtime import PreflightFailed

ATTEMPTS = 3
#: How long a started agent has to write its keep file or stay up before its start counts as made.
START_SECONDS = 300.0
INSTALL_SECONDS = 1800.0
SOURCE_DIR = "src"
ENCODING = "utf-8"
GIT = "git"
SHIM_DIR = "bin"
#: The lines of an output a failure keeps, the end of the output, where the cause usually is.
TAIL_LINES = 20

DOCKER_SHIM = """#!/usr/bin/env python3
# Runs `docker exec` and `docker cp` in place, for an agent written to drive a container; the sandbox has no daemon.
# A ROCR_VISIBLE_DEVICES the agent passes names cards of its own container, numbered from 0: it is read as indices
# into the sandbox's cards, since set as given it would reach every card of the host.
import os, shutil, subprocess, sys

def own_cards(value):
    cards = os.environ.get("ROCR_VISIBLE_DEVICES", "").split(",")
    if not value.strip():
        return ",".join(cards)
    try:
        return ",".join(cards[int(index)] for index in value.split(","))
    except (ValueError, IndexError):
        sys.exit(f"docker exec: ROCR_VISIBLE_DEVICES={value} names a card beyond the sandbox's {len(cards)}")

args = sys.argv[1:]
if args[:1] == ["exec"]:
    rest, env, cwd = args[1:], dict(os.environ), None
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if flag in ("-e", "--env"):
            name, _, value = rest.pop(0).partition("=")
            env[name] = own_cards(value) if name == "ROCR_VISIBLE_DEVICES" else value
        elif flag in ("-w", "--workdir"):
            cwd = rest.pop(0)
    sys.exit(subprocess.call(rest[1:], cwd=cwd, env=env))
if args[:1] == ["cp"]:
    source, target = (path.split(":", 1)[-1] for path in args[1:3])
    copy = shutil.copytree if os.path.isdir(source) else shutil.copy2
    copy(source, target)
    sys.exit(0)
sys.exit(f"docker {' '.join(args[:1])}: this sandbox runs commands in place and has no docker daemon")
"""


@dataclass(frozen=True)
class AgentSpec:
    """What MB asks of the agent, which its config must be derived from: the workload, the metric it is judged on,
    the server arguments and environment it starts from (the root's), ``workspace``, the framework checkout it edits,
    ``cards``, the GPUs it may use, ``seconds``, how long it may run (0: until MB stops it), and ``port``, the first
    of the ports its servers may listen on."""

    model: str
    framework: str
    tp: int
    precision: str
    gpu_type: str
    workload: Mapping[str, object]
    metric: str
    server_args: str
    envs: Mapping[str, str]
    workspace: str
    cards: tuple[int, ...]
    seconds: float
    port: int

    def to_document(self) -> dict[str, object]:
        return {
            "model": self.model,
            "framework": self.framework,
            "tp": self.tp,
            "precision": self.precision,
            "gpu_type": self.gpu_type,
            "workload": dict(self.workload),
            "metric": self.metric,
            "server_args": self.server_args,
            "envs": dict(self.envs),
            "workspace": self.workspace,
            "cards": list(self.cards),
            "seconds": self.seconds,
            "port": self.port,
        }


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    output: str


class AgentProcess(Protocol):
    """A command started in the sandbox."""

    @property
    def returncode(self) -> int | None: ...

    async def output(self) -> str: ...

    async def kill(self) -> None:
        """Kill it and everything it started."""


class AgentSandbox(Protocol):
    """The branch's sandbox: its own cards, the session's image, a shell; commands run with ``env`` added."""

    async def run(self, command: str, cwd: Path, env: Mapping[str, str], timeout: float) -> CommandResult: ...

    async def start(self, command: str, cwd: Path, env: Mapping[str, str]) -> AgentProcess: ...


#: Asks the onboarding model ``prompt`` with the agent's checkout as its working directory; its answer.
Onboard = Callable[[Path, str], Awaitable[str]]


@dataclass(frozen=True)
class OnboardingSetup:
    sandbox: AgentSandbox
    onboard: Onboard
    attempts: int = ATTEMPTS
    start_seconds: float = START_SECONDS
    install_seconds: float = INSTALL_SECONDS
    poll_seconds: float = 5.0
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    clock: Callable[[], float] = time.monotonic


@dataclass(frozen=True)
class ReadyAgent:
    """An agent preflight made ready: its manifest, its checkout and the process it runs in."""

    manifest: AgentManifest
    source_dir: Path
    process: AgentProcess


class AttemptFailed(RuntimeError):
    """One onboarding attempt failed; the message says at which step and what the step printed last."""


async def make_ready(source: str, spec: AgentSpec, agent_dir: Path, setup: OnboardingSetup) -> ReadyAgent:
    """Fetch ``source`` into ``agent_dir``, onboard it for ``spec`` and start it; raise
    :class:`~hyperloom_multibranch.specialist.runtime.PreflightFailed` with one line per failed attempt."""

    failures: list[str] = []
    for attempt in range(1, setup.attempts + 1):
        try:
            return await _attempt(source, spec, agent_dir, setup, tuple(failures))
        except (AttemptFailed, OSError) as error:
            failures.append(f"attempt {attempt}: {error}")
    raise PreflightFailed(failures)


async def _attempt(
    source: str, spec: AgentSpec, agent_dir: Path, setup: OnboardingSetup, failures: tuple[str, ...]
) -> ReadyAgent:
    checkout = agent_dir / SOURCE_DIR
    commit = await _fetch(source, checkout)
    answer = await setup.onboard(checkout, onboarding_prompt(spec, source, checkout, commit, failures))
    try:
        manifest = AgentManifest.read(_json_object(answer))
    except (ManifestError, ValueError) as error:
        raise AttemptFailed(f"the manifest: {error}") from error
    (checkout / manifest.config_file).parent.mkdir(parents=True, exist_ok=True)
    (checkout / manifest.config_file).write_text(json.dumps(manifest.config, indent=2), encoding=ENCODING)
    env = agent_env(agent_dir)
    for command in manifest.install:
        done = await setup.sandbox.run(command, checkout, env, setup.install_seconds)
        if done.returncode != 0:
            raise AttemptFailed(f"install `{command}` exited {done.returncode}: {tail(done.output)}")
    process = await setup.sandbox.start(manifest.launch, checkout, env)
    started = False
    try:
        await _started(process, manifest, checkout, setup)
        started = True
    finally:
        if not started:
            await process.kill()
    return ReadyAgent(manifest, checkout, process)


async def _started(process: AgentProcess, manifest: AgentManifest, checkout: Path, setup: OnboardingSetup) -> None:
    """Return once the agent wrote its keep file or stayed up :attr:`OnboardingSetup.start_seconds`; raise
    :class:`AttemptFailed` when it exited first."""

    until = setup.clock() + setup.start_seconds
    keep = None if manifest.keep is None else checkout / manifest.keep.path
    while setup.clock() < until and not (keep is not None and keep.exists()):
        if process.returncode is not None:
            raise AttemptFailed(f"`{manifest.launch}` exited {process.returncode}: {tail(await process.output())}")
        await setup.sleep(setup.poll_seconds)


async def _fetch(source: str, checkout: Path) -> str:
    if not (checkout / ".git").exists():
        url, ref = split_source(source)
        clone = [GIT, "clone", "--quiet", url, str(checkout)]
        done = await _git(clone, checkout.parent)
        if done.returncode != 0:
            raise AttemptFailed(f"fetching {url}: {tail(done.output)}")
        if ref:
            checked = await _git([GIT, "checkout", "--quiet", ref], checkout)
            if checked.returncode != 0:
                # Left in place, the clone would pass the next attempt on its default branch.
                shutil.rmtree(checkout)
                raise AttemptFailed(f"checking out {ref}: {tail(checked.output)}")
    head = await _git([GIT, "rev-parse", "HEAD"], checkout)
    return head.output.strip()


async def _git(argv: list[str], cwd: Path) -> CommandResult:
    cwd.mkdir(parents=True, exist_ok=True)
    process = await asyncio.create_subprocess_exec(
        *argv, cwd=cwd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    output, _ = await process.communicate()
    return CommandResult(process.returncode or 0, output.decode(errors="replace"))


def split_source(source: str) -> tuple[str, str]:
    """``source`` as the repository and the ref after its last ``@``; the ref is ``""`` when what comes before that
    ``@`` is no whole repository but a user, as in ``git@host:org/repo`` or ``ssh://git@host/org/repo``."""

    url, at, ref = source.rpartition("@")
    if not at or not ref or ":" in ref or not _is_repository(url):
        return source, ""
    return url, ref


def _is_repository(url: str) -> bool:
    """Whether ``url`` names a whole repository: a path, a URL with a path past its host, or ``host:path``."""

    _, scheme, rest = url.partition("://")
    if scheme:
        return "/" in rest
    return url.startswith(("/", ".", "~")) or ":" in url


def agent_env(agent_dir: Path) -> dict[str, str]:
    """What the agent of ``agent_dir`` runs with on top of its sandbox's environment: :data:`DOCKER_SHIM` first on its
    ``PATH``."""

    shim = agent_dir / SHIM_DIR / "docker"
    shim.parent.mkdir(parents=True, exist_ok=True)
    shim.write_text(DOCKER_SHIM, encoding=ENCODING)
    shim.chmod(0o755)
    return {"PATH": f"{shim.parent}{os.pathsep}{os.environ.get('PATH', '')}"}


def _json_object(answer: str) -> object:
    """The JSON object the onboarding answer holds, inside a code fence or not."""

    match = re.search(r"\{.*\}", answer, re.DOTALL)
    if match is None:
        raise ValueError("the answer holds no JSON object")
    return json.loads(match.group(0))


def tail(output: str) -> str:
    """The last :data:`TAIL_LINES` lines of ``output``, on one line."""

    return " | ".join(line.strip() for line in output.strip().splitlines()[-TAIL_LINES:])


def onboarding_prompt(spec: AgentSpec, source: str, checkout: Path, commit: str, failures: tuple[str, ...]) -> str:
    """What the onboarding model is asked: read the agent's ``checkout`` of ``source`` and answer with its manifest for
    ``spec``."""

    earlier = "\n".join(f"- {line}" for line in failures) or "- none"
    document = json.dumps(spec.to_document(), indent=2)
    return ONBOARDING_PROMPT.format(source=source, checkout=checkout, commit=commit, spec=document, failures=earlier)


ONBOARDING_PROMPT = """\
You are onboarding a third-party LLM-serving optimization agent so it can run unattended in a sandbox. Your working
directory is the agent's checkout, {checkout}, a clone of {source} at {commit}. Read its README, its docs and its
entry points before you answer. Every file the agent writes (its config, the packs or scripts the install adds, its session
and run directories) must be in that checkout or in "workspace", never in {source}: give absolute paths under them.

It will optimize this workload, and every value its config needs must come from here, not be invented. "server_args"
and "envs" are what the root adds to a server of "model" at "tp"; the agent's own model, host, port and TP flags come
on top of them:
{spec}

The sandbox:
- has the session's image and framework and no docker daemon. It sees only the GPUs in "cards", numbered from 0
  (ROCR_VISIBLE_DEVICES is set): a GPU id the agent's config gives is one of those, 0 the first; never set
  ROCR_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES in a command of your own. A `docker` on PATH runs `docker exec [-w DIR] [-e K=V] CONTAINER CMD...` as CMD in DIR and `docker cp
  C:SRC DST` as a copy, so an agent written to drive a container works when its config names any container;
- shares the host with other servers: `pkill`, `pgrep` and `killall` reach only the agent's own processes, but turn
  off any other sweep that kills what holds GPU memory, and give the agent's servers a port no one else uses;
- the agent edits the framework source in "workspace", in place, and launches its servers from it;
- "seconds" is how long the agent may run, 0 when MB stops it: then give it a time budget it cannot reach.

Answer with one JSON object and nothing else, with exactly these fields:
- "source": "{source}", "commit": "{commit}";
- "install": shell commands run once in the checkout, [] when none;
- "config_file": where the agent's config goes, relative to the checkout; "config": that config, as an object;
- "launch": the shell command that runs the agent end to end from the checkout; "resume": the one that continues a
  run after a crash, "" when launching again does;
- "keep": how a kept change shows, {{"path": file relative to the checkout, "field": dotted path of the counter (a
  number or a list) as the file has it once the first keep is in, or "" when every change of the file is a keep,
  "where": {{field: value}} that a list item must match}}, or null when the agent keeps nothing and only the state its
  workspace ends in counts;
- "config_source": {{"path": file relative to "workspace" holding the server arguments and environment the agent runs
  with, "server_args": the field of the arguments string, "envs": the field of the environment object}}, or null;
- "summary_field": dotted path in the keep file to the agent's own words for its latest keep (an integer step indexes
  a list, -1 its last item), "" when none;
- "needs": what it requires, from "anthropic", "openai", "network", "docker".

Earlier attempts that failed, to correct rather than repeat:
{failures}
"""


__all__ = [
    "ATTEMPTS",
    "ENCODING",
    "SOURCE_DIR",
    "DOCKER_SHIM",
    "ONBOARDING_PROMPT",
    "AgentProcess",
    "AgentSandbox",
    "AgentSpec",
    "AttemptFailed",
    "CommandResult",
    "OnboardingSetup",
    "ReadyAgent",
    "agent_env",
    "make_ready",
    "onboarding_prompt",
    "split_source",
    "tail",
]
