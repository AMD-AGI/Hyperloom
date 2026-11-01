# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Making an autonomous branch's agent ready to run: its preflight.

The agent's source is fetched once into the branch's directory. Each attempt then asks the onboarding model, which
reads that checkout, for the manifest (:mod:`agent_manifest`), writes the agent's config from it, runs its install
commands in the branch's sandbox and starts it. An attempt fails with one line that says which step failed and what
the step printed last, and the next attempt's prompt carries every earlier line, so the model corrects the manifest
instead of repeating it; it starts :data:`RETRY_SECONDS` after the failure. An attempt whose model call could not reach
the gateway or found it overloaded (:class:`GatewayDown`) is no attempt: the next one starts after a pause that doubles
up to :data:`GATEWAY_PAUSE_SECONDS`, for as long as :func:`gateway_wait` allows, and only a failure past that counts.
After :data:`ATTEMPTS` failures the preflight raises :class:`~hyperloom_multibranch.specialist.runtime.PreflightFailed`
with every line, and at once after a model call refused for a reason the next attempt would meet again
(:class:`ModelRefused`).

An agent that writes a snapshot of each keep runs as a component of MB, in the mode Arbor's README describes: MB sets
that mode, its host, the absolute deadline and the commit its keeps are read against in its config itself
(:func:`component_config`), whatever the manifest says of them, and, when the session's root was profiled, the file in
its session that holds the root's op shares (:data:`PROFILE_FILE`). Arbor itself is not onboarded by the model: its
checkout says what it is, and :mod:`arbor_adapter` writes its manifest and config from the spec, which MB then sets
as it sets any agent's in that mode. A config refused then, by the adapter's check or by Arbor as it starts, is the
last attempt (:class:`ConfigRefused`): another would write the same config.

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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from hyperloom.common.workload_defaults import DEFAULT_EP
from hyperloom_multibranch.contracts.blackboard.records import OpShares
from hyperloom_multibranch.specialist.agent_manifest import AgentManifest, ManifestError
from hyperloom_multibranch.specialist.arbor_adapter import (
    BRANCH_RUN,
    EXIT_REFUSED,
    ArborCheckout,
    ArborRefused,
    ArborRun,
    arbor_manifest,
    is_arbor,
    refused_reasons,
)
from hyperloom_multibranch.specialist.runtime import PreflightFailed

ATTEMPTS = 3
#: How long the next attempt waits after a failed one, and the first pause after one that met the gateway down.
RETRY_SECONDS = 60.0
#: The longest pause between attempts that met the gateway down.
GATEWAY_PAUSE_SECONDS = 600.0
#: The longest a preflight waits for the gateway: past an outage of an hour and a half.
GATEWAY_WAIT_SECONDS = 7200.0
#: How long a started agent has to write its keep file or stay up before its start counts as made.
START_SECONDS = 300.0
INSTALL_SECONDS = 1800.0
SOURCE_DIR = "src"
ENCODING = "utf-8"
GIT = "git"
SHIM_DIR = "bin"
#: The lines of an output a failure keeps, the end of the output, where the cause usually is.
TAIL_LINES = 20
#: The file, in the session of an agent run as a component, that holds the op shares of the session's root profile.
PROFILE_FILE = "op_shares.json"
#: What the root's shares are, as the profile file names their source: its op shares, the blackboard's.
ROOT_PROFILE_SOURCE = "PRELUDE root profile"

DOCKER_SHIM = """#!/usr/bin/env python3
# Runs `docker exec` and `docker cp` in place, for an agent written to drive a container; the sandbox has no daemon.
# A device mask the agent passes names cards of its own container, numbered from 0: it is read as indices into the
# sandbox's cards, which HIP_VISIBLE_DEVICES numbers from 0 under ROCR_VISIBLE_DEVICES, and set there alone, since a
# ROCR_VISIBLE_DEVICES set as given would reach other cards of the host.
import os, shutil, subprocess, sys

MASKS = ("ROCR_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "CUDA_VISIBLE_DEVICES")


def own_cards(value):
    cards = os.environ.get("HIP_VISIBLE_DEVICES", "").split(",")
    if not value.strip():
        return ",".join(cards)
    try:
        return ",".join(cards[int(index)] for index in value.split(","))
    except (ValueError, IndexError):
        sys.exit(f"docker exec: a device mask of {value} names a card beyond the sandbox's {len(cards)}")

args = sys.argv[1:]
if args[:1] == ["exec"]:
    rest, env, cwd = args[1:], dict(os.environ), None
    while rest and rest[0].startswith("-"):
        flag = rest.pop(0)
        if flag in ("-e", "--env"):
            name, _, value = rest.pop(0).partition("=")
            if name in MASKS:
                env["HIP_VISIBLE_DEVICES"] = own_cards(value)
            else:
                env[name] = value
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
    """What MB asks of the agent, which its config must be derived from: the workload, the metric it is judged on, the
    server arguments and environment of the commit its branch starts on, ``workspace``, the framework checkout it edits,
    ``cards``, the GPUs it may use, ``seconds``, how long it may run (0: until MB stops it), ``port``, the first of
    the ports its servers may listen on, ``client``, the command the gate measures a server on ``port`` with (empty when
    the session has none to give), ``baseline``, what the gate measured of the metric on the session's root (``None``
    before it did), ``knowledge``, the notes on the integration head the agent reads each iteration, and, for an agent
    run as a component, ``deadline``, the clock time it ends by, ``root``, the commit of ``workspace`` its patches
    are made against, and ``root_shares``, the op shares of the session root's profile (``None`` when it has none),
    which the profile file says came from ``root_shares_source``. Arbor (:mod:`arbor_adapter`) is also given ``ep``,
    the expert-parallel size the workload is served at, and may be given ``plan``, the file of the branch's share of
    the session's directions, ``pack``, the name of the pack it runs (``None`` for the adapter's default),
    ``launch``, the file of the launch the recipe's adds to (``None`` when the recipe's is the whole launch),
    ``dossier``, the dossier the session's dossier pass wrote, for its checkout (``None`` without one), and
    ``review``, how its review gate is handed over (:data:`~hyperloom_multibranch.specialist.arbor_adapter.REVIEW_HOST`
    when MB reviews before its branches open, :data:`~hyperloom_multibranch.specialist.arbor_adapter.REVIEW_OFF` when
    nobody does, ``""`` to leave the config without either); the onboarding model is told none of them."""

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
    client: str = ""
    baseline: float | None = None
    knowledge: str = ""
    deadline: float = 0.0
    root: str = ""
    root_shares: OpShares | None = None
    plan: str | None = None
    ep: int = DEFAULT_EP
    pack: str | None = None
    launch: str | None = None
    dossier: str | None = None
    root_shares_source: str = ROOT_PROFILE_SOURCE
    review: str = ""

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
            "client": self.client,
            "baseline": self.baseline,
            "knowledge": self.knowledge,
            "deadline": self.deadline,
            "root": self.root,
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

    def watched(self) -> str:
        """Start the watch confining the sandbox's processes again when it ended; what to say of it, empty when it
        runs."""


#: Asks the onboarding model ``prompt`` with the agent's checkout as its working directory; its answer.
Onboard = Callable[[Path, str], Awaitable[str]]


@dataclass(frozen=True)
class OnboardingSetup:
    """``run`` is the run of Arbor the preflight starts, when the agent is Arbor: a branch's, or the session's
    dossier pass."""

    sandbox: AgentSandbox
    onboard: Onboard
    run: ArborRun = BRANCH_RUN
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


class ModelRefused(AttemptFailed):
    """The onboarding model's call was refused for a reason no later attempt escapes: a client error of the gateway,
    or a CLI too old for the model."""


class ConfigRefused(AttemptFailed):
    """An attempt whose config was refused when no model wrote it: another attempt would write the same, so it is the
    last."""


class GatewayDown(AttemptFailed):
    """The onboarding model's call could not reach the gateway or found it overloaded; the agent is not at fault, and
    a later attempt may pass."""


async def make_ready(source: str, spec: AgentSpec, agent_dir: Path, setup: OnboardingSetup) -> ReadyAgent:
    """Fetch ``source`` into ``agent_dir``, onboard it for ``spec`` and start it; raise
    :class:`~hyperloom_multibranch.specialist.runtime.PreflightFailed` with one line per failed attempt. An attempt
    that met the gateway down is made again, not counted, until :func:`gateway_wait` of ``spec`` has passed."""

    failures: list[str] = []
    attempt, pause = 1, RETRY_SECONDS
    until = setup.clock() + gateway_wait(spec)
    while True:
        try:
            return await _attempt(source, spec, agent_dir, setup, tuple(failures))
        except (ModelRefused, ConfigRefused) as error:
            raise PreflightFailed([*failures, f"attempt {attempt}: {error}"]) from error
        except GatewayDown as error:
            if setup.clock() + pause <= until:
                await setup.sleep(pause)
                pause = min(pause * 2, GATEWAY_PAUSE_SECONDS)
                continue
            failures.append(f"attempt {attempt}: {error}")
        except (AttemptFailed, OSError) as error:
            failures.append(f"attempt {attempt}: {error}")
        if attempt == setup.attempts:
            raise PreflightFailed(failures)
        attempt += 1
        await setup.sleep(RETRY_SECONDS)


def gateway_wait(spec: AgentSpec) -> float:
    """How long the preflight for ``spec`` waits for a gateway that is down: :data:`GATEWAY_WAIT_SECONDS`, and no more
    than half of the agent's time, so that it still runs for the other half."""

    return min(GATEWAY_WAIT_SECONDS, spec.seconds / 2) if spec.seconds else GATEWAY_WAIT_SECONDS


async def _attempt(
    source: str, spec: AgentSpec, agent_dir: Path, setup: OnboardingSetup, failures: tuple[str, ...]
) -> ReadyAgent:
    checkout = agent_dir / SOURCE_DIR
    commit = await fetch_source(source, checkout)
    adapted = is_arbor(checkout)
    if adapted:
        try:
            manifest = await asyncio.to_thread(arbor_manifest, spec, ArborCheckout(source, commit, checkout), setup.run)
        except ArborRefused as error:
            raise ConfigRefused(f"Arbor would refuse the config written for it: {error}") from error
    else:
        answer = await setup.onboard(checkout, onboarding_prompt(spec, source, checkout, commit, failures))
        try:
            manifest = AgentManifest.read(_json_object(answer))
        except (ManifestError, ValueError) as error:
            raise AttemptFailed(f"the manifest: {error}") from error
    if manifest.snapshots is not None:
        profile = root_profile(spec.root_shares, checkout / manifest.snapshots.session, spec.root_shares_source)
        manifest = replace(manifest, config=component_config(manifest.config, spec, profile))
    (checkout / manifest.config_file).parent.mkdir(parents=True, exist_ok=True)
    (checkout / manifest.config_file).write_text(json.dumps(manifest.config, indent=2), encoding=ENCODING)
    env = agent_env(agent_dir)
    await install(manifest, setup, checkout, env)
    process = await setup.sandbox.start(manifest.launch, checkout, env)
    started = False
    try:
        await _started(process, manifest, checkout, setup, final=adapted)
        started = True
    finally:
        if not started:
            await process.kill()
    return ReadyAgent(manifest, checkout, process)


async def install(manifest: AgentManifest, setup: OnboardingSetup, checkout: Path, env: Mapping[str, str]) -> None:
    """Run ``manifest``'s install commands in ``setup``'s sandbox, in ``checkout``; raise :class:`AttemptFailed` with
    the output of the first that fails."""

    for command in manifest.install:
        done = await setup.sandbox.run(command, checkout, env, setup.install_seconds)
        if done.returncode != 0:
            raise AttemptFailed(f"install `{command}` exited {done.returncode}: {tail(done.output)}")


def component_config(config: Mapping[str, object], spec: AgentSpec, profile: Path | None = None) -> dict[str, object]:
    """``config`` in Arbor's component mode, which runs on the host only, ending by ``spec``'s deadline, in whole unix
    seconds, making every patch against ``spec``'s root and reading the root's op shares from ``profile``, if any."""

    component = config.get("component")
    given = dict(component) if isinstance(component, Mapping) else {}
    profiled = {} if profile is None else {"profile": str(profile)}
    return {
        **config,
        "mode": "component",
        "docker_container": "@host",
        "component": {**given, "deadline": int(spec.deadline), "root": spec.root, **profiled},
    }


def root_profile(shares: OpShares | None, session: Path, source: str = ROOT_PROFILE_SOURCE) -> Path | None:
    """Write ``shares``, which came from ``source``, to :data:`PROFILE_FILE` in ``session``, the form Arbor's
    ``component.profile`` reads; the file, ``None`` with no shares."""

    if shares is None:
        return None
    document = {
        "source": source,
        "commit": shares.commit_id,
        "job": shares.job_id,
        "unit": "percent",
        "shares": dict(shares.shares),
    }
    profile = session / PROFILE_FILE
    profile.parent.mkdir(parents=True, exist_ok=True)
    profile.write_text(json.dumps(document, indent=2), encoding=ENCODING)
    return profile


async def _started(
    process: AgentProcess, manifest: AgentManifest, checkout: Path, setup: OnboardingSetup, *, final: bool = False
) -> None:
    """Return once the agent wrote its keep file or stayed up :attr:`OnboardingSetup.start_seconds`, or, run as a
    component, ended of itself with its exit file written, as one started with too little time left does: its run
    finishes with the exit file's reason. Raise :class:`AttemptFailed` when it exited otherwise first, saying why from
    its refusal file when it ran as a component and refused its config, and :class:`ConfigRefused` then when the
    config is ``final``."""

    until = setup.clock() + setup.start_seconds
    keep = None if manifest.keep is None else checkout / manifest.keep.path
    while setup.clock() < until and not (keep is not None and keep.exists()):
        if (
            process.returncode == 0
            and manifest.snapshots is not None
            and (checkout / manifest.snapshots_exit).is_file()
        ):
            return
        if process.returncode is not None:
            said = tail(await process.output())
            if process.returncode == EXIT_REFUSED and manifest.snapshots is not None:
                why = refused_reasons(checkout / manifest.snapshots_refused) or said
                raise (ConfigRefused if final else AttemptFailed)(f"`{manifest.launch}` refused its config: {why}")
            raise AttemptFailed(f"`{manifest.launch}` exited {process.returncode}: {said}")
        await setup.sleep(setup.poll_seconds)


async def fetch_source(source: str, checkout: Path) -> str:
    """Clone the agent ``source`` (:func:`split_source`'s form) into ``checkout`` unless it is there, and return the
    commit it is at; raise :class:`AttemptFailed` with git's words when it cannot."""

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
directory is the agent's checkout, {checkout}, a clone of {source} at {commit}. Read its README, its docs and its entry
points before you answer. Every file the agent writes (its config, the packs or scripts the install adds, its session
and run directories) must be in that checkout or in "workspace", never in {source}: give absolute paths under them.

It will optimize this workload, and every value its config needs must come from here, not be invented. "server_args" and
"envs" are what the recipe the branch starts on adds to a server of "model" at "tp"; the agent's own model, host, port
and TP flags come on top of them:
{spec}

The sandbox:
- has the session's image and framework and no docker daemon. It sees only the GPUs in "cards", numbered from 0
  (ROCR_VISIBLE_DEVICES holds them): a GPU id the agent's config gives is one of those, 0 the first; never set
  ROCR_VISIBLE_DEVICES, HIP_VISIBLE_DEVICES or CUDA_VISIBLE_DEVICES in a command of your own: a command whose
  ROCR_VISIBLE_DEVICES reaches other cards is killed. A `docker` on PATH runs `docker exec [-w DIR] [-e K=V]
  CONTAINER CMD...` as CMD in DIR and `docker cp C:SRC DST` as a copy, so an agent written to drive a container works
  when its config names any container;
- shares the host with other servers: `pkill`, `pgrep` and `killall` reach only the agent's own processes, but turn
  off any other sweep that kills what holds GPU memory, and give the agent's servers a port no one else uses;
- the agent edits the framework source in "workspace", in place, and launches its servers from it;
- "seconds" is how long the agent may run, 0 when MB stops it: then give it a time budget it cannot reach.

If the agent can run inside a host (Arbor's component mode, which its docs/hosts.md describes: an absolute deadline, the
commit its patches are made against, and a snapshot directory written for each kept change), run it in that mode: MB
sets its "mode", "docker_container", "component.deadline" ("deadline"), "component.root" ("root") and, when the
session's root was profiled, "component.profile" itself; set everything else docs/hosts.md asks of a host, give its
session a directory in the checkout, and answer "snapshots".

MB judges every change the agent keeps by measuring it with "client", the gate's own benchmark client, against a
server on "port"; another client sends other prompts and ranks changes otherwise. Unless "client" is empty, make the
agent measure every A/B, its baseline included, with exactly that command, changing only the port of --base-url to the
server it measures and --result-dir to a directory of its own: no other client, dataset, prompt count or warmup, even
when the agent ships a benchmark of its own. "baseline" is what the gate measured of "metric" on the session's root
with it, which the agent's own measurement of that recipe should come close to.

"knowledge" is a markdown file MB rewrites whenever the session's integration branch takes a change from another
branch: its server arguments, environment, patches and the changes merged into it. Make the agent read it at the start
of every iteration, through its goal, notes or instructions, and never redo a change it lists.

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
- "config_source": where the agent keeps the server arguments and environment of what it kept, {{"path": a JSON file,
  "root": "workspace" when the path is relative to "workspace", "checkout" when relative to the checkout, "server_args":
  the dotted field of the arguments string, "envs": the dotted field of the environment, an object or one string of
  NAME=VALUE words, "adds": true when the two hold only what the agent accepted on top of the "server_args" and "envs"
  it was started with, false when they are its whole config}}. A tuned kernel table the agent deploys through an
  environment variable (an AITER_CONFIG_* path) is a change it keeps: when its result file reports the accepted
  environment (for example accepted_config.env beside accepted_config.flags), name that file here. null only when the
  agent keeps neither arguments nor environment of its own;
- "summary_field": dotted path in the keep file to the agent's own words for its latest keep (an integer step indexes
  a list, -1 its last item), "" when none;
- "needs": what it requires, from "anthropic", "openai", "network", "docker";
- "ledger": where the agent writes down what it found, {{"path": JSON-lines file relative to the checkout, "id": the
  field of an entry's id, "kind": the field of its kind, "text": the field of the finding, "kinds": {{each of the
  agent's kinds that states a measured fact or a dead end: "insight" or "pitfall"}}}}, or null when it writes none;
- "snapshots": for an agent run in that mode, {{"session": its session directory relative to the checkout, where it
  writes its snapshot outbox and its exit file}}, with "keep" and "config_source" null; null otherwise.

Earlier attempts that failed, to correct rather than repeat:
{failures}
"""


__all__ = [
    "ATTEMPTS",
    "GATEWAY_PAUSE_SECONDS",
    "GATEWAY_WAIT_SECONDS",
    "RETRY_SECONDS",
    "ENCODING",
    "SOURCE_DIR",
    "DOCKER_SHIM",
    "ONBOARDING_PROMPT",
    "PROFILE_FILE",
    "AgentProcess",
    "AgentSandbox",
    "AgentSpec",
    "AttemptFailed",
    "ConfigRefused",
    "CommandResult",
    "GatewayDown",
    "ModelRefused",
    "OnboardingSetup",
    "ReadyAgent",
    "agent_env",
    "component_config",
    "gateway_wait",
    "install",
    "make_ready",
    "onboarding_prompt",
    "root_profile",
    "split_source",
    "tail",
]
