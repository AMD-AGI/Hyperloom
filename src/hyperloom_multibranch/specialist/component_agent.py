# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An agent run as a component of its host, and the adapter that makes one ready without the onboarding model.

A third-party agent is onboarded by a model that reads its repository and writes its manifest
(:mod:`agent_onboarding`). An agent that runs as a **component** of a host needs none of that: it takes one config,
writes a snapshot of each kept change into its session's outbox and an exit file before every exit 0, and exits
:data:`EXIT_REFUSED` for a config it refuses, which a host never starts again. For such an agent an **adapter**
(:class:`ComponentAgent`) says what only the agent's own code knows: how its checkout is recognised, the manifest and
config that follow from a branch's spec, which config keys a checkout's version takes, how its map is made and split,
and where its dossier is.

This module is the host's side of that:

* the adapter's record, and the registry of the installed ones (:func:`installed`, :func:`adapter_for`): a
  package declares its adapter under the entry-point group :data:`ENTRY_POINT_GROUP`, and this one carries none;
* the runs MB starts in an agent's checkout (:data:`BRANCH_RUN`, :data:`DOSSIER_PASS`);
* the launch a recipe adds to, as a launch file or a base launch record (:func:`read_launch`);
* how a run ended (:func:`exit_reason`, :func:`refused_reasons`, :func:`mapped`);
* how a spec hands an agent's review gate over (:data:`REVIEW_HOST`, :data:`REVIEW_OFF`).

With no adapter installed, every agent source takes the onboarding model's path.
"""

from __future__ import annotations

import functools
import json
import os
import re
import shlex
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from importlib import import_module, metadata
from pathlib import Path
from typing import Any

from hyperloom_multibranch.blackboard.local.agent_manifest import REDACTED
from hyperloom_multibranch.specialist.agent_manifest import EXIT_FILE, AgentManifest
from hyperloom_multibranch.specialist.agent_spec import AgentSpec
from inference_suite.argv import reserialize_json_blobs
from inference_suite.contracts import BASE_LAUNCH_SCHEMA, BaseLaunch, BaseLaunchError, CredentialNotHeld, held_env

ENCODING = "utf-8"
#: Where MB writes an agent's configs and keeps its sessions, in the agent's checkout: a config must be under the
#: checkout, where the agent finds its own files from it.
MB_DIR = "hyperloom-mb"
#: The keys of a launch file (:func:`read_launch`): the first two it must give, the others it may.
LAUNCH_KEYS = ("server_flags", "server_env", "model_name")
#: A placeholder of the repository's launch template still to be filled: ``<NAME>``, or a description such as ``<the
#: name the server answers to>``. A literal tag in a real flag, ``<think>``, is neither.
_PLACEHOLDER = re.compile(r"<(?:[A-Z_]+|[^<>]*\s[^<>]*)>")
#: A component agent's exit code for a config it refuses, and only for that; its session's refusal file says why.
EXIT_REFUSED = 2
#: The exit reason of a dossier pass that set its map.
MAPPED = "mapped"
#: How a spec hands an agent's review gate over: the host reviewed the campaign before it starts the run, or nobody
#: reviews it.
REVIEW_HOST = "host"
REVIEW_OFF = "off"
#: The entry-point group another package declares its adapter under: ``name = "package.module:ADAPTER"``.
ENTRY_POINT_GROUP = "hyperloom_multibranch.component_agents"


class AgentRefused(ValueError):
    """A config the agent would refuse at its start, or a launch that cannot be served; ``reasons`` says why, one
    line each."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = tuple(reasons)


class SplitFailed(RuntimeError):
    """The agent could not generate a map, or split one into shares; the message says what it printed."""


@dataclass(frozen=True)
class ResolvedLaunch:
    """A server's launch as the recipe's framework resolves it: its ``server_flags`` and ``server_env``, the
    ``model_name`` the server answers to, ``""`` when not given. The recipe's
    own launch is added to it, as a round adds a root recipe's ``extra_server_args`` and ``extra_envs``."""

    server_flags: str = ""
    server_env: Mapping[str, str] = field(default_factory=dict)
    model_name: str = ""
    #: A base launch record's file, its digest and its entrypoint; empty for a launch file.
    record: str = ""
    digest: str = ""
    entrypoint: tuple[str, ...] = ()


def read_launch(path: Path) -> ResolvedLaunch:
    """The launch in the file ``path``, a JSON object of :data:`LAUNCH_KEYS` (the repository's
    ``examples/multibranch-agent-launch`` has a template), or a base launch record (:func:`_read_record`); raise
    :class:`AgentRefused` with every reason it cannot be used: unreadable, a key it does not have, no ``server_flags``
    or ones with a quote left open, a ``server_env`` that is not an object of strings, or a template placeholder left
    unfilled. The gate gives every agent its client, so a launch names none."""

    try:
        document = json.loads(path.read_text(encoding=ENCODING))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise AgentRefused([f"the launch file {path} cannot be read: {error}"]) from error
    if not isinstance(document, dict):
        raise AgentRefused([f"the launch file {path} is not a JSON object"])
    if document.get("schema") == BASE_LAUNCH_SCHEMA:
        return _read_record(path, document)
    unknown = sorted(set(document) - set(LAUNCH_KEYS))
    reasons = [f"it has keys a launch does not: {', '.join(unknown)}"] if unknown else []
    flags, env = document.get("server_flags"), document.get("server_env")
    if not isinstance(flags, str) or not flags.strip():
        reasons.append("it gives no server_flags, the launch's server arguments")
    elif not _splits_as_shell(flags):
        reasons.append("its server_flags do not split into shell words: a quote is left open")
    if not isinstance(env, dict) or not all(isinstance(value, str) for value in env.values()):
        reasons.append("its server_env is not an object of strings")
    texts = {key: document.get(key, "") for key in LAUNCH_KEYS[2:]}
    reasons += [f"its {key} is not a string" for key, value in texts.items() if not isinstance(value, str)]
    unfilled = [key for key in LAUNCH_KEYS if _PLACEHOLDER.search(json.dumps(document.get(key, "")))]
    if unfilled:
        reasons.append(f"its {', '.join(unfilled)} still hold the template's placeholders: fill them in")
    if reasons:
        raise AgentRefused([f"the launch file {path}: {reason}" for reason in reasons])
    return ResolvedLaunch(flags, env, texts["model_name"])


def served_config(config: Mapping[str, Any], spec: AgentSpec) -> dict[str, Any]:
    """``config`` as MB stores it, every credential masked, with each masked ``server_env`` value that ``spec``'s
    launch file serves read from it again (:func:`read_launch`): a frozen base's from this host. Raise
    :class:`AgentRefused` when the host no longer holds that value."""

    env = config.get("server_env")
    masked = [name for name, value in env.items() if value == REDACTED] if isinstance(env, Mapping) else []
    if not masked or not spec.launch:
        return dict(config)
    served = read_launch(Path(spec.launch)).server_env
    return {**config, "server_env": {**env, **{name: served[name] for name in masked if name in served}}}


def _read_record(path: Path, document: Mapping[str, Any]) -> ResolvedLaunch:
    """The launch the base launch record ``document`` read from ``path`` serves: its server argv and environment as the
    gate seeds a launch with them (:func:`~hyperloom.orchestrator.state.base_launch.server_base`) and each redacted
    value from this host (:func:`~inference_suite.contracts.held_env`, as the suite's own round fills it), the record
    itself, its digest and its entrypoint, which an agent may check its own launch against at its start. The record
    names no served model and no client command: those are the gate's client's. Raise :class:`AgentRefused` when it is
    no valid record, a scriptable base, which launches no server, or one whose redacted value this host does not
    hold."""

    from hyperloom.orchestrator.state.base_launch import server_argv, server_base

    try:
        base = BaseLaunch.from_document(document)
    except BaseLaunchError as error:
        raise AgentRefused([f"the base launch record {path} cannot be used: {error}"]) from error
    if base.scriptable:
        raise AgentRefused([f"the base launch record {path} runs a script and launches no server for the agent"])
    try:
        held = held_env(base.env, os.environ)
    except CredentialNotHeld as error:
        raise AgentRefused([f"the base launch record {path} cannot be served: {error}"]) from error
    return ResolvedLaunch(
        server_flags=" ".join(_flag_word(word) for word in server_argv(base)),
        server_env={**held, **server_base(base).env},
        record=str(path),
        digest=base.digest(),
        entrypoint=base.entrypoint,
    )


def _flag_word(word: str) -> str:
    """``word`` as a launch's server_flags carry it, which an agent launches through a shell: quoted for the shell, a
    JSON value compacted first, and of a ``--name=value`` word only its value, so that it still reads as a flag,
    unless the value holds whitespace, which a non-POSIX split would break inside a quoted value."""

    flag, value = _flag_and_value(word)
    if value[:1] in "{[":
        try:
            value = json.dumps(json.loads(value), separators=(",", ":"))
        except ValueError:
            pass  # not JSON after all: the value is quoted as it stands
    if any(char.isspace() for char in value):
        return shlex.quote(flag + value)
    return flag + shlex.quote(value)


def shell_line(args: str) -> str:
    """Server arguments as MB composes them, as the shell line a launch's server_flags are: each of
    :func:`_shell_words` quoted by :func:`_flag_word`."""

    return " ".join(_flag_word(word) for word in _shell_words(args))


def _shell_words(args: str) -> list[str]:
    """The words a shell gives ``args``, in which a JSON value may also stand unquoted, as MB composes one; raise
    ``ValueError`` when a quote is left open."""

    text, decoder = reserialize_json_blobs(args), json.JSONDecoder()
    parts: list[str] = []
    blobs: list[str] = []
    i = 0
    while i < len(text):
        if text[i] in "{[":
            try:
                end = decoder.raw_decode(text, i)[1]
            except ValueError:
                pass  # no JSON value starts here: the character is the shell's
            else:
                parts.append(f"\0{len(blobs)}\0")
                blobs.append(text[i:end])
                i = end
                continue
        parts.append(text[i])
        i += 1
    return [re.sub(r"\0(\d+)\0", lambda found: blobs[int(found[1])], word) for word in shlex.split("".join(parts))]


def _splits_as_shell(args: str) -> bool:
    try:
        _shell_words(args)
    except ValueError:
        return False
    return True


def _flag_and_value(word: str) -> tuple[str, str]:
    """A ``--name=value`` word as ``--name=`` and its value; any other word as ``""`` and itself."""

    if word.startswith("-") and "=" in word:
        name, _, value = word.partition("=")
        return f"{name}=", value
    return "", word


@dataclass(frozen=True)
class AgentCheckout:
    """An agent at ``commit`` of ``source``, checked out at ``path``."""

    source: str
    commit: str
    path: Path


@dataclass(frozen=True)
class ComponentRun:
    """One run of an agent in its checkout: where its config and its session are, and where it stops of itself,
    ``None`` when only its deadline or MB stops it."""

    config_file: str
    session_dir: str
    stop_at: str | None = None


#: A branch's run, and the dossier pass, which writes the dossier and its map on one lease before the branches open and
#: exits 0 once the map is set (:func:`mapped`).
BRANCH_RUN = ComponentRun(f"{MB_DIR}/config.json", f"{MB_DIR}/session")
DOSSIER_PASS = ComponentRun(f"{MB_DIR}/dossier.json", f"{MB_DIR}/dossier", stop_at="map")


@dataclass(frozen=True)
class ComponentAgent:
    """What only an agent's own code knows, for MB to run it as a component of this host (the module's docstring).

    ``name`` is the agent's, as a message says it, and ``recognises`` whether a checkout is the agent's. ``manifest``
    is the manifest that follows from a branch's spec, its config in it, for one run in the checkout; it raises
    :class:`AgentRefused` for a config the agent would refuse at its start. ``fit_keys`` is that config with every key
    the checkout's version cannot take left out, and a note for each. ``eval_envs`` names the evaluation settings the
    agent reads from its config's server environment, which no keep carries into a recipe.

    The agent plans before its branches open. ``map_file`` is its map's file in a dossier pass's session.
    ``generate_directions`` writes a map from a root profile, with no GPU time, and ``split_directions`` splits a
    map's open directions into disjoint plans, in order; each raises :class:`SplitFailed` with what the agent printed.
    ``dossier_index`` is the file that makes a directory one of its dossiers, ``pass_dossier`` the dossier a pass wrote
    in its checkout, ``None`` without one, and ``dossier_notes`` what a branch should note of the dossier it was
    given."""

    name: str
    recognises: Callable[[Path], bool]
    manifest: Callable[[AgentSpec, AgentCheckout, ComponentRun], AgentManifest]
    fit_keys: Callable[[Mapping[str, Any], Path], tuple[dict[str, Any], tuple[str, ...]]]
    eval_envs: tuple[str, ...]
    map_file: str
    generate_directions: Callable[[Path, Path, Path], Awaitable[Path]]
    split_directions: Callable[[Path, Path, int, Path], Awaitable[tuple[Path, ...]]]
    dossier_index: str
    pass_dossier: Callable[[Path, Path], Path | None]
    dossier_notes: Callable[[Path, str | None], tuple[str, ...]]


@functools.cache
def installed() -> tuple[ComponentAgent, ...]:
    """The adapters this install has: those other packages declare under :data:`ENTRY_POINT_GROUP`, by name. This
    package carries none. ``TypeError`` for a declaration that names no :class:`ComponentAgent`."""

    declared = sorted((point.name, point.value) for point in metadata.entry_points(group=ENTRY_POINT_GROUP))
    agents = []
    for name, target in declared:
        module, _, attribute = target.partition(":")
        agent = getattr(import_module(module), attribute, None)
        if not isinstance(agent, ComponentAgent):
            raise TypeError(f"the component agent {name} ({target}) names no ComponentAgent")
        agents.append(agent)
    return tuple(agents)


def adapter_for(checkout: Path) -> ComponentAgent | None:
    """The installed adapter whose agent ``checkout`` is, ``None`` for any other agent's."""

    return next((agent for agent in installed() if agent.recognises(checkout)), None)


def agent_of(checkout: Path) -> ComponentAgent:
    """The adapter of ``checkout``, a component agent's; ``LookupError`` when no installed adapter recognises it."""

    agent = adapter_for(checkout)
    if agent is None:
        raise LookupError(f"no installed adapter recognises the agent at {checkout}")
    return agent


def eval_envs() -> frozenset[str]:
    """The evaluation settings of every installed adapter's agent (``eval_envs``)."""

    return frozenset(name for agent in installed() for name in agent.eval_envs)


def dossier_lacks(path: Path) -> str:
    """What the directory ``path`` lacks to be a dossier an installed adapter's agent takes: ``""`` when it holds one
    adapter's ``dossier_index``, else the index files it holds none of."""

    indexes = [agent.dossier_index for agent in installed()]
    if any((path / index).is_file() for index in indexes):
        return ""
    return " or ".join(indexes) or "index an installed component agent knows"


def refused_reasons(path: Path) -> str:
    """Why the agent refused its config, as the refusal file at ``path`` lists it; ``""`` when there is none to
    read."""

    try:
        document = json.loads(path.read_text(encoding=ENCODING))
    except (OSError, ValueError):
        return ""
    reasons = document.get("reasons") if isinstance(document, dict) else None
    if not isinstance(reasons, list):
        return ""
    return "; ".join(str(reason) for reason in reasons if str(reason).strip())


def exit_reason(path: Path) -> str:
    """Why the agent says its run ended, as the exit file at ``path`` it writes before an exit 0 has it; else what is
    wrong with that file."""

    try:
        document = json.loads(path.read_text(encoding=ENCODING))
    except FileNotFoundError:
        return "it wrote no exit reason"
    except (OSError, ValueError) as error:
        return f"its exit reason cannot be read: {error}"
    reason = document.get("reason") if isinstance(document, dict) else None
    return reason if isinstance(reason, str) and reason else f"its exit file names no reason: {document!r}"


def mapped(session: Path) -> bool:
    """Whether the dossier pass whose session is ``session`` ended with its map set, as its exit file says; the map is
    then the file the agent's adapter names in that session (``map_file``)."""

    return exit_reason(session / EXIT_FILE) == MAPPED


__all__ = [
    "BRANCH_RUN",
    "DOSSIER_PASS",
    "ENCODING",
    "ENTRY_POINT_GROUP",
    "EXIT_REFUSED",
    "LAUNCH_KEYS",
    "MAPPED",
    "MB_DIR",
    "REVIEW_HOST",
    "REVIEW_OFF",
    "AgentCheckout",
    "AgentRefused",
    "ComponentAgent",
    "ComponentRun",
    "ResolvedLaunch",
    "SplitFailed",
    "adapter_for",
    "agent_of",
    "dossier_lacks",
    "eval_envs",
    "exit_reason",
    "installed",
    "mapped",
    "read_launch",
    "refused_reasons",
    "served_config",
    "shell_line",
]
