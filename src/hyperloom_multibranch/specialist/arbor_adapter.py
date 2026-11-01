# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Arbor as an autonomous branch's agent, made ready without the onboarding model: the Arbor lifecycle adapter.

Arbor runs inside a host in its component mode, whose contract is its ``docs/hosts.md``: one config, a snapshot of each
kept change in its session's outbox, an exit file written before every exit 0, and exit 2 for a config it refuses,
which a host never starts again. A checkout that has that mode, its entry point and that contract is Arbor's
(:func:`is_arbor`), and the preflight (:mod:`agent_onboarding`) takes its manifest from here instead of asking the
model: the config follows from the branch's spec alone (:func:`arbor_config`), on Arbor's serving template pack, with
the gate's client as its bench command (:func:`bench_command`) and its session in its checkout, the directory the
manifest's snapshots name. Before Arbor starts, the config is checked against what Arbor refuses at its start
(:func:`refusals`): Arbor's ``--dry-run`` runs its loop on synthetic numbers and checks none of it. Another attempt
would write the same config, so a refusal, this side's or Arbor's own, fails the preflight at once.

The spec's ``plan``, the file of the branch's share of the session's directions, becomes ``component.plan``. Those
shares come from one map, which Arbor's dossier pass writes on one lease before the branches open
(:data:`DOSSIER_PASS`, its end read by :func:`mapped`) or Arbor generates from the root's profile with no GPU time
(:func:`generate_directions`), split by Arbor's own command line (:func:`split_directions`); the scheduling calls these.
The dossier the pass wrote (:func:`pass_dossier`) is copied into each branch's checkout (:func:`carry_dossier`), where
Arbor looks a dossier up. The root's op shares are not the adapter's: the preflight sets
``component.profile`` for Arbor as for any agent run as a component
(:func:`~hyperloom_multibranch.specialist.agent_onboarding.component_config`).

The spec's ``launch``, a launch file (:func:`read_launch`), stands in for what MB does not carry yet: the launch a
Magpie AgentX recipe resolves from InferenceX, of which the root recipe holds only what is added to it. Without one,
an AgentX branch is refused before Arbor starts, which would otherwise serve defaults MB does not measure.

The spec's ``review`` hands Arbor's human review gate over (ArborDevelopment #30): Arbor waits at the start of a new
session for a person's approval unless the host says it reviewed (``component.review: "host"``, :data:`REVIEW_HOST`)
or the config turns the gate off (``"review_gate": false``, :data:`REVIEW_OFF`). Arbor refuses a component key it does
not know, so either is written only for a checkout whose component mode has the gate (:func:`has_review_gate`); an
older Arbor gets neither and runs as before.
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from hyperloom.inference_optimizer.framework_registry import EXPERT_PARALLEL_FLAGS
from hyperloom.inference_optimizer.grid_server_args import merge_server_args, tokenize_server_args_preserving_json
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    PackageLocator,
    UnsupportedComponentError,
    locate_installed_package,
)
from hyperloom_multibranch.specialist.agent_manifest import EXIT_FILE, AgentManifest
from hyperloom_multibranch.specialist.worktree import SKIPPED

if TYPE_CHECKING:
    from hyperloom_multibranch.specialist.agent_onboarding import AgentSpec

#: What makes a checkout Arbor's, run as a component: the mode, the entry point and the host's contract.
MARKERS = ("arbor/component.py", "arbor/__main__.py", "docs/hosts.md")
HOST = "hyperloom-mb"
#: Arbor's serving template, the pack a host runs it on unless the spec names another, and the validator scripts every
#: pack the adapter runs has.
PACK = "serving"
BENCH_SCRIPT = "validator/bench.sh"
CORRECTNESS_SCRIPT = "validator/correctness.sh"
#: Where MB writes Arbor's configs and keeps its sessions, in Arbor's checkout: a config must be under the checkout,
#: where Arbor finds its packs from it.
MB_DIR = "hyperloom-mb"
#: Where a branch's share of the directions is copied in its checkout, beside its config: the share MB splits off is
#: under the session, which the branch's sandbox need not see.
PLAN_FILE = f"{MB_DIR}/plan.json"
OUTBOX = "keeps"
#: The keys of a launch file (:func:`read_launch`): the first two it must give, the others it may.
LAUNCH_KEYS = ("server_flags", "server_env", "model_name", "client")
#: The workload kind whose recipe Magpie resolves, the root's launch holding only what is added to it.
AGENTX = "agentx"
#: Arbor gives vLLM its TP itself and SGLang none, which reads it from these flags only.
SGLANG = "sglang"
SGLANG_TP_FLAGS = ("--tp-size", "--tp", "--tensor-parallel-size")
#: The interpreter Arbor runs under, the sandbox's.
PYTHON = "python3"
#: What Arbor needs beyond the image (its docs/getting-started.md, and the third-party modules arbor/ imports outside
#: its tests): Python 3.11, PyYAML and the Claude agent SDK, which brings its CLI. Each is installed only when it does
#: not import, and the last command checks that both import.
INSTALL = (
    f"{PYTHON} -c 'import sys; sys.exit(sys.version_info < (3, 11) and \"Arbor needs Python 3.11 or newer\")'",
    f"{PYTHON} -c 'import yaml' 2>/dev/null || {PYTHON} -m pip install --quiet pyyaml",
    f"{PYTHON} -c 'import claude_agent_sdk' 2>/dev/null || {PYTHON} -m pip install --quiet claude-agent-sdk",
    f"{PYTHON} -c 'import yaml, claude_agent_sdk'",
)
#: Arbor's exit code for a config it refuses, and only for that; its session's refusal file says why.
EXIT_REFUSED = 2
#: The exit reason of a dossier pass that set its map, and the map's file in its session.
MAPPED = "mapped"
MAP_FILE = "directions.json"
#: Where Arbor's dossier skill writes a dossier in its checkout and looks one up: a directory per dossier, with its
#: index. Arbor ships dossiers of its own there, which git tracks.
DOSSIER_DIR = "docs/dossier"
DOSSIER_INDEX = "README.md"
#: The keys of Arbor's component block; another is refused.
BLOCK_KEYS = (
    "host",
    "deadline",
    "seconds",
    "notes",
    "bench_command",
    "outbox",
    "phase_minutes",
    "root",
    "plan",
    "stop_at",
    "profile",
    "review",
)
#: How a spec hands Arbor's review gate over: the host reviewed the campaign before it starts the run (the only value
#: ``component.review`` takes), or nobody reviews it (``"review_gate": false`` at the config's top level).
REVIEW_KEY = "review"
REVIEW_HOST = "host"
REVIEW_OFF = "off"
REVIEW_GATE_KEY = "review_gate"
#: Where an Arbor checkout declares its component block's keys, which says whether it has the review gate.
COMPONENT_MODULE = "arbor/component.py"
#: Arbor's ledger: FACT and DEAD_END records are what it measured and ruled out.
LEDGER = {"id": "id", "kind": "type", "text": "claim", "kinds": {"FACT": "insight", "DEAD_END": "pitfall"}}
ENCODING = "utf-8"
#: The client's flags Arbor fills: its server's port and an empty directory per replay.
BASE_URL, PORT, RESULT_DIR, MODEL = "--base-url", "--port", "--result-dir", "--model"
_SHA = re.compile(r"[0-9a-fA-F]{7,40}")
#: A word of nothing but these is a shell operator, carried as it stands.
_OPERATORS = frozenset("();<>|&")
#: The lines of a command's output a failure keeps, its end.
_TAIL_LINES = 20
#: The built files of an installed package, its compiled extensions among them: the source base's (:data:`SKIPPED`)
#: but its bytecode, which Python writes again.
BUILT = tuple(pattern for pattern in SKIPPED if pattern.startswith("*.") and pattern not in ("*.pyc", "*.pyo"))


class ArborRefused(ValueError):
    """A config Arbor would refuse at its start; ``reasons`` says why, one line each."""

    def __init__(self, reasons: list[str]) -> None:
        super().__init__("; ".join(reasons))
        self.reasons = tuple(reasons)


class SplitFailed(RuntimeError):
    """Arbor could not generate a map, or split one into shares; the message says what it printed."""


@dataclass(frozen=True)
class ResolvedLaunch:
    """A server's launch as the recipe's framework resolves it: its ``server_flags`` and ``server_env``, the
    ``model_name`` the server answers to and ``client``, the gate's client, each ``""`` when not given. The recipe's
    own launch is added to it, as Magpie adds a root recipe's ``extra_server_args`` and ``extra_envs``."""

    server_flags: str = ""
    server_env: Mapping[str, str] = field(default_factory=dict)
    model_name: str = ""
    client: str = ""


def read_launch(path: Path) -> ResolvedLaunch:
    """The launch in the file ``path``, a JSON object of :data:`LAUNCH_KEYS` (the repository's
    ``examples/multibranch-arbor-launch`` has a template); raise :class:`ArborRefused` with every reason it cannot be
    used: unreadable, a key it does not have, no ``server_flags``, a ``server_env`` that is not an object of strings,
    or a ``client`` Arbor could not fill."""

    try:
        document = json.loads(path.read_text(encoding=ENCODING))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ArborRefused([f"the launch file {path} cannot be read: {error}"]) from error
    if not isinstance(document, dict):
        raise ArborRefused([f"the launch file {path} is not a JSON object"])
    unknown = sorted(set(document) - set(LAUNCH_KEYS))
    reasons = [f"it has keys a launch does not: {', '.join(unknown)}"] if unknown else []
    flags, env = document.get("server_flags"), document.get("server_env")
    if not isinstance(flags, str) or not flags.strip():
        reasons.append("it gives no server_flags, the launch's server arguments")
    if not isinstance(env, dict) or not all(isinstance(value, str) for value in env.values()):
        reasons.append("its server_env is not an object of strings")
    texts = {key: document.get(key, "") for key in LAUNCH_KEYS[2:]}
    reasons += [f"its {key} is not a string" for key, value in texts.items() if not isinstance(value, str)]
    if reasons:
        raise ArborRefused([f"the launch file {path}: {reason}" for reason in reasons])
    if texts["client"]:
        bench_command(texts["client"])
    return ResolvedLaunch(flags, env, texts["model_name"], texts["client"])


@dataclass(frozen=True)
class ArborCheckout:
    """Arbor at ``commit`` of ``source``, checked out at ``path``, run with ``env``."""

    source: str
    commit: str
    path: Path
    env: Mapping[str, str] = field(default_factory=lambda: os.environ)


@dataclass(frozen=True)
class ArborRun:
    """One run of Arbor in its checkout: where its config and its session are, and where it stops of itself, ``None``
    when only its deadline or MB stops it."""

    config_file: str
    session_dir: str
    stop_at: str | None = None


#: A branch's run, and the dossier pass, which writes the dossier and its map on one lease before the branches open and
#: exits 0 once the map is set (:func:`mapped`).
BRANCH_RUN = ArborRun(f"{MB_DIR}/config.json", f"{MB_DIR}/session")
DOSSIER_PASS = ArborRun(f"{MB_DIR}/dossier.json", f"{MB_DIR}/dossier", stop_at="map")


def is_arbor(checkout: Path) -> bool:
    """Whether ``checkout`` is Arbor's, which runs as a component of a host: it has every one of :data:`MARKERS`."""

    return all((checkout / marker).is_file() for marker in MARKERS)


def has_review_gate(checkout: Path) -> bool:
    """Whether the Arbor in ``checkout`` has the human review gate: its component block takes :data:`REVIEW_KEY`, as
    its :data:`COMPONENT_MODULE` declares ``BLOCK_KEYS``, read without importing Arbor (Arbor ``1fa2069`` and later);
    ``False`` for a checkout without that module, or whose keys cannot be read."""

    try:
        tree = ast.parse((checkout / COMPONENT_MODULE).read_text(encoding=ENCODING))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BLOCK_KEYS" for t in node.targets):
            if isinstance(node.value, (ast.Tuple, ast.List)):
                return any(isinstance(e, ast.Constant) and e.value == REVIEW_KEY for e in node.value.elts)
    return False


def arbor_manifest(
    spec: AgentSpec, checkout: ArborCheckout, run: ArborRun, locator: PackageLocator = locate_installed_package
) -> AgentManifest:
    """The manifest of ``run`` of Arbor in ``checkout`` for the branch ``spec`` names, its session made and its
    workspace given the installed framework's built files (:func:`link_built`, ``locator`` finding the package);
    raise :class:`ArborRefused` with every reason Arbor would refuse its config for.

    The root's op shares are its caller's to add, as the preflight does for every agent run as a component; the
    dossier pass profiles only what they do not cover."""

    path = checkout.path.absolute()
    if Path(spec.workspace).is_dir():
        link_built(Path(spec.workspace), spec.framework, locator)
    if spec.dossier:
        carry_dossier(path, Path(spec.dossier))
    config = arbor_config(spec, path, run)
    reasons = [*_launch_refusals(spec), *refusals(config, checkout.env)]
    if reasons:
        raise ArborRefused(reasons)
    (path / run.session_dir).mkdir(parents=True, exist_ok=True)
    component = config["component"]
    if component.get("plan"):
        carried = path / PLAN_FILE
        carried.write_bytes(Path(component["plan"]).read_bytes())
        component["plan"] = str(carried)
    launch = f"{PYTHON} -m arbor run --config {shlex.quote(run.config_file)}"
    return AgentManifest.read(
        {
            "source": checkout.source,
            "commit": checkout.commit,
            "install": list(INSTALL),
            "config_file": run.config_file,
            "config": config,
            "launch": launch,
            "resume": launch,
            "keep": None,
            "config_source": None,
            "summary_field": "",
            "needs": ["anthropic"],
            "ledger": {"path": f"{run.session_dir}/ledger.jsonl", **LEDGER},
            "snapshots": {"session": run.session_dir},
        }
    )


def link_built(workspace: Path, framework: str, locator: PackageLocator = locate_installed_package) -> int:
    """Link the built files of ``framework``'s installed package (:data:`BUILT`) into ``workspace``, a branch's
    worktree, where the package's upstream layout puts them; how many it linked.

    The source base lays the package out without them, and MB's own tests lay a change over the installed package.
    Arbor serves from its workspace, and a vLLM without its compiled extensions cannot start. A file already there
    stays, and the base's ``info/exclude`` keeps the links out of every kept change. A framework with no layout, or
    not installed, links nothing: Arbor refuses at its start a workspace that cannot serve."""

    layout = COMPONENT_LAYOUTS.get(framework)
    if layout is None:
        return 0
    try:
        package = locator(layout)
    except UnsupportedComponentError:
        return 0
    target, linked = workspace / layout.upstream_prefix, 0
    for source in package.path.rglob("*"):
        if "__pycache__" in source.parts or source.is_dir() or not any(fnmatch(source.name, p) for p in BUILT):
            continue
        destination = target / source.relative_to(package.path)
        if os.path.lexists(destination):
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(source, destination)
        linked += 1
    return linked


def carry_dossier(checkout: Path, dossier: Path) -> None:
    """Copy ``dossier``, the one the session's dossier pass wrote, into ``checkout``'s :data:`DOSSIER_DIR`, where
    Arbor's dossier skill looks one up: a branch run on a share of the pass's map has no bootstrap round to write its
    own. One already there stays."""

    target = checkout / DOSSIER_DIR / dossier.name
    if dossier.is_dir() and not target.exists():
        shutil.copytree(dossier, target)


def pass_dossier(checkout: Path) -> Path | None:
    """The dossier a dossier pass in ``checkout`` wrote: the newest directory of :data:`DOSSIER_DIR` with an index that
    git does not track (the ones it tracks are Arbor's own); ``None`` when it wrote none."""

    root = checkout / DOSSIER_DIR
    if not root.is_dir():
        return None
    listed = subprocess.run(["git", "ls-files", "--", DOSSIER_DIR], cwd=checkout, capture_output=True, text=True)
    depth = len(PurePosixPath(DOSSIER_DIR).parts)
    tracked = (PurePosixPath(line).parts for line in listed.stdout.splitlines())
    shipped = {parts[depth] for parts in tracked if len(parts) > depth}
    written = [path for path in root.iterdir() if path.name not in shipped and (path / DOSSIER_INDEX).is_file()]
    return max(written, key=lambda path: (path / DOSSIER_INDEX).stat().st_mtime, default=None)


def arbor_config(spec: AgentSpec, checkout: Path, run: ArborRun) -> dict[str, Any]:
    """Arbor's config for ``run`` of ``spec``: in component mode on the host, the recipe's launch on the pack the spec
    names in ``checkout`` (the serving template by default), ending by the spec's deadline, its patches made against
    its root, the integration notes read every iteration, every measurement by the gate's client when the spec has
    one, and the host's plan when it gives one, which a run that stops at its map does not take. A spec with a launch
    file is served the file's launch with the recipe's added to it, and measured by the file's client when the spec
    has none.

    The cards are numbered from 0, as the sandbox shows them; Arbor never masks them itself in this mode."""

    name = spec.pack or PACK
    pack = checkout / "packs" / name
    launch = read_launch(Path(spec.launch)) if spec.launch else ResolvedLaunch()
    client = spec.client or launch.client
    component: dict[str, Any] = {"host": HOST, "deadline": int(spec.deadline), "root": spec.root, "outbox": OUTBOX}
    if spec.knowledge:
        component["notes"] = spec.knowledge
    if client:
        component["bench_command"] = bench_command(client)
    if run.stop_at:
        component["stop_at"] = run.stop_at
    elif spec.plan:
        component["plan"] = spec.plan
    gate = spec.review if spec.review and has_review_gate(checkout) else ""
    if gate == REVIEW_HOST:
        component[REVIEW_KEY] = REVIEW_HOST
    return {
        **({REVIEW_GATE_KEY: False} if gate == REVIEW_OFF else {}),
        "pack": name,
        "identity": f"{HOST} {Path(spec.model).name}",
        "goal": f"{spec.metric} of {spec.model} on {spec.framework} at TP{spec.tp} on {spec.gpu_type}, as MB measures it",
        "mode": "component",
        "docker_container": "@host",
        "framework": spec.framework,
        "model_path": spec.model,
        "model_name": launch.model_name or _client_value(client, MODEL) or spec.model,
        "tp": spec.tp,
        "server_flags": server_flags(spec, launch.server_flags),
        "server_env": {**launch.server_env, **spec.envs},
        "bench_script": str(pack / BENCH_SCRIPT),
        "correctness_script": str(pack / CORRECTNESS_SCRIPT),
        "workspace_root": spec.workspace,
        "session_root": str(checkout / run.session_dir),
        "port": spec.port,
        "gpu_ids": list(range(len(spec.cards))),
        **_operating_point(spec.workload),
        "component": component,
    }


def server_flags(spec: AgentSpec, resolved: str = "") -> str:
    """The recipe's server arguments as Arbor launches them, after the ``resolved`` launch's when there is one: on
    SGLang, with the workload's TP size and its expert-parallel size, each unless the arguments set it; the
    expert-parallel size only when it divides the TP, as MB's own launch has it."""

    args = merge_server_args(resolved, spec.server_args)
    if spec.framework != SGLANG:
        return args
    named = _flag_names(args)
    added = [] if named & set(SGLANG_TP_FLAGS) else [f"{SGLANG_TP_FLAGS[0]} {spec.tp}"]
    flag, aliases, _ = EXPERT_PARALLEL_FLAGS[SGLANG]
    if spec.ep > 1 and spec.tp % spec.ep == 0 and not named & set(aliases):
        added.append(f"{flag} {spec.ep}")
    return merge_server_args(args, *added)


def _launch_refusals(spec: AgentSpec) -> list[str]:
    """Why ``spec``'s branch cannot serve what MB measures: an AgentX recipe's launch, which holds only what Magpie
    adds to the one it resolves from InferenceX, given without that launch."""

    if spec.workload.get("kind") != AGENTX or spec.launch:
        return []
    return [
        "an AgentX recipe's launch holds only what Magpie adds to the one it resolves from InferenceX, so Arbor would"
        " serve defaults MB does not measure: give the resolved launch with --agent-launch"
        " (examples/multibranch-arbor-launch)"
    ]


def _flag_names(args: str) -> set[str]:
    split = tokenize_server_args_preserving_json(args)
    words = args.split() if split is None else split[1]
    return {word.partition("=")[0] for word in words if word.startswith("-")}


def _operating_point(workload: Mapping[str, object]) -> dict[str, int]:
    """The workload's concurrency and request shape, for the pack's own bench when the gate gives no client."""

    named = {"concurrency": workload.get("conc"), "isl": workload.get("isl"), "osl": workload.get("osl")}
    return {key: value for key, value in named.items() if isinstance(value, int) and not isinstance(value, bool)}


def bench_command(client: str) -> str:
    """The gate's ``client`` as Arbor's ``component.bench_command``: the port it measures, in ``--base-url`` or
    ``--port``, as ``{port}``, and its ``--result-dir`` as ``{out}``, which Arbor quotes as it fills it; every other
    word as it was, the client's warm-up requests among them. Raise :class:`ArborRefused` when the client names no
    port or no result directory, or cannot be read word by word."""

    words = _words(client)
    templated: list[str] = []
    found: set[str] = set()
    pending = ""
    for word in words:
        if pending:
            templated.append(_filled(pending, word))
            found.add(pending)
            pending = ""
            continue
        flag, joined, value = word.partition("=")
        if flag in (BASE_URL, PORT, RESULT_DIR):
            if joined:
                templated.append(f"{flag}={_filled(flag, value)}")
                found.add(flag)
            else:
                templated.append(flag)
                pending = flag
            continue
        templated.append(word if set(word) <= _OPERATORS else shlex.quote(word))
    reasons = [f"the gate's client ends at {pending}, which takes a value"] if pending else []
    if not found & {BASE_URL, PORT}:
        reasons.append(f"the gate's client names no port to measure Arbor's server on ({BASE_URL} or {PORT})")
    if RESULT_DIR not in found:
        reasons.append(f"the gate's client names no {RESULT_DIR} for Arbor to read each replay from")
    if reasons:
        raise ArborRefused([f"{reason}: {client}" for reason in reasons])
    return " ".join(templated)


def _words(client: str) -> list[str]:
    """``client``'s words, its shell operators words of their own; refused when it expands a variable or a command,
    which its words, quoted again, would carry as text."""

    if "$" in client or "`" in client:
        raise ArborRefused([f"the gate's client expands variables or commands, which its words cannot carry: {client}"])
    lexer = shlex.shlex(client, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError as error:
        raise ArborRefused([f"the gate's client cannot be read word by word ({error}): {client}"]) from error


def _filled(flag: str, value: str) -> str:
    """The value of ``flag`` as Arbor fills it: a placeholder, or the base URL with its port one."""

    if flag == RESULT_DIR:
        return "{out}"
    if flag == PORT:
        return "{port}"
    try:
        url = urlsplit(value)
        port = url.port
    except ValueError as error:
        raise ArborRefused([f"the gate's client's {BASE_URL} {value} is no URL with a port: {error}"]) from error
    if port is None:
        raise ArborRefused([f"the gate's client's {BASE_URL} {value} names no port"])
    head = f"{url.scheme}://{url.netloc.rsplit(':', 1)[0]}:"
    rest = value[len(f"{url.scheme}://{url.netloc}") :]
    return shlex.quote(head) + "{port}" + (shlex.quote(rest) if rest else "")


def _client_value(client: str, flag: str) -> str:
    """The value ``client`` gives ``flag``, ``""`` when it gives none or cannot be read."""

    try:
        words = _words(client) if client else []
    except ArborRefused:
        return ""
    for index, word in enumerate(words):
        if word.startswith(f"{flag}="):
            return word.partition("=")[2]
        if word == flag and index + 1 < len(words):
            return words[index + 1]
    return ""


def refusals(config: Mapping[str, Any], env: Mapping[str, str] = os.environ) -> list[str]:
    """Why Arbor would refuse ``config`` at its start, run with ``env``, as far as this side can tell; ``[]`` when it
    would not.

    These are the refusals of Arbor's ``docs/hosts.md`` that a config written from the spec can meet: no session, no
    deadline, a container other than the host, ``ARBOR_ROLL=1``, a component block with a key Arbor does not have, a
    root that is no commit of the workspace, a workspace, script or pack that is not there, a plan that cannot be read,
    and a plan beside a dossier pass. What only Arbor can check, a plan's directions and the pack's own validation
    among them, it refuses at its start, with exit 2."""

    component = config.get("component")
    component = component if isinstance(component, Mapping) else {}
    return [
        *_block_refusals(component),
        *_host_refusals(config, env),
        *_path_refusals(config),
        *_root_refusals(component.get("root"), Path(str(config.get("workspace_root") or ""))),
        *_plan_refusals(component.get("plan")),
    ]


def _block_refusals(component: Mapping[str, Any]) -> list[str]:
    reasons = []
    unknown = sorted(set(component) - set(BLOCK_KEYS))
    if unknown:
        reasons.append(f"the component block has keys Arbor does not: {', '.join(unknown)}")
    deadline = component.get("deadline")
    if not isinstance(deadline, int) or isinstance(deadline, bool) or deadline <= 0:
        reasons.append(f"no deadline: component.deadline is {deadline!r}")
    if component.get("plan") is not None and component.get("stop_at") is not None:
        reasons.append("a dossier pass (component.stop_at) takes no plan")
    if component.get(REVIEW_KEY) not in (None, REVIEW_HOST):
        reasons.append(f"component.review is {component.get(REVIEW_KEY)!r}, not {REVIEW_HOST!r}")
    return reasons


def _host_refusals(config: Mapping[str, Any], env: Mapping[str, str]) -> list[str]:
    reasons = []
    if not config.get("session_root"):
        reasons.append("no session_root: the snapshots, the heartbeat and the exit file are written there")
    if config.get("docker_container") != "@host":
        reasons.append(f"docker_container is {config.get('docker_container')!r}, not @host")
    if env.get("ARBOR_ROLL") == "1":
        reasons.append("ARBOR_ROLL=1 is set: a component's clock is its host's")
    return reasons


def _path_refusals(config: Mapping[str, Any]) -> list[str]:
    reasons = []
    if not config.get("workspace_root") or not Path(str(config["workspace_root"])).is_dir():
        reasons.append(f"the workspace {config.get('workspace_root')!r} is not a directory")
    for key in ("bench_script", "correctness_script"):
        if not Path(str(config.get(key) or "")).is_file():
            reasons.append(f"{key} {config.get(key)!r} is not a file")
    pack = Path(str(config.get("bench_script") or "")).parent.parent
    if not (pack / "pack.yaml").is_file():
        reasons.append(f"the pack {config.get('pack')!r} has no pack.yaml at {pack}")
    return reasons


def _root_refusals(root: object, workspace: Path) -> list[str]:
    if not isinstance(root, str) or not _SHA.fullmatch(root):
        return [f"component.root {root!r} is not a commit sha"]
    if workspace.is_dir() and not _has_commit(workspace, root):
        return [f"component.root {root} is not a commit of the workspace {workspace}"]
    return []


def _has_commit(workspace: Path, sha: str) -> bool:
    found = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    return found.returncode == 0 and bool(found.stdout.strip())


def _plan_refusals(plan: object) -> list[str]:
    """What is wrong with the plan file ``plan``, if any, as far as reading it goes: a JSON list of directions or
    ``{"directions": [...]}``, not empty, at an absolute path."""

    if plan is None:
        return []
    if not isinstance(plan, str) or not Path(plan).is_absolute():
        return [f"component.plan {plan!r} is not an absolute path"]
    try:
        document = json.loads(Path(plan).read_text(encoding=ENCODING))
    except (OSError, ValueError) as error:
        return [f"component.plan {plan} cannot be read: {error}"]
    directions = document.get("directions") if isinstance(document, dict) else document
    if not isinstance(directions, list) or not directions:
        return [f"component.plan {plan} holds no directions"]
    return []


def refused_reasons(path: Path) -> str:
    """Why Arbor refused its config, as the refusal file at ``path`` lists it; ``""`` when there is none to read."""

    try:
        document = json.loads(path.read_text(encoding=ENCODING))
    except (OSError, ValueError):
        return ""
    reasons = document.get("reasons") if isinstance(document, dict) else None
    if not isinstance(reasons, list):
        return ""
    return "; ".join(str(reason) for reason in reasons if str(reason).strip())


def mapped(session: Path) -> bool:
    """Whether the dossier pass whose session is ``session`` ended with its map set, as its exit file says; the map is
    then the session's :data:`MAP_FILE`."""

    try:
        document = json.loads((session / EXIT_FILE).read_text(encoding=ENCODING))
    except (OSError, ValueError):
        return False
    return isinstance(document, dict) and document.get("reason") == MAPPED


async def split_directions(checkout: Path, map_file: Path, shares: int, out: Path) -> tuple[Path, ...]:
    """Split the open directions of ``map_file`` into ``shares`` disjoint plans in ``out``, with Arbor's own command
    line in ``checkout``; the plan files, in order, each a ``component.plan``. Raise :class:`SplitFailed` with what
    Arbor printed when it cannot: a map with fewer independent groups than shares, or one that does not validate."""

    paths = tuple(out / f"share-{number}.json" for number in range(1, shares + 1))
    await _directions(checkout, paths, "split", str(map_file), "--shares", str(shares), "--out", str(out))
    return paths


async def generate_directions(checkout: Path, profile: Path, out: Path) -> Path:
    """A map generated from the root profile ``profile``, the file ``component.profile`` names, by Arbor's own
    command line in ``checkout``, with no GPU time: one open direction per op family, ranked by its share. Written to
    ``out``, which :func:`split_directions` splits as any map. Raise :class:`SplitFailed` with what Arbor printed when
    it cannot: a profile with no share above its cut, or an Arbor older than ``generate``."""

    await _directions(checkout, (out,), "generate", "--profile", str(profile), "--out", str(out))
    return out


async def _directions(checkout: Path, written: tuple[Path, ...], *args: str) -> None:
    """Run ``python3 -m arbor.directions`` with ``args`` in ``checkout``; raise :class:`SplitFailed` with what it
    printed when it fails or leaves any of ``written`` unwritten."""

    argv = [PYTHON, "-m", "arbor.directions", *args]
    process = await asyncio.create_subprocess_exec(
        *argv, cwd=checkout, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT
    )
    output, _ = await process.communicate()
    said = " | ".join(line.strip() for line in output.decode(errors="replace").strip().splitlines()[-_TAIL_LINES:])
    if process.returncode != 0:
        raise SplitFailed(f"`{shlex.join(argv)}` exited {process.returncode}: {said}")
    missing = [path.name for path in written if not path.is_file()]
    if missing:
        raise SplitFailed(f"`{shlex.join(argv)}` wrote no {', '.join(missing)}: {said}")


__all__ = [
    "BLOCK_KEYS",
    "BRANCH_RUN",
    "BUILT",
    "DOSSIER_DIR",
    "DOSSIER_INDEX",
    "DOSSIER_PASS",
    "EXIT_REFUSED",
    "INSTALL",
    "LAUNCH_KEYS",
    "COMPONENT_MODULE",
    "MAP_FILE",
    "MAPPED",
    "MARKERS",
    "REVIEW_GATE_KEY",
    "REVIEW_HOST",
    "REVIEW_KEY",
    "REVIEW_OFF",
    "ArborCheckout",
    "ArborRefused",
    "ArborRun",
    "ResolvedLaunch",
    "SplitFailed",
    "arbor_config",
    "arbor_manifest",
    "bench_command",
    "carry_dossier",
    "generate_directions",
    "has_review_gate",
    "is_arbor",
    "link_built",
    "mapped",
    "pass_dossier",
    "read_launch",
    "refusals",
    "refused_reasons",
    "server_flags",
    "split_directions",
]
