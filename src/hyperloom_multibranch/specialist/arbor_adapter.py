# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Arbor as an autonomous branch's agent, made ready without the onboarding model: the Arbor lifecycle adapter.

Arbor runs inside a host in its component mode, whose contract is its ``docs/hosts.md``: one config, a snapshot of each
kept change in its session's outbox, an exit file written before every exit 0, and exit 2 for a config it refuses,
which a host never starts again. A checkout that has that mode, its entry point and that contract is Arbor's
(:func:`is_arbor`), and the preflight (:mod:`agent_onboarding`) takes its manifest from here instead of asking the
model: the config follows from the branch's spec alone (:func:`arbor_config`), on Arbor's serving template pack, with
the gate's client as its bench command (:func:`bench_command`), an AgentX round's report read by Arbor's harness at
:data:`REPORT_PATHS`, and its session in its checkout, the directory the manifest's snapshots name. Before Arbor
starts, the config is checked against what Arbor refuses at its start (:func:`refusals`): Arbor's ``--dry-run`` runs
its loop on synthetic numbers and checks none of it. Another attempt would write the same config, so a refusal, this
side's or Arbor's own, fails the preflight at once.

The spec's ``plan``, the file of the branch's share of the session's directions, becomes ``component.plan``. Those
shares come from one map, which Arbor's dossier pass writes on one lease before the branches open
(:data:`DOSSIER_PASS`, its end read by :func:`mapped`) or Arbor generates from the root's profile with no GPU time
(:func:`generate_directions`), split by Arbor's own command line (:func:`split_directions`); the scheduling calls these.
The dossier the pass wrote (:func:`pass_dossier`) is copied into each branch's checkout (:func:`carry_dossier`), where
Arbor looks a dossier up. The root's op shares are not the adapter's: the preflight sets
``component.profile`` for Arbor as for any agent run as a component
(:func:`~hyperloom_multibranch.specialist.agent_onboarding.component_config`).

The spec's ``launch``, a launch file (:func:`read_launch`), stands in for what MB does not carry yet: the launch an
AgentX recipe resolves from InferenceX, of which the root recipe holds only what is added to it. Without one,
an AgentX branch is refused before Arbor starts, which would otherwise serve defaults MB does not measure. The file
may also be the session's frozen base launch (:data:`BASE_LAUNCH_SCHEMA`), the one MB's gate measures every change on.
Arbor is then served its launch, and checks at its start that the launch it would run is the record's
(``component.base_launch``, carried into its checkout), refusing before any GPU work when it is not; it serves with the
record's entrypoint when that is not the framework's own (``component.entrypoint``).

The spec's ``review`` hands Arbor's human review gate over: Arbor waits at the start of a new
session for a person's approval unless the host says it reviewed (``component.review: "host"``, :data:`REVIEW_HOST`)
or the config turns the gate off (``"review_gate": false``, :data:`REVIEW_OFF`). Arbor refuses a component key it does
not know, so either is written only for a checkout whose component mode has the gate (:func:`has_review_gate`); an
older Arbor gets neither and runs as before. The keys a checkout declares (:func:`declared_keys`) decide the others
too: a key that changes what Arbor runs, which the pinned Arbor lacks, refuses its start, naming the Arbor commit to
pin (:data:`KEY_SINCE`); an input it can run without (:data:`OPTIONAL_KEYS`) is left out with a note (:func:`fit_keys`).
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
import os
import re
import shlex
import shutil
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass, field
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlsplit

import yaml

from hyperloom.common.gain_math import (
    ACCURACY_THRESHOLD,
    AGENTX_ERROR_RATE_THRESHOLD_PCT,
    DEFAULT_ENABLEMENT_ACCURACY_FLOOR,
)
from hyperloom.common.perf_metric import (
    GRADED_INTVTY,
    GRADED_INTVTY_P50,
    GRADED_OUTPUT,
    TOTAL_TOKEN_THROUGHPUT,
)
from hyperloom.inference_optimizer.framework_registry import expert_parallel_flag
from hyperloom.inference_optimizer.grid_server_args import merge_server_args
from hyperloom.measure.compile import (
    COMPONENT_LAYOUTS,
    UnsupportedComponentError,
    locate_installed_package,
)
from hyperloom.measure.surface import POINT_ENV_KEYS, REQUEST_ENV_KEYS, SURFACE_ENV_KEYS
from hyperloom_multibranch.repo.isolated_git import pinned_git
from hyperloom_multibranch.rulegate.agentx import AgentXObjective
from hyperloom_multibranch.specialist.agent_manifest import EXIT_FILE, AgentManifest
from hyperloom_multibranch.specialist.agent_spec import AgentSpec
from hyperloom_multibranch.specialist.worktree import SKIPPED
from inference_suite.argv import remove_server_args, tokenize_server_args_preserving_json
from inference_suite.contracts import BASE_LAUNCH_SCHEMA, BaseLaunch, BaseLaunchError, RunStatus
from inference_suite.contracts.layout import REPORT_FILE


#: What makes a checkout Arbor's, run as a component: the mode, the entry point and the host's contract.
MARKERS = ("arbor/component.py", "arbor/__main__.py", "docs/hosts.md")
HOST = "hyperloom-mb"
#: Arbor's serving template, the pack a host runs it on unless the spec names another, and the validator scripts a pack
#: has when its pack.yaml names none (``validator.bench`` and ``validator.correctness`` name its own: the agentx pack's
#: is ``validator/bench_agentx.sh``).
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
#: A placeholder of the repository's launch template still to be filled: ``<NAME>``, or a description such as ``<the
#: name the server answers to>``. A literal tag in a real flag, ``<think>``, is neither.
_PLACEHOLDER = re.compile(r"<(?:[A-Z_]+|[^<>]*\s[^<>]*)>")
#: Where a branch's checkout carries the base launch record it is served, beside its config.
BASE_LAUNCH_FILE = f"{MB_DIR}/base_launch.json"
#: The pass floor of the serving pack's GSM8K check, which Arbor reads from the config's ``server_env``; without one
#: it books every measured iteration whose accuracy it must judge as invalid. MB gives it its own absolute floor,
#: ``DEFAULT_ENABLEMENT_ACCURACY_FLOOR``, the part of MB's accuracy rule that holds on any eval's scale: MB's eval
#: and the pack's lm_eval score one model points apart, so the rule's other part, at most ``ACCURACY_THRESHOLD``
#: under the baseline's score, Arbor applies on its own eval: it measures the root's full GSM8K itself and holds a
#: numerics change to that less ``ACCURACY_POLICY``'s tolerance; this floor stands when it could not. MB still judges
#: its own, at its confirm test. A launch file or the recipe may set its own floor.
ACCURACY_FLOOR_ENV = "GSM8K_MIN"
#: Arbor's policy under MB: a numerics change may cost what MB's rule allows, ``ACCURACY_THRESHOLD`` under the root,
#: measured on Arbor's own eval against the root it scores itself.
ACCURACY_POLICY = {"accuracy_tolerance": ACCURACY_THRESHOLD}
#: The pack's evaluation settings, which Arbor reads from its campaign config and never from what an iteration
#: changed: no keep carries them into a recipe.
EVAL_ENVS = (ACCURACY_FLOOR_ENV, "GSM8K_LIMIT")
#: How each of MB's rules starts among Arbor's ``directives``, which every agent reads in its program and the dossier
#: copies into part 4's rules table (Arbor's docs/hosts.md, the config's ``directives``).
HOST_RULE = "Host rule (MB)"
#: The workload kind whose recipe the round resolves, the root's launch holding only what is added to it.
AGENTX = "agentx"
#: Arbor's key (``arbor/objective.py``) for each AgentX round metric MB keeps on, guards or records.
ARBOR_METRICS = {
    GRADED_INTVTY_P50: "e2e_norm_intvty_p50",
    GRADED_INTVTY: "e2e_norm_intvty_p90",
    TOTAL_TOKEN_THROUGHPUT: "total_token_throughput",
    GRADED_OUTPUT: "output_throughput",
}
#: The config key Arbor optimizes a metric by in place of its pack's (``arbor/objective.py``), and the key it reads as
#: each AgentX objective MB keeps on: ``e2e_norm_intvty_p50`` as MB reads it, and the total token throughput per GPU,
#: which ranks a change as MB's per-server figure does at a session's fixed TP. Another session's metric is its pack's.
OBJECTIVE_KEY = "opt_objective"
ARBOR_OBJECTIVES = {objective.metric: ARBOR_METRICS[objective.metric] for objective in AgentXObjective}
#: Where an Arbor checkout lists the metrics it optimizes on :data:`OBJECTIVE_KEY` (``KNOWN``), and the commit from
#: which it knows each: ``e8d31b7`` added the key but read P50 interactivity as decode alone until ``9c455df``, and
#: ``df9b55a`` added the total token throughput.
OBJECTIVE_MODULE = "arbor/objective.py"
OBJECTIVE_SINCE = {"e2e_norm_intvty_p50": "9c455df", "total_token_throughput": "df9b55a"}
#: Where the gate's AgentX round report (``report.json``) holds each metric Arbor's harness reads from it
#: (``component.bench_read``), by Arbor's key: either objective, every guard of either, and the output throughput MB
#: records. Throughput is the server's, not per GPU.
REPORT_PATHS = {
    ARBOR_METRICS[GRADED_INTVTY_P50]: "agentx.e2e_norm_intvty.p50",
    ARBOR_METRICS[GRADED_INTVTY]: "agentx.e2e_norm_intvty.p90",
    ARBOR_METRICS[TOTAL_TOKEN_THROUGHPUT]: "agentx.throughput.total_tokens_per_second",
    ARBOR_METRICS[GRADED_OUTPUT]: "agentx.throughput.output_tokens_per_second",
}
#: SGLang reads its TP from these flags only. Arbor gives vLLM its TP itself, but MB names it too, so a vLLM branch's
#: server_flags are never empty, which Arbor refuses (a plain baseline root has no launch of its own).
SGLANG = "sglang"
SGLANG_TP_FLAGS = ("--tp-size", "--tp", "--tensor-parallel-size")
VLLM = "vllm"
VLLM_TP_FLAGS = ("--tensor-parallel-size", "-tp")
#: The interpreter Arbor runs under, the sandbox's.
PYTHON = "python3"
#: Where Arbor imports what pip installed for it, which Arbor puts on its own import path at its import: a sandbox
#: whose pip installs out of every interpreter's reach (``PIP_TARGET``, as a confined one does) gives that directory
#: to Arbor's own commands, its driver and the ``arbor.*`` commands its agents run, never to the sandbox's
#: environment, where an inference server started from Arbor's shell would import the SDK's dependencies ahead of the
#: image's. An Arbor that does not read it runs as before, and its install's last command says why the SDK does not
#: import for it.
SITE_ENV = "ARBOR_SITE"
#: The Arbor commits that read :data:`SITE_ENV`: on main, and the tag on its v0.9.2 line.
SITE_SINCE = "545a850 (v0.9.2.1)"
#: The file Arbor (v0.9.4 on) reads its ``ARBOR_*`` defaults from, ``~/.arbor/settings.env`` when unset or empty.
#: Pointed at an empty file, so a local run never takes the running user's settings (an ``ARBOR_ROLL``, a review gate,
#: an agent backend) past the preflight that refuses them in the environment.
SETTINGS_ENV = "ARBOR_SETTINGS"
#: :data:`PYTHON` run as Arbor: with no site settings file, and given :data:`SITE_ENV` where the sandbox's pip installs
#: out of reach.
ARBOR_PYTHON = f'env {SETTINGS_ENV}=/dev/null ${{PIP_TARGET:+"{SITE_ENV}=$PIP_TARGET"}} {PYTHON}'
#: Run once Arbor's imports failed, says so when the cause is an Arbor that does not read :data:`SITE_ENV`: it names a
#: directory, which is not on the import path Arbor's import left.
_SITE_UNREAD = (
    f'import os, sys, arbor; site = os.environ.get("{SITE_ENV}", ""); '
    "sys.exit(os.path.isdir(site) and os.path.realpath(site) not in map(os.path.realpath, sys.path) and "
    f'f"this Arbor does not read {SITE_ENV}, so it cannot import what pip installed in {{site}}: '
    f'pin Arbor {SITE_SINCE} or later")'
)
#: The oldest Python the checkout runs on: its ``arbor/version.py`` ``PYTHON_MIN`` (3.10 from Arbor ``6126676``), read
#: without importing Arbor, since nothing is installed yet; 3.11, what Arbor needed before it named one, without it.
_PYTHON_MIN = (
    'import os, re, sys; p = "arbor/version.py"; t = open(p).read() if os.path.exists(p) else ""; '
    'm = re.search(r"PYTHON_MIN[^=]*=\\s*\\((\\d+),\\s*(\\d+)\\)", t); '
    "need = tuple(int(g) for g in m.groups()) if m else (3, 11); "
    'sys.exit(sys.version_info[:2] < need and "Arbor needs Python %d.%d or newer" % need)'
)
#: What Arbor needs beyond the image (its docs/getting-started.md, and the third-party modules arbor/ imports outside
#: its tests): the Python its checkout names, PyYAML and the Claude agent SDK, which brings its CLI. Each is installed
#: only when Arbor cannot import it, and the last command checks that Arbor imports both.
INSTALL = (
    f"{PYTHON} -c '{_PYTHON_MIN}'",
    f"{ARBOR_PYTHON} -c 'import arbor, yaml' 2>/dev/null || {PYTHON} -m pip install --quiet pyyaml",
    f"{ARBOR_PYTHON} -c 'import arbor, claude_agent_sdk' 2>/dev/null"
    f" || {PYTHON} -m pip install --quiet claude-agent-sdk",
    f"{ARBOR_PYTHON} -c 'import arbor, yaml, claude_agent_sdk' || {{ {ARBOR_PYTHON} -c '{_SITE_UNREAD}'; exit 1; }}",
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
    "base_launch",
    "entrypoint",
    "bench_read",
)
#: The server command each framework's Arbor adapter runs unless ``component.entrypoint`` gives another.
ENTRYPOINTS = {SGLANG: ("python3", "-m", "sglang.launch_server"), VLLM: ("vllm", "serve")}
#: How a spec hands Arbor's review gate over: the host reviewed the campaign before it starts the run (the only value
#: ``component.review`` takes), or nobody reviews it (``"review_gate": false`` at the config's top level).
REVIEW_KEY = "review"
REVIEW_HOST = "host"
REVIEW_OFF = "off"
REVIEW_GATE_KEY = "review_gate"
#: Where an Arbor checkout declares its component block's keys, which says whether it has the review gate.
COMPONENT_MODULE = "arbor/component.py"
#: The Arbor commit that added each key MB writes, which a refusal names; the rest came with component mode (32bf0c7).
KEY_SINCE = {
    "root": "4249384",
    "plan": "0a17a7b",
    "stop_at": "15a46e6",
    "profile": "15a46e6",
    "review": "e28249f",
    "base_launch": "ba2f5a8",
    "entrypoint": "ba2f5a8",
    "bench_read": "6a62def",
}
COMPONENT_SINCE = "32bf0c7"
#: The keys an Arbor that lacks them runs without, losing only an input or a check: left out, with a note, rather than
#: refused. An older Arbor runs the record's launch without checking it.
OPTIONAL_KEYS = ("profile", "base_launch")
#: Arbor's ledger: FACT and DEAD_END records are what it measured and ruled out.
LEDGER = {"id": "id", "kind": "type", "text": "claim", "kinds": {"FACT": "insight", "DEAD_END": "pitfall"}}
ENCODING = "utf-8"
#: The client's flags Arbor fills: its server's port and an empty directory per replay, ``--run-dir`` an
#: inference_suite round's, whose config names its server.
BASE_URL, PORT, RESULT_DIR, RUN_DIR, MODEL = "--base-url", "--port", "--result-dir", "--run-dir", "--model"
RESULT_DIRS = (RESULT_DIR, RUN_DIR)
#: Arbor's phase that runs the bench command, whose cap (``component.phase_minutes``) MB sets when its client is
#: bounded otherwise than by Arbor's default.
BENCH_PHASE = "bench"
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
    own launch is added to it, as a round adds a root recipe's ``extra_server_args`` and ``extra_envs``."""

    server_flags: str = ""
    server_env: Mapping[str, str] = field(default_factory=dict)
    model_name: str = ""
    client: str = ""
    #: A base launch record's file, its digest and its entrypoint; empty for a launch file.
    record: str = ""
    digest: str = ""
    entrypoint: tuple[str, ...] = ()


def read_launch(path: Path) -> ResolvedLaunch:
    """The launch in the file ``path``, a JSON object of :data:`LAUNCH_KEYS` (the repository's
    ``examples/multibranch-arbor-launch`` has a template), or a base launch record (:func:`_read_record`); raise
    :class:`ArborRefused` with every reason it cannot be used: unreadable, a key it does not have, no ``server_flags``,
    a ``server_env`` that is not an object of strings, a template placeholder left unfilled, or a ``client`` Arbor
    could not fill."""

    try:
        document = json.loads(path.read_text(encoding=ENCODING))
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ArborRefused([f"the launch file {path} cannot be read: {error}"]) from error
    if not isinstance(document, dict):
        raise ArborRefused([f"the launch file {path} is not a JSON object"])
    if document.get("schema") == BASE_LAUNCH_SCHEMA:
        return _read_record(path, document)
    unknown = sorted(set(document) - set(LAUNCH_KEYS))
    reasons = [f"it has keys a launch does not: {', '.join(unknown)}"] if unknown else []
    flags, env = document.get("server_flags"), document.get("server_env")
    if not isinstance(flags, str) or not flags.strip():
        reasons.append("it gives no server_flags, the launch's server arguments")
    if not isinstance(env, dict) or not all(isinstance(value, str) for value in env.values()):
        reasons.append("its server_env is not an object of strings")
    texts = {key: document.get(key, "") for key in LAUNCH_KEYS[2:]}
    reasons += [f"its {key} is not a string" for key, value in texts.items() if not isinstance(value, str)]
    unfilled = [key for key in LAUNCH_KEYS if _PLACEHOLDER.search(json.dumps(document.get(key, "")))]
    if unfilled:
        reasons.append(f"its {', '.join(unfilled)} still hold the template's placeholders: fill them in")
    if reasons:
        raise ArborRefused([f"the launch file {path}: {reason}" for reason in reasons])
    if texts["client"]:
        bench_command(texts["client"])
    return ResolvedLaunch(flags, env, texts["model_name"], texts["client"])


def _read_record(path: Path, document: Mapping[str, Any]) -> ResolvedLaunch:
    """The launch the base launch record ``document`` read from ``path`` serves: its server argv and environment as the
    gate seeds a launch with them (:func:`~hyperloom.orchestrator.state.base_launch.server_base`), the record itself,
    its digest and its entrypoint, which Arbor checks its own launch against at its start. The record names no
    served model and no client command: those are the gate's client's. Raise :class:`ArborRefused` when it is no
    valid record, or a scriptable base, which launches no server."""

    from hyperloom.orchestrator.state.base_launch import server_argv, server_base

    try:
        base = BaseLaunch.from_document(document)
    except BaseLaunchError as error:
        raise ArborRefused([f"the base launch record {path} cannot be used: {error}"]) from error
    if base.scriptable:
        raise ArborRefused([f"the base launch record {path} runs a script and launches no server for Arbor"])
    return ResolvedLaunch(
        server_flags=" ".join(_flag_word(word) for word in server_argv(base)),
        server_env=dict(server_base(base).env),
        record=str(path),
        digest=base.digest(),
        entrypoint=base.entrypoint,
    )


def _flag_word(word: str) -> str:
    """A record's word as Arbor's server_flags carry it: a JSON value compact and unquoted, as Arbor splits one;
    another word quoted for the shell only when it must be."""

    if word[:1] in "{[":
        try:
            return json.dumps(json.loads(word), separators=(",", ":"))
        except ValueError:
            pass
    return shlex.quote(word)


@dataclass(frozen=True)
class ArborCheckout:
    """Arbor at ``commit`` of ``source``, checked out at ``path``."""

    source: str
    commit: str
    path: Path


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
    """Whether the Arbor in ``checkout`` has the human review gate: its component block takes :data:`REVIEW_KEY`
    (Arbor ``1fa2069`` and later); ``False`` when its keys cannot be read (:func:`declared_keys`)."""

    return REVIEW_KEY in (declared_keys(checkout) or ())


def declared_keys(checkout: Path) -> frozenset[str] | None:
    """The component keys the Arbor in ``checkout`` takes, as its :data:`COMPONENT_MODULE` declares ``BLOCK_KEYS``,
    read without importing Arbor; ``None`` for a checkout without that module, or whose keys cannot be read."""

    try:
        tree = ast.parse((checkout / COMPONENT_MODULE).read_text(encoding=ENCODING))
    except (OSError, SyntaxError, UnicodeDecodeError):
        return None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "BLOCK_KEYS" for t in node.targets):
            if isinstance(node.value, (ast.Tuple, ast.List)):
                keys = [e.value for e in node.value.elts if isinstance(e, ast.Constant)]
                return frozenset(key for key in keys if isinstance(key, str))
    return None


def fit_keys(config: Mapping[str, Any], checkout: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    """``config`` without the :data:`OPTIONAL_KEYS` the Arbor in ``checkout`` lacks, and a note for each one left out:
    an older pin runs without that input rather than refuse its start. Unchanged when its keys cannot be read."""

    declared, component = declared_keys(checkout), dict(config.get("component") or {})
    lacking = [key for key in OPTIONAL_KEYS if key in component and declared is not None and key not in declared]
    for key in lacking:
        del component[key]
    notes = tuple(
        f"the pinned Arbor has no component.{key} (Arbor {KEY_SINCE[key]}): started without it" for key in lacking
    )
    return {**config, "component": component}, notes


def arbor_manifest(spec: AgentSpec, checkout: ArborCheckout, run: ArborRun) -> AgentManifest:
    """The manifest of ``run`` of Arbor in ``checkout`` for the branch ``spec`` names, its session made and its
    workspace given the installed framework's built files (:func:`link_built`); raise :class:`ArborRefused` with every
    reason Arbor would refuse its config for.

    The root's op shares are its caller's to add, as the preflight does for every agent run as a component; the
    dossier pass profiles only what they do not cover."""

    path = checkout.path.absolute()
    if Path(spec.workspace).is_dir():
        link_built(Path(spec.workspace), spec.framework)
    if spec.dossier:
        carry_dossier(path, Path(spec.dossier))
    config = arbor_config(spec, path, run)
    reasons = [
        *_launch_refusals(spec),
        *_objective_refusals(config, path),
        *refusals(config, declared_keys(path)),
    ]
    if reasons:
        raise ArborRefused(reasons)
    (path / run.session_dir).mkdir(parents=True, exist_ok=True)
    component = config["component"]
    if component.get("plan"):
        carried = path / PLAN_FILE
        carried.write_bytes(Path(component["plan"]).read_bytes())
        component["plan"] = str(carried)
    if component.get("base_launch"):
        carried = path / BASE_LAUNCH_FILE
        carried.parent.mkdir(parents=True, exist_ok=True)
        carried.write_bytes(Path(component["base_launch"]["path"]).read_bytes())
        component["base_launch"] = {**component["base_launch"], "path": str(carried)}
    launch = f"{ARBOR_PYTHON} -m arbor run --config {shlex.quote(run.config_file)}"
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


def link_built(workspace: Path, framework: str) -> int:
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
        package = locate_installed_package(layout)
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


def dossier_notes(checkout: Path, dossier: str | None) -> tuple[str, ...]:
    """A note for the branch when ``dossier``, the session's dossier pass's, is named but neither there nor already in
    ``checkout``: :func:`carry_dossier` copies nothing then, and the branch runs on its share with no dossier."""

    if not dossier or Path(dossier).is_dir() or (checkout / DOSSIER_DIR / Path(dossier).name).is_dir():
        return ()
    return (f"the dossier pass's dossier {dossier} is not there: started without a dossier",)


def pass_dossier(checkout: Path, session: Path) -> Path | None:
    """The dossier a dossier pass in ``checkout`` wrote: the newest directory with an index under :data:`DOSSIER_DIR`
    of the checkout, where git does not track it (the ones it tracks are Arbor's own), or of the pass's ``session``,
    where Arbor's dossier skill writes it when its agent works from there;
    ``None`` when it wrote none."""

    written = [*_written(checkout / DOSSIER_DIR, _shipped(checkout)), *_written(session / DOSSIER_DIR, frozenset())]
    return max(written, key=lambda path: (path / DOSSIER_INDEX).stat().st_mtime, default=None)


def _shipped(checkout: Path) -> frozenset[str]:
    """The dossiers ``checkout`` ships, the ones git tracks under :data:`DOSSIER_DIR`, read with no program the
    checkout's config names run (:func:`~hyperloom_multibranch.repo.isolated_git.pinned_git`): Arbor writes it."""

    try:
        listed = pinned_git(checkout, "ls-files", "--", DOSSIER_DIR)
    except (OSError, subprocess.SubprocessError):
        return frozenset()
    depth = len(PurePosixPath(DOSSIER_DIR).parts)
    tracked = (PurePosixPath(line).parts for line in listed.stdout.splitlines())
    return frozenset(parts[depth] for parts in tracked if len(parts) > depth)


def _written(root: Path, shipped: frozenset[str]) -> list[Path]:
    """The directories of ``root`` with a dossier index, but for the ``shipped`` ones."""

    if not root.is_dir():
        return []
    return [path for path in root.iterdir() if path.name not in shipped and (path / DOSSIER_INDEX).is_file()]


def validator_script(pack: Path, key: str, default: str) -> str:
    """The script ``pack``'s pack.yaml names for ``key`` under ``validator:`` (``bench``, ``correctness``), relative to
    the pack; ``default`` when it names none, or the file does not read (Arbor then refuses the pack itself)."""

    try:
        manifest = yaml.safe_load((pack / "pack.yaml").read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        return default
    validator = manifest.get("validator") if isinstance(manifest, dict) else None
    named = validator.get(key) if isinstance(validator, dict) else None
    return named.strip() if isinstance(named, str) and named.strip() else default


def arbor_config(spec: AgentSpec, checkout: Path, run: ArborRun) -> dict[str, Any]:
    """Arbor's config for ``run`` of ``spec``: in component mode on the host, the recipe's launch on the pack the spec
    names in ``checkout`` (the serving template by default), ending by the spec's deadline, its patches made against
    its root, the integration notes read every iteration, its measurements as :func:`_bench_keys` gives them, and the
    host's plan when it gives one, which a run that stops at its map does not take. A spec with a launch file is served the file's launch with the
    recipe's added to it, and measured by the file's client when the spec has none.

    The cards are numbered from 0, as the sandbox shows them; Arbor never masks them itself in this mode."""

    name = spec.pack or PACK
    pack = checkout / "packs" / name
    launch = read_launch(Path(spec.launch)) if spec.launch else ResolvedLaunch()
    client = spec.client or launch.client
    component: dict[str, Any] = {"host": HOST, "deadline": int(spec.deadline), "root": spec.root, "outbox": OUTBOX}
    if spec.knowledge:
        component["notes"] = spec.knowledge
    component.update(_bench_keys(spec, client))
    if run.stop_at:
        component["stop_at"] = run.stop_at
    elif spec.plan:
        component["plan"] = spec.plan
    gate = spec.review if spec.review and has_review_gate(checkout) else ""
    if gate == REVIEW_HOST:
        component[REVIEW_KEY] = REVIEW_HOST
    if launch.record:
        component["base_launch"] = {"path": launch.record, "digest": launch.digest}
    if launch.entrypoint and launch.entrypoint != ENTRYPOINTS.get(spec.framework):
        component["entrypoint"] = list(launch.entrypoint)
    objective = ARBOR_OBJECTIVES.get(spec.metric)
    return {
        **({REVIEW_GATE_KEY: False} if gate == REVIEW_OFF else {}),
        **({OBJECTIVE_KEY: objective} if objective else {}),
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
        "server_env": {ACCURACY_FLOOR_ENV: str(DEFAULT_ENABLEMENT_ACCURACY_FLOOR), **launch.server_env, **spec.envs},
        "bench_script": str(pack / validator_script(pack, "bench", BENCH_SCRIPT)),
        "correctness_script": str(pack / validator_script(pack, "correctness", CORRECTNESS_SCRIPT)),
        "workspace_root": spec.workspace,
        "session_root": str(checkout / run.session_dir),
        "port": spec.port,
        "gpu_ids": list(range(len(spec.cards))),
        **_operating_point(spec.workload),
        "directives": host_rules(spec),
        "policy": dict(ACCURACY_POLICY),
        "component": component,
    }


def _bench_keys(spec: AgentSpec, client: str) -> dict[str, Any]:
    """The component keys of how Arbor measures ``spec``'s branch: every measurement by ``client`` when there is one,
    the report of the gate's AgentX round read by Arbor's harness (:func:`_report_keys`), and its bench phase as long
    as one measurement by the gate's client may take."""

    keys: dict[str, Any] = {}
    if client:
        keys["bench_command"] = bench_command(client)
    if spec.client and spec.workload.get("kind") == AGENTX:
        keys["bench_read"] = {"file": REPORT_FILE, "keys": _report_keys(_agentx_objective(spec.metric))}
    if spec.client_seconds:
        keys["phase_minutes"] = {BENCH_PHASE: math.ceil(spec.client_seconds / 60)}
    return keys


def _report_keys(objective: AgentXObjective) -> dict[str, str]:
    """Arbor's key and :data:`REPORT_PATHS` path of each metric MB reads from an AgentX round under ``objective``: the
    objective, its guards and the output throughput."""

    metrics = (objective.metric, *(guard.metric for guard in objective.guards), GRADED_OUTPUT)
    return {ARBOR_METRICS[metric]: REPORT_PATHS[ARBOR_METRICS[metric]] for metric in metrics}


def server_flags(spec: AgentSpec, resolved: str = "") -> str:
    """The recipe's server arguments as Arbor launches them, after the ``resolved`` launch's, less the flags the recipe
    removes from it, when there is one, with the workload's TP size unless the arguments set it (SGLang and vLLM); on
    SGLang also its expert-parallel size, unless the arguments set it, and only when it divides the TP, as MB's own
    launch has it."""

    args = merge_server_args(remove_server_args(resolved, spec.remove_args), spec.server_args)
    tp_flags = {SGLANG: SGLANG_TP_FLAGS, VLLM: VLLM_TP_FLAGS}.get(spec.framework)
    if tp_flags is None:
        return args
    named = _flag_names(args)
    added = [] if named & set(tp_flags) else [f"{tp_flags[0]} {spec.tp}"]
    flag, aliases, _ = expert_parallel_flag(SGLANG)
    if spec.framework == SGLANG and spec.ep > 1 and spec.tp % spec.ep == 0 and not named & set(aliases):
        added.append(f"{flag} {spec.ep}")
    return merge_server_args(args, *added)


def _launch_refusals(spec: AgentSpec) -> list[str]:
    """Why ``spec``'s branch cannot serve what MB measures: an AgentX recipe's launch, which holds only what the
    round adds to the one it resolves from InferenceX, given without that launch."""

    if spec.workload.get("kind") != AGENTX or spec.launch:
        return []
    return [
        "an AgentX recipe's launch holds only what the round adds to the one it resolves from InferenceX, so Arbor would"
        " serve defaults MB does not measure: give the resolved launch with --agent-launch, or the session's base"
        " with --reference-script (examples/multibranch-arbor-launch)"
    ]


def _flag_names(args: str) -> set[str]:
    split = tokenize_server_args_preserving_json(args)
    words = args.split() if split is None else split[1]
    return {word.partition("=")[0] for word in words if word.startswith("-")}


def _operating_point(workload: Mapping[str, object]) -> dict[str, int]:
    """The workload's concurrency and request shape, for the pack's own bench when the gate gives no client."""

    named = {"concurrency": workload.get("conc"), "isl": workload.get("isl"), "osl": workload.get("osl")}
    return {key: value for key, value in named.items() if isinstance(value, int) and not isinstance(value, bool)}


def host_rules(spec: AgentSpec) -> list[str]:
    """MB's rules for ``spec``'s runs, as Arbor's ``directives`` (:data:`HOST_RULE`): the workload it holds fixed and
    the variables it owns, how it keeps a change, its accuracy rule and, when the spec says why, that no source change
    can be kept. MB refuses or rejects a change that breaks one, whatever Arbor measured; the dossier lists them among
    the benchmark's rules."""

    point = ", ".join(f"{key} {value}" for key, value in _operating_point(spec.workload).items())
    owned = ", ".join(sorted({*SURFACE_ENV_KEYS, *POINT_ENV_KEYS, *REQUEST_ENV_KEYS}))
    kind = str(spec.workload.get("kind") or "the session's workload")
    objective = _agentx_objective(spec.metric) if kind == AGENTX else None
    kept = _agentx_kept(objective) if objective is not None else "that a second measurement confirms"
    rules = [
        f"{HOST_RULE}: the workload is fixed: {kind}{f' at {point}' if point else ''}, measured by MB's client, the"
        f" only bench of record. A change never sets {owned}: MB owns them and refuses a launch that does.",
        f"{HOST_RULE}: MB keeps a change only on its own measurement of {spec.metric} on its own cards, a gain above"
        f" its bar {kept}; Arbor's numbers decide only Arbor's own KEEPs.",
    ]
    if kind == AGENTX:
        rules.append(
            f"{HOST_RULE}: accuracy: the request error rate stays at or under {AGENTX_ERROR_RATE_THRESHOLD_PCT:g}%."
        )
    else:
        rules.append(
            f"{HOST_RULE}: accuracy: MB's own eval of a change scores at least {DEFAULT_ENABLEMENT_ACCURACY_FLOOR:g} and"
            f" at most {ACCURACY_THRESHOLD:g} under the root's score; a numerics change that costs more is never merged."
            f" Arbor's GSM8K reads on another scale, so Arbor holds the change to the same drop against the root it"
            f" measures itself, and to {DEFAULT_ENABLEMENT_ACCURACY_FLOOR:g} when it could not."
        )
    if objective is not None and spec.client:
        rules.append(
            f"{HOST_RULE}: MB's client is its own AgentX round, a replay of the workload's trace window, whose"
            f" {REPORT_FILE} Arbor's harness reads: a run whose run_status there is not {RunStatus.COMPLETED.value}"
            " measured nothing, whatever numbers it holds."
        )
    if spec.source_unkept:
        rules.append(
            f"{HOST_RULE}: a change to {spec.framework}'s source cannot be kept on this image: {spec.source_unkept}."
            " Spend the work on launch flags and environment variables."
        )
    return rules


def _agentx_objective(metric: str) -> AgentXObjective:
    return {objective.metric: objective for objective in AgentXObjective}[metric]


def _agentx_kept(objective: AgentXObjective) -> str:
    guards = " and ".join(guard.metric for guard in objective.guards)
    return (
        f"of +{objective.bar_pct:g}% in one measurement, which is final unless a guard holds it for a retest: {guards}"
        f" within {objective.guard_band_pct:g}% of the parent, and no more failed requests"
    )


def bench_command(client: str) -> str:
    """The gate's ``client`` as Arbor's ``component.bench_command``: the port it measures, in ``--base-url`` or
    ``--port``, as ``{port}``, and its ``--result-dir`` or ``--run-dir`` as ``{out}``, which Arbor quotes as it fills
    it; every other word as it was, the client's warm-up requests among them. Raise :class:`ArborRefused` when the
    client names no result directory, no port beside a ``--result-dir`` (a ``--run-dir`` is an inference_suite
    round's, whose config names its server), or cannot be read word by word."""

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
        if flag in (BASE_URL, PORT, *RESULT_DIRS):
            if joined:
                templated.append(f"{flag}={_filled(flag, value)}")
                found.add(flag)
            else:
                templated.append(flag)
                pending = flag
            continue
        templated.append(word if set(word) <= _OPERATORS else shlex.quote(word))
    reasons = [f"the gate's client ends at {pending}, which takes a value"] if pending else []
    if RUN_DIR not in found and not found & {BASE_URL, PORT}:
        reasons.append(f"the gate's client names no port to measure Arbor's server on ({BASE_URL} or {PORT})")
    if not found & set(RESULT_DIRS):
        reasons.append(f"the gate's client names no {RESULT_DIR} or {RUN_DIR} for Arbor to read each replay from")
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

    if flag in RESULT_DIRS:
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


def refusals(config: Mapping[str, Any], declared: frozenset[str] | None = None) -> list[str]:
    """Why Arbor would refuse ``config`` at its start, run with this process's environment, as far as this side can
    tell; ``[]`` when it would not. ``declared`` is the component keys the pinned Arbor takes (:func:`declared_keys`),
    when known.

    These are the refusals of Arbor's ``docs/hosts.md`` that a config written from the spec can meet: no session, no
    deadline, a container other than the host, ``ARBOR_ROLL=1``, a component block with a key Arbor does not have, a
    root that is no commit of the workspace, a workspace, script or pack that is not there, a plan that cannot be read,
    and a plan beside a dossier pass. What only Arbor can check, a plan's directions and the pack's own validation
    among them, it refuses at its start, with exit 2."""

    component = config.get("component")
    component = component if isinstance(component, Mapping) else {}
    return [
        *_block_refusals(component),
        *_pin_refusals(component, declared),
        *_host_refusals(config),
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


def _pin_refusals(component: Mapping[str, Any], declared: frozenset[str] | None) -> list[str]:
    """A key MB writes that the pinned Arbor lacks, naming the Arbor commit that added it: Arbor would refuse it too.
    An optional key is not refused: :func:`fit_keys` leaves it out, with a note, before Arbor starts."""

    known = declared if declared is not None else set(BLOCK_KEYS)
    lacking = sorted(set(component) & set(BLOCK_KEYS) - set(OPTIONAL_KEYS) - known)
    return [
        f"the pinned Arbor has no component.{key}: pin Arbor {KEY_SINCE.get(key, COMPONENT_SINCE)} or later"
        for key in lacking
    ]


def _objective_refusals(config: Mapping[str, Any], checkout: Path) -> list[str]:
    """An :data:`OBJECTIVE_KEY` the Arbor in ``checkout`` cannot optimize: one whose :data:`OBJECTIVE_MODULE` does not
    list it (``e8d31b7`` read P50 interactivity as decode alone), or, from an Arbor without that module, which ignores
    the key and climbs its pack's metric, any but P50 interactivity, which a session ran on such pins before the key
    existed (the vendored v0.9.3 among them)."""

    wanted = config.get(OBJECTIVE_KEY)
    known = _known_objectives(checkout)
    if not wanted or (wanted in known if known is not None else wanted == ARBOR_OBJECTIVES[GRADED_INTVTY_P50]):
        return []
    return [
        f"the pinned Arbor cannot optimize {wanted} ({OBJECTIVE_KEY}): pin Arbor {OBJECTIVE_SINCE[wanted]} or later"
    ]


def _known_objectives(checkout: Path) -> frozenset[str] | None:
    """The metric keys :data:`OBJECTIVE_MODULE` lists as ``KNOWN``, read without importing Arbor; ``None`` for a
    checkout without that module, empty when it does not read."""

    try:
        text = (checkout / OBJECTIVE_MODULE).read_text(encoding=ENCODING)
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError):
        return frozenset()
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return frozenset()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "KNOWN" for t in node.targets):
            if isinstance(node.value, ast.Dict):
                return frozenset(k.value for k in node.value.keys if isinstance(k, ast.Constant))
    return frozenset()


def _host_refusals(config: Mapping[str, Any]) -> list[str]:
    reasons = []
    if not config.get("session_root"):
        reasons.append("no session_root: the snapshots, the heartbeat and the exit file are written there")
    if config.get("docker_container") != "@host":
        reasons.append(f"docker_container is {config.get('docker_container')!r}, not @host")
    if os.environ.get("ARBOR_ROLL") == "1":
        reasons.append("ARBOR_ROLL=1 is set: a component's clock is its host's")
    if not str(config.get("server_flags") or "").strip():
        reasons.append("no server_flags: Arbor serves the launch MB measures, and an empty one serves defaults")
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
    """Whether ``sha`` is a commit of the workspace, the agent's to write, asked with no program its config names run."""
    try:
        found = pinned_git(workspace, "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}")
    except (OSError, subprocess.SubprocessError):
        return False
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


def exit_reason(path: Path) -> str:
    """Why Arbor says its run ended, as the exit file at ``path`` it writes before an exit 0 has it; else what is
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
    then the session's :data:`MAP_FILE`."""

    return exit_reason(session / EXIT_FILE) == MAPPED


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
    """Run ``python3 -m arbor.directions`` with ``args`` in ``checkout``, as Arbor's other commands with no site
    settings file (:data:`SETTINGS_ENV`); raise :class:`SplitFailed` with what it printed when it fails or leaves any
    of ``written`` unwritten."""

    argv = [PYTHON, "-m", "arbor.directions", *args]
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=checkout,
        env={**os.environ, SETTINGS_ENV: os.devnull},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    output, _ = await process.communicate()
    said = " | ".join(line.strip() for line in output.decode(errors="replace").strip().splitlines()[-_TAIL_LINES:])
    if process.returncode != 0:
        raise SplitFailed(f"`{shlex.join(argv)}` exited {process.returncode}: {said}")
    missing = [path.name for path in written if not path.is_file()]
    if missing:
        raise SplitFailed(f"`{shlex.join(argv)}` wrote no {', '.join(missing)}: {said}")


__all__ = [
    "ACCURACY_FLOOR_ENV",
    "ACCURACY_POLICY",
    "ARBOR_PYTHON",
    "BASE_LAUNCH_FILE",
    "BLOCK_KEYS",
    "BRANCH_RUN",
    "BUILT",
    "DOSSIER_DIR",
    "DOSSIER_INDEX",
    "DOSSIER_PASS",
    "EVAL_ENVS",
    "EXIT_REFUSED",
    "HOST_RULE",
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
    "SETTINGS_ENV",
    "SITE_ENV",
    "SITE_SINCE",
    "ArborCheckout",
    "ArborRefused",
    "ArborRun",
    "ResolvedLaunch",
    "SplitFailed",
    "arbor_config",
    "arbor_manifest",
    "bench_command",
    "carry_dossier",
    "dossier_notes",
    "exit_reason",
    "generate_directions",
    "has_review_gate",
    "host_rules",
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
