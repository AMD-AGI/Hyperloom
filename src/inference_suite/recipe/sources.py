# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Take the snapshot a session's base launch is resolved from: the one module that reaches the network or runs a
source's own code.

Every repository is read at the head of its ``main`` and recorded with that commit. The vLLM recipes site, the
model's HF card and the framework's latest release are recorded as the responses they were. Which variant an
InferenceX point binds, its acceptance length, its golden overrides and the golden length of a cookbook launch are
InferenceX's own code: they run here once, in an isolated interpreter of a throwaway virtual environment holding
InferenceX's ``recipes`` dependencies, and their answers are recorded as data that resolving reads. The environment is
built on the pinned CPython build (:func:`inference_suite.client_env.pinned_python`), never on the interpreter running
this module, and only once that build satisfies the checkout's ``requires-python``.

A source that cannot be read, InferenceX included, stops the reading there, and the snapshot names it. The
operator's own launch file needs no source read: the load client is pinned in :mod:`inference_suite.pins`.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote

from packaging.specifiers import InvalidSpecifier, SpecifierSet

from inference_suite.client_env import ClientEnvError, pinned_python
from inference_suite.contracts.base_launch import LaunchQuery, Workload
from inference_suite.recipe.atom_catalog import ATOM_REPO, CATALOG, RECIPES_DIR
from inference_suite.recipe.golden_curve import GoldenLength, GoldenRequest, golden_requests
from inference_suite.recipe.image_version import release_version
from inference_suite.recipe.inferencex import (
    GOLDEN_DIR,
    INFERENCEX_REPO,
    MASTER,
    PROJECT,
    RUNNERS,
    InferencexPoint,
    served_points,
)
from inference_suite.recipe.model import ApiResponse, InferencexSelection, InferencexSnapshot, RepoBundle, SourceMiss
from inference_suite.recipe.resolve import SourceSnapshot, UnreadSource
from inference_suite.recipe.sglang_cookbook import (
    AGENTX_DIR,
    COOKBOOK_DIR,
    CONFIG_DIR,
    SGLANG_REPO,
    sglang_cookbook_launches,
)
from inference_suite.recipe.shell_launch import ShellLaunch
from inference_suite.recipe.user_file import UserFile
from inference_suite.recipe.vllm_recipes import VllmRecipesSnapshot, vllm_recipes_launches, vllm_recipes_requests

#: The only branch a source is read from: what it has merged, never a pull request or a tag.
MAIN = "main"
VLLM_REPO = "https://github.com/vllm-project/vllm"
HF_MODELS_API = "https://huggingface.co/api/models"
#: The srt-slurm submodule InferenceX pins, under its project; its commit pins srtctl.
SRT_SLURM = f"{PROJECT}/utils/srt-slurm"

_FRAMEWORK_REPOS = {"sglang": SGLANG_REPO, "vllm": VLLM_REPO, "atom": ATOM_REPO}
_INFERENCEX_SOURCE = "InferenceX"
_RECIPES_EXTRA = "recipes"
_QUIET = "--quiet"
_OK = 200
# HF answers 401, not 404, for a repository it does not know when asked anonymously.
_NO_MODEL_CARD = frozenset({401, 404})
_HTTP_TIMEOUT_S = 60
_COMMAND_TIMEOUT_S = 900
_OUTPUT_TAIL = 2000

# Run as ``python -I -c`` in the project directory of an InferenceX checkout: reads a request on stdin and writes the
# answers as JSON on stdout. Upstream output is sent to stderr so stdout carries the answers alone. A ValueError is
# how InferenceX refuses a point or a draft config, so it is recorded as the answer.
_INFERENCEX_CHILD = """
import hashlib
import json
import sys
from pathlib import Path

request = json.load(sys.stdin)
answers, sys.stdout = sys.stdout, sys.stderr
project = Path(request["project"])
sys.path[:0] = [str(project), str(project / "utils" / "srt-slurm" / "src")]

from infx.golden_al_distribution import GOLDEN_DIR, curve_name, golden_length
from infx.srt_slurm.single_node import select_recipe
from infx.srt_slurm.synthetic_acceptance import (
    ENGINES, acceptance_length, build_overrides, generation_role, spec_parameters, verifier,
)
from infx.srt_slurm.workload import NATIVE_TELEMETRY_BLOCK, SHARED_BLOCKS


def selection(recipe_path, environment):
    variant, recipe = select_recipe(str(project / recipe_path), environment, root=project)
    recorded = json.loads(json.dumps(recipe))
    framework = environment["FRAMEWORK"]
    length = acceptance_length(recipe, framework, environment)
    curve = None
    if length is not None:
        engine = ENGINES[framework]
        generation = generation_role(recipe)
        spec = spec_parameters(generation, engine)
        curve = curve_name(environment["MODEL_PREFIX"], spec, GOLDEN_DIR, verifier(generation, engine).value)
    composed = (
        recipe_path.partition(":")[0],
        str(SHARED_BLOCKS[environment["IS_AGENTIC"] == "1", False]),
        str(NATIVE_TELEMETRY_BLOCK),
    )
    return {
        "variant": variant.removeprefix(f"{project}/"),
        "recipe": recorded,
        "files": {path: hashlib.sha256((project / path).read_bytes()).hexdigest() for path in composed},
        "acceptance_length": length,
        "golden_overrides": build_overrides(recipe, framework, environment),
        "golden_curve": curve,
    }


def golden(asked):
    try:
        value = golden_length(asked["model_prefix"], asked["spec"], asked["thinking"], GOLDEN_DIR, asked["verify"])
        curve = curve_name(asked["model_prefix"], asked["spec"], GOLDEN_DIR, asked["verify"])
    except ValueError as error:
        return {"value": None, "curve": None, "error": str(error)}
    return {"value": value, "curve": curve, "error": None}


selections, rejections = {}, {}
for point in request["points"]:
    try:
        selections[point["key"]] = selection(point["recipe"], point["environment"])
    except ValueError as error:
        rejections[point["key"]] = str(error)
lengths = {asked["key"]: golden(asked["request"]) for asked in request["golden"]}
answers.write(json.dumps({"selections": selections, "rejections": rejections, "golden_lengths": lengths}))
"""

# Run as ``python -I -c`` with an InferenceX ``pyproject.toml``: the interpreter's version and the project's
# ``requires-python`` ("" when it states none) as JSON. Run by the interpreter it asks about, which has tomllib.
_REQUIRES_PYTHON_CHILD = """
import json
import platform
import sys
import tomllib

with open(sys.argv[1], "rb") as pyproject:
    requires = tomllib.load(pyproject).get("project", {}).get("requires-python", "")
print(json.dumps({"version": platform.python_version(), "requires": requires}))
"""


class SourceFetchError(RuntimeError):
    """A source could not be read, or its own code failed rather than answer."""


def _run(command: Sequence[str], *, cwd: Path | None = None, stdin: str | None = None) -> str:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            input=stdin,
            capture_output=True,
            text=True,
            timeout=_COMMAND_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise SourceFetchError(f"{command[0]} {command[1]} ran past {_COMMAND_TIMEOUT_S} s") from error
    if completed.returncode != 0:
        raise SourceFetchError(
            f"{command[0]} {command[1]} exited {completed.returncode}: {completed.stderr.strip()[-_OUTPUT_TAIL:]}"
        )
    return completed.stdout


def _git(*args: str, cwd: Path | None = None) -> str:
    return _run(("git", *args), cwd=cwd).strip()


@dataclass(frozen=True)
class Checkout:
    """A sparse checkout of ``repo`` at ``commit`` in ``root``."""

    repo: str
    commit: str
    root: Path

    def paths(self, directory: str, suffix: str, *, nested: bool = True) -> list[str]:
        """The files ending in ``suffix`` under ``directory``, in it alone unless ``nested``."""
        base = self.root / directory
        found = base.rglob(f"*{suffix}") if nested else base.glob(f"*{suffix}")
        return sorted(path.relative_to(self.root).as_posix() for path in found if path.is_file())

    def bundle(self, paths: Sequence[str]) -> RepoBundle:
        """``paths`` as the checkout holds them, byte for byte."""
        return RepoBundle(
            self.repo, self.commit, {path: (self.root / path).read_bytes().decode("utf-8") for path in paths}
        )


def checkout_main(repo: str, directories: Sequence[str], into: Path) -> Checkout:
    """The head of ``repo``'s ``main`` checked out in ``into``, with only ``directories`` (and the root's files)."""
    _git(
        *("clone", _QUIET, "--no-tags", "--depth=1", "--filter=blob:none", "--sparse"),
        *("--single-branch", "--branch", MAIN, repo, str(into)),
    )
    _git("sparse-checkout", "set", *directories, cwd=into)
    return Checkout(repo, _git("rev-parse", "HEAD", cwd=into), into)


def fetch_response(url: str) -> ApiResponse:
    """The response ``url`` answers with now, an HTTP error status included."""
    request = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "inference-suite"})
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT_S) as response:
            status, body = response.status, response.read()
    except urllib.error.HTTPError as error:
        status, body = error.code, error.read()
    except OSError as error:
        raise SourceFetchError(f"{url} could not be fetched: {error}") from error
    try:
        text = body.decode("utf-8")
    except UnicodeDecodeError as error:
        raise SourceFetchError(f"{url} answered with a body that is not UTF-8: {error}") from error
    return ApiResponse(url, status, text, int(time.time()))


def _require_python(python: Path, project: Path) -> None:
    """Refuse unless ``python`` satisfies the ``requires-python`` of the InferenceX project at ``project``: pip would
    otherwise backtrack through every release of each dependency until the command timeout."""
    asked = json.loads(_run((str(python), "-I", "-c", _REQUIRES_PYTHON_CHILD, str(project / "pyproject.toml"))))
    try:
        satisfied = SpecifierSet(asked["requires"]).contains(asked["version"], prereleases=True)
    except InvalidSpecifier as error:
        raise SourceFetchError(f"InferenceX's requires-python {asked['requires']!r} does not read: {error}") from error
    if not satisfied:
        raise SourceFetchError(
            f"InferenceX requires Python {asked['requires']}; the pinned interpreter {python} is Python "
            f"{asked['version']}"
        )


def inferencex_python(project: Path, into: Path) -> Path:
    """The interpreter of a new virtual environment in the new directory ``into``, on the pinned CPython build, holding
    the ``recipes`` dependencies of the InferenceX project at ``project``.

    Raises:
        SourceFetchError: The build cannot be had, it does not satisfy the project's ``requires-python``, or the
            environment cannot be built.
    """
    into.mkdir()
    try:
        base = pinned_python(into / "python")
    except (ClientEnvError, OSError) as error:
        raise SourceFetchError(f"the pinned Python build cannot be had: {error}") from error
    _require_python(base, project)
    environment = into / "venv"
    _run((str(base), "-I", "-m", "venv", str(environment)))
    python = environment / "bin" / "python"
    _run(
        (
            str(python),
            "-I",
            "-m",
            "pip",
            "install",
            _QUIET,
            "--disable-pip-version-check",
            f"{project}[{_RECIPES_EXTRA}]",
        )
    )
    return python


@dataclass(frozen=True)
class InferencexAnswers:
    """What InferenceX's own code answered: a selection or a refusal per point key, a length per golden request."""

    selections: Mapping[str, InferencexSelection] = field(default_factory=dict)
    rejections: Mapping[str, str] = field(default_factory=dict)
    golden_lengths: Mapping[str, GoldenLength] = field(default_factory=dict)


def inferencex_answers(
    python: Path, project: Path, points: Sequence[InferencexPoint], requests: Sequence[GoldenRequest]
) -> InferencexAnswers:
    """Run InferenceX's selection, acceptance and golden-length code at ``project`` with ``python`` on ``points``
    and ``requests``, and record its answers."""
    asked = {
        "project": str(project),
        "points": [{"key": point.key, "recipe": point.recipe, "environment": point.environment()} for point in points],
        "golden": [{"key": request.key, "request": request.to_document()} for request in requests],
    }
    output = _run((str(python), "-I", "-c", _INFERENCEX_CHILD), cwd=project, stdin=json.dumps(asked))
    try:
        answers = json.loads(output)
    except ValueError as error:
        raise SourceFetchError(f"InferenceX answered with no JSON: {output[-_OUTPUT_TAIL:]!r}") from error
    return InferencexAnswers(
        selections={key: InferencexSelection.from_document(value) for key, value in answers["selections"].items()},
        rejections=dict(answers["rejections"]),
        golden_lengths={key: GoldenLength.from_document(value) for key, value in answers["golden_lengths"].items()},
    )


def _inferencex(into: Path) -> tuple[Checkout, InferencexSnapshot]:
    checkout = checkout_main(INFERENCEX_REPO, (PROJECT,), into)
    _git("submodule", "update", _QUIET, "--init", "--depth=1", "--", SRT_SLURM, cwd=into)
    bundle = checkout.bundle((MASTER, RUNNERS, *checkout.paths(GOLDEN_DIR, ".yaml", nested=False)))
    return checkout, InferencexSnapshot(bundle, _git("rev-parse", f"HEAD:{SRT_SLURM}", cwd=into))


def _sglang_cookbook(into: Path) -> RepoBundle:
    checkout = checkout_main(SGLANG_REPO, (CONFIG_DIR, AGENTX_DIR, COOKBOOK_DIR), into)
    return checkout.bundle(
        (
            *checkout.paths(CONFIG_DIR, ".jsx"),
            *checkout.paths(AGENTX_DIR, ".jsx"),
            *checkout.paths(COOKBOOK_DIR, ".mdx"),
        )
    )


def _atom_catalog(into: Path) -> RepoBundle:
    checkout = checkout_main(ATOM_REPO, (str(Path(CATALOG).parent), RECIPES_DIR), into)
    return checkout.bundle((CATALOG, *checkout.paths(RECIPES_DIR, ".md", nested=False)))


def _vllm_recipes(query: LaunchQuery) -> VllmRecipesSnapshot:
    snapshot = VllmRecipesSnapshot({})
    while urls := vllm_recipes_requests(query, snapshot):
        snapshot = VllmRecipesSnapshot({**snapshot.responses, **{url: fetch_response(url) for url in urls}})
    return snapshot


def _with_cookbook(snapshot: SourceSnapshot, query: LaunchQuery, scratch: Path) -> SourceSnapshot:
    """``snapshot`` carrying the cookbook of the query's framework."""
    if query.framework == "sglang":
        return dataclasses.replace(snapshot, sglang_cookbook=_sglang_cookbook(scratch / "sglang"))
    if query.framework == "vllm":
        return dataclasses.replace(snapshot, vllm_recipes=_vllm_recipes(query))
    if query.framework == "atom":
        return dataclasses.replace(snapshot, atom_catalog=_atom_catalog(scratch / "atom"))
    return snapshot


def _base_model(model_id: str) -> str | None:
    """The model's HF ``cardData.base_model`` when it names exactly one, one level up and no further."""
    response = fetch_response(f"{HF_MODELS_API}/{quote(model_id, safe='/')}")
    if response.status in _NO_MODEL_CARD:
        return None
    if response.status != _OK:
        raise SourceFetchError(f"{response.url} answered {response.status}")
    named = (response.json().get("cardData") or {}).get("base_model")
    if isinstance(named, list) and len(named) == 1:
        named = named[0]
    return named if isinstance(named, str) and named else None


def _version_key(version: str) -> tuple[tuple[int, ...], int]:
    release, _, post = version.partition(".post")
    return tuple(map(int, release.split("."))), int(post) if post else -1


def latest_release(repo: str) -> str | None:
    """The tag of ``repo``'s newest release: the highest tag naming a version, release candidates and nightlies
    passed over; ``None`` when no tag names one."""
    refs = _git("ls-remote", "--tags", "--refs", repo).splitlines()
    versions = {
        tag: version for tag in (ref.rpartition("refs/tags/")[2] for ref in refs) if (version := release_version(tag))
    }
    return max(versions, key=lambda tag: _version_key(versions[tag]), default=None)


def _latest_releases(framework: str) -> dict[str, str]:
    repo = _FRAMEWORK_REPOS.get(framework)
    release = None if repo is None else latest_release(repo)
    return {} if release is None else {framework: release}


def _cookbook_agentx_requests(
    query: LaunchQuery, snapshot: SourceSnapshot, inferencex: InferencexSnapshot
) -> tuple[GoldenRequest, ...]:
    """What ``golden_length`` is asked so resolving can pair whichever cookbook AgentX launch it takes."""
    if snapshot.sglang_cookbook is not None:
        found = sglang_cookbook_launches(query, snapshot.sglang_cookbook)
    elif snapshot.vllm_recipes is not None:
        found = vllm_recipes_launches(query, snapshot.vllm_recipes)
    else:
        return ()
    if isinstance(found, SourceMiss):
        return ()
    return golden_requests(query, found, inferencex.bundle, snapshot.base_model)


def _with_inferencex_answers(
    snapshot: SourceSnapshot, query: LaunchQuery, checkout: Checkout, inferencex: InferencexSnapshot, scratch: Path
) -> SourceSnapshot:
    points = served_points(query, inferencex.bundle)
    requests = _cookbook_agentx_requests(query, snapshot, inferencex) if query.workload is Workload.AGENTX else ()
    if not points and not requests:
        return snapshot
    project = checkout.root / PROJECT
    python = inferencex_python(project, scratch / "resolver")
    answers = inferencex_answers(python, project, points, requests)
    answered = dataclasses.replace(inferencex, selections=answers.selections, rejections=answers.rejections)
    return dataclasses.replace(snapshot, inferencex=answered, golden_lengths=answers.golden_lengths)


_Read = Callable[[SourceSnapshot], SourceSnapshot]


def _recipe_reads(
    query: LaunchQuery, checkout: Checkout, inferencex: InferencexSnapshot, scratch: Path
) -> tuple[tuple[str, _Read], ...]:
    """What is read after InferenceX's checkout, in order, each with the name of the source it reads."""
    reads: list[tuple[str, _Read]] = []
    if query.workload is Workload.AGENTX:
        reads.append(
            (
                "the Hugging Face model card",
                lambda snapshot: dataclasses.replace(snapshot, base_model=_base_model(query.model_id)),
            )
        )
    reads += [
        (
            f"the {query.framework} releases",
            lambda snapshot: dataclasses.replace(snapshot, latest_releases=_latest_releases(query.framework)),
        ),
        (f"the {query.framework} cookbook", lambda snapshot: _with_cookbook(snapshot, query, scratch)),
        (
            "InferenceX's own code",
            lambda snapshot: _with_inferencex_answers(snapshot, query, checkout, inferencex, scratch),
        ),
    ]
    return tuple(reads)


def _with_corpus_source(fetched_unix: int, user_file: UserFile) -> SourceSnapshot:
    with tempfile.TemporaryDirectory(prefix="inference-suite-sources-") as scratch_dir:
        try:
            _, inferencex = _inferencex(Path(scratch_dir) / "inferencex")
        except SourceFetchError as error:
            unread = UnreadSource(_INFERENCEX_SOURCE, str(error))
            return SourceSnapshot(None, fetched_unix, user_file=user_file, unread=unread)
    return SourceSnapshot(inferencex, fetched_unix, user_file=user_file)


def fetch_sources(query: LaunchQuery, *, user_file: UserFile | None = None) -> SourceSnapshot:
    """Read every source ``query`` resolves from, at the head of its ``main`` and as of now.

    With the operator's ``user_file``, which is the base, only InferenceX is read, and only when the file is a shell
    launch of an AgentX query: InferenceX names the corpus the replay runs. A source that cannot be read is named as
    the snapshot's ``unread``, and nothing after it is read.
    """
    fetched_unix = int(time.time())
    if user_file is not None:
        if query.workload is Workload.AGENTX and isinstance(user_file.launch, ShellLaunch):
            return _with_corpus_source(fetched_unix, user_file)
        return SourceSnapshot(None, fetched_unix, user_file=user_file)
    with tempfile.TemporaryDirectory(prefix="inference-suite-sources-") as scratch_dir:
        scratch = Path(scratch_dir)
        try:
            checkout, inferencex = _inferencex(scratch / "inferencex")
        except SourceFetchError as error:
            return SourceSnapshot(None, fetched_unix, unread=UnreadSource(_INFERENCEX_SOURCE, str(error)))
        snapshot = SourceSnapshot(inferencex, fetched_unix)
        for source, read in _recipe_reads(query, checkout, inferencex, scratch):
            try:
                snapshot = read(snapshot)
            except SourceFetchError as error:
                return dataclasses.replace(snapshot, unread=UnreadSource(source, str(error)))
        return snapshot


__all__ = [
    "HF_MODELS_API",
    "MAIN",
    "SRT_SLURM",
    "VLLM_REPO",
    "Checkout",
    "InferencexAnswers",
    "SourceFetchError",
    "checkout_main",
    "fetch_response",
    "fetch_sources",
    "inferencex_answers",
    "inferencex_python",
    "latest_release",
]
