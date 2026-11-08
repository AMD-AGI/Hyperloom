# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Taking a source snapshot: repositories read at ``main`` with their commit, responses kept with their status and
body, and InferenceX's own answers recorded as data.

Checkouts and responses run against local repositories and a local HTTP server. The recording protocol runs a real
isolated interpreter over a stand-in ``infx``; the ``upstream_sources`` test runs InferenceX itself at the commit
of the main InferenceX fixture, over the network, and holds the fixture to what InferenceX answers there.
"""

from __future__ import annotations

import gzip
import hashlib
import http.server
import json
import socket
import subprocess
import sys
import textwrap
import threading
from collections.abc import Iterator
from functools import cache
from pathlib import Path

import pytest

from inference_suite.contracts import (
    BaseLaunch,
    ClientLaunch,
    ImageFacts,
    LaunchModel,
    LaunchPoint,
    LaunchQuery,
    LaunchSource,
    SourceKind,
    Topology,
    VersionBasis,
    Workload,
)
from inference_suite.recipe import sources
from inference_suite.recipe.golden_curve import GoldenLength, GoldenRequest
from inference_suite.recipe.inferencex import INFERENCEX_REPO, PROJECT, InferencexPoint, Scenario, inferencex_points
from inference_suite.recipe.model import InferencexSelection, InferencexSnapshot
from inference_suite.recipe.resolve import UnreadSource
from inference_suite.recipe.user_file import UserFile, parse_user_file
from inference_suite.recipe.sources import (
    MAIN,
    SRT_SLURM,
    Checkout,
    SourceFetchError,
    checkout_main,
    fetch_response,
    fetch_sources,
    inferencex_answers,
    inferencex_python,
    latest_release,
)

_FIXTURE = Path(__file__).parent / "fixtures" / "inferencex" / "ce35bc935.json.gz"
_GIT = ("git", "-c", "user.name=suite", "-c", "user.email=suite@example.com", "-c", "init.defaultBranch=trunk")


@cache
def _main_fixture() -> InferencexSnapshot:
    return InferencexSnapshot.from_document(json.loads(gzip.decompress(_FIXTURE.read_bytes()))["snapshot"])


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run((*_GIT, *args), cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def _commit(repo: Path, files: dict[str, str], message: str) -> str:
    for path, text in files.items():
        (repo / path).parent.mkdir(parents=True, exist_ok=True)
        (repo / path).write_text(text, encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", message)
    return _git(repo, "rev-parse", "HEAD")


@pytest.fixture
def origin(tmp_path: Path) -> Path:
    """A repository whose default branch is not ``main``, with a ``main`` that differs from it."""
    repo = tmp_path / "origin"
    repo.mkdir()
    _git(repo, "init", "-q")
    _commit(repo, {"docs/a.txt": "trunk\n"}, "trunk")
    _git(repo, "checkout", "-q", "-b", MAIN)
    _commit(repo, {"docs/a.txt": "main\r\n", "docs/deep/b.txt": "b\n", "other/c.txt": "c\n"}, "main")
    _git(repo, "checkout", "-q", "trunk")
    _commit(repo, {"docs/a.txt": "trunk moved on\n"}, "trunk again")
    return repo


def test_checkout_reads_main_not_the_default_branch_and_records_its_commit(origin: Path, tmp_path: Path) -> None:
    main_commit = _git(origin, "rev-parse", MAIN)

    checkout = checkout_main(f"file://{origin}", ("docs",), tmp_path / "clone")
    bundle = checkout.bundle(checkout.paths("docs", ".txt"))

    assert checkout.commit == main_commit
    assert (bundle.repo, bundle.commit) == (f"file://{origin}", main_commit)
    assert bundle.files == {"docs/a.txt": "main\r\n", "docs/deep/b.txt": "b\n"}
    assert bundle.sha256("docs/a.txt") == hashlib.sha256(b"main\r\n").hexdigest()
    assert checkout.paths("docs", ".txt", nested=False) == ["docs/a.txt"]
    assert not (tmp_path / "clone" / "other").exists()


def test_a_repository_with_no_main_is_refused(origin: Path, tmp_path: Path) -> None:
    _git(origin, "branch", "-q", "-m", MAIN, "merged")

    with pytest.raises(SourceFetchError, match="exited"):
        checkout_main(f"file://{origin}", ("docs",), tmp_path / "clone")


def test_the_latest_release_is_the_highest_version_tag_and_no_candidate_or_nightly(origin: Path) -> None:
    for tag in ("v0.5.9", "v0.5.10", "v0.5.10.post1", "v0.5.11rc1", "nightly-20261106", "gateway-v9.0.0"):
        _git(origin, "tag", tag)

    assert latest_release(f"file://{origin}") == "v0.5.10.post1"


def test_a_repository_with_no_version_tag_has_no_latest_release(origin: Path) -> None:
    _git(origin, "tag", "v0.6.0rc1")

    assert latest_release(f"file://{origin}") is None


class _Site(http.server.BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        status, body = (200, b'{"recipe": "\xc3\xa9"}') if self.path == "/found.json" else (404, b"no such recipe")
        self.send_response(status)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: object) -> None:
        pass


@pytest.fixture
def site() -> Iterator[str]:
    server = http.server.HTTPServer(("127.0.0.1", 0), _Site)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_a_response_is_recorded_with_its_status_and_the_sha256_of_its_body(site: str) -> None:
    found = fetch_response(f"{site}/found.json")
    missing = fetch_response(f"{site}/missing.json")

    assert (found.url, found.status, found.json()) == (f"{site}/found.json", 200, {"recipe": "é"})
    assert found.provenance().response_sha256 == hashlib.sha256('{"recipe": "é"}'.encode()).hexdigest()
    assert (missing.status, missing.body) == (404, "no such recipe")


def test_an_unreachable_site_is_refused() -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]

    with pytest.raises(SourceFetchError, match="could not be fetched"):
        fetch_response(f"http://127.0.0.1:{port}/recipe.json")


_QUERY = LaunchQuery("deepseek-ai/DeepSeek-R1-0528", "sglang", "mi355x", Workload.FIXED, (64,), isl=8192, osl=1024)
_AGENTX_QUERY = LaunchQuery("deepseek-ai/DeepSeek-R1-0528", "sglang", "mi355x", Workload.AGENTX, (8,))


@pytest.fixture
def read_inferencex(monkeypatch, tmp_path: Path) -> None:
    """InferenceX reads as the main fixture; a test makes a later source unreadable."""
    fixture = _main_fixture()
    checkout = Checkout(INFERENCEX_REPO, fixture.bundle.commit, tmp_path)
    monkeypatch.setattr(sources, "_inferencex", lambda into: (checkout, fixture))


def _unreachable(*args, **kwargs):
    raise SourceFetchError("git ls-remote exited 128: could not resolve host")


def _never(*args, **kwargs):
    raise AssertionError("a source after the unreadable one was read")


def test_a_source_that_cannot_be_read_is_named_and_stops_the_reading(read_inferencex, monkeypatch) -> None:
    monkeypatch.setattr(sources, "_latest_releases", _unreachable)
    monkeypatch.setattr(sources, "_with_cookbook", _never)

    snapshot = fetch_sources(_QUERY)

    assert snapshot.unread == UnreadSource("the sglang releases", "git ls-remote exited 128: could not resolve host")
    assert snapshot.inferencex == _main_fixture()
    assert snapshot.sglang_cookbook is None


def _frozen_agentx_file() -> UserFile:
    query = LaunchQuery("moonshotai/Kimi-K3", "sglang", "mi355x", Workload.AGENTX, (32,))
    base = BaseLaunch(
        query=query,
        source=LaunchSource(kind=SourceKind.USER_FILE, path="kimi.sh", files={"kimi.sh": "c" * 64}),
        point=LaunchPoint(conc=32, exact=True),
        image=ImageFacts(run=None, recipe=None, framework_version=None, version_basis=VersionBasis.NOT_APPLICABLE),
        framework="sglang",
        entrypoint=("sglang", "serve"),
        model=LaunchModel("moonshotai/Kimi-K3"),
        topology=Topology(tp=8),
        args=("--tp", "8"),
        env={},
        client=ClientLaunch(tool="aiperf", commit="d" * 40, env={"WEKA_LOADER_OVERRIDE": "corpus"}),
        resolved_unix=1_700_000_000,
    )
    return parse_user_file("base_launch.json", json.dumps(base.to_document()).encode(), framework="sglang")


@pytest.mark.parametrize("agentx", [False, True], ids=["fixed-shell", "agentx-frozen-base"])
def test_a_user_file_reads_no_source(monkeypatch, agentx) -> None:
    monkeypatch.setattr(sources, "_inferencex", _never)
    monkeypatch.setattr(sources, "_recipe_reads", _never)
    if agentx:
        query, user_file = _AGENTX_QUERY, _frozen_agentx_file()
    else:
        query = _QUERY
        user_file = parse_user_file("launch.sh", b"python3 -m sglang.launch_server --tp-size 8\n", framework="sglang")

    snapshot = fetch_sources(query, user_file=user_file)

    assert (snapshot.user_file, snapshot.inferencex, snapshot.unread) == (user_file, None, None)


def test_an_agentx_shell_user_file_reads_inferencex_alone_for_the_corpus(read_inferencex, monkeypatch) -> None:
    monkeypatch.setattr(sources, "_recipe_reads", _never)
    user_file = parse_user_file("launch.sh", b"python3 -m sglang.launch_server --tp-size 8\n", framework="sglang")

    snapshot = fetch_sources(_AGENTX_QUERY, user_file=user_file)

    assert (snapshot.user_file, snapshot.inferencex, snapshot.unread) == (user_file, _main_fixture(), None)


def test_an_agentx_shell_user_file_names_an_unreadable_inferencex(monkeypatch) -> None:
    def unreachable(into):
        raise SourceFetchError("git clone exited 128: could not resolve host")

    monkeypatch.setattr(sources, "_inferencex", unreachable)
    user_file = parse_user_file("launch.sh", b"python3 -m sglang.launch_server --tp-size 8\n", framework="sglang")

    snapshot = fetch_sources(_AGENTX_QUERY, user_file=user_file)

    assert snapshot.user_file == user_file
    assert snapshot.unread == UnreadSource("InferenceX", "git clone exited 128: could not resolve host")


def test_an_unreadable_inferencex_is_named_and_stops_the_reading(monkeypatch) -> None:
    def unreachable(into):
        raise SourceFetchError("git clone exited 128: could not resolve host")

    monkeypatch.setattr(sources, "_inferencex", unreachable)
    monkeypatch.setattr(sources, "_recipe_reads", _never)

    snapshot = fetch_sources(_QUERY)

    assert snapshot.unread == UnreadSource("InferenceX", "git clone exited 128: could not resolve host")
    assert snapshot.inferencex is None


_STAND_IN = {
    "infx/__init__.py": "",
    "infx/srt_slurm/__init__.py": "",
    "infx/golden_al_distribution/__init__.py": """
        GOLDEN_DIR = None

        def curve_name(prefix, spec, golden_dir, verify):
            return f"{prefix}__{spec['method']}__{verify}"

        def golden_length(prefix, spec, thinking, golden_dir, verify):
            if prefix == "unknown":
                raise ValueError("no golden curve for unknown")
            return 3.25
        """,
    "infx/srt_slurm/single_node.py": """
        def select_recipe(config, environment, *, root):
            print("upstream says hello on stdout")
            if environment["FRAMEWORK"] == "atom":
                raise ValueError("Expected exactly one matching single-node SRT recipe")
            path, _, selector = config.partition(":")
            return f"{path}:{selector or 'base'}", {"roles": {"agg": {"args": {"tp": int(environment["TP"])}}}}
        """,
    "infx/srt_slurm/synthetic_acceptance.py": """
        ENGINES = {"sglang": "sglang", "vllm": "vllm"}

        def acceptance_length(recipe, framework, environment):
            return 2.5 if environment["IS_AGENTIC"] == "1" else None

        def build_overrides(recipe, framework, environment):
            return ["--set", "roles.agg.env.X=1"] if environment["IS_AGENTIC"] == "1" else []

        def generation_role(recipe):
            return recipe["roles"]["agg"]

        def spec_parameters(role, engine):
            return {"method": "mtp"}

        class _Fact:
            value = "standard"

        def verifier(role, engine):
            return _Fact()
        """,
    "infx/srt_slurm/workload.py": """
        from pathlib import Path

        SHARED_BLOCKS = {(False, False): Path("configs/fixed.yaml"), (True, False): Path("configs/agentic.yaml")}
        NATIVE_TELEMETRY_BLOCK = Path("configs/telemetry.yaml")
        """,
}


def _stand_in_project(root: Path, points: list[InferencexPoint]) -> Path:
    files = {path: textwrap.dedent(text) for path, text in _STAND_IN.items()}
    files.update(
        {path: f"{path}\n" for path in ("configs/fixed.yaml", "configs/agentic.yaml", "configs/telemetry.yaml")}
    )
    files.update({point.recipe.partition(":")[0]: f"{point.recipe}\n" for point in points})
    for path, text in files.items():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text(text, encoding="utf-8")
    return root


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_inferencex_answers_are_recorded_as_data_from_an_isolated_interpreter(tmp_path: Path) -> None:
    points = inferencex_points(_main_fixture().bundle)
    fixed = next(point for point in points if (point.framework, point.scenario) == ("sglang", Scenario.FIXED))
    agentic = next(point for point in points if (point.framework, point.scenario) == ("sglang", Scenario.AGENTIC))
    refused = next(point for point in points if point.framework == "atom")
    project = _stand_in_project(tmp_path / "project", [fixed, agentic, refused])
    asked = GoldenRequest("glm5.3", {"method": "mtp", "num_speculative_tokens": 3}, "thinking_on", "standard")
    unknown = GoldenRequest("unknown", {"method": "mtp", "num_speculative_tokens": 3}, "thinking_on", None)

    answers = inferencex_answers(Path(sys.executable), project, [fixed, agentic, refused], [asked, unknown])

    recipe_path = agentic.recipe.partition(":")[0]
    assert answers.selections[agentic.key] == InferencexSelection(
        variant=f"{recipe_path}:{agentic.recipe.partition(':')[2] or 'base'}",
        recipe={"roles": {"agg": {"args": {"tp": agentic.tp}}}},
        files={
            recipe_path: _sha256(f"{agentic.recipe}\n"),
            "configs/agentic.yaml": _sha256("configs/agentic.yaml\n"),
            "configs/telemetry.yaml": _sha256("configs/telemetry.yaml\n"),
        },
        acceptance_length=2.5,
        golden_overrides=("--set", "roles.agg.env.X=1"),
        golden_curve=f"{agentic.model_prefix}__mtp__standard",
    )
    assert answers.selections[fixed.key].acceptance_length is None
    assert "configs/fixed.yaml" in answers.selections[fixed.key].files
    assert answers.rejections == {refused.key: "Expected exactly one matching single-node SRT recipe"}
    assert answers.golden_lengths == {
        asked.key: GoldenLength(3.25, curve="glm5.3__mtp__standard"),
        unknown.key: GoldenLength(None, error="no golden curve for unknown"),
    }


def test_upstream_code_that_fails_rather_than_answers_is_refused(tmp_path: Path) -> None:
    point = inferencex_points(_main_fixture().bundle)[0]
    project = _stand_in_project(tmp_path / "project", [point])
    (project / "infx/srt_slurm/workload.py").write_text("raise ImportError('srtctl is missing')\n", encoding="utf-8")

    with pytest.raises(SourceFetchError, match="srtctl is missing"):
        inferencex_answers(Path(sys.executable), project, [point], [])


def test_inferencex_needing_a_newer_python_than_the_pinned_build_is_refused_before_installing(
    tmp_path: Path, monkeypatch
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text('[project]\nname = "infx"\nrequires-python = ">=3.99"\n')
    monkeypatch.setattr(sources, "pinned_python", lambda into: Path(sys.executable))
    version = ".".join(map(str, sys.version_info[:3]))

    with pytest.raises(SourceFetchError, match=rf"requires Python >=3\.99.*is Python {version}"):
        inferencex_python(project, tmp_path / "resolver")

    assert not (tmp_path / "resolver" / "venv").exists()


def _checkout_commit(repo: str, commit: str, into: Path) -> None:
    into.mkdir()
    _git(into, "init", "-q")
    _git(into, "remote", "add", "origin", repo)
    _git(into, "sparse-checkout", "set", PROJECT)
    _git(into, "fetch", "-q", "--depth=1", "--filter=blob:none", "origin", commit)
    _git(into, "checkout", "-q", "FETCH_HEAD")
    _git(into, "submodule", "update", "-q", "--init", "--depth=1", "--", SRT_SLURM)


@pytest.mark.upstream_sources
def test_the_main_inferencex_fixture_is_what_inferencex_answers_at_its_commit(tmp_path: Path) -> None:
    if subprocess.run(("git", "ls-remote", "--exit-code", INFERENCEX_REPO, "HEAD"), capture_output=True).returncode:
        pytest.skip(f"{INFERENCEX_REPO} is not reachable")
    recorded = _main_fixture()
    _checkout_commit(INFERENCEX_REPO, recorded.bundle.commit, tmp_path / "inferencex")
    project = tmp_path / "inferencex" / PROJECT
    points = inferencex_points(recorded.bundle)
    drafted = next(
        point
        for point in points
        if point.framework == "vllm" and recorded.selections[point.key].acceptance_length is not None
    )
    selection = recorded.selections[drafted.key]
    spec = json.loads(selection.recipe["roles"]["agg"]["args"]["speculative-config"])
    asked = GoldenRequest(drafted.model_prefix, spec, "thinking_on", "block")

    answers = inferencex_answers(inferencex_python(project, tmp_path / "resolver"), project, points, [asked])

    assert _git(tmp_path / "inferencex", "rev-parse", f"HEAD:{SRT_SLURM}") == recorded.srt_slurm_commit
    assert answers.rejections == recorded.rejections
    assert answers.selections == recorded.selections
    assert answers.golden_lengths == {
        asked.key: GoldenLength(selection.acceptance_length, curve=selection.golden_curve)
    }
