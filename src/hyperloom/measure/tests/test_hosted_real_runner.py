# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The hosted backend against the shipped runner flow in a local sandbox; only the GPU measurement is stubbed."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from hyperloom.measure.hosted_log import STATE_FINISHED
from hyperloom.measure.local_templates import image_env
from hyperloom.measure.returncodes import HOSTED_SPEC_REJECTED_RC
from hyperloom.measure.tests.hosted_fixtures import Rig, make_call, make_rig
from hyperloom.sandbox.local import LocalTemplate

REAL_RUNNER = (sys.executable, "-m", "hyperloom.measure.tests.real_runner_entry")
#: The shipped command line, with the sandbox's own directories: the local sandbox maps nothing, it only sets the root.
RUNNER_CLI = (
    "/bin/sh",
    "-c",
    'exec "$0" -m hyperloom.measure.runner run --in "$HYPERLOOM_SANDBOX_ROOT/in" --out "$HYPERLOOM_SANDBOX_ROOT/out"'
    ' --scratch "$HYPERLOOM_SANDBOX_ROOT/work/scratch" --spec-wait-sec 60',
    sys.executable,
)
MARKER = b"MARKED = True\n"


def _git_repo(path: Path) -> str:
    """A tiny repository with the file the runner checks for in an InferenceX checkout; returns its commit."""
    (path / "benchmarks").mkdir(parents=True)
    (path / "benchmarks" / "benchmark_lib.sh").write_text("# stand-in\n")
    env = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@t"}
    for args in (["init", "-q"], ["add", "."], ["commit", "-q", "-m", "x"]):
        subprocess.run(["git", *args], cwd=path, check=True, env={**env, "PATH": "/usr/bin:/bin"})
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=path, check=True, capture_output=True, text=True)
    return done.stdout.strip()


class _Environment:
    """What a compiled environment hands the sheet: one root that adds a file to the installed ``yaml`` package."""

    def __init__(self, blob_sha: str, blob_root: Path) -> None:
        self.attachments = ()
        self._doc = {
            "roots": [
                {
                    "id": "yaml",
                    "kind": "package",
                    "import_name": "yaml",
                    "dist": {"name": "PyYAML", "version": "0"},
                    "entries": [
                        {"path": "_hl_marker.py", "op": "upsert", "blob": f"sha256:{blob_sha}", "mode": "0644"}
                    ],
                }
            ],
            "runtime": None,
            "blob_root": str(blob_root),
        }

    def to_document(self) -> dict:
        return self._doc


def _rigs(tmp_path: Path, command: tuple[str, ...]):
    """A factory of rigs over the runner ``command``: ``envs`` are added to the template's, ``config`` to the backend's."""
    repo = tmp_path / "inferencex"
    commit = _git_repo(repo)
    built: list[Rig] = []

    def make(*, envs: Mapping[str, str] | None = None, config: Mapping[str, Any] | None = None, **spec) -> Rig:
        template = LocalTemplate(
            "runner-1",
            "measurement",
            command=command,
            gpus=1,
            sealed=True,
            signing=True,
            envs={**image_env(), "INFERENCEX_REPO": str(repo), **(envs or {})},
        )
        r = make_rig(
            tmp_path,
            templates=[template],
            spec={"inferencex_commit": commit, "benchmark_backend": "bypass", **spec},
            poll_interval_sec=0.1,
            **(config or {}),
        )
        built.append(r)
        return r

    yield make
    for r in built:
        r.backend.close()
        r.provider.close()


@pytest.fixture
def real(tmp_path: Path):
    """The shipped runner flow with the GPU measurement stubbed."""
    yield from _rigs(tmp_path, REAL_RUNNER)


@pytest.fixture
def real_cli(tmp_path: Path):
    """The shipped command line, exactly as a local runner template starts it."""
    yield from _rigs(tmp_path, RUNNER_CLI)


def _point(tmp: Path, **envs: str) -> Path:
    lines = [
        "benchmark:",
        "  model: /nonexistent/model",
        "  framework: sglang",
        "  envs:",
        "    TP: '1'",
        "    RUN_EVAL: 'false'",
    ]
    lines += [f"    {key}: '{value}'" for key, value in envs.items()]
    path = tmp / "real.yaml"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_the_shipped_runner_flow_produces_a_result_the_backend_accepts(real, tmp_path) -> None:
    r = real()
    call = make_call(tmp_path, point=_point(tmp_path), timeout_sec=120)
    outcome = r.backend.run_test(call)
    assert outcome.returncode == 0, outcome.stderr
    assert outcome.failure_class is None
    record = r.backend.journal.latest("req-1")
    assert record.state == STATE_FINISHED and record.extra["verified"] is True
    seen = json.loads((call.output_dir / "point_seen.json").read_text())
    envs = seen["benchmark"]["envs"]
    assert envs["ROCR_VISIBLE_DEVICES"] == "0", "the rewritten point carries this sandbox's cards"
    inferencex = seen["benchmark"]["inferencex_path"]
    assert inferencex.endswith("InferenceX-" + r.config.spec.inferencex_commit)
    assert inferencex.startswith(str(tmp_path / "sandboxes")), "the runner's scratch is the sandbox's own directory"
    assert [x.phase for x in outcome.rounds] == ["measure"]
    assert outcome.rounds[0].output_dir == call.output_dir / "benchmark_stub_20261014_000000"
    assert (outcome.sandbox.gpu_ids, outcome.sandbox.node_name) == ((0,), "node-a")


def test_a_rebuilt_framework_is_what_the_server_would_import(real, tmp_path) -> None:
    r = real()
    blob = r.config.spec.blobs.put(MARKER)
    environment = _Environment(blob, r.config.spec.blobs.root)
    call = make_call(
        tmp_path,
        point=_point(tmp_path, HYPERLOOM_FRAMEWORK_PYTHON=sys.executable),
        timeout_sec=120,
        environment=environment,
    )
    outcome = r.backend.run_test(call)
    assert outcome.returncode == 0, outcome.stderr
    assert (call.output_dir / "import_probe.txt").read_text() == "True", "the overlay file is in the imported package"
    pythonpath = json.loads((call.output_dir / "point_seen.json").read_text())["benchmark"]["envs"]["PYTHONPATH"]
    assert pythonpath.split(":")[0].endswith("farm/site"), "the farm is first on the server's import path"


@pytest.mark.parametrize("fixture", ["real", "real_cli"])
def test_a_sheet_the_real_runner_refuses_is_a_refused_sheet_not_a_failed_candidate(fixture, request, tmp_path) -> None:
    r = request.getfixturevalue(fixture)(model_hashes={"config.json": "00" * 32})
    call = make_call(tmp_path, point=_point(tmp_path), timeout_sec=120)
    outcome = r.backend.run_test(call)
    assert outcome.returncode == HOSTED_SPEC_REJECTED_RC
    assert outcome.failure_class is None
    assert "spec_rejected" in outcome.stderr and "config.json" in outcome.stderr
    assert r.provider.find({}, role="measurement") == [], "the sandbox is closed after the refusal"
