# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Contract of the AgentX client environment: the pinned build and the lock's wheels by hash, installed relocatably,
and its normalised identity read back from what was installed."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from inference_suite import client_env
from inference_suite.client_env import (
    BUILD_FILE,
    CLIENT_PYTHON,
    INPUT_ARCHIVE,
    INPUT_WHEELS,
    ClientEnvError,
    client_identity,
    fetch_inputs,
    install,
    read_lock,
    staged_files,
)
from inference_suite.pins import AGENTX_PYTHON_BUILD, INFERENCEX_CLIENT_COMMIT, LM_EVAL_VERSION, LM_EVAL_WHEEL_SHA256
from inference_suite.tests.client_env_fakes import VERSION, WHEELS, FakeClient

_REPO = Path(__file__).resolve().parents[3]


@pytest.fixture
def fake(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeClient:
    return FakeClient(tmp_path, monkeypatch)


def _inputs(tmp_path: Path) -> Path:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    fetch_inputs(inputs)
    return inputs


def test_fetch_downloads_the_pinned_build_and_the_lock_wheels_by_hash(fake, tmp_path) -> None:
    inputs = _inputs(tmp_path)
    assert fake.downloads == [fake.build.url]
    assert hashlib.sha256((inputs / INPUT_ARCHIVE).read_bytes()).hexdigest() == fake.build.sha256
    assert sorted(path.name.split("-")[0] for path in (inputs / INPUT_WHEELS).iterdir()) == sorted(
        name.replace("-", "_") for name in WHEELS
    )


def test_fetch_refuses_a_build_whose_sha256_is_not_the_pin(fake, tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(fake, "archive", fake.archive + b"tampered")
    with pytest.raises(ClientEnvError, match="pinned at"):
        _inputs(tmp_path)


def test_fetch_refuses_a_wheel_off_the_lock_hash(fake, tmp_path) -> None:
    next(fake.index.glob("aiohttp-*")).write_bytes(b"another aiohttp")
    with pytest.raises(ClientEnvError, match="downloading the wheels"):
        _inputs(tmp_path)


def test_install_fetches_nothing_and_the_environment_runs_after_its_directory_is_renamed(fake, tmp_path) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / ".staging-x"
    root.mkdir()
    install(inputs, root)
    assert fake.downloads == [fake.build.url]
    moved = tmp_path / "published"
    os.rename(root, moved)
    identity = client_identity(moved, INFERENCEX_CLIENT_COMMIT)
    assert {package.name: package.wheel_sha256 for package in identity.packages} == fake.lock_shas()
    assert (identity.interpreter_version, identity.interpreter_sha256) == (VERSION, fake.build.sha256)
    assert "pip" not in {package.name for package in identity.packages}
    assert (moved / CLIENT_PYTHON).is_file()
    assert {path.relative_to(moved).as_posix() for path in staged_files(moved)} >= {CLIENT_PYTHON, BUILD_FILE}


def test_install_refuses_inputs_without_a_wheel_the_lock_pins(fake, tmp_path) -> None:
    inputs = _inputs(tmp_path)
    next((inputs / INPUT_WHEELS).glob("datasets-*")).unlink()
    root = tmp_path / "root"
    root.mkdir()
    with pytest.raises(ClientEnvError, match="installing the AgentX client"):
        install(inputs, root)


def test_a_build_of_another_python_version_is_refused(tmp_path, monkeypatch) -> None:
    FakeClient(tmp_path, monkeypatch, version="3.12.14")
    with pytest.raises(ClientEnvError, match="3.12.14"):
        _inputs(tmp_path)


def test_identity_refuses_an_environment_that_is_not_its_lock(fake, tmp_path) -> None:
    inputs = _inputs(tmp_path)
    root = tmp_path / "root"
    root.mkdir()
    install(inputs, root)
    site = next((root / "agentx-python").glob("lib/python3.12/site-packages"))
    (site / "extra_pkg-1.0.dist-info").mkdir()
    with pytest.raises(ClientEnvError, match="extra-pkg"):
        client_identity(root, INFERENCEX_CLIENT_COMMIT)


def test_a_lock_line_that_is_not_a_hashed_pin_or_a_pin_twice_is_refused() -> None:
    sha = "a" * 64
    with pytest.raises(ClientEnvError, match="not name"):
        read_lock("aiohttp>=3\n")
    with pytest.raises(ClientEnvError, match="twice"):
        read_lock(f"aiohttp==1 --hash=sha256:{sha}\nAIOHTTP==1 --hash=sha256:{sha}\n")


def test_lock_regeneration_writes_each_resolved_wheel_by_hash_under_a_header_that_says_how(fake, tmp_path, capsys):
    requirements = tmp_path / "infx" / "bench" / "agentic" / "requirements.txt"
    requirements.parent.mkdir(parents=True)
    requirements.write_text("agentx-harness==1.0.13\n")
    assert client_env.main(["lock", "--requirements", str(requirements)]) == 0
    text = capsys.readouterr().out
    assert {pin.name: pin.sha256 for pin in read_lock(text)} == fake.lock_shas()
    assert "python -m inference_suite.client_env lock --requirements" in text


def test_the_shipped_lock_pins_the_harness_with_one_wheel_hash_per_distribution() -> None:
    text = client_env.AGENTX_LOCK.read_text()
    pins = {pin.name: pin.version for pin in read_lock(text)}
    assert pins["agentx-harness"] == "1.0.13"
    assert "datasets" in pins
    assert AGENTX_PYTHON_BUILD.sha256 in text and INFERENCEX_CLIENT_COMMIT in text


@pytest.mark.parametrize("lock", ["runner.lock", "runner-cp314.lock"])
def test_the_lm_eval_wheel_pin_is_a_hash_the_sealed_runner_lock_installs(lock) -> None:
    text = (_REPO / "requirements" / lock).read_text()
    block = text[text.index(f"lm-eval=={LM_EVAL_VERSION}") :].split("# via")[0]
    assert f"--hash=sha256:{LM_EVAL_WHEEL_SHA256}" in block


def test_the_build_is_downloaded_with_a_user_agent_its_mirror_serves(tmp_path, monkeypatch) -> None:
    sent = []

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc) -> None:
            return None

        def read(self, size: int = -1) -> bytes:
            return b""

    def urlopen(request, **_):
        sent.append(request)
        return _Response()

    monkeypatch.setattr(client_env.urllib.request, "urlopen", urlopen)
    client_env._download(AGENTX_PYTHON_BUILD.url, tmp_path / "build.tar.gz")

    (request,) = sent
    assert request.full_url == AGENTX_PYTHON_BUILD.url
    assert not request.get_header("User-agent", "Python-urllib").startswith("Python-urllib")
