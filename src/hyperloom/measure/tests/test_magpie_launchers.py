# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Magpie's launcher is put in the InferenceX checkout the way Magpie puts it, before the first round reads it."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
import yaml

from hyperloom.measure.magpie_launchers import install_launchers

LAUNCHER = "sglang_mi355x.sh"


@pytest.fixture
def magpie(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A Magpie checkout with its launchers, named by ``MAGPIE_PATH`` as the runner names the one it pinned."""
    launchers = tmp_path / "magpie" / "Magpie" / "scripts" / "benchmark"
    launchers.mkdir(parents=True)
    (launchers / LAUNCHER).write_text("export SGLANG_USE_AITER=1\n")
    monkeypatch.setenv("MAGPIE_PATH", str(tmp_path / "magpie"))
    monkeypatch.delenv("INFERENCEX_PATH", raising=False)
    return launchers


def _checkout(tmp_path: Path, name: str = "ix", *, benchmarks: bool = True) -> Path:
    checkout = tmp_path / name
    (checkout / "benchmarks" if benchmarks else checkout).mkdir(parents=True)
    return checkout


def _point(tmp_path: Path, checkout: Path | None, **bench: object) -> Path:
    path = tmp_path / "point.yaml"
    named = {"inferencex_path": str(checkout)} if checkout is not None else {}
    path.write_text(
        yaml.safe_dump({"benchmark": {"framework": "sglang", "benchmark_script": LAUNCHER, **named, **bench}})
    )
    return path


def test_the_launcher_a_point_names_is_copied_into_its_checkout_executable(tmp_path: Path, magpie: Path) -> None:
    checkout = _checkout(tmp_path)

    install_launchers([_point(tmp_path, checkout)])

    placed = checkout / "benchmarks" / LAUNCHER
    assert placed.read_text() == "export SGLANG_USE_AITER=1\n"
    assert stat.S_IMODE(placed.stat().st_mode) == 0o700
    assert not list((checkout / "benchmarks").glob(".*.tmp")), "no half-written copy is left"


def test_a_launcher_the_checkout_has_that_differs_is_replaced_and_one_that_is_the_same_is_left_alone(
    tmp_path: Path, magpie: Path
) -> None:
    checkout = _checkout(tmp_path)
    placed = checkout / "benchmarks" / LAUNCHER
    placed.write_text("export OLD=1\n")

    install_launchers([_point(tmp_path, checkout)])
    first = placed.stat().st_ino
    install_launchers([_point(tmp_path, checkout)])

    assert placed.read_text() == "export SGLANG_USE_AITER=1\n", "Magpie's copy is the one that runs"
    assert placed.stat().st_ino == first, "an identical script is not replaced under a bash that may hold it open"


def test_the_checkout_in_the_process_environment_gets_it_when_the_point_names_none(
    tmp_path: Path, magpie: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _checkout(tmp_path)
    monkeypatch.setenv("INFERENCEX_PATH", str(checkout))

    install_launchers([_point(tmp_path, None)])

    assert (checkout / "benchmarks" / LAUNCHER).is_file()


@pytest.mark.parametrize(
    "case", ["no launcher named", "Magpie ships none", "no benchmarks directory", "an AgentX point", "no Magpie"]
)
def test_what_cannot_or_need_not_be_installed_is_left_as_it_is(
    tmp_path: Path, magpie: Path, monkeypatch: pytest.MonkeyPatch, case: str
) -> None:
    checkout = _checkout(tmp_path, benchmarks=case != "no benchmarks directory")
    bench: dict[str, object] = {}
    if case == "no launcher named":
        bench["benchmark_script"] = ""
    if case == "Magpie ships none":
        (magpie / LAUNCHER).unlink()
    if case == "an AgentX point":
        bench["agentx"] = {"enabled": True}
    if case == "no Magpie":
        monkeypatch.delenv("MAGPIE_PATH")
        monkeypatch.setattr("importlib.util.find_spec", lambda name: None)

    install_launchers([_point(tmp_path, checkout, **bench)])

    assert sorted(path.name for path in checkout.rglob("*") if path.is_file()) == []


def test_a_copy_that_cannot_be_made_is_logged_and_does_not_stop_the_test(
    tmp_path: Path, magpie: Path, caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkout = _checkout(tmp_path)

    def full_disk(*args: object, **kwargs: object) -> None:
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(os, "replace", full_disk)

    install_launchers([_point(tmp_path, checkout)])

    assert not (checkout / "benchmarks" / LAUNCHER).exists()
    assert not list((checkout / "benchmarks").glob(".*.tmp")), "the temporary copy is removed"
    assert "No space left" in caplog.text
