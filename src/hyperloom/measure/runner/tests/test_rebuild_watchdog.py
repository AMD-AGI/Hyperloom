# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""An sglang point whose tuned tables make a prebuilt aiter module stale gets a watchdog that waits for the rebuild.

The image is the aiter package of ``test_table_modules`` with its module config: a table set for
``AITER_CONFIG_GEMM_A8W8_BLOCKSCALE_BPRESHUFFLE`` that differs from the image's makes ``module_gemm_a8w8_blockscale_
bpreshuffle`` (and ``module_deepgemm_opus``, which reads the same table) stale.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from hyperloom.measure.framework.tests.test_table_modules import BPRESHUFFLE, CONFIG, _package, _table
from hyperloom.measure.runner import rebuild_watchdog
from hyperloom.measure.runner.rebuild_watchdog import raise_rebuild_watchdog, rebuild_watchdog_sec
from hyperloom.measure.socket import TestCall

_UTF8 = "utf-8"
_FLAG = "--watchdog-timeout"


@pytest.fixture
def image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    package = _package(tmp_path / "image" / "aiter")
    (package / "jit" / "optCompilerConfig.json").write_text(json.dumps(CONFIG), encoding=_UTF8)
    monkeypatch.setattr(rebuild_watchdog, "locate_package", lambda name, python, cwd: package)
    return package


def _point(
    tmp_path: Path, name: str, *, framework: str = "sglang", agentx: bool = False, args: str = "", **envs
) -> Path:
    if agentx:
        launch = {"env": envs, "append_args": args.split()}
        bench = {"framework": framework, "agentx": {"enabled": True, "launch_overrides": launch}}
    else:
        bench = {"framework": framework, "envs": {**envs, **({"EXTRA_SGLANG_ARGS": args} if args else {})}}
    path = tmp_path / name
    path.write_text(yaml.safe_dump({"benchmark": bench}), encoding=_UTF8)
    return path


def _call(tmp_path: Path, *points: Path) -> TestCall:
    return TestCall(points=points, output_dir=tmp_path, timeout_sec=1, cwd=str(tmp_path), magpie_python="", task_id="t")


def _bench(point: Path) -> dict:
    return yaml.safe_load(point.read_text(encoding=_UTF8))["benchmark"]


def _tuned(tmp_path: Path) -> dict[str, str]:
    return {BPRESHUFFLE: _table(tmp_path, "tuned.csv", "M,N,K,kernelName\n9,9,9,bp_new\n")}


def test_a_table_that_makes_modules_stale_raises_the_watchdog_of_both_carriers(image, tmp_path: Path) -> None:
    tuned = _tuned(tmp_path)
    plain = _point(tmp_path, "plain.yaml", args="--tp 4", **tuned)
    agentx = _point(tmp_path, "agentx.yaml", agentx=True, args="--tp 4", **tuned)

    note = raise_rebuild_watchdog(_call(tmp_path, plain, agentx), None, {}, tmp_path)

    seconds = rebuild_watchdog_sec(2)  # the blockscale module and the opus module that reads the same table
    assert _bench(plain)["envs"]["EXTRA_SGLANG_ARGS"] == f"--tp 4 {_FLAG} {seconds}"
    assert _bench(agentx)["agentx"]["launch_overrides"]["append_args"] == ["--tp", "4", _FLAG, str(seconds)]
    assert note is not None and f"raised to {seconds}s" in note and "module_gemm_a8w8_blockscale_bpreshuffle" in note


def test_the_estimate_is_the_builds_plus_the_margin() -> None:
    assert rebuild_watchdog_sec(1) == rebuild_watchdog.MODULE_BUILD_SEC + rebuild_watchdog.WATCHDOG_MARGIN_SEC
    assert rebuild_watchdog_sec(1) > 300, "longer than sglang's default watchdog"


def _watchdog(point: Path, agentx: bool) -> str:
    bench = _bench(point)
    if agentx:
        return " ".join(bench["agentx"]["launch_overrides"]["append_args"])
    return bench["envs"]["EXTRA_SGLANG_ARGS"]


#: Every way a launch carries the flag: AgentX's argument list, the plain launch's argument string, and ``=``.
_CARRIERS = [(False, " "), (False, "="), (True, " "), (True, "=")]


@pytest.mark.parametrize(("agentx", "sep"), _CARRIERS)
def test_a_present_watchdog_below_the_estimate_is_raised_to_it(
    image, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, agentx: bool, sep: str
) -> None:
    """The AgentX default (1800 s) is shorter than rebuilding three modules (2100 s): the default must not cut the
    rebuild."""
    monkeypatch.setattr(rebuild_watchdog, "stale_table_modules", lambda *args: frozenset({"m1", "m2", "m3"}))
    point = _point(tmp_path, "point.yaml", agentx=agentx, args=f"--tp 4 {_FLAG}{sep}1800 --mem 0.8", **_tuned(tmp_path))

    note = raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path)

    assert rebuild_watchdog_sec(3) == 2100
    assert _watchdog(point, agentx) == f"--tp 4 {_FLAG}{sep}2100 --mem 0.8"
    assert note is not None and "raised to 2100s" in note


@pytest.mark.parametrize(("agentx", "sep"), _CARRIERS)
@pytest.mark.parametrize("present", ["4000", "1500", "fast"])
def test_a_present_watchdog_at_or_above_the_estimate_is_kept(
    image, tmp_path: Path, agentx: bool, sep: str, present: str
) -> None:
    """A candidate's 4000 s and a value equal to the estimate stay; a value that is not a number is not rewritten."""
    assert rebuild_watchdog_sec(2) == 1500
    point = _point(tmp_path, "point.yaml", agentx=agentx, args=f"--tp 4 {_FLAG}{sep}{present}", **_tuned(tmp_path))
    before = point.read_text(encoding=_UTF8)

    assert raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path) is None
    assert point.read_text(encoding=_UTF8) == before


@pytest.mark.parametrize("agentx", [False, True])
def test_the_last_of_two_watchdogs_is_the_one_raised(image, tmp_path: Path, agentx: bool) -> None:
    """sglang's parser takes the last value, so that is the one compared and raised."""
    point = _point(tmp_path, "point.yaml", agentx=agentx, args=f"{_FLAG} 4000 {_FLAG} 120", **_tuned(tmp_path))

    raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path)

    assert _watchdog(point, agentx) == f"{_FLAG} 4000 {_FLAG} 1500"


def test_a_test_whose_tables_are_the_images_keeps_sglangs_watchdog(image, tmp_path: Path) -> None:
    point = _point(tmp_path, "point.yaml", args="--tp 4")

    assert raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path) is None
    assert _FLAG not in point.read_text(encoding=_UTF8)


def test_a_vllm_point_is_left_as_it_is(image, tmp_path: Path) -> None:
    point = _point(tmp_path, "point.yaml", framework="vllm", **_tuned(tmp_path))

    assert raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path) is None
    assert _FLAG not in point.read_text(encoding=_UTF8)
