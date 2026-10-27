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


@pytest.mark.parametrize(("args", "agentx"), [(f"{_FLAG} 120", False), (f"{_FLAG}=120", False), (f"{_FLAG} 120", True)])
def test_a_launch_that_sets_the_watchdog_keeps_its_own(image, tmp_path: Path, args: str, agentx: bool) -> None:
    point = _point(tmp_path, "point.yaml", agentx=agentx, args=args, **_tuned(tmp_path))
    before = point.read_text(encoding=_UTF8)

    assert raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path) is None
    assert point.read_text(encoding=_UTF8) == before


def test_a_test_whose_tables_are_the_images_keeps_sglangs_watchdog(image, tmp_path: Path) -> None:
    point = _point(tmp_path, "point.yaml", args="--tp 4")

    assert raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path) is None
    assert _FLAG not in point.read_text(encoding=_UTF8)


def test_a_vllm_point_is_left_as_it_is(image, tmp_path: Path) -> None:
    point = _point(tmp_path, "point.yaml", framework="vllm", **_tuned(tmp_path))

    assert raise_rebuild_watchdog(_call(tmp_path, point), None, {}, tmp_path) is None
    assert _FLAG not in point.read_text(encoding=_UTF8)
