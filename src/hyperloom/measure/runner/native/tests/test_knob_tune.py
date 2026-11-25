# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``python -m hyperloom.measure.runner.native.knob_tune``: its exit status, its report and its patch."""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from hyperloom.common.visible_devices import ROCR_LEVEL_VARS, VISIBLE_DEVICE_VARS
from hyperloom.measure.runner.native import knob_tune
from hyperloom.orchestrator.actions.executors.candidate_builds import TreeBuild, TreeBuildStatus

_MMVQ = "int f() {\n    if (table_id == MMVQ_PARAMETERS_RDNA3_0) {\n        return 1;\n    }\n}\n"
_HEADER = (
    "enum ggml_type { GGML_TYPE_F32 = 0, GGML_TYPE_Q8_0 = 8, };\nenum ggml_op { GGML_OP_NONE, GGML_OP_MUL_MAT, };\n"
)


class _Builds:
    def __init__(self, root: Path, gfx: str) -> None:
        self.root = root
        self.session = SimpleNamespace(
            planner=SimpleNamespace(site=SimpleNamespace(machine=SimpleNamespace(gfx_arch=gfx)))
        )
        self.patches: list[bytes] = []

    def read_upstream(self, path: str) -> str:
        return _HEADER if path.endswith(".h") else _MMVQ

    def build_patch(self, patch: bytes) -> TreeBuild:
        self.patches.append(patch)
        forced = re.findall(rb"return (\d+); }", patch)
        bin_dir = self.root / f"bin-{forced[0].decode() if forced else 'base'}"
        bin_dir.mkdir(parents=True, exist_ok=True)
        return TreeBuild(TreeBuildStatus.BUILT, key=bin_dir.name, bin=bin_dir)


def _timed(executable: Path, cases: Path, names, env, timeout: float) -> dict[str, float]:
    faster = executable.parent.name == "bin-4"
    return {name: (50.0 if faster and "_2_" in name else 100.0) for name in names}


@pytest.fixture
def builds(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Builds:
    found = _Builds(tmp_path / "tunes", "gfx1100")
    monkeypatch.setattr(knob_tune, "tune_native_builds", lambda session: found)
    monkeypatch.setattr(knob_tune, "time_cases", _timed)
    monkeypatch.setattr(knob_tune, "check_cases", lambda exe, cases, names, env, timeout: dict.fromkeys(names, "OK"))
    return found


def _argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "--knob=mmvq_nwarps_rdna3",
        "--axis=type=Q8_0",
        "--axis=ncols=1,2",
        "--values=2,4",
        "--shape=64x128",
        f"--session={tmp_path}",
        f"--out={tmp_path / 'out'}",
        f"--gpu-lock={tmp_path / 'card.lock'}",
        *extra,
    ]


def test_a_sweep_prints_its_report_and_writes_the_winning_patch(
    builds: _Builds, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert knob_tune.main(_argv(tmp_path)) == 0

    printed = json.loads(capsys.readouterr().out)
    assert printed["winners"] == {"type=Q8_0,ncols=2": "4"}
    patch = (tmp_path / "out" / "winning.patch").read_text()
    assert "+        if (type == GGML_TYPE_Q8_0 && ncols_dst == 2) { return 4; }" in patch
    assert json.loads((tmp_path / "out" / "report.json").read_text())["winners"] == printed["winners"]
    assert builds.patches[0] == b""


def test_a_knob_the_session_does_not_compile_for_is_refused(
    builds: _Builds, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    builds.session.planner.site.machine.gfx_arch = "gfx942"

    assert knob_tune.main(_argv(tmp_path)) == 1
    assert "gfx942" in json.loads(capsys.readouterr().out)["error"]
    assert builds.patches == []


def test_a_run_that_names_no_knob_is_refused_before_any_build(
    builds: _Builds, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert knob_tune.main(_argv(tmp_path)[1:]) == 1

    assert "no kernel knob" in json.loads(capsys.readouterr().out)["error"]
    assert builds.patches == []


def test_a_session_that_builds_nothing_from_source_exits_2_with_the_refusal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert knob_tune.main(_argv(tmp_path)) == 2
    assert "builds no framework server from source" in json.loads(capsys.readouterr().out)["error"]


def test_a_round_names_its_card_by_the_rocr_mask_alone_whatever_hip_mask_the_process_was_launched_with(
    builds: _Builds, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A HIP-level mask indexes into what ROCr exposed: beside a ROCr mask of card 3 it names a fourth device of one."""
    seen: list[dict[str, str]] = []

    def timed(executable: Path, cases: Path, names, env, timeout: float) -> dict[str, float]:
        seen.append({name: env[name] for name in VISIBLE_DEVICE_VARS if name in env})
        return _timed(executable, cases, names, env, timeout)

    monkeypatch.setattr(knob_tune, "time_cases", timed)
    for name in VISIBLE_DEVICE_VARS:
        monkeypatch.setenv(name, "3")

    assert knob_tune.main(_argv(tmp_path)) == 0

    assert seen and all(masks == dict.fromkeys(ROCR_LEVEL_VARS, "3") for masks in seen)
