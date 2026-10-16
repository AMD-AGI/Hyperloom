# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from __future__ import annotations

import yaml

from hyperloom.measure.farm import Farm
from hyperloom.measure.runner.rewrite import Placement, rewrite_config

SHEET = """
benchmark:
  framework: sglang
  inferencex_path: /session/InferenceX
  server_lifecycle: {enabled: true, pid_dir: /session/pids}
  envs:
    TP: 1
    ROCR_VISIBLE_DEVICES: "6,7"
    HIP_VISIBLE_DEVICES: "6,7"
    PYTHONPATH: /rt/site
    AITER_CONFIG: ${HL_ATTACH:aiter}
"""


def _rewrite(tmp_path, farm=None, visible="0,1"):
    out = tmp_path / "point.yaml"
    rewrite_config(
        SHEET,
        out,
        Placement(
            placed={"aiter": tmp_path / "attach" / "aiter"},
            pid_dir=tmp_path / "pids",
            inferencex_path=tmp_path / "ix",
            visible_devices=visible,
            farm=farm,
        ),
    )
    return yaml.safe_load(out.read_text())["benchmark"]


def test_the_yaml_is_made_this_sandboxes(tmp_path):
    bench = _rewrite(tmp_path)

    assert bench["envs"]["ROCR_VISIBLE_DEVICES"] == "0,1"
    assert "HIP_VISIBLE_DEVICES" not in bench["envs"]
    assert bench["envs"]["AITER_CONFIG"] == f"{tmp_path}/attach/aiter"
    assert bench["server_lifecycle"]["pid_dir"] == f"{tmp_path}/pids"
    assert bench["inferencex_path"] == f"{tmp_path}/ix"
    assert bench["envs"]["PYTHONPATH"] == "/rt/site"
    assert not list(tmp_path.glob("*.mask*")) and not list(tmp_path.glob("*.ray*"))


def test_the_rebuilt_framework_goes_first_on_the_servers_path(tmp_path):
    bench = _rewrite(tmp_path, farm=Farm(tmp_path / "farm" / "site", ("sglang",)))

    assert bench["envs"]["PYTHONPATH"] == f"{tmp_path}/farm/site:/rt/site"
