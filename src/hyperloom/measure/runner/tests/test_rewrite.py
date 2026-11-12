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


AGENTX_SHEET = """
benchmark:
  framework: sglang
  agentx: {enabled: true, mode: canonical, launch_overrides: {version: 1, env: {SGLANG_X: "1", PYTHONPATH: /rt/site}}}
  envs:
    TP: 1
"""


def _rewrite(tmp_path, farm=None, visible="0,1", sheet=SHEET, path=None):
    out = tmp_path / "point.yaml"
    rewrite_config(
        sheet,
        out,
        Placement(
            placed={"aiter": tmp_path / "attach" / "aiter"},
            pid_dir=tmp_path / "pids",
            inferencex_path=tmp_path / "ix",
            visible_devices=visible,
            farm=farm,
            path=path,
        ),
    )
    return yaml.safe_load(out.read_text())["benchmark"]


HOST_PATH_SHEET = """
benchmark:
  framework: sglang
  envs:
    TP: 1
    PATH: /root/.local/bin:/usr/local/bin:/usr/bin
"""


def test_the_servers_path_is_the_sandboxes_not_the_hosts_that_rendered_the_sheet(tmp_path):
    # A sheet rendered on the application host carried the host's PATH, so the hosted server started with
    # /usr/bin/python3, which has no sglang; the image's interpreter is first on the sandbox's own PATH.
    bench = _rewrite(tmp_path, sheet=HOST_PATH_SHEET, path="/opt/venv/bin:/usr/bin")

    assert bench["envs"]["PATH"] == "/opt/venv/bin:/usr/bin"


def test_a_sheet_that_names_no_path_is_given_none(tmp_path):
    bench = _rewrite(tmp_path, path="/opt/venv/bin:/usr/bin")

    assert "PATH" not in bench["envs"]


def test_without_the_sandboxes_path_the_sheets_path_stays(tmp_path):
    bench = _rewrite(tmp_path, sheet=HOST_PATH_SHEET)

    assert bench["envs"]["PATH"] == "/root/.local/bin:/usr/local/bin:/usr/bin"


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


def test_the_rebuilt_framework_goes_first_on_an_agentx_servers_path_not_in_the_benchmark_envs(tmp_path):
    (tmp_path / "farm" / "site").mkdir(parents=True)

    bench = _rewrite(tmp_path, farm=Farm(tmp_path / "farm" / "site", ("sglang",)), sheet=AGENTX_SHEET)

    launch = bench["agentx"]["launch_overrides"]
    assert launch["env"]["PYTHONPATH"] == f"{tmp_path}/farm/site:/rt/site" and launch["env"]["SGLANG_X"] == "1"
    assert "PYTHONPATH" not in bench["envs"]


def test_a_rebuilt_aiter_is_told_the_images_meta_directory_in_the_benchmark_envs(tmp_path):
    farm = Farm(tmp_path / "farm" / "site", ("aiter",), aiter_meta=tmp_path / "image" / "aiter_meta")

    bench = _rewrite(tmp_path, farm=farm)

    assert bench["envs"]["AITER_META_DIR"] == f"{tmp_path}/image/aiter_meta"
    assert bench["envs"]["PYTHONPATH"] == f"{tmp_path}/farm/site:/rt/site"


def test_a_rebuilt_aiter_is_told_the_images_meta_directory_in_an_agentx_launch_not_the_benchmark_envs(tmp_path):
    (tmp_path / "farm" / "site").mkdir(parents=True)
    farm = Farm(tmp_path / "farm" / "site", ("aiter",), aiter_meta=tmp_path / "image" / "aiter_meta")

    bench = _rewrite(tmp_path, farm=farm, sheet=AGENTX_SHEET)

    launch = bench["agentx"]["launch_overrides"]
    assert launch["env"]["AITER_META_DIR"] == f"{tmp_path}/image/aiter_meta" and launch["env"]["SGLANG_X"] == "1"
    assert "AITER_META_DIR" not in bench["envs"]


def test_a_meta_directory_the_sheet_names_is_the_servers_not_the_images(tmp_path):
    farm = Farm(tmp_path / "farm" / "site", ("aiter",), aiter_meta=tmp_path / "image" / "aiter_meta")

    bench = _rewrite(tmp_path, farm=farm, sheet=SHEET + "    AITER_META_DIR: /patched/meta\n")

    assert bench["envs"]["AITER_META_DIR"] == "/patched/meta"


def test_a_farm_without_aiter_and_a_sandbox_without_a_farm_put_no_meta_directory_in_the_environment(tmp_path):
    (tmp_path / "farm" / "site").mkdir(parents=True)

    for farm, sheet in ((Farm(tmp_path / "farm" / "site", ("sglang",)), SHEET), (None, SHEET)):
        assert "AITER_META_DIR" not in _rewrite(tmp_path, farm=farm, sheet=sheet)["envs"]
    agentx = _rewrite(tmp_path, farm=Farm(tmp_path / "farm" / "site", ("sglang",)), sheet=AGENTX_SHEET)
    assert "AITER_META_DIR" not in agentx["agentx"]["launch_overrides"]["env"]
