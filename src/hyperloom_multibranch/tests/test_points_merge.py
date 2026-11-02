# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""``points merge``: the points' best trees in one vLLM patch, each behind HYPERLOOM_OPERATING_POINT.

The vLLM is a stand-in with the three files the patch hooks (``envs.py``, ``env_override.py``, ``engine/arg_utils.py``)
and one module both points patched; each point's session holds the recipes, blobs and installed tree the merge reads.
The patched tree is then imported under each operating point, the way ``vllm serve`` builds its parser.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from hyperloom_multibranch.points_merge import POINT_VARIABLE, MergeRefused, _branched, _Edit, merge

VLLM = {
    "vllm/__init__.py": "",
    "vllm/envs.py": (
        "import os\nfrom collections.abc import Callable\nfrom typing import TYPE_CHECKING, Any\n\n"
        'if TYPE_CHECKING:\n    VLLM_HOST_IP: str = ""\n\n'
        'environment_variables: dict[str, Callable[[], Any]] = {\n    "VLLM_HOST_IP": lambda: os.getenv("VLLM_HOST_IP", ""),\n}\n'
    ),
    "vllm/env_override.py": 'import os\n\nos.environ["SEEN_AT_IMPORT"] = os.environ.get("FAKE_MHA", "unset")\n',
    "vllm/engine/__init__.py": "",
    "vllm/engine/arg_utils.py": (
        "import argparse\nimport json\n\n\nclass EngineArgs:\n    @staticmethod\n"
        "    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:\n"
        '        parser.add_argument("--block-size", type=int, default=None)\n'
        '        parser.add_argument("--kv-cache-dtype", default="auto")\n'
        '        parser.add_argument("--compilation-config", "-cc", type=json.loads, default=None)\n'
        "        return parser\n"
    ),
    "vllm/fusion.py": (
        '"""Fusion."""\n\nfrom __future__ import annotations\n\n\ndef sites():\n    return "debug"\n\n\n'
        "def width():\n    return total(\n        1,\n        0,\n    )\n\n\ndef total(a, b):\n    return a + b\n"
    ),
}
FUSION = '-cc \'{"pass_config": {"fusion": true}}\''
ROOT = {
    "base_image_digest": "sha256:image",
    "components": [{"component": "vllm", "patches": [], "upstream_ref": "u", "upstream_repo": "r"}],
    "kernels": [],
    "launch": {"extra_envs": {}, "extra_server_args": ""},
    "runtime_snapshot": None,
    "schema_version": 1,
}
#: aiter's ``jit/core.py`` as the table helpers read it: one variable, its default table and its tuned name.
CORE = """import os
AITER_ROOT_DIR = os.path.dirname(__file__)
AITER_CONFIG_GEMM_BF16 = os.getenv("AITER_CONFIG_GEMM_BF16", f"{AITER_ROOT_DIR}/aiter/configs/bf16_tuned_gemm.csv")


class AITER_CONFIG:
    @property
    def AITER_CONFIG_GEMM_BF16_FILE(self):
        return self.get_config_file("AITER_CONFIG_GEMM_BF16", AITER_CONFIG_GEMM_BF16, "bf16_tuned_gemm")
"""
PROBE = """
import argparse, json, os, sys
import vllm.env_override
from vllm import fusion
from vllm.engine.arg_utils import EngineArgs
args = EngineArgs.add_cli_args(argparse.ArgumentParser()).parse_args(sys.argv[1:])
print(json.dumps({"block": args.block_size, "dtype": args.kv_cache_dtype, "cc": args.compilation_config,
                  "mha": os.environ.get("FAKE_MHA"), "seen": os.environ.get("SEEN_AT_IMPORT"),
                  "bf16": os.environ.get("AITER_CONFIG_GEMM_BF16"), "sites": fusion.sites(), "width": fusion.width()}))
"""


def _git(cwd: Path, *args: str) -> str:
    env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    command = ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-C", str(cwd), *args]
    return subprocess.run(command, capture_output=True, text=True, check=True, env=env).stdout


def _write(root: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text, encoding="utf-8")


@pytest.fixture
def installed(tmp_path: Path) -> Path:
    repo = tmp_path / "installed"
    repo.mkdir()
    _git(repo, "init", "-q")
    _write(repo, VLLM)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "installed")
    return repo


@pytest.fixture
def aiter(tmp_path: Path) -> Path:
    package = tmp_path / "aiter"
    _write(package, {"jit/core.py": CORE, "configs/bf16_tuned_gemm.csv": "M,N,K,kernelName\n1,2,3,default\n"})
    return package


def _patch(installed: Path, scratch: Path, changes: list[tuple[str, str, str]]) -> bytes:
    """A patch of the installed tree that makes each replacement ``old`` -> ``new`` in its file."""

    _git(installed, "clone", "-q", ".", str(scratch))
    for path, old, new in changes:
        text = (scratch / path).read_text(encoding="utf-8")
        assert old in text
        (scratch / path).write_text(text.replace(old, new, 1), encoding="utf-8")
    return _git(scratch, "diff").encode()


def _point(
    tmp_path: Path,
    installed: Path,
    conc: int,
    *,
    launch: dict,
    patches: list[tuple[str, str, str]] | None = None,
    table: bytes | None = None,
    exit_code: int = 0,
) -> Path:
    """A point of ``tmp_path/points`` whose run's FINAL chose a recipe of ``launch``, ``patches`` and ``table``."""

    session = tmp_path / f"session-c{conc}"
    blobs = session / "MB-blobs"
    blobs.mkdir(parents=True)
    recipe = json.loads(json.dumps(ROOT))
    recipe["launch"] = launch

    def blob(data: bytes) -> str:
        digest = hashlib.sha256(data).hexdigest()
        (blobs / digest).write_bytes(data)
        return digest

    if patches:
        digest = blob(_patch(installed, tmp_path / f"patch-c{conc}", patches))
        recipe["components"][0]["patches"] = [{"path": f"patch/vllm/{digest}.patch", "sha256": digest}]
    if table is not None:
        digest = blob(table)
        load = {"env": "AITER_CONFIG_GEMM_BF16", "rows_only": True, "via": "env_file"}
        recipe["kernels"] = [
            {"kind": "gemm", "metadata": {"load": load}, "ref": f"kernel/gemm/bf16/{digest}.csv", "sha256": digest}
        ]
    repo = session / "MB-repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    for message, document in (("baseline", ROOT), ("chosen", recipe)):
        (repo / "recipe.json").write_text(json.dumps(document), encoding="utf-8")
        _git(repo, "add", "recipe.json")
        _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    chosen = _git(repo, "rev-parse", "HEAD").strip()
    _write(
        session, {"MB-reports/MB-report.json": json.dumps({"final": {"chosen": {"label": "b01", "commit": chosen}}})}
    )
    _git(installed, "clone", "-q", ".", str(session / "MB-source"))
    run = {"session": str(session), "exit_code": exit_code}
    _write(tmp_path / "points", {f"c{conc}/point.json": json.dumps({"conc": conc, "run": run})})
    return session


def _two_points(tmp_path: Path, installed: Path) -> Path:
    _point(
        tmp_path,
        installed,
        8,
        launch={"extra_envs": {"FAKE_MHA": "0"}, "extra_server_args": f"--block-size 64 {FUSION}"},
        patches=[("vllm/fusion.py", 'return "debug"', 'return "info"'), ("vllm/env_override.py", "unset", "none")],
    )
    _point(
        tmp_path,
        installed,
        64,
        launch={"extra_envs": {}, "extra_server_args": f"--kv-cache-dtype fp8 {FUSION}"},
        patches=[
            ("vllm/fusion.py", 'return "debug"', 'return "info"'),
            ("vllm/fusion.py", "        1,", "        64,"),
        ],
        table=b"M,N,K,kernelName\n64,2,3,tuned\n",
    )
    return tmp_path / "points"


def _serve(tree: Path, point: str | None, *args: str) -> dict:
    env = {key: value for key, value in os.environ.items() if key not in (POINT_VARIABLE, "FAKE_MHA")}
    env.update({"PYTHONPATH": str(tree), **({POINT_VARIABLE: point} if point else {})})
    done = subprocess.run([sys.executable, "-c", PROBE, *args], capture_output=True, text=True, env=env, check=False)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_each_point_s_tree_runs_under_its_flag_and_the_root_s_without_one(
    tmp_path: Path, installed: Path, aiter: Path
) -> None:
    merged = merge(_two_points(tmp_path, installed), aiter)

    tree = tmp_path / "applied"
    _git(installed, "clone", "-q", ".", str(tree))
    _git(tree, "apply", str(merged.patch))
    assert merged.points == (8, 64)
    assert _serve(tree, None) == {
        "block": None,
        "dtype": "auto",
        "cc": None,
        "mha": None,
        "seen": "unset",
        "bf16": None,
        "sites": "debug",
        "width": 1,
    }
    eight = _serve(tree, "8")
    assert (eight["block"], eight["dtype"], eight["cc"]) == (64, "auto", {"pass_config": {"fusion": True}})
    assert (eight["mha"], eight["seen"], eight["sites"], eight["width"]) == ("0", "0", "info", 1)
    assert eight["bf16"] is None
    sixty_four = _serve(tree, "64")
    assert (sixty_four["block"], sixty_four["dtype"], sixty_four["sites"], sixty_four["width"]) == (
        None,
        "fp8",
        "info",
        64,
    )
    assert (sixty_four["mha"], sixty_four["seen"]) == (None, "unset"), "C8's environment and code stay C8's"
    listed = sixty_four["bf16"].split(os.pathsep)
    assert listed[0] == str(aiter / "configs" / "bf16_tuned_gemm.csv")
    assert Path(listed[1]).read_bytes() == b"M,N,K,kernelName\n64,2,3,tuned\n"
    assert _serve(tree, "8", "--block-size", "16")["block"] == 16, "an argument given still wins"

    summary = json.loads(merged.summary.read_text())
    assert summary["shared_code"] == ["vllm/fusion.py"]
    assert [point["code"] for point in summary["points"]] == [
        ["vllm/env_override.py", "vllm/fusion.py"],
        ["vllm/fusion.py"],
    ]
    assert (
        summary["points"][1]["tables"]["AITER_CONFIG_GEMM_BF16"][1]
        == "c64/aiter_config_gemm_bf16/"
        + Path(
            json.loads(_git(tmp_path / "session-c64" / "MB-repo", "show", "HEAD:recipe.json"))["kernels"][0]["ref"]
        ).name
    )
    recipe = json.loads((merged.patch.parent / "recipe.json").read_text())
    assert recipe["components"][0]["patches"] == [
        {"path": f"patch/vllm/{summary['patch_sha256']}.patch", "sha256": summary["patch_sha256"]}
    ]


def test_a_point_the_tree_does_not_hold_is_refused(tmp_path: Path, installed: Path, aiter: Path) -> None:
    merged = merge(_two_points(tmp_path, installed), aiter)
    tree = tmp_path / "applied"
    _git(installed, "clone", "-q", ".", str(tree))
    _git(tree, "apply", str(merged.patch))

    env = {**os.environ, "PYTHONPATH": str(tree), POINT_VARIABLE: "32"}
    done = subprocess.run([sys.executable, "-c", "import vllm.env_override"], capture_output=True, text=True, env=env)
    assert done.returncode != 0 and "names no operating point of this tree, which has 8, 64" in done.stderr


def test_a_point_without_final_s_choice_is_left_out_and_what_cannot_branch_is_refused(
    tmp_path: Path, installed: Path, aiter: Path
) -> None:
    _point(tmp_path, installed, 8, launch={"extra_envs": {}, "extra_server_args": "--block-size 64"})
    _point(tmp_path, installed, 64, launch={"extra_envs": {}, "extra_server_args": ""}, exit_code=1)

    merged = merge(tmp_path / "points", aiter)
    assert merged.points == (8,) and "exited 1" in merged.left_out[64]

    other = tmp_path / "other"
    _point(
        other,
        installed,
        16,
        launch={"extra_envs": {}, "extra_server_args": ""},
        patches=[("vllm/fusion.py", "1,", "1, (")],
    )
    with pytest.raises(MergeRefused, match="no longer Python|whole statements"):
        merge(other / "points", aiter)


def _wrapped(text: str, *edits: _Edit) -> str:
    out = _branched("f.py", text, edits)
    ast.parse(out)
    return out


def test_a_change_inside_a_statement_takes_the_whole_statement_and_points_that_agree_share_a_branch() -> None:
    text = "def f():\n    x = g(\n        1,\n    )\n    return x\n"
    out = _wrapped(text, _Edit("8", 2, 3, ("        8,\n",)), _Edit("64", 2, 3, ("        64,\n",)))
    assert out == (
        "def f():\n"
        '    if _hyperloom_points.active("8"):\n        x = g(\n            8,\n        )\n'
        '    elif _hyperloom_points.active("64"):\n        x = g(\n            64,\n        )\n'
        "    else:\n        x = g(\n            1,\n        )\n"
        "    return x\n"
    )
    shared = _wrapped(text, _Edit("8", 4, 5, ("    return 2\n",)), _Edit("64", 4, 5, ("    return 2\n",)))
    assert '    if _hyperloom_points.active("8", "64"):\n        return 2\n    else:\n        return x\n' in shared
    added = _wrapped(text, _Edit("8", 4, 4, ("    x += 1\n",)))
    assert '    if _hyperloom_points.active("8"):\n        x += 1\n    return x\n' in added


def test_the_shipped_server_arguments_are_the_launch_s_words(tmp_path: Path, installed: Path, aiter: Path) -> None:
    merged = merge(_two_points(tmp_path, installed), aiter)
    summary = json.loads(merged.summary.read_text())
    assert summary["points"][0]["server_args"] == ["--block-size", "64", *shlex.split(FUSION)]
