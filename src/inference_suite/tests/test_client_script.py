# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import os
import subprocess
import sys
from pathlib import Path

from inference_suite import CLIENT_ROOT_ENV, CLIENT_SCRIPT
from inference_suite.staging import CLIENT_FILE, CLIENT_MODULE


def _staged_root(tmp_path: Path) -> Path:
    root = tmp_path / "inferencex"
    client = root / CLIENT_FILE
    client.parent.mkdir(parents=True)
    for package in (root / "infx", client.parent):
        (package / "__init__.py").write_text("", encoding="utf-8")
    (client.parent / "helper.py").write_text("ANSWER = 'staged client'\n", encoding="utf-8")
    client.write_text(
        "import sys\nfrom .helper import ANSWER\nif __name__ == '__main__':\n    print(ANSWER, sys.argv[1:])\n",
        encoding="utf-8",
    )
    return root


def test_the_client_script_runs_the_staged_package_module_a_harness_cannot_run_as_a_file(tmp_path):
    root = _staged_root(tmp_path)
    env = {**os.environ, CLIENT_ROOT_ENV: str(root)}

    done = subprocess.run(
        [sys.executable, "-I", str(CLIENT_SCRIPT), "--num-prompts", "8"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "staged client ['--num-prompts', '8']"


def test_the_staged_client_file_alone_cannot_run_as_a_script(tmp_path):
    root = _staged_root(tmp_path)

    done = subprocess.run([sys.executable, "-I", str(root / CLIENT_FILE)], capture_output=True, text=True, check=False)

    assert done.returncode != 0
    assert "relative import" in done.stderr


def test_the_client_script_names_the_root_env_and_module_this_package_stages():
    text = CLIENT_SCRIPT.read_text(encoding="utf-8")

    assert f'os.environ["{CLIENT_ROOT_ENV}"]' in text
    assert f'runpy.run_module("{CLIENT_MODULE}"' in text
