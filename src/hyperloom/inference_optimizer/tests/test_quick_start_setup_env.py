# Copyright Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[4]
SCRIPT = REPO_ROOT / "quick-start" / "setup_env.sh"


def _run_setup_env_script(tmp_path: Path, env: dict[str, str]) -> str:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            [
                "SAFE_API_KEY=ak-placeholder",
                "# OPENAI_BASE_URL=",
                "# TRACELENS_ROOT=",
                "USER_DATA_PATH=/workspace/hyperloom",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script_copy = tmp_path / "setup_env.sh"
    text = SCRIPT.read_text(encoding="utf-8")
    text = text.replace('ENV_FILE="/opt/Hyperloom/.env"', f'ENV_FILE="{env_file}"')
    text = text.replace("tail -f /dev/null", ":")
    script_copy.write_text(text, encoding="utf-8")

    run_env = os.environ.copy()
    run_env.update(env)
    subprocess.run(
        ["bash", str(script_copy)],
        cwd=REPO_ROOT,
        env=run_env,
        text=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return env_file.read_text(encoding="utf-8")


def test_setup_env_preserves_sed_replacement_metacharacters(tmp_path: Path) -> None:
    text = _run_setup_env_script(
        tmp_path,
        {
            "SAFE_API_KEY": "ak-safe",
            "OPENAI_BASE_URL": r"https://gateway.example/v1?team=a&env=b|stage",
            "TRACELENS_ROOT": r"/opt/Trace\Lens",
            "USER_DATA_PATH": "/workspace/hyperloom",
        },
    )

    assert r"OPENAI_BASE_URL=https://gateway.example/v1?team=a&env=b|stage" in text
    assert r"TRACELENS_ROOT=/opt/Trace\Lens" in text
