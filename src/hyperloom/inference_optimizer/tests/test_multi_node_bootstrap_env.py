# SPDX-FileCopyrightText: 2025 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node bootstrap.sh env-forwarding coverage.

Both custom-header env vars must be forwarded to remote workers
(OpenAI/Codex reads OPENAI_CUSTOM_HEADERS, Claude reads ANTHROPIC_CUSTOM_HEADERS).
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

import hyperloom.inference_optimizer.multi_node as _multi_node

_BOOTSTRAP = Path(_multi_node.__file__).parent / "scripts" / "bootstrap.sh"

pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="bash not available")


def _run_bootstrap(tmp_path: Path, extra_env: dict[str, str]) -> str:
    """Run bootstrap.sh in a sandbox and return the rendered env file text."""
    venv_bin = tmp_path / "venv" / "bin"
    venv_bin.mkdir(parents=True)
    py = venv_bin / "python3"
    py.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    py.chmod(0o755)

    env_file = tmp_path / "hyperloom-env.sh"
    env = {
        "PATH": "/usr/bin:/bin",
        "HYPERLOOM_VENV": str(tmp_path / "venv"),
        "ENV_FILE": str(env_file),
        "BOOTSTRAP_MARKER": str(tmp_path / ".bootstrap_done"),
        "LOG_DIR": str(tmp_path / "log"),
    }
    env.update(extra_env)
    subprocess.run(
        ["bash", str(_BOOTSTRAP)],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return env_file.read_text(encoding="utf-8")


def test_bootstrap_forwards_both_custom_headers(tmp_path: Path) -> None:
    rendered = _run_bootstrap(
        tmp_path,
        {
            "OPENAI_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: openai-key",
            "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: anthropic-key",
        },
    )
    assert "export OPENAI_CUSTOM_HEADERS='Ocp-Apim-Subscription-Key: openai-key'" in rendered
    assert "export ANTHROPIC_CUSTOM_HEADERS='Ocp-Apim-Subscription-Key: anthropic-key'" in rendered


def test_bootstrap_omits_unset_custom_headers(tmp_path: Path) -> None:
    rendered = _run_bootstrap(tmp_path, {"OPENAI_CUSTOM_HEADERS": "X-Team: hyperloom"})
    assert "export OPENAI_CUSTOM_HEADERS='X-Team: hyperloom'" in rendered
    # An unset var is not emitted as an empty placeholder.
    assert "ANTHROPIC_CUSTOM_HEADERS" not in rendered
