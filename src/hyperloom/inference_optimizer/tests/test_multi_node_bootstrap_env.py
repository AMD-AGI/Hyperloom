# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Multi-node bootstrap.sh credential-minimization coverage.

bootstrap.sh renders ``/etc/profile.d/hyperloom-env.sh`` with the framework
venv PATH only. Credential-bearing env (``*_API_KEY`` / ``*_BASE_URL`` /
``*_CUSTOM_HEADERS`` — the latter carry subscription keys) must NOT be written
into that world-readable (0644) file; later Ray Dashboard REST jobs inherit
them from the head-pod container env instead.
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


def test_bootstrap_does_not_leak_custom_headers(tmp_path: Path) -> None:
    """Custom-header creds must never be written to the world-readable env file."""
    rendered = _run_bootstrap(
        tmp_path,
        {
            "OPENAI_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: openai-key",
            "ANTHROPIC_CUSTOM_HEADERS": "Ocp-Apim-Subscription-Key: anthropic-key",
        },
    )
    assert "OPENAI_CUSTOM_HEADERS" not in rendered
    assert "ANTHROPIC_CUSTOM_HEADERS" not in rendered
    assert "openai-key" not in rendered
    assert "anthropic-key" not in rendered


def test_bootstrap_renders_path_only(tmp_path: Path) -> None:
    """The rendered env file carries the venv PATH export and nothing else."""
    rendered = _run_bootstrap(tmp_path, {"OPENAI_CUSTOM_HEADERS": "X-Team: hyperloom"})
    assert "OPENAI_CUSTOM_HEADERS" not in rendered
    assert "X-Team: hyperloom" not in rendered
    # Only the framework venv PATH is exported (no credential re-exports).
    assert f'export PATH="{tmp_path / "venv" / "bin"}:${{PATH}}"' in rendered
    assert rendered.count("export ") == 1
