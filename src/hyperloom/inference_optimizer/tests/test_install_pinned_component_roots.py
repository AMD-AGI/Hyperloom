# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A rerun of the kernel-agent install follows the pinned Magpie, InferenceX and TraceLens revisions."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from ._kernel_tools_harness import sourceable_kernel_tools


def _source_kernel_tools(tmp_path: Path, exports: dict[str, Path], then: str) -> str:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(exist_ok=True)
    (tmp_path / "cache").mkdir(exist_ok=True)
    lines = [
        "set -euo pipefail",
        "export ANTHROPIC_API_KEY=sk-test ANTHROPIC_BASE_URL=https://api.anthropic.com",
        f"export REPO_ROOT={repo_root} USER_DATA_PATH={tmp_path} HYPERLOOM_CACHE_DIR={tmp_path / 'cache'}",
        f"export HYPERLOOM_RUNTIME_DIR={tmp_path / 'runtime'} KERNEL_AGENT_ENV={tmp_path / 'runtime' / 'env.sh'}",
        f"export GEAK_ROOT={tmp_path / 'geak'}",
        *(f"export {key}={value}" for key, value in exports.items()),
        f"source {sourceable_kernel_tools(tmp_path)}",
        then,
    ]
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_", "GEAK_", "TRACELENS_"))}
    proc = subprocess.run(["bash", "-c", "\n".join(lines)], env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    return proc.stdout


def test_magpie_and_inferencex_paths_stay_out_of_dotenv(tmp_path: Path) -> None:
    dotenv = tmp_path / "repo" / ".env"
    dotenv.parent.mkdir()
    dotenv.write_text(f"MAGPIE_PATH={tmp_path / 'old'}\nINFERENCEX_PATH={tmp_path / 'old'}\n", encoding="utf-8")

    _source_kernel_tools(
        tmp_path, {"MAGPIE_PATH": tmp_path / "Magpie", "INFERENCEX_PATH": tmp_path / "InferenceX"}, "write_env_file"
    )

    text = dotenv.read_text(encoding="utf-8")
    assert "MAGPIE_PATH=" not in text
    assert "INFERENCEX_PATH=" not in text


@pytest.mark.parametrize("stale", [True, False], ids=["stale-cache-checkout", "operator-checkout"])
def test_tracelens_root_follows_the_pin_unless_the_operator_chose_it(tmp_path: Path, stale: bool) -> None:
    root = tmp_path / "cache" / "TraceLens@0000000" if stale else tmp_path / "my-tracelens"

    resolved, sha = _source_kernel_tools(
        tmp_path, {"TRACELENS_ROOT": root}, 'printf "%s\\n%s" "$TRACELENS_ROOT" "$_TRACELENS_SHA"'
    ).split("\n")

    assert resolved == str(tmp_path / "cache" / f"TraceLens@{sha}" if stale else root)
