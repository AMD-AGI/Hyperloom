# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A variable the shell exported keeps its value through the kernel-agent install steps."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from ._kernel_tools_harness import sourceable_kernel_tools


def test_exported_geak_run_mode_wins_over_dotenv(tmp_path: Path) -> None:
    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    (repo_root / ".env").write_text("GEAK_RUN_MODE=full\n", encoding="utf-8")
    sourceable = sourceable_kernel_tools(tmp_path)
    # No subscription token, so a step that reloads .env whenever a credential is unset would reload it here.
    script = f"""set -euo pipefail
export ANTHROPIC_API_KEY=sk-test
export ANTHROPIC_BASE_URL=https://api.anthropic.com
export REPO_ROOT={repo_root}
export USER_DATA_PATH={tmp_path}
export HYPERLOOM_RUNTIME_DIR={tmp_path / "runtime"}
export KERNEL_AGENT_ENV={tmp_path / "runtime" / "kernel-agent.env.sh"}
export GEAK_ROOT={tmp_path / "geak"}
export TRACELENS_ROOT={tmp_path / "tracelens"}
export GEAK_RUN_MODE=quick
source {sourceable}
printf '%s' "$GEAK_RUN_MODE_VAL"
"""
    env = {k: v for k, v in os.environ.items() if not k.startswith(("ANTHROPIC_", "CLAUDE_", "GEAK_"))}
    proc = subprocess.run(["bash", "-c", script], env=env, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == "quick", proc.stderr
