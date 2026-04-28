"""Regression tests for the Marathon launcher preflight."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path


def _symlink_tool(fake_bin: Path, name: str) -> None:
    target = shutil.which(name)
    assert target is not None, f"test host missing required tool: {name}"
    (fake_bin / name).symlink_to(target)


def _write_executable(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(0o755)


def test_preflight_apt_get_install_is_wallclock_bounded(tmp_path: Path) -> None:
    """A hung apt-get install must fail preflight instead of polling forever."""

    repo = Path(__file__).resolve().parents[3]
    run_sh = repo / ".cursor/skills/marathon-inference-optimization/scripts/launcher/run.sh"

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()

    # Keep PATH hermetic so `jq` is genuinely missing, while the launcher still
    # has the basic shell utilities it needs before the install attempt.
    for tool in ("bash", "date", "dirname", "hostname", "tail", "timeout", "uname"):
        _symlink_tool(fake_bin, tool)

    for tool in ("claude", "curl", "tmux"):
        _write_executable(fake_bin / tool, "#!/usr/bin/env bash\nexit 0\n")

    _write_executable(
        fake_bin / "apt-get",
        "#!/usr/bin/env bash\n/bin/sleep 60\n",
    )

    env = os.environ.copy()
    env.update({
        "PATH": str(fake_bin),
        "PREFLIGHT_STEP_TIMEOUT_S": "1",
        "STAGE_ONLY": "1",
    })

    started = time.monotonic()
    proc = subprocess.run(
        [str(fake_bin / "bash"), str(run_sh)],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
        timeout=10,
    )
    elapsed = time.monotonic() - started

    assert elapsed < 8
    assert proc.returncode == 14
    assert "missing apt pkgs: jq" in proc.stdout
    assert "preflight step 'apt-get update' timed out after 1s" in proc.stderr
