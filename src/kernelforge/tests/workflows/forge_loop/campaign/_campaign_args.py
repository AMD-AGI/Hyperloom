# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Campaign arguments and the git workspace they resolve against, shared by the campaign and loop tests."""

from __future__ import annotations

import subprocess

from kernelforge.workflows.forge_loop.campaign.campaign_config import CampaignInputs


def _git_workspace(tmp_path, name="workspace"):
    workspace = tmp_path / name
    workspace.mkdir()
    subprocess.run(
        ["git", "init", "-b", "feature/test-campaign"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Tests"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "tests@example.com"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    kernel = workspace / "kernel.py"
    driver = workspace / "driver.py"
    kernel.write_text("def k(): return 1\n")
    driver.write_text("pass\n")
    subprocess.run(["git", "add", "."], cwd=workspace, check=True)
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=workspace,
        check=True,
        capture_output=True,
    )
    return workspace, kernel, driver


def _base_args(workspace, kernel, driver, **inputs):
    return dict(
        workspace_dir=str(workspace),
        inputs=CampaignInputs(
            kernel=str(kernel), driver=str(driver), kernel_backend="triton", snr_threshold=2.0, **inputs
        ),
        resume=False,
        prepare_task=False,
    )
