# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""A remote sandbox whose commands run as subprocesses of this host, for the tests of the remote agent sandbox."""

import os
import subprocess

from hyperloom.sandbox import CommandResult, RunSettings, SandboxGone, SandboxUnavailable

#: The sandbox's own environment, which a command runs on.
SANDBOX_ENV = {"PATH": os.environ["PATH"], "ROCR_VISIBLE_DEVICES": "6,7", "IMAGE_ONLY": "yes"}


class SubprocessSandbox:
    """Runs each command as a local subprocess on :data:`SANDBOX_ENV`; ``gone`` ends it."""

    sandbox_id = "sbx-1"

    def __init__(self) -> None:
        self.gone = False
        self.commands: list[tuple[str, dict[str, str]]] = []

    def run(self, cmd: str, *, background: bool = False, settings: RunSettings = RunSettings(), sinks=None):
        if self.gone:
            raise SandboxGone("sandbox sbx-1 ended")
        envs = dict(settings.envs or {})
        self.commands.append((cmd, envs))
        try:
            done = subprocess.run(
                ["bash", "-c", cmd],
                env={**SANDBOX_ENV, **envs},
                capture_output=True,
                text=True,
                timeout=settings.timeout_sec,
            )
        except subprocess.TimeoutExpired as error:
            raise SandboxUnavailable(f"timed out: {cmd}") from error
        return CommandResult(done.returncode, done.stdout, done.stderr)
