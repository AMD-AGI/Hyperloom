# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node inference optimization helpers."""

from ._internal.external_state import external_has_server_control
from ._internal.ssh_client import SshTarget, ssh_run

__all__ = ["SshTarget", "external_has_server_control", "ssh_run"]
