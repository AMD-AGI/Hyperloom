# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node inference optimization helpers."""

from ._internal.ssh_client import SshTarget, ssh_run

__all__ = ["SshTarget", "ssh_run"]
