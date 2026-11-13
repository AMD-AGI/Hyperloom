# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node launches, prefill/decode disaggregated or aggregated: resolve one from its sources, freeze it, run it over
SSH."""

from __future__ import annotations

from inference_suite.multinode.commands import engine_urls
from inference_suite.multinode.launch import (
    Cluster,
    LaunchError,
    LaunchOptions,
    engines_gone,
    replace_flags,
    restart,
    start,
    stop,
)
from inference_suite.multinode.nodes import NODE_LIST_ENV, NodeList, NodeListError, parse_node_list, read_node_list
from inference_suite.multinode.recipe import OperatorArgs, PdResolveError, ResolveInputs, resolve
from inference_suite.multinode.record import (
    RECORD_NAME,
    PdLaunch,
    PdMode,
    PdRecordError,
    Role,
    read_record,
    write_record,
)
from inference_suite.multinode.session import cluster_of, placed_launch, resolve_session
from inference_suite.multinode.ssh import SshClient, SshError

__all__ = [
    "engine_urls",
    "engines_gone",
    "replace_flags",
    "placed_launch",
    "cluster_of",
    "NODE_LIST_ENV",
    "RECORD_NAME",
    "Cluster",
    "LaunchError",
    "LaunchOptions",
    "NodeList",
    "NodeListError",
    "OperatorArgs",
    "PdLaunch",
    "PdMode",
    "PdRecordError",
    "PdResolveError",
    "ResolveInputs",
    "Role",
    "SshClient",
    "SshError",
    "parse_node_list",
    "read_node_list",
    "read_record",
    "resolve",
    "resolve_session",
    "restart",
    "start",
    "stop",
    "write_record",
]
