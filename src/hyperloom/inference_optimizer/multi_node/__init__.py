# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Multi-node inference optimization helpers."""

from ._internal.baseline_args import (
    BaselineArgsMissing,
    export_baseline_args,
    record_baseline_args,
    resolve_baseline_args,
)
from ._internal.env_safety import server_log_dir
from ._internal.external_state import external_has_server_control
from ._internal.infera_support import INFERA_POD_LOG_FILE
from ._internal.pd_cluster import pd_pod_trace_dir, previous_cluster_state, record_profile_paths
from ._internal.pd_profile import (
    add_pd_analysis,
    await_pd_role_traces,
    drop_graph_captures,
    graph_capture_profile_settings,
    infera_profile_payload,
    pd_graph_capture_profiling,
    pd_node_traces,
    pd_profile_by_steps,
    post_pd_engine_profile,
    record_analysed_profile,
    stop_profile_timeout_s,
    trace_fs_round_start,
)
from ._internal.serving_probe import (
    completion_probe_min_tokens,
    completion_probe_tokens,
    generated_tokens,
    poll_worker_health,
    worker_health_targets,
)
from ._internal.ssh_client import SshTarget, ssh_run
from ._internal.topology_file import (
    MN_BACKENDS,
    PD_ROUTER_BACKENDS,
    SSH_FANOUT_BACKENDS,
    TopologyHandoffError,
    normalize_mn_backend,
    topology_file_overlay,
)

__all__ = [
    "BaselineArgsMissing",
    "INFERA_POD_LOG_FILE",
    "MN_BACKENDS",
    "PD_ROUTER_BACKENDS",
    "SSH_FANOUT_BACKENDS",
    "SshTarget",
    "TopologyHandoffError",
    "add_pd_analysis",
    "await_pd_role_traces",
    "completion_probe_min_tokens",
    "completion_probe_tokens",
    "drop_graph_captures",
    "export_baseline_args",
    "external_has_server_control",
    "generated_tokens",
    "graph_capture_profile_settings",
    "infera_profile_payload",
    "normalize_mn_backend",
    "pd_graph_capture_profiling",
    "pd_node_traces",
    "pd_pod_trace_dir",
    "pd_profile_by_steps",
    "poll_worker_health",
    "post_pd_engine_profile",
    "previous_cluster_state",
    "record_analysed_profile",
    "record_baseline_args",
    "record_profile_paths",
    "resolve_baseline_args",
    "server_log_dir",
    "ssh_run",
    "stop_profile_timeout_s",
    "topology_file_overlay",
    "trace_fs_round_start",
    "worker_health_targets",
]
