# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The prefill-decode flags of the multi-node ``restart-server`` subcommand."""

from __future__ import annotations

import argparse
import os


def add_restart_pd_shape_arguments(sp: argparse.ArgumentParser) -> None:
    """Add the prefill-decode mode and group-shape flags of ``restart-server``."""
    # Prefill-Decode disaggregation: aggregated (default) keeps a single server group; disaggregated splits into
    # prefill + decode groups fronted by a router on the head pod that binds the public 8888 port.
    sp.add_argument(
        "--pd-mode",
        choices=("aggregated", "disaggregated"),
        default="aggregated",
        help="PD disaggregation mode (default aggregated).",
    )
    sp.add_argument(
        "--pd-prefill-nodes",
        type=int,
        default=0,
        help="number of prefill nodes (disaggregated only); pn+dn==nnodes",
    )
    sp.add_argument(
        "--pd-decode-nodes",
        type=int,
        default=0,
        help="number of decode nodes (disaggregated only)",
    )
    sp.add_argument(
        "--pd-prefill-tp",
        type=int,
        default=0,
        help="TP for prefill group (disaggregated only); default = --tp",
    )
    sp.add_argument(
        "--pd-decode-tp",
        type=int,
        default=0,
        help="TP for decode group (disaggregated only); default = --tp",
    )


def add_restart_pd_role_arguments(sp: argparse.ArgumentParser) -> None:
    """Add the prefill-decode per-role, transport and router flags of ``restart-server``."""
    # Per-role EP / extra server args (disaggregated only), so prefill and decode can use different MoE topologies.
    sp.add_argument(
        "--pd-prefill-ep",
        type=int,
        default=int(os.environ.get("PD_PREFILL_EP", "0") or 0),
        help="EP for prefill group (disaggregated only); 0 = use --ep",
    )
    sp.add_argument(
        "--pd-decode-ep",
        type=int,
        default=int(os.environ.get("PD_DECODE_EP", "0") or 0),
        help="EP for decode group (disaggregated only); 0 = use --ep",
    )
    sp.add_argument(
        "--pd-prefill-extra-args",
        default=os.environ.get("PD_PREFILL_EXTRA_ARGS", ""),
        help="extra sglang args appended to the PREFILL group only "
        "(merged after the shared --extra-args); disaggregated only",
    )
    sp.add_argument(
        "--pd-decode-extra-args",
        default=os.environ.get("PD_DECODE_EXTRA_ARGS", ""),
        help="extra sglang args appended to the DECODE group only "
        "(merged after the shared --extra-args); disaggregated only",
    )
    sp.add_argument(
        "--pd-transfer-backend",
        default="",
        help="sglang: mooncake|nixl ; vllm: NixlConnector|P2pNcclConnector|"
        "MooncakeConnector|LMCacheConnectorV1; empty = framework default",
    )
    sp.add_argument(
        "--pd-ib-device",
        default="",
        help="comma-separated IB/RoCE device list (e.g. mlx5_0,mlx5_1). Empty = read $NCCL_IB_HCA from RayJob pod env.",
    )
    sp.add_argument(
        "--pd-bootstrap-port",
        type=int,
        default=8998,
        help="sglang PD bootstrap rendezvous port (default 8998)",
    )
    sp.add_argument(
        "--pd-vllm-router-cmd",
        default="",
        help="vllm-only override for router cmdline; supports {prefill}/{decode}/{port} placeholders",
    )
