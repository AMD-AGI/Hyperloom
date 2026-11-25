# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Node-list documents and launches the multi-node tests build on."""

from typing import Any

from inference_suite.frameworks import Framework
from inference_suite.multinode.record import (
    Frontend,
    FrontendType,
    PdLaunch,
    PdMode,
    PdSource,
    PdSourceKind,
    Role,
    RoleLaunch,
)

SSH = {"port": 22, "user": "root", "key_file": "/keys/id"}


def node_document(**fields: Any) -> dict[str, Any]:
    """A disaggregated sglang node list of ``/models/M`` on one prefill, one decode and a router host, with
    ``fields`` over it; a field given as ``None`` is left out."""
    document = {
        "framework": "sglang",
        "model": "/models/M",
        "service_url": "http://svc:8000",
        "router": {"host": "r0"},
        "prefill": [{"host": "p0"}],
        "decode": [{"host": "d0"}],
        "gpus_per_node": 8,
        **fields,
    }
    return {key: value for key, value in document.items() if value is not None}


def pd_launch(roles: dict[Role, RoleLaunch], kind: PdSourceKind = PdSourceKind.PD_TABLE, **fields: Any) -> PdLaunch:
    """A sglang launch of ``/models/M`` served as ``org/M`` behind the router on ``r0``, with ``fields`` over it:
    aggregated with no context or transfer backend when ``roles`` name a worker, else disaggregated over mori."""
    aggregated = Role.WORKER in roles
    launch: dict[str, Any] = {
        "source": PdSource(kind),
        "framework": Framework.SGLANG,
        "model": "/models/M",
        "served_model_name": "org/M",
        "context_length": None if aggregated else 2048,
        "transfer_backend": "" if aggregated else "mori",
        "roles": roles,
        "frontend": Frontend(FrontendType.SGLANG_ROUTER, "r0", ()),
        "mode": PdMode.AGGREGATED if aggregated else PdMode.DISAGGREGATED,
    }
    return PdLaunch(**{**launch, **fields})
