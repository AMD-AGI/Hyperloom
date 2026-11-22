# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

from dataclasses import replace

import pytest

from inference_suite.contracts.base_launch import Workload
from inference_suite.frameworks import Framework
from inference_suite.multinode.query import PdQuery, PdQueryError, RoleAsk
from inference_suite.multinode.record import FrontendType, PdMode, Role


def _query(**fields):
    query = PdQuery(
        model_id="deepseek-ai/DeepSeek-R1",
        model_path="/models/DeepSeek-R1/",
        framework=Framework.SGLANG,
        board="mi355x",
        gpus_per_node=8,
        workload=Workload.FIXED,
        concurrencies=(4, 64, 16),
        roles={Role.PREFILL: RoleAsk(("p1",), tp=8), Role.DECODE: RoleAsk(("d1", "d2"))},
        isl=1024,
        osl=1024,
    )
    return replace(query, **fields)


def test_a_query_exposes_its_roles_model_frontend_and_target():
    query = _query()
    assert query.roles[Role.PREFILL].tp == 8 and query.roles[Role.DECODE].hosts == ("d1", "d2")
    assert query.model_name == "DeepSeek-R1" and query.target_concurrency == 64
    assert query.served_frontend is FrontendType.SGLANG_ROUTER
    assert _query(frontend=FrontendType.INFERA).served_frontend is FrontendType.INFERA
    assert _query(framework=Framework.ATOM).served_frontend is FrontendType.ATOMESH


@pytest.mark.parametrize(("hosts", "fields"), [((), {}), (("",), {}), (("h",), {"tp": 0}), (("h",), {"ep": "2"})])
def test_a_role_ask_refuses_missing_hosts_or_bad_parallelism(hosts, fields):
    with pytest.raises(PdQueryError):
        RoleAsk(hosts, **fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"framework": Framework.VLLM},
        {"workload": Workload.SCRIPTABLE},
        {"board": ""},
        {"gpus_per_node": 0},
        {"concurrencies": ()},
        {"framework": Framework.ATOM, "frontend": FrontendType.INFERA},
        {"mode": PdMode.AGGREGATED},
        {"osl": None},
        {"workload": Workload.AGENTX},
    ],
)
def test_a_query_refuses_a_broken_contract(fields):
    with pytest.raises(PdQueryError):
        _query(**fields)
