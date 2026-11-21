# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Regression: the ``optimize --mn-backend`` flag must be registered."""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.cli import multi_node as cli_mn
from hyperloom.inference_optimizer.cli.multi_node import _resolve_mn_backend, _stamp_pd_backend
from hyperloom.inference_optimizer.cli.parser import _build_parser


def _args(*extra: str) -> list[str]:
    return ["optimize", "--model", "/tmp/no-such-model", *extra]


def test_mn_backend_flag_is_registered_and_parses_infera():
    parser = _build_parser()
    ns = parser.parse_args(_args("--mn-backend", "infera"))
    assert ns.mn_backend == "infera"
    assert _resolve_mn_backend(ns) == "infera"


def test_mn_backend_flag_parses_rayjob():
    parser = _build_parser()
    ns = parser.parse_args(_args("--mn-backend", "rayjob"))
    assert ns.mn_backend == "rayjob"
    assert _resolve_mn_backend(ns) == "rayjob"


def test_mn_backend_default_none_falls_back_to_rayjob(monkeypatch):
    monkeypatch.delenv("INFERENCE_OPTIMIZER_MN_BACKEND", raising=False)
    parser = _build_parser()
    ns = parser.parse_args(_args())
    assert ns.mn_backend is None
    assert _resolve_mn_backend(ns) == "rayjob"


def test_mn_backend_flag_overrides_env(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "rayjob")
    parser = _build_parser()
    ns = parser.parse_args(_args("--mn-backend", "infera"))
    assert _resolve_mn_backend(ns) == "infera"


def test_mn_backend_env_used_when_flag_absent(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "infera")
    parser = _build_parser()
    ns = parser.parse_args(_args())
    assert ns.mn_backend is None
    assert _resolve_mn_backend(ns) == "infera"


@pytest.mark.parametrize("value", ["atomesh", "ATOMESH"])
def test_mn_backend_flag_names_atomesh(value):
    ns = _build_parser().parse_args(_args("--mn-backend", value))
    assert ns.mn_backend == "atomesh"
    assert _resolve_mn_backend(ns) == "atomesh"


def test_mn_backend_env_names_atomesh(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "atomesh")
    ns = _build_parser().parse_args(_args())
    assert _resolve_mn_backend(ns) == "atomesh"


def test_pd_router_backend_exits_2_without_pd_mode_disaggregated(monkeypatch):
    monkeypatch.delenv("PD_MODE", raising=False)
    with pytest.raises(SystemExit) as ei:
        _stamp_pd_backend({"backend": "atomesh"})
    assert ei.value.code == 2


def test_pd_router_backend_records_the_handed_off_pods(monkeypatch):
    monkeypatch.setenv("PD_MODE", "disaggregated")
    monkeypatch.setenv("HYPERLOOM_MN_EXT_SSH_PORT", "29000")
    monkeypatch.setattr(cli_mn, "_settle_pd_baseline_args", lambda _state: None)
    state = {
        "backend": "sglangrouter",
        "service_url": "",
        "prefill_pod_ips": ["10.0.0.1"],
        "decode_pod_ips": ["10.0.0.2"],
    }
    _stamp_pd_backend(state)
    pods = state["router_pods"] + state["prefill_pods"] + state["decode_pods"]
    assert [pod["role"] for pod in pods] == ["prefill", "decode"]
    assert {pod["sshPort"] for pod in pods} == {29000}


def test_mn_backend_rejects_invalid_choice():
    parser = _build_parser()
    with pytest.raises(SystemExit) as ei:
        parser.parse_args(_args("--mn-backend", "tensorrt"))
    assert ei.value.code == 2
