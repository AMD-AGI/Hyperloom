# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression: the ``optimize --mn-backend`` flag must be registered.

Lock the flag in and pin the resolution precedence (flag > env > rayjob).
"""

from __future__ import annotations

import pytest

from hyperloom.inference_optimizer.cli.multi_node import _resolve_mn_backend
from hyperloom.inference_optimizer.cli.parser import _build_parser


def _args(*extra: str) -> list[str]:
    return ["optimize", "--model", "/tmp/no-such-model", *extra]


def test_mn_backend_flag_is_registered_and_parses_dynamo():
    parser = _build_parser()
    ns = parser.parse_args(_args("--mn-backend", "dynamo"))
    assert ns.mn_backend == "dynamo"
    assert _resolve_mn_backend(ns) == "dynamo"


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
    ns = parser.parse_args(_args("--mn-backend", "dynamo"))
    assert _resolve_mn_backend(ns) == "dynamo"


def test_mn_backend_env_used_when_flag_absent(monkeypatch):
    monkeypatch.setenv("INFERENCE_OPTIMIZER_MN_BACKEND", "dynamo")
    parser = _build_parser()
    ns = parser.parse_args(_args())
    assert ns.mn_backend is None
    assert _resolve_mn_backend(ns) == "dynamo"


def test_mn_backend_rejects_invalid_choice():
    parser = _build_parser()
    with pytest.raises(SystemExit) as ei:
        parser.parse_args(_args("--mn-backend", "tensorrt"))
    assert ei.value.code == 2
