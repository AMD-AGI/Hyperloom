# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Regression: the ``optimize --mn-backend`` flag must be registered.

``_resolve_mn_backend`` reads ``getattr(args, "mn_backend", ...)`` and SKILL.md /
error messages advertise ``optimize --mn-backend dynamo`` as the primary way to
select the Dynamo backend, but the flag was never ``add_argument``-ed — so the
documented command died with ``error: unrecognized arguments`` and only the
``INFERENCE_OPTIMIZER_MN_BACKEND`` env var worked. These lock the flag in and
pin the resolution precedence (flag > env > rayjob).
"""

from __future__ import annotations

import pytest

from inference_optimizer.cli import _build_parser, _resolve_mn_backend


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
    # No flag, no env -> rayjob (legacy default), and the attr still exists.
    monkeypatch.delenv("INFERENCE_OPTIMIZER_MN_BACKEND", raising=False)
    parser = _build_parser()
    ns = parser.parse_args(_args())
    assert ns.mn_backend is None
    assert _resolve_mn_backend(ns) == "rayjob"


def test_mn_backend_flag_overrides_env(monkeypatch):
    # Flag wins over env (precedence: --mn-backend > $INFERENCE_OPTIMIZER_MN_BACKEND).
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
