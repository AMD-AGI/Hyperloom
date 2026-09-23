# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Tests for the guard that keeps a refused connection from ending the eval on UnboundLocalError."""

from __future__ import annotations

import asyncio
import sys
import types

import pytest

from hyperloom.orchestrator.actions.executors._inferencex_patcher import (
    _EVAL_UNBOUND_OUTPUTS_PY,
    ensure_eval_unbound_outputs_patched,
)


def _install_stub_lm_eval(monkeypatch, amodel_call):
    """Publish a ``lm_eval.models.api_models.TemplateAPI`` carrying ``amodel_call``."""
    api_models = types.ModuleType("lm_eval.models.api_models")
    api_models.TemplateAPI = type("TemplateAPI", (), {"amodel_call": amodel_call})
    models = types.ModuleType("lm_eval.models")
    models.api_models = api_models
    root = types.ModuleType("lm_eval")
    root.models = models
    for name, module in (
        ("lm_eval", root),
        ("lm_eval.models", models),
        ("lm_eval.models.api_models", api_models),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    return api_models.TemplateAPI


def _run_block(monkeypatch, amodel_call):
    """Execute the injected block against the stub and return the resulting class."""
    api = _install_stub_lm_eval(monkeypatch, amodel_call)
    exec(compile(_EVAL_UNBOUND_OUTPUTS_PY, "<sitecustomize>", "exec"), {})
    return api


def test_a_refused_connection_reports_the_connection_not_the_handler(monkeypatch):
    """Upstream's handler destroys the error it is reporting; the round must still say why it stopped."""

    async def amodel_call(self, *args, **kwargs):
        # The shape of the unpatched harness: the handler logs ``{outputs}``, which
        # was never bound because the POST failed before a response was parsed.
        try:
            raise ConnectionRefusedError("connect refused")
        except BaseException:
            raise UnboundLocalError("cannot access local variable 'outputs'") from None

    api = _run_block(monkeypatch, amodel_call)

    with pytest.raises(RuntimeError) as excinfo:
        asyncio.run(api.amodel_call(object()))
    assert "refused connection" in str(excinfo.value)
    assert "3293" in str(excinfo.value)


def test_a_successful_call_is_untouched(monkeypatch):
    """The guard only intercepts the fault; every other outcome passes through."""

    async def amodel_call(self, *args, **kwargs):
        return ["answer"]

    api = _run_block(monkeypatch, amodel_call)

    assert asyncio.run(api.amodel_call(object())) == ["answer"]


def test_an_ordinary_failure_keeps_its_own_exception(monkeypatch):
    """A request that failed for a reason the handler reported correctly must not be relabelled."""

    async def amodel_call(self, *args, **kwargs):
        raise TimeoutError("read timed out")

    api = _run_block(monkeypatch, amodel_call)

    with pytest.raises(TimeoutError):
        asyncio.run(api.amodel_call(object()))


def test_a_harness_that_already_carries_the_upstream_fix_is_left_alone(monkeypatch):
    """Rebinding a fixed method would add a frame for nothing; the source check prevents it."""

    async def amodel_call(self, *args, **kwargs):
        # Post-fix form: reads the name through locals(), so the bare-name marker is absent.
        raise UnboundLocalError("unrelated")

    api = _run_block(monkeypatch, amodel_call)

    assert api.amodel_call is amodel_call


def test_applying_it_twice_appends_one_block(tmp_path, monkeypatch):
    """The sentinel makes the append idempotent, as every other patch here is."""
    target = tmp_path / "utils" / "evals" / "patches" / "lm_eval_sitecustomize.py"
    target.parent.mkdir(parents=True)
    target.write_text("# upstream sitecustomize\n", encoding="utf-8")
    monkeypatch.setenv("INFERENCEX_PATH", str(tmp_path))

    assert ensure_eval_unbound_outputs_patched(tmp_path)
    once = target.read_text(encoding="utf-8")
    assert ensure_eval_unbound_outputs_patched(tmp_path)

    assert target.read_text(encoding="utf-8") == once
    assert once.count("def _hl_eval_unbound_outputs_install") == 1
