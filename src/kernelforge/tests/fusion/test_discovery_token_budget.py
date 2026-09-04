# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Architecture guards for the registered discovery backend."""

from __future__ import annotations

import inspect

from kernelforge.fusion import command, discover


def test_discovery_requires_an_injected_llm() -> None:
    parameter = inspect.signature(discover.discover_recipes).parameters["llm_fn"]
    assert parameter.default is inspect.Parameter.empty


def test_cli_injects_the_registered_agent_backend() -> None:
    source = inspect.getsource(command.run.callback)
    assert "registered_agent_llm_fn" in source
    assert "llm_fn=registered_agent_llm_fn(" in source


def test_legacy_direct_client_fallback_is_absent() -> None:
    assert not hasattr(discover, "default_llm_fn")
    assert not hasattr(discover, "complete_with_retry")
