# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Architecture guards for the registered discovery backend."""

from __future__ import annotations

import inspect

from kernelforge.workflows.forge_fusion import discover, discovery_stage


def test_discovery_requires_an_injected_llm() -> None:
    parameter = inspect.signature(discover.discover_recipes).parameters["llm_fn"]
    assert parameter.default is inspect.Parameter.empty


def test_cli_injects_the_registered_agent_backend() -> None:
    """Every discovery path reaches the model through the backend the CLI selected.

    None of them may fall back to a client of its own: the selected backend is
    the one whose model, sandbox mode and prompt log the run reports.
    """
    for discovery in (discovery_stage._discover_anchored, discovery_stage._discover_by_llm):
        source = inspect.getsource(discovery)
        assert "registered_agent_llm_fn" in source, discovery.__name__
        assert "llm_fn=registered_agent_llm_fn(" in source, discovery.__name__


def test_legacy_direct_client_fallback_is_absent() -> None:
    assert not hasattr(discover, "default_llm_fn")
    assert not hasattr(discover, "complete_with_retry")
