# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""forge-fuse accepts exactly the providers the registry knows, plus ``auto``."""

from __future__ import annotations

import click
import pytest

from kernelforge.agent_backends.registry import list_agent_providers
from kernelforge.fusion.command import run as main


def _agent_backend_option() -> click.Option:
    return next(param for param in main.params if param.name == "agent_backend")


@pytest.mark.parametrize("name", ["auto", *list_agent_providers()])
def test_every_registered_provider_is_accepted(name: str) -> None:
    """A live dsh run was refused: 'deepseek' is not one of 'auto', 'claude', 'codex'."""
    ctx = click.Context(main)
    assert _agent_backend_option().process_value(ctx, name) == name


def test_an_unknown_provider_is_refused() -> None:
    ctx = click.Context(main)
    with pytest.raises(click.BadParameter):
        _agent_backend_option().process_value(ctx, "gemini")
