# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Which optional arguments the loop hands a session callback."""

from __future__ import annotations

from kernelforge.workflows.forge_loop.roles.agent_extras import declared_extras

EXTRAS = {"session_sink": {}, "gate_scoring": object()}


def test_a_callback_is_handed_the_extras_it_names():
    async def names_one(kernel_path, history, session_sink):
        return ""

    async def names_none(kernel_path, history):
        return ""

    assert declared_extras(names_one, EXTRAS) == {"session_sink": EXTRAS["session_sink"]}
    assert declared_extras(names_none, EXTRAS) == {}


def test_a_callback_that_accepts_any_keyword_is_handed_every_extra():
    """A wrapper that forwards its keywords would otherwise lose every extra on the way to what it wraps."""

    async def forwards(kernel_path, history, **extras):
        return ""

    assert declared_extras(forwards, EXTRAS) == EXTRAS


def test_a_callback_whose_signature_cannot_be_read_is_handed_none():
    assert declared_extras(object(), EXTRAS) == {}
