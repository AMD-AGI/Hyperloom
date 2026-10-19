# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Group click options into typed objects, so a command receives one value per concern."""

from __future__ import annotations

import dataclasses
import functools


def option_group(name, group_type, *declarations):
    """Declare click options, or nested groups, that reach the command as one ``group_type`` named ``name``.

    ``group_type`` is a dataclass whose field names are the declared option
    destinations (or the names of the nested groups).
    """

    def decorate(func):
        @functools.wraps(func)
        def collect(*args, **kwargs):
            values = {field.name: kwargs.pop(field.name) for field in dataclasses.fields(group_type)}
            return func(*args, **kwargs, **{name: group_type(**values)})

        for declaration in reversed(declarations):
            collect = declaration(collect)
        return collect

    return decorate
