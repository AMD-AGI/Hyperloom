# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Every value-carrying click option must have a matching callback parameter."""

from __future__ import annotations

import dataclasses
import inspect
import typing

import click

from kernelforge.cli import main


def _walk(group: click.Group, prefix: str = ""):
    """Yield (qualified name, command) for every leaf command under a group."""
    for name, cmd in group.commands.items():
        if isinstance(cmd, click.Group):
            yield from _walk(cmd, f"{prefix}{name} ")
        else:
            yield f"{prefix}{name}", cmd


def _accepted_names(callback) -> set[str]:
    """Option names a callback receives: its own parameters plus the leaves of every option-group object."""
    hints = typing.get_type_hints(callback)
    names: set[str] = set()

    def leaves(group_type) -> None:
        for field in dataclasses.fields(group_type):
            nested = typing.get_type_hints(group_type)[field.name]
            if dataclasses.is_dataclass(nested):
                leaves(nested)
            else:
                names.add(field.name)

    for pname in inspect.signature(callback).parameters:
        group_type = hints.get(pname)
        if dataclasses.is_dataclass(group_type):
            leaves(group_type)
        else:
            names.add(pname)
    return names


def test_every_option_is_accepted_by_its_callback():
    missing: list[str] = []
    for name, cmd in _walk(main):
        if cmd.callback is None:
            continue
        sig = inspect.signature(cmd.callback)
        # A **kwargs callback absorbs anything, so nothing can mismatch.
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            continue
        accepted = _accepted_names(cmd.callback)
        for param in cmd.params:
            # An expose_value=False option is never passed to the callback, so it needs no parameter there: it is the
            # --help / --version pattern of an eager flag whose own callback does the work and exits.
            if not param.expose_value:
                continue
            if param.name not in accepted:
                missing.append(f"{name}: --{param.name.replace('_', '-')}")
    assert not missing, "click options with no callback parameter: " + ", ".join(missing)


def test_callbacks_have_no_required_parameter_click_never_supplies():
    """The inverse gap: a required parameter with no option and no default."""
    unfilled: list[str] = []
    for name, cmd in _walk(main):
        if cmd.callback is None:
            continue
        supplied = {p.name for p in cmd.params}
        for pname in _accepted_names(cmd.callback):
            if pname not in supplied:
                unfilled.append(f"{name}: {pname}")
    assert not unfilled, "callback parameters click cannot fill: " + ", ".join(unfilled)


def test_no_command_absorbs_an_undeclared_option():
    """An option no command declares must cost an exit code, never a default."""
    tolerant: list[str] = []
    for name, cmd in _walk(main):
        settings = cmd.context_settings or {}
        if settings.get("ignore_unknown_options") or settings.get("allow_extra_args"):
            tolerant.append(name)
        elif getattr(cmd, "ignore_unknown_options", False) or getattr(cmd, "allow_extra_args", False):
            tolerant.append(name)
    assert not tolerant, "commands that swallow undeclared options: " + ", ".join(tolerant)
