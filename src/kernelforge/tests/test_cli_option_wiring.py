# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Every value-carrying click option must have a matching callback parameter."""

from __future__ import annotations

import inspect

import click

from kernelforge.cli import main


def _walk(group: click.Group, prefix: str = ""):
    """Yield (qualified name, command) for every leaf command under a group."""
    for name, cmd in group.commands.items():
        if isinstance(cmd, click.Group):
            yield from _walk(cmd, f"{prefix}{name} ")
        else:
            yield f"{prefix}{name}", cmd


def test_every_option_is_accepted_by_its_callback():
    missing: list[str] = []
    for name, cmd in _walk(main):
        if cmd.callback is None:
            continue
        sig = inspect.signature(cmd.callback)
        # A **kwargs callback absorbs anything, so nothing can mismatch.
        if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
            continue
        for param in cmd.params:
            # An expose_value=False option is never passed to the callback, so it needs no parameter there: it is the
            # --help / --version pattern of an eager flag whose own callback does the work and exits.
            if not param.expose_value:
                continue
            if param.name not in sig.parameters:
                missing.append(f"{name}: --{param.name.replace('_', '-')}")
    assert not missing, "click options with no callback parameter: " + ", ".join(missing)


def test_callbacks_have_no_required_parameter_click_never_supplies():
    """The inverse gap: a required parameter with no option and no default."""
    unfilled: list[str] = []
    for name, cmd in _walk(main):
        if cmd.callback is None:
            continue
        supplied = {p.name for p in cmd.params}
        for pname, p in inspect.signature(cmd.callback).parameters.items():
            if p.kind in (p.VAR_KEYWORD, p.VAR_POSITIONAL):
                continue
            if p.default is p.empty and pname not in supplied:
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
