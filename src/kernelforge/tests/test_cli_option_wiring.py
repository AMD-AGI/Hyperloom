# Copyright Advanced Micro Devices, Inc. All rights reserved.

"""Every value-carrying click option must have a matching callback parameter.

click passes each option to the command callback as a keyword argument, so an
option whose name is absent from the callback signature raises TypeError the
moment the command actually runs. `--help` does NOT catch this: it renders the
option list without ever invoking the callback, so a mismatch stays invisible
until a real run fails at startup.

An option declared `expose_value=False` is exempt: click keeps it out of the
callback arguments entirely, so there is no keyword for the signature to accept.
"""

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
            # An expose_value=False option is never passed to the callback, so
            # it needs no parameter there: it is the --help / --version pattern
            # of an eager flag whose own callback does the work and exits.
            if not param.expose_value:
                continue
            if param.name not in sig.parameters:
                missing.append(f"{name}: --{param.name.replace('_', '-')}")
    assert not missing, "click options with no callback parameter: " + ", ".join(missing)


def test_callbacks_have_no_required_parameter_click_never_supplies():
    """The inverse gap: a required parameter with no option and no default.

    click supplies only what its params declare, so such a callback also fails
    at invocation time rather than at --help time.
    """
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
    """An option no command declares must cost an exit code, never a default.

    ``forge-loop`` and ``forge-rewrite-by-flydsl`` used to be exempt: they
    dropped unknown options with a warning so a consumer in a separate
    repository could ship ahead of the installed producer. Vendoring put
    producer and consumer in one tree, and what the tolerance actually absorbed
    was typos -- seven shipped examples kept passing the retired spelling of
    ``--kernel-backend`` and ran an inferred backend, exiting 0 every time (see
    ``test_rename_completeness``). This pins the exemption shut for every
    command, including ones added later.
    """
    tolerant: list[str] = []
    for name, cmd in _walk(main):
        settings = cmd.context_settings or {}
        if settings.get("ignore_unknown_options") or settings.get("allow_extra_args"):
            tolerant.append(name)
        elif getattr(cmd, "ignore_unknown_options", False) or getattr(cmd, "allow_extra_args", False):
            tolerant.append(name)
    assert not tolerant, "commands that swallow undeclared options: " + ", ".join(tolerant)
