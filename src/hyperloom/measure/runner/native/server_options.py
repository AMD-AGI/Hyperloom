# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The options a session's native server takes, and which of them its base launch sets.

A native server's ``--help`` is its whole option list, printed by the program the session built. A session keeps it as
``server_options.json`` in its directory (:func:`write_inventory`): every entry the help does not mark removed, with
its spellings and its help line, and whether the base launch sets it: its arguments spell one of them, or the runner
sets the option itself on every launch of the base. The options the base leaves unset
(:attr:`OptionInventory.unset`) are the ones a session can still try, so a search that is to cover them reads them here
rather than recalling them.

An entry's spellings are the server's own: two that differ only by case are two options, and an alias belongs to an
entry because the help lists it there, not by any table kept here. The file names the program by the SHA-256 of the
file that ran and carries the base's arguments, so a reader can tell the build and the launch it describes.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from hyperloom_common.fs_utils import file_sha256
from hyperloom_common.io import atomic_write_json
from inference_suite import help_options, native_executable, native_help
from inference_suite.argv import is_flag, tokenize_server_args_preserving_json
from inference_suite.frameworks import Framework, framework_facts

__all__ = [
    "SERVER_OPTIONS_FILE",
    "OptionInventory",
    "ServerOption",
    "build_inventory",
    "inventory_path",
    "read_inventory",
    "server_inventory",
    "write_inventory",
]

SERVER_OPTIONS_FILE = "server_options.json"
_SCHEMA = 1
#: How a help line names the environment variable the server reads the option from when its arguments do not
#: give it.
_ENV = re.compile(r"\(env: (\w+)\)")
#: How the long spelling of a switch's negation begins. The help lists a switch and its negation as one entry,
#: so the spellings of such an entry do not all mean the same.
_NEGATION = "--no-"


@dataclass(frozen=True)
class ServerOption:
    """One option of the server: every spelling its help lists, the help line, and whether the base launch sets it."""

    spellings: tuple[str, ...]
    help: str
    set_by_base: bool

    @property
    def name(self) -> str:
        """The longest spelling, the one a review or a message cites."""
        return max(self.spellings, key=len)

    @property
    def env(self) -> str | None:
        """The environment variable the server reads the option from, as its help line names it; ``None`` when it
        names none."""
        named = _ENV.search(self.help)
        return named.group(1) if named else None


@dataclass(frozen=True)
class OptionInventory:
    """The options of the server ``program`` whose file has ``executable_sha256``, read against ``base_args``."""

    program: str
    executable_sha256: str
    base_args: str
    options: tuple[ServerOption, ...]

    @property
    def unset(self) -> tuple[ServerOption, ...]:
        """The options the base launch does not set, in help order."""
        return tuple(option for option in self.options if not option.set_by_base)

    def option(self, spelling: str) -> ServerOption | None:
        """The option ``spelling`` is a spelling of; ``None`` when the server lists none."""
        return next((option for option in self.options if spelling in option.spellings), None)

    def spelled_as(self, args: str, like: str) -> str:
        """``args`` with each option ``like`` spells another way spelled as ``like`` spells it, so that the two
        compare by what they set; ``args`` itself when it has none. A switch listed with its negation keeps its
        spelling, and so does an option ``like`` spells in two ways, which has no one spelling to take."""
        spelled = _spelled(like)
        tokens = _tokens(args)
        respelled = [self._spelled_among(token, spelled) for token in tokens]
        return " ".join(respelled) if respelled != tokens else args

    def _spelled_among(self, token: str, spelled: frozenset[str]) -> str:
        """``token`` under the one spelling in ``spelled`` of the option it spells another way; else ``token``."""
        flag, equals, value = token.partition("=")
        option = self.option(flag)
        if option is None or flag in spelled or any(name.startswith(_NEGATION) for name in option.spellings):
            return token
        theirs = [name for name in option.spellings if name in spelled]
        return f"{theirs[0]}{equals}{value}" if len(theirs) == 1 else token


def _tokens(args: str) -> list[str]:
    split = tokenize_server_args_preserving_json(args)
    return split[1] if split is not None else args.split()


def _spelled(args: str) -> frozenset[str]:
    """The flags ``args`` spell, as written: ``--flag=value`` counts as ``--flag``."""
    return frozenset(token.partition("=")[0] for token in _tokens(args) if is_flag(token))


def build_inventory(
    program: str, executable: Path, help_text: str, base_args: str, *, runner_owned: Collection[str] = ()
) -> OptionInventory:
    """The inventory of ``program``, the file ``executable``, from its ``--help`` text and the base's ``base_args``.

    ``runner_owned`` are the flags the runner sets itself on every launch of the base (the model, the address): the
    base launch sets those too, though its arguments do not spell them.
    """
    spelled = _spelled(base_args) | frozenset(runner_owned)
    options = tuple(
        ServerOption(entry.spellings, entry.help, set_by_base=not spelled.isdisjoint(entry.spellings))
        for entry in help_options(help_text)
        if not entry.removed
    )
    return OptionInventory(program, file_sha256(executable), base_args, options)


def write_inventory(inventory: OptionInventory, directory: Path) -> Path:
    """Write ``inventory`` as ``directory``'s option file, whole or not at all; return the file."""
    path = directory / SERVER_OPTIONS_FILE
    atomic_write_json(path, {"schema": _SCHEMA, **asdict(inventory)})
    return path


def read_inventory(path: Path) -> OptionInventory:
    """The inventory written at ``path``.

    Raises:
        ValueError: The file is of another schema than this module writes.
    """
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != _SCHEMA:
        raise ValueError(f"{path} is an option inventory of schema {document.get('schema')!r}, not {_SCHEMA}")
    options = tuple(
        ServerOption(tuple(entry["spellings"]), entry["help"], entry["set_by_base"]) for entry in document["options"]
    )
    return OptionInventory(document["program"], document["executable_sha256"], document["base_args"], options)


def inventory_path(directory: Path) -> Path | None:
    """``directory``'s option file; ``None`` when it holds none."""
    path = directory / SERVER_OPTIONS_FILE
    return path if path.is_file() else None


def server_inventory(
    framework: str, env: Mapping[str, str], base_args: str, *, cwd: Path, runner_owned: Collection[str] = ()
) -> OptionInventory | None:
    """The inventory of ``framework``'s server as ``env``'s ``PATH`` resolves it, run in ``cwd`` for its ``--help``,
    against ``base_args`` and the base's ``runner_owned`` flags (:func:`build_inventory`); ``None`` for a framework
    whose server is no native program.

    ``env``'s ``PATH`` must put the session's base build first: the inventory is of the program found there.

    Raises:
        FrameworkProbeError: The program is not on ``env``'s ``PATH``, or cannot print its ``--help``.
    """
    command = framework_facts(Framework(framework)).server_command
    if command is None or not command.native:
        return None
    program = command.entrypoint[0]
    help_text = native_help(program, env, cwd=cwd)
    executable = Path(native_executable(program, env))
    return build_inventory(program, executable, help_text, base_args, runner_owned=runner_owned)
