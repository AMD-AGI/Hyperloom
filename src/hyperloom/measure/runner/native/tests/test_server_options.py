# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The option inventory of a session's native server: every ``--help`` entry, and which the base launch sets."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyperloom.measure.runner.native.server_options import (
    SERVER_OPTIONS_FILE,
    OptionInventory,
    build_inventory,
    inventory_path,
    read_inventory,
    server_inventory,
    write_inventory,
)

_HELP = """\
----- common params -----

-t,    --threads N                      number of CPU threads to use during generation (default:
                                        -1)
-c,    --ctx-size N                     size of the prompt context (default: 0)
-C,    --cpu-mask M                     CPU affinity mask
--jinja, --no-jinja                     whether to use the chat template engine
--keep N                                number of tokens to keep from the initial prompt
--old-size N                            the argument has been removed. use --ctx-size
"""


def _program(root: Path) -> dict[str, str]:
    """A stand-in server on a ``PATH`` of its own that prints ``_HELP`` for any argument."""
    bin_dir = root / "bin"
    bin_dir.mkdir()
    program = bin_dir / "llama-server"
    (root / "help.txt").write_text(_HELP, encoding="utf-8")
    program.write_text(f'#!/bin/sh\n/bin/cat "{root / "help.txt"}"\n', encoding="utf-8")
    program.chmod(0o755)
    return {"PATH": str(bin_dir)}


def _inventory(tmp_path: Path, base_args: str) -> OptionInventory:
    executable = tmp_path / "server"
    executable.write_bytes(b"binary")
    return build_inventory("llama-server", executable, _HELP, base_args)


_NAMES = ["--threads", "--ctx-size", "--cpu-mask", "--no-jinja", "--keep"]


@pytest.mark.parametrize(
    ("base_args", "set_by_base", "unset"),
    [
        ("-c 4096 --no-jinja --keep=8", ["--ctx-size", "--jinja", "--keep"], ["--threads", "--cpu-mask"]),
        ("-C 0xff", ["-C"], ["--threads", "-c", "--no-jinja", "--keep"]),
    ],
    ids=["any-spelling", "the-servers-own-case"],
)
def test_an_option_is_set_when_the_base_spells_any_of_its_spellings_in_the_servers_own_case(
    tmp_path, base_args, set_by_base, unset
):
    inventory = server_inventory("llamacpp", _program(tmp_path), base_args, cwd=tmp_path)

    assert [option.name for option in inventory.options] == _NAMES
    assert all(inventory.option(spelling).set_by_base for spelling in set_by_base)
    assert [inventory.option(spelling) for spelling in unset] == list(inventory.unset)
    assert inventory.option("--old-size") is None and inventory.option("--nope") is None


def test_a_written_inventory_reads_back_equal(tmp_path):
    inventory = _inventory(tmp_path, "-c 4096")
    assert inventory_path(tmp_path) is None

    written = write_inventory(inventory, tmp_path)

    assert written == tmp_path / SERVER_OPTIONS_FILE == inventory_path(tmp_path)
    assert read_inventory(written) == inventory
    document = json.loads(written.read_text(encoding="utf-8"))
    assert document["schema"] == 1
    assert document["options"][1] == {
        "spellings": ["-c", "--ctx-size"],
        "help": "N size of the prompt context (default: 0)",
        "set_by_base": True,
    }
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(SERVER_OPTIONS_FILE)] == [
        SERVER_OPTIONS_FILE
    ]


def test_an_inventory_of_another_schema_is_refused(tmp_path):
    path = write_inventory(_inventory(tmp_path, ""), tmp_path)
    path.write_text(json.dumps({**json.loads(path.read_text(encoding="utf-8")), "schema": 2}), encoding="utf-8")

    with pytest.raises(ValueError, match="schema"):
        read_inventory(path)
