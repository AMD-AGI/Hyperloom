# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The trace fixtures in ``data/`` are what ``data/_make_traces.py`` writes, so each can be read, changed and written
again from the script; none is a copy of a profile."""

from __future__ import annotations

import gzip
import json
import runpy
from pathlib import Path
from typing import Any

DATA = Path(__file__).parent / "data"
GENERATOR = DATA / "_make_traces.py"


def _document(path: Path) -> Any:
    raw = path.read_bytes()
    return json.loads(gzip.decompress(raw) if path.suffix == ".gz" else raw)


def test_every_fixture_of_the_data_directory_is_what_the_generator_writes(tmp_path: Path) -> None:
    written = runpy.run_path(str(GENERATOR))["write_all"](tmp_path)

    names = sorted(path.name for path in written)
    assert names == sorted(path.name for path in DATA.iterdir() if path.is_file() and path != GENERATOR)
    for name in names:
        assert _document(tmp_path / name) == _document(DATA / name), f"{name} is not what the generator writes"


def test_the_traces_are_written_without_a_file_name_or_time_in_their_gzip_header(tmp_path: Path) -> None:
    for path in DATA.glob("*.gz"):
        header = path.read_bytes()[:10]
        assert header[3] & 0x08 == 0 and header[4:8] == b"\0\0\0\0", f"{path.name} carries a file name or a time"
