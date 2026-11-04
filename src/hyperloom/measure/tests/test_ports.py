# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The ports local measurement never gives a test's server: the block the dev sandboxes serve on.

The OS picks a free port from its own range (32768 and up on Linux), which overlaps that block, so the pick is made
through a source the test injects and offers the reserved ports on purpose.
"""

from __future__ import annotations

from collections.abc import Iterator
from itertools import cycle
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from hyperloom.measure import local
from hyperloom.measure.ports import DEV_SANDBOX_PORTS, pick_port_outside_dev_sandboxes

#: The first and last port of the block, and the ports just outside it.
_FIRST, _LAST = 32100, 33099
_BEFORE, _AFTER = _FIRST - 1, _LAST + 1


def _offers(*ports: int) -> Iterator[int]:
    return iter(ports)


def test_the_reserved_block_holds_every_slot_of_a_hundred() -> None:
    assert (DEV_SANDBOX_PORTS.start, DEV_SANDBOX_PORTS[-1]) == (_FIRST, _LAST)
    for slot in range(100):
        block = range(32100 + 10 * slot, 32100 + 10 * slot + 10)
        assert block.start in DEV_SANDBOX_PORTS and block[-1] in DEV_SANDBOX_PORTS


@pytest.mark.parametrize("reserved", [_FIRST, 32768, 33000, _LAST])
def test_a_reserved_port_the_source_offers_is_dropped_and_the_next_one_taken(reserved: int) -> None:
    offers = _offers(reserved, reserved + 1 if reserved < _LAST else _FIRST, 40001)
    assert pick_port_outside_dev_sandboxes(lambda: next(offers)) == 40001


@pytest.mark.parametrize("port", [_BEFORE, _AFTER, 8888, 60999])
def test_a_port_next_to_the_block_is_taken_as_offered(port: int) -> None:
    assert pick_port_outside_dev_sandboxes(lambda: port) == port


def test_a_source_that_only_offers_reserved_ports_ends_as_a_refusal_not_a_reserved_port() -> None:
    reserved = cycle([_FIRST, _LAST])
    with pytest.raises(OSError, match="no port outside the dev sandboxes' 32100-33099"):
        pick_port_outside_dev_sandboxes(lambda: next(reserved))


def _point(tmp_path: Path) -> Path:
    path = tmp_path / "point.yaml"
    path.write_text(yaml.safe_dump({"benchmark": {"framework": "sglang", "envs": {"TP": "1"}}}), encoding="utf-8")
    return path


def _os_offers(monkeypatch: pytest.MonkeyPatch, *ports: int) -> None:
    """The OS's picks, as ``assign_free_port`` hands them to the backend."""
    offers = _offers(*ports)
    monkeypatch.setattr(local, "assign_free_port", lambda _current: next(offers))


def test_a_round_without_a_lifecycle_gets_a_port_outside_the_reserved_block_though_the_os_offered_one_inside(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _os_offers(monkeypatch, 32768, 33099, 41234)
    with local._free_port_for(_point(tmp_path), None) as pinned:
        envs = yaml.safe_load(pinned.read_text(encoding="utf-8"))["benchmark"]["envs"]
    assert envs == {"TP": "1", "PORT": 41234}


def _lifecycle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, picked: int) -> Any:
    info = {"eligible": True, "port": picked, "framework": "sglang"}
    monkeypatch.setattr(local, "resolve_lifecycle_params", lambda _point: info)
    call = SimpleNamespace(output_dir=str(tmp_path), points=[_point(tmp_path)])
    return local._Lifecycle(SimpleNamespace(call=call, phases=None))


def test_a_lifecycle_whose_port_fell_in_the_reserved_block_serves_on_another(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _os_offers(monkeypatch, 32200, 45678)
    lifecycle = _lifecycle(tmp_path, monkeypatch, picked=32105)
    assert (lifecycle.port, lifecycle.settings(cleanup=True)["port"]) == (45678, 45678)


def test_a_lifecycle_port_outside_the_block_is_kept(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _os_offers(monkeypatch)  # asking the OS again would be an error: the iterator is empty
    assert _lifecycle(tmp_path, monkeypatch, picked=8123).port == 8123
