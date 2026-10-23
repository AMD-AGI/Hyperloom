# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The pack a session reads is staged for its sandboxes as the bytes whose digest the session recorded."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from hyperloom.common.rule_pack import PACK_ENV, RulePackError, load_pack, rule_pack_identity
from hyperloom.measure.rule_pack_stage import rule_pack_envs
from hyperloom.measure.tests.rule_pack_fixtures import identity_of, other_pack, reading_pack  # noqa: F401

Reading = Callable[[Path | None], None]


def test_a_session_that_reads_the_reference_pack_sets_nothing(reading_pack: Reading, tmp_path: Path) -> None:
    reading_pack(None)
    assert rule_pack_envs(tmp_path / "rule-pack") == {}
    assert not (tmp_path / "rule-pack").exists()


def test_the_named_pack_is_staged_under_its_digest_and_the_variable_names_the_copy(
    reading_pack: Reading, tmp_path: Path
) -> None:
    named = other_pack(tmp_path)
    reading_pack(named)
    envs = rule_pack_envs(tmp_path / "session" / "rule-pack")
    assert set(envs) == {PACK_ENV}
    staged = Path(envs[PACK_ENV])
    assert staged.is_absolute() and staged != named
    assert staged.parent == (tmp_path / "session" / "rule-pack").resolve()
    assert staged.name == identity_of(named)["digest"].removeprefix("sha256:") + ".yaml"
    assert staged.read_bytes() == named.read_bytes()
    assert load_pack(staged).digest == rule_pack_identity()["digest"] == identity_of(named)["digest"]


def test_a_staged_copy_that_was_damaged_is_written_again(reading_pack: Reading, tmp_path: Path) -> None:
    named = other_pack(tmp_path)
    reading_pack(named)
    staged = Path(rule_pack_envs(tmp_path / "rule-pack")[PACK_ENV])
    staged.write_bytes(b"not the pack")
    assert rule_pack_envs(tmp_path / "rule-pack") == {PACK_ENV: str(staged)}
    assert staged.read_bytes() == named.read_bytes()


def test_a_pack_file_that_changed_since_the_process_read_it_is_refused(reading_pack: Reading, tmp_path: Path) -> None:
    named = other_pack(tmp_path)
    reading_pack(named)
    rule_pack_identity()
    named.write_bytes(named.read_bytes() + b"\n# edited after the session read it\n")
    with pytest.raises(RulePackError, match="changed since this process read it") as refused:
        rule_pack_envs(tmp_path / "rule-pack")
    assert str(named) in str(refused.value)
    assert not (tmp_path / "rule-pack").exists(), "nothing is staged for a digest the session did not record"


def test_a_pack_file_that_is_gone_is_refused(reading_pack: Reading, tmp_path: Path) -> None:
    named = other_pack(tmp_path)
    reading_pack(named)
    rule_pack_identity()
    named.unlink()
    with pytest.raises(RulePackError, match="cannot be read to stage it"):
        rule_pack_envs(tmp_path / "rule-pack")
