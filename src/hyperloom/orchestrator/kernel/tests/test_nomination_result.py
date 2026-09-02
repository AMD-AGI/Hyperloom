# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""Consumer rules for a multi-patch forge result: drop, dedupe, tolerate empty."""

from __future__ import annotations

from typing import Any

import pytest

from hyperloom.orchestrator.kernel import nomination_result as nres


def _entry(name: str, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "kernel_name": name,
        "patch_path": f"/out/{name}.patch",
        "target_file": f"/repo/{name}.py",
        "micro_speedup": 1.0,
    }
    row.update(overrides)
    return row


def _envelope(entries: list[Any], **overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {"patches": entries}
    payload.update(overrides)
    return payload


def test_reads_every_declared_field() -> None:
    outcome = nres.parse_outcome(
        _envelope(
            [
                _entry(
                    "a",
                    kernel_repo="/repo",
                    snapshot_dir="/snap",
                    base_commit="abc123",
                    micro_speedup=1.4,
                    write_paths=["/repo/a.py", " "],
                    env_flag="ZAYA_FUSED_A",
                )
            ]
        )
    )
    (patch,) = outcome.patches
    assert patch.kernel_name == "a"
    assert patch.patch_path == "/out/a.patch"
    assert patch.target_file == "/repo/a.py"
    assert patch.kernel_repo == "/repo"
    assert patch.snapshot_dir == "/snap"
    assert patch.base_commit == "abc123"
    assert patch.micro_speedup == 1.4
    assert patch.write_paths == ("/repo/a.py",)
    # The env flag gates the fused path; it has to survive parsing or the
    # consumer queues a sibling that integrate measures un-fused.
    assert patch.env_flag == "ZAYA_FUSED_A"


def test_env_flag_defaults_to_empty_for_a_self_activating_sibling() -> None:
    """A compile-pass sibling flips a default in source and carries no flag."""
    (patch,) = nres.parse_outcome(_envelope([_entry("a")])).patches
    assert patch.env_flag == ""


def test_source_file_substitutes_for_target_file() -> None:
    entry = _entry("a")
    entry.pop("target_file")
    entry["source_file"] = "/repo/a.py"
    (patch,) = nres.parse_outcome(_envelope([entry])).patches
    assert patch.target_file == "/repo/a.py"


@pytest.mark.parametrize(
    ("field_name", "reason"),
    [
        ("kernel_name", nres.DROP_MISSING_KERNEL_NAME),
        ("patch_path", nres.DROP_MISSING_PATCH_PATH),
        ("target_file", nres.DROP_MISSING_TARGET_FILE),
    ],
)
def test_one_bad_entry_does_not_cost_its_siblings(field_name: str, reason: str) -> None:
    """The batch is N independent results; a malformed entry drops alone."""
    bad = _entry("bad")
    bad.pop(field_name)
    outcome = nres.parse_outcome(_envelope([_entry("good_one"), bad, _entry("good_two")]))
    assert sorted(patch.kernel_name for patch in outcome.patches) == ["good_one", "good_two"]
    assert [drop.reason for drop in outcome.dropped] == [reason]


def test_non_object_entry_is_dropped_with_a_reason() -> None:
    outcome = nres.parse_outcome(_envelope(["junk", _entry("good")]))
    assert [patch.kernel_name for patch in outcome.patches] == ["good"]
    assert [drop.reason for drop in outcome.dropped] == [nres.DROP_NOT_AN_OBJECT]


def test_duplicate_names_collapse_to_the_stronger_claim() -> None:
    """Duplicates would share one ledger key, one retry budget, one rejection."""
    outcome = nres.parse_outcome(
        _envelope(
            [
                _entry("same", patch_path="/out/weak.patch", micro_speedup=1.1),
                _entry("same", patch_path="/out/strong.patch", micro_speedup=1.9),
            ]
        )
    )
    (patch,) = outcome.patches
    assert patch.patch_path == "/out/strong.patch"
    assert [drop.reason for drop in outcome.dropped] == [nres.DROP_DUPLICATE_KERNEL_NAME]
    assert outcome.dropped[0].raw["patch_path"] == "/out/weak.patch"


def test_duplicate_order_does_not_change_the_winner() -> None:
    stronger_first = nres.parse_outcome(
        _envelope(
            [
                _entry("same", patch_path="/out/strong.patch", micro_speedup=1.9),
                _entry("same", patch_path="/out/weak.patch", micro_speedup=1.1),
            ]
        )
    )
    assert stronger_first.patches[0].patch_path == "/out/strong.patch"


def test_patches_are_ordered_strongest_first() -> None:
    outcome = nres.parse_outcome(
        _envelope([_entry("mid", micro_speedup=1.3), _entry("top", micro_speedup=2.0), _entry("low")])
    )
    assert [patch.kernel_name for patch in outcome.patches] == ["top", "mid", "low"]


def test_empty_patches_is_a_valid_outcome() -> None:
    outcome = nres.parse_outcome(_envelope([]))
    assert outcome.patches == ()
    assert outcome.dropped == ()
    assert outcome.is_empty is True


@pytest.mark.parametrize("payload", [None, [], "text", 7])
def test_non_mapping_envelope_yields_an_empty_outcome(payload: Any) -> None:
    assert nres.parse_outcome(payload).is_empty is True


def test_missing_patches_key_yields_an_empty_outcome() -> None:
    assert nres.parse_outcome({"baseline_ms": 12.0}).is_empty is True


def test_nomination_summary_is_read_when_present() -> None:
    outcome = nres.parse_outcome(
        _envelope([_entry("a")], nomination={"candidates_seen": 40, "resolved": 12, "selected": 1})
    )
    assert (outcome.candidates_seen, outcome.resolved, outcome.selected) == (40, 12, 1)


def test_absent_summary_counts_as_zero() -> None:
    outcome = nres.parse_outcome(_envelope([_entry("a")]))
    assert (outcome.candidates_seen, outcome.resolved, outcome.selected) == (0, 0, 0)


def test_unparsable_summary_counts_as_zero() -> None:
    outcome = nres.parse_outcome(_envelope([_entry("a")], nomination={"candidates_seen": "many"}))
    assert outcome.candidates_seen == 0


def test_non_finite_speedup_sorts_last_without_crashing() -> None:
    outcome = nres.parse_outcome(_envelope([_entry("nan", micro_speedup=None), _entry("real", micro_speedup=1.2)]))
    assert [patch.kernel_name for patch in outcome.patches] == ["real", "nan"]
