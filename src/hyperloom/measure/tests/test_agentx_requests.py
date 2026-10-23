# SPDX-FileCopyrightText: 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

"""The requests a measured AgentX round's client marked failed are counted by class from the round's export (#117).

The export lines are rebuilt from the pinned client's source and docs (``agentx_export``: no export of a real run was
at hand). What is pinned: which failed request is transport and which is no-content, that a request the client counted
successful is none of them, that a count the export cannot give is ``None`` with its reason and never zero, that
warm-up requests count in none, and that a file that is missing, ambiguous or damaged says so instead of reading as a
round with nothing failed.
"""

from __future__ import annotations

import errno
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from hyperloom.common.outside_facts.rule_pack import DEFAULT_PACK_PATH
from hyperloom.contracts.measurement import RequestClasses
from hyperloom.measure.agentx_requests import export_files, read_request_classes, read_round_classes
from hyperloom.measure.socket import RoundOutcome
from hyperloom.measure.tests import agentx_export as export

EXPORT = "aiperf_artifacts/profile_export.jsonl"


def _round(tmp_path: Path, lines: list[str], *, below: str = "") -> Path:
    """A round directory whose export holds ``lines``; ``below`` puts the benchmark one directory down."""
    path = tmp_path / "round" / below / EXPORT
    path.parent.mkdir(parents=True)
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return tmp_path / "round"


def _read(round_dir: Path, root: Path | None = None) -> RequestClasses:
    return read_request_classes(round_dir, root or round_dir.parent, 0, 128)


def _counts(classes: RequestClasses) -> tuple[int | None, ...]:
    return (
        classes.requests_total,
        classes.failed_total,
        classes.failed_transport,
        classes.failed_no_content,
        classes.failed_empty,
        classes.failed_short,
    )


def _mixed() -> list[str]:
    """Twelve profiling requests: seven served, four transport failures, one no-content; and warm-up lines."""
    records = [export.served(n) for n in range(7)]
    records += [export.refused(7), export.refused(8, status=400, reason="Bad Request"), export.unreachable(9)]
    records += [export.cancelled(10), export.no_content(11)]
    records += [
        export.served(20, phase="warmup"),
        export.refused(21, phase="warmup"),
        export.no_content(22, phase="warmup"),
    ]
    return export.text(*records)


def test_each_failed_request_is_classed_by_what_the_server_did(tmp_path: Path) -> None:
    classes = _read(_round(tmp_path, _mixed()))
    assert classes.requests_total == 12
    assert classes.failed_total == 5
    assert classes.failed_transport == 4, (
        "an HTTP error status, a connection error and a cancellation: no usable answer"
    )
    assert classes.failed_no_content == 1, "the server answered and the client found nothing in the answer"


def test_a_request_the_client_counted_successful_is_in_no_failed_class(tmp_path: Path) -> None:
    classes = _read(_round(tmp_path, export.text(*(export.served(n, output_tokens=1) for n in range(5)))))
    assert _counts(classes) == (5, 0, 0, 0, 0, 0), "a one-token answer with content is a success, not a short failure"
    assert classes.reason == ""


def test_warmup_requests_are_in_no_count_and_a_line_without_a_phase_is_a_profiling_one(tmp_path: Path) -> None:
    old = export.served(1)
    del old["metadata"]["benchmark_phase"]  # artifacts that predate the field: InferenceX reads them as profiling
    lines = export.text(old, export.refused(2, phase="warmup"), export.served(3, phase="warmup"))
    assert _counts(_read(_round(tmp_path, lines))) == (1, 0, 0, 0, 0, 0)


def test_empty_and_short_are_told_apart_exactly_by_the_output_token_count_the_patched_client_exports(
    tmp_path: Path,
) -> None:
    records = [export.served(n) for n in range(4)]
    records += [export.no_content(10, output_tokens=0), export.no_content(11, output_tokens=1)]
    records += [export.no_content(12, output_tokens=5), export.no_content(13, output_tokens=0), export.refused(14)]
    classes = _read(_round(tmp_path, export.text(*records)))
    assert _counts(classes) == (9, 5, 1, 4, 2, 2), (
        "two zero-token replies are empty, a one-token and a five-token short"
    )
    assert classes.reason == ""


def test_a_client_that_exports_no_token_count_on_a_failed_line_leaves_empty_and_short_not_computed(
    tmp_path: Path,
) -> None:
    """An export of the client as released: the old exports of a run the runner did not patch the client of."""
    records = [export.served(n) for n in range(6)] + [export.no_content(10), export.no_content(11), export.refused(12)]
    classes = _read(_round(tmp_path, export.text(*records)))
    assert _counts(classes) == (9, 3, 1, 2, None, None), "the split is None, not 0: nothing says how many were empty"
    assert classes.reason == (
        "the client's export carries no output token count for a request it counted failed, so 2 of the 2 no-content "
        "failures cannot be told empty from short"
    ), "recorded results hold this text and are compared with it when reverified: it is not reworded"


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(-1, id="negative"),
        pytest.param("3", id="text"),
        pytest.param(True, id="bool"),
        pytest.param(1.5, id="float"),
    ],
)
def test_a_token_count_that_is_not_a_count_is_no_count(tmp_path: Path, value: object) -> None:
    damaged = export.no_content(2, output_tokens=0)
    damaged["metrics"]["error_osl"]["value"] = value
    classes = _read(_round(tmp_path, export.text(export.served(1), damaged)))
    assert (classes.failed_no_content, classes.failed_empty, classes.failed_short) == (1, None, None)


def test_the_output_length_of_a_served_request_is_not_the_count_of_a_failed_one(tmp_path: Path) -> None:
    """``output_sequence_length`` is the metric of a request the client counted successful; only ``error_osl`` counts."""
    failed = export.no_content(2)
    failed["metrics"]["output_sequence_length"] = {"value": 4, "unit": "tokens"}
    classes = _read(_round(tmp_path, export.text(export.served(1), failed)))
    assert (classes.failed_no_content, classes.failed_empty, classes.failed_short) == (1, None, None)


def test_a_split_that_only_some_failed_lines_support_is_not_computed_either(tmp_path: Path) -> None:
    lines = export.text(export.served(1), export.no_content(2, output_tokens=0), export.no_content(3))
    classes = _read(_round(tmp_path, lines))
    assert (classes.failed_no_content, classes.failed_empty, classes.failed_short) == (2, None, None)
    assert "1 of the 2" in classes.reason, "a half-known split would read as a full one"


def test_a_round_with_no_no_content_failure_has_a_split_of_zero_because_there_is_nothing_to_split(
    tmp_path: Path,
) -> None:
    classes = _read(_round(tmp_path, export.text(export.served(1), export.refused(2), export.unreachable(3))))
    assert _counts(classes) == (3, 2, 2, 0, 0, 0)


def test_an_export_that_is_missing_has_no_count_and_says_why(tmp_path: Path) -> None:
    (tmp_path / "round").mkdir()
    classes = _read(tmp_path / "round")
    assert _counts(classes) == (None,) * 6
    assert (classes.export, classes.reason) == ("", "the round left no per-request export")


def test_two_exports_below_one_round_are_ambiguous_and_none_is_taken(tmp_path: Path) -> None:
    round_dir = _round(tmp_path, export.text(export.served(1)))
    other = round_dir / "again" / EXPORT
    other.parent.mkdir(parents=True)
    other.write_text("\n".join(export.text(export.refused(2))) + "\n", encoding="utf-8")
    classes = _read(round_dir)
    assert _counts(classes) == (None,) * 6
    assert classes.reason == "the round left 2 per-request exports, so none of them is the round's"


@pytest.mark.parametrize(
    "damage",
    [
        pytest.param("not json at all", id="not-json"),
        pytest.param('{"metadata": {"benchmark_phase": "profiling"}', id="truncated"),
        pytest.param("[1, 2, 3]", id="not-an-object"),
        pytest.param('{"metrics": {}}', id="no-metadata"),
        pytest.param('{"metadata": {}, "error": "boom"}', id="error-that-is-not-an-object"),
        pytest.param("[" * 100_000, id="too-deeply-nested"),
        pytest.param('{"metadata": {"benchmark_phase": "profiling"}}', id="no-metrics"),
    ],
)
def test_a_damaged_line_leaves_no_count_instead_of_a_count_without_it(tmp_path: Path, damage: str) -> None:
    classes = _read(_round(tmp_path, [*export.text(export.served(1), export.refused(2)), damage]))
    assert _counts(classes) == (None,) * 6
    assert classes.reason == "1 line(s) of the export are not a request record, so no count of it can be trusted"
    assert classes.export.endswith(EXPORT), "the file is still named, for whoever reads it"


def _padded(length: int) -> str:
    """A request line that is otherwise sound and is ``length`` bytes long."""
    record = export.served(1)
    record["metadata"]["pad"] = ""
    short = len(json.dumps(record, separators=(",", ":")))
    record["metadata"]["pad"] = "x" * (length - short)
    line = json.dumps(record, separators=(",", ":"))
    assert len(line.encode()) == length
    return line


@pytest.mark.parametrize("length", [1 << 20, (1 << 20) + 1, 1 << 21], ids=["at-the-limit", "one-over", "far-over"])
def test_a_line_past_the_size_limit_is_damage_even_when_it_is_otherwise_a_request(tmp_path: Path, length: int) -> None:
    """The limit counts the line with its newline; a longer one is never read whole, and no count rests on it."""
    classes = _read(_round(tmp_path, [*export.text(export.refused(2)), _padded(length)]))
    assert _counts(classes) == (None,) * 6
    assert classes.reason.startswith("1 line(s) of the export are not a request record")


def test_the_longest_line_the_limit_admits_is_read(tmp_path: Path) -> None:
    classes = _read(_round(tmp_path, [_padded((1 << 20) - 1)]))
    assert _counts(classes) == (1, 0, 0, 0, 0, 0)


def test_an_export_with_no_profiling_request_is_not_a_round_that_failed_none(tmp_path: Path) -> None:
    classes = _read(_round(tmp_path, export.text(export.served(1, phase="warmup"))))
    assert _counts(classes) == (None,) * 6
    assert classes.reason == "the export holds no profiling request"


def test_blank_lines_are_not_damage_and_an_empty_error_is_no_error(tmp_path: Path) -> None:
    served = export.served(1)
    served["error"] = {}
    path = _round(tmp_path, ["", *export.text(served), "   "])
    assert _counts(_read(path)) == (1, 0, 0, 0, 0, 0)


def test_the_export_is_named_relative_to_the_tests_output_and_found_one_level_down_too(tmp_path: Path) -> None:
    round_dir = _round(tmp_path, export.text(export.served(1)), below="benchmark_agentx_1")
    classes = _read(round_dir, root=tmp_path)
    assert classes.export == f"round/benchmark_agentx_1/{EXPORT}"
    assert export_files(round_dir) == [round_dir / "benchmark_agentx_1" / EXPORT]


def test_an_export_outside_the_tests_output_is_not_named(tmp_path: Path) -> None:
    round_dir = _round(tmp_path, export.text(export.served(1)))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    classes = _read(round_dir, root=elsewhere)
    assert _counts(classes) == (None,) * 6 and classes.export == ""
    assert classes.reason == "the export is not below the test's output"


def test_an_export_that_cannot_be_read_says_so_and_does_not_raise(tmp_path: Path, monkeypatch) -> None:
    round_dir = _round(tmp_path, export.text(export.served(1)))
    opened = Path.open

    def failing(self: Path, *args: Any, **kwargs: Any):
        if self.name == "profile_export.jsonl":
            raise OSError(errno.EIO, "input/output error")
        return opened(self, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing)
    classes = _read(round_dir)
    assert _counts(classes) == (None,) * 6
    assert classes.reason == "the export could not be read (OSError)"


def test_every_measured_round_has_an_entry_at_its_own_point_and_load(tmp_path: Path) -> None:
    first, second = tmp_path / "p0", tmp_path / "p1"
    for directory, lines in ((first, export.text(export.served(1))), (second, export.text(export.refused(2)))):
        (directory / "aiperf_artifacts").mkdir(parents=True)
        (directory / EXPORT).write_text("\n".join(lines) + "\n", encoding="utf-8")
    rounds = (
        RoundOutcome(0, "warmup", first, 0),
        RoundOutcome(0, "measure", first, 0),
        RoundOutcome(1, "measure", second, 0),
        RoundOutcome(1, "accuracy", second, 0),
        RoundOutcome(2, "measure", tmp_path / "nothing", 0),
    )
    entries = read_round_classes(rounds, tmp_path, (128, 256))
    assert [(c.point, c.concurrency, c.requests_total, c.failed_transport) for c in entries] == [
        (0, 128, 1, 0),
        (1, 256, 1, 1),
        (2, None, None, None),
    ], "a warm-up and an accuracy round have none; a measured round with no export says so"
    assert [c.export for c in entries] == [f"p0/{EXPORT}", f"p1/{EXPORT}", ""]


def test_the_failure_class_is_read_from_the_error_not_from_its_wording(tmp_path: Path) -> None:
    """A transport error whose text names the other class stays transport; the class name is what decides."""
    quoting = export.refused(1, status=500, reason="Internal Server Error")
    quoting["error"]["message"] = json.dumps({"type": "InvalidInferenceResultError"})
    classes = _read(_round(tmp_path, export.text(quoting, export.served(2))))
    assert (classes.failed_transport, classes.failed_no_content) == (1, 0)


def test_a_line_that_is_not_a_record_of_this_export_is_not_counted_even_with_a_matching_error(tmp_path: Path) -> None:
    other: dict[str, Any] = {"error": export.INVALID_REPLY}
    classes = _read(_round(tmp_path, [*export.text(export.served(1)), json.dumps(other)]))
    assert classes.requests_total is None


def test_the_example_lines_the_rule_pack_holds_read_as_the_pack_says_they_do(tmp_path: Path) -> None:
    """The pack's own examples are the reader's fixtures: a fact moved in a pack must move them too."""
    facts = yaml.safe_load(DEFAULT_PACK_PATH.read_text(encoding="utf-8"))["facts"]
    lines = [e[e.index(": {") + 2 :] for e in facts["aiperf_request_export"]["examples"] if ": {" in e]
    expected = [
        ("a served request", (1, 0, 0, 0, 0, 0)),
        ("a refused request", (1, 1, 1, 0, 0, 0)),
        ("an unreachable server", (1, 1, 1, 0, 0, 0)),
        ("a no-content reply of the client as released", (1, 1, 0, 1, None, None)),
        ("a no-content reply of the patched client: nothing produced", (1, 1, 0, 1, 1, 0)),
    ]
    assert len(lines) == len(expected)
    for index, (line, (what, counts)) in enumerate(zip(lines, expected)):
        assert _counts(_read(_round(tmp_path / str(index), [line]))) == counts, what
